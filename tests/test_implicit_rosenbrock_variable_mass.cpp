#include "implicit_rosenbrock_variable_mass_problem.h"

using namespace rosenbrock_tests;

// The three systems in the support header isolate genuine state-dependent
// decay, time-dependent mass with nonzero q_t, and coupled state/time mass.
// Check q derivatives against centered differences at fixed state/time. The
// deliberately incomplete derivatives must fail the same checks visibly.
template <class Model>
void derivatives()
{
    auto             ops = std::make_shared<operations_type>( 2 );
    auto             p   = std::make_shared<variable_mass_problem<Model>>( ops );
    vector_wrap_type in( *ops ), direction( *ops ), plus( *ops ), minus( *ops ), fp( *ops ), fm( *ops ), exact( *ops );
    for ( auto *v : { &in, &direction, &plus, &minus, &fp, &fm, &exact } )
        v->start_use();
    for ( double time : { .17, .37, .63 } )
    {
        p->set_time( time );
        ops->assign_scalar( .8, *in );
        ops->assign_scalar( .3, *direction );
        const double epsilon = 3e-6;
        ops->assign_lin_comb( 1., *in, epsilon, *direction, *plus );
        ops->assign_lin_comb( 1., *in, -epsilon, *direction, *minus );
        p->evaluate_rate( *plus, *fp );
        p->evaluate_rate( *minus, *fm );
        ops->assign_lin_comb( 1 / ( 2 * epsilon ), *fp, -1 / ( 2 * epsilon ), *fm, *plus );
        // With a=0,b=1 the normalized shifted operator is exactly -Dq.
        p->set_affine_shift( 0., 1. );
        p->set_linearization_point( *in );
        p->jacobi->apply( *direction, *exact );
        ops->add_lin_comb( 1., *exact, *plus );
        check( ops->norm( *plus ) < 2e-9, "Variable-mass rate Jacobian failed finite differences" );
        p->set_time( time + epsilon );
        p->evaluate_rate( *in, *fp );
        p->set_time( time - epsilon );
        p->evaluate_rate( *in, *fm );
        p->set_time( time );
        p->evaluate_time_derivative( *in, *exact );
        ops->assign_lin_comb( 1 / ( 2 * epsilon ), *fp, -1 / ( 2 * epsilon ), *fm, *minus );
        ops->add_lin_comb( -1., *exact, *minus );
        const double time_error = ops->norm( *minus );
        if ( time_error >= 2e-9 )
            std::cout << "q_t finite-difference error=" << time_error << " at t=" << time << '\n';
        check( time_error < 2e-9, "Variable-mass partial q_t failed finite differences" );
        p->model.omit_mass_derivative = true;
        p->set_linearization_point( *in );
        if constexpr ( variable_mass_autonomous<Model> )
            p->jacobi->apply( *direction, *minus );
        else
            p->evaluate_time_derivative( *in, *minus );
        if constexpr ( variable_mass_autonomous<Model> )
        {
            p->model.omit_mass_derivative = false;
            p->set_linearization_point( *in );
            p->jacobi->apply( *direction, *exact );
        }
        ops->add_lin_comb( -1., *exact, *minus );
        check( ops->norm( *minus ) > 1e-3, "Missing mass derivative was not detected" );
        p->model.omit_mass_derivative = false;
    }
}

template <class Model, bool Dense = true>
std::array<double, 2> trajectory( const char *method, double h, double factor = 1 )
{
    using P                    = variable_mass_problem<Model>;
    using S                    = rosenbrock::time_step<operations_type, P, adaptation_type, direct_solver<P>, Dense>;
    auto ops                   = std::make_shared<operations_type>( 2 );
    auto p                     = std::make_shared<P>( ops );
    p->model.factor            = factor;
    auto               control = std::make_shared<adaptation_type>( adaptation_type::params{ std::abs( h ) } );
    typename S::params sp;
    sp.method     = method;
    sp.autonomous = variable_mass_autonomous<Model>;
    S                step( ops, p, control, std::make_shared<direct_solver<P>>( ops ), sp );
    vector_wrap_type in( *ops ), out( *ops ), sample( *ops );
    in.start_use();
    out.start_use();
    sample.start_use();
    const double start = .37, end = start + std::copysign( .4, h );
    operations_type::for_each_type{}( variable_mass_initial<Model, vector_type>{ start, *in }, 2 );
    std::array<double, 2> error{};
    const auto            count = static_cast<unsigned int>( std::lround( .4 / std::abs( h ) ) );
    for ( unsigned int n = 0; n < count; ++n )
    {
        const double t = start + n * h;
        step.set_time( t );
        step.set_target_time( end );
        step.apply( *in, *out );
        check( step.get_status() == single_step_status::converged, "Variable-mass rate step failed" );
        check( p->time == t && p->jacobi->model.time == t, "Variable mass corrupted frozen base time" );
        if constexpr ( Dense )
        {
            auto         view        = step.get_continuous_integration();
            const double sample_time = view.evaluate( .43, *sample );
            for ( unsigned int i = 0; i < 2; ++i )
                error[1] = std::max(
                    error[1],
                    std::abs(
                        ops->get_value_at_point( i, *sample ) - variable_mass_exact<Model>( i, sample_time, start )
                    )
                );
        }
        const auto endpoint = step.get_dt() == end - t ? end : t + step.get_dt();
        step.finalize( adaptation_status::accepted, endpoint, *out );
        ops->assign( *out, *in );
    }
    for ( unsigned int i = 0; i < 2; ++i )
        error[0] = std::max(
            error[0], std::abs( ops->get_value_at_point( i, *out ) - variable_mass_exact<Model>( i, end, start ) )
        );
    check(
        p->linearizations == count && p->residuals == 0 && p->derivatives == ( sp.autonomous ? 0 : count ),
        "Variable mass must use normalized rates, not a native mass residual"
    );
    return error;
}

template <class Model>
void orders()
{
    derivatives<Model>();
    for ( const char *method : { "ROS2", "ROS3", "ROS34PW2" } )
        for ( double sign : { 1., -1. } )
        {
            // Backward time-mass dense errors enter their asymptotic range
            // later than endpoint errors; retain the same required order.
            const double h     = std::is_same_v<Model, time_mass_rate_model> && sign < 0 ? .02 : .04;
            const auto   a     = trajectory<Model>( method, sign * h );
            const auto   b     = trajectory<Model>( method, sign * h / 2 );
            const auto   c     = trajectory<Model>( method, sign * h / 4 );
            const auto   order = rosenbrock::method_table( method ).order();
            std::cout << method << " model="
                      << ( variable_mass_autonomous<Model>               ? "state"
                           : std::is_same_v<Model, time_mass_rate_model> ? "time"
                                                                         : "coupled" )
                      << " sign=" << sign << " primary=" << a[0] << ',' << b[0] << ',' << c[0] << " dense=" << a[1]
                      << ',' << b[1] << ',' << c[1] << '\n';
            for ( unsigned int i = 0; i < 2; ++i )
                check(
                    a[i] / b[i] > std::pow( 2, order - .4 ) && b[i] / c[i] > std::pow( 2, order - .4 ),
                    "Variable-mass primary/dense order was lost"
                );
            const auto plain = trajectory<Model, false>( method, sign * h );
            check( std::abs( plain[0] - a[0] ) < 1e-13, "Dense preparation changed variable-mass endpoint" );
        }
    const auto a = trajectory<Model>( "ROS34PW2", .02, .5 );
    const auto b = trajectory<Model>( "ROS34PW2", .01, .5 );
    for ( unsigned int i = 0; i < 2; ++i )
        check( a[i] / b[i] > 6., "Variable-mass W approximation lost third order" );
}

int main()
try
{
    backend_type::init_device();
    orders<state_mass_rate_model>();
    orders<time_mass_rate_model>();
    orders<coupled_variable_mass_model>();
    std::cout << "PASS: Rosenbrock variable-mass rates and dense output (" << checks << " checks)\n";
}
catch ( const std::exception &e )
{
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
