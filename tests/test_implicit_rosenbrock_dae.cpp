#include "implicit_rosenbrock_dae_problem.h"
#include <nmfd/time_steppers/integration/time_step_adaptation_filter.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_matlab.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_scipy.h>

using namespace rosenbrock_tests;

// Singular M=diag(mass,0), with z=x or z=x^2 as described in the support
// header. Track differential and algebraic errors independently; nonlinear
// constraints have an O(h^3) defect, not exact projection onto z=x^2.
std::array<double, 3> fixed_trajectory( bool nonlinear, double factor, double h, double mass = 2 )
{
    using P      = dae_problem<>;
    using S      = rosenbrock::time_step<operations_type, P, adaptation_type, direct_solver<P>>;
    auto ops     = std::make_shared<operations_type>( 2 );
    auto p       = std::make_shared<P>( ops );
    p->nonlinear = nonlinear;
    p->factor    = factor;
    p->mass      = mass;
    auto solver  = std::make_shared<direct_solver<P>>( ops );
    S    step(
        ops, p, std::make_shared<adaptation_type>( adaptation_type::params{ std::abs( h ) } ), solver, dae_params<S>()
    );
    vector_wrap_type in( *ops ), out( *ops ), scratch( *ops );
    in.start_use();
    out.start_use();
    scratch.start_use();
    ops->assign_scalar( 1., *in );
    const double          start = .37, duration = std::copysign( .4, h ), end = start + duration;
    const auto            count = static_cast<unsigned int>( std::lround( duration / h ) );
    std::array<double, 3> errors{};
    for ( unsigned int n = 0; n < count; ++n )
    {
        step.set_time( start + n * h );
        step.set_target_time( end );
        step.apply( *in, *out );
        check( step.get_status() == single_step_status::converged, "Index-1 fixed step failed" );
        check( p->jacobi->a == p->a && p->jacobi->b == p->b, "DAE stage changed frozen shift" );
        errors[2]           = std::max( errors[2], p->constraint_norm( *out, *scratch ) );
        const auto time     = start + n * h;
        const auto endpoint = step.get_dt() == end - time ? end : time + step.get_dt();
        step.finalize( adaptation_status::accepted, endpoint, *out );
        ops->assign( *out, *in );
    }
    errors[0] = std::abs( ops->get_value_at_point( 0, *out ) - std::exp( -duration ) );
    errors[1] = std::abs( ops->get_value_at_point( 1, *out ) - std::exp( -( nonlinear ? 2 : 1 ) * duration ) );
    check(
        p->linearizations == count && p->residuals == 4 * count && solver->calls == 4 * count,
        "Index-1 stages must use four full shifted solves per frozen operator"
    );
    if ( !nonlinear )
        check( errors[2] < 2e-13, "Linear algebraic constraint drifted" );
    return errors;
}

// Unscaled g and g' must remain visible for every differential scaling.
// The reference buffer and known RHS must not accidentally enter algebraic M.
void row_scaling()
{
    auto             ops = std::make_shared<operations_type>( 2 );
    auto             p   = std::make_shared<dae_problem<>>( ops );
    vector_wrap_type in( *ops ), out( *ops ), direction( *ops ), product( *ops );
    in.start_use();
    out.start_use();
    direction.start_use();
    product.start_use();
    ops->assign_scalar( 2., *in );
    ops->assign_scalar( 3., *direction );
    ops->assign_scalar( 77., p->previous_state() );
    ops->assign_scalar( 0., p->known_rhs() );
    for ( double a : { -1., 1e-12, 1e6 } )
        for ( double b : { 1e-12, 1., 1e6 } )
        {
            p->set_affine_shift( a, b );
            p->set_linearization_point( *in );
            p->apply( *in, *out );
            check( ops->get_value_at_point( 1, *out ) == -2., "DAE residual scaled its algebraic equation" );
            p->jacobi->apply( *direction, *out );
            check( ops->get_value_at_point( 1, *out ) == -9., "DAE Jacobian scaled its algebraic row" );
        }
    p->set_affine_shift( 1., .1 );
    p->set_linearization_point( *in );
    p->jacobi->solve( *direction, *out );
    p->jacobi->apply( *out, *product );
    ops->add_lin_comb( -1., *direction, *product );
    check( ops->norm( *product ) < 1e-13, "Full DAE shifted solve failed its true residual check" );
}

// On z=x, the full singular-mass solve must give the scalar stability function
// for x'=-x, including balanced small/large steps. This is a discrete reference,
// not an assertion that an unresolved large step reproduces exp(-h).
void balanced_steps()
{
    using P      = dae_problem<>;
    using S      = rosenbrock::time_step<operations_type, P, adaptation_type, direct_solver<P>>;
    auto ops     = std::make_shared<operations_type>( 2 );
    auto p       = std::make_shared<P>( ops );
    p->nonlinear = false;
    vector_wrap_type in( *ops ), out( *ops );
    in.start_use();
    out.start_use();
    ops->assign_scalar( 1., *in );
    for ( double reference_time : { 1e-6, 1., 1e6 } )
        for ( double h : { -1e-4, 1e-12, .01, 1., 1000. } )
        {
            auto sp           = dae_params<S>();
            sp.reference_time = reference_time;
            S step(
                ops, p, std::make_shared<adaptation_type>( adaptation_type::params{ std::abs( h ) } ),
                std::make_shared<direct_solver<P>>( ops ), sp
            );
            step.set_target_time( h );
            step.apply( *in, *out );
            check( step.get_status() == single_step_status::converged, "Balanced index-1 step failed" );
            const double g = step.table().gamma, z = -h;
            const double exact = ( 1 + ( 1 - 4 * g ) * z + ( .5 - 4 * g + 6 * g * g ) * z * z +
                                   ( 1. / 6 - 2 * g + 6 * g * g - 4 * g * g * g ) * z * z * z ) /
                                 std::pow( 1 - g * z, 4 );
            for ( unsigned int i = 0; i < 2; ++i )
                check(
                    std::abs( ops->get_value_at_point( i, *out ) - exact ) < 2e-12,
                    "Full singular-mass step disagrees with scalar stability function"
                );
        }
}

struct capture_estimate : adaptation_type
{
    std::shared_ptr<operations_type> ops;
    vector_wrap_type                 error;
    capture_estimate( std::shared_ptr<operations_type> o, double h )
        : adaptation_type( params{ h } ), ops( std::move( o ) ), error( *ops )
    {
        error.start_use();
    }
    bool requires_error_estimate() const
    {
        return true;
    }
    adaptation_status assess(
        double, double, const vector_type &, const vector_type &, double &, unsigned int order = 0,
        const vector_type *estimate = nullptr
    )
    {
        check( order == 3 && estimate, "DAE estimator did not provide its order-2 embedded defect" );
        ops->assign( *estimate, *error );
        return adaptation_status::accepted;
    }
};

// From exact initial data, the embedded state has a third-order local error
// in both components. The nonlinear primary algebraic error is also O(h^3),
// so state-error acceptance cannot replace a separate constraint tolerance.
void embedded_orders( bool nonlinear )
{
    using P      = dae_problem<>;
    using S      = rosenbrock::time_step<operations_type, P, capture_estimate, direct_solver<P>>;
    auto ops     = std::make_shared<operations_type>( 2 );
    auto p       = std::make_shared<P>( ops );
    p->nonlinear = nonlinear;
    vector_wrap_type in( *ops ), out( *ops );
    in.start_use();
    out.start_use();
    ops->assign_scalar( 1., *in );
    std::array<double, 2> previous{};
    for ( double h : { .04, .02, .01 } )
    {
        auto control = std::make_shared<capture_estimate>( ops, h );
        S    step( ops, p, control, std::make_shared<direct_solver<P>>( ops ), dae_params<S>() );
        step.apply( *in, *out );
        check( step.get_status() == single_step_status::converged, "DAE embedded test failed" );
        ops->add_lin_comb( -1., *control->error, *out );
        for ( unsigned int i = 0; i < 2; ++i )
        {
            const double error =
                std::abs( ops->get_value_at_point( i, *out ) - std::exp( -h * ( i && nonlinear ? 2 : 1 ) ) );
            if ( previous[i] )
                check( previous[i] / error > 6., "DAE embedded local order was lost" );
            previous[i] = error;
        }
    }
}

template <class Controller>
double adaptive_trajectory( double tolerance, const char *filter = nullptr )
{
    using P                         = dae_problem<>;
    using C                         = dae_control<Controller, P>;
    using S                         = rosenbrock::time_step<operations_type, P, C, direct_solver<P>>;
    auto                        ops = std::make_shared<operations_type>( 2 );
    auto                        p   = std::make_shared<P>( ops );
    typename Controller::params cp;
    cp.relative_tolerance = tolerance;
    cp.absolute_tolerance = tolerance * .01;
    cp.initial_step = cp.maximum_step = .2;
    if constexpr ( std::is_same_v<Controller, integration::time_step_adaptation_filter<operations_type>> )
        cp.controller = filter;
    auto controller                  = std::make_shared<C>( p, cp );
    controller->constraint_tolerance = tolerance * .1;
    auto step = std::make_shared<S>( ops, p, controller, std::make_shared<direct_solver<P>>( ops ), dae_params<S>() );
    integration::time_integrator<operations_type, S> run( ops, step );
    run.set_time_interval( .37, .77 );
    vector_wrap_type in( *ops ), out( *ops ), scratch( *ops );
    in.start_use();
    out.start_use();
    scratch.start_use();
    ops->assign_scalar( 1., *in );
    double      first = 0;
    std::size_t steps = 0;
    for ( unsigned int repeat = 0; repeat < 2; ++repeat )
    {
        run.apply( *in, *out );
        check(
            run.get_status() == integration_status::completed && run.get_final_time() == .77,
            "Adaptive index-1 trajectory failed"
        );
        check(
            p->constraint_norm( *out, *scratch ) <= controller->constraint_tolerance,
            "DAE acceptance ignored the independent constraint tolerance"
        );
        const double result = ops->get_value_at_point( 0, *out );
        if ( !repeat )
        {
            first = result;
            steps = run.get_steps();
        }
        else
            check( first == result && steps == run.get_steps(), "DAE restart was not deterministic" );
        check( ops->get_value_at_point( 0, *in ) == 1., "DAE integration modified its initial state" );
    }
    check( controller->constraint_rejections > 0, "DAE constraint rejection was not exercised" );
    return std::max(
        std::abs( first - std::exp( -.4 ) ), std::abs( ops->get_value_at_point( 1, *out ) - std::exp( -.8 ) )
    );
}

int main()
try
{
    backend_type::init_device();
    static_assert(
        !detail::has_rate_evaluation<dae_problem<>, vector_type>::value,
        "Index-1 regression must not provide a singular mass inverse"
    );
    const rosenbrock::method_table table( "ROS34PW2" );
    check(
        table.supports_constant_mass_index1() && !rosenbrock::method_table( "ROS2" ).supports_constant_mass_index1() &&
            !rosenbrock::method_table( "ROS3" ).supports_constant_mass_index1(),
        "ODE orders silently enabled unsupported DAE tables"
    );
    // Index-1 algebraic order condition b^T*(A+Gamma)^-1*c^2=1.
    // In transformed coordinates this is m^T*(I+a)^-1*c^2=1.
    std::array<double, 4> w{};
    double                condition = 0;
    for ( unsigned int i = 0; i < 4; ++i )
    {
        w[i] = table.stage_time( i ) * table.stage_time( i );
        for ( unsigned int j = 0; j < i; ++j )
            w[i] -= table.a( i, j ) * w[j];
        condition += table.m( i ) * w[i];
    }
    check( std::abs( condition - 1 ) < 2e-14, "ROS34PW2 index-1 algebraic order condition failed" );
    row_scaling();
    balanced_steps();
    embedded_orders( false );
    embedded_orders( true );
    for ( bool nonlinear : { false, true } )
        for ( double factor : { 1., .5 } )
            for ( double sign : { 1., -1. } )
            {
                // The backward approximate-Jacobian constraint defect has a
                // coarse-grid cancellation; measure its asymptotic order later.
                const double h = nonlinear && factor != 1 && sign < 0 ? .01 : .04;
                const auto   a = fixed_trajectory( nonlinear, factor, sign * h );
                const auto   b = fixed_trajectory( nonlinear, factor, sign * h / 2 );
                const auto   c = fixed_trajectory( nonlinear, factor, sign * h / 4 );
                for ( unsigned int i = 0; i < ( nonlinear ? 3u : 2u ); ++i )
                    check( a[i] / b[i] > 6. && b[i] / c[i] > 6., "Index-1 differential/algebraic order was lost" );
                const auto scaled = fixed_trajectory( nonlinear, factor, sign * h, 1e-6 );
                for ( unsigned int i = 0; i < 3; ++i )
                    check( std::abs( scaled[i] - a[i] ) < 5e-13, "Differential mass scaling changed DAE accuracy" );
            }
    auto refine = []( auto tag, const char *filter = nullptr ) {
        using C             = typename decltype( tag )::type;
        const double coarse = adaptive_trajectory<C>( 1e-3, filter );
        const double fine   = adaptive_trajectory<C>( 1e-5, filter );
        check( fine < coarse * .1 && fine < 3e-6, "DAE adaptive tolerance refinement failed" );
    };
    refine( std::common_type<integration::time_step_adaptation_matlab<operations_type>>{} );
    refine( std::common_type<integration::time_step_adaptation_scipy<operations_type>>{} );
    for ( const char *filter : { "I", "H211", "PC", "PID", "H312", "PPID", "H321" } )
        refine( std::common_type<integration::time_step_adaptation_filter<operations_type>>{}, filter );
    std::cout << "PASS: Rosenbrock index-1 DAE (" << checks << " checks)\n";
}
catch ( const std::exception &e )
{
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
