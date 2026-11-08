#include "ode_benchmark_rosenbrock.h"

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;

// Exercise the actual visual-runner facades, not just equivalent analytical
// fixtures. Physical equations remain unchanged; only the Rosenbrock residual
// and shifted Jacobian are normalized to q=M^-1*f. All systems are scalar.
unsigned int checks = 0;
void         check( bool condition, const char *message )
{
    ++checks;
    require( condition, message );
}

template <class Equation>
void benchmark_facade()
{
    using P                     = benchmark_rosenbrock_problem<Equation>;
    using L                     = typename newton_types<P>::linear_type;
    using S                     = rosenbrock::time_step<operations_type, P, adaptation_type, L, true>;
    auto                    ops = std::make_shared<operations_type>( 1 );
    auto                    p   = std::make_shared<P>( ops );
    stage_problem<Equation> physical( ops );
    vector_wrap_type        in( *ops ), out( *ops ), raw( *ops ), direction( *ops );
    for ( auto *v : { &in, &out, &raw, &direction } )
        v->start_use();
    ops->assign_scalar( .8, *in );
    ops->assign_scalar( 1., *direction );
    for ( double t : { .25, .63 } )
    {
        const double mass = std::is_same_v<Equation, variable_mass> ? 1 + .8 * .8 : 1 + t;
        const double q    = std::is_same_v<Equation, variable_mass> ? -.8 : -.8 / ( 1 + t );
        const double dq   = std::is_same_v<Equation, variable_mass> ? -1 : -1 / ( 1 + t );
        p->set_time( t );
        physical.set_time( t );
        p->set_affine_shift( .5, .1 );
        physical.set_affine_shift( .5, .1 );
        ops->assign_scalar( .6, p->previous_state() );
        ops->assign_scalar( .6, physical.previous_state() );
        ops->assign_scalar( 0., p->known_rhs() );
        ops->assign_scalar( 0., physical.known_rhs() );
        p->evaluate_rate( *in, *out );
        check( std::abs( ops->get_value_at_point( 0, *out ) - q ) < 1e-14, "Benchmark returned force instead of rate" );
        p->apply( *in, *out );
        physical.apply( *in, *raw );
        check(
            std::abs( ops->get_value_at_point( 0, *out ) * mass - ops->get_value_at_point( 0, *raw ) ) < 1e-14,
            "Benchmark normalization changed the physical stage equation"
        );
        p->set_linearization_point( *in );
        p->get_jacobi_operator()->apply( *direction, *out );
        check(
            std::abs( ops->get_value_at_point( 0, *out ) - ( .5 - .1 * dq ) ) < 1e-14,
            "Benchmark retained the physical-mass Jacobian in rate form"
        );
        p->evaluate_time_derivative( *in, *out );
        const double qt = std::is_same_v<Equation, variable_mass> ? 0 : .8 / ( ( 1 + t ) * ( 1 + t ) );
        check(
            std::abs( ops->get_value_at_point( 0, *out ) - qt ) < 1e-14, "Benchmark partial rate derivative is wrong"
        );
    }
    typename L::params lp;
    lp.basis_size            = 2;
    lp.batch_size            = 1;
    lp.monitor.rel_tol       = 1e-12;
    lp.monitor.max_iters_num = 20;
    auto linear              = std::make_shared<L>( ops, nullptr, lp );
    for ( const char *method : { "ROS2", "ROS3", "ROS34PW2" } )
    {
        typename S::params sp;
        sp.method      = method;
        sp.autonomous  = benchmark_rosenbrock_autonomous<Equation>;
        const double t = .37, h = 1e-4;
        S            step( ops, p, std::make_shared<adaptation_type>( adaptation_type::params{ h } ), linear, sp );
        step.set_time( t );
        step.set_target_time( t + h );
        step.apply( *in, *out );
        check( step.get_status() == single_step_status::converged, "Benchmark facade shifted solve failed" );
        auto view = step.get_continuous_integration();
        for ( double theta : { .37, 1. } )
        {
            const double time  = view.evaluate( theta, *out );
            const double exact = std::is_same_v<Equation, variable_mass> ? .8 * std::exp( -( time - t ) )
                                                                         : .8 * ( 1 + t ) / ( 1 + time );
            check(
                std::abs( ops->get_value_at_point( 0, *out ) - exact ) < 1e-8,
                "Benchmark facade failed analytical flow/dense check"
            );
        }
        step.finalize( adaptation_status::accepted, t + step.get_dt(), *out );
    }
}

int main()
try
{
    backend_type::init_device();
    benchmark_facade<variable_mass>();
    benchmark_facade<time_dependent_mass>();
    std::cout << "PASS: Rosenbrock variable-mass benchmark facades (" << checks << " checks)\n";
}
catch ( const std::exception &e )
{
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
