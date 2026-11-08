#include "implicit_rosenbrock_time_problem.h"
#include <nmfd/time_steppers/integration/time_step_adaptation_scipy.h>

using namespace rosenbrock_tests;

// These tests use the manufactured exponential system documented in the
// companion problem header. They distinguish partial f_t from d(f(t,u(t)))/dt,
// and require a frozen base Jacobian even though j(t) varies between stages.
using P = time_problem<>;
using L = direct_solver<P>;
using S = rosenbrock::time_step<operations_type, P, adaptation_type, L>;

struct wrong_derivative_result : problem<decay>
{
    void evaluate_time_derivative( const vector_type &, vector_type & )
    {
    }
};
static_assert( detail::has_time_derivative_evaluation<P, vector_type>::value );
static_assert( !detail::has_time_derivative_evaluation<problem<decay>, vector_type>::value );
static_assert( !detail::has_time_derivative_evaluation<wrong_derivative_result, vector_type>::value );

double fixed_error( bool varying, double start, double end, double h )
{
    auto ops          = std::make_shared<operations_type>( 1 );
    auto p            = std::make_shared<P>( ops );
    p->varying        = varying;
    auto      control = std::make_shared<adaptation_type>( adaptation_type::params{ h } );
    auto      linear  = std::make_shared<L>( ops );
    S::params params;
    params.autonomous     = false;
    auto             step = std::make_shared<S>( ops, p, control, linear, params );
    vector_wrap_type in( *ops ), out( *ops );
    in.start_use();
    out.start_use();
    ops->assign_scalar( std::exp( 2 * start ), *in );
    integration::time_integrator<operations_type, S> run( ops, step );
    run.set_time_interval( start, end );
    run.apply( *in, *out );
    check(
        run.get_status() == integration_status::completed && run.get_final_time() == end,
        "Non-autonomous ROS2 missed its target"
    );
    check(
        p->derivatives == run.get_steps() && p->rates == 2 * p->derivatives && linear->bindings == p->derivatives,
        "ROS2 did not prepare exactly one base derivative/Jacobian per step"
    );
    check(
        p->time == p->prepared_time && p->derivative_time == p->prepared_time, "ROS2 did not restore its base time"
    );
    return std::abs( ops->get_value_at_point( 0, *out ) - std::exp( 2 * end ) );
}

void test_order_and_timing()
{
    for ( bool varying : { false, true } )
        for ( bool backward : { false, true } )
        {
            const double start = .37, end = backward ? .17 : .57;
            // Keep lambda*h small for the stiff order check. Backward relaxation
            // is unstable; this checks discretization order, not reverse stability.
            const double h  = varying ? .02 : .0005;
            const double e1 = fixed_error( varying, start, end, h );
            const double e2 = fixed_error( varying, start, end, h / 2 );
            std::cout << "ROS2 time varying=" << varying << " backward=" << backward << " h=" << h << " errors=" << e1
                      << ',' << e2 << " order=" << std::log2( e1 / e2 ) << '\n';
            check( e1 / e2 > 3.7 && e1 / e2 < 4.3, "Non-autonomous ROS2 lost second-order accuracy" );
        }
    auto      ops = std::make_shared<operations_type>( 1 );
    auto      p   = std::make_shared<P>( ops );
    S::params params;
    params.autonomous = false;
    S step(
        ops, p, std::make_shared<adaptation_type>( adaptation_type::params{ .1 } ), std::make_shared<L>( ops ), params
    );
    vector_wrap_type in( *ops ), out( *ops );
    in.start_use();
    out.start_use();
    ops->assign_scalar( std::exp( .74 ), *in );
    step.set_time( .37 );
    step.set_target_time( .4 );
    step.apply( *in, *out );
    check(
        step.get_status() == single_step_status::converged && p->first_rate_time == .37 && p->last_rate_time == .4 &&
            p->prepared_time == .37 && p->derivative_time == .37 && p->jacobi->model.lambda == -.37,
        "ROS2 used the wrong clipped stage time or changed its frozen Jacobian"
    );
    step.finalize( adaptation_status::accepted, .4, *out );
}

struct retry_controller : adaptation_type
{
    double       next     = .1;
    unsigned int failures = 0;
    double       initialize( double, const vector_type & )
    {
        return next;
    }
    double get_dt() const
    {
        return next;
    }
    adaptation_status reject_step( double h )
    {
        ++failures;
        next = std::abs( h ) / 2;
        return adaptation_status::rejected;
    }
};

void test_failures()
{
    auto             ops = std::make_shared<operations_type>( 1 );
    vector_wrap_type in( *ops ), out( *ops );
    in.start_use();
    out.start_use();
    ops->assign_scalar( 1., *in );
    S::params settings;
    settings.autonomous = false;
    for ( bool derivative : { false, true } )
        for ( bool invalid : { false, true } )
        {
            auto p = std::make_shared<P>( ops );
            if ( derivative )
            {
                if ( invalid )
                    p->invalid_derivative_on = 1;
                else
                    p->fail_derivative_on = 1;
            }
            else
            {
                if ( invalid )
                    p->invalid_rate_on = 2;
                else
                    p->fail_rate_on = 2;
            }
            ops->assign_scalar( 42., *out );
            S step( ops, p, std::make_shared<adaptation_type>(), std::make_shared<L>( ops ), settings );
            step.apply( *in, *out );
            check(
                step.get_status() ==
                        ( invalid ? single_step_status::failed_nonfinite : single_step_status::failed_solver ) &&
                    ops->get_value_at_point( 0, *out ) == 42 && p->time == 0,
                "ROS2 derivative/rate failure changed output or left endpoint time active"
            );
            p->fail_derivative_on = p->invalid_derivative_on = p->fail_rate_on = p->invalid_rate_on = 0;
            step.reset();
            step.apply( *in, *out );
            check(
                step.get_status() == single_step_status::converged, "ROS2 did not recover after a time-provider failure"
            );
            step.finalize( adaptation_status::accepted, step.get_dt(), *out );
        }
    for ( bool derivative : { false, true } )
    {
        auto p              = std::make_shared<P>( ops );
        p->throw_derivative = derivative;
        if ( !derivative )
            p->throw_rate_on = 2;
        ops->assign_scalar( 42., *out );
        S    step( ops, p, std::make_shared<adaptation_type>(), std::make_shared<L>( ops ), settings );
        bool failed = false;
        try
        {
            step.apply( *in, *out );
        }
        catch ( const std::runtime_error & )
        {
            failed = true;
        }
        check(
            failed && p->time == 0 && ops->get_value_at_point( 0, *out ) == 42,
            "ROS2 did not propagate a provider exception transactionally"
        );
    }
    auto p                = std::make_shared<P>( ops );
    p->fail_derivative_on = 1;
    auto control          = std::make_shared<retry_controller>();
    using Retry           = rosenbrock::time_step<operations_type, P, retry_controller, L>;
    Retry::params params;
    params.autonomous = false;
    Retry retry( ops, p, control, std::make_shared<L>( ops ), params );
    retry.apply( *in, *out );
    check(
        retry.get_status() == single_step_status::converged && retry.get_attempts() == 2 && control->failures == 1 &&
            p->derivatives == 2 && p->time == 0 && p->last_rate_time == .05,
        "ROS2 reused a failed time derivative or failed to reclip the retried stage"
    );
    retry.finalize( adaptation_status::accepted, .05, *out );
}

void test_adaptive_time_problem()
{
    auto ops   = std::make_shared<operations_type>( 1 );
    auto p     = std::make_shared<P>( ops );
    using C    = integration::time_step_adaptation_scipy<operations_type>;
    using Step = rosenbrock::time_step<operations_type, P, C, L>;
    C::params parameters;
    parameters.relative_tolerance = 2e-6;
    parameters.absolute_tolerance = 1e-8;
    parameters.initial_step = parameters.maximum_step = .2;
    auto         control                              = std::make_shared<C>( ops, parameters );
    Step::params settings;
    settings.autonomous = false;
    auto step           = std::make_shared<Step>( ops, p, control, std::make_shared<L>( ops ), settings );
    integration::time_integrator<operations_type, Step> run( ops, step );
    vector_wrap_type                                    in( *ops ), out( *ops );
    in.start_use();
    out.start_use();
    for ( bool backward : { false, true } )
    {
        const double start = backward ? .57 : .37, end = backward ? .37 : .57;
        ops->assign_scalar( std::exp( 2 * start ), *in );
        run.set_time_interval( start, end );
        for ( int restart = 0; restart < 2; ++restart )
        {
            run.apply( *in, *out );
            check(
                run.get_status() == integration_status::completed && run.get_final_time() == end &&
                    std::abs( ops->get_value_at_point( 0, *out ) - std::exp( 2 * end ) ) < 2e-5,
                "Adaptive non-autonomous ROS2 reference mismatch"
            );
        }
    }
}

int main()
try
{
    backend_type::init_device();
    test_order_and_timing();
    test_failures();
    test_adaptive_time_problem();
    std::cout << "Non-autonomous ROS2 partial derivatives, stage timing, refinement and rollback: " << checks
              << " checks PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
