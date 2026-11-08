#include "implicit_rosenbrock_mass_time_problem.h"
#include <nmfd/time_steppers/integration/time_step_adaptation_scipy.h>

using namespace rosenbrock_tests;

// Manufactured coupled M*u'=M*q(t,u), exact (exp(t),exp(2*t)). Injected
// derivative, stage and dense-only failures must leave committed data intact
// and restore base time, including when evaluation throws at the endpoint.
void failures( const char *method )
{
    using P              = mass_time_problem<>;
    using L              = direct_solver<P>;
    using S              = rosenbrock::time_step<operations_type, P, adaptation_type, L, true>;
    auto             ops = std::make_shared<operations_type>( 2 );
    vector_wrap_type in( *ops ), out( *ops ), sample( *ops );
    in.start_use();
    out.start_use();
    sample.start_use();
    ops->assign_scalar( 1., *in );
    const auto stages      = rosenbrock::method_table( method ).stage_count();
    const auto evaluations = stages + ( stages == 4 ? 2 : stages == 3 ? 1 : 0 );
    for ( unsigned int failure = 0; failure < 3 + 4 * evaluations; ++failure )
    {
        auto      p      = std::make_shared<P>( ops );
        auto      solver = std::make_shared<L>( ops );
        S::params sp;
        sp.method     = method;
        sp.mass_form  = true;
        sp.autonomous = false;
        S step( ops, p, std::make_shared<adaptation_type>( adaptation_type::params{ .1 } ), solver, sp );
        step.set_time( .37 );
        step.set_target_time( .4 );
        const bool throws = failure == 2 || ( failure >= 3 && ( failure - 3 ) % 4 == 1 );
        if ( failure == 0 )
            p->fail_derivative_on = 1;
        else if ( failure == 1 )
            p->invalid_derivative_on = 1;
        else if ( failure == 2 )
            p->throw_derivative = true;
        else
        {
            const auto evaluation = ( failure - 3 ) / 4 + 1;
            const auto kind       = ( failure - 3 ) % 4;
            if ( kind == 0 )
                p->invalid_on = evaluation;
            if ( kind == 1 )
                p->throw_on = evaluation;
            if ( kind == 2 )
                solver->fail_on = evaluation;
            if ( kind == 3 )
                solver->invalid_on = evaluation;
        }
        ops->assign_scalar( 77., *out );
        bool threw = false;
        try
        {
            step.apply( *in, *out );
        }
        catch ( const std::runtime_error & )
        {
            threw = true;
        }
        check( threw == throws, "Mass time callback exception was lost" );
        check(
            ops->get_value_at_point( 0, *out ) == 77. && ops->get_value_at_point( 0, *in ) == 1.,
            "Mass time failure changed committed input/output"
        );
        check(
            p->time == .37 && p->jacobi->model.time == .37 && p->derivative_time == .37,
            "Mass time failure left a stage time active or corrupted linearization"
        );
        bool unavailable = false;
        try
        {
            step.get_continuous_integration();
        }
        catch ( const std::logic_error & )
        {
            unavailable = true;
        }
        check( unavailable, "Failed mass time preparation published a dense view" );
        p->fail_derivative_on = p->invalid_derivative_on = p->invalid_on = p->throw_on = 0;
        p->throw_derivative                                                            = false;
        solver->fail_on = solver->invalid_on = 0;
        step.reset();
        step.apply( *in, *out );
        check(
            step.get_status() == single_step_status::converged && p->derivatives == 2 && solver->bindings == 2,
            "Mass time recovery reused a derivative or shifted operator"
        );
        auto         view      = step.get_continuous_integration();
        const double committed = view.evaluate( .5, *sample );
        step.finalize( adaptation_status::accepted_modified, committed, *sample );
        bool expired = false;
        try
        {
            view.evaluate( .5, *sample );
        }
        catch ( const std::logic_error & )
        {
            expired = true;
        }
        check( expired, "Modified mass time commit left a dense view valid" );
        step.set_time( committed );
        step.set_target_time( .4 );
        step.apply( *sample, *sample );
        check(
            step.get_status() == single_step_status::converged && p->time == committed &&
                p->derivative_time == committed && p->jacobi->model.time == committed,
            "Mass time modified/in-place continuation retained old base data"
        );
        step.finalize( adaptation_status::accepted, .4, *sample );
    }
}

// Failed q_t, endpoint residuals and shifted solves all trigger the same bounded
// numerical retry. The controller must neither commit nor assess a failed trial.
void retry( const char *method )
{
    using P              = mass_time_problem<>;
    using L              = direct_solver<P>;
    using C              = integration::time_step_adaptation_scipy<operations_type>;
    using S              = rosenbrock::time_step<operations_type, P, C, L, true>;
    auto             ops = std::make_shared<operations_type>( 2 );
    vector_wrap_type in( *ops ), out( *ops );
    in.start_use();
    out.start_use();
    ops->assign_scalar( 1., *in );
    for ( unsigned int kind = 0; kind < 3; ++kind )
    {
        auto p      = std::make_shared<P>( ops );
        auto solver = std::make_shared<L>( ops );
        if ( kind == 0 )
            p->fail_derivative_on = 1;
        if ( kind == 1 )
            p->invalid_on = rosenbrock::method_table( method ).stage_count() == 2
                                ? 2
                                : rosenbrock::method_table( method ).stage_count() + 1;
        if ( kind == 2 )
            solver->fail_on = 2;
        C::params cp;
        cp.relative_tolerance = cp.absolute_tolerance = .1;
        cp.initial_step = cp.maximum_step = .05;
        S::params sp;
        sp.method     = method;
        sp.mass_form  = true;
        sp.autonomous = false;
        S step( ops, p, std::make_shared<C>( ops, cp ), solver, sp );
        step.set_time( .37 );
        step.set_target_time( .42 );
        step.apply( *in, *out );
        check(
            step.get_status() == single_step_status::converged && step.get_attempts() > 1 && step.get_dt() < .05,
            "Mass time numerical failure did not shrink/retry"
        );
        check(
            p->derivatives == step.get_attempts() && solver->bindings == step.get_attempts() && p->time == .37 &&
                p->prepared_time == .37 && ops->get_value_at_point( 0, *in ) == 1.,
            "Mass time retry retained a stale derivative, shift or state"
        );
    }
}

int main()
try
{
    backend_type::init_device();
    for ( const char *method : { "ROS2", "ROS3", "ROS34PW2" } )
    {
        failures( method );
        retry( method );
    }
    std::cout << "PASS: Rosenbrock mass time lifecycle (" << checks << " checks)\n";
}
catch ( const std::exception &e )
{
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
