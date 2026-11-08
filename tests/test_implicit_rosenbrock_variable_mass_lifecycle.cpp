#include "implicit_rosenbrock_variable_mass_problem.h"
#include <nmfd/time_steppers/integration/time_step_adaptation_scipy.h>

using namespace rosenbrock_tests;

// Coupled time/state mass, with failures injected into the problem-owned rate
// solve, partial derivative and shifted solve (including dense-only work).
// Numerical failure must not publish either component or a borrowed dense view.
template <bool Dense>
void lifecycle( const char *method )
{
    using P                    = variable_mass_problem<coupled_variable_mass_model>;
    using L                    = direct_solver<P>;
    using S                    = rosenbrock::time_step<operations_type, P, adaptation_type, L, Dense>;
    auto               ops     = std::make_shared<operations_type>( 2 );
    auto               p       = std::make_shared<P>( ops );
    auto               solver  = std::make_shared<L>( ops );
    auto               control = std::make_shared<adaptation_type>( adaptation_type::params{ .02 } );
    typename S::params sp;
    sp.method     = method;
    sp.autonomous = false;
    vector_wrap_type in( *ops ), out( *ops );
    in.start_use();
    out.start_use();
    ops->assign_scalar( 1., *in );
    const rosenbrock::method_table table( method );
    const unsigned int extra = Dense ? ( table.stage_count() == 4 ? 2 : table.stage_count() == 3 ? 1 : 0 ) : 0;
    for ( unsigned int stage = 1; stage <= table.stage_count() + extra; ++stage )
        for ( unsigned int kind = 0; kind < 4; ++kind )
        {
            p->rates = p->derivatives = solver->calls = p->linearizations = 0;
            p->fail_rate_on = p->invalid_rate_on = solver->fail_on = solver->invalid_on = 0;
            const unsigned int rates = ( table.stage_count() == 4 ? 4 : 2 ) + ( Dense && extra ? 1 : 0 );
            if ( kind == 0 )
                solver->fail_on = stage;
            if ( kind == 1 )
                solver->invalid_on = stage;
            if ( kind == 2 )
                p->fail_rate_on = std::min( stage, rates );
            if ( kind == 3 )
                p->invalid_rate_on = std::min( stage, rates );
            S step( ops, p, control, solver, sp );
            step.set_time( .37 );
            ops->assign_scalar( 77., *out );
            step.apply( *in, *out );
            check(
                step.get_status() != single_step_status::converged && p->time == .37,
                "Failed variable-mass callback converged or changed base time"
            );
            for ( unsigned int i = 0; i < 2; ++i )
                check(
                    ops->get_value_at_point( i, *out ) == 77. && ops->get_value_at_point( i, *in ) == 1.,
                    "Failed variable-mass attempt published partial output"
                );
            if constexpr ( Dense )
            {
                bool refused = false;
                try
                {
                    step.get_continuous_integration();
                }
                catch ( const std::logic_error & )
                {
                    refused = true;
                }
                check( refused, "Failed variable-mass attempt published a dense view" );
            }
            p->fail_rate_on = p->invalid_rate_on = solver->fail_on = solver->invalid_on = 0;
            step.apply( *in, *out );
            check(
                step.get_status() == single_step_status::converged && p->linearizations == 2,
                "Variable-mass recovery failed to reprepare its Jacobian"
            );
            step.finalize( adaptation_status::accepted_modified, .38, *in );
            p->model.lambda = 2;
            step.set_time( .38 );
            step.apply( *in, *out );
            check(
                step.get_status() == single_step_status::converged && p->jacobi->model.lambda == 2,
                "Variable-mass rebinding retained stale problem data"
            );
            step.finalize( adaptation_status::rejected, .38, *in );
        }
    for ( unsigned int kind = 0; kind < 4; ++kind )
    {
        p->fail_derivative_on = p->invalid_derivative_on = 0;
        p->throw_derivative = p->throw_rate = false;
        p->derivatives                      = 0;
        if ( kind == 0 )
            p->fail_derivative_on = 1;
        if ( kind == 1 )
            p->invalid_derivative_on = 1;
        if ( kind == 2 )
            p->throw_derivative = true;
        if ( kind == 3 )
            p->throw_rate = true;
        S step( ops, p, control, solver, sp );
        step.set_time( .37 );
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
        check(
            threw == ( kind >= 2 ) && p->time == .37 && ops->get_value_at_point( 0, *out ) == 77.,
            "Variable-mass partial-derivative/exception rollback failed"
        );
    }
}

void adaptive_retry()
{
    using P          = variable_mass_problem<coupled_variable_mass_model>;
    using C          = integration::time_step_adaptation_scipy<operations_type>;
    using S          = rosenbrock::time_step<operations_type, P, C, direct_solver<P>>;
    auto      ops    = std::make_shared<operations_type>( 2 );
    auto      p      = std::make_shared<P>( ops );
    auto      solver = std::make_shared<direct_solver<P>>( ops );
    C::params cp;
    cp.relative_tolerance = cp.absolute_tolerance = .1;
    cp.initial_step = cp.maximum_step = .05;
    S::params sp;
    sp.method       = "ROS34PW2";
    sp.autonomous   = false;
    p->fail_rate_on = 1;
    S                step( ops, p, std::make_shared<C>( ops, cp ), solver, sp );
    vector_wrap_type in( *ops ), out( *ops );
    in.start_use();
    out.start_use();
    ops->assign_scalar( 1., *in );
    step.apply( *in, *out );
    check(
        step.get_status() == single_step_status::converged && step.get_attempts() == 2 && step.get_dt() < .05 &&
            p->linearizations == 2 && solver->bindings == 2,
        "Problem-owned mass-solve failure did not shrink/rebind and recover"
    );
}

int main()
try
{
    backend_type::init_device();
    for ( const char *method : { "ROS2", "ROS3", "ROS34PW2" } )
    {
        lifecycle<false>( method );
        lifecycle<true>( method );
    }
    adaptive_retry();
    std::cout << "PASS: Rosenbrock variable-mass lifecycle (" << checks << " checks)\n";
}
catch ( const std::exception &e )
{
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
