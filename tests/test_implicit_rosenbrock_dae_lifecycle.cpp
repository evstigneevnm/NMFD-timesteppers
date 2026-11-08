#include "implicit_rosenbrock_dae_problem.h"
#include <nmfd/time_steppers/integration/time_step_adaptation_scipy.h>

using namespace rosenbrock_tests;

// x'=-x-z+x^2, 0=z-x^2. Failed full-system solves/residuals and constraint
// rejection must leave both differential/algebraic committed components intact.
int main()
try
{
    backend_type::init_device();
    using P                     = dae_problem<>;
    using L                     = direct_solver<P>;
    using S                     = rosenbrock::time_step<operations_type, P, adaptation_type, L>;
    auto             ops        = std::make_shared<operations_type>( 2 );
    auto             p          = std::make_shared<P>( ops );
    auto             solver     = std::make_shared<L>( ops );
    auto             controller = std::make_shared<adaptation_type>( adaptation_type::params{ .02 } );
    vector_wrap_type in( *ops ), out( *ops );
    in.start_use();
    out.start_use();
    ops->assign_scalar( 1., *in );
    for ( unsigned int stage = 1; stage <= 4; ++stage )
        for ( unsigned int kind = 0; kind < 4; ++kind )
        {
            p->residuals = p->linearizations = solver->calls = solver->bindings = 0;
            p->invalid_on = p->throw_on = solver->fail_on = solver->invalid_on = 0;
            if ( kind == 0 )
                solver->fail_on = stage;
            if ( kind == 1 )
                solver->invalid_on = stage;
            if ( kind == 2 )
                p->invalid_on = stage;
            if ( kind == 3 )
                p->throw_on = stage;
            S step( ops, p, controller, solver, dae_params<S>() );
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
            check( threw == ( kind == 3 ), "DAE exception propagation changed" );
            check( threw || step.get_status() != single_step_status::converged, "Failed DAE stage converged" );
            for ( unsigned int i = 0; i < 2; ++i )
                check(
                    ops->get_value_at_point( i, *out ) == 77. && ops->get_value_at_point( i, *in ) == 1.,
                    "Failed DAE attempt published partial differential/algebraic data"
                );
            p->invalid_on = p->throw_on = solver->fail_on = solver->invalid_on = 0;
            step.apply( *in, *out );
            check(
                step.get_status() == single_step_status::converged && p->linearizations == 2 && solver->bindings == 2,
                "DAE recovery did not reprepare/rebind its full operator"
            );
            step.finalize( adaptation_status::rejected, 0., *in );
            step.apply( *in, *out );
            check(
                step.get_status() == single_step_status::converged, "Externally rejected DAE candidate poisoned retry"
            );
            step.finalize( adaptation_status::accepted_modified, .01, *in );
            p->mass   = 3;
            p->factor = .5;
            step.set_time( .01 );
            step.apply( *in, *out );
            check(
                p->jacobi->mass == 3 && p->jacobi->factor == .5, "Modified DAE endpoint retained stale operator data"
            );
            step.finalize( adaptation_status::accepted, .03, *out );
            step.reset();
        }
    for ( unsigned int invalid = 0; invalid < 5; ++invalid )
    {
        auto sp = dae_params<S>();
        if ( invalid == 0 )
            sp.method = "ROS2";
        if ( invalid == 1 )
            sp.method = "ROS3";
        if ( invalid == 2 )
            sp.autonomous = false;
        if ( invalid == 3 )
            sp.mass_form = false;
        bool refused = false;
        try
        {
            if ( invalid == 4 )
            {
                using D = rosenbrock::time_step<operations_type, P, adaptation_type, L, true>;
                D step( ops, p, controller, solver, dae_params<D>() );
            }
            else
            {
                S step( ops, p, controller, solver, sp );
            }
        }
        catch ( const std::invalid_argument & )
        {
            refused = true;
        }
        check( refused, "Unsupported index-1 capability was accepted" );
    }
    using Base = integration::time_step_adaptation_scipy<operations_type>;
    using C    = dae_control<Base, P>;
    using A    = rosenbrock::time_step<operations_type, P, C, L>;
    Base::params cp;
    cp.relative_tolerance = cp.absolute_tolerance = .1;
    cp.initial_step = cp.maximum_step = .2;
    auto checked                      = std::make_shared<C>( p, cp );
    A    step( ops, p, checked, solver, dae_params<A>() );
    ops->assign_scalar( 2., *in );
    ops->assign_scalar( 77., *out );
    const auto before  = solver->calls;
    bool       refused = false;
    try
    {
        step.apply( *in, *out );
    }
    catch ( const std::invalid_argument & )
    {
        refused = true;
    }
    check(
        refused && before == solver->calls && ops->get_value_at_point( 1, *out ) == 77.,
        "Problem-owned initialization failed to reject inconsistent data before solving"
    );
    ops->assign_scalar( 1., *in );
    checked->constraint_tolerance = 1e-9;
    step.apply( *in, *out );
    check(
        step.get_status() == single_step_status::converged && step.get_attempts() > 1 &&
            checked->constraint_rejections > 0 && step.get_dt() < .2,
        "Independent constraint rejection failed to shrink and recover"
    );
    step.finalize( adaptation_status::rejected, 0., *in );
    auto sp                       = dae_params<A>();
    sp.maximum_attempts           = 1;
    checked->constraint_tolerance = 1e-30;
    A exhausted( ops, p, checked, solver, sp );
    ops->assign_scalar( 77., *out );
    exhausted.apply( *in, *out );
    check(
        exhausted.get_status() == single_step_status::attempt_limit_reached &&
            ops->get_value_at_point( 1, *out ) == 77.,
        "Constraint retry budget published a failed candidate"
    );
    std::cout << "PASS: Rosenbrock index-1 lifecycle (" << checks << " checks)\n";
}
catch ( const std::exception &e )
{
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
