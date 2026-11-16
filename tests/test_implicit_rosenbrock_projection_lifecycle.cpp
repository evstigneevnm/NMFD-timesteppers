#include "implicit_rosenbrock_projection_problem.h"
using namespace projection_tests;
using O = space_type;
using P = rosenbrock_projection_problem<O>;
using L = projection_linear_solver<O>;
using C = rosenbrock_recording_control<O>;
using S = rosenbrock_projected_step<O, C, true>;

#if defined( TEST_PROJECTION_BAD_RETURN_MUST_FAIL ) || defined( TEST_PROJECTION_MISSING_APPLY_MUST_FAIL )
struct invalid_projector
{
#    ifdef TEST_PROJECTION_BAD_RETURN_MUST_FAIL
    int apply( O::vector_type & )
    {
        return 1;
    }
#    endif
};
void reject_invalid( std::shared_ptr<O> ops, std::shared_ptr<P> p, std::shared_ptr<C> c, std::shared_ptr<L> l )
{
    rosenbrock::time_step<O, P, C, L, true, invalid_projector> step(
        ops, p, c, l, {}, std::make_shared<invalid_projector>()
    );
}
#endif

template <class Action>
void refuses( Action action, const char *message )
{
    bool caught = false;
    try
    {
        action();
    }
    catch ( const std::exception & )
    {
        caught = true;
    }
    check( caught, message );
}

// Fault every ordinary and dense-only solve/projection, including aliases.
// No candidate/view is published; a clean retry starts from the same input.
void failure( const char *method, unsigned int location, int mode, bool alias, bool retry = false )
{
    auto ops             = std::make_shared<O>( 3 );
    auto p               = std::make_shared<P>( ops );
    auto c               = std::make_shared<C>( ops );
    c->retry             = retry;
    auto      l          = std::make_shared<L>( ops );
    auto      projection = std::make_shared<projector<O>>( ops );
    S::params sp{ method, 3 };
    sp.autonomous = false;
    S         step( ops, p, c, l, sp, projection );
    buffer<O> state( *ops ), initial( *ops ), out( *ops ), sentinel( *ops ), scratch( *ops );
    exact( *ops, *state, .3 );
    ops->assign( *state, *initial );
    ops->assign_scalar( 7., *out );
    ops->assign( *out, *sentinel );
    step.set_time( .3 );
    step.set_target_time( .5 );
    if ( mode == 0 )
        projection->fail_on = location;
    if ( mode == 1 )
        projection->invalid_on = location;
    if ( mode == 2 )
        projection->throw_on = location;
    if ( mode == 3 )
        l->fail_on = location;
    if ( mode == 4 )
        l->invalid_on = location;
    bool threw = false;
    try
    {
        step.apply( *state, alias ? *state : *out );
    }
    catch ( const std::runtime_error & )
    {
        threw = true;
    }
    check( threw == ( mode == 2 ), "Rosenbrock swallowed projection exception" );
    if ( !retry )
    {
        if ( !threw )
            check(
                step.get_status() == ( mode == 0   ? single_step_status::failed_projection
                                       : mode == 3 ? single_step_status::failed_solver
                                                   : single_step_status::failed_nonfinite ),
                "Rosenbrock projection failure status"
            );
        check( difference( *ops, *state, *initial, *scratch ) == 0, "Failed Rosenbrock changed input" );
        check( difference( *ops, *out, *sentinel, *scratch ) == 0, "Failed Rosenbrock published output" );
        const auto stages = step.table().stage_count();
        check( c->assessments == ( location > stages ? 1u : 0u ), "Invalid Rosenbrock stage was assessed" );
        if ( mode >= 3 )
            check( projection->calls == location - 1, "Failed linear solve was projected" );
        refuses( [&] { step.get_continuous_integration(); }, "Failed Rosenbrock exposed dense output" );
        projection->fail_on = projection->invalid_on = projection->throw_on = l->fail_on = l->invalid_on = 0;
        step.apply( *state, *out );
    }
    else
        check( step.get_attempts() == 2 && step.get_dt() == .01, "Projection failure did not shrink/retry" );
    check(
        step.get_status() == single_step_status::converged && p->time == .3, "Rosenbrock did not recover frozen base"
    );
    auto         view = step.get_continuous_integration();
    const double next = view.evaluate( .5, *scratch );
    check( constraint( *ops, *scratch, *out ) < 1e-12, "Shortened Rosenbrock endpoint left range" );
    step.finalize( adaptation_status::accepted_modified, next, *scratch );
    refuses( [&] { view.evaluate( .5, *out ); }, "Modified Rosenbrock commit retained view" );
    step.set_time( next );
    step.apply( *scratch, *out );
    check( step.get_status() == single_step_status::converged, "Rosenbrock failed after shortened commit" );
    auto second = step.get_continuous_integration();
    step.reset();
    refuses( [&] { second.evaluate( .5, *out ); }, "Rosenbrock reset retained dense view" );
}

void ownership_and_scope()
{
    using Default = rosenbrock::time_step<O, P, fixed_control<O>, L>;
    using Custom  = rosenbrock_projected_step<O>;
    static_assert( !Default::projects_increments && Custom::projects_increments );
    auto       ops    = std::make_shared<O>( 3 );
    auto       p      = std::make_shared<P>( ops );
    auto       l      = std::make_shared<L>( ops );
    auto       c      = std::make_shared<fixed_control<O>>();
    const auto before = ops->allocations;
    refuses( [&] { Custom missing( ops, p, c, l ); }, "Rosenbrock accepted a null custom projector" );
    check( ops->allocations == before, "Null projector allocated Rosenbrock workspace" );
    auto projection = std::make_shared<projector<O>>( ops );
    for ( int mode = 0; mode < 4; ++mode )
    {
        typename Custom::params sp;
        sp.mass_form  = mode == 0;
        sp.index1_dae = mode == 1;
        sp.index2_dae = mode == 2;
        if ( mode == 3 )
            sp.method = "ROSI2P2";
        refuses(
            [&] { Custom invalid( ops, p, c, l, sp, projection ); },
            "Rosenbrock accepted unsupported projected mass/DAE mode"
        );
        check( ops->allocations == before, "Unsupported projection allocated Rosenbrock workspace" );
    }
    buffer<O> state( *ops ), out( *ops );
    exact( *ops, *state );
    std::size_t storage = 0, scans = 0;
    {
        const auto count = ops->allocations;
        Default    step( ops, p, c, l );
        storage = ops->allocations - count;
        step.apply( *state, *out );
        step.finalize( adaptation_status::accepted, step.get_dt(), *out );
        step.reset();
        const auto valid = ops->validity_checks;
        step.apply( *state, *out );
        scans = ops->validity_checks - valid;
    }
    {
        const auto count = ops->allocations;
        Default    step( ops, p, c, l, {}, std::make_shared<nmfd::operations::ident_operator<O>>( ops ) );
        check( ops->allocations - count == storage, "Identity projection added Rosenbrock vectors" );
        const auto valid = ops->validity_checks;
        step.apply( *state, *out );
        check( ops->validity_checks - valid == scans, "Identity projection added Rosenbrock scans" );
    }
    std::weak_ptr<projector<O>> weak = projection;
    {
        const auto count = ops->allocations;
        Custom     step( ops, p, c, l, {}, projection );
        projection.reset();
        check( !weak.expired(), "Rosenbrock lost shared projector ownership" );
        check( ops->allocations - count == storage, "Custom projection added Rosenbrock vectors" );
    }
    check( weak.expired(), "Rosenbrock projector ownership cycle" );
    // Repeated numerical failures exhaust the unchanged attempt budget.
    auto always         = std::make_shared<projector<O>>( ops );
    always->always_fail = true;
    auto retry          = std::make_shared<C>( ops );
    retry->retry        = true;
    S failed( ops, p, retry, l, S::params{ "ROS2", 3 }, always );
    failed.apply( *state, *out );
    check(
        failed.get_status() == single_step_status::attempt_limit_reached && failed.get_attempts() == 3,
        "Rosenbrock projection retry budget"
    );
}
int main()
try
{
    backend_type::init_device();
    ownership_and_scope();
    for ( const char *method : { "ROS2", "ROS3", "ROS34PW2" } )
    {
        const unsigned int solves = std::string( method ) == "ROS2" ? 2 : std::string( method ) == "ROS3" ? 4 : 6;
        for ( unsigned int location = 1; location <= solves; ++location )
        {
            for ( int mode = 0; mode < 5; ++mode )
                for ( bool alias : { false, true } )
                    failure( method, location, mode, alias );
            failure( method, location, 0, false, true );
        }
    }
    std::cout << checks << " Rosenbrock projection lifecycle checks passed\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
