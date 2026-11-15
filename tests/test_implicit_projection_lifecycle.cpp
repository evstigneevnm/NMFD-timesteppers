#include "implicit_projection_problem.h"

using namespace projection_tests;

#ifdef TEST_PROJECTION_VOID_MUST_FAIL
struct void_projector
{
    void apply( space_type::vector_type & )
    {
    }
};
void reject_void_projector(
    std::shared_ptr<space_type> ops, std::shared_ptr<implicit_problem<space_type>> p,
    std::shared_ptr<fixed_control<space_type>> control
)
{
    using S = runge_kutta::implicit_time_step<
        space_type, implicit_problem<space_type>, fixed_control<space_type>, implicit_newton<space_type>::solver, false,
        void_projector>;
    S step( ops, p, control, implicit_newton<space_type>::make( ops ), {}, std::make_shared<void_projector>() );
}
#endif

struct observed_control : fixed_control<space_type>
{
    using fixed_control<space_type>::fixed_control;
    std::size_t       assessments = 0;
    adaptation_status assess(
        double, double, const vector_type &, const vector_type &, double &next, unsigned int = 0,
        const vector_type * = nullptr
    )
    {
        ++assessments;
        next = get_dt();
        return adaptation_status::accepted;
    }
};
struct retry_control : observed_control
{
    double      h          = .1;
    std::size_t rejections = 0;
    double      initialize( double, const vector_type &, double = -1 )
    {
        reset();
        return h;
    }
    void reset()
    {
        h           = .1;
        assessments = rejections = 0;
    }
    double get_dt() const
    {
        return h;
    }
    adaptation_status reject_step( double )
    {
        ++rejections;
        h *= .5;
        return adaptation_status::rejected;
    }
};

void ownership()
{
    using O = space_type;
    using S = implicit_projected_step<O>;
    using R = implicit_reference_step<O>;
    static_assert( S::projects_stages && !R::projects_stages );
    auto      ops     = std::make_shared<O>( 3 );
    auto      p       = std::make_shared<implicit_problem<O>>( ops );
    auto      control = std::make_shared<fixed_control<O>>( fixed_control<O>::params{ .1 } );
    auto      solver  = implicit_newton<O>::make( ops );
    buffer<O> state( *ops ), out( *ops );
    exact( *ops, *state );
    const auto before  = ops->allocations;
    bool       refused = false;
    try
    {
        S step( ops, p, control, solver );
    }
    catch ( const std::invalid_argument & )
    {
        refused = true;
    }
    check( refused && before == ops->allocations, "Missing DIRK projector allocated workspace" );
    std::size_t storage, scans;
    {
        R step( ops, p, control, solver, R::params{ "ESDIRK2(1)3" } );
        storage = ops->allocations - before;
        step.apply( *state, *out ); // Warm up the injected Newton solver.
        step.finalize( adaptation_status::accepted, .1, *out );
        step.reset();
        auto count = ops->validity_checks;
        step.apply( *state, *out );
        scans = ops->validity_checks - count;
        step.finalize( adaptation_status::accepted, .1, *out );
    }
    {
        const auto count = ops->allocations;
        auto       ident = std::make_shared<nmfd::operations::ident_operator<O>>( ops );
        R          step( ops, p, control, solver, runge_kutta::make_butcher_table( "ESDIRK2(1)3" ), {}, ident );
        check( ops->allocations - count == storage, "Default identity changed DIRK storage" );
        auto checks_before = ops->validity_checks;
        step.apply( *state, *out );
        check( ops->validity_checks - checks_before == scans, "Default identity added DIRK scans" );
        step.finalize( adaptation_status::accepted, .1, *out );
    }
    std::weak_ptr<projector<O>> weak;
    {
        auto project     = std::make_shared<projector<O>>( ops );
        weak             = project;
        const auto count = ops->allocations;
        S          step( ops, p, control, solver, S::params{ "ESDIRK2(1)3" }, project );
        check( ops->allocations - count == storage, "Projector added DIRK stage vectors" );
        project.reset();
        check( !weak.expired(), "DIRK lost projector ownership" );
    }
    check( weak.expired(), "DIRK projector ownership cycle" );
}

void failure( const char *method, std::size_t stage, int mode, bool alias )
{
    using O           = space_type;
    using S           = implicit_projected_step<O, observed_control, true>;
    auto      ops     = std::make_shared<O>( 3 );
    auto      p       = std::make_shared<implicit_problem<O>>( ops );
    auto      project = std::make_shared<projector<O>>( ops );
    auto      control = std::make_shared<observed_control>( observed_control::params{ .1 } );
    S         step( ops, p, control, implicit_newton<O>::make( ops ), S::params{ method }, project );
    buffer<O> state( *ops ), initial( *ops ), out( *ops ), sentinel( *ops ), scratch( *ops );
    exact( *ops, *state );
    exact( *ops, *initial );
    ops->assign_scalar( 7., *out );
    ops->assign( *out, *sentinel );
    if ( mode == 0 )
        project->fail_on = stage;
    if ( mode == 1 )
        project->invalid_on = stage;
    if ( mode == 2 )
        project->throw_on = stage;
    bool threw = false;
    try
    {
        step.apply( *state, alias ? *state : *out );
    }
    catch ( const std::runtime_error & )
    {
        threw = true;
    }
    check( threw == ( mode == 2 ), "DIRK projection exception contract" );
    if ( !threw )
        check(
            step.get_status() ==
                ( mode == 0 ? single_step_status::failed_projection : single_step_status::failed_nonfinite ),
            "DIRK projection failure status"
        );
    check( difference( *ops, *state, *initial, *scratch ) == 0, "Failed projection changed input" );
    check( difference( *ops, *out, *sentinel, *scratch ) == 0, "Failed projection published output" );
    check( project->calls == stage, "Unexpected projection failure call count" );
    check( control->assessments == ( stage > step.table().size() ? 1 : 0 ), "Invalid DIRK candidate assessed" );
    bool refused = false;
    try
    {
        step.get_continuous_integration();
    }
    catch ( const std::logic_error & )
    {
        refused = true;
    }
    check( refused, "Failed DIRK projection exposed dense view" );
    project->fail_on = project->invalid_on = project->throw_on = 0;
    step.reset();
    step.apply( *state, *out );
    check( step.get_status() == single_step_status::converged, "DIRK projection restart failed" );
    auto view = step.get_continuous_integration();
    step.finalize( adaptation_status::accepted, .05, *out );
    refused = false;
    try
    {
        view.evaluate( .3, *scratch );
    }
    catch ( const std::logic_error & )
    {
        refused = true;
    }
    check( refused, "Finalization retained stale projected dense output" );
}
void retries()
{
    using O           = space_type;
    using S           = implicit_projected_step<O, retry_control>;
    auto      ops     = std::make_shared<O>( 3 );
    auto      p       = std::make_shared<implicit_problem<O>>( ops );
    auto      project = std::make_shared<projector<O>>( ops );
    auto      control = std::make_shared<retry_control>();
    buffer<O> state( *ops ), out( *ops ), initial( *ops ), scratch( *ops );
    exact( *ops, *state );
    exact( *ops, *initial );
    S step( ops, p, control, implicit_newton<O>::make( ops ), S::params{ "ESDIRK2(1)3", 3 }, project );
    project->fail_on = 2;
    step.apply( *state, *out );
    check(
        step.get_status() == single_step_status::converged && step.get_attempts() == 2, "DIRK projection did not retry"
    );
    check( step.get_dt() == .05 && control->rejections == 1 && control->assessments == 1, "DIRK retry accounting" );
    check( difference( *ops, *state, *initial, *scratch ) == 0, "DIRK retry changed committed input" );
    check( constraint( *ops, *out, *scratch ) < 1e-13, "DIRK retry retained corrupt stage" );
    step.finalize( adaptation_status::accepted, .05, *out );
    step.reset();
    project->always_fail = true;
    step.apply( *state, *out );
    check(
        step.get_status() == single_step_status::attempt_limit_reached && step.get_attempts() == 3,
        "Unbounded DIRK projection retries"
    );
}

void invalid_physical_rate( const char *method, std::size_t projected_stages )
{
    using O           = space_type;
    using S           = implicit_projected_step<O, fixed_control<O>, true>;
    auto      ops     = std::make_shared<O>( 3 );
    auto      p       = std::make_shared<implicit_problem<O>>( ops );
    auto      project = std::make_shared<projector<O>>( ops );
    auto      control = std::make_shared<fixed_control<O>>( fixed_control<O>::params{ .1 } );
    S         step( ops, p, control, implicit_newton<O>::make( ops ), S::params{ method }, project );
    buffer<O> state( *ops ), initial( *ops ), out( *ops ), scratch( *ops );
    exact( *ops, *state );
    exact( *ops, *initial );
    ops->assign_scalar( 7., *out );
    p->invalid_on = 1;
    step.apply( *state, *out );
    check( step.get_status() == single_step_status::failed_nonfinite, "Invalid physical rate not rejected" );
    check( project->calls == projected_stages, "Projector inspected an invalid physical rate" );
    check( difference( *ops, *state, *initial, *scratch ) == 0, "Invalid physical rate changed input" );
    ops->assign_scalar( 7., *scratch );
    ops->add_lin_comb( -1., *scratch, *out );
    check( ops->norm( *out ) == 0, "Invalid physical rate published an endpoint" );
}
int main()
try
{
    backend_type::init_device();
    ownership();
    retries();
    invalid_physical_rate( "ESDIRK2(1)3", 0 );
    invalid_physical_rate( "SDIRK3(1)3", 3 );
    for ( bool alias : { false, true } )
        for ( int mode = 0; mode < 3; ++mode )
        {
            failure( "ESDIRK2(1)3", 1, mode, alias ); // Explicit first rate.
            failure( "SDIRK2(1)2", 2, mode, alias );  // Final stiffly-accurate stage.
            failure( "SDIRK3(1)3", 4, mode, alias );  // Dense-only endpoint rate.
        }
    std::cout << checks << " implicit projection lifecycle checks passed\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
