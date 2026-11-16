#include "implicit_imex_projection_problem.h"

using namespace projection_tests;
using O = space_type;
using P = imex_projection_problem<O>;
using I = P::implicit_type;

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
void reject_invalid_projector( std::shared_ptr<O> ops, std::shared_ptr<P> p, std::shared_ptr<fixed_control<O>> control )
{
    using S =
        runge_kutta::imex_time_step<O, P, fixed_control<O>, implicit_newton<O, I>::solver, false, invalid_projector>;
    S invalid( ops, p, control, implicit_newton<O, I>::make( ops ), {}, std::make_shared<invalid_projector>() );
}
#endif

struct observed_control : fixed_control<O>
{
    using fixed_control<O>::fixed_control;
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
    using S = imex_projected_step<O>;
    using R = imex_reference_step<O>;
    static_assert( S::projects_stages && !R::projects_stages );
    auto      ops     = std::make_shared<O>( 3 );
    auto      p       = std::make_shared<P>( ops );
    auto      solver  = implicit_newton<O, I>::make( ops );
    auto      control = std::make_shared<fixed_control<O>>( fixed_control<O>::params{ .1 } );
    buffer<O> state( *ops ), out( *ops );
    imex_exact( *ops, *state );
    const auto before  = ops->allocations;
    bool       refused = false;
    try
    {
        S invalid( ops, p, control, solver );
    }
    catch ( const std::invalid_argument & )
    {
        refused = true;
    }
    check( refused && ops->allocations == before, "Null IMEX projector not rejected before allocation" );
    std::size_t storage, scans;
    {
        R step( ops, p, control, solver, R::params{ "IMEX_ARK324L2SA" } );
        storage = ops->allocations - before;
        step.apply( *state, *out );
        step.finalize( adaptation_status::accepted, .1, *out );
        step.reset();
        const auto count = ops->validity_checks;
        step.apply( *state, *out );
        scans = ops->validity_checks - count;
        step.finalize( adaptation_status::accepted, .1, *out );
    }
    {
        const auto count = ops->allocations;
        auto       ident = std::make_shared<nmfd::operations::ident_operator<O>>( ops );
        R step( ops, p, control, solver, runge_kutta::make_imex_butcher_table( "IMEX_ARK324L2SA" ), {}, ident );
        check( ops->allocations - count == storage, "Default identity changed IMEX workspace" );
        const auto validity = ops->validity_checks;
        step.apply( *state, *out );
        check( ops->validity_checks - validity == scans, "Default identity added IMEX validity scans" );
        step.finalize( adaptation_status::accepted, .1, *out );
    }
    std::weak_ptr<projector<O>> weak;
    {
        auto project     = std::make_shared<projector<O>>( ops );
        weak             = project;
        const auto count = ops->allocations;
        S step( ops, p, control, solver, runge_kutta::make_imex_butcher_table( "IMEX_ARK324L2SA" ), {}, project );
        check( ops->allocations - count == storage, "IMEX projector added stage vectors" );
        project.reset();
        check( !weak.expired(), "IMEX lost shared projector ownership" );
        step.apply( *state, *out );
        check( step.get_status() == single_step_status::converged, "Custom-table projected IMEX failed" );
        step.finalize( adaptation_status::accepted, .1, *out );
    }
    check( weak.expired(), "IMEX projector ownership cycle" );
}

// Fail an initial implicit/explicit rate, a solved stage state, or either fresh
// dense endpoint rate. Publication and committed input must be transactional.
void failure( unsigned int location, unsigned int mode, bool alias )
{
    using S           = imex_projected_step<O, observed_control, true>;
    auto      ops     = std::make_shared<O>( 3 );
    auto      p       = std::make_shared<P>( ops );
    auto      project = std::make_shared<projector<O>>( ops );
    auto      control = std::make_shared<observed_control>( observed_control::params{ .1 } );
    S         step( ops, p, control, implicit_newton<O, I>::make( ops ), S::params{ "IMEX_ARK324L2SA" }, project );
    buffer<O> state( *ops ), initial( *ops ), out( *ops ), sentinel( *ops ), scratch( *ops );
    imex_exact( *ops, *state );
    imex_exact( *ops, *initial );
    step.apply( *state, *out );
    check( step.get_status() == single_step_status::converged, "IMEX lifecycle baseline failed" );
    const auto   total    = project->calls;
    const double frozen_a = p->bundle->implicit_part->jacobian->a, frozen_b = p->bundle->implicit_part->jacobian->b;
    step.finalize( adaptation_status::accepted, .1, *out );
    step.reset();
    project->calls          = 0;
    control->assessments    = 0;
    const std::size_t stage = location < 3 ? location + 1 : total - ( location == 3 ? 1 : 0 );
    if ( mode == 0 )
        project->fail_on = stage;
    if ( mode == 1 )
        project->invalid_on = stage;
    if ( mode == 2 )
        project->throw_on = stage;
    ops->assign_scalar( 7., *out );
    ops->assign( *out, *sentinel );
    bool threw = false;
    try
    {
        step.apply( *state, alias ? *state : *out );
    }
    catch ( const std::runtime_error & )
    {
        threw = true;
    }
    check( threw == ( mode == 2 ), "IMEX projection exception handling" );
    if ( !threw )
        check(
            step.get_status() ==
                ( mode == 0 ? single_step_status::failed_projection : single_step_status::failed_nonfinite ),
            "IMEX projection failure status"
        );
    check( project->calls == stage, "Wrong IMEX failure projection count" );
    check( control->assessments == ( location >= 3 ? 1 : 0 ), "Invalid IMEX candidate reached assessment" );
    check( difference( *ops, *state, *initial, *scratch ) == 0, "Failed projection changed committed IMEX input" );
    check( difference( *ops, *out, *sentinel, *scratch ) == 0, "Failed projection published IMEX output" );
    if ( location >= 3 )
    {
        check(
            p->bundle->implicit_part->time == .1 && p->bundle->explicit_part->time == .1,
            "Dense IMEX failure left partition times changed"
        );
        check(
            p->bundle->implicit_part->jacobian->a == frozen_a && p->bundle->implicit_part->jacobian->b == frozen_b,
            "Dense IMEX failure changed frozen Jacobian"
        );
    }
    bool refused = false;
    try
    {
        step.get_continuous_integration();
    }
    catch ( const std::logic_error & )
    {
        refused = true;
    }
    check( refused, "Failed IMEX projection published dense view" );
    project->fail_on = project->invalid_on = project->throw_on = 0;
    step.reset();
    step.apply( *state, *out );
    check( step.get_status() == single_step_status::converged, "IMEX projection restart failed" );
    auto view = step.get_continuous_integration();
    view.evaluate( .5, *scratch );
    step.finalize( adaptation_status::accepted, .05, *scratch );
    refused = false;
    try
    {
        view.evaluate( .3, *scratch );
    }
    catch ( const std::logic_error & )
    {
        refused = true;
    }
    check( refused, "Shortened commitment retained stale projected IMEX view" );
}

void invalid_rate( bool implicit, bool nonfinite )
{
    using S      = imex_projected_step<O>;
    auto ops     = std::make_shared<O>( 3 );
    auto p       = std::make_shared<P>( ops );
    auto project = std::make_shared<projector<O>>( ops );
    auto control = std::make_shared<fixed_control<O>>( fixed_control<O>::params{ .1 } );
    S    step( ops, p, control, implicit_newton<O, I>::make( ops ), S::params{ "IMEX_ARK324L2SA" }, project );
    if ( implicit )
    {
        if ( nonfinite )
            p->bundle->implicit_part->invalid_on = 1;
        else
            p->bundle->implicit_part->fail_on = 1;
    }
    else
    {
        if ( nonfinite )
            p->bundle->explicit_part->invalid_on = 1;
        else
            p->bundle->explicit_part->fail_on = 1;
    }
    buffer<O> state( *ops ), initial( *ops ), out( *ops ), scratch( *ops );
    imex_exact( *ops, *state );
    imex_exact( *ops, *initial );
    ops->assign_scalar( 7., *out );
    step.apply( *state, *out );
    check(
        step.get_status() == ( nonfinite ? single_step_status::failed_nonfinite : single_step_status::failed_solver ),
        "Invalid IMEX rate status"
    );
    check( project->calls == ( implicit ? 0 : 1 ), "Projector inspected an invalid IMEX physical rate" );
    check( difference( *ops, *state, *initial, *scratch ) == 0, "Invalid IMEX rate changed input" );
    ops->assign_scalar( 7., *scratch );
    ops->add_lin_comb( -1., *scratch, *out );
    check( ops->norm( *out ) == 0, "Invalid IMEX rate published output" );
}
void retries()
{
    using S           = imex_projected_step<O, retry_control>;
    auto      ops     = std::make_shared<O>( 3 );
    auto      p       = std::make_shared<P>( ops );
    auto      project = std::make_shared<projector<O>>( ops );
    auto      control = std::make_shared<retry_control>();
    S         step( ops, p, control, implicit_newton<O, I>::make( ops ), S::params{ "IMEX_ARK324L2SA", 3 }, project );
    buffer<O> state( *ops ), initial( *ops ), out( *ops ), scratch( *ops );
    imex_exact( *ops, *state );
    imex_exact( *ops, *initial );
    project->fail_on = 3;
    step.apply( *state, *out );
    check(
        step.get_status() == single_step_status::converged && step.get_attempts() == 2, "IMEX projection did not retry"
    );
    check( step.get_dt() == .05 && control->rejections == 1 && control->assessments == 1, "IMEX retry accounting" );
    check( difference( *ops, *state, *initial, *scratch ) == 0, "IMEX retry changed input" );
    check( constraint( *ops, *out, *scratch ) < 1e-13, "IMEX retry retained a failed stage" );
    step.finalize( adaptation_status::accepted, .05, *out );
    step.reset();
    project->always_fail = true;
    step.apply( *state, *out );
    check(
        step.get_status() == single_step_status::attempt_limit_reached && step.get_attempts() == 3,
        "Unbounded IMEX projection retries"
    );
}
int main()
try
{
    backend_type::init_device();
    ownership();
    retries();
    for ( bool implicit : { false, true } )
        for ( bool nonfinite : { false, true } )
            invalid_rate( implicit, nonfinite );
    for ( bool alias : { false, true } )
        for ( unsigned int mode = 0; mode < 3; ++mode )
            for ( unsigned int location = 0; location < 5; ++location )
                failure( location, mode, alias );
    std::cout << checks << " IMEX projection lifecycle checks passed\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
