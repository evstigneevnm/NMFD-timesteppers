#include "implicit_bdf_projection_test.h"
using namespace projection_tests;
using O = space_type;

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
void reject_invalid_projector( std::shared_ptr<O> ops, std::shared_ptr<implicit_problem<O>> p )
{
    using S = bdf::implicit_time_step<
        O, implicit_problem<O>, fixed_control<O>, implicit_newton<O>::solver, false, invalid_projector>;
    S invalid(
        ops, p, std::make_shared<fixed_control<O>>(), implicit_newton<O>::make( ops ), {},
        std::make_shared<invalid_projector>()
    );
}
#endif

struct control : bdf_accepting_control<O>
{
    using bdf_accepting_control<O>::bdf_accepting_control;
    double            h           = .1;
    bool              retry       = false;
    std::size_t       assessments = 0, rejections = 0;
    adaptation_status last = adaptation_status::failed;
    void              reset()
    {
        h           = .1;
        assessments = rejections = 0;
    }
    double initialize( double, const vector_type &, double = -1 )
    {
        return h;
    }
    double get_dt() const
    {
        return h;
    }
    adaptation_status assess(
        double, double, const vector_type &, const vector_type &, double &next, unsigned int exponent = 0,
        const vector_type *error = nullptr
    )
    {
        check( error && exponent >= 2, "Missing BDF projected estimate" );
        ++assessments;
        next = h;
        return adaptation_status::accepted;
    }
    adaptation_status reject_step( double dt )
    {
        ++rejections;
        if ( !retry )
            return adaptation_status::failed;
        h = std::abs( dt ) * .5;
        return adaptation_status::rejected;
    }
    void update( adaptation_status s, double, double, const vector_type & )
    {
        last = s;
    }
};

void ownership()
{
    using S = bdf_projected_step<O>;
    using R = bdf_reference_step<O>;
    static_assert( S::projects_states && !R::projects_states );
    auto      ops    = std::make_shared<O>( 3 );
    auto      p      = std::make_shared<implicit_problem<O>>( ops );
    auto      solver = implicit_newton<O>::make( ops );
    auto      c      = std::make_shared<fixed_control<O>>( fixed_control<O>::params{ .1 } );
    buffer<O> state( *ops ), out( *ops );
    exact( *ops, *state );
    const auto before  = ops->allocations;
    bool       refused = false;
    try
    {
        S step( ops, p, c, solver );
    }
    catch ( const std::invalid_argument & )
    {
        refused = true;
    }
    check( refused && ops->allocations == before, "BDF missing projector allocated workspace" );
    std::size_t storage, scans;
    {
        R step( ops, p, c, solver );
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
        const auto allocated = ops->allocations;
        R          step( ops, p, c, solver, {}, std::make_shared<nmfd::operations::ident_operator<O>>( ops ) );
        check( ops->allocations - allocated == storage, "BDF identity added backend storage" );
        const auto count = ops->validity_checks;
        step.apply( *state, *out );
        check( ops->validity_checks - count == scans, "BDF identity added validity scans" );
        step.finalize( adaptation_status::accepted, .1, *out );
    }
    std::weak_ptr<projector<O>> weak;
    {
        auto projection      = std::make_shared<projector<O>>( ops );
        weak                 = projection;
        const auto allocated = ops->allocations;
        S          step( ops, p, c, solver, {}, projection );
        projection.reset();
        check( !weak.expired(), "BDF lost shared projector ownership" );
        check( ops->allocations - allocated == storage, "BDF projection added backend vectors" );
    }
    check( weak.expired(), "BDF projector ownership cycle" );
}

void startup_failure( unsigned int location, unsigned int mode, bool alias, bool retry )
{
    using S         = bdf_projected_step<O, control, true>;
    auto ops        = std::make_shared<O>( 3 );
    auto p          = std::make_shared<implicit_problem<O>>( ops );
    p->record_times = true;
    auto projection = std::make_shared<projector<O>>( ops );
    auto c          = std::make_shared<control>();
    c->retry        = retry;
    S::params prm{ "BDF2", 4 };
    prm.adaptive = true;
    S         step( ops, p, c, implicit_newton<O>::make( ops ), prm, projection );
    buffer<O> state( *ops ), initial( *ops ), out( *ops ), sentinel( *ops ), scratch( *ops );
    exact( *ops, *state );
    exact( *ops, *initial );
    ops->assign_scalar( 7., *out );
    ops->assign( *out, *sentinel );
    if ( mode == 0 )
        projection->fail_on = location;
    if ( mode == 1 )
        projection->invalid_on = location;
    if ( mode == 2 )
        projection->throw_on = location;
    bool threw = false;
    try
    {
        step.apply( *state, alias ? *state : *out );
    }
    catch ( const std::runtime_error & )
    {
        threw = true;
    }
    check( threw == ( mode == 2 ), "BDF projection exception was swallowed" );
    if ( !retry || threw )
    {
        if ( !threw )
            check(
                step.get_status() ==
                    ( mode == 0 ? single_step_status::failed_projection : single_step_status::failed_nonfinite ),
                "BDF projection failure status"
            );
        check( c->assessments == 0, "BDF assessed invalid startup work" );
        check( difference( *ops, *out, *sentinel, *scratch ) == 0, "BDF failure published output" );
        check( difference( *ops, *state, *initial, *scratch ) == 0, "BDF failure changed committed input" );
        bool refused = false;
        try
        {
            step.get_continuous_integration();
        }
        catch ( const std::logic_error & )
        {
            refused = true;
        }
        check( refused, "BDF failure published a dense view" );
        projection->fail_on = projection->invalid_on = projection->throw_on = 0;
        step.apply( *state, *out );
        check(
            step.get_status() == single_step_status::converged, "BDF startup did not recover without resetting history"
        );
    }
    else
    {
        check(
            step.get_status() == single_step_status::converged && step.get_attempts() == 2,
            "BDF projection failure did not retry"
        );
        check( c->rejections == 1 && step.get_dt() == .05, "BDF retry accounting" );
        check( projection->calls == location + 3, "BDF retry reused an invalid half-step" );
    }
    check( step.get_order() == 1, "Startup failure changed committed BDF order" );
    check(
        constraint( *ops, alias && retry && !threw ? *state : *out, *scratch ) < 1e-12,
        "BDF retry retained corrupt projection"
    );
    auto view = step.get_continuous_integration();
    view.evaluate( .5, *scratch );
    step.finalize( adaptation_status::accepted, step.get_dt() * .5, *scratch );
    check( c->last == adaptation_status::accepted_modified, "Shortened BDF endpoint was not marked modified" );
    bool expired = false;
    try
    {
        view.evaluate( .4, *scratch );
    }
    catch ( const std::logic_error & )
    {
        expired = true;
    }
    check( expired, "Shortened BDF commit retained dense view" );
}

void history_transaction()
{
    using S              = bdf_projected_step<O, control, true>;
    using R              = bdf_reference_step<O, control, true>;
    auto      ops        = std::make_shared<O>( 3 );
    auto      p          = std::make_shared<implicit_problem<O>>( ops );
    auto      r          = std::make_shared<implicit_problem<O>>( ops, true );
    auto      projection = std::make_shared<projector<O>>( ops );
    auto      c          = std::make_shared<control>();
    S::params sp{ "BDF2" };
    sp.adaptive = true;
    R::params rp{ "BDF2" };
    rp.adaptive = true;
    S         step( ops, p, c, implicit_newton<O>::make( ops ), sp, projection );
    R         reference( ops, r, std::make_shared<control>(), implicit_newton<O>::make( ops ), rp );
    buffer<O> state( *ops ), ref( *ops ), out( *ops ), expected( *ops ), scratch( *ops ), initial( *ops );
    exact( *ops, *state );
    ops->assign( *state, *ref );
    double t = 0;
    for ( int j = 0; j < 3; ++j )
    {
        step.set_time( t );
        reference.set_time( t );
        step.apply( *state, *out );
        reference.apply( *ref, *expected );
        check( step.get_status() == single_step_status::converged, "Projected BDF warmup failed" );
        check( step.get_order() == ( j < 2 ? 1u : 2u ), "Ordinary projected commit restarted BDF history" );
        t += step.get_dt();
        step.finalize( adaptation_status::accepted, t, *out );
        reference.finalize( adaptation_status::accepted, t, *expected );
        ops->assign( *out, *state );
        ops->assign( *expected, *ref );
    }
    ops->assign( *state, *initial );
    step.set_time( t );
    reference.set_time( t );
    ops->assign_scalar( 7., *out );
    projection->fail_on = projection->calls + 1;
    step.apply( *state, *out );
    check( step.get_status() == single_step_status::failed_projection, "Warmed BDF projection did not fail" );
    check( difference( *ops, *state, *initial, *scratch ) == 0, "Failed warmed BDF changed input" );
    projection->fail_on = 0;
    reference.apply( *ref, *expected );
    for ( auto outcome : { adaptation_status::rejected, adaptation_status::failed, adaptation_status::accepted } )
    {
        step.apply( *state, *out );
        check(
            step.get_status() == single_step_status::converged && step.get_order() == 2,
            "BDF failed/rejected work changed order/history"
        );
        check( difference( *ops, *out, *expected, *scratch ) < 1e-11, "BDF failure polluted committed differences" );
        step.finalize(
            outcome, outcome == adaptation_status::accepted ? t + step.get_dt() : t,
            outcome == adaptation_status::accepted ? *out : *state
        );
    }
    t += step.get_dt();
    reference.finalize( adaptation_status::accepted, t, *expected );
    ops->assign( *out, *state );
    step.set_time( t );
    step.apply( *state, *out );
    auto view = step.get_continuous_integration();
    view.evaluate( .5, *scratch );
    t += step.get_dt() * .5;
    step.finalize( adaptation_status::accepted, t, *scratch );
    ops->assign( *scratch, *state );
    step.set_time( t );
    step.apply( *state, *out );
    check(
        step.get_status() == single_step_status::converged && step.get_order() == 1,
        "Shortened BDF did not restart at order one"
    );
    step.finalize( adaptation_status::accepted, t + step.get_dt(), *out );
    step.reset();
    projection->always_fail = true;
    c->retry                = true;
    step.apply( *state, *out );
    check(
        step.get_status() == single_step_status::attempt_limit_reached && step.get_attempts() == sp.maximum_attempts,
        "Unbounded BDF projection retries"
    );
}
int main()
try
{
    backend_type::init_device();
    ownership();
    history_transaction();
    for ( bool alias : { false, true } )
        for ( unsigned int mode = 0; mode < 3; ++mode )
            for ( unsigned int location = 1; location <= 3; ++location )
                startup_failure( location, mode, alias, false );
    for ( unsigned int location = 1; location <= 3; ++location )
        startup_failure( location, 0, false, true );
    std::cout << checks << " BDF projection lifecycle checks passed\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
