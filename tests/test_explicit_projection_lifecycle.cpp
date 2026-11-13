#include "explicit_projection_problem.h"

using namespace projection_tests;

#ifdef TEST_PROJECTION_VOID_MUST_FAIL
struct void_projector
{
    void apply( space_type::vector_type & )
    {
    }
};
void reject_void_projector(
    std::shared_ptr<space_type> ops, std::shared_ptr<problem<space_type>> p,
    std::shared_ptr<fixed_control<space_type>> control
)
{
    using invalid_step = runge_kutta::explicit_time_step<
        space_type, problem<space_type>, fixed_control<space_type>, false, void_projector>;
    invalid_step invalid( ops, p, control, {}, std::make_shared<void_projector>() );
}
#endif

// The same constrained linear system as the accuracy test, with injected
// projection failures. Failed stage work must never replace committed data.
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
        h          = .1;
        rejections = assessments = 0;
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

void check_default_and_ownership()
{
    using R = reference_step<space_type>;
    using S = projected_step<space_type>;
    static_assert( !R::projects_stage_rates && S::projects_stage_rates );
    static_assert( int( single_step_status::step_size_underflow ) == 7, "Existing status values changed" );
    auto               ops     = std::make_shared<space_type>( 3 );
    auto               p       = std::make_shared<problem<space_type>>( ops );
    auto               control = std::make_shared<fixed_control<space_type>>();
    buffer<space_type> state( *ops ), out( *ops );
    exact( *ops, *state );
    const auto before  = ops->allocations;
    bool       refused = false;
    try
    {
        S invalid( ops, p, control );
    }
    catch ( const std::invalid_argument & )
    {
        refused = true;
    }
    check( refused && ops->allocations == before, "Missing projector was not refused before allocation" );
    std::size_t default_storage, default_checks;
    {
        R step( ops, p, control, R::params{ "RK4" } );
        default_storage  = ops->allocations - before;
        const auto scans = ops->validity_checks;
        step.apply( *state, *out );
        default_checks = ops->validity_checks - scans;
        step.finalize( adaptation_status::accepted, step.get_dt(), *out );
    }
    {
        const auto allocations = ops->allocations;
        // The canonical identity returns void. It must never be instantiated
        // as a Boolean projector or add a validity scan/vector allocation.
        auto identity = std::make_shared<nmfd::operations::ident_operator<space_type>>( ops );
        R    step( ops, p, control, R::params{ "RK4" }, identity );
        check( ops->allocations - allocations == default_storage, "Identity added workspace" );
        const auto scans = ops->validity_checks;
        step.apply( *state, *out );
        check( ops->validity_checks - scans == default_checks, "Identity added validity scans" );
        step.finalize( adaptation_status::accepted, step.get_dt(), *out );
    }
    std::weak_ptr<projector<space_type>> weak;
    {
        auto projection        = std::make_shared<projector<space_type>>( ops );
        weak                   = projection;
        const auto allocations = ops->allocations;
        S          step( ops, p, control, S::params{ "RK4" }, projection );
        projection.reset();
        check( !weak.expired(), "Stepper did not retain the projector" );
        check( ops->allocations - allocations == default_storage, "Projection added stage workspace" );
        const auto scans = ops->validity_checks;
        step.apply( *state, *out );
        check(
            ops->validity_checks - scans == default_checks + step.table().size(),
            "Expected exactly one extra validity check per projection"
        );
        step.finalize( adaptation_status::accepted, step.get_dt(), *out );
    }
    check( weak.expired(), "Projector ownership cycle" );
}

void check_failure( bool dense_failure, unsigned int mode, bool alias )
{
    using S                       = projected_step<space_type, observed_control, true>;
    auto               ops        = std::make_shared<space_type>( 3 );
    auto               p          = std::make_shared<problem<space_type>>( ops );
    auto               projection = std::make_shared<projector<space_type>>( ops );
    auto               control    = std::make_shared<observed_control>( observed_control::params{ .1 } );
    S                  step( ops, p, control, S::params{ "DOP853" }, projection );
    buffer<space_type> state( *ops ), out( *ops ), initial( *ops ), sentinel( *ops ), scratch( *ops );
    exact( *ops, *state );
    exact( *ops, *initial );
    ops->assign_scalar( 7., *out );
    ops->assign( *out, *sentinel );
    const auto failure = dense_failure ? step.table().size() + 1 : 2;
    if ( mode == 0 )
        projection->fail_on = failure;
    else if ( mode == 1 )
        projection->invalid_on = failure;
    else if ( mode == 2 )
        projection->throw_on = failure;
    else
        p->invalid_on = failure;
    bool threw = false;
    try
    {
        step.apply( *state, alias ? *state : *out );
    }
    catch ( const std::runtime_error & )
    {
        threw = true;
    }
    check( threw == ( mode == 2 ), "Projection exception was swallowed or fabricated" );
    if ( mode != 2 )
        check(
            step.get_status() ==
                ( mode == 0 ? single_step_status::failed_projection : single_step_status::failed_nonfinite ),
            "Wrong projection failure status"
        );
    check( control->assessments == ( dense_failure ? 1 : 0 ), "Invalid candidate reached error assessment" );
    check( difference( *ops, *state, *initial, *scratch ) == 0, "Failed projection changed committed input" );
    check( difference( *ops, *out, *sentinel, *scratch ) == 0, "Failed projection published output" );
    check(
        projection->calls == failure - ( mode == 3 ? 1 : 0 ), "Projection ran after failed/nonfinite physical rate"
    );
    bool refused = false;
    try
    {
        step.get_continuous_integration();
    }
    catch ( const std::logic_error & )
    {
        refused = true;
    }
    check( refused, "Failed projection published a dense view" );
    projection->fail_on = projection->invalid_on = projection->throw_on = p->invalid_on = 0;
    step.reset();
    step.apply( *state, *out );
    check( step.get_status() == single_step_status::converged, "Projection failure prevented restart" );
    step.finalize( adaptation_status::accepted, .1, *out );
}

void check_retries()
{
    using S                       = projected_step<space_type, retry_control>;
    auto               ops        = std::make_shared<space_type>( 3 );
    auto               p          = std::make_shared<problem<space_type>>( ops );
    auto               projection = std::make_shared<projector<space_type>>( ops );
    auto               control    = std::make_shared<retry_control>();
    buffer<space_type> state( *ops ), out( *ops ), scratch( *ops ), expected( *ops );
    for ( unsigned int failure_stage : { 1u, 2u } )
    {
        S step( ops, p, control, S::params{ "RK45", 3, true }, projection );
        exact( *ops, *state );
        exact( *ops, *expected );
        projection->fail_on = projection->calls + failure_stage;
        const auto calls    = p->calls;
        step.apply( *state, *out );
        check(
            step.get_status() == single_step_status::converged && step.get_attempts() == 2,
            "Projection failure did not retry"
        );
        check(
            step.get_dt() == .05 && control->rejections == 1 && control->assessments == 1,
            "Projection retry changed adaptation accounting"
        );
        check( difference( *ops, *state, *expected, *scratch ) == 0, "Retry changed input" );
        const auto stages = step.table().size();
        check(
            p->calls - calls == failure_stage + stages - ( failure_stage == 2 ? 1 : 0 ),
            "Retry reused a failed projection or discarded a valid first rate"
        );
        check( constraint( *ops, *out, *scratch ) < 2e-15, "Retry retained a failed stage buffer" );
        step.finalize( adaptation_status::accepted, .05, *out );
    }
    projection->always_fail = true;
    S exhausted( ops, p, control, S::params{ "RK45", 3, true }, projection );
    exhausted.apply( *state, *out );
    check(
        exhausted.get_status() == single_step_status::attempt_limit_reached && exhausted.get_attempts() == 3,
        "Projection retries exceeded the attempt budget"
    );
}

void check_fsal()
{
    using S         = projected_step<space_type>;
    auto ops        = std::make_shared<space_type>( 3 );
    auto p          = std::make_shared<problem<space_type>>( ops );
    auto projection = std::make_shared<projector<space_type>>( ops );
    auto control    = std::make_shared<fixed_control<space_type>>( fixed_control<space_type>::params{ .125 } );
    S    step( ops, p, control, S::params{ "RK45", 32, true }, projection );
    buffer<space_type> state( *ops ), sample( *ops );
    exact( *ops, *state );
    for ( unsigned int i = 0; i < 5; ++i )
    {
        if ( i == 2 )
        {
            projection->enabled = false;
            step.invalidate_rhs_cache();
        }
        if ( i == 4 )
            step.reset();
        step.set_time( i * .125 );
        const auto calls = projection->calls;
        step.apply( *state, *state );
        check( step.get_status() == single_step_status::converged, "Projected FSAL step failed" );
        check(
            projection->calls - calls == step.table().size() - ( i == 1 || i == 3 ? 1 : 0 ),
            "Projected FSAL reuse/invalidation count"
        );
        step.finalize(
            i == 3 ? adaptation_status::accepted_modified : adaptation_status::accepted, ( i + 1 ) * .125, *state
        );
    }
    check( projection->calls == p->calls, "FSAL projected a cached rate twice or skipped a new rate" );
}

int main()
try
{
    backend_type::init_device();
    check_default_and_ownership();
    for ( bool dense : { false, true } )
        for ( unsigned int mode = 0; mode < 4; ++mode )
            for ( bool alias : { false, true } )
                check_failure( dense, mode, alias );
    check_retries();
    check_fsal();
    std::cout << checks << " explicit projection lifecycle checks passed\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
