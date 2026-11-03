#include "implicit_rate_test.h"
#include "nonautonomous_test_problems.h"
#include <nmfd/time_steppers/integration/time_step_adaptation_matlab.h>
#include <limits>

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;

// u'=t*u, u(.3)=1. The time-dependent coefficient makes incorrect explicit
// stage times observable. Instrumentation checks that zero diagonals evaluate
// a rate, never a Newton stage, including an explicit stage after an implicit one.
using equation_type = rate_scalar_equation<coefficient_growth<double>>;
using problem_type  = rate_problem<equation_type>;
using types         = newton_types<problem_type>;

void test_rate_and_stages()
{
    auto             operations = std::make_shared<operations_type>( 1 );
    auto             problem    = std::make_shared<problem_type>( operations );
    vector_wrap_type state( *operations ), out( *operations ), before( *operations );
    state.start_use();
    out.start_use();
    before.start_use();
    operations->assign_scalar( 1., *state );
    operations->assign_scalar( .7, problem->previous_state() );
    operations->assign_scalar( .2, problem->known_rhs() );
    problem->set_time( .3 );
    problem->set_affine_shift( 7 );
    problem->apply( *state, *before );
    require( problem->evaluate_rate( *state, *out ), "Rate evaluation failed" );
    require(
        std::abs( operations->get_value_at_point( 0, *out ) - .3 ) < 1e-14,
        "Rate evaluation used the affine shift or known RHS"
    );
    problem->apply( *state, *out );
    require(
        operations->get_value_at_point( 0, *out ) == operations->get_value_at_point( 0, *before ) &&
            operations->get_value_at_point( 0, problem->previous_state() ) == .7 &&
            operations->get_value_at_point( 0, problem->known_rhs() ) == .2,
        "Rate evaluation altered the residual formulation or stage buffers"
    );

    for ( const char *method : { "CN", "ESDIRK2(1)3", "ARK324L2SA" } )
    {
        problem->rate_times.clear();
        problem->implicit_times.clear();
        auto step = make_step( operations, problem, .1, method );
        step->set_time( .3 );
        step->set_target_time( .4 );
        step->apply( *state, *out );
        require( step->get_status() == single_step_status::converged, "CN/ESDIRK step failed" );
        require(
            problem->rate_times.size() == 1 && problem->rate_times[0] == .3 &&
                problem->implicit_times.size() + 1 == step->table().size(),
            "Wrong explicit/implicit stage counts"
        );
        for ( std::size_t i = 1; i < step->table().size(); ++i )
            require(
                std::abs( problem->implicit_times[i - 1] - ( .3 + .1 * step->table().c( i ) ) ) < 1e-14,
                "Implicit stage used the wrong time"
            );
        step->finalize( adaptation_status::accepted, .4, *out );
    }

    const runge_kutta::butcher_table mixed(
        { { .25L, 0, 0 }, { .5L, 0, 0 }, { .25L, .25L, .5L } }, { .25L, .25L, .5L }, 1
    );
    problem->rate_times.clear();
    problem->implicit_times.clear();
    types::step_type step(
        operations, problem, std::make_shared<adaptation_type>( adaptation_type::params{ .1 } ),
        types::make_solver( operations ), mixed
    );
    step.set_time( .3 );
    step.set_target_time( .4 );
    step.apply( *state, *out );
    const auto y0       = 1 / ( 1 - .1 * .25 * .325 );
    const auto k0       = .325 * y0;
    const auto k1       = .35 * ( 1 + .1 * .5 * k0 );
    const auto expected = ( 1 + .1 * .25 * ( k0 + k1 ) ) / ( 1 - .1 * .5 * .4 );
    require(
        step.get_status() == single_step_status::converged && problem->rate_times.size() == 1 &&
            std::abs( problem->rate_times[0] - .35 ) < 1e-14 && problem->implicit_times.size() == 2 &&
            std::abs( operations->get_value_at_point( 0, *out ) - expected ) < 1e-10,
        "An interior zero diagonal used stale time, shift, state or rate"
    );
    step.finalize( adaptation_status::accepted, .4, *out );
}

struct failing_problem : problem_type
{
    std::shared_ptr<operations_type> operations;
    unsigned int                     failures = 0, calls = 0;
    bool                             nonfinite = false, throw_error = false;
    explicit failing_problem( std::shared_ptr<operations_type> ops )
        : problem_type( ops ), operations( std::move( ops ) )
    {
    }
    bool evaluate_rate( const vector_type &state, vector_type &out )
    {
        ++calls;
        if ( throw_error )
            throw std::runtime_error( "Injected rate backend error" );
        if ( failures )
        {
            --failures;
            return false;
        }
        if ( nonfinite )
        {
            operations->assign_scalar( std::numeric_limits<double>::quiet_NaN(), out );
            return true;
        }
        return problem_type::evaluate_rate( state, out );
    }
};

void test_failure_and_lifecycle()
{
    using failure_types = newton_types<failing_problem>;
    using controller    = integration::time_step_adaptation_matlab<operations_type>;
    using adaptive_step =
        runge_kutta::implicit_time_step<operations_type, failing_problem, controller, failure_types::solver_type>;
    auto             operations = std::make_shared<operations_type>( 1 );
    auto             problem    = std::make_shared<failing_problem>( operations );
    vector_wrap_type state( *operations ), out( *operations ), reference( *operations );
    state.start_use();
    out.start_use();
    reference.start_use();
    operations->assign_scalar( 1., *state );
    controller::params p;
    p.initial_step = p.maximum_step = .1;
    p.minimum_step                  = .001;
    p.relative_tolerance = p.absolute_tolerance = 1;
    problem->failures                           = 1;
    adaptive_step step(
        operations, problem, std::make_shared<controller>( operations, p ), failure_types::make_solver( operations ),
        adaptive_step::params{ "ESDIRK2(1)3" }
    );
    step.apply( *state, *out );
    require(
        step.get_status() == single_step_status::converged && step.get_attempts() == 2 && problem->calls == 2 &&
            step.get_dt() < .1,
        "Failed rate did not shrink and retry from the input state"
    );
    step.finalize( adaptation_status::accepted, step.get_dt(), *out );

    for ( const bool nonfinite : { false, true } )
    {
        problem->failures  = nonfinite ? 0 : 1;
        problem->nonfinite = nonfinite;
        problem->implicit_times.clear();
        operations->assign_scalar( 42., *out );
        auto failed = make_step( operations, problem, .1, "CN" );
        failed->apply( *state, *out );
        require(
            failed->get_status() ==
                    ( nonfinite ? single_step_status::failed_nonfinite : single_step_status::failed_solver ) &&
                problem->implicit_times.empty() && operations->get_value_at_point( 0, *out ) == 42 &&
                operations->get_value_at_point( 0, *state ) == 1,
            "Invalid rate reached Newton or corrupted the committed state"
        );
    }
    problem->nonfinite = false;
    problem->failures  = 99;
    adaptive_step exhausted(
        operations, problem, std::make_shared<controller>( operations, p ), failure_types::make_solver( operations ),
        adaptive_step::params{ "ESDIRK2(1)3", 2 }
    );
    exhausted.apply( *state, *out );
    require(
        exhausted.get_status() == single_step_status::attempt_limit_reached && exhausted.get_attempts() == 2 &&
            operations->get_value_at_point( 0, *out ) == 42,
        "Failed rate retries escaped the attempt budget"
    );
    problem->failures    = 0;
    problem->throw_error = true;
    bool threw           = false;
    try
    {
        auto failed = make_step( operations, problem, .1, "CN" );
        failed->apply( *state, *out );
    }
    catch ( const std::runtime_error & )
    {
        threw = true;
    }
    require(
        threw && operations->get_value_at_point( 0, *out ) == 42, "Rate backend exception was swallowed or committed"
    );
    problem->throw_error = false;

    auto modified = make_step( operations, problem, .1, "CN" );
    modified->apply( *state, *out );
    operations->assign_scalar( 2., *state );
    modified->finalize( adaptation_status::accepted_modified, .04, *state );
    modified->set_time( .04 );
    modified->apply( *state, *out );
    auto fresh = make_step( operations, problem, .1, "CN" );
    fresh->set_time( .04 );
    fresh->apply( *state, *reference );
    require(
        std::abs( operations->get_value_at_point( 0, *out ) - operations->get_value_at_point( 0, *reference ) ) < 1e-13,
        "Modified endpoint reused an obsolete explicit rate"
    );
    modified->finalize( adaptation_status::accepted, .14, *out );
    fresh->finalize( adaptation_status::accepted, .14, *reference );
}

void test_unavailable_estimate()
{
    using controller = integration::time_step_adaptation_matlab<operations_type>;
    using step_type  = runge_kutta::implicit_time_step<operations_type, problem_type, controller, types::solver_type>;
    auto             operations = std::make_shared<operations_type>( 1 );
    auto             problem    = std::make_shared<problem_type>( operations );
    vector_wrap_type state( *operations ), out( *operations );
    state.start_use();
    out.start_use();
    operations->assign_scalar( 1., *state );
    operations->assign_scalar( 42., *out );
    step_type step(
        operations, problem, std::make_shared<controller>( operations ), types::make_solver( operations ),
        step_type::params{ "CN" }
    );
    step.apply( *state, *out );
    require(
        step.get_status() == single_step_status::error_estimate_unavailable && problem->rate_calls == 0 &&
            operations->get_value_at_point( 0, *out ) == 42,
        "CN invented an embedded estimator or evaluated an unnecessary rate"
    );
}

void test_optional_capability()
{
    using legacy_problem        = stage_problem<equation_type>;
    using legacy_types          = newton_types<legacy_problem>;
    auto             operations = std::make_shared<operations_type>( 1 );
    auto             problem    = std::make_shared<legacy_problem>( operations );
    vector_wrap_type initial( *operations ), out( *operations );
    initial.start_use();
    out.start_use();
    operations->assign_scalar( 1., *initial );
    const auto g = 1.L - std::sqrt( 2.L ) / 2;
    // A padded SDIRK2 table: only its embedding needs the dummy first rate.
    const runge_kutta::butcher_table padded(
        { { 0, 0, 0 }, { 0, g, 0 }, { 0, 1 - g, g } }, { 0, 1 - g, g }, 2, {}, { .5L, 0, .5L }, 1
    );
    legacy_types::step_type step(
        operations, problem, std::make_shared<adaptation_type>(), legacy_types::make_solver( operations ), padded
    );
    step.apply( *initial, *out );
    require( step.get_status() == single_step_status::converged, "An unused explicit rate became mandatory" );
    step.finalize( adaptation_status::accepted, step.get_dt(), *out );
    using controller = integration::time_step_adaptation_matlab<operations_type>;
    using adaptive_step =
        runge_kutta::implicit_time_step<operations_type, legacy_problem, controller, legacy_types::solver_type>;
    bool rejected = false;
    try
    {
        adaptive_step invalid(
            operations, problem, std::make_shared<controller>( operations ), legacy_types::make_solver( operations ),
            padded
        );
    }
    catch ( const std::invalid_argument & )
    {
        rejected = true;
    }
    require( rejected, "Embedding consumed an unavailable explicit rate" );
    auto capable = std::make_shared<problem_type>( operations );
    rejected     = false;
    try
    {
        auto invalid = make_step( operations, capable, .1, "EE" );
    }
    catch ( const std::invalid_argument & )
    {
        rejected = true;
    }
    require( rejected, "Implicit engine accepted an entirely explicit table" );
}

struct invalid_rate_signature
{
    void evaluate_rate( const vector_type &, vector_type & );
};

int main()
try
{
    backend_type::init_device();
    static_assert( runge_kutta::detail::has_rate_evaluation<problem_type, vector_type>::value );
    static_assert( !runge_kutta::detail::has_rate_evaluation<stage_problem<equation_type>, vector_type>::value );
    static_assert( !runge_kutta::detail::has_rate_evaluation<invalid_rate_signature, vector_type>::value );
    static_assert( std::is_same_v<decltype( &problem_type::apply ), decltype( &stage_problem<equation_type>::apply )> );
    test_rate_and_stages();
    test_failure_and_lifecycle();
    test_unavailable_estimate();
    test_optional_capability();
    std::cout << "CN/ESDIRK rate evaluation, timed/interior stages, failures and commitment: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
