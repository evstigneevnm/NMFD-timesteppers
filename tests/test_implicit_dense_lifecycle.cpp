#include "implicit_dense_test.h"
#include <limits>
#include <optional>

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;

// u'=t*u, exact u=exp(t^2/2). This smooth non-autonomous ODE isolates
// pending-step lifetime, error rejection and failures of dense-only rate evaluation.
using problem_type = rate_problem<dense_growth>;
using types        = newton_types<problem_type>;
using step_type    = dense_step_type<problem_type>;

template <class Step, class = void>
struct has_dense_view : std::false_type
{
};
template <class Step>
struct has_dense_view<Step, std::void_t<decltype( std::declval<const Step &>().get_continuous_integration() )>>
    : std::true_type
{
};
static_assert( !has_dense_view<types::step_type>::value && has_dense_view<step_type>::value );

void test_lifetime()
{
    auto             operations = std::make_shared<operations_type>( 1 );
    auto             problem    = std::make_shared<problem_type>( operations );
    auto             step       = make_dense_step( operations, problem, .1, "IM" );
    vector_wrap_type state( *operations ), out( *operations ), sample( *operations ), reference( *operations );
    state.start_use();
    out.start_use();
    sample.start_use();
    reference.start_use();
    require_exception<std::logic_error>(
        [&] { step->get_continuous_integration(); }, "Dense view available before apply"
    );
    operations->assign_scalar( 1., *state );
    step->apply( *state, *state );
    require( step->get_status() == single_step_status::converged, "In-place dense step failed" );
    const auto old = step->get_continuous_integration();
    old.evaluate( 0, *sample );
    require( operations->get_value_at_point( 0, *sample ) == 1, "In-place apply lost the initial snapshot" );
    old.evaluate( 1, *sample );
    require(
        operations->get_value_at_point( 0, *sample ) == operations->get_value_at_point( 0, *state ),
        "In-place dense endpoint mismatch"
    );
    operations->assign( *sample, *reference );
    old.evaluate( .5, *state );
    old.evaluate( 1, *sample );
    require(
        operations->get_value_at_point( 0, *sample ) == operations->get_value_at_point( 0, *reference ),
        "Sampling into the caller candidate corrupted the endpoint"
    );
    require_exception<std::logic_error>( [&] { step->apply( *state, *out ); }, "Pending step was overwritten" );
    old.evaluate( .4, *sample );
    for ( const auto theta :
          { -.01, 1.01, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity() } )
    {
        operations->assign_scalar( 77., *sample );
        require_exception<std::invalid_argument>(
            [&] { old.evaluate( theta, *sample ); }, "Invalid dense theta was accepted"
        );
        require( operations->get_value_at_point( 0, *sample ) == 77, "Invalid dense request changed output" );
    }
    operations->assign_scalar( 2., *state );
    step->finalize( adaptation_status::accepted_modified, .04, *state );
    require_exception<std::logic_error>(
        [&] { old.evaluate( .5, *sample ); }, "Modified commit kept an old view alive"
    );
    step->set_time( .04 );
    step->apply( *state, *out );
    auto fresh = make_dense_step( operations, problem, .1, "IM" );
    fresh->set_time( .04 );
    fresh->apply( *state, *reference );
    require(
        std::abs( operations->get_value_at_point( 0, *out ) - operations->get_value_at_point( 0, *reference ) ) < 1e-13,
        "Shortened/modified endpoint retained obsolete dense rates"
    );
    require_exception<std::logic_error>( [&] { old.evaluate( .5, *sample ); }, "Old view observed a new pending step" );
    const auto current = step->get_continuous_integration();
    fresh->finalize( adaptation_status::accepted, .14, *reference );
    step->reset();
    require_exception<std::logic_error>( [&] { current.evaluate( .5, *sample ); }, "Reset kept dense output alive" );
    step->apply( *state, *out );
    require_exception<std::logic_error>(
        [&] { current.evaluate( .5, *sample ); }, "Reset allowed an old view to revive"
    );
    const auto failed = step->get_continuous_integration();
    step->finalize( adaptation_status::failed, .04, *state );
    require_exception<std::logic_error>(
        [&] { failed.evaluate( .5, *sample ); }, "Failed commit retained dense output"
    );
}

struct failing_rate_problem : problem_type
{
    using problem_type::problem_type;
    unsigned int calls = 0, failures = 0;
    bool         throw_error = false;
    bool         evaluate_rate( const vector_type &state, vector_type &rate )
    {
        ++calls;
        if ( throw_error )
            throw std::runtime_error( "Injected dense rate backend error" );
        if ( failures )
        {
            --failures;
            return false;
        }
        return problem_type::evaluate_rate( state, rate );
    }
};

struct invalid_rate_problem : problem_type
{
    std::shared_ptr<operations_type> operations;
    explicit invalid_rate_problem( std::shared_ptr<operations_type> ops )
        : problem_type( ops ), operations( std::move( ops ) )
    {
    }
    bool evaluate_rate( const vector_type &, vector_type &rate )
    {
        operations->assign_scalar( std::numeric_limits<double>::infinity(), rate );
        return true;
    }
};

void test_preparation_failure()
{
    auto             operations = std::make_shared<operations_type>( 1 );
    vector_wrap_type state( *operations ), out( *operations );
    state.start_use();
    out.start_use();
    operations->assign_scalar( 1., *state );
    auto problem = std::make_shared<failing_rate_problem>( operations );
    for ( const bool throws : { false, true } )
    {
        problem->failures    = throws ? 0 : 1;
        problem->throw_error = throws;
        auto step            = make_dense_step( operations, problem, .1, "IM" );
        operations->assign_scalar( 77., *out );
        if ( throws )
            require_exception<std::runtime_error>(
                [&] { step->apply( *state, *out ); }, "Dense rate exception was swallowed"
            );
        else
        {
            step->apply( *state, *out );
            require( step->get_status() == single_step_status::failed_solver, "Dense rate failure status was lost" );
        }
        require(
            problem->linearizations > 0 && problem->time == .05 && operations->get_value_at_point( 0, *out ) == 77 &&
                operations->get_value_at_point( 0, *state ) == 1,
            "Dense failure changed output or prepared stage time"
        );
        require_exception<std::logic_error>(
            [&] { step->get_continuous_integration(); }, "Failed preparation published a dense view"
        );
    }
    auto invalid = std::make_shared<invalid_rate_problem>( operations );
    auto step    = make_dense_step( operations, invalid, .1, "IM" );
    step->apply( *state, *out );
    require(
        step->get_status() == single_step_status::failed_nonfinite && operations->get_value_at_point( 0, *out ) == 77,
        "Nonfinite dense rate was published"
    );

    using controller    = integration::time_step_adaptation_matlab<operations_type>;
    using adaptive_step = dense_step_type<failing_rate_problem, controller>;
    controller::params p;
    p.initial_step = p.maximum_step = .1;
    p.relative_tolerance = p.absolute_tolerance = 1;
    problem->throw_error                        = false;
    problem->failures                           = 1;
    problem->calls                              = 0;
    adaptive_step retry(
        operations, problem, std::make_shared<controller>( operations, p ),
        newton_types<failing_rate_problem>::make_solver( operations ), adaptive_step::params{ "SDIRK3(1)3" }
    );
    retry.apply( *state, *out );
    require(
        retry.get_status() == single_step_status::converged && retry.get_attempts() == 2 && problem->calls == 2 &&
            retry.get_dt() < .1,
        "Dense preparation failure did not shrink and retry"
    );
    retry.get_continuous_integration().evaluate( 0, *out );
    require( operations->get_value_at_point( 0, *out ) == 1, "Dense retry used a rejected state" );
    retry.finalize( adaptation_status::accepted_modified, retry.get_dt(), *out );

    problem->failures = 99;
    adaptive_step exhausted(
        operations, problem, std::make_shared<controller>( operations, p ),
        newton_types<failing_rate_problem>::make_solver( operations ), adaptive_step::params{ "SDIRK3(1)3", 2 }
    );
    operations->assign_scalar( 77., *out );
    exhausted.apply( *state, *out );
    require(
        exhausted.get_status() == single_step_status::attempt_limit_reached && exhausted.get_attempts() == 2 &&
            operations->get_value_at_point( 0, *out ) == 77,
        "Dense failure escaped bounded rejection"
    );

    p.initial_step = p.maximum_step = .4;
    p.relative_tolerance = p.absolute_tolerance = 1e-9;
    problem->failures = problem->calls = 0;
    adaptive_step rejected(
        operations, problem, std::make_shared<controller>( operations, p ),
        newton_types<failing_rate_problem>::make_solver( operations ), adaptive_step::params{ "SDIRK3(1)3" }
    );
    rejected.apply( *state, *out );
    require(
        rejected.get_status() == single_step_status::converged && rejected.get_attempts() > 1 && problem->calls == 1,
        "Error-rejected attempts performed dense-only rate evaluations"
    );
    rejected.finalize( adaptation_status::accepted, rejected.get_dt(), *out );
}

struct counting_operations : operations_type
{
    using operations_type::operations_type;
    mutable unsigned int allocations = 0, live = 0, activations = 0, fail_at = 0;
    void                 start_use_vector( vector_type &v ) const override
    {
        const auto fresh = v.is_free();
        operations_type::start_use_vector( v );
        if ( fresh )
        {
            ++allocations;
            ++live;
        }
        if ( ++activations == fail_at )
            throw std::runtime_error( "Injected dense allocation activation error" );
    }
    void free_vector( vector_type &v ) const override
    {
        if ( !v.is_free() )
            --live;
        operations_type::free_vector( v );
    }
};

template <bool Dense>
void test_storage()
{
    using counted_step =
        runge_kutta::implicit_time_step<counting_operations, problem_type, adaptation_type, types::solver_type, Dense>;
    auto operations = std::make_shared<counting_operations>( 1 );
    {
        vector_wrap_type state( *operations ), out( *operations ), sample( *operations );
        state.start_use();
        out.start_use();
        sample.start_use();
        for ( const auto *method : dense_methods )
        {
            auto       problem      = std::make_shared<problem_type>( operations );
            const auto dependencies = operations->live;
            auto       solver       = types::make_solver( operations );
            const auto baseline     = operations->live;
            {
                counted_step step(
                    operations, problem, std::make_shared<adaptation_type>( adaptation_type::params{ .1 } ), solver,
                    typename counted_step::params{ method }
                );
                const auto &table = step.table();
                const auto  rates =
                    Dense ? table.dense_outout_stage_count() : table.size() - ( std::string( method ) == "IM" ? 0 : 1 );
                require(
                    operations->live - baseline == 1 + rates + ( Dense ? 1 : 0 ),
                    "Implicit dense vector storage budget mismatch"
                );
                require(
                    step.dense_output_order() == ( Dense ? table.dense_order() : 0 ), "Disabled dense order is not zero"
                );
                operations->assign_scalar( 1., *state );
                step.apply( *state, *out );
                require( step.get_status() == single_step_status::converged, "Counted dense step failed" );
                if constexpr ( Dense )
                {
                    const auto allocations = operations->allocations, calls = problem->rate_calls;
                    const auto view = step.get_continuous_integration();
                    for ( int i = 0; i <= 20; ++i )
                        view.evaluate( i / 20., *sample );
                    require(
                        operations->allocations == allocations && problem->rate_calls == calls,
                        "Sampling allocated or evaluated a rate"
                    );
                }
                else
                    require(
                        problem->rate_calls == ( table.a( 0, 0 ) == 0 ? 1 : 0 ),
                        "Disabled dense output added a rate call"
                    );
                step.finalize( adaptation_status::accepted, step.get_dt(), *out );
            }
            // GMRES activates its retained scratch lazily on the first solve.
            solver.reset();
            require( operations->live == dependencies, "Dense step or solver leaked vector storage" );
        }
        auto       problem  = std::make_shared<problem_type>( operations );
        auto       solver   = types::make_solver( operations );
        const auto baseline = operations->live;
        const auto count    = Dense ? 4u : 2u; // IM: work, midpoint rate, optional initial rate and snapshot.
        for ( unsigned int n = 1; n <= count; ++n )
        {
            operations->fail_at = operations->activations + n;
            require_exception<std::runtime_error>(
                [&] {
                    counted_step step(
                        operations, problem, std::make_shared<adaptation_type>(), solver,
                        typename counted_step::params{ "IM" }
                    );
                },
                "Injected vector activation failure was not reached"
            );
            operations->fail_at = 0;
            require( operations->live == baseline, "Partially constructed dense workspace leaked" );
        }
    }
    require( operations->live == 0, "Dense dependencies retained vectors after destruction" );
}

void test_optional_support()
{
    using plain_problem = stage_problem<dense_growth>;
    using plain_types   = newton_types<plain_problem>;
    auto operations     = std::make_shared<operations_type>( 1 );
    auto problem        = std::make_shared<plain_problem>( operations );
    auto solver         = plain_types::make_solver( operations );
    auto adaptation     = std::make_shared<adaptation_type>();
    for ( const auto *method : { "IM", "SDIRK3(1)3" } )
    {
        plain_types::step_type ordinary(
            operations, problem, adaptation, solver, plain_types::step_type::params{ method }
        );
        require_exception<std::invalid_argument>(
            [&] {
                dense_step_type<plain_problem> dense(
                    operations, problem, adaptation, solver, dense_step_type<plain_problem>::params{ method }
                );
            },
            "Dense-only required rate provider was silently omitted"
        );
    }
    require_exception<std::invalid_argument>(
        [&] {
            dense_step_type<plain_problem> invalid(
                operations, problem, adaptation, solver, runge_kutta::butcher_table( { { 1 } }, { 1 }, 1 )
            );
        },
        "Custom table without a continuous extension was accepted"
    );
    // This explicit dummy rate is unused even by the linear continuous extension.
    const runge_kutta::butcher_table padded(
        { { 0, 0 }, { 0, 1 } }, { 0, 1 }, 1, {}, {}, 0, 2e-15L, { { 0 }, { 1 } }, 1
    );
    dense_step_type<plain_problem> dense( operations, problem, adaptation, solver, padded );
    vector_wrap_type               state( *operations ), out( *operations );
    state.start_use();
    out.start_use();
    operations->assign_scalar( 1., *state );
    dense.apply( *state, *out );
    require( dense.get_status() == single_step_status::converged, "Unused dense dummy rate became mandatory" );
    dense.finalize( adaptation_status::accepted, dense.get_dt(), *out );
}

void test_additional_stages()
{
    auto operations = std::make_shared<operations_type>( 1 );
    auto problem    = std::make_shared<problem_type>( operations );
    // A first-order extension whose two extra explicit rates depend on earlier
    // derivatives. Their contributions cancel at the endpoint, not in the interior.
    const runge_kutta::butcher_table table(
        { { 1 } }, { 1 }, 1, {}, {}, 0, 2e-15L, { { 1, 0 }, { 1, -1 }, { -1, 1 } }, 1,
        { { .5L, 0, 0 }, { 0, .25L, 0 } }, { .5L, .25L }
    );
    step_type step(
        operations, problem, std::make_shared<adaptation_type>( adaptation_type::params{ .1 } ),
        types::make_solver( operations ), table
    );
    vector_wrap_type state( *operations ), out( *operations ), sample( *operations );
    state.start_use();
    out.start_use();
    sample.start_use();
    operations->assign_scalar( 1., *state );
    step.set_time( .3 );
    step.apply( *state, *out );
    require(
        step.get_status() == single_step_status::converged && problem->rate_times.size() == 2 &&
            std::abs( problem->rate_times[0] - .35 ) < 1e-14 && std::abs( problem->rate_times[1] - .325 ) < 1e-14 &&
            problem->time == .4,
        "Additional dense stages used incorrect times"
    );
    const auto k0    = .4 / ( 1 - .1 * .4 );
    const auto k1    = .35 * ( 1 + .05 * k0 );
    const auto k2    = .325 * ( 1 + .025 * k1 );
    const auto theta = .37;
    step.get_continuous_integration().evaluate( theta, *sample );
    require(
        std::abs(
            operations->get_value_at_point( 0, *sample ) -
            ( 1 + .1 * ( theta * k0 + theta * ( 1 - theta ) * ( k1 - k2 ) ) )
        ) < 1e-13,
        "Additional dense stages lost their derivative dependency"
    );
    step.finalize( adaptation_status::accepted, .4, *out );

    // These finite long-double coefficients overflow when converted to double.
    // Reject the invalid extra state before calling the problem's rate provider.
    const auto huge = std::numeric_limits<long double>::max() / 4;
    if ( huge > std::numeric_limits<double>::max() )
    {
        const runge_kutta::butcher_table overflow(
            { { 1 } }, { 1 }, 1, {}, {}, 0, 2e-15L, { { 1, 0 }, { 1, -1 }, { -1, 1 } }, 1,
            { { 0, 0, 0 }, { huge, -huge, 0 } }, { 0, 0 }
        );
        problem->rate_times.clear();
        step_type invalid(
            operations, problem, std::make_shared<adaptation_type>( adaptation_type::params{ .1 } ),
            types::make_solver( operations ), overflow
        );
        invalid.set_time( .3 );
        operations->assign_scalar( 77., *out );
        invalid.apply( *state, *out );
        require(
            invalid.get_status() == single_step_status::failed_nonfinite && problem->rate_times.size() == 1 &&
                problem->time == .4 && operations->get_value_at_point( 0, *out ) == 77,
            "Invalid additional dense state reached the provider or was committed"
        );
    }
}

using section_problem = rate_problem<rate_scalar_equation<polynomial_growth>>;
using section_step    = dense_step_type<section_problem>;
struct section_manager
{
    std::shared_ptr<operations_type>                                 operations;
    std::shared_ptr<section_step>                                    step;
    double                                                          *end    = nullptr;
    integration_status                                               status = integration_status::running;
    std::optional<runge_kutta::continuous_integration<section_step>> last_view;
    void                                                             set_time_interval( const double &, double &next )
    {
        end    = &next;
        status = integration_status::running;
    }
    bool apply( const vector_type &in, vector_type &out )
    {
        const auto threshold = polynomial_growth::exact( .47 );
        if ( operations->get_value_at_point( 0, in ) > threshold ||
             operations->get_value_at_point( 0, out ) < threshold )
            return false;
        last_view   = step->get_continuous_integration();
        double left = 0, right = 1;
        for ( int i = 0; i < 45; ++i )
        {
            const auto theta = ( left + right ) / 2;
            last_view->evaluate( theta, out );
            if ( operations->get_value_at_point( 0, out ) < threshold )
                left = theta;
            else
                right = theta;
        }
        *end   = last_view->evaluate( ( left + right ) / 2, out );
        status = integration_status::stopped_by_external_operation;
        return true;
    }
    integration_status get_status() const
    {
        return status;
    }
};

void test_section()
{
    auto operations = std::make_shared<operations_type>( 1 );
    auto problem    = std::make_shared<section_problem>( operations );
    auto step       = make_dense_step( operations, problem, .1, "SDIRK3(1)3" );
    auto external   = std::make_shared<section_manager>( section_manager{ operations, step } );
    integration::time_integrator<operations_type, section_step, section_manager> integrator(
        operations, step, {}, external
    );
    vector_wrap_type initial( *operations ), out( *operations );
    initial.start_use();
    out.start_use();
    operations->assign_scalar( polynomial_growth::exact( .3 ), *initial );
    integrator.set_time_interval( .3, .8 );
    for ( int repeat = 0; repeat < 2; ++repeat )
    {
        integrator.apply( *initial, *out );
        require(
            integrator.get_status() == integration_status::stopped_by_external_operation &&
                std::abs( integrator.get_final_time() - .47 ) < 2e-12 &&
                std::abs( operations->get_value_at_point( 0, *out ) - polynomial_growth::exact( .47 ) ) < 2e-12,
            "Implicit dense section did not commit matching time/state on restart"
        );
        require_exception<std::logic_error>(
            [&] { external->last_view->evaluate( .5, *out ); }, "Section retained a finalized dense view"
        );
    }
}

int main()
try
{
    backend_type::init_device();
    test_lifetime();
    test_preparation_failure();
    test_storage<false>();
    test_storage<true>();
    test_optional_support();
    test_additional_stages();
    test_section();
    std::cout << "Implicit dense lifetime, storage, failures, rollback, ownership and section commitment: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
