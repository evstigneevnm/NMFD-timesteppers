#include "implicit_imex_dense_test.h"
#include <limits>

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;

// u'=.25*u-1.25*u=-u. Fresh endpoint-rate failures must not publish a
// candidate, advance controller history, alter stage bindings or retain stale views.
struct dense_partition : rate_problem<imex_linear_equation>
{
    using base = rate_problem<imex_linear_equation>;
    std::shared_ptr<operations_type> ops;
    unsigned int                     calls = 0, fail_on = 0, failure_period = 0;
    int                              mode       = 0; // false, nonfinite output, or exception.
    double                           last_state = 0, last_time = 0;
    dense_partition( std::shared_ptr<operations_type> operations, double lambda )
        : base( operations, imex_linear_equation{ lambda } ), ops( std::move( operations ) )
    {
    }
    bool evaluate_rate( const vector_type &state, vector_type &rate )
    {
        ++calls;
        last_state = ops->get_value_at_point( 0, state );
        last_time  = this->time;
        if ( calls == fail_on || ( failure_period && calls % failure_period == 0 ) )
        {
            if ( mode == 2 )
                throw std::runtime_error( "Injected IMEX endpoint rate exception" );
            if ( mode == 0 )
                return false;
            ops->assign_scalar( std::numeric_limits<double>::infinity(), rate );
            return true;
        }
        return base::evaluate_rate( state, rate );
    }
};
using problem_type = imex_problem<dense_partition, dense_partition>;
using types        = newton_types<dense_partition>;
using step_type    = imex_dense_step<problem_type>;

auto assembly( std::shared_ptr<operations_type> ops )
{
    return std::make_shared<problem_type>(
        std::make_shared<dense_partition>( ops, .25 ), std::make_shared<dense_partition>( ops, -1.25 )
    );
}

template <class Step, class = void>
struct has_imex_dense_view : std::false_type
{
};
template <class Step>
struct has_imex_dense_view<Step, std::void_t<decltype( std::declval<const Step &>().get_continuous_integration() )>>
    : std::true_type
{
};
static_assert( !has_imex_dense_view<imex_step<problem_type>>::value && has_imex_dense_view<step_type>::value );

void test_views_and_rates()
{
    auto             ops     = std::make_shared<operations_type>( 1 );
    auto             problem = assembly( ops );
    auto             step    = make_imex_dense_step( ops, problem, .1 );
    vector_wrap_type state( *ops ), out( *ops ), sample( *ops ), reference( *ops );
    state.start_use();
    out.start_use();
    sample.start_use();
    reference.start_use();
    require_exception<std::logic_error>( [&] { step->get_continuous_integration(); }, "Premature IMEX dense view" );
    ops->assign_scalar( 1., *state );
    step->apply( *state, *state );
    require( step->get_status() == single_step_status::converged, "In-place IMEX dense solve failed" );
    const auto old = step->get_continuous_integration();
    old.evaluate( 0, *sample );
    require( ops->get_value_at_point( 0, *sample ) == 1, "Dense in-place initial snapshot was overwritten" );
    old.evaluate( 1, *sample );
    const auto endpoint = ops->get_value_at_point( 0, *state );
    auto      &e        = *problem->partitions->explicit_part;
    auto      &i        = *problem->partitions->implicit_part;
    require(
        e.calls == 5 && i.calls == 2 && e.last_state == endpoint && i.last_state == endpoint && e.last_time == .1 &&
            i.last_time == .1,
        "Dense output used internal stage rates instead of the actual combined endpoint"
    );
    const auto calls          = e.calls + i.calls;
    const auto linearizations = i.linearizations;
    const auto jacobian       = i.get_jacobi_operator()->matrix;
    const auto previous       = ops->get_value_at_point( 0, i.previous_state() );
    const auto rhs            = ops->get_value_at_point( 0, i.known_rhs() );
    old.evaluate( .4, *state );
    old.evaluate( 1, *sample );
    require(
        ops->get_value_at_point( 0, *sample ) == endpoint && e.calls + i.calls == calls &&
            i.linearizations == linearizations && i.get_jacobi_operator()->matrix( 0, 0 ) == jacobian( 0, 0 ) &&
            ops->get_value_at_point( 0, i.previous_state() ) == previous &&
            ops->get_value_at_point( 0, i.known_rhs() ) == rhs,
        "Dense sampling changed the endpoint, problem buffers, Jacobian or rate counts"
    );
    for ( const double theta :
          { -.1, 1.1, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity() } )
    {
        ops->assign_scalar( 77., *sample );
        require_exception<std::invalid_argument>( [&] { old.evaluate( theta, *sample ); }, "Invalid IMEX dense theta" );
        require( ops->get_value_at_point( 0, *sample ) == 77, "Invalid dense theta modified output" );
    }
    require_exception<std::logic_error>( [&] { step->apply( *state, *out ); }, "Pending IMEX view was overwritten" );
    ops->assign_scalar( 2., *state );
    step->finalize( adaptation_status::accepted_modified, .04, *state );
    require_exception<std::logic_error>( [&] { old.evaluate( .5, *sample ); }, "Modified endpoint retained IMEX view" );
    step->set_time( .04 );
    step->apply( *state, *out );
    auto fresh = make_imex_dense_step( ops, assembly( ops ), .1 );
    fresh->set_time( .04 );
    fresh->apply( *state, *reference );
    require(
        std::abs( ops->get_value_at_point( 0, *out ) - ops->get_value_at_point( 0, *reference ) ) < 1e-13,
        "Modified endpoint retained obsolete IMEX dense rates"
    );
    require_exception<std::logic_error>( [&] { old.evaluate( .5, *sample ); }, "Old IMEX view revived" );
    fresh->finalize( adaptation_status::accepted, .14, *reference );
    const auto current = step->get_continuous_integration();
    step->reset();
    require_exception<std::logic_error>( [&] { current.evaluate( .5, *sample ); }, "Reset retained IMEX view" );
    step->apply( *state, *out );
    require_exception<std::logic_error>( [&] { current.evaluate( .5, *sample ); }, "Reset view revived in a new step" );
    const auto failed = step->get_continuous_integration();
    step->finalize( adaptation_status::failed, .04, *state );
    require_exception<std::logic_error>( [&] { failed.evaluate( .5, *sample ); }, "Failed commit retained IMEX view" );
}

void test_endpoint_failures()
{
    auto             ops = std::make_shared<operations_type>( 1 );
    vector_wrap_type state( *ops ), out( *ops );
    state.start_use();
    out.start_use();
    ops->assign_scalar( 1., *state );
    // ARS233's last internal stage is NOT at the endpoint. Opt in explicitly
    // to its order-three Hermite extension to verify time restoration as well.
    auto                                  pair = runge_kutta::make_imex_butcher_table( "IMEX_ARS233" );
    const runge_kutta::imex_butcher_table table(
        pair.explicit_table, pair.implicit_table, 3, 0,
        runge_kutta::imex_butcher_table::dense_output_type::endpoint_hermite
    );
    for ( int source = 0; source < 2; ++source )
        for ( int mode = 0; mode < 3; ++mode )
        {
            auto  problem = assembly( ops );
            auto &e       = *problem->partitions->explicit_part;
            auto &i       = *problem->partitions->implicit_part;
            e.mode = i.mode = mode;
            if ( source == 0 )
                e.fail_on = 4;
            else
                i.fail_on = 2;
            step_type step(
                ops, problem, std::make_shared<adaptation_type>( adaptation_type::params{ .1 } ),
                types::make_solver( ops ), table
            );
            ops->assign_scalar( 77., *out );
            if ( mode == 2 )
                require_exception<std::runtime_error>(
                    [&] { step.apply( *state, *out ); }, "Dense exception swallowed"
                );
            else
            {
                step.apply( *state, *out );
                require(
                    step.get_status() ==
                        ( mode == 0 ? single_step_status::failed_solver : single_step_status::failed_nonfinite ),
                    "Dense endpoint failure status was lost"
                );
            }
            const double stage_time = .1 * double( pair.explicit_table.c( 2 ) );
            require(
                ops->get_value_at_point( 0, *out ) == 77 && ops->get_value_at_point( 0, *state ) == 1 &&
                    e.time == stage_time && i.time == stage_time && i.linearizations > 0,
                "Failed preparation changed committed state or prepared stage time"
            );
            require_exception<std::logic_error>(
                [&] { step.get_continuous_integration(); }, "Failed endpoint published view"
            );
        }

    using controller    = observed_controller<integration::time_step_adaptation_scipy<operations_type>>;
    using adaptive_step = imex_dense_step<problem_type, controller>;
    for ( int source = 0; source < 2; ++source )
    {
        auto problem = assembly( ops );
        if ( source == 0 )
            problem->partitions->explicit_part->fail_on = 5;
        else
            problem->partitions->implicit_part->fail_on = 2;
        auto p         = controller_parameters<controller>( 1. );
        p.initial_step = p.maximum_step = .1;
        auto control                    = std::make_shared<controller>( ops, p );
        control->expected_order         = 3;
        adaptive_step step(
            ops, problem, control, types::make_solver( ops ), adaptive_step::params{ "IMEX_ARK324L2SA" }
        );
        step.apply( *state, *out );
        require(
            step.get_status() == single_step_status::converged && step.get_attempts() == 2 && step.get_dt() < .1 &&
                control->commits == 0 && control->numerical_rejections == 1,
            "Dense-only failure did not reject without committing"
        );
        auto             fresh = make_imex_dense_step( ops, assembly( ops ), step.get_dt() );
        vector_wrap_type reference( *ops );
        reference.start_use();
        fresh->apply( *state, *reference );
        require(
            std::abs( ops->get_value_at_point( 0, *out ) - ops->get_value_at_point( 0, *reference ) ) < 1e-13,
            "Dense retry used failed rates"
        );
        step.finalize( adaptation_status::accepted, step.get_dt(), *out );
        fresh->finalize( adaptation_status::accepted, fresh->get_dt(), *reference );
        require( control->commits == 1, "Dense retry committed at the wrong boundary" );
    }
    auto problem   = assembly( ops );
    auto p         = controller_parameters<controller>( 1e-9 );
    p.initial_step = p.maximum_step = .4;
    auto control                    = std::make_shared<controller>( ops, p );
    control->expected_order         = 3;
    adaptive_step rejected(
        ops, problem, control, types::make_solver( ops ), adaptive_step::params{ "IMEX_ARK324L2SA" }
    );
    rejected.apply( *state, *out );
    require(
        rejected.get_status() == single_step_status::converged && rejected.get_attempts() > 1 &&
            problem->partitions->explicit_part->calls == 4 * rejected.get_attempts() + 1 &&
            problem->partitions->implicit_part->calls == rejected.get_attempts() + 1,
        "Error-rejected attempts evaluated dense-only endpoint rates"
    );
    rejected.finalize( adaptation_status::accepted, rejected.get_dt(), *out );
    problem                                            = assembly( ops );
    problem->partitions->explicit_part->failure_period = 5;
    p.relative_tolerance = p.absolute_tolerance = 1;
    control                                     = std::make_shared<controller>( ops, p );
    control->expected_order                     = 3;
    adaptive_step exhausted(
        ops, problem, control, types::make_solver( ops ), adaptive_step::params{ "IMEX_ARK324L2SA", 2 }
    );
    ops->assign_scalar( 77., *out );
    exhausted.apply( *state, *out );
    require(
        exhausted.get_status() == single_step_status::attempt_limit_reached && exhausted.get_attempts() == 2 &&
            ops->get_value_at_point( 0, *out ) == 77 && control->commits == 0,
        "Unbounded dense rejection"
    );
}

struct counted_operations : operations_type
{
    using operations_type::operations_type;
    mutable unsigned int allocations = 0, live = 0;
    void                 start_use_vector( vector_type &v ) const override
    {
        const auto fresh = v.is_free();
        operations_type::start_use_vector( v );
        if ( fresh )
        {
            ++allocations;
            ++live;
        }
    }
    void free_vector( vector_type &v ) const override
    {
        if ( !v.is_free() )
            --live;
        operations_type::free_vector( v );
    }
};

template <bool Dense>
unsigned int storage_budget()
{
    using counted_step =
        runge_kutta::imex_time_step<counted_operations, problem_type, adaptation_type, types::solver_type, Dense>;
    auto         ops    = std::make_shared<counted_operations>( 1 );
    unsigned int budget = 0;
    {
        auto         problem  = assembly( ops );
        auto         solver   = types::make_solver( ops );
        const auto   baseline = ops->live;
        counted_step step(
            ops, problem, std::make_shared<adaptation_type>( adaptation_type::params{ .1 } ), solver,
            typename counted_step::params{ "IMEX_ARK324L2SA" }
        );
        budget = ops->live - baseline;
        vector_wrap_type state( *ops ), out( *ops ), sample( *ops );
        state.start_use();
        out.start_use();
        sample.start_use();
        ops->assign_scalar( 1., *state );
        unsigned int warmed = 0;
        for ( int run = 0; run < 3; ++run )
        {
            step.reset();
            step.apply( *state, *out );
            require( step.get_status() == single_step_status::converged, "Dense workspace repeated solve failed" );
            if constexpr ( Dense )
                for ( int n = 0; n < 20; ++n )
                    step.get_continuous_integration().evaluate( .37, *sample );
            if ( run )
                require( ops->allocations == warmed, "IMEX dense sampling/solve allocated vectors after warmup" );
            warmed = ops->allocations;
            step.finalize( adaptation_status::accepted, step.get_dt(), *out );
        }
    }
    require( ops->live == 0, "IMEX dense workspace leaked" );
    return budget;
}

struct binding_failure
{
    void set_nonlinear_operator( std::shared_ptr<dense_partition> )
    {
        throw std::runtime_error( "Injected dense solver binding failure" );
    }
    bool solve( vector_type & )
    {
        return false;
    }
};

void test_construction_cleanup()
{
    auto       ops      = std::make_shared<counted_operations>( 1 );
    auto       problem  = assembly( ops );
    const auto baseline = ops->live;
    using failed_step =
        runge_kutta::imex_time_step<counted_operations, problem_type, adaptation_type, binding_failure, true>;
    require_exception<std::runtime_error>(
        [&] {
            failed_step step(
                ops, problem, std::make_shared<adaptation_type>(), std::make_shared<binding_failure>(),
                failed_step::params{ "IMEX_ARK324L2SA" }
            );
        },
        "Dense binding failure was swallowed"
    );
    require( ops->live == baseline, "Failed construction leaked IMEX dense snapshot or rates" );
}

void test_optional_capabilities()
{
    using explicit_type = rate_problem<imex_linear_equation>;
    using implicit_type = stage_problem<imex_linear_equation>;
    using problem       = imex_problem<explicit_type, implicit_type>;
    using dense_type    = imex_dense_step<problem>;
    auto ops            = std::make_shared<operations_type>( 1 );
    auto assembly       = std::make_shared<problem>(
        std::make_shared<explicit_type>( ops, imex_linear_equation{ .25 } ),
        std::make_shared<implicit_type>( ops, imex_linear_equation{ -1.25 } )
    );
    auto                            pair = runge_kutta::make_imex_butcher_table( "IMEX_ARS233" );
    runge_kutta::imex_butcher_table custom(
        pair.explicit_table, pair.implicit_table, 3, 0,
        runge_kutta::imex_butcher_table::dense_output_type::endpoint_hermite
    );
    // Ordinary ARS233 does not consume the implicit initial rate; dense output does.
    auto             ordinary = make_imex_step( ops, assembly, .1, "IMEX_ARS233" );
    vector_wrap_type state( *ops ), out( *ops );
    state.start_use();
    out.start_use();
    ops->assign_scalar( 1., *state );
    ordinary->apply( *state, *out );
    require( ordinary->get_status() == single_step_status::converged, "Dense support became mandatory" );
    ordinary->finalize( adaptation_status::accepted, .1, *out );
    require_exception<std::invalid_argument>(
        [&] {
            dense_type invalid(
                ops, assembly, std::make_shared<adaptation_type>(), newton_types<implicit_type>::make_solver( ops ),
                custom
            );
        },
        "Dense output accepted a missing implicit rate provider"
    );
    require_exception<std::invalid_argument>(
        [&] {
            step_type invalid(
                ops, ::assembly( ops ), std::make_shared<adaptation_type>(), types::make_solver( ops ),
                step_type::params{ "IMEX_ARS233" }
            );
        },
        "Named method silently borrowed a partition extension"
    );
}

int main()
try
{
    backend_type::init_device();
    test_views_and_rates();
    test_endpoint_failures();
    test_construction_cleanup();
    test_optional_capabilities();
    require(
        storage_budget<true>() == storage_budget<false>() + 1, "Combined Hermite requires only one extra snapshot"
    );
    std::cout << "IMEX dense actual endpoint rates, views, bounded rejection, rollback and workspace reuse: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
