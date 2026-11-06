#include "implicit_bdf_dense_test.h"
#include <limits>

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;
namespace model = nmfd::time_steppers::tests::bdf_test;

template <class Step, class = void>
struct has_view : std::false_type
{
};
template <class Step>
struct has_view<Step, std::void_t<decltype( std::declval<const Step &>().get_continuous_integration() )>>
    : std::true_type
{
};
static_assert( !has_view<model::step_type<model::problem_type>>::value );
static_assert( has_view<model::dense_step<model::problem_type>>::value );

// u'=t-u: real Newton stages, including in-place publication, must retain the
// starting snapshot and never promote a rejected or externally changed endpoint.
template <class Controller>
void test_lifecycle( const char *method, std::shared_ptr<Controller> control, bool adaptive )
{
    using step_type                    = model::dense_step<model::problem_type, Controller>;
    auto                       ops     = std::make_shared<operations_type>( 1 );
    auto                       problem = std::make_shared<model::problem_type>( ops );
    typename step_type::params p{ method };
    p.adaptive = adaptive;
    step_type        step( ops, problem, control, newton_types<model::problem_type>::make_solver( ops ), p );
    vector_wrap_type state( *ops ), candidate( *ops ), sample( *ops );
    state.start_use();
    candidate.start_use();
    sample.start_use();
    model::throws<std::logic_error>( [&] { step.get_continuous_integration(); }, "View before BDF apply" );
    ops->assign_scalar( 1., *state );
    double time = 0;
    step.set_target_time( 10 );
    for ( int i = 0; i < 24; ++i )
    {
        step.set_time( time );
        step.apply( *state, *state );
        require( step.get_status() == single_step_status::converged, "Dense lifecycle warmup failed" );
        time += step.get_dt();
        step.finalize( adaptation_status::accepted, time, *state );
    }
    step.set_time( time );
    const auto initial = ops->get_value_at_point( 0, *state );
    step.apply( *state, *candidate );
    const auto order    = step.get_order();
    const auto old      = step.get_continuous_integration();
    const auto endpoint = old.evaluate( 1, *sample );
    const auto value    = ops->get_value_at_point( 0, *sample );
    old.evaluate( .37, *sample );
    const auto middle = ops->get_value_at_point( 0, *sample );
    old.evaluate( .37, *candidate );
    old.evaluate( 1, *sample );
    require( ops->get_value_at_point( 0, *sample ) == value, "Caller sampling corrupted the BDF candidate" );
    for ( double theta :
          { -.01, 1.01, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity() } )
    {
        ops->assign_scalar( 77., *sample );
        model::throws<std::invalid_argument>( [&] { old.evaluate( theta, *sample ); }, "Invalid dense coordinate" );
        require( ops->get_value_at_point( 0, *sample ) == 77, "Invalid request wrote output" );
    }
    model::throws<std::logic_error>( [&] { step.apply( *state, *candidate ); }, "Pending BDF view overwritten" );
    for ( auto outcome : { adaptation_status::failed, adaptation_status::rejected } )
    {
        auto view = step.get_continuous_integration();
        step.finalize( outcome, time, *state );
        model::throws<std::logic_error>( [&] { view.evaluate( .5, *sample ); }, "Failed commit retained view" );
        step.apply( *state, *candidate );
        model::throws<std::logic_error>( [&] { view.evaluate( .5, *sample ); }, "Stale BDF view revived" );
        require( step.get_order() == order, "Failed commit changed BDF order history" );
        step.get_continuous_integration().evaluate( .37, *sample );
        require( ops->get_value_at_point( 0, *sample ) == middle, "Failed commit changed dense history" );
    }
    auto current   = step.get_continuous_integration();
    auto shortened = current.evaluate( .4, *candidate );
    // Even finalize(accepted) must recognize a shortened time as modified.
    step.finalize( adaptation_status::accepted, shortened, *candidate );
    model::throws<std::logic_error>( [&] { current.evaluate( .5, *sample ); }, "Shortened commit retained view" );
    ops->assign( *candidate, *state );
    step.set_time( shortened );
    step.apply( *state, *candidate );
    require( step.get_order() == 1 && step.dense_output_order() == 1, "Shortened BDF retained high-order history" );
    current   = step.get_continuous_integration();
    auto full = current.evaluate( 1, *sample );
    // A state change at the full endpoint must also invalidate history.
    ops->assign_scalar( 2., *candidate );
    step.finalize( adaptation_status::accepted_modified, full, *candidate );
    ops->assign( *candidate, *state );
    step.set_time( full );
    step.apply( *state, *candidate );
    require( step.get_order() == 1, "Modified full endpoint retained BDF history" );
    current = step.get_continuous_integration();
    current.evaluate( 0, *sample );
    require( ops->get_value_at_point( 0, *sample ) == 2, "Restart has obsolete initial snapshot" );
    step.reset();
    model::throws<std::logic_error>( [&] { current.evaluate( .5, *sample ); }, "Reset retained BDF view" );
    step.apply( *state, *candidate );
    model::throws<std::logic_error>( [&] { current.evaluate( .5, *sample ); }, "Reset revived BDF view" );
    step.finalize( adaptation_status::failed, full, *state );
    require( initial != value && endpoint > time, "Lifecycle fixture did not advance" );
}

struct counted_operations : operations_type
{
    using operations_type::operations_type;
    mutable unsigned int allocations = 0, live = 0, fail_at = 0;
    void                 start_use_vector( vector_type &x ) const override
    {
        const bool fresh = x.is_free();
        operations_type::start_use_vector( x );
        if ( fresh )
        {
            ++live;
            if ( ++allocations == fail_at )
                throw std::runtime_error( "Injected dense BDF activation failure" );
        }
    }
    void free_vector( vector_type &x ) const override
    {
        if ( !x.is_free() )
            --live;
        operations_type::free_vector( x );
    }
};

template <bool Dense>
void test_storage()
{
    using step_type = bdf::implicit_time_step<
        counted_operations, model::problem_type, model::dense_controller,
        newton_types<model::problem_type>::solver_type, Dense>;
    auto ops = std::make_shared<counted_operations>( 1 );
    {
        auto             problem = std::make_shared<model::problem_type>( ops );
        auto             solver  = newton_types<model::problem_type>::make_solver( ops );
        vector_wrap_type state( *ops ), sample( *ops );
        state.start_use();
        sample.start_use();
        // Initialize the injected solver's lazy workspace before measuring only
        // timestepper buffers and injecting construction failures.
        {
            using warmup_type = model::dense_step<model::problem_type>;
            warmup_type warmup( ops, problem, std::make_shared<adaptation_type>(), solver, { "BDF1" } );
            ops->assign_scalar( 1., *state );
            warmup.apply( *state, *state );
            require( warmup.get_status() == single_step_status::converged, "Dense storage warmup failed" );
            warmup.finalize( adaptation_status::accepted, warmup.get_dt(), *state );
        }
        for ( const char *method : { "BDF1", "BDF2", "BDF" } )
            for ( bool adaptive : { false, true } )
            {
                if ( std::string( method ) == "BDF" && !adaptive )
                    continue;
                typename step_type::params p{ method };
                p.adaptive          = adaptive;
                const auto budget   = adaptive                          ? ( std::string( method ) == "BDF"    ? 9u
                                                                            : std::string( method ) == "BDF2" ? 5u
                                                                                                              : 4u )
                                      : std::string( method ) == "BDF2" ? 3u
                                                                        : 1u + unsigned( Dense );
                const auto baseline = ops->live;
                auto       control  = std::make_shared<model::dense_controller>( .02 );
                // A non-estimating controller is needed for fixed-mode storage.
                if ( !adaptive )
                {
                    using fixed_step = bdf::implicit_time_step<
                        counted_operations, model::problem_type, adaptation_type,
                        newton_types<model::problem_type>::solver_type, Dense>;
                    for ( unsigned int failure = 1; failure <= budget; ++failure )
                    {
                        ops->fail_at = ops->allocations + failure;
                        model::throws<std::runtime_error>(
                            [&] {
                                fixed_step invalid(
                                    ops, problem, std::make_shared<adaptation_type>(), solver, { method }
                                );
                            },
                            "Fixed dense BDF construction did not fail"
                        );
                        require( ops->live == baseline, "Partial fixed BDF construction leaked storage" );
                    }
                    ops->fail_at = 0;
                    fixed_step step( ops, problem, std::make_shared<adaptation_type>(), solver, { method } );
                    require( ops->live - baseline == budget, "Fixed BDF dense workspace budget" );
                    ops->assign_scalar( 1., *state );
                    step.apply( *state, *state );
                    require( step.get_status() == single_step_status::converged, "Counted fixed BDF failed" );
                    const auto allocated = ops->allocations;
                    if constexpr ( Dense )
                        for ( int i = 0; i <= 20; ++i )
                            step.get_continuous_integration().evaluate( i / 20., *sample );
                    require( ops->allocations == allocated, "BDF sampling allocated vectors" );
                    step.finalize( adaptation_status::accepted, step.get_dt(), *state );
                }
                else
                {
                    for ( unsigned int failure = 1; failure <= budget; ++failure )
                    {
                        ops->fail_at = ops->allocations + failure;
                        model::throws<std::runtime_error>(
                            [&] { step_type invalid( ops, problem, control, solver, p ); },
                            "Dense BDF construction did not fail"
                        );
                        require( ops->live == baseline, "Partial dense BDF construction leaked storage" );
                    }
                    ops->fail_at = 0;
                    step_type step( ops, problem, control, solver, p );
                    require( ops->live - baseline == budget, "Adaptive BDF dense workspace budget" );
                    ops->assign_scalar( 1., *state );
                    step.apply( *state, *state );
                    require( step.get_status() == single_step_status::converged, "Counted adaptive BDF failed" );
                    const auto allocated = ops->allocations;
                    if constexpr ( Dense )
                        for ( int i = 0; i <= 20; ++i )
                            step.get_continuous_integration().evaluate( i / 20., *sample );
                    require( ops->allocations == allocated, "BDF interpolation allocated backend storage" );
                    step.finalize( adaptation_status::accepted, step.get_dt(), *state );
                }
                require( ops->live == baseline, "Dense BDF workspace leaked" );
            }
    }
    require( ops->live == 0, "Dense BDF dependencies leaked storage" );
}

struct refusing_solver
{
    std::shared_ptr<operations_type>                                ops;
    std::shared_ptr<newton_types<model::problem_type>::solver_type> solver;
    bool                                                            fail = true;
    void set_nonlinear_operator( std::shared_ptr<model::problem_type> problem )
    {
        solver->set_nonlinear_operator( std::move( problem ) );
    }
    bool solve( vector_type &state )
    {
        if ( !fail )
            return solver->solve( state );
        ops->assign_scalar( 99., state );
        return false;
    }
};

void test_solver_failure()
{
    using step_type =
        bdf::implicit_time_step<operations_type, model::problem_type, adaptation_type, refusing_solver, true>;
    auto ops     = std::make_shared<operations_type>( 1 );
    auto problem = std::make_shared<model::problem_type>( ops );
    auto solver  = std::make_shared<refusing_solver>(
        refusing_solver{ ops, newton_types<model::problem_type>::make_solver( ops ) }
    );
    step_type        step( ops, problem, std::make_shared<adaptation_type>(), solver );
    vector_wrap_type state( *ops ), out( *ops );
    state.start_use();
    out.start_use();
    ops->assign_scalar( 1., *state );
    ops->assign_scalar( 77., *out );
    step.apply( *state, *out );
    require(
        step.get_status() == single_step_status::failed_solver && ops->get_value_at_point( 0, *out ) == 77,
        "Failed BDF solve published its workspace"
    );
    model::throws<std::logic_error>(
        [&] { step.get_continuous_integration(); }, "Failed solve published dense output"
    );
    solver->fail = false;
    step.apply( *state, *out );
    require( step.get_status() == single_step_status::converged, "BDF did not recover after solver failure" );
    const auto view = step.get_continuous_integration();
    view.evaluate( 0, *out );
    require( ops->get_value_at_point( 0, *out ) == 1, "Solver failure corrupted dense starting state" );
    step.finalize( adaptation_status::failed, 0, *state );
}

void test_rejection()
{
    auto ops        = std::make_shared<operations_type>( 1 );
    auto problem    = std::make_shared<model::problem_type>( ops );
    auto control    = std::make_shared<model::dense_controller>( .1 );
    using step_type = model::dense_step<model::problem_type, model::dense_controller>;
    step_type::params p{ "BDF" };
    p.adaptive = true;
    step_type        step( ops, problem, control, newton_types<model::problem_type>::make_solver( ops ), p );
    vector_wrap_type state( *ops ), out( *ops );
    state.start_use();
    out.start_use();
    ops->assign_scalar( 1., *state );
    control->reject = true;
    step.apply( *state, *out );
    require(
        step.get_status() == single_step_status::converged && step.get_attempts() == 2 && step.get_dt() == .05,
        "Dense BDF did not retry the rejected candidate"
    );
    const auto view = step.get_continuous_integration();
    view.evaluate( .5, *out );
    require(
        std::abs( ops->get_value_at_point( 0, *out ) - ( 1 + ( 1 + .05 * .05 ) / 1.05 ) / 2 ) < 2e-10,
        "Dense starter used a rejected candidate or auxiliary half steps"
    );
    step.finalize( adaptation_status::failed, 0, *state );
    ops->assign_scalar( std::numeric_limits<double>::infinity(), *state );
    step.apply( *state, *out );
    require( step.get_status() == single_step_status::failed_nonfinite, "Nonfinite BDF input not refused" );
    model::throws<std::logic_error>( [&] { step.get_continuous_integration(); }, "Failed attempt published view" );
    model::throws<std::logic_error>( [&] { view.evaluate( .5, *out ); }, "Failed attempt revived stale view" );
}

int main()
try
{
    backend_type::init_device();
    for ( const char *method : { "BDF1", "BDF2" } )
        test_lifecycle( method, std::make_shared<adaptation_type>( adaptation_type::params{ .02 } ), false );
    for ( const char *method : { "BDF1", "BDF2", "BDF" } )
        test_lifecycle( method, std::make_shared<model::dense_controller>( .02 ), true );
    test_storage<false>();
    test_storage<true>();
    test_rejection();
    test_solver_failure();
    std::cout << "BDF dense lifetime, rejection, shortened/modified commits and allocation budgets: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
