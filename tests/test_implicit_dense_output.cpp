#include "implicit_dense_test.h"
#include <nmfd/time_steppers/integration/time_step_adaptation_scipy.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_filter.h>

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;

// u'=t*u, u(0)=1, exact u=exp(t^2/2). Non-autonomous identity-mass ODE.
// Check local interior order from exact initial data and global interior order
// across multiple steps, separately from endpoint accuracy and adaptive control.
struct timed_problem : rate_problem<dense_growth>
{
    using rate_problem<dense_growth>::rate_problem;
    unsigned int time_calls = 0;
    void         set_time( double time )
    {
        ++time_calls;
        rate_problem<dense_growth>::set_time( time );
    }
};

void test_orders()
{
    auto             operations = std::make_shared<operations_type>( 1 );
    vector_wrap_type initial( *operations ), out( *operations ), sample( *operations );
    initial.start_use();
    out.start_use();
    sample.start_use();
    for ( const auto *method : dense_methods )
        for ( const double direction : { 1., -1. } )
        {
            double     local[3]{}, global[3]{};
            const auto order = runge_kutta::make_butcher_table( method ).order();
            for ( int level = 0; level < 3; ++level )
            {
                auto       problem = std::make_shared<timed_problem>( operations );
                const auto h       = .16 / ( 1 << level );
                auto       step    = make_dense_step( operations, problem, h, method );
                operations->assign_scalar( dense_growth{}.exact( .35 ), *initial );
                step->set_time( .35 );
                step->set_target_time( .35 + direction * h );
                step->apply( *initial, *out );
                require( step->get_status() == single_step_status::converged, "Dense implicit step failed" );
                require( step->dense_output_order() == order, "Incorrect dense order report" );
                const auto view  = step->get_continuous_integration();
                const auto calls = problem->rate_calls, linearizations = problem->linearizations;
                const auto time_calls = problem->time_calls;
                const auto stage_time = problem->time,
                           previous   = operations->get_value_at_point( 0, problem->previous_state() );
                const auto matrix     = problem->get_jacobi_operator()->matrix;
                const auto extra      = step->table().dense_outout_stage_count() - step->table().size();
                require( calls == extra + ( step->table().a( 0, 0 ) == 0 ? 1 : 0 ), "Dense rate evaluation budget" );
                require(
                    time_calls == step->table().size() + 2 * extra, "Dense preparation performed redundant time updates"
                );
                require(
                    std::abs( stage_time - ( .35 + direction * h * step->table().c( step->table().size() - 1 ) ) ) <
                        1e-14,
                    "Dense preparation changed the prepared stage time"
                );
                for ( const double theta : { 0., .17, .43, .81, 1. } )
                {
                    const auto time = view.evaluate( theta, *sample );
                    require( time == .35 + theta * step->get_dt(), "Incorrect dense physical time" );
                    const auto value = operations->get_value_at_point( 0, *sample );
                    if ( theta == 0 )
                        require(
                            value == operations->get_value_at_point( 0, *initial ), "Initial dense endpoint mismatch"
                        );
                    else if ( theta == 1 )
                        require( value == operations->get_value_at_point( 0, *out ), "Final dense endpoint mismatch" );
                    else
                        local[level] = std::max( local[level], std::abs( value - dense_growth{}.exact( time ) ) );
                }
                require(
                    calls == problem->rate_calls && linearizations == problem->linearizations &&
                        time_calls == problem->time_calls && stage_time == problem->time &&
                        previous == operations->get_value_at_point( 0, problem->previous_state() ) &&
                        operations->get_value_at_point( 0, problem->known_rhs() ) == 0 &&
                        matrix( 0, 0 ) == problem->get_jacobi_operator()->matrix( 0, 0 ),
                    "Sampling modified the problem, stage buffers or prepared Jacobian"
                );
                step->finalize( adaptation_status::accepted, view.evaluate( 1, *sample ), *out );

                const auto end = .35 + direction * .4;
                const auto dt  = .1 / ( 1 << level );
                step           = make_dense_step( operations, problem, dt, method );
                operations->assign_scalar( dense_growth{}.exact( .35 ), *initial );
                for ( int n = 0; n < 4 * ( 1 << level ); ++n )
                {
                    step->set_time( .35 + direction * n * dt );
                    step->set_target_time( end );
                    step->apply( *initial, *out );
                    require( step->get_status() == single_step_status::converged, "Dense trajectory failed" );
                    const auto dense = step->get_continuous_integration();
                    const auto time  = dense.evaluate( .37, *sample );
                    global[level]    = std::max(
                        global[level],
                        std::abs( operations->get_value_at_point( 0, *sample ) - dense_growth{}.exact( time ) )
                    );
                    const auto endpoint = dense.evaluate( 1, *sample );
                    step->finalize( adaptation_status::accepted, endpoint, *out );
                    operations->assign( *out, *initial );
                }
            }
            std::cout << method << " direction=" << direction
                      << " dense local order=" << std::log2( local[1] / local[2] )
                      << ", global=" << std::log2( global[1] / global[2] ) << '\n';
            require(
                local[0] / local[1] > std::pow( 2., order + .4 ) && local[1] / local[2] > std::pow( 2., order + .4 ),
                "Implicit dense local order mismatch"
            );
            require( global[1] / global[2] > std::pow( 2., order - .3 ), "Implicit dense global order mismatch" );
        }
}

void test_quadratic_trajectory()
{
    using equation              = rate_scalar_equation<polynomial_growth>;
    auto             operations = std::make_shared<operations_type>( 1 );
    auto             problem    = std::make_shared<rate_problem<equation>>( operations );
    vector_wrap_type initial( *operations ), out( *operations ), sample( *operations );
    initial.start_use();
    out.start_use();
    sample.start_use();
    for ( const auto *method : dense_methods )
    {
        if ( std::string( method ) == "IE" )
            continue;
        auto step = make_dense_step( operations, problem, .2, method );
        operations->assign_scalar( polynomial_growth::exact( .4 ), *initial );
        step->set_time( .4 );
        step->set_target_time( .6 );
        step->apply( *initial, *out );
        require( step->get_status() == single_step_status::converged, "Polynomial dense step failed" );
        const auto dense = step->get_continuous_integration();
        for ( const auto theta : { .13, .37, .79 } )
        {
            const auto t = dense.evaluate( theta, *sample );
            require(
                std::abs( operations->get_value_at_point( 0, *sample ) - polynomial_growth::exact( t ) ) < 2e-13,
                "Quadratic trajectory is not reproduced by dense output"
            );
        }
        step->finalize( adaptation_status::accepted, .6, *out );
    }
}

struct trajectory_result
{
    double      endpoint;
    std::size_t steps, attempts;
};

template <class Controller, bool Dense>
trajectory_result adaptive_trajectory( const char *method )
{
    using problem_type = rate_problem<dense_growth>;
    using types        = newton_types<problem_type>;
    using step_type =
        runge_kutta::implicit_time_step<operations_type, problem_type, Controller, types::solver_type, Dense>;
    auto                        operations = std::make_shared<operations_type>( 1 );
    auto                        problem    = std::make_shared<problem_type>( operations );
    typename Controller::params p;
    p.initial_step = p.maximum_step = .4;
    p.relative_tolerance = p.absolute_tolerance = 1e-7;
    step_type step(
        operations, problem, std::make_shared<Controller>( operations, p ), types::make_solver( operations ),
        typename step_type::params{ method }
    );
    vector_wrap_type state( *operations ), out( *operations ), sample( *operations );
    state.start_use();
    out.start_use();
    sample.start_use();
    operations->assign_scalar( dense_growth{}.exact( .35 ), *state );
    trajectory_result result{};
    double            time = .35;
    while ( time != .75 )
    {
        require( result.steps < 10000, "Adaptive dense trajectory budget exhausted" );
        step.set_time( time );
        step.set_target_time( .75 );
        step.apply( *state, *out );
        require( step.get_status() == single_step_status::converged, "Adaptive dense step failed" );
        result.attempts += step.get_attempts();
        if constexpr ( Dense )
        {
            const auto t = step.get_continuous_integration().evaluate( .41, *sample );
            require(
                std::abs( operations->get_value_at_point( 0, *sample ) - dense_growth{}.exact( t ) ) < 2e-5,
                "Adaptive dense interior accuracy failed"
            );
        }
        time = step.get_dt() == .75 - time ? .75 : time + step.get_dt();
        step.finalize( adaptation_status::accepted, time, *out );
        operations->assign( *out, *state );
        ++result.steps;
    }
    result.endpoint = operations->get_value_at_point( 0, *state );
    return result;
}

template <class Controller>
void test_adaptation()
{
    for ( const auto *method : { "ESDIRK2(1)3", "ARK324L2SA" } )
    {
        const auto plain = adaptive_trajectory<Controller, false>( method );
        const auto dense = adaptive_trajectory<Controller, true>( method );
        require(
            plain.steps == dense.steps && plain.attempts == dense.attempts && plain.attempts > plain.steps &&
                std::abs( plain.endpoint - dense.endpoint ) < 1e-13,
            "Dense output changed error estimation, rejection or endpoint integration"
        );
    }
}

int main()
try
{
    backend_type::init_device();
    test_orders();
    test_quadratic_trajectory();
    test_adaptation<integration::time_step_adaptation_matlab<operations_type>>();
    test_adaptation<integration::time_step_adaptation_scipy<operations_type>>();
    test_adaptation<integration::time_step_adaptation_filter<operations_type>>();
    std::cout << "Implicit dense orders, endpoints, timed stages and unchanged adaptation: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
