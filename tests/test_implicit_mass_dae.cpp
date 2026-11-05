#include "implicit_mass_dae_problems.h"

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;

// Deliberately inconsistent points must retain their full constraint residual,
// even when the differential rows have been strongly scaled. Then check actual
// small/large and backward IE stages from consistent initial data.
template <class Equation>
void test_dae_scaling(
    std::shared_ptr<operations_type> operations, const scfd::static_vec::vec<double, Equation::size> &point
)
{
    using problem_type = stage_problem<Equation>;
    using monitor_type = constraint_monitor<Equation>;
    vector_wrap_type state( *operations ), result( *operations );
    state.start_use();
    result.start_use();
    for ( const double reference_time : { .25, 4. } )
    {
        problem_type problem( operations );
        operations->assign_scalar( .2, problem.previous_state() );
        operations->assign_scalar( .1, problem.known_rhs() );
        operations->set( point.d, *state, Equation::size );
        problem.set_time( .4 );
        for ( const double alpha : { -1e12, -1e-12, 0., 1e-12, 1e12 } )
        {
            set_test_shift( problem, alpha, reference_time );
            problem.apply( *state, *result );
            const auto actual = operations->get_value_at_point( Equation::size - 1, *result );
            require(
                std::abs( actual - ( monitor_type::constraint( point, .4 ) - .1 ) ) < 2e-15,
                "Balanced scaling hid an algebraic constraint or changed its known RHS"
            );
        }
    }
    auto       problem = std::make_shared<problem_type>( operations );
    const auto initial = exact_state<Equation>( .25 );
    operations->set( initial.d, *state, Equation::size );
    for ( const double h : { -1e-6, 1e-6, 10. } )
    {
        auto   step = make_step( operations, problem, std::abs( h ), "IE" );
        double end  = .25 + h;
        step->set_time( .25 );
        step->set_target_time( end );
        step->apply( *state, *result );
        require( step->get_status() == single_step_status::converged, "Balanced small/large-step DAE solve failed" );
        monitor_type monitor{ operations };
        monitor.set_time_interval( .25, end );
        monitor.apply( *state, *result );
        step->finalize( adaptation_status::accepted, end, *result );
    }
}

template <class Equation>
void test_mass( const scfd::static_vec::vec<double, Equation::size> &point, const char *label, double start = 0 )
{
    using problem_type = stage_problem<Equation>;
    using types        = newton_types<problem_type>;
    using monitor_type = constraint_monitor<Equation>;
    auto operations    = std::make_shared<operations_type>( Equation::size );
    auto problem       = std::make_shared<problem_type>( operations );
    for ( const double time : { 0., .4, .9 } )
        for ( const double alpha : { -1e6, -.01, 0., .01, 1e6 } )
            check_jacobian( operations, *problem, point, time, alpha );
    vector_wrap_type initial( *operations ), result( *operations );
    initial.start_use();
    result.start_use();
    const auto initial_values = exact_state<Equation>( start );
    const auto reference      = exact_state<Equation>( start + .4 );
    operations->set( initial_values.d, *initial, Equation::size );
    for ( const char *method : { "IE", "SDIRK2(1)2" } )
    {
        double errors[2]{};
        for ( int level = 0; level < 2; ++level )
        {
            auto step    = make_step( operations, problem, .04 / ( level + 1 ), method );
            auto monitor = std::make_shared<monitor_type>( monitor_type{ operations } );
            if constexpr ( std::is_same_v<Equation, coupled_dae> || std::is_same_v<Equation, stokes_dae> ||
                           std::is_same_v<Equation, moving_constraint_dae> )
            {
                integration::time_integrator<operations_type, typename types::step_type, monitor_type> integrator(
                    operations, step, {}, monitor
                );
                integrator.set_time_interval( start, start + .4 );
                for ( int run = 0; run < 2; ++run )
                {
                    integrator.apply( *initial, *result );
                    require(
                        integrator.get_status() == integration_status::completed && monitor->calls > 0,
                        "Coupled DAE integration or restart failed"
                    );
                }
            }
            else
            {
                integration::time_integrator<operations_type, typename types::step_type> integrator( operations, step );
                integrator.set_time_interval( start, start + .4 );
                for ( int run = 0; run < 2; ++run )
                {
                    integrator.apply( *initial, *result );
                    require(
                        integrator.get_status() == integration_status::completed, "Mass integration or restart failed"
                    );
                }
            }
            typename problem_type::state_type actual;
            operations->get( *result, actual.d, Equation::size );
            for ( int i = 0; i < Equation::size; ++i )
                errors[level] = std::max( errors[level], std::abs( actual[i] - reference[i] ) );
        }
        require( errors[1] < .6 * errors[0], "Mass/DAE refinement did not improve accuracy" );
    }
    if constexpr ( std::is_same_v<Equation, coupled_dae> || std::is_same_v<Equation, stokes_dae> ||
                   std::is_same_v<Equation, moving_constraint_dae> )
    {
        test_dae_scaling<Equation>( operations, point );
        test_adaptive_system(
            Equation{}, initial_values, reference, start, start + .4, label, operations,
            std::make_shared<monitor_type>( monitor_type{ operations } )
        );
    }
    else
        test_adaptive_system( Equation{}, initial_values, reference, start, start + .4, label, operations );
}

int main()
try
{
    backend_type::init_device();
    test_mass<coupled_mass>( { .7, .4 }, "Coupled mass" );
    test_mass<variable_mass>( { .7 }, "State-dependent mass" );
    test_mass<time_dependent_mass>( { .7 }, "Time-dependent mass" );
    test_mass<coupled_dae>( { .7, .4 }, "Nonlinear index-1 DAE" );
    test_mass<moving_constraint_dae>( { .7, .4 }, "Moving index-1 DAE" );
    test_mass<moving_constraint_dae>( { .7, .4 }, "Shifted moving index-1 DAE", .25 );
    test_mass<stokes_dae>( { .7, -.7, .2 }, "Stokes index-2 DAE" );
    std::cout << "DIRK coupled/state/time-dependent mass, moving constraints, index-1 and Stokes index-2 DAEs: PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
