#include "implicit_bdf_adaptive_test.h"
#include "implicit_mass_dae_problems.h"

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;
namespace model = nmfd::time_steppers::tests::bdf_test;

// Manufactured systems in implicit_mass_dae_problems.h have exact trajectories:
// coupled/state-dependent/time-dependent masses, nonlinear and moving index-1
// constraints, and a small velocity-pressure index-2 system. The residual-only
// problem deliberately offers no rate or mass inverse. Tightening the local
// state tolerance must improve the endpoint, without relaxing any constraint.
template <class Equation>
void test_system( const char *label )
{
    auto ops         = std::make_shared<operations_type>( Equation::size );
    using controller = integration::time_step_adaptation_scipy<operations_type>;
    for ( const char *method : { "BDF1", "BDF2", "BDF" } )
        for ( const double direction : { 1., -1. } )
        {
            const double start = .25, end = start + direction * .4;
            double       errors[2]{};
            for ( int level = 0; level < 2; ++level )
            {
                const auto tolerance = level ? 2e-6 : 2e-3;
                auto       run       = [&]( auto monitor ) {
                    return model::adaptive_trajectory<controller>(
                        Equation{}, exact_state<Equation>( start ), exact_state<Equation>( end ), start, end, method,
                        label, tolerance, ops, monitor
                    );
                };
                if constexpr ( std::is_same_v<Equation, coupled_dae> ||
                               std::is_same_v<Equation, moving_constraint_dae> || std::is_same_v<Equation, stokes_dae> )
                {
                    auto monitor =
                        std::make_shared<constraint_monitor<Equation>>( constraint_monitor<Equation>{ ops } );
                    errors[level] = run( monitor );
                    require( monitor->calls > 0, "Adaptive BDF skipped accepted constraints" );
                }
                else
                    errors[level] = run( std::shared_ptr<detail::no_external_operations>{} );
            }
            require( errors[1] < .2 * errors[0], "Adaptive BDF mass/DAE refinement failed" );
            require(
                errors[1] < ( std::string( method ) == "BDF1" ? .003 : .0005 ), "Adaptive BDF mass/DAE endpoint error"
            );
        }
}

template <class Equation>
void test_high_order_constraints(
    double start = .25, double tolerance = 1e-4, double initial_step = .01, double maximum_step = .01
)
{
    auto ops           = std::make_shared<operations_type>( Equation::size );
    using problem_type = stage_problem<Equation>;
    using control_type = model::observed_bdf_controller<integration::time_step_adaptation_scipy<operations_type>>;
    using step_type    = bdf::implicit_time_step<
           operations_type, problem_type, control_type, typename newton_types<problem_type>::solver_type>;
    auto cp         = controller_parameters<integration::time_step_adaptation_scipy<operations_type>>( tolerance );
    cp.initial_step = initial_step;
    cp.maximum_step = maximum_step;
    auto                       control = std::make_shared<control_type>( ops, cp );
    typename step_type::params p{ "BDF" };
    p.adaptive      = true;
    p.maximum_order = 5;
    auto step       = std::make_shared<step_type>(
        ops, std::make_shared<problem_type>( ops ), control, newton_types<problem_type>::make_solver( ops ), p
    );
    auto monitor = std::make_shared<constraint_monitor<Equation>>( constraint_monitor<Equation>{ ops } );
    integration::time_integrator<operations_type, step_type, constraint_monitor<Equation>> flow(
        ops, step, {}, monitor
    );
    flow.set_time_interval( start, start + .4 );
    vector_wrap_type state( *ops );
    state.start_use();
    const auto initial = exact_state<Equation>( start );
    for ( int run = 0; run < 2; ++run )
    {
        ops->set( initial.d, *state, Equation::size );
        flow.apply( *state, *state );
        require(
            flow.get_status() == integration_status::completed && ( control->committed_orders & ( 1u << 5 ) ),
            "Constrained BDF trajectory did not reach order five"
        );
        typename problem_type::state_type actual;
        ops->get( *state, actual.d, Equation::size );
        const auto expected = exact_state<Equation>( start + .4 );
        for ( int i = 0; i < Equation::size; ++i )
            require(
                std::abs( actual[i] - expected[i] ) < std::max( 1e-6, 10 * tolerance ),
                "High-order DAE cold-start endpoint error"
            );
    }
}

int main()
try
{
    backend_type::init_device();
    test_system<coupled_mass>( "Coupled SPD mass" );
    test_system<variable_mass>( "State-dependent mass" );
    test_system<time_dependent_mass>( "Time-dependent mass" );
    test_system<coupled_dae>( "Nonlinear index-1 DAE" );
    test_system<moving_constraint_dae>( "Moving index-1 DAE" );
    test_system<stokes_dae>( "Stokes index-2 DAE" );
    test_high_order_constraints<coupled_dae>();
    test_high_order_constraints<moving_constraint_dae>();
    test_high_order_constraints<stokes_dae>();
    // An extrapolated guess could pass Newton's residual test without correcting
    // pressure, then drive the controller to h_min. Keep the same tight settings.
    test_high_order_constraints<stokes_dae>( 0, 1e-7, .001, .05 );
    std::cout << "Adaptive BDF mass/DAE refinement, constraints, backward integration and restart: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
