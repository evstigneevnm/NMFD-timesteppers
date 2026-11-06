#include "implicit_bdf_dense_test.h"
#include "implicit_mass_dae_problems.h"

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;
namespace model = nmfd::time_steppers::tests::bdf_test;

// Residual-only systems: coupled SPD M, M(u)=1+u^2, M(t)=1+t;
// nonlinear/moving index-1 constraints and a velocity-pressure saddle system.
// Dense sampling must never require a mass inverse or evaluate_rate().
template <class Equation>
struct reconstruction_mapping
{
    double              time;
    vector_type         state;
    __DEVICE_TAG__ void operator()( std::ptrdiff_t ) const
    {
        if constexpr ( std::is_same_v<Equation, coupled_dae> )
            state( 1 ) = state( 0 ) * state( 0 );
        else if constexpr ( std::is_same_v<Equation, moving_constraint_dae> )
            state( 1 ) = std::sin( time ) * state( 0 );
        else if constexpr ( std::is_same_v<Equation, stokes_dae> )
        {
            state( 1 ) = -state( 0 );
            // Problem-owned pressure reconstruction, not polynomial pressure.
            state( 2 ) = std::sin( time );
        }
    }
};
template <class Equation>
void reconstruct( double time, vector_type &state )
{
    execute_small_mapping( reconstruction_mapping<Equation>{ time, state } );
}

template <class Equation>
void test_system()
{
    constexpr bool constrained = std::is_same_v<Equation, coupled_dae> ||
                                 std::is_same_v<Equation, moving_constraint_dae> ||
                                 std::is_same_v<Equation, stokes_dae>;
    using problem_type       = stage_problem<Equation>;
    auto             ops     = std::make_shared<operations_type>( Equation::size );
    auto             problem = std::make_shared<problem_type>( ops );
    vector_wrap_type state( *ops ), sample( *ops );
    state.start_use();
    sample.start_use();
    double raw_constraint = 0;
    for ( const char *method : { "BDF1", "BDF2", "BDF" } )
        for ( double direction : { -1., 1. } )
        {
            double errors[2]{};
            for ( int level = 0; level < 2; ++level )
            {
                using step_type                    = model::dense_step<problem_type, model::dense_controller>;
                const double               h       = .02 / ( level + 1 );
                auto                       control = std::make_shared<model::dense_controller>( h );
                typename step_type::params p{ method };
                p.adaptive = true;
                step_type  step( ops, problem, control, newton_types<problem_type>::make_solver( ops ), p );
                double     time    = .35;
                const auto initial = exact_state<Equation>( time );
                ops->set( initial.d, *state, Equation::size );
                step.set_target_time( time + direction );
                for ( int i = 0; i < 20 * ( level + 1 ); ++i )
                {
                    step.set_time( time );
                    step.apply( *state, *state );
                    require( step.get_status() == single_step_status::converged, "BDF dense mass/DAE solve failed" );
                    const auto view = step.get_continuous_integration();
                    for ( double theta : { .23, .67 } )
                    {
                        auto                              t = view.evaluate( theta, *sample );
                        typename problem_type::state_type values;
                        if constexpr ( constrained )
                        {
                            ops->get( *sample, values.d, Equation::size );
                            raw_constraint = std::max(
                                raw_constraint, std::abs( constraint_monitor<Equation>::constraint( values, t ) )
                            );
                            reconstruct<Equation>( t, *sample );
                        }
                        ops->get( *sample, values.d, Equation::size );
                        const auto reference = exact_state<Equation>( t );
                        if constexpr ( constrained )
                            require(
                                std::abs( constraint_monitor<Equation>::constraint( values, t ) ) < 1e-13,
                                "Reconstructed BDF sample violates constraints"
                            );
                        for ( int j = 0; j < Equation::size; ++j )
                            errors[level] = std::max( errors[level], std::abs( values[j] - reference[j] ) );
                    }
                    time = view.evaluate( 1, *sample );
                    if constexpr ( constrained )
                    {
                        typename problem_type::state_type values;
                        ops->get( *sample, values.d, Equation::size );
                        require(
                            std::abs( constraint_monitor<Equation>::constraint( values, time ) ) < 2e-9,
                            "Dense endpoint lost its solved constraint"
                        );
                    }
                    step.finalize( adaptation_status::accepted, time, *state );
                }
            }
            require( errors[1] < .75 * errors[0] && errors[1] < .02, "Dense mass/DAE refinement failed" );
        }
    if constexpr ( std::is_same_v<Equation, coupled_dae> || std::is_same_v<Equation, moving_constraint_dae> )
        require( raw_constraint > 1e-7, "DAE samples did not distinguish interpolation and reconstruction" );
}

using problem_type = stage_problem<moving_constraint_dae>;
using step_type    = model::dense_step<problem_type>;
struct constrained_manager
{
    std::shared_ptr<step_type> step;
    double                    *end    = nullptr;
    bool                       fail   = false;
    integration_status         status = integration_status::running;
    void                       set_time_interval( const double &, double &to )
    {
        end = &to;
    }
    bool apply( const vector_type &, vector_type &out )
    {
        *end = step->get_continuous_integration().evaluate( .4, out );
        reconstruct<moving_constraint_dae>( *end, out );
        status =
            fail ? integration_status::external_operation_failure : integration_status::stopped_by_external_operation;
        return true;
    }
    integration_status get_status() const
    {
        return status;
    }
};

void test_constrained_commit()
{
    auto ops     = std::make_shared<operations_type>( 2 );
    auto problem = std::make_shared<problem_type>( ops );
    auto step    = model::make_dense( ops, problem, .1, "BDF2" );
    auto manager = std::make_shared<constrained_manager>( constrained_manager{ step } );
    integration::time_integrator<operations_type, step_type, constrained_manager> flow( ops, step, {}, manager );
    vector_wrap_type                                                              initial( *ops ), out( *ops );
    initial.start_use();
    out.start_use();
    const auto exact = exact_state<moving_constraint_dae>( .35 );
    ops->set( exact.d, *initial, 2 );
    flow.set_time_interval( .35, .75 );
    for ( int run = 0; run < 2; ++run )
    {
        flow.apply( *initial, *out );
        require(
            flow.get_status() == integration_status::stopped_by_external_operation && flow.get_steps() == 1 &&
                std::abs( flow.get_final_time() - .39 ) < 1e-14,
            "BDF reconstructed event was not committed"
        );
        scfd::static_vec::vec<double, 2> values;
        ops->get( *out, values.d, 2 );
        require(
            std::abs( constraint_monitor<moving_constraint_dae>::constraint( values, .39 ) ) < 1e-14,
            "Committed BDF sample violates moving constraint"
        );
    }
    manager->fail = true;
    flow.apply( *initial, *out );
    require(
        flow.get_status() == integration_status::external_operation_failure && flow.get_steps() == 0 &&
            flow.get_final_time() == .35,
        "Failed reconstruction committed BDF event"
    );
    for ( int i = 0; i < 2; ++i )
        require( ops->get_value_at_point( i, *out ) == exact[i], "Reconstruction failure changed committed state" );
    model::throws<std::logic_error>(
        [&] { step->get_continuous_integration(); }, "Failed manager retained BDF dense view"
    );
}

int main()
try
{
    backend_type::init_device();
    test_system<coupled_mass>();
    test_system<variable_mass>();
    test_system<time_dependent_mass>();
    test_system<coupled_dae>();
    test_system<moving_constraint_dae>();
    test_system<stokes_dae>();
    test_constrained_commit();
    std::cout << "BDF dense mass/DAE samples, problem-owned reconstruction and rollback: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
