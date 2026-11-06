#include "implicit_bdf_test.h"
#include "implicit_mass_dae_problems.h"

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;
namespace model = nmfd::time_steppers::tests::bdf_test;

// Reuse the documented manufactured systems in implicit_mass_dae_problems.h:
// coupled SPD, state-dependent and time-dependent masses; nonlinear/moving
// index-1 constraints; and a constrained velocity-pressure index-2 system.
// Every problem is residual-only: BDF must not request a mass inverse or rate.
template <class Equation>
void test_system( const char *label, double start = .25 )
{
    using problem_type = stage_problem<Equation>;
    using step_type    = model::step_type<problem_type>;
    using monitor_type = constraint_monitor<Equation>;
    auto       ops     = std::make_shared<operations_type>( Equation::size );
    auto       problem = std::make_shared<problem_type>( ops );
    const auto initial = exact_state<Equation>( start );
    check_jacobian( ops, *problem, initial, start, 37.5 );
    vector_wrap_type state( *ops ), out( *ops );
    state.start_use();
    out.start_use();
    ops->set( initial.d, *state, Equation::size );
    for ( const char *method : { "BDF1", "BDF2" } )
        for ( const double direction : { 1., -1. } )
        {
            const auto end       = start + direction * .4;
            const auto reference = exact_state<Equation>( end );
            double     errors[2]{};
            for ( int level = 0; level < 2; ++level )
            {
                auto step      = model::make_step( ops, problem, .02 / ( level + 1 ), method );
                auto monitor   = std::make_shared<monitor_type>( monitor_type{ ops } );
                auto integrate = [&]( auto external ) {
                    using external_type = typename decltype( external )::element_type;
                    integration::time_integrator<operations_type, step_type, external_type> flow(
                        ops, step, {}, external
                    );
                    flow.set_time_interval( start, end );
                    for ( int run = 0; run < 2; ++run )
                    {
                        flow.apply( *state, *out );
                        require(
                            flow.get_status() == integration_status::completed && flow.get_final_time() == end,
                            "BDF mass/DAE trajectory or restart failed"
                        );
                    }
                };
                if constexpr ( std::is_same_v<Equation, coupled_dae> ||
                               std::is_same_v<Equation, moving_constraint_dae> || std::is_same_v<Equation, stokes_dae> )
                {
                    integrate( monitor );
                    require( monitor->calls > 0, "BDF did not check accepted constraints" );
                }
                else
                    integrate( std::shared_ptr<detail::no_external_operations>{} );
                typename problem_type::state_type actual;
                ops->get( *out, actual.d, Equation::size );
                for ( int i = 0; i < Equation::size; ++i )
                    errors[level] = std::max( errors[level], std::abs( actual[i] - reference[i] ) );
                require(
                    step->get_order() == ( std::string( method ) == "BDF2" ? 2 : 1 ),
                    "BDF dropped order on a roundoff-clipped final step"
                );
            }
            const double       observed = std::log2( errors[0] / errors[1] );
            const unsigned int order    = std::string( method ) == "BDF2" ? 2 : 1;
            std::cout << label << ' ' << method << " direction=" << direction << " order=" << observed << '\n';
            require( observed > order - .2 && observed < order + .2, "BDF mass/DAE analytical order mismatch" );
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
    std::cout << "BDF mass derivatives, unscaled constraints, forward/backward order and repeated trajectories: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
