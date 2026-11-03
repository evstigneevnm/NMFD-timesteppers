#include "implicit_rate_test.h"
#include "implicit_mass_dae_problems.h"

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;

// Reuse the manufactured systems in implicit_mass_dae_problems.h:
// [[2,1],[1,3]]*u'=-M*diag(1,2)*u; (1+u^2)*u'=-(1+u^2)*u;
// x'=-x-z+sin(t)*x, 0=z-sin(t)*x. The last case needs the algebraic
// rate z'=cos(t)*x+sin(t)*x', not zero or an inverse of its singular mass.
// All runs start at a nonzero time with consistent data. DAE checks establish
// convergence and endpoint constraints, not universal classical algebraic order.
template <class Equation>
void test_mass( const char *label )
{
    using problem_type          = rate_problem<Equation>;
    using types                 = newton_types<problem_type>;
    auto             operations = std::make_shared<operations_type>( Equation::size );
    auto             problem    = std::make_shared<problem_type>( operations );
    vector_wrap_type initial( *operations ), out( *operations );
    initial.start_use();
    out.start_use();
    const auto start = exact_state<Equation>( .25 );
    const auto end   = exact_state<Equation>( .65 );
    operations->set( start.d, *initial, Equation::size );
    problem->set_time( .25 );
    require( problem->evaluate_rate( *initial, *out ), "Manufactured mass rate solve failed" );
    for ( int i = 0; i < Equation::size; ++i )
    {
        const double expected = std::is_same_v<Equation, moving_constraint_dae> && i == 1
                                    ? std::exp( -.25 ) * ( std::cos( .25 ) - std::sin( .25 ) )
                                    : -( i + 1 ) * start[i];
        require(
            std::abs( operations->get_value_at_point( i, *out ) - expected ) < 1e-13,
            "Mass provider returned raw RHS or an inconsistent algebraic rate"
        );
    }
    for ( const char *method : { "CN", "ESDIRK2(1)3", "ARK324L2SA" } )
    {
        double errors[3]{};
        for ( int level = 0; level < 3; ++level )
        {
            auto step    = make_step( operations, problem, .1 / ( 1 << level ), method );
            auto monitor = std::make_shared<constraint_monitor<Equation>>( constraint_monitor<Equation>{ operations } );
            // Only the DAE has algebraic constraints; ordinary ODEs use the
            // unmodified integrator without a synthetic constraint monitor.
            auto run = [&]( auto &integrator ) {
                integrator.set_time_interval( .25, .65 );
                double      first = 0;
                std::size_t steps = 0;
                for ( int repeat = 0; repeat < 2; ++repeat )
                {
                    integrator.apply( *initial, *out );
                    require(
                        integrator.get_status() == integration_status::completed && integrator.get_final_time() == .65,
                        "ESDIRK mass/DAE integration failed"
                    );
                    for ( int i = 0; i < Equation::size; ++i )
                    {
                        errors[level] =
                            std::max( errors[level], std::abs( operations->get_value_at_point( i, *out ) - end[i] ) );
                        require(
                            operations->get_value_at_point( i, *initial ) == start[i],
                            "Mass integration changed its input"
                        );
                    }
                    const auto actual = operations->get_value_at_point( 0, *out );
                    if ( repeat )
                        require(
                            integrator.get_steps() == steps && std::abs( actual - first ) < 1e-13,
                            "Mass/DAE restart retained rate history"
                        );
                    first = actual;
                    steps = integrator.get_steps();
                }
            };
            if constexpr ( std::is_same_v<Equation, moving_constraint_dae> )
            {
                integration::time_integrator<operations_type, typename types::step_type, constraint_monitor<Equation>>
                    integrator( operations, step, {}, monitor );
                run( integrator );
                require( monitor->calls > 0, "DAE endpoint constraints were not checked" );
            }
            else
            {
                integration::time_integrator<operations_type, typename types::step_type> integrator( operations, step );
                run( integrator );
            }
        }
        std::cout << label << ' ' << method << " errors=" << errors[0] << " -> " << errors[1] << " -> " << errors[2]
                  << '\n';
        require(
            errors[1] < .7 * errors[0] && errors[2] < .7 * errors[1] && errors[2] < 2e-4,
            "ESDIRK mass/DAE refinement or endpoint accuracy failed"
        );
        if constexpr ( !std::is_same_v<Equation, moving_constraint_dae> )
            require(
                std::abs( std::log2( errors[1] / errors[2] ) - runge_kutta::make_butcher_table( method ).order() ) < .4,
                "ESDIRK mass ODE order mismatch"
            );
        operations->set( end.d, *initial, Equation::size );
        auto step = make_step( operations, problem, .025, method );
        integration::time_integrator<operations_type, typename types::step_type> backward( operations, step );
        backward.set_time_interval( .65, .25 );
        backward.apply( *initial, *out );
        require( backward.get_status() == integration_status::completed, "Backward ESDIRK mass solve failed" );
        for ( int i = 0; i < Equation::size; ++i )
            require(
                std::abs( operations->get_value_at_point( i, *out ) - start[i] ) < 2e-4,
                "Backward mass/DAE endpoint mismatch"
            );
        operations->set( start.d, *initial, Equation::size );
    }
}

void test_unsupported_dae()
{
    // The existing index-2 saddle problem has no full consistent-rate provider.
    // Its valid implicit DIRK support must not silently become ESDIRK support.
    using problem_type = stage_problem<stokes_dae>;
    auto operations    = std::make_shared<operations_type>( 3 );
    auto problem       = std::make_shared<problem_type>( operations );
    for ( const char *method : { "CN", "ESDIRK2(1)3", "ARK324L2SA" } )
    {
        bool rejected = false;
        try
        {
            auto step = make_step( operations, problem, .1, method );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        require( rejected, "ESDIRK silently reconstructed an unsupported DAE rate" );
    }
}

int main()
try
{
    backend_type::init_device();
    test_mass<coupled_mass>( "Coupled mass" );
    test_mass<variable_mass>( "State-dependent mass" );
    test_mass<moving_constraint_dae>( "Moving index-1 constraint" );
    test_unsupported_dae();
    std::cout
        << "CN/ESDIRK coupled/state-dependent mass, consistent moving DAE rates and unsupported DAE refusal: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
