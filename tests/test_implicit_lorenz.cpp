#include "implicit_lorenz_problem.h"

using namespace nmfd::time_steppers::tests;

// Damped Lorenz: x'=10*(y-x)-epsilon*x^3, y'=28*x-y-x*z, z'=x*y-(8/3)*z.
// Autonomous nonlinear ODE with identity mass; epsilon=0.0055 models spontaneous relaminarization.
// Setting epsilon=0 recovers the classical Lorenz equations.
// Initial state (2.2,30.5,2.5), t in [0,0.1]; reference from SciPy DOP853/Radau.
// Checks Jacobians, short-time trajectory accuracy, refinement, and restart, not sustained chaos.

// Reproduce the first rejected SDIRK2 stage from the strict [0,10] run. Its raw
// residual stalled at 3.9315e-10; balanced scaling removes the artificial 1/h floor.
void test_stalled_stage()
{
    auto             operations = std::make_shared<operations_type>( 3 );
    auto             problem    = std::make_shared<stage_problem<lorenz_equation>>( operations );
    auto             solver     = newton_types<stage_problem<lorenz_equation>>::make_solver( operations );
    vector_wrap_type state( *operations );
    state.start_use();
    const scfd::static_vec::vec<double, 3> previous{ 14.800392922564242, .021396443387958788, 46.30755920818253 };
    const scfd::static_vec::vec<double, 3> reference{ 14.799685394732823, .0200993166414588, 46.30696948402977 };
    operations->set( previous.d, problem->previous_state(), 3 );
    operations->assign_scalar( 0., problem->known_rhs() );
    operations->set( previous.d, *state, 3 );
    problem->set_time( .40390075177026186 );
    problem->set_affine_shift( 208890.52588143403 );
    solver->set_nonlinear_operator( problem );
    require( solver->solve( *state ), "Balanced Lorenz stage retained the raw-residual stagnation" );
    scfd::static_vec::vec<double, 3> actual;
    operations->get( *state, actual.d, 3 );
    for ( int i = 0; i < 3; ++i )
        require( std::abs( actual[i] - reference[i] ) < 2e-10, "Balanced Lorenz stage changed its root" );
    require( !solver->convergence_strategy()->used_relaxed_tolerance(), "Lorenz stage needed relaxed convergence" );
}

template <class Controller>
void test_zero_crossing()
{
    auto operations    = std::make_shared<operations_type>( 3 );
    using problem_type = stage_problem<lorenz_equation>;
    using types        = newton_types<problem_type>;
    using step_type    = nmfd::time_steppers::runge_kutta::implicit_time_step<
           operations_type, problem_type, Controller, typename types::solver_type>;
    auto problem   = std::make_shared<problem_type>( operations );
    auto p         = controller_parameters<Controller>( 1e-5 );
    p.initial_step = .001;
    p.maximum_step = .05;
    auto solver    = types::make_solver( operations );
    auto step      = std::make_shared<step_type>(
        operations, problem, std::make_shared<Controller>( operations, p ), solver,
        typename step_type::params{ "SDIRK2(1)2" }
    );
    nmfd::time_steppers::integration::time_integrator<operations_type, step_type> integrator( operations, step );
    vector_wrap_type initial( *operations ), result( *operations );
    initial.start_use();
    result.start_use();
    operations->assign_scalar( 1., *initial );
    integrator.set_time_interval( 0, .5 );
    integrator.apply( *initial, *result );
    require(
        integrator.get_status() == nmfd::time_steppers::integration_status::completed &&
            integrator.get_final_time() == .5,
        "Balanced SDIRK2 stopped at the Lorenz zero crossing"
    );
    // Independent SciPy DOP853/Radau reference, checked offline at tight tolerances.
    const scfd::static_vec::vec<double, 3> reference{ 1.198272968049551, -8.867197729737070, 32.454740211503726 };
    scfd::static_vec::vec<double, 3>       actual;
    operations->get( *result, actual.d, 3 );
    for ( int i = 0; i < 3; ++i )
        require( std::abs( actual[i] - reference[i] ) < 5e-4, "Lorenz zero-crossing trajectory reference mismatch" );
}
int main()
try
{
    backend_type::init_device();
    test_stalled_stage();
    test_zero_crossing<nmfd::time_steppers::integration::time_step_adaptation_matlab<operations_type>>();
    test_zero_crossing<nmfd::time_steppers::integration::time_step_adaptation_scipy<operations_type>>();
    lorenz_equation equation;
    equation.epsilon = .0055;
    // Same damped Lorenz reference used by the explicit regression (SciPy DOP853/Radau).
    test_ode( equation, { 2.2, 30.5, 2.5 }, { 21.527600219626638, 31.740750381438271, 45.939726836031042 }, .1, .0005 );
    // Classical Lorenz from (1,1,1), without relaminarization damping. A 60-digit Taylor
    // reference (orders 24/28, steps .001/.0005) agrees through 55 decimal places at t=.1.
    equation.epsilon = 0;
    test_adaptive_system(
        equation, { 1, 1, 1 }, { 2.133107618644515, 4.471420177185415, 1.113898885778636 }, 0, .1, "Classical Lorenz"
    );
    std::cout << "Implicit Lorenz, analytical Jacobian, reference refinement and restart: PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
