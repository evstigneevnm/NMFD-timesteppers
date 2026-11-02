#include "implicit_lorenz_problem.h"
#include "implicit_steady_state_test.h"
#include <nmfd/time_steppers/runge_kutta/explicit_time_step.h>

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;

// Classical Lorenz: x'=10*(y-x), y'=28*x-y-x*z, z'=x*y-(8/3)*z.
// epsilon=delta=0: no relaminarization damping. All three equilibria are unstable.
// The origin has a positive eigenvalue 11.828; the nonzero roots have the pair
// .09396+/-10.1945i. IE and the stiffly accurate SDIRK methods can damp these
// modes with sufficiently large pseudo-steps. Small-step trajectory integration
// need not converge, and no physical stability or arbitrary-seed guarantee follows.
void test_classical_lorenz()
{
    lorenz_equation equation;
    equation.epsilon = equation.delta = 0;
    const double magnitude            = std::sqrt( equation.beta * ( equation.rho - 1 ) );
    for ( const double sign : { 0., 1., -1. } )
    {
        const scfd::static_vec::vec<double, 3> equilibrium{
            sign * magnitude, sign * magnitude, sign == 0 ? 0 : equation.rho - 1
        };
        const double                           direction = sign < 0 ? -1 : 1;
        const scfd::static_vec::vec<double, 3> initial{
            equilibrium[0] + direction * .05, equilibrium[1] - direction * .025, equilibrium[2] + .075
        };
        test_unstable_equilibrium(
            equation, initial, equilibrium, 1,
            sign == 0  ? "Classical Lorenz origin"
            : sign > 0 ? "Classical Lorenz positive root"
                       : "Classical Lorenz negative root"
        );
    }
    test_unstable_equilibrium( equation, { 1, 1, 1 }, { 0, 0, 0 }, 1, "Classical Lorenz seed (1,1,1)" );
}

// Damped Lorenz with epsilon=.05, sigma=10, rho=28, beta=8/3, delta=0.
// The stable nonzero equilibria are (+/-x,+/-y,z), with s=x^2 solving
// (1+epsilon*s/sigma)*(1+s/beta)=rho, y=x*(1+epsilon*s/sigma), z=x*y/beta.
// Small perturbations test both attracting basins without a long chaotic search.
// The stored equilibrium was independently checked with SciPy DOP853 and Radau;
// error-controlled RK45 below supplies a separate physical-time comparison.
void test_damped_lorenz()
{
    using problem_type = stage_problem<lorenz_equation>;
    using types        = newton_types<problem_type>;
    using merit_type   = rhs_merit<problem_type>;
    using controller   = integration::time_step_adaptation_globalization<operations_type, merit_type>;
    using step_type    = runge_kutta::implicit_time_step<operations_type, problem_type, controller, types::solver_type>;
    auto            operations = std::make_shared<operations_type>( 3 );
    lorenz_equation equation;
    equation.epsilon         = .05;
    auto             problem = std::make_shared<problem_type>( operations, equation );
    auto             merit   = std::make_shared<merit_type>( operations, problem );
    vector_wrap_type initial( *operations ), result( *operations ), reference( *operations );
    initial.start_use();
    result.start_use();
    reference.start_use();
    const double equilibrium[] = { 7.4651576970186655, 9.545270865821479, 26.721357102793213 };

    using accuracy_control   = integration::time_step_adaptation_scipy<operations_type>;
    using physical_step_type = runge_kutta::explicit_time_step<operations_type, lorenz_equation, accuracy_control>;
    accuracy_control::params accuracy;
    accuracy.relative_tolerance = 1e-9;
    accuracy.absolute_tolerance = 1e-11;
    accuracy.initial_step       = .001;
    accuracy.maximum_step       = .1;
    auto physical_step          = std::make_shared<physical_step_type>(
        operations, std::make_shared<lorenz_equation>( equation ),
        std::make_shared<accuracy_control>( operations, accuracy )
    );

    for ( const double sign : { 1., -1. } )
    {
        const double start[] = {
            sign * ( equilibrium[0] + .5 ), sign * ( equilibrium[1] - .25 ), equilibrium[2] + .75
        };
        operations->set( start, *initial, 3 );
        check_jacobian( operations, *problem, { start[0], start[1], start[2] } );
        std::cout << "Damped Lorenz physical RK45, sign=" << sign << ": ";
        run_to_steady( operations, physical_step, merit, *initial, *reference, 1e-7 );
        for ( const char *method : { "IE", "SDIRK2(1)2", "SDIRK3(1)3" } )
        {
            auto control = std::make_shared<controller>( merit );
            auto step    = std::make_shared<step_type>(
                operations, problem, control, types::make_solver( operations ), step_type::params{ method }
            );
            std::cout << "Damped Lorenz pseudo-time " << method << ", sign=" << sign << ": ";
            run_to_steady( operations, step, merit, *initial, *result, 1e-7 );
            for ( int i = 0; i < 3; ++i )
            {
                const auto expected = ( i < 2 ? sign : 1 ) * equilibrium[i];
                const auto physical = operations->get_value_at_point( i, *reference );
                const auto pseudo   = operations->get_value_at_point( i, *result );
                require(
                    std::abs( physical - expected ) < 1e-6 && std::abs( pseudo - expected ) < 1e-6 &&
                        std::abs( pseudo - physical ) < 1e-6,
                    "Pseudo-time and ordinary damped Lorenz did not reach the known same-basin equilibrium"
                );
                require(
                    operations->get_value_at_point( i, *initial ) == start[i],
                    "Lorenz relaxation modified its initial state"
                );
            }
        }
    }
}

int main()
try
{
    backend_type::init_device();
    test_classical_lorenz();
    test_damped_lorenz();
    std::cout << "Classical unstable and damped stable Lorenz equilibria: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
