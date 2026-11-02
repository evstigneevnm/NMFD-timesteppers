#include "implicit_rossler_problem.h"
#include "implicit_steady_state_test.h"

using namespace nmfd::time_steppers::tests;

// Rossler: x'=-y-z, y'=x+a*y, z'=b+z*(x-c), a=b=.2, c=5.7.
// Both equilibria (a*z,-z,z), a*z^2-c*z+b=0, are physically unstable.
// The small root has eigenvalues approximately .097+/- .995i and -5.687;
// the large root has a positive eigenvalue .193 and a weakly stable pair.
// Large implicit pseudo-steps damp unstable modes; this is a local root search,
// not physical-time integration of the chaotic attractor or global convergence.
int main()
try
{
    backend_type::init_device();
    rossler_equation equation;
    const auto       discriminant = std::sqrt( equation.c * equation.c - 4 * equation.a * equation.b );
    for ( const double sign : { -1., 1. } )
    {
        // Avoid cancellation in the small root of the quadratic.
        const auto                             z = sign < 0 ? 2 * equation.b / ( equation.c + discriminant )
                                                            : ( equation.c + discriminant ) / ( 2 * equation.a );
        const scfd::static_vec::vec<double, 3> equilibrium{ equation.a * z, -z, z };
        const scfd::static_vec::vec<double, 3> initial{
            equilibrium[0] + .01, equilibrium[1] - .02, equilibrium[2] + .015
        };
        test_unstable_equilibrium(
            equation, initial, equilibrium, 100, sign < 0 ? "Rossler small root" : "Rossler large root"
        );
    }
    std::cout << "Rossler unstable steady states: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
