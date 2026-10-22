#include "implicit_ode_test.h"
#include "explicit_lorenz_problem.h"

using namespace nmfd::time_steppers::tests;

// Damped Lorenz: x'=10*(y-x)-epsilon*x^3, y'=28*x-y-x*z, z'=x*y-(8/3)*z.
// Autonomous nonlinear ODE with identity mass; epsilon=0.0055 models spontaneous relaminarization.
// Setting epsilon=0 recovers the classical Lorenz equations.
// Initial state (2.2,30.5,2.5), t in [0,0.1]; reference from SciPy DOP853/Radau.
// Checks Jacobians, short-time trajectory accuracy, refinement, and restart, not sustained chaos.
struct lorenz_equation : explicit_lorenz_problem<operations_type>
{
    static constexpr int size = 3;
    __DEVICE_TAG__ void  residual(
         double, double alpha, const vector_type &x, const vector_type &previous, const vector_type &rhs,
         vector_type &out
     ) const
    {
        identity_mass_residual<3>( mapping{ x, out, sigma, rho, beta, epsilon, delta }, alpha, previous, rhs );
    }
    auto jacobian(
        double, double alpha, const scfd::static_vec::vec<double, 3> &x, const scfd::static_vec::vec<double, 3> &
    ) const
    {
        return scfd::static_mat::mat<double, 3, 3>{ alpha + sigma + 3 * epsilon * x[0] * x[0],
                                                    -sigma,
                                                    0,
                                                    x[2] - rho,
                                                    alpha + 1,
                                                    x[0],
                                                    -x[1],
                                                    -x[0],
                                                    alpha + beta };
    }
};

int main()
try
{
    backend_type::init_device();
    lorenz_equation equation;
    equation.epsilon = .0055;
    // Same damped Lorenz reference used by the explicit regression (SciPy DOP853/Radau).
    test_ode( equation, { 2.2, 30.5, 2.5 }, { 21.527600219626638, 31.740750381438271, 45.939726836031042 }, .1, .0005 );
    std::cout << "Implicit Lorenz, analytical Jacobian, reference refinement and restart: PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
