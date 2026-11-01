#include "implicit_ode_test.h"
#include "explicit_rossler_problem.h"

using namespace nmfd::time_steppers::tests;

// Rossler: x'=-y-z, y'=x+a*y, z'=b+z*(x-c), with a=b=0.2 and c=5.7.
// Autonomous nonlinear three-dimensional ODE with identity mass.
// Initial state (2,0,0), t in [0,1]; compared against the independent numerical reference.
// Checks Jacobians, refinement, and restart; this short trajectory does not certify chaos.
struct rossler_equation : explicit_rossler_problem<operations_type>
{
    static constexpr int size = 3;
    __DEVICE_TAG__ void  residual(
         double, const stage_scaling &scaling, const vector_type &x, const vector_type &previous, const vector_type &rhs,
         vector_type &out
     ) const
    {
        identity_mass_residual<3>( mapping{ x, out, a, b, c }, scaling, previous, rhs );
    }
    auto jacobian(
        double, const stage_scaling &scaling, const scfd::static_vec::vec<double, 3> &x,
        const scfd::static_vec::vec<double, 3> &
    ) const
    {
        return scfd::static_mat::mat<double, 3, 3>{
            scaling.mass,
            scaling.rhs,
            scaling.rhs,
            -scaling.rhs,
            scaling.mass - scaling.rhs * a,
            0,
            -scaling.rhs * x[2],
            0,
            scaling.mass + scaling.rhs * ( c - x[0] )
        };
    }
};

int main()
try
{
    backend_type::init_device();
    test_ode(
        rossler_equation{}, { 2, 0, 0 }, { .98415292572457735, 1.8475471904515657, .045002515579204969 }, 1, .01
    );
    test_adaptive_system(
        rossler_equation{}, { 2, 0, 0 }, { .98415292572457735, 1.8475471904515657, .045002515579204969 }, 0, 1,
        "Rossler"
    );
    std::cout << "Implicit Rossler, analytical Jacobian, reference refinement and restart: PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
