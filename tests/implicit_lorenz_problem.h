#ifndef NMFD_TIME_STEPPERS_TESTS_IMPLICIT_LORENZ_PROBLEM_H
#define NMFD_TIME_STEPPERS_TESTS_IMPLICIT_LORENZ_PROBLEM_H

#include "implicit_ode_test.h"
#include "explicit_lorenz_problem.h"

namespace nmfd
{
namespace time_steppers
{
namespace tests
{
// Lorenz with optional cubic damping: x'=sigma*(y-x)-epsilon*x^3,
// y'=rho*x-y-x*z+delta, z'=x*y-beta*z-delta. Identity mass, analytical stage Jacobian.
// epsilon=0 is classical Lorenz; positive epsilon models spontaneous relaminarization.
struct lorenz_equation : explicit_lorenz_problem<operations_type>
{
    static constexpr int size = 3;
    __DEVICE_TAG__ void  residual(
         double, const stage_scaling &scaling, const vector_type &x, const vector_type &previous, const vector_type &rhs,
         vector_type &out
     ) const
    {
        identity_mass_residual<3>( mapping{ x, out, sigma, rho, beta, epsilon, delta }, scaling, previous, rhs );
    }
    auto jacobian(
        double, const stage_scaling &scaling, const scfd::static_vec::vec<double, 3> &x,
        const scfd::static_vec::vec<double, 3> &
    ) const
    {
        return scfd::static_mat::mat<double, 3, 3>{
            scaling.mass + scaling.rhs * ( sigma + 3 * epsilon * x[0] * x[0] ),
            -scaling.rhs * sigma,
            0,
            scaling.rhs * ( x[2] - rho ),
            scaling.mass + scaling.rhs,
            scaling.rhs * x[0],
            -scaling.rhs * x[1],
            -scaling.rhs * x[0],
            scaling.mass + scaling.rhs * beta
        };
    }
};
}
}
}
#endif
