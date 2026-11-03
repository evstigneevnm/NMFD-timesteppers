#ifndef NMFD_TIME_STEPPERS_TESTS_IMPLICIT_DENSE_H
#define NMFD_TIME_STEPPERS_TESTS_IMPLICIT_DENSE_H

#include "implicit_rate_test.h"
#include "nonautonomous_test_problems.h"
#include <nmfd/time_steppers/integration/time_step_adaptation_matlab.h>

namespace nmfd
{
namespace time_steppers
{
namespace tests
{
inline const char *const dense_methods[]{ "IE", "IM", "CN", "SDIRK2(1)2", "ESDIRK2(1)3", "SDIRK3(1)3", "ARK324L2SA" };

using dense_growth = rate_scalar_equation<coefficient_growth<double>>;

// u'=1+2*t, u(0)=0, exact u=t+t^2. The quadratic trajectory makes
// endpoint agreement and section-time reconstruction independently checkable.
struct polynomial_growth
{
    __DEVICE_TAG__ double mass( double ) const
    {
        return 1;
    }
    __DEVICE_TAG__ double rhs( double t, double ) const
    {
        return 1 + 2 * t;
    }
    __DEVICE_TAG__ double rhs_derivative( double, double ) const
    {
        return 0;
    }
    static double exact( double t )
    {
        return t + t * t;
    }
};

template <class Problem, class Adaptation = adaptation_type>
using dense_step_type = runge_kutta::implicit_time_step<
    operations_type, Problem, Adaptation, typename newton_types<Problem>::solver_type, true>;

template <class Problem>
std::shared_ptr<dense_step_type<Problem>> make_dense_step(
    std::shared_ptr<operations_type> operations, std::shared_ptr<Problem> problem, double dt, const char *method
)
{
    using step_type = dense_step_type<Problem>;
    return std::make_shared<step_type>(
        operations, problem, std::make_shared<adaptation_type>( adaptation_type::params{ dt } ),
        newton_types<Problem>::make_solver( operations ), typename step_type::params{ method }
    );
}

template <class Exception, class Action>
void require_exception( Action action, const char *message )
{
    bool caught = false;
    try
    {
        action();
    }
    catch ( const Exception & )
    {
        caught = true;
    }
    require( caught, message );
}
}
}
}
#endif
