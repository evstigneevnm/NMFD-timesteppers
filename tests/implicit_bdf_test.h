#ifndef NMFD_TIME_STEPPERS_TESTS_IMPLICIT_BDF_TEST_H
#define NMFD_TIME_STEPPERS_TESTS_IMPLICIT_BDF_TEST_H

#include "implicit_test_support.h"
#include <nmfd/time_steppers/bdf/implicit_time_step.h>

namespace nmfd::time_steppers::tests::bdf_test
{
// u'=t-u, u(0)=1; exact u=t-1+2*exp(-t). Identity mass, smooth
// non-autonomous forcing and a stable linear mode isolate history and order.
struct forced_decay
{
    __DEVICE_TAG__ double mass( double ) const
    {
        return 1;
    }
    __DEVICE_TAG__ double rhs( double t, double u ) const
    {
        return t - u;
    }
    __DEVICE_TAG__ double rhs_derivative( double, double ) const
    {
        return -1;
    }
    static double exact( double t )
    {
        return t - 1 + 2 * std::exp( -t );
    }
};
using problem_type = stage_problem<scalar_time_equation<forced_decay>>;

template <class Problem, class Controller = adaptation_type>
using step_type =
    bdf::implicit_time_step<operations_type, Problem, Controller, typename newton_types<Problem>::solver_type>;

template <class Problem>
std::shared_ptr<step_type<Problem>> make_step(
    std::shared_ptr<operations_type> operations, std::shared_ptr<Problem> problem, double h, const char *method = "BDF2"
)
{
    return std::make_shared<step_type<Problem>>(
        operations, problem, std::make_shared<adaptation_type>( adaptation_type::params{ h } ),
        newton_types<Problem>::make_solver( operations ), typename step_type<Problem>::params{ method }
    );
}
}
#endif
