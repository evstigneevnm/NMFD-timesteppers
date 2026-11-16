#ifndef NMFD_TESTS_IMPLICIT_BDF_PROJECTION_TEST_H
#define NMFD_TESTS_IMPLICIT_BDF_PROJECTION_TEST_H
#include "implicit_projection_problem.h"
#include <nmfd/time_steppers/bdf/implicit_time_step.h>

namespace projection_tests
{
// Exercise all automatic orders on a fixed admissible grid. Error estimates
// are required and checked for availability, but deliberately not used to
// reject: accuracy/tolerance control is tested independently with SciPy.
template <class Space>
struct bdf_accepting_control : fixed_control<Space>
{
    using base = fixed_control<Space>;
    using base::base;
    using vector_type = typename Space::vector_type;
    bool requires_error_estimate() const
    {
        return true;
    }
    adaptation_status assess(
        double, double, const vector_type &, const vector_type &, double &next, unsigned int exponent = 0,
        const vector_type *error = nullptr
    )
    {
        check( exponent >= 2 && exponent <= 6 && error, "BDF lost its projected error estimate" );
        next = this->get_dt();
        return adaptation_status::accepted;
    }
};
template <class Space, class Control = fixed_control<Space>, bool Dense = false>
using bdf_projected_step = bdf::implicit_time_step<
    Space, implicit_problem<Space>, Control, typename implicit_newton<Space>::solver, Dense, projector<Space>>;
template <class Space, class Control = fixed_control<Space>, bool Dense = false>
using bdf_reference_step =
    bdf::implicit_time_step<Space, implicit_problem<Space>, Control, typename implicit_newton<Space>::solver, Dense>;
}
#endif
