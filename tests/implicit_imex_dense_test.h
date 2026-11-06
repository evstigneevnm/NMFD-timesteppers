#ifndef NMFD_TIME_STEPPERS_TESTS_IMPLICIT_IMEX_DENSE_H
#define NMFD_TIME_STEPPERS_TESTS_IMPLICIT_IMEX_DENSE_H

#include "implicit_imex_adaptive_test.h"
#include "implicit_dense_test.h"

namespace nmfd::time_steppers::tests
{
template <
    class Problem, class Controller = adaptation_type,
    class Solver = typename newton_types<typename Problem::implicit_operator_type>::solver_type>
using imex_dense_step = runge_kutta::imex_time_step<operations_type, Problem, Controller, Solver, true>;

template <class Problem>
auto make_imex_dense_step( std::shared_ptr<operations_type> ops, std::shared_ptr<Problem> problem, double dt )
{
    using step_type = imex_dense_step<Problem>;
    return std::make_shared<step_type>(
        ops, problem, std::make_shared<adaptation_type>( adaptation_type::params{ dt } ),
        newton_types<typename Problem::implicit_operator_type>::make_solver( ops ),
        typename step_type::params{ "IMEX_ARK324L2SA" }
    );
}
}
#endif
