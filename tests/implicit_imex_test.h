#ifndef NMFD_TIME_STEPPERS_TESTS_IMPLICIT_IMEX_H
#define NMFD_TIME_STEPPERS_TESTS_IMPLICIT_IMEX_H

#include "implicit_rate_test.h"
#include <nmfd/time_steppers/runge_kutta/imex_time_step.h>

namespace nmfd::time_steppers::tests
{
inline constexpr const char *imex_methods[] = { "IMEX_EULER", "IMEX_HEUN_TR2", "IMEX_ARS222", "IMEX_ARS233" };

template <class Explicit, class Implicit>
struct imex_problem
{
    using implicit_operator_type = Implicit;
    using imex_operator_type     = runge_kutta::imex_operator<Explicit, Implicit>;
    std::shared_ptr<imex_operator_type> partitions;
    unsigned int                        requests = 0;

    imex_problem( std::shared_ptr<Explicit> e, std::shared_ptr<Implicit> i )
        : partitions( std::make_shared<imex_operator_type>( std::move( e ), std::move( i ) ) )
    {
    }
    std::shared_ptr<imex_operator_type> get_imex_operator()
    {
        ++requests;
        return partitions;
    }
};

template <
    class Problem, class Controller = adaptation_type,
    class Solver = typename newton_types<typename Problem::implicit_operator_type>::solver_type>
using imex_step = runge_kutta::imex_time_step<operations_type, Problem, Controller, Solver>;

template <class Problem>
auto make_imex_step(
    std::shared_ptr<operations_type> ops, std::shared_ptr<Problem> problem, double dt, const char *method
)
{
    using types = newton_types<typename Problem::implicit_operator_type>;
    return std::make_shared<imex_step<Problem>>(
        ops, problem, std::make_shared<adaptation_type>( adaptation_type::params{ dt } ), types::make_solver( ops ),
        typename imex_step<Problem>::params{ method }
    );
}

// Scalar partition f(u)=lambda*u; the combined exact flow is exp((lambda_E+lambda_I)*t).
struct imex_linear_equation
{
    static constexpr int size   = 1;
    double               lambda = -1;
    __DEVICE_TAG__ void  rate( double, const vector_type &x, vector_type &out ) const
    {
        out( 0 ) = lambda * x( 0 );
    }
    __DEVICE_TAG__ void residual(
        double, const stage_scaling &s, const vector_type &x, const vector_type &previous, const vector_type &rhs,
        vector_type &out
    ) const
    {
        out( 0 ) = s.mass * ( x( 0 ) - previous( 0 ) ) - s.rhs * ( lambda * x( 0 ) + rhs( 0 ) );
    }
    auto jacobian(
        double, const stage_scaling &s, const scfd::static_vec::vec<double, 1> &,
        const scfd::static_vec::vec<double, 1> &
    ) const
    {
        return scfd::static_mat::mat<double, 1, 1>{ s.mass - s.rhs * lambda };
    }
};
}
#endif
