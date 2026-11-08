#ifndef NMFD_TIME_STEPPERS_ROSENBROCK_CONTINUOUS_INTEGRATION_H
#define NMFD_TIME_STEPPERS_ROSENBROCK_CONTINUOUS_INTEGRATION_H

#include <array>
#include <cmath>
#include <stdexcept>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/time_steppers/rosenbrock/coefficients.h>
#include <nmfd/time_steppers/runge_kutta/continuous_integration.h>

namespace nmfd::time_steppers::rosenbrock
{
// P=(1-theta)*u0+theta*u1+theta*(1-theta)*sum((d_i+theta*f_i)*k_i).
// ROS3 adds k4 through A*k4=f(t1,u1)-sum(m_i*k_i)/h, using the base A.
// Coefficients satisfy the exact-Jacobian rooted-tree conditions through order
// three (two for ROS2); see the derivation in docs/rosenbrock_dense_output.md.
template <class T>
std::array<T, 4> continuous_weights( const method_table &table, T theta )
{
    if ( !std::isfinite( theta ) || theta < 0 || theta > 1 )
        throw std::invalid_argument( "Dense-output theta must be in [0,1]" );
    std::array<T, 4> weights{};
    const T          factor = theta * ( 1 - theta );
    if ( table.stage_count() == 2 )
        weights[0] = weights[1] = factor / T( 2 * table.gamma * ( 2 * table.gamma - 1 ) );
    else
    {
        constexpr std::array<double, 4> d = {
            7.7181786580594462837544743072342861555, -18.615520228222446988630640572344279710,
            .52993716110849007609025199845704232835, 3.4006232527153181413299843702258389904
        };
        constexpr std::array<double, 4> f = {
            -3.6845559252568873532147776293191732202, 7.1384959096922085460341466435534366346,
            -.058627234848502026925279121796584180136, -3.0976359654927675708384313769915114658
        };
        for ( std::size_t i = 0; i < weights.size(); ++i )
            weights[i] = factor * ( T( d[i] ) + theta * T( f[i] ) );
    }
    return weights;
}

}

namespace nmfd::time_steppers::detail
{
template <class Operations, bool Enabled>
struct rosenbrock_dense_storage
{
    explicit rosenbrock_dense_storage( const Operations & )
    {
    }
};

template <class Operations>
struct rosenbrock_dense_storage<Operations, true>
{
    nmfd::detail::vector_wrap<Operations, false, false> previous, correction;
    explicit rosenbrock_dense_storage( const Operations &ops ) : previous( ops ), correction( ops )
    {
    }
};
}
#endif
