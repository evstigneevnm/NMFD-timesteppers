#ifndef NMFD_TIME_STEPPERS_ROSENBROCK_COEFFICIENTS_H
#define NMFD_TIME_STEPPERS_ROSENBROCK_COEFFICIENTS_H

#include <cmath>
#include <stdexcept>
#include <string>

namespace nmfd::time_steppers::rosenbrock
{
// ROS2 in the increment formulation documented by KPP:
// https://kpp.readthedocs.io/en/stable/num_methods/rosenbrock-methods.html#ros-2
// A*k1=f(u), A*k2=f(u+a21*k1)+(c21/h)*k1, A=I/(gamma*h)-J.
struct coefficients
{
    static constexpr unsigned int stages = 2, order = 2, embedded_order = 1;
    const double                  gamma = 1 + 1 / std::sqrt( 2. );
    const double                  a21 = 1 / gamma, c21 = -2 / gamma;
    const double                  m1 = 3 / ( 2 * gamma ), m2 = 1 / ( 2 * gamma );
    const double                  e1 = 1 / ( 2 * gamma ), e2 = 1 / ( 2 * gamma );

    explicit coefficients( const std::string &method = "ROS2" )
    {
        if ( method != "ROS2" )
            throw std::invalid_argument( "Unknown Rosenbrock method: " + method );
    }
};
}
#endif
