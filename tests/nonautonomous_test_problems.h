#ifndef NMFD_TIME_STEPPERS_TESTS_NONAUTONOMOUS_PROBLEMS_H
#define NMFD_TIME_STEPPERS_TESTS_NONAUTONOMOUS_PROBLEMS_H

#include <cmath>
#include <scfd/utils/device_tag.h>

namespace nmfd
{
namespace time_steppers
{
namespace tests
{
// u'=t*u, u(0)=1; exact u=exp(t^2/2). The RHS Jacobian changes with time.
template <class T>
struct coefficient_growth
{
    using scalar_type = T;
    __DEVICE_TAG__ T mass( T ) const
    {
        return T( 1 );
    }
    __DEVICE_TAG__ T rhs( T t, T u ) const
    {
        return t * u;
    }
    __DEVICE_TAG__ T rhs_derivative( T t, T ) const
    {
        return t;
    }
    __DEVICE_TAG__ T exact( T t ) const
    {
        return std::exp( t * t / T( 2 ) );
    }
};

// (1+t)*u'=-u, u(0)=1; exact u=1/(1+t). The mass is positive for t>-1.
// Unlike a common factor multiplying both sides, this mass changes the actual rate.
template <class T>
struct time_mass_decay
{
    using scalar_type = T;
    __DEVICE_TAG__ T mass( T t ) const
    {
        return T( 1 ) + t;
    }
    __DEVICE_TAG__ T rhs( T, T u ) const
    {
        return -u;
    }
    __DEVICE_TAG__ T rhs_derivative( T, T ) const
    {
        return T( -1 );
    }
    __DEVICE_TAG__ T exact( T t ) const
    {
        return T( 1 ) / ( T( 1 ) + t );
    }
};

// u'=-kappa*(u-sin(t))+cos(t), u(0)=0; exact u=sin(t).
// Perturbations decay at rate kappa. Large kappa introduces stiffness without changing the exact trajectory.
template <class T>
struct stiff_tracking
{
    using scalar_type      = T;
    T                kappa = T( 1 );
    __DEVICE_TAG__ T mass( T ) const
    {
        return T( 1 );
    }
    __DEVICE_TAG__ T rhs( T t, T u ) const
    {
        return -kappa * ( u - std::sin( t ) ) + std::cos( t );
    }
    __DEVICE_TAG__ T rhs_derivative( T, T ) const
    {
        return -kappa;
    }
    __DEVICE_TAG__ T exact( T t ) const
    {
        return std::sin( t );
    }
};
}
}
}

#endif
