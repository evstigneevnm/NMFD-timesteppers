#ifndef NMFD_TIME_STEPPERS_BDF_CONTINUOUS_INTEGRATION_H
#define NMFD_TIME_STEPPERS_BDF_CONTINUOUS_INTEGRATION_H

#include <array>
#include <cmath>
#include <stdexcept>
#include <nmfd/time_steppers/runge_kutta/continuous_integration.h>

namespace nmfd::time_steppers::bdf
{
template <class T>
struct interpolation_coefficients
{
    T                endpoint;
    std::array<T, 4> differences{};
};

// p=in+L*(candidate-in)+sum((B_k(theta*r)-L*B_k(r))*D_k).
// D_k uses the previous signed step as its scale; no absolute times enter.
template <class T>
interpolation_coefficients<T> make_interpolation( unsigned int order, T theta, T h, const std::array<T, 5> &steps = {} )
{
    if ( order < 1 || order > 5 || !std::isfinite( theta ) || theta < 0 || theta > 1 || !std::isfinite( h ) || h == 0 )
        throw std::invalid_argument( "Invalid BDF interpolation order, coordinate or step" );
    interpolation_coefficients<T> result{ theta };
    if ( order == 1 )
        return result;
    std::array<T, 5> nodes{};
    for ( unsigned int i = 1; i < order; ++i )
    {
        if ( !std::isfinite( steps[i - 1] ) || steps[i - 1] == 0 || std::signbit( steps[i - 1] ) != std::signbit( h ) )
            throw std::invalid_argument( "Incomplete or reversed BDF interpolation history" );
        nodes[i] = nodes[i - 1] - steps[i - 1] / steps[0];
        if ( !std::isfinite( nodes[i] ) )
            throw std::invalid_argument( "Unrepresentable BDF interpolation grid" );
    }
    const T r = h / steps[0], z = theta * r;
    if ( !std::isfinite( r ) || r <= 0 )
        throw std::invalid_argument( "Unrepresentable BDF interpolation step ratio" );
    T weight = 1;
    for ( unsigned int i = 0; i < order; ++i )
        weight *= ( z - nodes[i] ) / ( r - nodes[i] );
    result.endpoint = weight;
    T at_sample = 1, at_endpoint = 1;
    for ( unsigned int i = 0; i + 1 < order; ++i )
    {
        at_sample *= z - nodes[i];
        at_endpoint *= r - nodes[i];
        result.differences[i] = at_sample - weight * at_endpoint;
    }
    return result;
}
}
#endif
