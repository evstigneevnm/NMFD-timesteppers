#ifndef NMFD_TIME_STEPPERS_AFFINE_SHIFT_H
#define NMFD_TIME_STEPPERS_AFFINE_SHIFT_H

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nmfd::time_steppers::detail
{
template <class T>
struct affine_shift
{
    T a, b;
};

// Preserve a/b=alpha while bounding the differential residual coefficients.
template <class T>
affine_shift<T> balanced_affine_shift( T alpha, T reference_time = T( 1 ) )
{
    if ( !std::isfinite( alpha ) || !std::isfinite( reference_time ) || reference_time <= T( 0 ) )
        throw std::invalid_argument( "Stage shift must be finite and reference time positive and finite" );
    const T inverse_time = T( 1 ) / reference_time;
    if ( !std::isfinite( inverse_time ) || inverse_time <= T( 0 ) )
        throw std::invalid_argument( "Stage reference inverse time is not representable" );
    const T               scale = std::max( std::abs( alpha ), inverse_time );
    const affine_shift<T> result{ alpha / scale, T( 1 ) / scale };
    if ( !std::isfinite( result.b ) || result.b <= T( 0 ) || ( alpha != T( 0 ) && result.a == T( 0 ) ) )
        throw std::invalid_argument( "Stage coefficients are not representable" );
    return result;
}
}
#endif
