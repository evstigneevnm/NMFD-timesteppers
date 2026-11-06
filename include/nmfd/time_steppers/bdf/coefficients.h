#ifndef NMFD_TIME_STEPPERS_BDF_COEFFICIENTS_H
#define NMFD_TIME_STEPPERS_BDF_COEFFICIENTS_H

#include <cmath>
#include <array>
#include <stdexcept>

namespace nmfd::time_steppers::bdf
{
template <class T>
struct coefficients
{
    T alpha, history_weight;
};

// Write the BDF equation as alpha*M(u)*(u-w)-f=0, where
// w=u_n+history_weight*(u_n-u_{n-1}). h and previous_h are signed.
template <class T>
coefficients<T> make_coefficients( unsigned int order, T h, T previous_h = T( 0 ) )
{
    if ( !std::isfinite( h ) || h == 0 || ( order != 1 && order != 2 ) )
        throw std::invalid_argument( "Invalid BDF order or step" );
    if ( order == 1 )
        return { T( 1 ) / h, T( 0 ) };
    const T ratio = h / previous_h;
    if ( !std::isfinite( previous_h ) || !std::isfinite( ratio ) || ratio <= 0 )
        throw std::invalid_argument( "BDF2 requires same-direction history" );
    return {
        ( ( T( 1 ) + T( 2 ) * ratio ) / ( T( 1 ) + ratio ) ) / h, ratio * ( ratio / ( T( 1 ) + T( 2 ) * ratio ) )
    };
}

template <class T>
struct formula
{
    unsigned int     order;
    T                alpha, error_weight;
    std::array<T, 5> reference{}, predictor{};
};

// Differentiate the Newton interpolant at the new endpoint to obtain alpha*(u-w).
// Basis coordinates are scaled by the previous signed step.
template <class T>
formula<T> make_formula( unsigned int order, T h, const std::array<T, 5> &steps )
{
    if ( order < 1 || order > 5 || !std::isfinite( h ) || h == 0 )
        throw std::invalid_argument( "Invalid BDF order or step" );
    const T          scale = steps[0];
    std::array<T, 6> nodes{};
    for ( unsigned int i = 0; i < order; ++i )
    {
        if ( !std::isfinite( steps[i] ) || steps[i] == 0 || std::signbit( steps[i] ) != std::signbit( h ) )
            throw std::invalid_argument( "Incomplete or reversed BDF history" );
        nodes[i + 1] = nodes[i] - steps[i] / scale;
    }
    const T r     = h / scale;
    T       alpha = 0;
    for ( unsigned int i = 0; i < order; ++i )
        alpha += T( 1 ) / ( r - nodes[i] );
    formula<T> result{ order, alpha / scale, T( 1 ) / ( T( 1 ) + alpha * ( r - nodes[order] ) ) };
    T          basis = 1, derivative = 0;
    for ( unsigned int i = 0; i < order; ++i )
    {
        const T distance = r - nodes[i];
        derivative       = derivative * distance + basis;
        basis *= distance;
        result.predictor[i] = basis;
        if ( i + 1 < order )
            result.reference[i] = basis - derivative / alpha;
    }
    return result;
}
}
#endif
