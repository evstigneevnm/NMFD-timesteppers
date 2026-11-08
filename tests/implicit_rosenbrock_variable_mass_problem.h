#ifndef NMFD_TIME_STEPPERS_TESTS_ROSENBROCK_VARIABLE_MASS_PROBLEM_H
#define NMFD_TIME_STEPPERS_TESTS_ROSENBROCK_VARIABLE_MASS_PROBLEM_H

#include "implicit_rosenbrock_mass_time_problem.h"

namespace rosenbrock_tests
{
// (1+u^2)*u'=-u, initially u=1. Positive mass changes the trajectory:
// log(u)+u^2/2=1/2-t. Both f_u and D_u M contribute to the rate Jacobian.
struct state_mass_rate_model
{
    double                       time = 0, factor = 1;
    bool                         omit_mass_derivative = false;
    __DEVICE_TAG__ static double mass( double u )
    {
        return 1 + u * u;
    }
    template <class X>
    __DEVICE_TAG__ double rate( std::ptrdiff_t i, const X &u ) const
    {
        return -u( i ) / mass( u( i ) );
    }
    template <class X, class V>
    __DEVICE_TAG__ double jacobian( std::ptrdiff_t i, const X &u, const V &v ) const
    {
        const double dmq = omit_mass_derivative ? 0 : 2 * u( i ) * rate( i, u );
        return factor * ( -1 - dmq ) * v( i ) / mass( u( i ) );
    }
    template <class X>
    __DEVICE_TAG__ double inverse( std::ptrdiff_t i, const X &u, const X &v, double a, double b ) const
    {
        const double dmq = omit_mass_derivative ? 0 : 2 * u( i ) * rate( i, u );
        return v( i ) / ( a + b * factor * ( 1 + dmq ) / mass( u( i ) ) );
    }
    template <class X>
    __DEVICE_TAG__ double time_derivative( std::ptrdiff_t, const X & ) const
    {
        return 0;
    }
};

// (1+t)*u'=-u, t>-1; u(t)=C/(1+t). f_t=0 but q_t=u/(1+t)^2
// is nonzero. The callback is a partial derivative, holding u fixed.
struct time_mass_rate_model
{
    double time = 0, factor = 1;
    bool   omit_mass_derivative = false;
    template <class X>
    __DEVICE_TAG__ double rate( std::ptrdiff_t i, const X &u ) const
    {
        return -u( i ) / ( 1 + time );
    }
    template <class X>
    __DEVICE_TAG__ double time_derivative( std::ptrdiff_t i, const X &u ) const
    {
        return omit_mass_derivative ? 0 : u( i ) / ( ( 1 + time ) * ( 1 + time ) );
    }
    template <class X, class V>
    __DEVICE_TAG__ double jacobian( std::ptrdiff_t i, const X &, const V &v ) const
    {
        return -factor * v( i ) / ( 1 + time );
    }
    template <class X>
    __DEVICE_TAG__ double inverse( std::ptrdiff_t i, const X &, const X &v, double a, double b ) const
    {
        return v( i ) / ( a + b * factor / ( 1 + time ) );
    }
};

// Coupled time/state-dependent nonsymmetric M*u'=f=M*q, where q and its exact
// exp(t),exp(2*t) trajectory are defined by manufactured_time_model. Each rate
// solves a 2x2 mass block. Differentiate f and M separately, then solve
// M*Dq[v]=Df[v]-DM[v]*q and M*q_t=f_t-M_t*q. No vector leaves its backend.
struct coupled_variable_mass_model : manufactured_time_model
{
    bool omit_mass_derivative = false;
    struct pair_view
    {
        double                x, y;
        __DEVICE_TAG__ double operator()( std::ptrdiff_t i ) const
        {
            return i % 2 ? y : x;
        }
    };
    template <class X>
    __DEVICE_TAG__ mass_matrix mass( std::ptrdiff_t j, const X &u ) const
    {
        return {
            2 + .1 * time + .2 * u( j ) * u( j ), .3 + .1 * sin( time ), -.2 + .05 * u( j + 1 ),
            1.5 + .1 * time * time + .3 * u( j + 1 ) * u( j + 1 )
        };
    }
    __DEVICE_TAG__ static double solve_mass( mass_matrix m, unsigned int row, double f0, double f1 )
    {
        return ( row ? m.m00 * f1 - m.m10 * f0 : m.m11 * f0 - m.m01 * f1 ) / ( m.m00 * m.m11 - m.m01 * m.m10 );
    }
    template <class X>
    __DEVICE_TAG__ double rate( std::ptrdiff_t i, const X &u ) const
    {
        const auto   j  = i - i % 2;
        const auto   m  = mass( j, u );
        const double q0 = manufactured_time_model::rate( j, u ), q1 = manufactured_time_model::rate( j + 1, u );
        return solve_mass( m, i % 2, m.multiply( 0, q0, q1 ), m.multiply( 1, q0, q1 ) );
    }
    template <class X, class V>
    __DEVICE_TAG__ double jacobian( std::ptrdiff_t i, const X &u, const V &v ) const
    {
        const auto        j  = i - i % 2;
        const auto        m  = mass( j, u );
        const double      q0 = rate( j, u ), q1 = rate( j + 1, u );
        const mass_matrix dm{ .4 * u( j ) * v( j ), 0, .05 * v( j + 1 ), .6 * u( j + 1 ) * v( j + 1 ) };
        const double      d0 = entry( 0, 0 ) * v( j ) + entry( 0, 1 ) * v( j + 1 );
        const double      d1 = entry( 1, 0 ) * v( j ) + entry( 1, 1 ) * v( j + 1 );
        double            f0 = dm.multiply( 0, q0, q1 ) + m.multiply( 0, d0, d1 );
        double            f1 = dm.multiply( 1, q0, q1 ) + m.multiply( 1, d0, d1 );
        if ( !omit_mass_derivative )
        {
            f0 -= dm.multiply( 0, q0, q1 );
            f1 -= dm.multiply( 1, q0, q1 );
        }
        const double delta = ( i % 2 ? -.3 : 1 ) * v( j ) + ( i % 2 ? -.7 : .5 ) * v( j + 1 );
        return factor * solve_mass( m, i % 2, f0, f1 ) + offset * delta;
    }
    template <class X>
    __DEVICE_TAG__ double time_derivative( std::ptrdiff_t i, const X &u ) const
    {
        const auto        j = i - i % 2;
        const auto        m = mass( j, u );
        const mass_matrix mt{ .1, .1 * cos( time ), 0, .2 * time };
        const double      q0 = rate( j, u ), q1 = rate( j + 1, u );
        const double      t0 = manufactured_time_model::time_derivative( j, u );
        const double      t1 = manufactured_time_model::time_derivative( j + 1, u );
        double            f0 = mt.multiply( 0, q0, q1 ) + m.multiply( 0, t0, t1 );
        double            f1 = mt.multiply( 1, q0, q1 ) + m.multiply( 1, t0, t1 );
        if ( !omit_mass_derivative )
        {
            f0 -= mt.multiply( 0, q0, q1 );
            f1 -= mt.multiply( 1, q0, q1 );
        }
        return solve_mass( m, i % 2, f0, f1 );
    }
    template <class X>
    __DEVICE_TAG__ double inverse( std::ptrdiff_t i, const X &u, const X &v, double a, double b ) const
    {
        const auto   j  = i - i % 2;
        const double d0 = a - b * jacobian( j, u, pair_view{ 1, 0 } );
        const double d1 = a - b * jacobian( j + 1, u, pair_view{ 0, 1 } );
        const double e0 = -b * jacobian( j, u, pair_view{ 0, 1 } );
        const double e1 = -b * jacobian( j + 1, u, pair_view{ 1, 0 } );
        return ( i % 2 ? d0 * v( j + 1 ) - e1 * v( j ) : d1 * v( j ) - e0 * v( j + 1 ) ) / ( d0 * d1 - e0 * e1 );
    }
};

template <class Model, class Operations = operations_type>
using variable_mass_problem = rate_time_problem<Operations, Model>;

template <class Model>
inline constexpr bool variable_mass_autonomous = std::is_same_v<Model, state_mass_rate_model>;

template <class Model, class Vector>
struct variable_mass_initial
{
    double              time;
    Vector              out;
    __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
    {
        out( i ) = std::is_same_v<Model, coupled_variable_mass_model> ? exp( time * ( i % 2 + 1 ) ) : 1;
    }
};

template <class Model>
double variable_mass_exact( unsigned int i, double time, double start )
{
    if constexpr ( std::is_same_v<Model, coupled_variable_mass_model> )
        return std::exp( time * ( i % 2 + 1 ) );
    else if constexpr ( std::is_same_v<Model, time_mass_rate_model> )
        return ( 1 + start ) / ( 1 + time );
    else
    {
        // Monotone implicit exact solution, independent of the stepping method.
        long double lo = 0, hi = 4, target = .5L - ( time - start );
        for ( unsigned int n = 0; n < 100; ++n )
        {
            const auto u = ( lo + hi ) / 2;
            if ( std::log( u ) + u * u / 2 < target )
                lo = u;
            else
                hi = u;
        }
        return static_cast<double>( ( lo + hi ) / 2 );
    }
}
}
#endif
