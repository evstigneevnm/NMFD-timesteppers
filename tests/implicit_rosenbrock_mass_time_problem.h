#ifndef NMFD_TIME_STEPPERS_TESTS_ROSENBROCK_MASS_TIME_PROBLEM_H
#define NMFD_TIME_STEPPERS_TESTS_ROSENBROCK_MASS_TIME_PROBLEM_H

#include "implicit_rosenbrock_mass_problem.h"
#include <array>

namespace rosenbrock_tests
{
// M*u'=M*q(t,u), q=L(t)*(u-g(t))+g'(t), g=(exp(t),exp(2*t)).
// L(t) is coupled, dissipative and optionally stiff. The exact solution is g.
// q_t=L_t*(u-g)-L*g'+g'' holds u fixed, unlike a total trajectory derivative.
struct manufactured_time_model
{
    double                time = 0, lambda = 1, factor = 1, offset = 0;
    bool                  varying = true;
    __DEVICE_TAG__ double entry( unsigned int row, unsigned int col, bool derivative = false ) const
    {
        const double slope = row ? ( col ? -.1 : .1 ) : ( col ? .2 : .3 );
        const double base  = row ? ( col ? -3 : -.25 ) : ( col ? .4 : -2 );
        return lambda * ( derivative ? ( varying ? slope : 0 ) : base + ( varying ? time * slope : 0 ) );
    }
    __DEVICE_TAG__ double approximate( unsigned int row, unsigned int col ) const
    {
        const double delta = row ? ( col ? -.7 : -.3 ) : ( col ? .5 : 1 );
        return factor * entry( row, col ) + offset * delta;
    }
    template <class X>
    __DEVICE_TAG__ double rate( std::ptrdiff_t i, const X &u ) const
    {
        const auto         j   = i - i % 2;
        const unsigned int row = i % 2;
        const double       g0 = exp( time ), g1 = exp( 2 * time );
        return entry( row, 0 ) * ( u( j ) - g0 ) + entry( row, 1 ) * ( u( j + 1 ) - g1 ) + ( row ? 2 * g1 : g0 );
    }
    template <class X>
    __DEVICE_TAG__ double time_derivative( std::ptrdiff_t i, const X &u ) const
    {
        const auto         j   = i - i % 2;
        const unsigned int row = i % 2;
        const double       g0 = exp( time ), g1 = exp( 2 * time );
        return entry( row, 0, true ) * ( u( j ) - g0 ) + entry( row, 1, true ) * ( u( j + 1 ) - g1 ) -
               entry( row, 0 ) * g0 - entry( row, 1 ) * 2 * g1 + ( row ? 4 * g1 : g0 );
    }
    template <class X>
    __DEVICE_TAG__ double jacobian( std::ptrdiff_t i, const X &, const X &v ) const
    {
        const auto j = i - i % 2;
        return approximate( i % 2, 0 ) * v( j ) + approximate( i % 2, 1 ) * v( j + 1 );
    }
    template <class X>
    __DEVICE_TAG__ double inverse( std::ptrdiff_t i, const X &, const X &v, double a, double b ) const
    {
        const auto   j  = i - i % 2;
        const double d0 = a - b * approximate( 0, 0 ), d1 = a - b * approximate( 1, 1 );
        const double e0 = -b * approximate( 0, 1 ), e1 = -b * approximate( 1, 0 );
        return ( i % 2 ? d0 * v( j + 1 ) - e1 * v( j ) : d1 * v( j ) - e0 * v( j + 1 ) ) / ( d0 * d1 - e0 * e1 );
    }
};

// M*u'=M*s*t*(1,2), u(0)=0; u(h)=s*h^2*(1,2)/2.
// Extreme h and s expose avoidable overflow/underflow in forming h^2*q_t.
struct polynomial_time_model
{
    double time = 0, slope = 1;
    template <class X>
    __DEVICE_TAG__ double rate( std::ptrdiff_t i, const X & ) const
    {
        return ( i % 2 + 1 ) * ( slope * time );
    }
    template <class X>
    __DEVICE_TAG__ double time_derivative( std::ptrdiff_t i, const X & ) const
    {
        return ( i % 2 + 1 ) * slope;
    }
    template <class X>
    __DEVICE_TAG__ double jacobian( std::ptrdiff_t, const X &, const X & ) const
    {
        return 0;
    }
    template <class X>
    __DEVICE_TAG__ double inverse( std::ptrdiff_t i, const X &, const X &v, double a, double ) const
    {
        return v( i ) / a;
    }
};

// The same physical model drives either the mass residual or the independent
// rate-form reference. Neither callback overwrites the prepared operator.
template <class Base>
struct time_problem_adapter : Base
{
    using Base::Base;
    using vector_type                   = typename Base::vector_type;
    using model_type                    = decltype( std::declval<Base>().model );
    double                prepared_time = 0, derivative_time = 0;
    unsigned int          derivatives = 0, fail_derivative_on = 0, invalid_derivative_on = 0;
    bool                  throw_derivative = false;
    std::array<double, 8> residual_times{};
    unsigned int          attempt_residuals = 0;
    void                  set_time( double t )
    {
        Base::set_time( t );
        this->model.time = t;
    }
    void set_linearization_point( const vector_type &state )
    {
        prepared_time     = this->time;
        attempt_residuals = 0;
        Base::set_linearization_point( state );
    }
    void apply( const vector_type &state, vector_type &out )
    {
        if ( attempt_residuals < residual_times.size() )
            residual_times[attempt_residuals] = this->time;
        ++attempt_residuals;
        Base::apply( state, out );
    }
    struct derivative_mapping
    {
        model_type          model;
        vector_type         state, out;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
        {
            out( i ) = model.time_derivative( i, state );
        }
    };
    bool evaluate_time_derivative( const vector_type &state, vector_type &out )
    {
        ++derivatives;
        derivative_time = this->time;
        if ( throw_derivative )
            throw std::runtime_error( "Injected mass rate-time derivative exception" );
        if ( derivatives == fail_derivative_on || derivatives == invalid_derivative_on )
        {
            this->ops->assign_scalar(
                derivatives == invalid_derivative_on ? std::numeric_limits<double>::infinity() : 99., out
            );
            return derivatives != fail_derivative_on;
        }
        using Operations = std::remove_reference_t<decltype( *this->ops )>;
        typename Operations::for_each_type{}( derivative_mapping{ this->model, state, out }, this->ops->size() );
        return true;
    }
};
template <class Operations = operations_type, class Model = manufactured_time_model>
using mass_time_problem = time_problem_adapter<mass_problem<Model, Operations>>;
template <class Operations = operations_type, class Model = manufactured_time_model>
using rate_time_problem = time_problem_adapter<problem<Model, Operations>>;
}
#endif
