#ifndef NMFD_TIME_STEPPERS_TESTS_ODE_BENCHMARK_ROSENBROCK_H
#define NMFD_TIME_STEPPERS_TESTS_ODE_BENCHMARK_ROSENBROCK_H

#include "ode_benchmark_imex.h"
#include <nmfd/time_steppers/rosenbrock/time_step.h>

namespace nmfd::time_steppers::tests
{
template <class Equation>
inline constexpr bool benchmark_supports_rosenbrock =
    std::is_same_v<Equation, lorenz_equation> || std::is_same_v<Equation, rossler_equation> ||
    std::is_same_v<Equation, vdp_equation> ||
    std::is_same_v<Equation, scalar_time_equation<coefficient_growth<double>>> ||
    std::is_same_v<Equation, scalar_time_equation<stiff_tracking<double>>> || std::is_same_v<Equation, variable_mass> ||
    std::is_same_v<Equation, time_dependent_mass>;

template <class Equation>
inline constexpr bool benchmark_rosenbrock_autonomous =
    std::is_same_v<Equation, lorenz_equation> || std::is_same_v<Equation, rossler_equation> ||
    std::is_same_v<Equation, vdp_equation> || std::is_same_v<Equation, variable_mass>;

// Normalize only the two nonsingular variable-mass benchmark equations.
// Their physical mass remains in the ordinary DIRK/BDF problem unchanged.
template <class Equation>
struct benchmark_rosenbrock_rate_equation : Equation
{
    __DEVICE_TAG__ explicit benchmark_rosenbrock_rate_equation( Equation e = {} ) : Equation( e )
    {
    }
    __DEVICE_TAG__ double physical_rate( double t, double u ) const
    {
        if constexpr ( std::is_same_v<Equation, variable_mass> )
            return this->physical_rhs( u ) / this->mass( u );
        else
            return this->rhs( t, u ) / this->mass( t );
    }
    __DEVICE_TAG__ void residual(
        double t, const stage_scaling &s, const vector_type &in, const vector_type &previous, const vector_type &rhs,
        vector_type &out
    ) const
    {
        out( 0 ) = s.mass * ( in( 0 ) - previous( 0 ) ) - s.rhs * ( physical_rate( t, in( 0 ) ) + rhs( 0 ) );
    }
    auto jacobian(
        double t, const stage_scaling &s, const scfd::static_vec::vec<double, 1> &,
        const scfd::static_vec::vec<double, 1> &
    ) const
    {
        // For (1+u^2)*u'=-(1+u^2)*u the DM terms cancel exactly: Dq=-1.
        const double dq = std::is_same_v<Equation, variable_mass> ? -1 : -1 / ( 1 + t );
        return scfd::static_mat::mat<double, 1, 1>{ s.mass - s.rhs * dq };
    }
};
template <class Equation>
using benchmark_rosenbrock_equation = std::conditional_t<
    std::is_same_v<Equation, variable_mass> || std::is_same_v<Equation, time_dependent_mass>,
    benchmark_rosenbrock_rate_equation<Equation>, Equation>;

// Identity-mass equations reuse their original Jacobian; nonsingular variable
// mass uses the normalized rate facade. No canonical timestepper interface changes.
template <class Equation>
struct benchmark_rosenbrock_problem : stage_problem<benchmark_rosenbrock_equation<Equation>>
{
    using base_type = stage_problem<benchmark_rosenbrock_equation<Equation>>;
    explicit benchmark_rosenbrock_problem( std::shared_ptr<operations_type> ops, Equation equation = {} )
        : base_type( std::move( ops ), benchmark_rosenbrock_equation<Equation>( equation ) )
    {
    }
    struct mapping
    {
        Equation            equation;
        double              time;
        vector_type         state, out;
        bool                derivative;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t ) const
        {
            if ( derivative )
            {
                for ( int i = 0; i < Equation::size; ++i )
                    out( i ) = 0;
                if constexpr ( std::is_same_v<Equation, scalar_time_equation<coefficient_growth<double>>> )
                    out( 0 ) = state( 0 );
                if constexpr ( std::is_same_v<Equation, scalar_time_equation<stiff_tracking<double>>> )
                    out( 0 ) = equation.kappa * std::cos( time ) - std::sin( time );
                if constexpr ( std::is_same_v<Equation, time_dependent_mass> )
                    out( 0 ) = state( 0 ) / ( ( 1 + time ) * ( 1 + time ) );
            }
            else
            {
                if constexpr ( std::is_same_v<Equation, variable_mass> ||
                               std::is_same_v<Equation, time_dependent_mass> )
                    out( 0 ) =
                        benchmark_rosenbrock_rate_equation<Equation>( equation ).physical_rate( time, state( 0 ) );
                else
                {
                    typename benchmark_split<Equation>::state_type point;
                    for ( int i = 0; i < Equation::size; ++i )
                        point[i] = state( i );
                    const auto value = benchmark_split<Equation>::explicit_rhs( equation, time, point ) +
                                       benchmark_split<Equation>::implicit_rhs( equation, time, point );
                    for ( int i = 0; i < Equation::size; ++i )
                        out( i ) = value[i];
                }
            }
        }
    };
    bool evaluate_rate( const vector_type &in, vector_type &out ) const
    {
        execute_small_mapping( mapping{ this->equation, this->time, in, out, false } );
        return true;
    }
    bool evaluate_time_derivative( const vector_type &in, vector_type &out ) const
    {
        execute_small_mapping( mapping{ this->equation, this->time, in, out, true } );
        return true;
    }
};
}
#endif
