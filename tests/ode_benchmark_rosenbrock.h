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
    std::is_same_v<Equation, scalar_time_equation<stiff_tracking<double>>>;

template <class Equation>
inline constexpr bool benchmark_rosenbrock_autonomous =
    std::is_same_v<Equation, lorenz_equation> || std::is_same_v<Equation, rossler_equation> ||
    std::is_same_v<Equation, vdp_equation>;

// Reuse each system's existing full RHS and analytical shifted Jacobian.
// This facade is restricted to the benchmark's identity-mass equations.
template <class Equation>
struct benchmark_rosenbrock_problem : stage_problem<Equation>
{
    using stage_problem<Equation>::stage_problem;
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
            }
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
