#ifndef NMFD_TIME_STEPPERS_TESTS_ODE_BENCHMARK_IMEX_H
#define NMFD_TIME_STEPPERS_TESTS_ODE_BENCHMARK_IMEX_H

#include "implicit_lorenz_problem.h"
#include "implicit_rossler_problem.h"
#include "implicit_vdp_problem.h"
#include "implicit_mass_dae_problems.h"
#include <nmfd/time_steppers/runge_kutta/imex_time_step.h>

namespace nmfd::time_steppers::tests
{
template <class Equation>
inline constexpr bool benchmark_supports_imex =
    !std::is_same_v<Equation, coupled_dae> && !std::is_same_v<Equation, moving_constraint_dae> &&
    !std::is_same_v<Equation, stokes_dae>;

template <int Size>
struct benchmark_split_types
{
    using state_type  = scfd::static_vec::vec<double, Size>;
    using matrix_type = scfd::static_mat::mat<double, Size, Size>;
};

template <class Equation>
struct benchmark_split;

// Lorenz: nonlinear coupling/forcing is explicit; the linear part is implicit.
template <>
struct benchmark_split<lorenz_equation> : benchmark_split_types<3>
{
    __DEVICE_TAG__ static state_type explicit_rhs( const lorenz_equation &e, double, const state_type &x )
    {
        return { -e.epsilon * x[0] * x[0] * x[0], -x[0] * x[2] + e.delta, x[0] * x[1] - e.delta };
    }
    __DEVICE_TAG__ static state_type implicit_rhs( const lorenz_equation &e, double, const state_type &x )
    {
        return { e.sigma * ( x[1] - x[0] ), e.rho * x[0] - x[1], -e.beta * x[2] };
    }
    static matrix_type explicit_jacobian( const lorenz_equation &e, double, const state_type &x )
    {
        return { -3 * e.epsilon * x[0] * x[0], 0, 0, -x[2], 0, -x[0], x[1], x[0], 0 };
    }
};

// Rossler: z*x is explicit; the remaining linear/affine terms are implicit.
template <>
struct benchmark_split<rossler_equation> : benchmark_split_types<3>
{
    __DEVICE_TAG__ static state_type explicit_rhs( const rossler_equation &, double, const state_type &x )
    {
        return { 0, 0, x[0] * x[2] };
    }
    __DEVICE_TAG__ static state_type implicit_rhs( const rossler_equation &e, double, const state_type &x )
    {
        return { -x[1] - x[2], x[0] + e.a * x[1], e.b - e.c * x[2] };
    }
    static matrix_type explicit_jacobian( const rossler_equation &, double, const state_type &x )
    {
        return { 0, 0, 0, 0, 0, 0, x[2], 0, x[0] };
    }
};

// Van der Pol: the oscillator is explicit; mu*(1-x^2)*v is implicit.
template <>
struct benchmark_split<vdp_equation> : benchmark_split_types<2>
{
    __DEVICE_TAG__ static state_type explicit_rhs( const vdp_equation &, double, const state_type &x )
    {
        return { x[1], -x[0] };
    }
    __DEVICE_TAG__ static state_type implicit_rhs( const vdp_equation &e, double, const state_type &x )
    {
        return { 0, e.mu * ( 1 - x[0] * x[0] ) * x[1] };
    }
    static matrix_type explicit_jacobian( const vdp_equation &, double, const state_type & )
    {
        return { 0, 1, -1, 0 };
    }
};

// Coupled mass: off-diagonal physical RHS terms are explicit, diagonal terms implicit.
template <>
struct benchmark_split<coupled_mass> : benchmark_split_types<2>
{
    __DEVICE_TAG__ static state_type explicit_rhs( const coupled_mass &, double, const state_type &x )
    {
        return { -2 * x[1], -x[0] };
    }
    __DEVICE_TAG__ static state_type implicit_rhs( const coupled_mass &, double, const state_type &x )
    {
        return { -2 * x[0], -6 * x[1] };
    }
    static matrix_type explicit_jacobian( const coupled_mass &, double, const state_type & )
    {
        return { 0, -2, -1, 0 };
    }
};

// State-dependent mass: split the physical decay RHS 1:3; retain DM in the Jacobian.
template <>
struct benchmark_split<variable_mass> : benchmark_split_types<1>
{
    __DEVICE_TAG__ static state_type explicit_rhs( const variable_mass &e, double, const state_type &x )
    {
        return { .25 * e.physical_rhs( x[0] ) };
    }
    __DEVICE_TAG__ static state_type implicit_rhs( const variable_mass &e, double, const state_type &x )
    {
        return { .75 * e.physical_rhs( x[0] ) };
    }
    static matrix_type explicit_jacobian( const variable_mass &, double, const state_type &x )
    {
        return { -.25 * ( 1 + 3 * x[0] * x[0] ) };
    }
};

// Scalar time-dependent problems use a 1:3 RHS split and their original mass.
template <class Equation>
struct benchmark_split<scalar_time_equation<Equation>> : benchmark_split_types<1>
{
    using equation_type = scalar_time_equation<Equation>;
    __DEVICE_TAG__ static state_type explicit_rhs( const equation_type &e, double t, const state_type &x )
    {
        return { .25 * e.rhs( t, x[0] ) };
    }
    __DEVICE_TAG__ static state_type implicit_rhs( const equation_type &e, double t, const state_type &x )
    {
        return { .75 * e.rhs( t, x[0] ) };
    }
    static matrix_type explicit_jacobian( const equation_type &e, double t, const state_type &x )
    {
        return { .25 * e.rhs_derivative( t, x[0] ) };
    }
};

// Stiff tracking: time-dependent forcing is explicit and -kappa*u is implicit.
template <>
struct benchmark_split<scalar_time_equation<stiff_tracking<double>>> : benchmark_split_types<1>
{
    using equation_type = scalar_time_equation<stiff_tracking<double>>;
    __DEVICE_TAG__ static state_type explicit_rhs( const equation_type &e, double t, const state_type & )
    {
        return { e.kappa * std::sin( t ) + std::cos( t ) };
    }
    __DEVICE_TAG__ static state_type implicit_rhs( const equation_type &e, double, const state_type &x )
    {
        return { -e.kappa * x[0] };
    }
    static matrix_type explicit_jacobian( const equation_type &, double, const state_type & )
    {
        return { 0 };
    }
};

// Small benchmark mass solves stay problem-owned and run inside the backend mapping.
template <class Equation>
__DEVICE_TAG__ auto benchmark_mass_solve(
    const Equation &e, double t, const typename benchmark_split<Equation>::state_type &x,
    typename benchmark_split<Equation>::state_type f
)
{
    if constexpr ( std::is_same_v<Equation, coupled_mass> )
        return typename benchmark_split<Equation>::state_type{ ( 3 * f[0] - f[1] ) / 5, ( 2 * f[1] - f[0] ) / 5 };
    else
    {
        if constexpr ( std::is_same_v<Equation, variable_mass> )
            f[0] /= e.mass( x[0] );
        else if constexpr ( Equation::size == 1 )
            f[0] /= e.mass( t );
        return f;
    }
}

template <class Equation, bool Implicit>
struct benchmark_rate_mapping
{
    Equation            equation;
    double              time;
    vector_type         state, out;
    __DEVICE_TAG__ void operator()( std::ptrdiff_t ) const
    {
        typename benchmark_split<Equation>::state_type x;
        for ( int j = 0; j < Equation::size; ++j )
            x[j] = state( j );
        const auto rhs  = Implicit ? benchmark_split<Equation>::implicit_rhs( equation, time, x )
                                   : benchmark_split<Equation>::explicit_rhs( equation, time, x );
        const auto rate = benchmark_mass_solve( equation, time, x, rhs );
        for ( int j = 0; j < Equation::size; ++j )
            out( j ) = rate[j];
    }
};

template <class Equation>
struct benchmark_implicit_equation : Equation
{
    explicit benchmark_implicit_equation( const Equation &equation ) : Equation( equation )
    {
    }
    __DEVICE_TAG__ void residual(
        double t, const stage_scaling &scaling, const vector_type &x, const vector_type &previous,
        const vector_type &rhs, vector_type &out
    ) const
    {
        Equation::residual( t, { scaling.mass, 0 }, x, previous, rhs, out );
        typename benchmark_split<Equation>::state_type point;
        for ( int j = 0; j < Equation::size; ++j )
            point[j] = x( j );
        const auto implicit_rhs = benchmark_split<Equation>::implicit_rhs( *this, t, point );
        for ( int j = 0; j < Equation::size; ++j )
            out( j ) -= scaling.rhs * ( implicit_rhs[j] + rhs( j ) );
    }
    auto jacobian(
        double t, const stage_scaling &scaling, const typename benchmark_split<Equation>::state_type &x,
        const typename benchmark_split<Equation>::state_type &previous
    ) const
    {
        auto       result = Equation::jacobian( t, scaling, x, previous );
        const auto e      = benchmark_split<Equation>::explicit_jacobian( *this, t, x );
        for ( int j = 0; j < Equation::size; ++j )
            for ( int k = 0; k < Equation::size; ++k )
                result( j, k ) += scaling.rhs * e( j, k );
        return result;
    }
};

template <class Equation>
struct benchmark_implicit_partition : stage_problem<benchmark_implicit_equation<Equation>>
{
    using base_type = stage_problem<benchmark_implicit_equation<Equation>>;
    benchmark_implicit_partition( std::shared_ptr<operations_type> ops, const Equation &equation )
        : base_type( std::move( ops ), benchmark_implicit_equation<Equation>( equation ) )
    {
    }
    bool evaluate_rate( const vector_type &state, vector_type &rate ) const
    {
        execute_small_mapping( benchmark_rate_mapping<Equation, true>{ this->equation, this->time, state, rate } );
        return true;
    }
};

template <class Equation>
struct benchmark_explicit_partition
{
    Equation equation;
    double   time = 0;
    void     set_time( double t )
    {
        time = t;
    }
    bool evaluate_rate( const vector_type &state, vector_type &rate ) const
    {
        execute_small_mapping( benchmark_rate_mapping<Equation, false>{ equation, time, state, rate } );
        return true;
    }
};

template <class Equation>
struct benchmark_imex_problem
{
    using imex_operator_type =
        runge_kutta::imex_operator<benchmark_explicit_partition<Equation>, benchmark_implicit_partition<Equation>>;
    std::shared_ptr<imex_operator_type> partitions;
    std::shared_ptr<imex_operator_type> get_imex_operator()
    {
        return partitions;
    }
};
}
#endif
