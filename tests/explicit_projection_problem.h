#ifndef NMFD_TESTS_EXPLICIT_PROJECTION_PROBLEM_H
#define NMFD_TESTS_EXPLICIT_PROJECTION_PROBLEM_H

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#ifdef TEST_VECTOR_BACKEND_CUDA
#    include <scfd/backend/cuda.h>
using backend_type = scfd::backend::cuda<std::ptrdiff_t>;
#elif defined( TEST_VECTOR_BACKEND_HIP )
#    include <scfd/backend/hip.h>
using backend_type = scfd::backend::hip<std::ptrdiff_t>;
#elif defined( TEST_VECTOR_BACKEND_OMP )
#    include <scfd/backend/omp.h>
using backend_type = scfd::backend::omp<std::ptrdiff_t>;
#else
#    include <scfd/backend/serial_cpu.h>
using backend_type = scfd::backend::serial_cpu<std::ptrdiff_t>;
#endif
#include <nmfd/operations/scfd_vector_space.h>
#include <nmfd/time_steppers/runge_kutta/explicit_time_step.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_constant.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_scipy.h>
#include <nmfd/time_steppers/integration/time_integrator.h>

namespace projection_tests
{
using namespace nmfd::time_steppers;
inline std::size_t checks = 0;
inline void        check( bool condition, const char *message )
{
    ++checks;
    if ( !condition )
        throw std::runtime_error( message );
}

template <class Backend>
struct counted_space : nmfd::operations::scfd_vector_space<Backend, double>
{
    using base = nmfd::operations::scfd_vector_space<Backend, double>;
    using base::base;
    using typename base::vector_type;
    mutable std::size_t allocations = 0, validity_checks = 0;
    void                start_use_vector( vector_type &x ) const override
    {
        allocations += x.is_free();
        base::start_use_vector( x );
    }
    bool is_valid_number( const vector_type &x ) const override
    {
        ++validity_checks;
        return base::is_valid_number( x );
    }
};
using space_type = counted_space<backend_type>;
template <class Space>
using buffer = nmfd::detail::vector_wrap<Space, true, true>;
template <class Space>
using fixed_control = integration::time_step_adaptation_constant<Space>;
template <class Space>
using adaptive_control = integration::time_step_adaptation_scipy<Space>;

template <class Space, class Mapping>
void map_blocks( const Space &space, Mapping mapping )
{
    typename Space::for_each_type execute;
    execute( mapping, space.size() / 3 );
    execute.wait();
}

// Independent 3-component blocks constrained by x+y=0. The physical RHS is
// (-x+z+g,-y-z+g,-2*z+x+y), g=1+t+(x-y)/4. Removing the normal component
// (g,g,0) gives z'= -2*z and x'= -x+z on the plane. From (1,-1,1),
// z=exp(-2*t), x=2*exp(-t)-exp(-2*t), y=-x. Normal forcing changes later
// stages unless every rate, not just the final state, is projected.
template <class Space>
struct problem
{
    using X = typename Space::vector_type;
    std::shared_ptr<Space> operations;
    bool                   reference = false, record_times = false;
    double                 time  = 0;
    std::size_t            calls = 0, invalid_on = 0;
    std::vector<double>    times;
    explicit problem( std::shared_ptr<Space> space, bool reduced = false )
        : operations( std::move( space ) ), reference( reduced )
    {
    }
    void set_time( double value )
    {
        time = value;
    }
    struct mapping
    {
        X                   in, out;
        double              time;
        bool                reference;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t block ) const
        {
            const auto   i = 3 * block;
            const double x = in( i ), y = in( i + 1 ), z = in( i + 2 );
            const double g = reference ? 0 : 1 + time + .25 * ( x - y );
            out( i )       = -x + z + g;
            out( i + 1 )   = -y - z + g;
            out( i + 2 )   = -2 * z + x + y;
        }
    };
    bool evaluate_rate( const X &state, X &rate )
    {
        ++calls;
        if ( record_times )
            times.push_back( time );
        if ( calls == invalid_on )
            operations->assign_scalar( std::numeric_limits<double>::quiet_NaN(), rate );
        else
            map_blocks( *operations, mapping{ state, rate, time, reference } );
        return true;
    }
};

// Fixed linear orthogonal projector P(x,y,z)=((x-y)/2,(y-x)/2,z).
// A block is loaded before writing to avoid in-place device read/write races.
template <class Space>
struct projector
{
    using X = typename Space::vector_type;
    std::shared_ptr<Space> operations;
    std::size_t            calls = 0, fail_on = 0, invalid_on = 0, throw_on = 0;
    bool                   always_fail = false, enabled = true;
    explicit projector( std::shared_ptr<Space> space ) : operations( std::move( space ) )
    {
    }
    struct mapping
    {
        X                   rate;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t block ) const
        {
            const auto i       = 3 * block;
            const auto tangent = .5 * ( rate( i ) - rate( i + 1 ) );
            rate( i )          = tangent;
            rate( i + 1 )      = -tangent;
        }
    };
    bool apply( X &rate )
    {
        ++calls;
        if ( calls == throw_on )
            throw std::runtime_error( "Injected stage projection exception" );
        if ( always_fail || calls == fail_on )
        {
            operations->assign_scalar( -99., rate );
            return false;
        }
        if ( calls == invalid_on )
            operations->assign_scalar( std::numeric_limits<double>::quiet_NaN(), rate );
        else if ( enabled )
            map_blocks( *operations, mapping{ rate } );
        return true;
    }
};

template <class Space, class Control = fixed_control<Space>, bool Dense = false>
using projected_step = runge_kutta::explicit_time_step<Space, problem<Space>, Control, Dense, projector<Space>>;
template <class Space, class Control = fixed_control<Space>, bool Dense = false>
using reference_step = runge_kutta::explicit_time_step<Space, problem<Space>, Control, Dense>;

template <class Space>
struct exact_mapping
{
    typename Space::vector_type state;
    double                      x, z;
    __DEVICE_TAG__ void         operator()( std::ptrdiff_t block ) const
    {
        state( 3 * block )     = x;
        state( 3 * block + 1 ) = -x;
        state( 3 * block + 2 ) = z;
    }
};

template <class Space>
void exact( const Space &space, typename Space::vector_type &state, double time = 0 )
{
    const double z = std::exp( -2 * time );
    map_blocks( space, exact_mapping<Space>{ state, 2 * std::exp( -time ) - z, z } );
}

template <class Space>
double difference(
    const Space &space, const typename Space::vector_type &a, const typename Space::vector_type &b,
    typename Space::vector_type &scratch
)
{
    space.assign_lin_comb( 1., a, -1., b, scratch );
    return space.norm( scratch );
}

template <class Space>
struct constraint_mapping
{
    typename Space::vector_type in, out;
    __DEVICE_TAG__ void         operator()( std::ptrdiff_t block ) const
    {
        const double defect = in( 3 * block ) + in( 3 * block + 1 );
        for ( int j = 0; j < 3; ++j )
            out( 3 * block + j ) = defect;
    }
};

template <class Space>
double constraint( const Space &space, const typename Space::vector_type &state, typename Space::vector_type &scratch )
{
    map_blocks( space, constraint_mapping<Space>{ state, scratch } );
    return space.norm( scratch );
}
}
#endif
