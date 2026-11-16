#ifndef NMFD_TESTS_IMPLICIT_IMEX_PROJECTION_PROBLEM_H
#define NMFD_TESTS_IMPLICIT_IMEX_PROJECTION_PROBLEM_H

#include "implicit_projection_problem.h"
#include <nmfd/time_steppers/runge_kutta/imex_time_step.h>

namespace projection_tests
{
// Independent blocks on x+y=0, with exact state (exp(-t),-exp(-t),exp(-2*t)).
// The implicit fraction is linear decay plus normal forcing. Explicit decay
// supplies the remaining fraction, a nonlinear tangential term vanishing on
// the exact trajectory, and normal forcing. Its dependence on x+y deliberately
// detects evaluation at an unprojected implicit stage. Only the implicit
// partition must satisfy P*f_I(v)=P*f_I(P*v).
template <class Space>
struct imex_implicit_partition
{
    using scalar_type = double;
    using vector_type = typename Space::vector_type;
    struct jacobi_operator_type
    {
        std::shared_ptr<Space> operations;
        double                 a = 0, b = 1, fraction = .6;
        struct mapping
        {
            vector_type         in, out;
            double              a, b, fraction;
            __DEVICE_TAG__ void operator()( std::ptrdiff_t block ) const
            {
                const auto i = 3 * block;
                out( i )     = ( a + b * fraction ) * in( i );
                out( i + 1 ) = ( a + b * fraction ) * in( i + 1 );
                out( i + 2 ) = ( a + 2 * b * fraction ) * in( i + 2 );
            }
        };
        void apply( const vector_type &in, vector_type &out ) const
        {
            map_blocks( *operations, mapping{ in, out, a, b, fraction } );
        }
    };
    std::shared_ptr<Space>                operations;
    buffer<Space>                         previous, rhs;
    std::shared_ptr<jacobi_operator_type> jacobian;
    double                                time = 0, a = 0, b = 1, fraction;
    bool                                  reduced;
    std::size_t                           rate_calls = 0, shift_calls = 0, invalid_on = 0, fail_on = 0;
    bool                                  record_times = false;
    std::vector<double>                   rate_times, stage_times;
    imex_implicit_partition( std::shared_ptr<Space> ops, bool reference = false, double weight = .6 )
        : operations( std::move( ops ) ), previous( *operations ), rhs( *operations ),
          jacobian( std::make_shared<jacobi_operator_type>() ), fraction( weight ), reduced( reference )
    {
        jacobian->operations = operations;
    }
    vector_type &previous_state()
    {
        return *previous;
    }
    const vector_type &previous_state() const
    {
        return *previous;
    }
    vector_type &known_rhs()
    {
        return *rhs;
    }
    const vector_type &known_rhs() const
    {
        return *rhs;
    }
    void set_time( double t )
    {
        time = t;
    }
    void set_affine_shift( double mass, double force )
    {
        a = mass;
        b = force;
        ++shift_calls;
        if ( record_times )
            stage_times.push_back( time );
    }
    struct mapping
    {
        vector_type         state, previous, rhs, out;
        double              a, b, time, fraction;
        bool                reduced, rate;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t block ) const
        {
            const auto   i    = 3 * block;
            const double g    = reduced ? 0 : 1 + time;
            const double f[3] = {
                fraction * ( -state( i ) + g ), fraction * ( -state( i + 1 ) + g ), -2 * fraction * state( i + 2 )
            };
            for ( int j = 0; j < 3; ++j )
                out( i + j ) = rate ? f[j] : a * ( state( i + j ) - previous( i + j ) ) - b * ( f[j] + rhs( i + j ) );
        }
    };
    void apply( const vector_type &state, vector_type &out ) const
    {
        map_blocks( *operations, mapping{ state, *previous, *rhs, out, a, b, time, fraction, reduced, false } );
    }
    bool evaluate_rate( const vector_type &state, vector_type &out )
    {
        ++rate_calls;
        if ( record_times )
            rate_times.push_back( time );
        if ( rate_calls == fail_on )
            return false;
        if ( rate_calls == invalid_on )
            operations->assign_scalar( std::numeric_limits<double>::quiet_NaN(), out );
        else
            map_blocks( *operations, mapping{ state, *previous, *rhs, out, a, b, time, fraction, reduced, true } );
        return true;
    }
    void set_linearization_point( const vector_type & )
    {
        jacobian->a        = a;
        jacobian->b        = b;
        jacobian->fraction = fraction;
    }
    std::shared_ptr<const jacobi_operator_type> get_jacobi_operator() const
    {
        return jacobian;
    }
};

template <class Space>
struct imex_explicit_partition
{
    using vector_type = typename Space::vector_type;
    std::shared_ptr<Space> operations;
    double                 time = 0, fraction;
    bool                   reduced, nonlinear;
    std::size_t            rate_calls = 0, invalid_on = 0, fail_on = 0;
    bool                   record_times = false;
    std::vector<double>    rate_times;
    imex_explicit_partition(
        std::shared_ptr<Space> ops, bool reference = false, double weight = .4, bool coupled = true
    )
        : operations( std::move( ops ) ), fraction( weight ), reduced( reference ), nonlinear( coupled )
    {
    }
    void set_time( double t )
    {
        time = t;
    }
    struct mapping
    {
        vector_type         state, out;
        double              time, fraction;
        bool                reduced, nonlinear;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t block ) const
        {
            const auto   i = 3 * block;
            const double x = state( i ), y = state( i + 1 ), z = state( i + 2 );
            const double g       = reduced ? 0 : 1 + time + .2 * z;
            const double tangent = nonlinear ? .2 * ( x - y ) * ( z - exp( -2 * time ) ) + .1 * ( x + y ) : 0;
            out( i )             = fraction * ( -x + tangent + g );
            out( i + 1 )         = fraction * ( -y - tangent + g );
            out( i + 2 )         = -2 * fraction * z;
        }
    };
    bool evaluate_rate( const vector_type &state, vector_type &out )
    {
        ++rate_calls;
        if ( record_times )
            rate_times.push_back( time );
        if ( rate_calls == fail_on )
            return false;
        if ( rate_calls == invalid_on )
            operations->assign_scalar( std::numeric_limits<double>::quiet_NaN(), out );
        else
            map_blocks( *operations, mapping{ state, out, time, fraction, reduced, nonlinear } );
        return true;
    }
};

template <class Space>
struct imex_projection_problem
{
    using implicit_type = imex_implicit_partition<Space>;
    using explicit_type = imex_explicit_partition<Space>;
    using bundle_type   = runge_kutta::imex_operator<explicit_type, implicit_type>;
    std::shared_ptr<bundle_type> bundle;
    std::size_t                  bindings = 0;
    imex_projection_problem(
        std::shared_ptr<Space> ops, bool reference = false, double implicit_fraction = .6, bool nonlinear = true
    )
        : bundle(
              std::make_shared<bundle_type>(
                  std::make_shared<explicit_type>( ops, reference, 1 - implicit_fraction, nonlinear ),
                  std::make_shared<implicit_type>( ops, reference, implicit_fraction )
              )
          )
    {
    }
    std::shared_ptr<bundle_type> get_imex_operator()
    {
        ++bindings;
        return bundle;
    }
};
template <class Space, class Control = fixed_control<Space>, bool Dense = false>
using imex_projected_step = runge_kutta::imex_time_step<
    Space, imex_projection_problem<Space>, Control,
    typename implicit_newton<Space, imex_implicit_partition<Space>>::solver, Dense, projector<Space>>;
template <class Space, class Control = fixed_control<Space>, bool Dense = false>
using imex_reference_step = runge_kutta::imex_time_step<
    Space, imex_projection_problem<Space>, Control,
    typename implicit_newton<Space, imex_implicit_partition<Space>>::solver, Dense>;
template <class Space>
void imex_exact( const Space &space, typename Space::vector_type &state, double time = 0 )
{
    map_blocks( space, exact_mapping<Space>{ state, std::exp( -time ), std::exp( -2 * time ) } );
}
}
#endif
