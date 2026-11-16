#ifndef NMFD_TESTS_IMPLICIT_ROSENBROCK_PROJECTION_PROBLEM_H
#define NMFD_TESTS_IMPLICIT_ROSENBROCK_PROJECTION_PROBLEM_H

#include "explicit_projection_problem.h"
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <nmfd/time_steppers/rosenbrock/time_step.h>
#include <scfd/utils/log_std.h>

namespace projection_tests
{
// In coordinates t=(x-y)/2, n=(x+y)/2, z, the physical system is
// t'=-t+z (or -t^2), n'=-n+1+time+t/2, z'=-2z. P removes n.
// The reduced equations have exact t=2*exp(-time)-exp(-2*time), or
// t=1/(1+time), and z=exp(-2*time). The raw Jacobian need not commute
// with P: it may map tangential perturbations into n, but not vice versa.
template <class Space>
struct rosenbrock_projection_problem
{
    using scalar_type = double;
    using vector_type = typename Space::vector_type;
    using X           = vector_type;
    struct jacobi_operator_type
    {
        std::shared_ptr<Space> operations;
        X                      frozen;
        double                 a = 0, b = 1, factor = 1, normal_feedback = 0;
        bool                   reduced = false, nonlinear = false, noncommuting = false;
        struct mapping
        {
            X                   in, out, frozen;
            double              a, b, factor, normal_feedback;
            bool                reduced, nonlinear, noncommuting;
            __DEVICE_TAG__ void operator()( std::ptrdiff_t block ) const
            {
                const auto   i = 3 * block;
                const double t = .5 * ( in(i)-in( i + 1 ) ), n = .5 * ( in( i ) + in( i + 1 ) ), z = in( i + 2 );
                const double base = .5 * ( frozen( i ) - frozen( i + 1 ) );
                const double bt   = noncommuting ? -1.4 * t + .2 * z : ( nonlinear ? -2 * base * t : -t + z ) * factor;
                const double bz   = noncommuting ? -.3 * t - 2.4 * z : -2 * factor * z;
                const double bn   = -factor * n + ( reduced ? 0 : .5 * factor * t );
                const double jt   = bt + normal_feedback * n;
                out( i )          = a * in(i)-b * ( jt + bn );
                out( i + 1 )      = a * in( i + 1 ) - b * ( -jt + bn );
                out( i + 2 )      = a * z - b * bz;
            }
        };
        void apply( const X &in, X &out ) const
        {
            map_blocks(
                *operations, mapping{ in, out, frozen, a, b, factor, normal_feedback, reduced, nonlinear, noncommuting }
            );
        }
    };
    std::shared_ptr<Space>                operations;
    buffer<Space>                         previous, rhs, frozen;
    std::shared_ptr<jacobi_operator_type> jacobian;
    double                                time = 0, a = 0, b = 1, factor = 1, normal_feedback = 0;
    bool        reduced = false, nonlinear = false, noncommuting = false, lagged = false, prepared = false;
    std::size_t rates = 0, derivatives = 0, linearizations = 0;
    explicit rosenbrock_projection_problem( std::shared_ptr<Space> ops, bool reference = false )
        : operations( std::move( ops ) ), previous( *operations ), rhs( *operations ), frozen( *operations ),
          jacobian( std::make_shared<jacobi_operator_type>() ), reduced( reference )
    {
        jacobian->operations = operations;
    }
    X &previous_state()
    {
        return *previous;
    }
    X &known_rhs()
    {
        return *rhs;
    }
    const X &previous_state() const
    {
        return *previous;
    }
    const X &known_rhs() const
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
    }
    struct mapping
    {
        X                   in, previous, rhs, out;
        double              time, a, b;
        bool                reduced, nonlinear, rate;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t block ) const
        {
            const auto   i = 3 * block;
            const double t = .5 * ( in(i)-in( i + 1 ) ), n = .5 * ( in( i ) + in( i + 1 ) ), z = in( i + 2 );
            const double ft   = nonlinear ? -t * t : -t + z;
            const double fn   = -n + ( reduced ? 0 : 1 + time + .5 * t );
            const double f[3] = { ft + fn, -ft + fn, -2 * z };
            for ( int j = 0; j < 3; ++j )
                out( i + j ) = rate ? f[j] : a * ( in( i + j ) - previous( i + j ) ) - b * ( f[j] + rhs( i + j ) );
        }
    };
    void apply( const X &in, X &out ) const
    {
        map_blocks( *operations, mapping{ in, *previous, *rhs, out, time, a, b, reduced, nonlinear, false } );
    }
    bool evaluate_rate( const X &in, X &out )
    {
        ++rates;
        map_blocks( *operations, mapping{ in, *previous, *rhs, out, time, a, b, reduced, nonlinear, true } );
        return true;
    }
    struct time_mapping
    {
        X                   out;
        bool                reduced;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t block ) const
        {
            out( 3 * block ) = out( 3 * block + 1 ) = reduced ? 0 : 1;
            out( 3 * block + 2 )                    = 0;
        }
    };
    bool evaluate_time_derivative( const X &, X &out )
    {
        ++derivatives;
        map_blocks( *operations, time_mapping{ out, reduced } );
        return true;
    }
    void set_linearization_point( const X &state )
    {
        ++linearizations;
        if ( !lagged || !prepared )
            operations->assign( state, *frozen );
        prepared                  = true;
        jacobian->frozen          = *frozen;
        jacobian->a               = a;
        jacobian->b               = b;
        jacobian->factor          = factor;
        jacobian->normal_feedback = normal_feedback;
        jacobian->reduced         = reduced;
        jacobian->nonlinear       = nonlinear;
        jacobian->noncommuting    = noncommuting;
    }
    std::shared_ptr<const jacobi_operator_type> get_jacobi_operator() const
    {
        return jacobian;
    }
};

template <class Space, class Problem = rosenbrock_projection_problem<Space>>
struct projection_linear_solver
{
    using log_type    = scfd::utils::log_std;
    using monitor     = nmfd::solvers::monitor_krylov<Space, log_type>;
    using solver_type = nmfd::solvers::gmres<Space, monitor, log_type, typename Problem::jacobi_operator_type>;
    using X           = typename Space::vector_type;
    std::shared_ptr<Space>       operations;
    std::shared_ptr<solver_type> solver;
    std::size_t                  calls = 0, bindings = 0, fail_on = 0, invalid_on = 0;
    explicit projection_linear_solver( std::shared_ptr<Space> ops ) : operations( std::move( ops ) )
    {
        typename solver_type::params p;
        p.basis_size            = 8;
        p.batch_size            = 1;
        p.monitor.abs_tol       = 2e-14;
        p.monitor.rel_tol       = 1e-13;
        p.monitor.max_iters_num = 40;
        solver                  = std::make_shared<solver_type>( operations, nullptr, p );
    }
    void set_operator( std::shared_ptr<const typename Problem::jacobi_operator_type> op )
    {
        ++bindings;
        solver->set_operator( std::move( op ) );
    }
    bool solve( const X &rhs, X &out )
    {
        ++calls;
        if ( calls == fail_on )
        {
            operations->assign_scalar( -99., out );
            return false;
        }
        if ( calls == invalid_on )
        {
            operations->assign_scalar( std::numeric_limits<double>::quiet_NaN(), out );
            return true;
        }
        return solver->solve( rhs, out );
    }
};
template <class Space>
void rosenbrock_exact( const Space &ops, typename Space::vector_type &out, double time, bool nonlinear )
{
    const double z = std::exp( -2 * time );
    map_blocks( ops, exact_mapping<Space>{ out, nonlinear ? 1 / ( 1 + time ) : 2 * std::exp( -time ) - z, z } );
}
template <class Space, class Control = fixed_control<Space>, bool Dense = false>
using rosenbrock_projected_step = rosenbrock::time_step<
    Space, rosenbrock_projection_problem<Space>, Control, projection_linear_solver<Space>, Dense, projector<Space>>;

// An error-recording fixed-grid controller isolates the embedded formula from
// adaptive mesh changes. Separate tests use the real SciPy controller.
template <class Space>
struct rosenbrock_recording_control : fixed_control<Space>
{
    using X = typename Space::vector_type;
    buffer<Space>          error;
    std::shared_ptr<Space> ops;
    std::size_t            assessments = 0;
    unsigned int           exponent    = 0;
    bool                   retry       = false;
    double                 h;
    rosenbrock_recording_control( std::shared_ptr<Space> operations, double step = .02 )
        : fixed_control<Space>( typename fixed_control<Space>::params{ step } ), error( *operations ),
          ops( std::move( operations ) ), h( step )
    {
    }
    bool requires_error_estimate() const
    {
        return true;
    }
    double get_dt() const
    {
        return h;
    }
    double initialize( double, const X &, double = -1 )
    {
        return h;
    }
    adaptation_status
    assess( double, double, const X &, const X &, double &next, unsigned int order = 0, const X *estimate = nullptr )
    {
        check( estimate && order >= 2, "Rosenbrock lost its projected embedded estimate" );
        ops->assign( *estimate, *error );
        exponent = order;
        ++assessments;
        next = h;
        return adaptation_status::accepted;
    }
    adaptation_status reject_step( double used )
    {
        if ( !retry )
            return adaptation_status::failed;
        h = std::abs( used ) * .5;
        return adaptation_status::rejected;
    }
};
}
#endif
