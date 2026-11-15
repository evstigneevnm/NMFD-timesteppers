#ifndef NMFD_TESTS_IMPLICIT_PROJECTION_PROBLEM_H
#define NMFD_TESTS_IMPLICIT_PROJECTION_PROBLEM_H

#include "explicit_projection_problem.h"
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/solvers/nonlinear_solver.h>
#include <nmfd/time_steppers/runge_kutta/implicit_time_step.h>
#include <scfd/utils/log_std.h>

namespace projection_tests
{
// On x+y=0 the solution is the same as the explicit projection model.
// Here z'=-2*z, so P*f(v)=P*f(P*v): projecting an implicit stage is
// equivalent to solving the projected equation. Setting coupling=1 instead
// makes z' depend on the discarded normal component, a counterexample.
template <class Space>
struct implicit_problem
{
    using scalar_type = double;
    using vector_type = typename Space::vector_type;
    using X           = vector_type;
    struct jacobi_operator_type
    {
        double                 a = 0, b = 1, coupling = 0;
        bool                   reduced = false;
        std::shared_ptr<Space> operations;
        struct mapping
        {
            X                   in, out;
            double              a, b, coupling;
            bool                reduced;
            __DEVICE_TAG__ void operator()( std::ptrdiff_t block ) const
            {
                const auto   i = 3 * block;
                const double x = in( i ), y = in( i + 1 ), z = in( i + 2 );
                const double g = reduced ? 0 : .25 * ( x - y );
                out( i )       = a * x - b * ( -x + z + g );
                out( i + 1 )   = a * y - b * ( -y - z + g );
                out( i + 2 )   = a * z - b * ( -2 * z + coupling * ( x + y ) );
            }
        };
        void apply( const X &in, X &out ) const
        {
            map_blocks( *operations, mapping{ in, out, a, b, coupling, reduced } );
        }
    };
    std::shared_ptr<Space>                operations;
    buffer<Space>                         previous, rhs;
    std::shared_ptr<jacobi_operator_type> jacobian;
    double                                time = 0, a = 0, b = 1, coupling = 0;
    bool                                  reduced;
    std::size_t                           rate_calls = 0, invalid_on = 0;
    std::vector<double>                   stage_times, rate_times;
    bool                                  record_times = false;
    explicit implicit_problem( std::shared_ptr<Space> ops, bool reference = false )
        : operations( std::move( ops ) ), previous( *operations ), rhs( *operations ),
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
        if ( record_times )
            stage_times.push_back( time );
    }
    struct mapping
    {
        X                   state, previous, rhs, out;
        double              a, b, time, coupling;
        bool                reduced, rate;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t block ) const
        {
            const auto   i = 3 * block;
            const double x = state( i ), y = state( i + 1 ), z = state( i + 2 );
            const double g    = reduced ? 0 : 1 + time + .25 * ( x - y );
            const double f[3] = { -x + z + g, -y - z + g, -2 * z + coupling * ( x + y ) };
            for ( int j = 0; j < 3; ++j )
                out( i + j ) = rate ? f[j] : a * ( state( i + j ) - previous( i + j ) ) - b * ( f[j] + rhs( i + j ) );
        }
    };
    void apply( const X &state, X &out ) const
    {
        map_blocks( *operations, mapping{ state, *previous, *rhs, out, a, b, time, coupling, reduced, false } );
    }
    bool evaluate_rate( const X &state, X &out )
    {
        ++rate_calls;
        if ( record_times )
            rate_times.push_back( time );
        if ( rate_calls == invalid_on )
            operations->assign_scalar( std::numeric_limits<double>::quiet_NaN(), out );
        else
            map_blocks( *operations, mapping{ state, *previous, *rhs, out, a, b, time, coupling, reduced, true } );
        return true;
    }
    void set_linearization_point( const X & )
    {
        jacobian->a        = a;
        jacobian->b        = b;
        jacobian->coupling = coupling;
        jacobian->reduced  = reduced;
    }
    std::shared_ptr<const jacobi_operator_type> get_jacobi_operator() const
    {
        return jacobian;
    }
};

template <class Space, class Problem = implicit_problem<Space>>
struct implicit_newton
{
    using log_type  = scfd::utils::log_std;
    using monitor   = nmfd::solvers::monitor_krylov<Space, log_type>;
    using linear    = nmfd::solvers::gmres<Space, monitor, log_type, typename Problem::jacobi_operator_type>;
    using iteration = nmfd::solvers::newton_iteration<Space, Problem, linear>;
    using strategy  = nmfd::solvers::default_convergence_strategy<Space, log_type, Problem>;
    using solver    = nmfd::solvers::nonlinear_solver<Space, log_type, Problem, iteration>;
    static std::shared_ptr<solver> make( std::shared_ptr<Space> ops )
    {
        typename linear::params lp;
        lp.basis_size                  = 8;
        lp.batch_size                  = 1;
        lp.monitor.abs_tol             = 2e-14;
        lp.monitor.rel_tol             = 1e-13;
        lp.monitor.max_iters_num       = 40;
        auto                      lin  = std::make_shared<linear>( ops, nullptr, lp );
        auto                      iter = std::make_shared<iteration>( ops, lin );
        typename strategy::params np;
        np.abs_tol       = 2e-13;
        np.rel_tol       = 1e-13;
        np.max_iters_num = 8;
        return std::make_shared<solver>( ops, nullptr, iter, std::make_shared<strategy>( ops, nullptr, np ) );
    }
};
template <class Space, class Control = fixed_control<Space>, bool Dense = false>
using implicit_projected_step = runge_kutta::implicit_time_step<
    Space, implicit_problem<Space>, Control, typename implicit_newton<Space>::solver, Dense, projector<Space>>;
template <class Space, class Control = fixed_control<Space>, bool Dense = false>
using implicit_reference_step = runge_kutta::implicit_time_step<
    Space, implicit_problem<Space>, Control, typename implicit_newton<Space>::solver, Dense>;
}
#endif
