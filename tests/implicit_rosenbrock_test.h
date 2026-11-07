#ifndef NMFD_TIME_STEPPERS_TESTS_ROSENBROCK_H
#define NMFD_TIME_STEPPERS_TESTS_ROSENBROCK_H

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <scfd/utils/log_std.h>
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
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <nmfd/time_steppers/integration/time_integrator.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_constant.h>
#include <nmfd/time_steppers/rosenbrock/time_step.h>

namespace rosenbrock_tests
{
using namespace nmfd::time_steppers;
using operations_type      = nmfd::operations::scfd_vector_space<backend_type, double>;
using vector_type          = operations_type::vector_type;
using vector_wrap_type     = nmfd::detail::vector_wrap<operations_type>;
using adaptation_type      = integration::time_step_adaptation_constant<operations_type>;
inline unsigned int checks = 0;
inline void         check( bool condition, const char *message )
{
    ++checks;
    if ( !condition )
        throw std::runtime_error( message );
}

// Independent scalar decay, u'=-lambda*u. Exact solution is u(0)*exp(-lambda*t).
struct decay
{
    double lambda = 1;
    template <class X>
    __DEVICE_TAG__ double rate( std::ptrdiff_t i, const X &u ) const
    {
        return -lambda * u( i );
    }
    template <class X>
    __DEVICE_TAG__ double jacobian( std::ptrdiff_t i, const X &, const X &v ) const
    {
        return -lambda * v( i );
    }
    template <class X>
    __DEVICE_TAG__ double inverse( std::ptrdiff_t i, const X &, const X &v, double a, double b ) const
    {
        return v( i ) / ( a + b * lambda );
    }
};

// u'=1, exact u=t+u(0). Isolates rounding, lifecycle and retry behavior.
struct constant_rate
{
    template <class X>
    __DEVICE_TAG__ double rate( std::ptrdiff_t, const X & ) const
    {
        return 1;
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

// Logistic ODE u'=u*(1-u), u(0)=1/4; u(t)=1/(1+3*exp(-t)).
// Its state-dependent Jacobian detects accidental stage-wise relinearization.
struct logistic
{
    template <class X>
    __DEVICE_TAG__ double rate( std::ptrdiff_t i, const X &u ) const
    {
        return u( i ) * ( 1 - u( i ) );
    }
    template <class X>
    __DEVICE_TAG__ double jacobian( std::ptrdiff_t i, const X &u, const X &v ) const
    {
        return ( 1 - 2 * u( i ) ) * v( i );
    }
    template <class X>
    __DEVICE_TAG__ double inverse( std::ptrdiff_t i, const X &u, const X &v, double a, double b ) const
    {
        return v( i ) / ( a - b * ( 1 - 2 * u( i ) ) );
    }
};

// Coupled dissipative blocks: x'=-2*x+y, y'=x-2*y; eigenvalues -1,-3.
// From (1,0), x=(exp(-t)+exp(-3*t))/2, y=(exp(-t)-exp(-3*t))/2.
struct coupled_decay
{
    template <class X>
    __DEVICE_TAG__ double rate( std::ptrdiff_t i, const X &u ) const
    {
        return -2 * u( i ) + u( i ^ 1 );
    }
    template <class X>
    __DEVICE_TAG__ double jacobian( std::ptrdiff_t i, const X &, const X &v ) const
    {
        return -2 * v( i ) + v( i ^ 1 );
    }
    template <class X>
    __DEVICE_TAG__ double inverse( std::ptrdiff_t i, const X &, const X &v, double a, double b ) const
    {
        const double d = a + 2 * b;
        return ( d * v( i ) + b * v( i ^ 1 ) ) / ( d * d - b * b );
    }
};

template <class Model, class Operations = operations_type>
struct problem
{
    using vector_type = typename Operations::vector_type;
    using wrap        = nmfd::detail::vector_wrap<Operations>;
    struct jacobian
    {
        std::shared_ptr<Operations> ops;
        wrap                        point;
        Model                       model;
        double                      a = 0, b = 1;
        explicit jacobian( std::shared_ptr<Operations> o ) : ops( std::move( o ) ), point( *ops )
        {
            point.start_use();
        }
        struct mapping
        {
            Model               model;
            double              a, b;
            vector_type         point, in, out;
            bool                inverse;
            __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
            {
                out( i ) = inverse ? model.inverse( i, point, in, a, b ) : a * in(i)-b * model.jacobian( i, point, in );
            }
        };
        void apply( const vector_type &in, vector_type &out ) const
        {
            typename Operations::for_each_type{}( mapping{ model, a, b, *point, in, out, false }, ops->size() );
        }
        void solve( const vector_type &in, vector_type &out ) const
        {
            typename Operations::for_each_type{}( mapping{ model, a, b, *point, in, out, true }, ops->size() );
        }
    };
    using jacobi_operator_type = jacobian;
    std::shared_ptr<Operations> ops;
    wrap                        previous, rhs;
    std::shared_ptr<jacobian>   jacobi;
    Model                       model;
    double                      a = 0, b = 1, time = 0;
    unsigned int                linearizations = 0, rates = 0, residuals = 0;
    unsigned int                fail_rate_on = 0, invalid_rate_on = 0;
    bool                        throw_rate = false, null_jacobian = false;
    explicit problem( std::shared_ptr<Operations> o, Model m = {} )
        : ops( std::move( o ) ), previous( *ops ), rhs( *ops ), jacobi( std::make_shared<jacobian>( ops ) ), model( m )
    {
        previous.start_use();
        rhs.start_use();
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
    struct mapping
    {
        Model               model;
        double              a, b;
        vector_type         in, previous, rhs, out;
        bool                residual;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
        {
            const double f = model.rate( i, in );
            out( i )       = residual ? a * ( in(i)-previous( i ) ) - b * ( f + rhs( i ) ) : f;
        }
    };
    void apply( const vector_type &in, vector_type &out )
    {
        ++residuals;
        typename Operations::for_each_type{}( mapping{ model, a, b, in, *previous, *rhs, out, true }, ops->size() );
    }
    bool evaluate_rate( const vector_type &in, vector_type &out )
    {
        ++rates;
        if ( throw_rate )
            throw std::runtime_error( "Injected rate exception" );
        if ( rates == fail_rate_on || rates == invalid_rate_on )
        {
            ops->assign_scalar( rates == invalid_rate_on ? std::numeric_limits<double>::infinity() : 99., out );
            return rates != fail_rate_on;
        }
        typename Operations::for_each_type{}( mapping{ model, a, b, in, {}, {}, out, false }, ops->size() );
        return true;
    }
    void set_linearization_point( const vector_type &in )
    {
        ++linearizations;
        jacobi->model = model;
        jacobi->a     = a;
        jacobi->b     = b;
        ops->assign( in, *jacobi->point );
    }
    std::shared_ptr<const jacobian> get_jacobi_operator() const
    {
        return null_jacobian ? nullptr : jacobi;
    }
};

// Exact scalar/block inverses only for the analytical test equations above.
// Production engine knows only set_operator() and bool solve(rhs, increment).
template <class Problem>
struct direct_solver
{
    std::shared_ptr<operations_type>                              ops;
    std::shared_ptr<const typename Problem::jacobi_operator_type> jacobi;
    unsigned int                                                  bindings = 0, calls = 0, fail_on = 0, invalid_on = 0;
    bool                                                          throws = false;
    explicit direct_solver( std::shared_ptr<operations_type> o ) : ops( std::move( o ) )
    {
    }
    void set_operator( std::shared_ptr<const typename Problem::jacobi_operator_type> j )
    {
        ++bindings;
        jacobi = std::move( j );
    }
    bool solve( const vector_type &in, vector_type &out )
    {
        ++calls;
        if ( throws )
            throw std::runtime_error( "Injected linear solver exception" );
        if ( calls == fail_on || calls == invalid_on )
        {
            ops->assign_scalar( calls == invalid_on ? std::numeric_limits<double>::infinity() : 99., out );
            return calls != fail_on;
        }
        jacobi->solve( in, out );
        return true;
    }
};

template <class Problem, class Operations = operations_type>
using krylov_solver = nmfd::solvers::gmres<
    Operations, nmfd::solvers::monitor_krylov<Operations, scfd::utils::log_std>, scfd::utils::log_std,
    typename Problem::jacobi_operator_type>;

template <class Solver, class Operations>
auto make_krylov( std::shared_ptr<Operations> ops )
{
    typename Solver::params p;
    p.basis_size            = 4;
    p.batch_size            = 1;
    p.monitor.rel_tol       = 1e-12;
    p.monitor.abs_tol       = 0;
    p.monitor.max_iters_num = 20;
    return std::make_shared<Solver>( ops, nullptr, p );
}
}
#endif
