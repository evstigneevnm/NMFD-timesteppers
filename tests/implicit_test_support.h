#ifndef NMFD_TIME_STEPPERS_TESTS_IMPLICIT_SUPPORT_H
#define NMFD_TIME_STEPPERS_TESTS_IMPLICIT_SUPPORT_H

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <scfd/static_mat/mat.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/log_std.h>
#ifdef TEST_VECTOR_BACKEND_CUDA
#    include <scfd/backend/cuda.h>
using backend_type = scfd::backend::cuda<std::ptrdiff_t>;
#elif defined( TEST_VECTOR_BACKEND_OMP )
#    include <scfd/backend/omp.h>
using backend_type = scfd::backend::omp<std::ptrdiff_t>;
#else
#    include <scfd/backend/serial_cpu.h>
using backend_type = scfd::backend::serial_cpu<std::ptrdiff_t>;
#endif
#include <nmfd/operations/scfd_vector_space.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/solvers/nonlinear_solver.h>
#include <nmfd/time_steppers/runge_kutta/implicit_time_step.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_constant.h>
#include <nmfd/time_steppers/integration/time_integrator.h>

namespace nmfd
{
namespace time_steppers
{
namespace tests
{
using operations_type  = nmfd::operations::scfd_vector_space<backend_type, double>;
using vector_type      = operations_type::vector_type;
using vector_wrap_type = nmfd::detail::vector_wrap<operations_type>;
using adaptation_type  = integration::time_step_adaptation_constant<operations_type>;

inline void require( bool condition, const char *message )
{
    if ( !condition )
        throw std::runtime_error( message );
}

template <class Mapping>
void execute_small_mapping( const Mapping &mapping )
{
    if constexpr ( operations_type::memory_type::is_host_visible )
        mapping( 0 );
    else
    {
        operations_type::for_each_type execute;
        execute( mapping, 1 );
        execute.wait();
    }
}

template <int Size>
struct matrix_jacobian
{
    using matrix_type  = scfd::static_mat::mat<double, Size, Size>;
    matrix_type matrix = matrix_type::make_zero();

    struct mapping
    {
        matrix_type         matrix;
        vector_type         in, out;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t ) const
        {
            scfd::static_vec::vec<double, Size> x;
            for ( int i = 0; i < Size; ++i )
                x[i] = in( i );
            for ( int i = 0; i < Size; ++i )
            {
                double value = 0;
                for ( int j = 0; j < Size; ++j )
                    value += matrix( i, j ) * x[j];
                out( i ) = value;
            }
        }
    };
    void apply( const vector_type &in, vector_type &out ) const
    {
        execute_small_mapping( mapping{ matrix, in, out } );
    }
};

// Equations supply their residual and analytical stage Jacobian, not a new solver layer.
template <class Equation>
class stage_problem
{
public:
    using scalar_type          = double;
    using vector_type          = tests::vector_type;
    using jacobi_operator_type = matrix_jacobian<Equation::size>;
    using state_type           = scfd::static_vec::vec<double, Equation::size>;

    Equation     equation;
    double       time = 0, alpha = 0;
    unsigned int linearizations = 0;

    explicit stage_problem( std::shared_ptr<operations_type> operations, Equation e = {} )
        : equation( e ), operations_( std::move( operations ) ), previous_( *operations_ ), rhs_( *operations_ ),
          jacobi_( std::make_shared<jacobi_operator_type>() )
    {
        require( operations_->size() == Equation::size, "Implicit test problem dimension mismatch" );
        previous_.start_use();
        rhs_.start_use();
    }
    void set_time( double t )
    {
        time = t;
    }
    void set_affine_shift( double value )
    {
        alpha = value;
    }
    vector_type &previous_state()
    {
        return *previous_;
    }
    const vector_type &previous_state() const
    {
        return *previous_;
    }
    vector_type &known_rhs()
    {
        return *rhs_;
    }
    const vector_type &known_rhs() const
    {
        return *rhs_;
    }

    struct mapping
    {
        Equation            equation;
        double              time, alpha;
        vector_type         state, previous, rhs, out;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t ) const
        {
            auto result = out;
            equation.residual( time, alpha, state, previous, rhs, result );
        }
    };
    void apply( const vector_type &state, vector_type &out ) const
    {
        execute_small_mapping( mapping{ equation, time, alpha, state, *previous_, *rhs_, out } );
    }
    void set_linearization_point( const vector_type &state )
    {
        state_type x, previous;
        operations_->get( state, x.d, Equation::size );
        operations_->get( *previous_, previous.d, Equation::size );
        jacobi_->matrix = equation.jacobian( time, alpha, x, previous );
        ++linearizations;
    }
    std::shared_ptr<const jacobi_operator_type> get_jacobi_operator() const
    {
        return jacobi_;
    }

private:
    std::shared_ptr<operations_type>      operations_;
    vector_wrap_type                      previous_, rhs_;
    std::shared_ptr<jacobi_operator_type> jacobi_;
};

template <class Problem>
struct newton_types
{
    using monitor_type = nmfd::solvers::monitor_krylov<operations_type, scfd::utils::log_std>;
    using linear_type  = nmfd::solvers::gmres<
         operations_type, monitor_type, scfd::utils::log_std, typename Problem::jacobi_operator_type>;
    using iteration_type = nmfd::solvers::newton_iteration<operations_type, Problem, linear_type>;
    using solver_type = nmfd::solvers::nonlinear_solver<operations_type, scfd::utils::log_std, Problem, iteration_type>;
    using strategy_type = nmfd::solvers::default_convergence_strategy<operations_type, scfd::utils::log_std, Problem>;
    using step_type     = runge_kutta::implicit_time_step<operations_type, Problem, adaptation_type, solver_type>;

    static std::shared_ptr<solver_type> make_solver( std::shared_ptr<operations_type> operations )
    {
        typename linear_type::params lp;
        lp.basis_size                            = 4;
        lp.batch_size                            = 1;
        lp.monitor.abs_tol                       = 1e-13;
        lp.monitor.rel_tol                       = 1e-11;
        lp.monitor.max_iters_num                 = 40;
        auto                           linear    = std::make_shared<linear_type>( operations, nullptr, lp );
        auto                           iteration = std::make_shared<iteration_type>( operations, linear );
        typename strategy_type::params np;
        np.abs_tol       = 2e-10;
        np.rel_tol       = 1e-12;
        np.max_iters_num = 30;
        auto strategy    = std::make_shared<strategy_type>( operations, nullptr, np );
        return std::make_shared<solver_type>( operations, nullptr, iteration, strategy );
    }
};

// Scalar test models have M(t)*u'=f(t,u); only the test stage formulation is shared.
template <class Equation>
struct scalar_time_equation : Equation
{
    static constexpr int size = 1;
    scalar_time_equation()    = default;
    explicit scalar_time_equation( Equation equation ) : Equation( equation )
    {
    }
    __DEVICE_TAG__ void residual(
        double t, double alpha, const vector_type &x, const vector_type &previous, const vector_type &rhs,
        vector_type &out
    ) const
    {
        out( 0 ) = alpha * this->mass( t ) * ( x( 0 ) - previous( 0 ) ) - this->rhs( t, x( 0 ) ) - rhs( 0 );
    }
    auto jacobian(
        double t, double alpha, const scfd::static_vec::vec<double, 1> &x, const scfd::static_vec::vec<double, 1> &
    ) const
    {
        return scfd::static_mat::mat<double, 1, 1>{ alpha * this->mass( t ) - this->rhs_derivative( t, x[0] ) };
    }
};

template <class Problem>
std::shared_ptr<typename newton_types<Problem>::step_type> make_step(
    std::shared_ptr<operations_type> operations, std::shared_ptr<Problem> problem, double dt,
    const char *method = "SDIRK2(1)2"
)
{
    using types     = newton_types<Problem>;
    auto adaptation = std::make_shared<adaptation_type>( adaptation_type::params{ dt } );
    return std::make_shared<typename types::step_type>(
        operations, problem, adaptation, types::make_solver( operations ), typename types::step_type::params{ method }
    );
}

template <class Equation>
void check_jacobian(
    std::shared_ptr<operations_type> operations, stage_problem<Equation> &problem,
    const scfd::static_vec::vec<double, Equation::size> &point, double time = .4, double alpha = 7
)
{
    vector_wrap_type state( *operations ), direction( *operations ), plus( *operations ), minus( *operations ),
        result( *operations ), derivative( *operations );
    state.start_use();
    direction.start_use();
    plus.start_use();
    minus.start_use();
    result.start_use();
    derivative.start_use();
    operations->set( point.d, *state, Equation::size );
    operations->assign_scalar( .3, *direction );
    operations->assign_scalar( .2, problem.previous_state() );
    operations->assign_scalar( .1, problem.known_rhs() );
    problem.set_time( time );
    problem.set_affine_shift( alpha );
    problem.set_linearization_point( *state );
    auto jacobi = problem.get_jacobi_operator();
    jacobi->apply( *direction, *derivative );
    operations->assign_lin_comb( 1., *state, 1e-5, *direction, *plus );
    operations->assign_lin_comb( 1., *state, -1e-5, *direction, *minus );
    problem.apply( *plus, *result );
    problem.apply( *minus, *plus );
    operations->assign_lin_comb( 5e4, *result, -5e4, *plus, *minus );
    operations->add_lin_comb( -1., *derivative, *minus );
    require(
        operations->norm_l2( *minus ) < 2e-6 * std::max( 1., operations->norm_l2( *derivative ) ),
        "Stage Jacobian finite difference mismatch"
    );
    jacobi->apply( *direction, *result );
    operations->add_lin_comb( -1., *derivative, *result );
    require( operations->norm_l2( *result ) == 0, "Trial residual changed the frozen Jacobian" );
}
}
}
}
#endif
