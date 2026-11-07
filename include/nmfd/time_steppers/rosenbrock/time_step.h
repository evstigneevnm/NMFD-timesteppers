#ifndef NMFD_TIME_STEPPERS_ROSENBROCK_TIME_STEP_H
#define NMFD_TIME_STEPPERS_ROSENBROCK_TIME_STEP_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/time_steppers/detail/affine_shift.h>
#include <nmfd/time_steppers/detail/problem_traits.h>
#include <nmfd/time_steppers/detail/status.h>
#include <nmfd/time_steppers/rosenbrock/coefficients.h>

namespace nmfd::time_steppers::rosenbrock
{
// This initial engine supports autonomous ODEs with identity mass and an exact
// base Jacobian. Rate evaluation must preserve the prepared shifted operator.
template <class VectorOperations, class Problem, class TimeStepAdaptation, class LinearSolver, bool DenseOutput = false>
class time_step
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    struct params
    {
        std::string  method           = "ROS2";
        unsigned int maximum_attempts = 32;
        scalar_type  reference_time   = scalar_type( 1 );
        bool         autonomous       = true;
    };

    static_assert(
        detail::has_implicit_stage_interface<Problem, vector_type, scalar_type>::value &&
            detail::has_rate_evaluation<Problem, vector_type>::value,
        "ROS2 requires the existing shifted-stage interface and bool evaluate_rate(state, rate)"
    );

    // Retain supplied dependencies and allocate four reusable backend vectors.
    // No nonlinear solver, mass inverse, estimator or dense storage is constructed.
    time_step(
        std::shared_ptr<VectorOperations> operations, std::shared_ptr<Problem> problem,
        std::shared_ptr<TimeStepAdaptation> adaptation, std::shared_ptr<LinearSolver> solver, const params &p = {}
    )
        : operations_(
              operations ? std::move( operations ) : throw std::invalid_argument( "ROS2 requires operations" )
          ),
          problem_( problem ? std::move( problem ) : throw std::invalid_argument( "ROS2 requires a problem" ) ),
          adaptation_(
              adaptation ? std::move( adaptation ) : throw std::invalid_argument( "ROS2 requires adaptation" )
          ),
          solver_( solver ? std::move( solver ) : throw std::invalid_argument( "ROS2 requires a linear solver" ) ),
          params_( p ), table_( p.method ), work_( *operations_ ), rhs_( *operations_ ), k1_( *operations_ ),
          k2_( *operations_ )
    {
        if ( !p.maximum_attempts || !p.autonomous )
            throw std::invalid_argument( "ROS2 requires a nonempty attempt budget and an autonomous problem" );
        if constexpr ( DenseOutput )
            throw std::invalid_argument( "ROS2 dense output is not implemented" );
        detail::balanced_affine_shift( scalar_type( 0 ), p.reference_time );
        for ( auto *v : { &work_, &rhs_, &k1_, &k2_ } )
        {
            v->init();
            v->start_use();
        }
    }

    // Change time or endpoint limits only after releasing the previous candidate.
    void set_time( scalar_type time )
    {
        require_finalized();
        if ( !std::isfinite( time ) )
            throw std::invalid_argument( "Nonfinite ROS2 time" );
        time_ = time;
    }
    void set_target_time( scalar_type target )
    {
        require_finalized();
        if ( !std::isfinite( target ) )
            throw std::invalid_argument( "Nonfinite ROS2 target" );
        target_ = target;
    }
    scalar_type get_dt() const
    {
        return dt_;
    }
    single_step_status get_status() const
    {
        return status_;
    }
    unsigned int get_attempts() const
    {
        return attempts_;
    }
    const coefficients &table() const
    {
        return table_;
    }
    unsigned int dense_output_order() const
    {
        return 0;
    }

    // Discard pending state and controller history without reallocating workspace.
    void reset()
    {
        adaptation_->reset();
        initialized_ = pending_ = false;
        attempts_               = 0;
        dt_                     = 0;
        status_                 = single_step_status::idle;
    }

    // Publish only a finite, accepted candidate; retries always start from in.
    // The caller, not apply(), owns time advancement and final commitment.
    void apply( const vector_type &in, vector_type &out )
    {
        require_finalized();
        attempts_ = 0;
        if ( adaptation_->requires_error_estimate() )
        {
            status_ = single_step_status::error_estimate_unavailable;
            return;
        }
        if ( !operations_->is_valid_number( in ) )
        {
            status_ = single_step_status::failed_nonfinite;
            return;
        }
        if ( !initialized_ )
        {
            adaptation_->initialize( time_, in );
            initialized_ = true;
        }
        while ( attempts_ < params_.maximum_attempts )
        {
            ++attempts_;
            const auto proposed  = adaptation_->get_dt();
            const auto remaining = target_ - time_;
            dt_                  = std::copysign( std::min( proposed, std::abs( remaining ) ), remaining );
            const auto roundoff  = scalar_type( 4 ) * std::numeric_limits<scalar_type>::epsilon() *
                                  std::max( { std::abs( time_ ), std::abs( target_ ), proposed } );
            if ( std::abs( remaining ) > proposed && std::abs( remaining ) - proposed <= roundoff )
                dt_ = remaining;
            if ( !std::isfinite( proposed ) || proposed <= 0 || !std::isfinite( dt_ ) || time_ + dt_ == time_ )
            {
                status_ = single_step_status::step_size_underflow;
                return;
            }
            if ( !compute( in ) )
            {
                if ( adaptation_->reject_step( dt_ ) != adaptation_status::rejected )
                    return;
                adaptation_->update( adaptation_status::rejected, time_, dt_, in );
                continue;
            }
            scalar_type next_dt  = proposed;
            const auto  decision = adaptation_->assess( time_, dt_, in, *work_, next_dt );
            if ( decision == adaptation_status::accepted )
            {
                operations_->assign( *work_, out );
                status_  = single_step_status::converged;
                pending_ = true;
                return;
            }
            adaptation_->update( decision, time_, dt_, in );
            if ( decision != adaptation_status::rejected )
            {
                status_ = single_step_status::failed_minimum_dt;
                return;
            }
        }
        status_ = single_step_status::attempt_limit_reached;
    }

    // Report the actual externally committed endpoint, including rejection or
    // shortening. No rates or shifted operators are reused on the next step.
    void finalize( adaptation_status outcome, scalar_type committed_time, const vector_type &state )
    {
        if ( !pending_ )
            throw std::logic_error( "No ROS2 candidate to finalize" );
        const auto endpoint = dt_ == target_ - time_ ? target_ : time_ + dt_;
        if ( !std::isfinite( committed_time ) || ( dt_ > 0 ? committed_time < time_ || committed_time > endpoint
                                                           : committed_time > time_ || committed_time < endpoint ) )
            throw std::invalid_argument( "Committed time is outside the pending ROS2 step" );
        adaptation_->update( outcome, committed_time, committed_time - time_, state );
        pending_ = false;
    }

private:
    using vector_wrap_type = nmfd::detail::vector_wrap<VectorOperations, false, false>;
    std::shared_ptr<VectorOperations>   operations_;
    std::shared_ptr<Problem>            problem_;
    std::shared_ptr<TimeStepAdaptation> adaptation_;
    std::shared_ptr<LinearSolver>       solver_;
    params                              params_;
    coefficients                        table_;
    vector_wrap_type                    work_, rhs_, k1_, k2_;
    scalar_type                         time_ = 0, target_ = 1, dt_ = 0;
    unsigned int                        attempts_    = 0;
    bool                                initialized_ = false, pending_ = false;
    single_step_status                  status_ = single_step_status::idle;

    void require_finalized() const
    {
        if ( pending_ )
            throw std::logic_error( "Finalize the pending ROS2 step first" );
    }

    // Scale A and both stage right-hand sides together. The coupling b*c21/h
    // equals a*gamma*c21, avoiding a division by a potentially tiny timestep.
    bool compute( const vector_type &in )
    {
        const auto alpha = scalar_type( 1 / table_.gamma ) / dt_;
        if ( !std::isfinite( alpha ) )
        {
            status_ = single_step_status::failed_nonfinite;
            return false;
        }
        const auto shift = detail::balanced_affine_shift( alpha, params_.reference_time );
        operations_->assign( in, problem_->previous_state() );
        operations_->assign_scalar( scalar_type( 0 ), problem_->known_rhs() );
        problem_->set_time( time_ );
        problem_->set_affine_shift( shift.a, shift.b );
        problem_->set_linearization_point( in );
        auto jacobian = problem_->get_jacobi_operator();
        if ( !jacobian )
            throw std::invalid_argument( "ROS2 requires a prepared Jacobian" );
        solver_->set_operator( std::move( jacobian ) );
        if ( !rate( in ) )
            return false;
        operations_->scale( shift.b, *rhs_ );
        if ( !solve( *k1_ ) )
            return false;
        operations_->assign_lin_comb( scalar_type( 1 ), in, scalar_type( table_.a21 ), *k1_, *work_ );
        if ( !operations_->is_valid_number( *work_ ) )
        {
            status_ = single_step_status::failed_nonfinite;
            return false;
        }
        if ( !rate( *work_ ) )
            return false;
        operations_->scale( shift.b, *rhs_ );
        operations_->add_lin_comb( shift.a * scalar_type( table_.gamma * table_.c21 ), *k1_, *rhs_ );
        if ( !solve( *k2_ ) )
            return false;
        operations_->assign_lin_comb( scalar_type( 1 ), in, scalar_type( table_.m1 ), *k1_, *work_ );
        operations_->add_lin_comb( scalar_type( table_.m2 ), *k2_, *work_ );
        if ( !operations_->is_valid_number( *work_ ) )
        {
            status_ = single_step_status::failed_nonfinite;
            return false;
        }
        return true;
    }

    bool rate( const vector_type &state )
    {
        if ( !problem_->evaluate_rate( state, *rhs_ ) )
        {
            status_ = single_step_status::failed_solver;
            return false;
        }
        if ( !operations_->is_valid_number( *rhs_ ) )
        {
            status_ = single_step_status::failed_nonfinite;
            return false;
        }
        return true;
    }

    bool solve( vector_type &increment )
    {
        if ( !operations_->is_valid_number( *rhs_ ) )
        {
            status_ = single_step_status::failed_nonfinite;
            return false;
        }
        operations_->assign_scalar( scalar_type( 0 ), increment );
        if ( !solver_->solve( *rhs_, increment ) )
        {
            status_ = single_step_status::failed_solver;
            return false;
        }
        if ( !operations_->is_valid_number( increment ) )
        {
            status_ = single_step_status::failed_nonfinite;
            return false;
        }
        return true;
    }
};
}
#endif
