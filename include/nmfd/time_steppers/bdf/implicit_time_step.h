#ifndef NMFD_TIME_STEPPERS_BDF_IMPLICIT_TIME_STEP_H
#define NMFD_TIME_STEPPERS_BDF_IMPLICIT_TIME_STEP_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/time_steppers/bdf/coefficients.h>
#include <nmfd/time_steppers/bdf/history.h>
#include <nmfd/time_steppers/bdf/order_policy.h>
#include <nmfd/time_steppers/detail/affine_shift.h>
#include <nmfd/time_steppers/detail/problem_traits.h>
#include <nmfd/time_steppers/detail/status.h>

namespace nmfd::time_steppers::bdf
{
template <class VectorOperations, class Problem, class TimeStepAdaptation, class NonlinearSolver>
class implicit_time_step
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    struct params
    {
        std::string  method             = "BDF2";
        unsigned int maximum_attempts   = 32;
        scalar_type  reference_time     = scalar_type( 1 );
        bool         adaptive           = false;
        scalar_type  minimum_step_ratio = scalar_type( .2 );
        scalar_type  maximum_step_ratio = scalar_type( 2 );
        unsigned int maximum_order      = 5;
    };

    static_assert(
        detail::has_implicit_stage_interface<Problem, vector_type, scalar_type>::value,
        "BDF requires problem-owned previous_state/known_rhs and a complete linearization"
    );

    // Allocate backend workspace once and bind the supplied nonlinear solver.
    implicit_time_step(
        std::shared_ptr<VectorOperations> operations, std::shared_ptr<Problem> problem,
        std::shared_ptr<TimeStepAdaptation> adaptation, std::shared_ptr<NonlinearSolver> solver, const params &p = {}
    )
        : operations_(
              operations ? std::move( operations ) : throw std::invalid_argument( "BDF requires vector operations" )
          ),
          problem_( problem ? std::move( problem ) : throw std::invalid_argument( "BDF requires a problem" ) ),
          adaptation_(
              adaptation ? std::move( adaptation ) : throw std::invalid_argument( "BDF requires time adaptation" )
          ),
          solver_( solver ? std::move( solver ) : throw std::invalid_argument( "BDF requires a nonlinear solver" ) ),
          params_( p ), work_( *operations_ ), base_( *operations_ ), older_( *operations_ ), error_( *operations_ ),
          history_( *operations_ )
    {
        automatic_order_ = p.method == "BDF";
        if ( p.method != "BDF1" && p.method != "BDF2" && !automatic_order_ )
            throw std::invalid_argument( "BDF requires BDF1, BDF2 or adaptive BDF" );
        if ( p.maximum_order < 1 || p.maximum_order > 5 )
            throw std::invalid_argument( "BDF maximum_order must be between 1 and 5" );
        if ( p.maximum_attempts == 0 )
            throw std::invalid_argument( "Empty BDF attempt budget" );
        if ( !std::isfinite( p.minimum_step_ratio ) || p.minimum_step_ratio <= 0 || p.minimum_step_ratio > 1 ||
             !std::isfinite( p.maximum_step_ratio ) || p.maximum_step_ratio < 1 || p.maximum_step_ratio > 2 )
            throw std::invalid_argument( "BDF step ratios require 0<minimum<=1<=maximum<=2" );
        detail::balanced_affine_shift( scalar_type( 0 ), p.reference_time );
        if constexpr ( detail::has_embedded_assessment<TimeStepAdaptation, vector_type, scalar_type>::value )
            estimate_error_ = p.adaptive && adaptation_->requires_error_estimate();
        else if ( p.adaptive && adaptation_->requires_error_estimate() )
            throw std::invalid_argument( "Adaptive BDF requires state-error assessment" );
        if ( automatic_order_ && !estimate_error_ )
            throw std::invalid_argument( "Automatic BDF requires adaptive=true and an error-requiring controller" );
        retain_history_ = p.method == "BDF2" || estimate_error_;
        work_.init();
        work_.start_use();
        if ( retain_history_ )
        {
            base_.init();
            base_.start_use();
            if ( p.adaptive )
            {
                auto capacity = estimate_error_ && p.method == "BDF2" ? 2u : 1u;
                if ( automatic_order_ )
                    capacity = p.maximum_order;
                history_.initialize( capacity );
            }
            else
            {
                older_.init();
                older_.start_use();
            }
        }
        if ( estimate_error_ )
        {
            error_.init();
            error_.start_use();
        }
        solver_->set_nonlinear_operator( problem_ );
    }

    // A discontinuous time change starts fresh history and controller state.
    void set_time( scalar_type time )
    {
        require_finalized();
        if ( !std::isfinite( time ) )
            throw std::invalid_argument( "Nonfinite BDF time" );
        if ( initialized_ && time != expected_time_ )
            reset();
        time_ = time;
    }

    void set_target_time( scalar_type target )
    {
        require_finalized();
        if ( !std::isfinite( target ) )
            throw std::invalid_argument( "Nonfinite BDF target" );
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
    // The order actually used by the latest attempt, including BDF1 restarts.
    unsigned int get_order() const
    {
        return order_;
    }
    unsigned int dense_output_order() const
    {
        return 0;
    }

    // Discard trajectory history without reallocating any workspace.
    void reset()
    {
        adaptation_->reset();
        initialized_ = pending_ = history_ready_ = false;
        dt_ = history_dt_ = scalar_type( 0 );
        expected_time_    = time_;
        attempts_         = 0;
        order_            = 1;
        status_           = single_step_status::idle;
        history_.reset();
        order_policy_.reset();
    }

    // Publish only a successfully solved and assessed candidate. Committed
    // history is read-only here, including retries and in-place stepping.
    void apply( const vector_type &in, vector_type &out )
    {
        require_finalized();
        attempts_ = 0;
        if ( adaptation_->requires_error_estimate() && !estimate_error_ )
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
            initialized_   = true;
            expected_time_ = time_;
        }
        if ( time_ != expected_time_ )
            throw std::logic_error( "Set BDF starting time to the committed endpoint" );
        for ( ; attempts_ < params_.maximum_attempts; )
        {
            ++attempts_;
            const auto proposed  = adaptation_->get_dt();
            const auto remaining = target_ - time_;
            auto       limited   = proposed;
            const auto growth    = automatic_order_ ? std::min( params_.maximum_step_ratio, scalar_type( 1.2 ) )
                                                    : params_.maximum_step_ratio;
            if ( params_.adaptive && history_.size() && std::signbit( remaining ) == std::signbit( history_.step() ) )
                limited = std::min( limited, growth * std::abs( history_.step() ) );
            dt_                 = std::copysign( std::min( limited, std::abs( remaining ) ), remaining );
            const auto roundoff = scalar_type( 4 ) * std::numeric_limits<scalar_type>::epsilon() *
                                  std::max( { std::abs( time_ ), std::abs( target_ ), proposed } );
            if ( std::abs( remaining ) > limited && std::abs( remaining ) - limited <= roundoff )
                dt_ = remaining;
            if ( !std::isfinite( proposed ) || proposed <= 0 || !std::isfinite( dt_ ) || time_ + dt_ == time_ )
            {
                status_ = single_step_status::step_size_underflow;
                return;
            }
            const bool roundoff_clipped = dt_ == remaining && equal_step( proposed, std::abs( history_dt_ ) ) &&
                                          std::signbit( dt_ ) == std::signbit( history_dt_ ) &&
                                          std::abs( dt_ - history_dt_ ) <= roundoff;
            restart_history_ = !history_.usable( dt_, params_.minimum_step_ratio, params_.maximum_step_ratio );
            if ( automatic_order_ )
            {
                order_ = order_policy_.select( history_.size(), attempts_ - 1, restart_history_ );
                if ( order_ > 2 && !history_.smooth( order_, dt_ ) )
                    order_ = 2;
            }
            else if ( params_.adaptive )
                order_ =
                    params_.method == "BDF2" && !restart_history_ && history_.size() >= ( estimate_error_ ? 2u : 1u )
                        ? 2
                        : 1;
            else
                order_ =
                    params_.method == "BDF2" && history_ready_ && ( equal_step( dt_, history_dt_ ) || roundoff_clipped )
                        ? 2
                        : 1;
            if ( !compute( in ) || ( estimate_error_ && !estimate( in ) ) )
            {
                if ( adaptation_->reject_step( dt_ ) != adaptation_status::rejected )
                    return;
                adaptation_->update( adaptation_status::rejected, time_, dt_, in );
                continue;
            }
            scalar_type next_dt  = proposed;
            const auto  decision = assess( in, next_dt );
            if ( decision == adaptation_status::accepted )
            {
                if ( retain_history_ )
                    operations_->assign( in, *base_ );
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

    // Promote the saved input only after an unchanged, full commit. Modified
    // endpoints restart at BDF1; rejected/failed commits retain the older history.
    void finalize( adaptation_status outcome, scalar_type committed_time, const vector_type &state )
    {
        if ( !pending_ )
            throw std::logic_error( "No BDF candidate to finalize" );
        const auto endpoint = endpoint_time();
        const bool accepted = outcome == adaptation_status::accepted || outcome == adaptation_status::accepted_modified;
        if ( !std::isfinite( committed_time ) ||
             ( dt_ > 0 ? committed_time < time_ || committed_time > endpoint
                       : committed_time > time_ || committed_time < endpoint ) ||
             ( accepted && ( committed_time == time_ || !operations_->is_valid_number( state ) ) ) )
            throw std::invalid_argument( "Invalid committed BDF endpoint" );
        if ( outcome == adaptation_status::accepted && committed_time != endpoint )
            outcome = adaptation_status::accepted_modified;
        adaptation_->update( outcome, committed_time, committed_time - time_, state );
        if ( accepted )
        {
            history_ready_ = outcome == adaptation_status::accepted && params_.method == "BDF2" && !params_.adaptive;
            if ( history_ready_ )
            {
                operations_->assign( *base_, *older_ );
                history_dt_ = dt_;
            }
            if ( params_.adaptive && retain_history_ )
            {
                if ( outcome == adaptation_status::accepted )
                    history_.commit( *base_, *work_, dt_, restart_history_ );
                else
                    history_.reset();
                if ( automatic_order_ )
                {
                    if ( outcome == adaptation_status::accepted )
                        order_policy_.commit( order_, params_.maximum_order, attempts_ == 1, restart_history_ );
                    else
                        order_policy_.reset();
                }
            }
            expected_time_ = committed_time;
        }
        pending_ = false;
    }

private:
    using vector_wrap_type = nmfd::detail::vector_wrap<VectorOperations, false, false>;
    std::shared_ptr<VectorOperations>            operations_;
    std::shared_ptr<Problem>                     problem_;
    std::shared_ptr<TimeStepAdaptation>          adaptation_;
    std::shared_ptr<NonlinearSolver>             solver_;
    params                                       params_;
    vector_wrap_type                             work_, base_, older_, error_;
    divided_difference_history<VectorOperations> history_;
    order_policy                                 order_policy_;
    scalar_type                                  time_ = 0, target_ = 1, dt_ = 0, expected_time_ = 0, history_dt_ = 0;
    unsigned int                                 attempts_ = 0, order_ = 1;
    bool                                         initialized_ = false, pending_ = false, history_ready_ = false;
    bool               retain_history_ = false, estimate_error_ = false, restart_history_ = true;
    bool               automatic_order_ = false;
    single_step_status status_          = single_step_status::idle;

    void require_finalized() const
    {
        if ( pending_ )
            throw std::logic_error( "Finalize the pending BDF step first" );
    }
    scalar_type endpoint_time() const
    {
        return dt_ == target_ - time_ ? target_ : time_ + dt_;
    }
    static bool equal_step( scalar_type x, scalar_type y )
    {
        return std::signbit( x ) == std::signbit( y ) &&
               std::abs( x - y ) <= scalar_type( 8 ) * std::numeric_limits<scalar_type>::epsilon() *
                                        std::max( std::abs( x ), std::abs( y ) );
    }

    adaptation_status assess( const vector_type &in, scalar_type &next_dt )
    {
        if constexpr ( detail::has_embedded_assessment<TimeStepAdaptation, vector_type, scalar_type>::value )
            if ( estimate_error_ )
                return adaptation_->assess( time_, dt_, in, *work_, next_dt, order_ + 1, &*error_ );
        return adaptation_->assess( time_, dt_, in, *work_, next_dt );
    }

    bool solve( vector_type &state, scalar_type time, scalar_type alpha )
    {
        if ( !std::isfinite( time ) || !std::isfinite( alpha ) )
        {
            status_ = single_step_status::failed_nonfinite;
            return false;
        }
        const auto shift = detail::balanced_affine_shift( alpha, params_.reference_time );
        problem_->set_time( time );
        problem_->set_affine_shift( shift.a, shift.b );
        if ( !solver_->solve( state ) )
        {
            status_ = single_step_status::failed_solver;
            return false;
        }
        if ( !operations_->is_valid_number( state ) )
        {
            status_ = single_step_status::failed_nonfinite;
            return false;
        }
        return true;
    }

    // Estimate startup by one full versus two half backward-Euler steps.
    // Solve the full step last, leaving the problem prepared for its candidate.
    bool compute( const vector_type &in )
    {
        auto &previous = problem_->previous_state();
        operations_->assign_scalar( scalar_type( 0 ), problem_->known_rhs() );
        if ( estimate_error_ && restart_history_ )
        {
            const auto half_time = time_ + dt_ / 2;
            if ( half_time == time_ || half_time == endpoint_time() )
            {
                status_ = single_step_status::step_size_underflow;
                return false;
            }
            operations_->assign( in, previous );
            operations_->assign( in, *error_ );
            if ( !solve( *error_, half_time, scalar_type( 2 ) / dt_ ) )
                return false;
            operations_->assign( *error_, previous );
            if ( !solve( *error_, endpoint_time(), scalar_type( 2 ) / dt_ ) )
                return false;
        }
        scalar_type alpha;
        if ( automatic_order_ && !restart_history_ )
        {
            const auto c = history_.coefficients( order_, dt_ );
            alpha        = c.alpha;
            history_.reference( c, in, previous );
        }
        else if ( params_.adaptive )
        {
            const auto c = make_coefficients( order_, dt_, history_.step() );
            alpha        = c.alpha;
            if ( order_ == 2 )
                history_.reference( c.history_weight, in, previous );
            else
                operations_->assign( in, previous );
        }
        else
        {
            alpha = ( order_ == 2 ? scalar_type( 1.5 ) : scalar_type( 1 ) ) / dt_;
            if ( order_ == 2 )
                operations_->assign_lin_comb( scalar_type( 4 ) / 3, in, scalar_type( -1 ) / 3, *older_, previous );
            else
                operations_->assign( in, previous );
        }
        if ( !operations_->is_valid_number( previous ) )
        {
            status_ = single_step_status::failed_nonfinite;
            return false;
        }
        // A small residual at an extrapolated guess need not imply a small
        // state error for an ill-conditioned mass/constraint Jacobian.
        operations_->assign( in, *work_ );
        return solve( *work_, endpoint_time(), alpha );
    }

    bool estimate( const vector_type &in )
    {
        if ( restart_history_ )
            operations_->assign_lin_comb( scalar_type( 2 ), *work_, scalar_type( -2 ), *error_, *error_ );
        else
            history_.estimate( order_, dt_, in, *work_, *error_ );
        if ( !operations_->is_valid_number( *error_ ) )
        {
            status_ = single_step_status::failed_nonfinite;
            return false;
        }
        return true;
    }
};
}
#endif
