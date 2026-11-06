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
        std::string  method           = "BDF2";
        unsigned int maximum_attempts = 32;
        scalar_type  reference_time   = scalar_type( 1 );
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
          params_( p ), work_( *operations_ ), base_( *operations_ ), older_( *operations_ )
    {
        if ( p.method != "BDF1" && p.method != "BDF2" )
            throw std::invalid_argument( "Fixed BDF supports only BDF1 and BDF2" );
        if ( p.maximum_attempts == 0 )
            throw std::invalid_argument( "Empty BDF attempt budget" );
        detail::balanced_affine_shift( scalar_type( 0 ), p.reference_time );
        work_.init();
        work_.start_use();
        if ( p.method == "BDF2" )
        {
            base_.init();
            base_.start_use();
            older_.init();
            older_.start_use();
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
    }

    // Publish only a successfully solved and assessed candidate. Committed
    // history is read-only here, including retries and in-place stepping.
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
            const bool roundoff_clipped = dt_ == remaining && equal_step( proposed, std::abs( history_dt_ ) ) &&
                                          std::signbit( dt_ ) == std::signbit( history_dt_ ) &&
                                          std::abs( dt_ - history_dt_ ) <= roundoff;
            order_ =
                params_.method == "BDF2" && history_ready_ && ( equal_step( dt_, history_dt_ ) || roundoff_clipped )
                    ? 2
                    : 1;
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
                if ( params_.method == "BDF2" )
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
            history_ready_ = outcome == adaptation_status::accepted && params_.method == "BDF2";
            if ( history_ready_ )
            {
                operations_->assign( *base_, *older_ );
                history_dt_ = dt_;
            }
            expected_time_ = committed_time;
        }
        pending_ = false;
    }

private:
    using vector_wrap_type = nmfd::detail::vector_wrap<VectorOperations, false, false>;
    std::shared_ptr<VectorOperations>   operations_;
    std::shared_ptr<Problem>            problem_;
    std::shared_ptr<TimeStepAdaptation> adaptation_;
    std::shared_ptr<NonlinearSolver>    solver_;
    params                              params_;
    vector_wrap_type                    work_, base_, older_;
    scalar_type                         time_ = 0, target_ = 1, dt_ = 0, expected_time_ = 0, history_dt_ = 0;
    unsigned int                        attempts_ = 0, order_ = 1;
    bool                                initialized_ = false, pending_ = false, history_ready_ = false;
    single_step_status                  status_ = single_step_status::idle;

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

    // Form alpha*M(u)*(u-w)-f(t,u)=0 without evaluating or inverting M.
    bool compute( const vector_type &in )
    {
        auto &previous = problem_->previous_state();
        if ( order_ == 2 )
            operations_->assign_lin_comb( scalar_type( 4 ) / 3, in, scalar_type( -1 ) / 3, *older_, previous );
        else
            operations_->assign( in, previous );
        operations_->assign_scalar( scalar_type( 0 ), problem_->known_rhs() );
        const auto alpha = ( order_ == 2 ? scalar_type( 1.5 ) : scalar_type( 1 ) ) / dt_;
        if ( !std::isfinite( endpoint_time() ) || !std::isfinite( alpha ) || !operations_->is_valid_number( previous ) )
        {
            status_ = single_step_status::failed_nonfinite;
            return false;
        }
        const auto shift = detail::balanced_affine_shift( alpha, params_.reference_time );
        problem_->set_time( endpoint_time() );
        problem_->set_affine_shift( shift.a, shift.b );
        operations_->assign( in, *work_ );
        if ( !solver_->solve( *work_ ) )
        {
            status_ = single_step_status::failed_solver;
            return false;
        }
        if ( !operations_->is_valid_number( *work_ ) )
        {
            status_ = single_step_status::failed_nonfinite;
            return false;
        }
        return true;
    }
};
}
#endif
