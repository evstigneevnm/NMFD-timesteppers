#ifndef NMFD_TIME_STEPPERS_ROSENBROCK_TIME_STEP_H
#define NMFD_TIME_STEPPERS_ROSENBROCK_TIME_STEP_H

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/time_steppers/detail/affine_shift.h>
#include <nmfd/time_steppers/detail/problem_traits.h>
#include <nmfd/time_steppers/detail/status.h>
#include <nmfd/time_steppers/rosenbrock/coefficients.h>
#include <nmfd/time_steppers/rosenbrock/continuous_integration.h>

namespace nmfd::time_steppers::rosenbrock
{
// Physical-rate ODEs (including problem-owned nonsingular mass solves) or
// opt-in constant-mass residuals. Only W
// tables permit approximate Jacobians. Evaluations preserve the frozen operator.
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
        bool         mass_form        = false;
        bool         index1_dae       = false;
    };

    static_assert(
        detail::has_implicit_stage_interface<Problem, vector_type, scalar_type>::value,
        "Rosenbrock requires the existing shifted-stage interface"
    );

    // Allocate stage increments and two scratch vectors, plus q_t only for
    // non-autonomous problems. Estimation and shared rates reuse scratch.
    time_step(
        std::shared_ptr<VectorOperations> operations, std::shared_ptr<Problem> problem,
        std::shared_ptr<TimeStepAdaptation> adaptation, std::shared_ptr<LinearSolver> solver, const params &p = {}
    )
        : operations_(
              operations ? std::move( operations ) : throw std::invalid_argument( "Rosenbrock requires operations" )
          ),
          problem_( problem ? std::move( problem ) : throw std::invalid_argument( "Rosenbrock requires a problem" ) ),
          adaptation_(
              adaptation ? std::move( adaptation ) : throw std::invalid_argument( "Rosenbrock requires adaptation" )
          ),
          solver_(
              solver ? std::move( solver ) : throw std::invalid_argument( "Rosenbrock requires a linear solver" )
          ),
          params_( p ), table_( p.method ), work_( *operations_ ), rhs_( *operations_ ),
          time_derivative_( *operations_ ), dense_( *operations_ )
    {
        if ( !p.maximum_attempts )
            throw std::invalid_argument( "Rosenbrock requires a nonempty attempt budget" );
        if ( p.index1_dae )
        {
            if ( !p.mass_form || !p.autonomous || !table_.supports_constant_mass_index1() )
                throw std::invalid_argument( "Index-1 Rosenbrock requires autonomous ROS34PW2 mass form" );
            if constexpr ( DenseOutput )
                throw std::invalid_argument( "Index-1 Rosenbrock dense output is not validated" );
        }
        if ( !p.mass_form && !detail::has_rate_evaluation<Problem, vector_type>::value )
            throw std::invalid_argument( "Rate-form Rosenbrock requires bool evaluate_rate(state, rate)" );
        if constexpr ( DenseOutput )
            if ( !table_.dense_output_order() )
                throw std::invalid_argument( "Selected Rosenbrock method has no validated dense output" );
        if ( !p.autonomous && !detail::has_time_derivative_evaluation<Problem, vector_type>::value )
            throw std::invalid_argument(
                "Non-autonomous Rosenbrock requires bool evaluate_time_derivative(state, out)"
            );
        detail::balanced_affine_shift( scalar_type( 0 ), p.reference_time );
        for ( auto *v : { &work_, &rhs_ } )
        {
            v->init();
            v->start_use();
        }
        increments_.reserve( table_.stage_count() );
        for ( std::size_t i = 0; i < table_.stage_count(); ++i )
        {
            increments_.emplace_back( *operations_ );
            increments_.back().init();
            increments_.back().start_use();
        }
        if ( !p.autonomous )
        {
            time_derivative_.init();
            time_derivative_.start_use();
        }
        if constexpr ( DenseOutput )
        {
            dense_.previous.init();
            dense_.previous.start_use();
            if ( table_.stage_count() >= 3 )
            {
                dense_.correction.init();
                dense_.correction.start_use();
            }
            if ( table_.stage_count() == 4 )
            {
                dense_.endpoint.init();
                dense_.endpoint.start_use();
            }
        }
    }

    // Change time or endpoint limits only after releasing the previous candidate.
    void set_time( scalar_type time )
    {
        require_finalized();
        if ( !std::isfinite( time ) )
            throw std::invalid_argument( "Nonfinite Rosenbrock time" );
        time_ = time;
    }
    void set_target_time( scalar_type target )
    {
        require_finalized();
        if ( !std::isfinite( target ) )
            throw std::invalid_argument( "Nonfinite Rosenbrock target" );
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
    const method_table &table() const
    {
        return table_;
    }
    unsigned int dense_output_order() const
    {
        return DenseOutput ? table_.dense_output_order() : 0;
    }

    // Borrow only the pending numerical step; the owner must outlive the view.
    template <bool Enabled = DenseOutput, std::enable_if_t<Enabled && DenseOutput, int> = 0>
    runge_kutta::continuous_integration<time_step> get_continuous_integration() const
    {
        if ( !pending_ )
            throw std::logic_error( "No pending Rosenbrock dense output" );
        return { *this, generation_ };
    }

    // Discard pending state and controller history without reallocating workspace.
    void reset()
    {
        adaptation_->reset();
        ++generation_;
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
        ++generation_;
        attempts_       = 0;
        estimate_error_ = adaptation_->requires_error_estimate();
        if ( estimate_error_ && !detail::has_embedded_assessment<TimeStepAdaptation, vector_type, scalar_type>::value )
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
            const auto  decision = assess( in, next_dt );
            if ( decision == adaptation_status::accepted )
            {
                if constexpr ( DenseOutput )
                    if ( !prepare_dense( in ) )
                    {
                        if ( adaptation_->reject_step( dt_ ) != adaptation_status::rejected )
                            return;
                        adaptation_->update( adaptation_status::rejected, time_, dt_, in );
                        continue;
                    }
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
            throw std::logic_error( "No Rosenbrock candidate to finalize" );
        const auto endpoint = dt_ == target_ - time_ ? target_ : time_ + dt_;
        if ( !std::isfinite( committed_time ) || ( dt_ > 0 ? committed_time < time_ || committed_time > endpoint
                                                           : committed_time > time_ || committed_time < endpoint ) )
            throw std::invalid_argument( "Committed time is outside the pending Rosenbrock step" );
        adaptation_->update( outcome, committed_time, committed_time - time_, state );
        ++generation_;
        pending_ = false;
    }

private:
    friend class runge_kutta::continuous_integration<time_step>;
    using vector_wrap_type = nmfd::detail::vector_wrap<VectorOperations, false, false>;
    std::shared_ptr<VectorOperations>                               operations_;
    std::shared_ptr<Problem>                                        problem_;
    std::shared_ptr<TimeStepAdaptation>                             adaptation_;
    std::shared_ptr<LinearSolver>                                   solver_;
    params                                                          params_;
    method_table                                                    table_;
    vector_wrap_type                                                work_, rhs_, time_derivative_;
    std::vector<vector_wrap_type>                                   increments_;
    detail::rosenbrock_dense_storage<VectorOperations, DenseOutput> dense_;
    std::size_t                                                     generation_ = 0;
    scalar_type                                                     time_ = 0, target_ = 1, dt_ = 0;
    unsigned int                                                    attempts_ = 0;
    bool               initialized_ = false, pending_ = false, estimate_error_ = false;
    single_step_status status_ = single_step_status::idle;

    void require_finalized() const
    {
        if ( pending_ )
            throw std::logic_error( "Finalize the pending Rosenbrock step first" );
    }

    adaptation_status assess( const vector_type &in, scalar_type &next_dt )
    {
        if constexpr ( detail::has_embedded_assessment<TimeStepAdaptation, vector_type, scalar_type>::value )
            if ( estimate_error_ )
                return adaptation_->assess( time_, dt_, in, *work_, next_dt, table_.error_order(), &*rhs_ );
        return adaptation_->assess( time_, dt_, in, *work_, next_dt );
    }

    // Scale A and stage right-hand sides together. Couplings b*c_ij/h equal
    // a*gamma*c_ij, avoiding division by a potentially tiny timestep.
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
            throw std::invalid_argument( "Rosenbrock requires a prepared Jacobian" );
        solver_->set_operator( std::move( jacobian ) );
        if ( !prepare_time_derivative( in ) )
            return false;
        for ( std::size_t i = 0; i < table_.stage_count(); ++i )
        {
            if ( params_.mass_form || !table_.reuse_previous_rate( i ) )
            {
                const vector_type *state = &in;
                if ( i )
                {
                    operations_->assign_lin_comb(
                        scalar_type( 1 ), in, scalar_type( table_.a( i, 0 ) ), *increments_[0], *work_
                    );
                    for ( std::size_t j = 1; j < i; ++j )
                        if ( table_.a( i, j ) != 0 )
                            operations_->add_lin_comb( scalar_type( table_.a( i, j ) ), *increments_[j], *work_ );
                    if ( !operations_->is_valid_number( *work_ ) )
                    {
                        status_ = single_step_status::failed_nonfinite;
                        return false;
                    }
                    state = &*work_;
                }
                const auto c          = table_.stage_time( i );
                const auto stage_time = params_.autonomous || c == 0 ? time_
                                        : c == 1 ? ( dt_ == target_ - time_ ? target_ : time_ + dt_ )
                                                 : time_ + dt_ * scalar_type( c );
                if ( params_.mass_form )
                {
                    std::array<double, 4> coupling{};
                    for ( std::size_t j = 0; j < i; ++j )
                        coupling[j] = table_.gamma * table_.c( i, j );
                    if ( !residual_rhs( *state, coupling, stage_time, table_.time_derivative( i ) ) ||
                         !solve( *increments_[i] ) )
                        return false;
                    continue;
                }
                if ( !rate( *state, stage_time ) )
                    return false;
                if constexpr ( DenseOutput )
                    if ( table_.stage_count() == 4 && i == 0 )
                        operations_->assign( *rhs_, *dense_.correction );
                // Once the stage state has been consumed, work_ can retain its
                // unshifted rate for consecutive stages sharing that state/time.
                if ( i + 1 < table_.stage_count() && table_.reuse_previous_rate( i + 1 ) )
                    operations_->assign( *rhs_, *work_ );
            }
            else
                operations_->assign( *work_, *rhs_ );
            operations_->scale( shift.b, *rhs_ );
            if ( !params_.autonomous )
                operations_->add_lin_comb(
                    shift.b * dt_ * scalar_type( table_.time_derivative( i ) ), *time_derivative_, *rhs_
                );
            for ( std::size_t j = 0; j < i; ++j )
                if ( table_.c( i, j ) != 0 )
                    operations_->add_lin_comb(
                        shift.a * scalar_type( table_.gamma * table_.c( i, j ) ), *increments_[j], *rhs_
                    );
            if ( !solve( *increments_[i] ) )
                return false;
        }
        operations_->assign_lin_comb( scalar_type( 1 ), in, scalar_type( table_.m( 0 ) ), *increments_[0], *work_ );
        for ( std::size_t i = 1; i < table_.stage_count(); ++i )
            operations_->add_lin_comb( scalar_type( table_.m( i ) ), *increments_[i], *work_ );
        if ( !operations_->is_valid_number( *work_ ) )
        {
            status_ = single_step_status::failed_nonfinite;
            return false;
        }
        if ( estimate_error_ )
        {
            // k_i are state increments, so the error already contains h.
            operations_->assign_lin_comb(
                scalar_type( table_.e( 0 ) ), *increments_[0], scalar_type( table_.e( 1 ) ), *increments_[1], *rhs_
            );
            for ( std::size_t i = 2; i < table_.stage_count(); ++i )
                operations_->add_lin_comb( scalar_type( table_.e( i ) ), *increments_[i], *rhs_ );
            if ( !operations_->is_valid_number( *rhs_ ) )
            {
                status_ = single_step_status::failed_nonfinite;
                return false;
            }
        }
        return true;
    }

    // For R(U)=a*M*(U-previous)-b*f(U), previous=U+sum(q_j*y_j)
    // gives -R=b*f(U)+a*M*sum(q_j*y_j), without a mass inverse or action API.
    // Index-1 problems keep algebraic residual/Jacobian rows unscaled; the
    // reference coupling affects only differential rows. No projection is inferred.
    // apply() must not change the operator prepared at the attempt's base state.
    bool residual_rhs(
        const vector_type &state, const std::array<double, 4> &coupling, scalar_type stage_time,
        double time_coefficient = 0
    )
    {
        auto &previous = problem_->previous_state();
        operations_->assign( state, previous );
        for ( std::size_t j = 0; j < increments_.size(); ++j )
            if ( coupling[j] != 0 )
                operations_->add_lin_comb( scalar_type( coupling[j] ), *increments_[j], previous );
        if ( !params_.autonomous && time_coefficient != 0 )
        {
            // M*q_t=f_t for constant mass. Encode h^2*gamma*d_i*q_t without
            // forming h^2, which can overflow/underflow despite a finite defect.
            operations_->assign_lin_comb( dt_, *time_derivative_, *rhs_ );
            operations_->add_lin_comb( dt_ * scalar_type( table_.gamma * time_coefficient ), *rhs_, previous );
        }
        if ( !operations_->is_valid_number( previous ) )
        {
            status_ = single_step_status::failed_nonfinite;
            return false;
        }
        return evaluate_at_time( stage_time, [&] {
            problem_->apply( state, *rhs_ );
            operations_->scale( scalar_type( -1 ), *rhs_ );
            return true;
        } );
    }

    // Recompute partial rate derivative q_t at the base point of every attempt.
    bool prepare_time_derivative( const vector_type &state )
    {
        if ( params_.autonomous )
            return true;
        if constexpr ( detail::has_time_derivative_evaluation<Problem, vector_type>::value )
            if ( !problem_->evaluate_time_derivative( state, *time_derivative_ ) )
            {
                status_ = single_step_status::failed_solver;
                return false;
            }
        if ( !operations_->is_valid_number( *time_derivative_ ) )
        {
            status_ = single_step_status::failed_nonfinite;
            return false;
        }
        return true;
    }

    // No rate callback is needed by residual-form problems.
    bool rate( const vector_type &state, scalar_type stage_time )
    {
        if constexpr ( !detail::has_rate_evaluation<Problem, vector_type>::value )
            throw std::logic_error( "Rate evaluation is unavailable for this Rosenbrock problem" );
        else
            return evaluate_at_time( stage_time, [&] { return problem_->evaluate_rate( state, *rhs_ ); } );
    }

    // Restore base time after stage/endpoint evaluations, including failures
    // and exceptions. Neither callback may overwrite the frozen linearization.
    template <class Evaluation>
    bool evaluate_at_time( scalar_type stage_time, Evaluation evaluate )
    {
        if ( !std::isfinite( stage_time ) )
        {
            status_ = single_step_status::failed_nonfinite;
            return false;
        }
        bool ok;
        if ( stage_time == time_ )
            ok = evaluate();
        else
        {
            try
            {
                problem_->set_time( stage_time );
                ok = evaluate();
            }
            catch ( ... )
            {
                problem_->set_time( time_ );
                throw;
            }
            problem_->set_time( time_ );
        }
        if ( !ok )
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

    // Dense-only solves run after error acceptance; the numerical endpoint and
    // frozen operator are unchanged. Preserve the base state before publication.
    bool prepare_dense( const vector_type &in )
    {
        if ( table_.stage_count() == 3 )
        {
            if ( params_.mass_form )
            {
                std::array<double, 4> coupling{};
                for ( std::size_t i = 0; i < increments_.size(); ++i )
                    coupling[i] = -table_.gamma * table_.m( i );
                const auto endpoint = dt_ == target_ - time_ ? target_ : time_ + dt_;
                if ( !residual_rhs( *work_, coupling, params_.autonomous ? time_ : endpoint ) ||
                     !solve( *dense_.correction ) )
                    return false;
            }
            else
            {
                const auto endpoint = dt_ == target_ - time_ ? target_ : time_ + dt_;
                if ( !rate( *work_, params_.autonomous ? time_ : endpoint ) )
                    return false;
                const auto shift =
                    detail::balanced_affine_shift( scalar_type( 1 / table_.gamma ) / dt_, params_.reference_time );
                operations_->scale( shift.b, *rhs_ );
                for ( std::size_t i = 0; i < increments_.size(); ++i )
                    operations_->add_lin_comb(
                        -shift.a * scalar_type( table_.gamma * table_.m( i ) ), *increments_[i], *rhs_
                    );
                if ( !solve( *dense_.correction ) )
                    return false;
            }
        }
        else if ( table_.stage_count() == 4 && !prepare_w_dense( in ) )
            return false;
        operations_->assign( in, *dense_.previous );
        return true;
    }

    // Prepare third-order slope increments using the same shifted inverse.
    // Rate form caches f0; mass form evaluates both coupled endpoint residuals.
    bool prepare_w_dense( const vector_type &in )
    {
        const auto shift =
            detail::balanced_affine_shift( scalar_type( 1 / table_.gamma ) / dt_, params_.reference_time );
        for ( std::size_t stage = 0; stage < 2; ++stage )
        {
            const auto &coupling = ros34pw2_dense_coupling( stage );
            if ( params_.mass_form )
            {
                const auto endpoint = dt_ == target_ - time_ ? target_ : time_ + dt_;
                if ( !residual_rhs( stage ? *work_ : in, coupling, params_.autonomous || !stage ? time_ : endpoint ) )
                    return false;
            }
            else
            {
                if ( !stage )
                    operations_->assign( *dense_.correction, *rhs_ );
                else
                {
                    const auto endpoint = dt_ == target_ - time_ ? target_ : time_ + dt_;
                    if ( !rate( *work_, params_.autonomous ? time_ : endpoint ) )
                        return false;
                }
                operations_->scale( shift.b, *rhs_ );
                for ( std::size_t i = 0; i < increments_.size(); ++i )
                    operations_->add_lin_comb( shift.a * scalar_type( coupling[i] ), *increments_[i], *rhs_ );
            }
            auto &slope = stage ? *dense_.endpoint : *dense_.correction;
            if ( !solve( slope ) )
                return false;
            for ( std::size_t i = 0; i < increments_.size(); ++i )
                operations_->add_lin_comb( -scalar_type( coupling[i] ), *increments_[i], slope );
            operations_->scale( scalar_type( 1 / table_.gamma ), slope );
            if ( !operations_->is_valid_number( slope ) )
            {
                status_ = single_step_status::failed_nonfinite;
                return false;
            }
        }
        return true;
    }

    // Interpolation is backend algebra only; never call the problem while sampling.
    scalar_type evaluate_dense( scalar_type theta, vector_type &out, std::size_t generation ) const
    {
        if ( !pending_ || generation != generation_ )
            throw std::logic_error( "Expired Rosenbrock dense output" );
        const auto weights = continuous_weights( table_, theta );
        if ( theta == 0 )
            operations_->assign( *dense_.previous, out );
        else if ( theta == 1 )
            operations_->assign( *work_, out );
        else
        {
            operations_->assign_lin_comb( 1 - theta, *dense_.previous, theta, *work_, out );
            for ( std::size_t i = 0; i < increments_.size(); ++i )
                operations_->add_lin_comb( weights[i], *increments_[i], out );
            if ( table_.stage_count() == 3 )
                operations_->add_lin_comb( weights[3], *dense_.correction, out );
            else if ( table_.stage_count() == 4 )
            {
                const auto factor = theta * ( 1 - theta );
                operations_->add_lin_comb( factor * ( 1 - theta ), *dense_.correction, out );
                operations_->add_lin_comb( -factor * theta, *dense_.endpoint, out );
            }
        }
        return theta == 1 ? ( dt_ == target_ - time_ ? target_ : time_ + dt_ ) : time_ + theta * dt_;
    }
};
}
#endif
