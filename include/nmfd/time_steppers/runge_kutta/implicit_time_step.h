#ifndef NMFD_TIME_STEPPERS_IMPLICIT_TIME_STEP_H
#define NMFD_TIME_STEPPERS_IMPLICIT_TIME_STEP_H

#include <algorithm>
#include <cmath>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/time_steppers/detail/status.h>
#include <nmfd/time_steppers/runge_kutta/butcher_tables.h>
#include <nmfd/time_steppers/runge_kutta/continuous_integration.h>

namespace nmfd
{
namespace time_steppers
{
namespace runge_kutta
{
namespace detail
{
template <class VectorOperations, bool Enabled>
struct implicit_dense_storage
{
    explicit implicit_dense_storage( const VectorOperations & )
    {
    }
};

template <class VectorOperations>
struct implicit_dense_storage<VectorOperations, true>
{
    nmfd::detail::vector_wrap<VectorOperations, false, false> previous;

    explicit implicit_dense_storage( const VectorOperations &operations ) : previous( operations )
    {
    }
};

template <class Problem, class Vector, class = void>
struct has_rate_evaluation : std::false_type
{
};

template <class Problem, class Vector>
struct has_rate_evaluation<
    Problem, Vector,
    std::void_t<decltype( std::declval<Problem &>()
                              .evaluate_rate( std::declval<const Vector &>(), std::declval<Vector &>() ) )>>
    : std::is_same<
          decltype( std::declval<Problem &>()
                        .evaluate_rate( std::declval<const Vector &>(), std::declval<Vector &>() ) ),
          bool>
{
};

template <class Adaptation, class Vector, class T, class = void>
struct has_embedded_assessment : std::false_type
{
};

template <class Adaptation, class Vector, class T>
struct has_embedded_assessment<
    Adaptation, Vector, T,
    std::void_t<decltype( std::declval<Adaptation &>().assess(
        std::declval<T>(), std::declval<T>(), std::declval<const Vector &>(), std::declval<const Vector &>(),
        std::declval<T &>(), std::declval<unsigned int>(), std::declval<const Vector *>()
    ) )>> : std::true_type
{
};

template <class Problem, class Vector, class T, class = void>
struct has_implicit_stage_interface : std::false_type
{
};

template <class Problem, class Vector, class T>
struct has_implicit_stage_interface<
    Problem, Vector, T,
    std::void_t<
        typename Problem::jacobi_operator_type, decltype( std::declval<Problem &>().set_time( std::declval<T>() ) ),
        decltype( std::declval<Problem &>().set_affine_shift( std::declval<T>() ) ),
        decltype( std::declval<Problem &>().apply( std::declval<const Vector &>(), std::declval<Vector &>() ) ),
        decltype( std::declval<Problem &>().set_linearization_point( std::declval<const Vector &>() ) ),
        decltype( std::declval<Problem &>().get_jacobi_operator() ),
        decltype( std::declval<Problem &>().previous_state() ), decltype( std::declval<Problem &>().known_rhs() ),
        decltype( std::declval<const Problem &>().previous_state() ),
        decltype( std::declval<const Problem &>().known_rhs() )>>
    : std::bool_constant<
          std::is_same_v<decltype( std::declval<Problem &>().previous_state() ), Vector &> &&
          std::is_same_v<decltype( std::declval<Problem &>().known_rhs() ), Vector &> &&
          std::is_same_v<decltype( std::declval<const Problem &>().previous_state() ), const Vector &> &&
          std::is_same_v<decltype( std::declval<const Problem &>().known_rhs() ), const Vector &> &&
          std::is_convertible_v<
              decltype( std::declval<Problem &>().get_jacobi_operator() ),
              std::shared_ptr<const typename Problem::jacobi_operator_type>>>
{
};
}

template <
    class VectorOperations, class Problem, class TimeStepAdaptation, class NonlinearSolver, bool DenseOutput = false>
class implicit_time_step
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    struct params
    {
        std::string  method           = "IE";
        unsigned int maximum_attempts = 32;
    };

    static_assert(
        detail::has_implicit_stage_interface<Problem, vector_type, scalar_type>::value,
        "DIRK requires problem-owned previous_state/known_rhs and a complete stage linearization"
    );

    // Select the named Butcher table from p.method and delegate construction.
    implicit_time_step(
        std::shared_ptr<VectorOperations> operations, std::shared_ptr<Problem> problem,
        std::shared_ptr<TimeStepAdaptation> adaptation, std::shared_ptr<NonlinearSolver> solver, const params &p = {}
    )
        : implicit_time_step(
              std::move( operations ), std::move( problem ), std::move( adaptation ), std::move( solver ),
              make_butcher_table( p.method ), p
          )
    {
    }

    // Validate a DIRK table and optional rate support, allocate reusable workspace,
    // and bind the nonlinear solver to the problem.
    implicit_time_step(
        std::shared_ptr<VectorOperations> operations, std::shared_ptr<Problem> problem,
        std::shared_ptr<TimeStepAdaptation> adaptation, std::shared_ptr<NonlinearSolver> solver, butcher_table table,
        const params &p = {}
    )
        : operations_(
              operations ? std::move( operations ) : throw std::invalid_argument( "DIRK requires vector operations" )
          ),
          problem_( problem ? std::move( problem ) : throw std::invalid_argument( "DIRK requires a problem" ) ),
          adaptation_(
              adaptation ? std::move( adaptation ) : throw std::invalid_argument( "DIRK requires time adaptation" )
          ),
          solver_( solver ? std::move( solver ) : throw std::invalid_argument( "DIRK requires a nonlinear solver" ) ),
          params_( p ), table_( std::move( table ) ), work_( *operations_ ), error_( *operations_ ),
          dense_( *operations_ )
    {
        if ( p.maximum_attempts == 0 )
            throw std::invalid_argument( "Empty DIRK attempt budget" );
        if ( table_.type() == butcher_table::scheme_type::explicit_rk )
            throw std::invalid_argument( "Use the explicit stepper for an entirely explicit table" );
        if constexpr ( DenseOutput )
            if ( !table_.has_dense_output() )
                throw std::invalid_argument( "DIRK dense output requires a continuous extension in the table" );
        for ( std::size_t i = 0; i < table_.size(); ++i )
        {
            for ( std::size_t j = i + 1; j < table_.size(); ++j )
                if ( table_.a( i, j ) != 0 )
                    throw std::invalid_argument( "DIRK requires a lower-triangular table" );
        }
        stiffly_accurate_ = true;
        for ( std::size_t i = 0; i < table_.size(); ++i )
            stiffly_accurate_ = stiffly_accurate_ && table_.a( table_.size() - 1, i ) == table_.b( i );
        if constexpr ( detail::has_embedded_assessment<TimeStepAdaptation, vector_type, scalar_type>::value )
            estimate_error_ = adaptation_->requires_error_estimate() && table_.is_embedded();
        for ( std::size_t i = 0; i < table_.size(); ++i )
            if ( table_.a( i, i ) == 0 && needs_rate( i ) && !detail::has_rate_evaluation<Problem, vector_type>::value )
                throw std::invalid_argument( "Zero-diagonal stages require bool evaluate_rate(state, rate)" );
        if constexpr ( DenseOutput )
            for ( std::size_t i = table_.size(); i < table_.dense_outout_stage_count(); ++i )
                if ( needs_rate( i ) && !detail::has_rate_evaluation<Problem, vector_type>::value )
                    throw std::invalid_argument(
                        "Additional dense-output stages require bool evaluate_rate(state, rate)"
                    );
        work_.init();
        work_.start_use();
        if ( estimate_error_ )
        {
            error_.init();
            error_.start_use();
        }
        const bool omit_endpoint_rate =
            stiffly_accurate_ && ( !estimate_error_ || table_.error_b( table_.size() - 1 ) == 0 );
        const auto rate_count =
            DenseOutput ? table_.dense_outout_stage_count() : table_.size() - ( omit_endpoint_rate ? 1 : 0 );
        rates_.reserve( rate_count );
        for ( std::size_t i = 0; i < rate_count; ++i )
        {
            rates_.emplace_back( *operations_ );
            rates_.back().init();
            rates_.back().start_use();
        }
        if constexpr ( DenseOutput )
        {
            dense_.previous.init();
            dense_.previous.start_use();
        }
        solver_->set_nonlinear_operator( problem_ );
    }

    // Set the starting time for the next step; requires no pending candidate.
    void set_time( scalar_type time )
    {
        require_finalized();
        if ( !std::isfinite( time ) )
            throw std::invalid_argument( "Nonfinite DIRK time" );
        time_ = time;
    }

    // Set the endpoint limit used to clip the next step, forward or backward.
    void set_target_time( scalar_type target )
    {
        require_finalized();
        if ( !std::isfinite( target ) )
            throw std::invalid_argument( "Nonfinite DIRK target" );
        target_ = target;
    }

    // Return the signed step size used by the most recent numerical attempt.
    scalar_type get_dt() const
    {
        return dt_;
    }
    // Return the current lifecycle or numerical-result status.
    single_step_status get_status() const
    {
        return status_;
    }
    // Return the number of trial steps attempted by the latest apply().
    unsigned int get_attempts() const
    {
        return attempts_;
    }
    // Return the selected Butcher table without copying its coefficients.
    const butcher_table &table() const
    {
        return table_;
    }

    // Report the validated continuous-extension order, or zero when disabled.
    unsigned int dense_output_order() const
    {
        return DenseOutput ? table_.dense_order() : 0;
    }

    // Borrow the accepted numerical step until apply(), reset(), or finalize().
    template <bool Enabled = DenseOutput, std::enable_if_t<Enabled && DenseOutput, int> = 0>
    continuous_integration<implicit_time_step> get_continuous_integration() const
    {
        if ( !pending_ )
            throw std::logic_error( "No pending DIRK dense output" );
        return { *this, generation_ };
    }

    // Clear adaptation history, discard any pending candidate, and restore idle.
    // Keep the configured time, target, dependencies, and allocated workspace.
    void reset()
    {
        adaptation_->reset();
        ++generation_;
        initialized_ = pending_ = false;
        dt_                     = 0;
        attempts_               = 0;
        status_                 = single_step_status::idle;
    }

    // Compute and assess a candidate, retrying rejected attempts within the budget.
    // Publish out only on success; numerical failure leaves it unchanged.
    // Success requires finalize() before another step; time is not advanced here.
    void apply( const vector_type &in, vector_type &out )
    {
        require_finalized();
        ++generation_;
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
            initialized_ = true;
        }
        for ( ; attempts_ < params_.maximum_attempts; )
        {
            ++attempts_;
            const auto proposed  = adaptation_->get_dt();
            const auto remaining = target_ - time_;
            dt_                  = std::copysign( std::min( proposed, std::abs( remaining ) ), remaining );
            // Avoid an unresolved final solve caused only by accumulated time rounding.
            const auto time_roundoff = scalar_type( 4 ) * std::numeric_limits<scalar_type>::epsilon() *
                                       std::max( { std::abs( time_ ), std::abs( target_ ), proposed } );
            if ( std::abs( remaining ) > proposed && std::abs( remaining ) - proposed <= time_roundoff )
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

    // Report the external-processing outcome and actual committed endpoint
    // to adaptation, then release the pending candidate. The integrator owns
    // state commitment and time advancement; this method does not copy state.
    void finalize( adaptation_status outcome, scalar_type committed_time, const vector_type &state )
    {
        if ( !pending_ )
            throw std::logic_error( "No DIRK candidate to finalize" );
        const auto elapsed  = committed_time - time_;
        const auto endpoint = endpoint_time();
        if ( !std::isfinite( committed_time ) || ( dt_ > 0 ? committed_time < time_ || committed_time > endpoint
                                                           : committed_time > time_ || committed_time < endpoint ) )
            throw std::invalid_argument( "Committed time is outside the pending DIRK step" );
        adaptation_->update( outcome, committed_time, elapsed, state );
        pending_ = false;
    }

private:
    using vector_wrap_type = nmfd::detail::vector_wrap<VectorOperations, false, false>;
    std::shared_ptr<VectorOperations>                             operations_;
    std::shared_ptr<Problem>                                      problem_;
    std::shared_ptr<TimeStepAdaptation>                           adaptation_;
    std::shared_ptr<NonlinearSolver>                              solver_;
    params                                                        params_;
    butcher_table                                                 table_;
    vector_wrap_type                                              work_, error_;
    std::vector<vector_wrap_type>                                 rates_;
    detail::implicit_dense_storage<VectorOperations, DenseOutput> dense_;
    std::size_t                                                   generation_ = 0;
    scalar_type                                                   time_ = 0, target_ = 1, dt_ = 0;
    unsigned int                                                  attempts_         = 0;
    bool                                                          stiffly_accurate_ = false, estimate_error_ = false;
    bool                                                          initialized_ = false, pending_ = false;
    single_step_status                                            status_ = single_step_status::idle;

    friend class continuous_integration<implicit_time_step>;

    scalar_type endpoint_time() const
    {
        return dt_ == target_ - time_ ? target_ : time_ + dt_;
    }

    // Retain only derivatives consumed by stages, endpoints, estimators or dense output.
    bool needs_rate( std::size_t i ) const
    {
        if ( i < table_.size() &&
             ( ( !stiffly_accurate_ && table_.b( i ) != 0 ) || ( estimate_error_ && table_.error_b( i ) != 0 ) ) )
            return true;
        for ( std::size_t j = i + 1; j < table_.size(); ++j )
            if ( table_.a( j, i ) != 0 )
                return true;
        if constexpr ( DenseOutput )
        {
            for ( std::size_t j = 0; j < table_.dense_degree(); ++j )
                if ( table_.dense_coefficient( i, j ) != 0 )
                    return true;
            for ( std::size_t j = std::max( table_.size(), i + 1 ); j < table_.dense_outout_stage_count(); ++j )
                if ( table_.dense_outout_a( j, i ) != 0 )
                    return true;
        }
        return false;
    }

    // Sample only stored rates; never invoke the problem or alter its stage data.
    scalar_type evaluate_dense( scalar_type theta, vector_type &out, std::size_t generation ) const
    {
        static_assert( DenseOutput, "Dense output is disabled" );
        if ( !pending_ || generation != generation_ )
            throw std::logic_error( "Expired DIRK dense output" );
        if ( !std::isfinite( theta ) || theta < 0 || theta > 1 )
            throw std::invalid_argument( "Dense-output theta must be in [0,1]" );
        if ( theta == 0 )
            operations_->assign( *dense_.previous, out );
        else if ( theta == 1 )
            operations_->assign( *work_, out );
        else
        {
            operations_->assign( *dense_.previous, out );
            for ( std::size_t i = 0; i < rates_.size(); ++i )
            {
                const auto weight = static_cast<scalar_type>( table_.dense_b( i, theta ) );
                if ( weight != 0 )
                    operations_->add_lin_comb( dt_ * weight, *rates_[i], out );
            }
        }
        return theta == 1 ? endpoint_time() : time_ + theta * dt_;
    }

    // Prepare extra explicit rates only after error acceptance, using the snapshot
    // as scratch. Restore stage time and preserve the endpoint and frozen Jacobian.
    bool prepare_dense( const vector_type &in )
    {
        static_assert( DenseOutput, "Dense output is disabled" );
        const auto restore_time = time_ + dt_ * scalar_type( stiffly_accurate_ ? 1 : table_.c( table_.size() - 1 ) );
        bool       ok = true, time_changed = false;
        try
        {
            for ( std::size_t i = table_.size(); i < rates_.size(); ++i )
            {
                if ( !needs_rate( i ) )
                    continue;
                auto &state = *dense_.previous;
                operations_->assign( in, state );
                for ( std::size_t j = 0; j < i; ++j )
                    if ( table_.dense_outout_a( i, j ) != 0 )
                        operations_->add_lin_comb(
                            dt_ * scalar_type( table_.dense_outout_a( i, j ) ), *rates_[j], state
                        );
                const auto stage_time = time_ + dt_ * scalar_type( table_.dense_outout_c( i ) );
                if ( !std::isfinite( stage_time ) || !operations_->is_valid_number( state ) )
                {
                    status_ = single_step_status::failed_nonfinite;
                    ok      = false;
                    break;
                }
                time_changed = true;
                problem_->set_time( stage_time );
                if constexpr ( detail::has_rate_evaluation<Problem, vector_type>::value )
                    if ( !problem_->evaluate_rate( state, *rates_[i] ) )
                    {
                        status_ = single_step_status::failed_solver;
                        ok      = false;
                        break;
                    }
                if ( !operations_->is_valid_number( *rates_[i] ) )
                {
                    status_ = single_step_status::failed_nonfinite;
                    ok      = false;
                    break;
                }
            }
        }
        catch ( ... )
        {
            if ( time_changed )
                problem_->set_time( restore_time );
            throw;
        }
        if ( time_changed )
            problem_->set_time( restore_time );
        if ( ok )
            operations_->assign( in, *dense_.previous );
        return ok;
    }

    // Reject operations that would overwrite an unfinalized candidate.
    void require_finalized() const
    {
        if ( pending_ )
            throw std::logic_error( "Finalize the pending DIRK step first" );
    }

    // Keep error-free custom controllers compatible with the five-argument assessment.
    adaptation_status assess( const vector_type &in, scalar_type &next_dt )
    {
        if constexpr ( detail::has_embedded_assessment<TimeStepAdaptation, vector_type, scalar_type>::value )
            if ( estimate_error_ )
                return adaptation_->assess( time_, dt_, in, *work_, next_dt, table_.error_order(), &*error_ );
        return adaptation_->assess( time_, dt_, in, *work_, next_dt );
    }

    // Perform one DIRK attempt: assemble stage references, times, and shifts,
    // solve implicit stages or evaluate explicit rates, and reconstruct work_.
    // Return false with a numerical failure status; do not assess or commit.
    bool compute( const vector_type &in )
    {
        auto &previous = problem_->previous_state();
        auto &rhs      = problem_->known_rhs();
        operations_->assign_scalar( scalar_type( 0 ), rhs );
        for ( std::size_t i = 0; i < table_.size(); ++i )
        {
            const bool store_rate = i < rates_.size() && ( table_.a( i, i ) != 0 || needs_rate( i ) );
            const auto stage_time =
                time_ + dt_ * scalar_type( stiffly_accurate_ && i + 1 == table_.size() ? 1 : table_.c( i ) );
            if ( !std::isfinite( stage_time ) )
            {
                status_ = single_step_status::failed_nonfinite;
                return false;
            }
            operations_->assign( in, previous );
            for ( std::size_t j = 0; j < i; ++j )
                if ( table_.a( i, j ) != 0 )
                    operations_->add_lin_comb( dt_ * scalar_type( table_.a( i, j ) ), *rates_[j], previous );
            if ( !operations_->is_valid_number( previous ) )
            {
                status_ = single_step_status::failed_nonfinite;
                return false;
            }
            problem_->set_time( stage_time );
            operations_->assign( previous, *work_ );
            if ( table_.a( i, i ) == 0 )
            {
                // Construction requires evaluate_rate() only for rates that are used.
                if constexpr ( detail::has_rate_evaluation<Problem, vector_type>::value )
                    if ( store_rate && !problem_->evaluate_rate( *work_, *rates_[i] ) )
                    {
                        status_ = single_step_status::failed_solver;
                        return false;
                    }
            }
            else
            {
                const auto alpha = scalar_type( 1 ) / ( dt_ * scalar_type( table_.a( i, i ) ) );
                if ( !std::isfinite( alpha ) )
                {
                    status_ = single_step_status::failed_nonfinite;
                    return false;
                }
                problem_->set_affine_shift( alpha );
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
                if ( store_rate )
                {
                    operations_->assign_lin_comb( scalar_type( 1 ), *work_, scalar_type( -1 ), previous, *rates_[i] );
                    operations_->scale( alpha, *rates_[i] );
                }
            }
            if ( store_rate && !operations_->is_valid_number( *rates_[i] ) )
            {
                status_ = single_step_status::failed_nonfinite;
                return false;
            }
        }
        if ( !stiffly_accurate_ )
        {
            operations_->assign( in, *work_ );
            for ( std::size_t i = 0; i < table_.size(); ++i )
                if ( table_.b( i ) != 0 )
                    operations_->add_lin_comb( dt_ * scalar_type( table_.b( i ) ), *rates_[i], *work_ );
        }
        if ( !operations_->is_valid_number( *work_ ) )
        {
            status_ = single_step_status::failed_nonfinite;
            return false;
        }
        if ( estimate_error_ )
        {
            operations_->assign_scalar( scalar_type( 0 ), *error_ );
            for ( std::size_t i = 0; i < table_.size(); ++i )
                if ( table_.error_b( i ) != 0 )
                    operations_->add_lin_comb( dt_ * scalar_type( table_.error_b( i ) ), *rates_[i], *error_ );
            if ( !operations_->is_valid_number( *error_ ) )
            {
                status_ = single_step_status::failed_nonfinite;
                return false;
            }
        }
        return true;
    }
};
}
}
}

#endif
