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

namespace nmfd
{
namespace time_steppers
{
namespace runge_kutta
{
namespace detail
{
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

template <class VectorOperations, class Problem, class TimeStepAdaptation, class NonlinearSolver>
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
          params_( p ), table_( std::move( table ) ), work_( *operations_ )
    {
        if ( p.maximum_attempts == 0 )
            throw std::invalid_argument( "Empty DIRK attempt budget" );
        for ( std::size_t i = 0; i < table_.size(); ++i )
        {
            if ( table_.a( i, i ) == 0 )
                throw std::invalid_argument( "DIRK requires nonzero diagonal stages" );
            for ( std::size_t j = i + 1; j < table_.size(); ++j )
                if ( table_.a( i, j ) != 0 )
                    throw std::invalid_argument( "DIRK requires a lower-triangular table" );
        }
        stiffly_accurate_ = true;
        for ( std::size_t i = 0; i < table_.size(); ++i )
            stiffly_accurate_ = stiffly_accurate_ && table_.a( table_.size() - 1, i ) == table_.b( i );
        work_.init();
        work_.start_use();
        rates_.reserve( table_.size() - ( stiffly_accurate_ ? 1 : 0 ) );
        for ( std::size_t i = 0; i < table_.size() - ( stiffly_accurate_ ? 1 : 0 ); ++i )
        {
            rates_.emplace_back( *operations_ );
            rates_.back().init();
            rates_.back().start_use();
        }
        solver_->set_nonlinear_operator( problem_ );
    }

    void set_time( scalar_type time )
    {
        require_finalized();
        if ( !std::isfinite( time ) )
            throw std::invalid_argument( "Nonfinite DIRK time" );
        time_ = time;
    }

    void set_target_time( scalar_type target )
    {
        require_finalized();
        if ( !std::isfinite( target ) )
            throw std::invalid_argument( "Nonfinite DIRK target" );
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
    const butcher_table &table() const
    {
        return table_;
    }

    void reset()
    {
        adaptation_->reset();
        initialized_ = pending_ = false;
        dt_                     = 0;
        attempts_               = 0;
        status_                 = single_step_status::idle;
    }

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

    void finalize( adaptation_status outcome, scalar_type committed_time, const vector_type &state )
    {
        if ( !pending_ )
            throw std::logic_error( "No DIRK candidate to finalize" );
        const auto elapsed  = committed_time - time_;
        const auto endpoint = dt_ == target_ - time_ ? target_ : time_ + dt_;
        if ( !std::isfinite( committed_time ) || ( dt_ > 0 ? committed_time < time_ || committed_time > endpoint
                                                           : committed_time > time_ || committed_time < endpoint ) )
            throw std::invalid_argument( "Committed time is outside the pending DIRK step" );
        adaptation_->update( outcome, committed_time, elapsed, state );
        pending_ = false;
    }

private:
    using vector_wrap_type = nmfd::detail::vector_wrap<VectorOperations, false, false>;
    std::shared_ptr<VectorOperations>   operations_;
    std::shared_ptr<Problem>            problem_;
    std::shared_ptr<TimeStepAdaptation> adaptation_;
    std::shared_ptr<NonlinearSolver>    solver_;
    params                              params_;
    butcher_table                       table_;
    vector_wrap_type                    work_;
    std::vector<vector_wrap_type>       rates_;
    scalar_type                         time_ = 0, target_ = 1, dt_ = 0;
    unsigned int                        attempts_         = 0;
    bool                                stiffly_accurate_ = false, initialized_ = false, pending_ = false;
    single_step_status                  status_ = single_step_status::idle;

    void require_finalized() const
    {
        if ( pending_ )
            throw std::logic_error( "Finalize the pending DIRK step first" );
    }

    bool compute( const vector_type &in )
    {
        auto &previous = problem_->previous_state();
        auto &rhs      = problem_->known_rhs();
        operations_->assign_scalar( scalar_type( 0 ), rhs );
        for ( std::size_t i = 0; i < table_.size(); ++i )
        {
            const auto alpha = scalar_type( 1 ) / ( dt_ * scalar_type( table_.a( i, i ) ) );
            const auto stage_time =
                time_ + dt_ * scalar_type( stiffly_accurate_ && i + 1 == table_.size() ? 1 : table_.c( i ) );
            if ( !std::isfinite( alpha ) || !std::isfinite( stage_time ) )
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
            problem_->set_affine_shift( alpha );
            operations_->assign( previous, *work_ );
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
            if ( i < rates_.size() )
            {
                operations_->assign_lin_comb( scalar_type( 1 ), *work_, scalar_type( -1 ), previous, *rates_[i] );
                operations_->scale( alpha, *rates_[i] );
                if ( !operations_->is_valid_number( *rates_[i] ) )
                {
                    status_ = single_step_status::failed_nonfinite;
                    return false;
                }
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
        return true;
    }
};
}
}
}

#endif
