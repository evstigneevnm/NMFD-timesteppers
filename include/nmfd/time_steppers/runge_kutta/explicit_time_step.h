#ifndef NMFD_TIME_STEPPERS_EXPLICIT_TIME_STEP_H
#define NMFD_TIME_STEPPERS_EXPLICIT_TIME_STEP_H
#include <cmath>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/ident_operator.h>
#include <nmfd/time_steppers/detail/problem_traits.h>
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
using nmfd::time_steppers::detail::has_set_time;

template <class Adaptation, class = void>
struct has_error_requirement : std::false_type
{
};

template <class Adaptation>
struct has_error_requirement<
    Adaptation, std::void_t<decltype( std::declval<const Adaptation &>().requires_error_estimate() )>> : std::true_type
{
};

template <class Adaptation, class T, class Vector, class = void>
struct has_dual_error_assessment : std::false_type
{
};

template <class Adaptation, class T, class Vector>
struct has_dual_error_assessment<
    Adaptation, T, Vector,
    std::void_t<decltype( std::declval<Adaptation &>().assess(
        std::declval<T>(), std::declval<T>(), std::declval<const Vector &>(), std::declval<const Vector &>(),
        std::declval<T &>(), std::declval<unsigned int>(), std::declval<const Vector *>(),
        std::declval<const Vector *>()
    ) )>> : std::true_type
{
};
}

template <
    class VectorOperations, class Problem, class TimeStepAdaptation, bool DenseOutput = false,
    class ProjectionOperator = nmfd::operations::ident_operator<VectorOperations>>
class explicit_time_step
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    static constexpr bool projects_stage_rates =
        !std::is_same_v<ProjectionOperator, nmfd::operations::ident_operator<VectorOperations>>;

    struct params
    {
        std::string  method              = "RK45";
        unsigned int maximum_attempts    = 32;
        bool         reuse_endpoint_rate = false;
    };
    static_assert(
        nmfd::time_steppers::detail::has_rate_evaluation<Problem, vector_type>::value,
        "Explicit RK requires bool evaluate_rate(state, rate); optional set_time(stage_time)"
    );

    explicit_time_step(
        std::shared_ptr<VectorOperations> operations, std::shared_ptr<Problem> problem,
        std::shared_ptr<TimeStepAdaptation> adaptation, const params &p = {},
        std::shared_ptr<ProjectionOperator> projection_operator = nullptr
    )
        : operations_(
              operations ? std::move( operations ) : throw std::invalid_argument( "RK requires vector operations" )
          ),
          problem_( problem ? std::move( problem ) : throw std::invalid_argument( "RK requires a problem" ) ),
          adaptation_(
              adaptation ? std::move( adaptation ) : throw std::invalid_argument( "RK requires time adaptation" )
          ),
          projection_operator_( std::move( projection_operator ) ), params_( p ),
          table_( make_butcher_table( p.method ) ), combined_error_( use_combined_error() ), work_( *operations_ ),
          error_( *operations_, table_.is_embedded() || ( combined_error_ && supports_combined_error ) ),
          secondary_error_( *operations_, combined_error_ && supports_combined_error ),
          dense_( *operations_, !table_.has_dense_output() )
    {
        if constexpr ( projects_stage_rates )
        {
            static_assert(
                std::is_same_v<
                    decltype( std::declval<ProjectionOperator &>().apply( std::declval<vector_type &>() ) ), bool>,
                "RK stage projection requires bool apply(vector_type&)"
            );
            if ( !projection_operator_ )
                throw std::invalid_argument( "RK requires the selected stage projector" );
        }
        if ( table_.type() != butcher_table::scheme_type::explicit_rk )
        {
            throw std::invalid_argument( "Explicit RK cannot execute an implicit table" );
        }
        if ( p.maximum_attempts == 0 )
        {
            throw std::invalid_argument( "Empty RK attempt budget" );
        }
        work_.start_use();
        if ( table_.is_embedded() || ( combined_error_ && supports_combined_error ) )
        {
            error_.start_use();
        }
        if ( combined_error_ && supports_combined_error )
        {
            secondary_error_.start_use();
        }
        const auto stage_count = DenseOutput ? table_.dense_outout_stage_count() : table_.size();
        reuse_endpoint_rate_   = p.reuse_endpoint_rate && table_.has_fsal();
        stages_.reserve( stage_count );
        stage_slots_.reserve( stage_count );
        for ( std::size_t i = 0; i < stage_count; ++i )
        {
            stages_.emplace_back( *operations_ );
            stages_.back().start_use();
            stage_slots_.push_back( i );
        }
    }

    void set_time( scalar_type time )
    {
        if ( pending_ )
        {
            throw std::logic_error( "Finalize the pending RK step first" );
        }
        if ( !std::isfinite( time ) )
        {
            throw std::invalid_argument( "Nonfinite RK time" );
        }
        if ( cached_rate_valid_ && time != cached_rate_time_ )
        {
            invalidate_rhs_cache();
        }
        time_ = time;
    }

    void set_target_time( scalar_type target )
    {
        if ( pending_ )
        {
            throw std::logic_error( "Finalize the pending RK step first" );
        }
        if ( !std::isfinite( target ) )
        {
            throw std::invalid_argument( "Nonfinite RK target" );
        }
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

    unsigned int dense_output_order() const
    {
        return DenseOutput ? ( table_.has_dense_output() ? table_.dense_order() : std::min( 3u, table_.order() ) ) : 0;
    }

    template <bool Enabled = DenseOutput, std::enable_if_t<Enabled && DenseOutput, int> = 0>
    continuous_integration<explicit_time_step> get_continuous_integration() const
    {
        if ( !pending_ )
        {
            throw std::logic_error( "No pending RK dense output" );
        }
        return {
            *this, generation_
        }; // continuous_integration<explicit_time_step>(const SingleStepMethod& owner, std::size_t generation)
    }

    const vector_type &error_estimate() const
    {
        if ( !pending_ || !table_.is_embedded() )
        {
            throw std::logic_error( "No pending RK error estimate" );
        }
        return *error_;
    }

    // DOP853 components are raw E5/E3 defects, not standalone combined estimates.
    const vector_type &error_estimate( std::size_t component ) const
    {
        if ( !pending_ || ( !table_.is_embedded() && !combined_error_ ) )
        {
            throw std::logic_error( "No pending RK error components" );
        }
        if ( component == 0 )
        {
            return *error_;
        }
        if ( component == 1 && combined_error_ )
        {
            return *secondary_error_;
        }
        throw std::out_of_range( "RK error component index" );
    }

    void invalidate_rhs_cache()
    {
        cached_rate_valid_ = false;
        ++rhs_generation_;
    }

    void reset()
    {
        adaptation_->reset();
        invalidate_rhs_cache();
        initialized_ = pending_ = false;
        ++generation_;
        dt_       = 0;
        attempts_ = 0;
        status_   = single_step_status::idle;
    }

    void apply( const vector_type &in, vector_type &out )
    {
        if ( pending_ )
        {
            throw std::logic_error( "Finalize the pending RK step first" );
        }
        ++generation_;
        bool first_rate_valid = reuse_endpoint_rate_ && cached_rate_valid_ && cached_rate_time_ == time_;
        cached_rate_valid_    = false;
        trial_rhs_generation_ = rhs_generation_;
        if constexpr ( detail::has_error_requirement<TimeStepAdaptation>::value )
        {
            if ( adaptation_->requires_error_estimate() && !table_.has_error_estimate() )
            {
                status_ = single_step_status::error_estimate_unavailable;
                return;
            }
        }
        if ( combined_error_ && !supports_combined_error )
        {
            status_ = single_step_status::error_estimate_unavailable;
            return;
        }
        if ( !initialized_ )
        {
            adaptation_->initialize( time_, in );
            initialized_ = true;
        }
        status_ = single_step_status::attempt_limit_reached;
        for ( attempts_ = 0; attempts_ < params_.maximum_attempts; )
        {
            ++attempts_;
            const auto proposed  = adaptation_->get_dt();
            const auto remaining = target_ - time_;
            dt_                  = std::copysign( std::min( proposed, std::abs( remaining ) ), remaining );
            if ( !std::isfinite( proposed ) || proposed <= 0 || !std::isfinite( dt_ ) || time_ + dt_ == time_ )
            {
                status_ = single_step_status::step_size_underflow;
                return;
            }
            if ( !compute( in, first_rate_valid ) )
            {
                // No assessment/callback may inspect an invalid numerical candidate.
                if ( adaptation_->reject_step( dt_ ) != adaptation_status::rejected )
                {
                    return;
                }
                adaptation_->update( adaptation_status::rejected, time_, dt_, in );
                continue;
            }
            scalar_type next_dt  = proposed;
            const auto  decision = assess( in, next_dt );
            if ( decision == adaptation_status::accepted )
            {
                if constexpr ( DenseOutput )
                {
                    if ( !prepare_dense( in ) )
                    {
                        if ( adaptation_->reject_step( dt_ ) != adaptation_status::rejected )
                        {
                            return;
                        }
                        adaptation_->update( adaptation_status::rejected, time_, dt_, in );
                        continue;
                    }
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

    void finalize( adaptation_status outcome, scalar_type committed_time, const vector_type &state )
    {
        if ( !pending_ )
        {
            throw std::logic_error( "No RK candidate to finalize" );
        }
        const auto elapsed = committed_time - time_;
        if ( !std::isfinite( committed_time ) || elapsed * dt_ < 0 ||
             std::abs( elapsed ) > std::abs( dt_ ) * scalar_type( 1.00000001 ) )
        {
            throw std::invalid_argument( "Committed time is outside the pending RK step" );
        }
        adaptation_->update( outcome, committed_time, elapsed, state );
        pending_           = false;
        cached_rate_valid_ = reuse_endpoint_rate_ && outcome == adaptation_status::accepted &&
                             committed_time == time_ + dt_ && trial_rhs_generation_ == rhs_generation_;
        if ( cached_rate_valid_ )
        {
            // Dense output has expired; only the logical buffer roles change.
            std::swap( stage_slots_.front(), stage_slots_[table_.size() - 1] );
            cached_rate_time_ = committed_time;
        }
    }

private:
    using vector_wrap_type = nmfd::detail::vector_wrap<VectorOperations>;
    static constexpr bool supports_combined_error =
        detail::has_dual_error_assessment<TimeStepAdaptation, scalar_type, vector_type>::value;
    std::shared_ptr<VectorOperations>                    operations_;
    std::shared_ptr<Problem>                             problem_;
    std::shared_ptr<TimeStepAdaptation>                  adaptation_;
    std::shared_ptr<ProjectionOperator>                  projection_operator_;
    params                                               params_;
    butcher_table                                        table_;
    bool                                                 combined_error_;
    vector_wrap_type                                     work_, error_, secondary_error_;
    std::vector<vector_wrap_type>                        stages_;
    std::vector<std::size_t>                             stage_slots_;
    detail::dense_storage<VectorOperations, DenseOutput> dense_;
    std::size_t                                          generation_     = 0;
    std::size_t                                          rhs_generation_ = 0, trial_rhs_generation_ = 0;
    scalar_type                                          cached_rate_time_    = 0;
    bool                                                 reuse_endpoint_rate_ = false, cached_rate_valid_ = false;
    scalar_type                                          time_ = 0, target_ = 1, dt_ = 0;
    unsigned int                                         attempts_    = 0;
    bool                                                 initialized_ = false, pending_ = false;
    single_step_status                                   status_ = single_step_status::idle;

    friend class continuous_integration<explicit_time_step>;

    vector_type &stage( std::size_t i )
    {
        return *stages_[stage_slots_[i]];
    }

    const vector_type &stage( std::size_t i ) const
    {
        return *stages_[stage_slots_[i]];
    }

    bool use_combined_error() const
    {
        if ( table_.error_estimator() != butcher_table::error_estimator_type::dop853_combined )
        {
            return false;
        }
        if constexpr ( detail::has_error_requirement<TimeStepAdaptation>::value )
        {
            return adaptation_->requires_error_estimate();
        }
        return supports_combined_error;
    }

    adaptation_status assess( const vector_type &in, scalar_type &next_dt )
    {
        if constexpr ( supports_combined_error )
        {
            if ( combined_error_ )
            {
                return adaptation_->assess(
                    time_, dt_, in, *work_, next_dt, table_.error_order(), &*error_, &*secondary_error_
                );
            }
        }
        return adaptation_->assess(
            time_, dt_, in, *work_, next_dt, table_.error_order(), table_.is_embedded() ? &*error_ : nullptr
        );
    }

    scalar_type evaluate_dense( scalar_type theta, vector_type &out, std::size_t generation ) const
    {
        static_assert( DenseOutput, "Dense output is disabled" );
        if ( !pending_ || generation != generation_ )
        {
            throw std::logic_error( "Expired RK dense output" );
        }
        if ( !std::isfinite( theta ) || theta < 0 || theta > 1 )
        {
            throw std::invalid_argument( "Dense-output theta must be in [0,1]" );
        }
        if ( theta == 0 )
        {
            operations_->assign( *dense_.previous, out );
        }
        else if ( theta == 1 )
        {
            operations_->assign( *work_, out );
        }
        else if ( table_.has_dense_output() )
        {
            operations_->assign( *dense_.previous, out );
            for ( std::size_t i = 0; i < table_.dense_outout_stage_count(); ++i )
            {
                const auto weight = static_cast<scalar_type>( table_.dense_b( i, theta ) );
                if ( weight != 0 )
                {
                    operations_->add_mul( dt_ * weight, stage( i ), out );
                }
            }
        }
        else
        {
            const auto a = theta * theta * ( 3 - 2 * theta ), b = theta * ( 1 - theta ) * ( 1 - theta ),
                       c = -theta * theta * ( 1 - theta );
            operations_->assign_mul( 1 - a, *dense_.previous, a, *work_, out );
            operations_->add_mul( dt_ * b, stage( 0 ), out );
            operations_->add_mul( dt_ * c, *dense_.endpoint_rate, out );
        }
        return time_ + theta * dt_;
    }

    bool prepare_dense( const vector_type &in )
    {
        static_assert( DenseOutput, "Dense output is disabled" );
        status_ = single_step_status::failed_nonfinite;
        // Reuse the snapshot buffer as scratch without disturbing the candidate.
        auto &scratch = *dense_.previous;
        for ( std::size_t i = table_.size(); i < table_.dense_outout_stage_count(); ++i )
        {
            operations_->assign( in, scratch );
            for ( std::size_t j = 0; j < i; ++j )
            {
                const auto weight = static_cast<scalar_type>( table_.dense_outout_a( i, j ) );
                if ( weight != 0 )
                {
                    operations_->add_mul( dt_ * weight, stage( j ), scratch );
                }
            }
            if ( !operations_->is_valid_number( scratch ) )
            {
                return false;
            }
            if constexpr ( detail::has_set_time<Problem, scalar_type>::value )
            {
                problem_->set_time( time_ + dt_ * static_cast<scalar_type>( table_.dense_outout_c( i ) ) );
            }
            if ( !evaluate_rate( scratch, stage( i ) ) )
            {
                return false;
            }
        }
        if constexpr ( detail::has_set_time<Problem, scalar_type>::value )
        {
            problem_->set_time( time_ + dt_ );
        }
        if ( !table_.has_dense_output() )
        {
            if ( !evaluate_rate( *work_, *dense_.endpoint_rate ) )
            {
                return false;
            }
        }
        // Snapshot before publishing the candidate: step.apply(x, x) is supported.
        operations_->assign( in, scratch );
        return true;
    }

    bool compute( const vector_type &in, bool &first_rate_valid )
    {
        status_    = single_step_status::failed_nonfinite;
        auto &work = *work_;
        for ( std::size_t i = 0; i < table_.size(); ++i )
        {
            if constexpr ( detail::has_set_time<Problem, scalar_type>::value )
            {
                problem_->set_time( time_ + dt_ * static_cast<scalar_type>( table_.c( i ) ) );
            }
            if ( i == 0 && first_rate_valid )
            {
                continue;
            }
            operations_->assign( in, work );
            for ( std::size_t j = 0; j < i; ++j )
            {
                if ( table_.a( i, j ) != 0 )
                {
                    operations_->add_mul( dt_ * static_cast<scalar_type>( table_.a( i, j ) ), stage( j ), work );
                }
            }
            if ( !evaluate_rate( work, stage( i ) ) )
            {
                return false;
            }
            if ( i == 0 )
            {
                first_rate_valid = reuse_endpoint_rate_;
            }
        }
        // All derivatives are retained; the last stage state can become the candidate.
        operations_->assign( in, work );
        const bool primary_error = table_.is_embedded() || combined_error_;
        if ( primary_error )
        {
            operations_->assign_scalar( scalar_type( 0 ), *error_ );
        }
        if ( combined_error_ )
        {
            operations_->assign_scalar( scalar_type( 0 ), *secondary_error_ );
        }
        for ( std::size_t i = 0; i < table_.size(); ++i )
        {
            if ( table_.b( i ) != 0 )
            {
                operations_->add_mul( dt_ * static_cast<scalar_type>( table_.b( i ) ), stage( i ), work );
            }
            if ( primary_error && table_.error_b( i ) != 0 )
            {
                operations_->add_mul( dt_ * static_cast<scalar_type>( table_.error_b( i ) ), stage( i ), *error_ );
            }
            if ( combined_error_ && table_.secondary_error_b( i ) != 0 )
            {
                operations_->add_mul(
                    dt_ * static_cast<scalar_type>( table_.secondary_error_b( i ) ), stage( i ), *secondary_error_
                );
            }
        }
        return operations_->is_valid_number( work ) && ( !primary_error || operations_->is_valid_number( *error_ ) ) &&
               ( !combined_error_ || operations_->is_valid_number( *secondary_error_ ) );
    }

    bool evaluate_rate( const vector_type &state, vector_type &rate )
    {
        if ( !problem_->evaluate_rate( state, rate ) )
        {
            status_ = single_step_status::failed_solver;
            return false;
        }
        if ( !operations_->is_valid_number( rate ) )
        {
            status_ = single_step_status::failed_nonfinite;
            return false;
        }
        // Store the projected RHS for every consumer, including dense-only
        // stages, embedded defects and FSAL. The identity path adds no work.
        if constexpr ( projects_stage_rates )
        {
            if ( !projection_operator_->apply( rate ) )
            {
                status_ = single_step_status::failed_projection;
                return false;
            }
            if ( !operations_->is_valid_number( rate ) )
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
