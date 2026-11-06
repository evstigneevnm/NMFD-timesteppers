#ifndef NMFD_TIME_STEPPERS_IMEX_TIME_STEP_H
#define NMFD_TIME_STEPPERS_IMEX_TIME_STEP_H

#include <algorithm>
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
#include <nmfd/time_steppers/runge_kutta/butcher_tables.h>
#include <nmfd/time_steppers/runge_kutta/continuous_integration.h>
#include <nmfd/time_steppers/runge_kutta/imex_operator.h>

namespace nmfd::time_steppers::runge_kutta
{
namespace detail
{
template <class VectorOperations, bool Enabled>
struct imex_dense_storage
{
    explicit imex_dense_storage( const VectorOperations & )
    {
    }
};

template <class VectorOperations>
struct imex_dense_storage<VectorOperations, true>
{
    nmfd::detail::vector_wrap<VectorOperations, false, false> previous;
    explicit imex_dense_storage( const VectorOperations &operations ) : previous( operations )
    {
    }
};
}

template <
    class VectorOperations, class Problem, class TimeStepAdaptation, class NonlinearSolver, bool DenseOutput = false>
class imex_time_step
{
public:
    using scalar_type            = typename VectorOperations::scalar_type;
    using vector_type            = typename VectorOperations::vector_type;
    using imex_operator_type     = typename decltype( std::declval<Problem &>().get_imex_operator() )::element_type;
    using explicit_operator_type = typename imex_operator_type::explicit_operator_type;
    using implicit_operator_type = typename imex_operator_type::implicit_operator_type;

    struct params
    {
        std::string  method           = "IMEX_EULER";
        unsigned int maximum_attempts = 32;
        scalar_type  reference_time   = scalar_type( 1 );
    };

    static_assert(
        nmfd::time_steppers::detail::has_rate_evaluation<explicit_operator_type, vector_type>::value,
        "IMEX explicit partition requires bool evaluate_rate(state, rate)"
    );
    static_assert(
        nmfd::time_steppers::detail::has_implicit_stage_interface<
            implicit_operator_type, vector_type, scalar_type>::value,
        "IMEX implicit partition requires the complete shifted stage interface"
    );

    imex_time_step(
        std::shared_ptr<VectorOperations> operations, std::shared_ptr<Problem> problem,
        std::shared_ptr<TimeStepAdaptation> adaptation, std::shared_ptr<NonlinearSolver> solver, const params &p = {}
    )
        : imex_time_step(
              std::move( operations ), std::move( problem ), std::move( adaptation ), std::move( solver ),
              make_imex_butcher_table( p.method ), p
          )
    {
    }

    // Cache the partition bundle and bind the solver once; allocate only consumed rates.
    imex_time_step(
        std::shared_ptr<VectorOperations> operations, std::shared_ptr<Problem> problem,
        std::shared_ptr<TimeStepAdaptation> adaptation, std::shared_ptr<NonlinearSolver> solver,
        imex_butcher_table table, const params &p = {}
    )
        : operations_(
              operations ? std::move( operations ) : throw std::invalid_argument( "IMEX requires vector operations" )
          ),
          partitions_(
              problem ? problem->get_imex_operator() : throw std::invalid_argument( "IMEX requires a problem" )
          ),
          adaptation_(
              adaptation ? std::move( adaptation ) : throw std::invalid_argument( "IMEX requires time adaptation" )
          ),
          solver_( solver ? std::move( solver ) : throw std::invalid_argument( "IMEX requires a nonlinear solver" ) ),
          params_( p ), table_(
                            std::move( table.explicit_table ), std::move( table.implicit_table ), table.order,
                            table.embedded_order(), table.dense_type()
                        ),
          work_( *operations_ ), error_( *operations_ ), dense_( *operations_ )
    {
        if ( !partitions_ || !partitions_->explicit_part || !partitions_->implicit_part )
            throw std::invalid_argument( "IMEX requires both partition operators" );
        if ( !p.maximum_attempts )
            throw std::invalid_argument( "Empty IMEX attempt budget" );
        nmfd::time_steppers::detail::balanced_affine_shift( scalar_type( 0 ), p.reference_time );
        if constexpr ( DenseOutput )
        {
            if ( !table_.has_dense_output() )
                throw std::invalid_argument( "IMEX requires a validated combined dense extension" );
            if constexpr ( !nmfd::time_steppers::detail::has_rate_evaluation<
                               implicit_operator_type, vector_type>::value )
                throw std::invalid_argument( "IMEX dense output requires both partition rate providers" );
        }
        if constexpr ( nmfd::time_steppers::detail::has_embedded_assessment<
                           TimeStepAdaptation, vector_type, scalar_type>::value )
            estimate_error_ = adaptation_->requires_error_estimate() && table_.is_embedded();
        const auto &e     = table_.explicit_table;
        const auto &i     = table_.implicit_table;
        stiffly_accurate_ = true;
        for ( std::size_t j = 0; j < e.size(); ++j )
            stiffly_accurate_ =
                stiffly_accurate_ && e.a( e.size() - 1, j ) == e.b( j ) && i.a( i.size() - 1, j ) == i.b( j );
        for ( std::size_t j = 0; j < i.size(); ++j )
            if ( i.a( j, j ) == 0 && needs_rate( i, j ) &&
                 !nmfd::time_steppers::detail::has_rate_evaluation<implicit_operator_type, vector_type>::value )
                throw std::invalid_argument( "Consumed zero-diagonal implicit stages require evaluate_rate()" );
        work_.init();
        work_.start_use();
        if ( estimate_error_ )
        {
            error_.init();
            error_.start_use();
        }
        allocate_rates( e, explicit_rates_ );
        allocate_rates( i, implicit_rates_ );
        if constexpr ( DenseOutput )
        {
            dense_.previous.init();
            dense_.previous.start_use();
        }
        solver_->set_nonlinear_operator( partitions_->implicit_part );
    }

    void set_time( scalar_type time )
    {
        require_finalized();
        if ( !std::isfinite( time ) )
            throw std::invalid_argument( "Nonfinite IMEX time" );
        time_ = time;
    }
    void set_target_time( scalar_type target )
    {
        require_finalized();
        if ( !std::isfinite( target ) )
            throw std::invalid_argument( "Nonfinite IMEX target" );
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
    const imex_butcher_table &table() const
    {
        return table_;
    }
    unsigned int dense_output_order() const
    {
        return DenseOutput ? table_.dense_order() : 0;
    }

    // Borrow the pending combined solution until finalize(), reset(), or apply().
    template <bool Enabled = DenseOutput, std::enable_if_t<Enabled && DenseOutput, int> = 0>
    continuous_integration<imex_time_step> get_continuous_integration() const
    {
        if ( !pending_ )
            throw std::logic_error( "No pending IMEX dense output" );
        return { *this, generation_ };
    }

    void reset()
    {
        adaptation_->reset();
        ++generation_;
        initialized_ = pending_ = false;
        attempts_               = 0;
        dt_                     = 0;
        status_                 = single_step_status::idle;
    }

    // Publish only an assessed candidate. Failed stages retry from the same input.
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
        if ( !valid( in ) )
            return;
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
                pending_ = true;
                status_  = single_step_status::converged;
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
            throw std::logic_error( "No IMEX candidate to finalize" );
        const auto endpoint = endpoint_time();
        if ( !std::isfinite( committed_time ) || ( dt_ > 0 ? committed_time < time_ || committed_time > endpoint
                                                           : committed_time > time_ || committed_time < endpoint ) )
            throw std::invalid_argument( "Committed time is outside the pending IMEX step" );
        adaptation_->update( outcome, committed_time, committed_time - time_, state );
        pending_ = false;
    }

private:
    using vector_wrap_type = nmfd::detail::vector_wrap<VectorOperations, false, false>;
    std::shared_ptr<VectorOperations>                         operations_;
    std::shared_ptr<imex_operator_type>                       partitions_;
    std::shared_ptr<TimeStepAdaptation>                       adaptation_;
    std::shared_ptr<NonlinearSolver>                          solver_;
    params                                                    params_;
    imex_butcher_table                                        table_;
    vector_wrap_type                                          work_, error_;
    std::vector<vector_wrap_type>                             explicit_rates_, implicit_rates_;
    detail::imex_dense_storage<VectorOperations, DenseOutput> dense_;
    std::size_t                                               generation_ = 0;
    scalar_type                                               time_ = 0, target_ = 1, dt_ = 0;
    unsigned int                                              attempts_ = 0;
    bool               stiffly_accurate_ = false, estimate_error_ = false, initialized_ = false, pending_ = false;
    single_step_status status_ = single_step_status::idle;

    friend class continuous_integration<imex_time_step>;

    scalar_type endpoint_time() const
    {
        return dt_ == target_ - time_ ? target_ : time_ + dt_;
    }

    void require_finalized() const
    {
        if ( pending_ )
            throw std::logic_error( "Finalize the pending IMEX step first" );
    }
    bool needs_rate( const butcher_table &table, std::size_t stage ) const
    {
        if constexpr ( DenseOutput )
            if ( stage == 0 )
                return true;
        if ( ( !stiffly_accurate_ && table.b( stage ) != 0 ) || ( estimate_error_ && table.error_b( stage ) != 0 ) )
            return true;
        for ( std::size_t j = stage + 1; j < table.size(); ++j )
            if ( table.a( j, stage ) != 0 )
                return true;
        return false;
    }
    void allocate_rates( const butcher_table &table, std::vector<vector_wrap_type> &rates )
    {
        rates.reserve( table.size() );
        for ( std::size_t j = 0; j < table.size(); ++j )
        {
            rates.emplace_back( *operations_ );
            if ( needs_rate( table, j ) || ( DenseOutput && j + 1 == table.size() ) )
            {
                rates.back().init();
                rates.back().start_use();
            }
        }
    }
    bool valid( const vector_type &state )
    {
        if ( operations_->is_valid_number( state ) )
            return true;
        status_ = single_step_status::failed_nonfinite;
        return false;
    }
    adaptation_status assess( const vector_type &in, scalar_type &next_dt )
    {
        if constexpr ( nmfd::time_steppers::detail::has_embedded_assessment<
                           TimeStepAdaptation, vector_type, scalar_type>::value )
            if ( estimate_error_ )
                return adaptation_->assess( time_, dt_, in, *work_, next_dt, table_.error_order(), &*error_ );
        return adaptation_->assess( time_, dt_, in, *work_, next_dt );
    }
    template <class Partition>
    bool evaluate_rate( Partition &partition, const vector_type &state, vector_type &rate )
    {
        if ( !partition.evaluate_rate( state, rate ) )
        {
            status_ = single_step_status::failed_solver;
            return false;
        }
        return valid( rate );
    }

    // Only accepted candidates need fresh rates at the actual combined endpoint.
    // Retired stage buffers become total initial/endpoint rates; no history is kept.
    bool prepare_dense( const vector_type &in )
    {
        static_assert( DenseOutput, "Dense output is disabled" );
        auto      &e = *partitions_->explicit_part;
        auto      &i = *partitions_->implicit_part;
        const auto restore_time =
            time_ +
            dt_ * scalar_type( stiffly_accurate_ ? 1 : table_.explicit_table.c( table_.explicit_table.size() - 1 ) );
        const auto restore = [&]() {
            if constexpr ( nmfd::time_steppers::detail::has_set_time<explicit_operator_type, scalar_type>::value )
                e.set_time( restore_time );
            i.set_time( restore_time );
        };
        bool ok = false;
        try
        {
            if constexpr ( nmfd::time_steppers::detail::has_set_time<explicit_operator_type, scalar_type>::value )
                e.set_time( endpoint_time() );
            i.set_time( endpoint_time() );
            if constexpr ( nmfd::time_steppers::detail::has_rate_evaluation<
                               implicit_operator_type, vector_type>::value )
                ok = evaluate_rate( e, *work_, *explicit_rates_.back() ) &&
                     evaluate_rate( i, *work_, *implicit_rates_.back() );
        }
        catch ( ... )
        {
            restore();
            throw;
        }
        restore();
        if ( !ok )
            return false;
        operations_->add_lin_comb( scalar_type( 1 ), *implicit_rates_.front(), *explicit_rates_.front() );
        operations_->add_lin_comb( scalar_type( 1 ), *implicit_rates_.back(), *explicit_rates_.back() );
        if ( !valid( *explicit_rates_.front() ) || !valid( *explicit_rates_.back() ) )
            return false;
        operations_->assign( in, *dense_.previous );
        return true;
    }

    // Sampling uses stored backend vectors only; caller-owned outputs may alias inputs.
    scalar_type evaluate_dense( scalar_type theta, vector_type &out, std::size_t generation ) const
    {
        static_assert( DenseOutput, "Dense output is disabled" );
        if ( !pending_ || generation != generation_ )
            throw std::logic_error( "Expired IMEX dense output" );
        if ( !std::isfinite( theta ) || theta < 0 || theta > 1 )
            throw std::invalid_argument( "Dense-output theta must be in [0,1]" );
        if ( theta == 0 )
            operations_->assign( *dense_.previous, out );
        else if ( theta == 1 )
            operations_->assign( *work_, out );
        else
        {
            const auto weights = table_.dense_weights( theta );
            operations_->assign_lin_comb(
                scalar_type( weights.initial_state ), *dense_.previous, scalar_type( weights.final_state ), *work_, out
            );
            operations_->add_lin_comb( dt_ * scalar_type( weights.initial_rate ), *explicit_rates_.front(), out );
            operations_->add_lin_comb( dt_ * scalar_type( weights.final_rate ), *explicit_rates_.back(), out );
        }
        return theta == 1 ? endpoint_time() : time_ + theta * dt_;
    }

    // The reference contains BOTH known rate sums; the shifted residual adds only f_I.
    bool compute( const vector_type &in )
    {
        auto       &explicit_part = *partitions_->explicit_part;
        auto       &implicit_part = *partitions_->implicit_part;
        auto       &previous      = implicit_part.previous_state();
        const auto &e             = table_.explicit_table;
        const auto &i             = table_.implicit_table;
        for ( std::size_t stage = 0; stage < e.size(); ++stage )
        {
            operations_->assign( in, previous );
            operations_->assign_scalar( scalar_type( 0 ), implicit_part.known_rhs() );
            for ( std::size_t j = 0; j < stage; ++j )
            {
                if ( e.a( stage, j ) != 0 )
                    operations_->add_lin_comb( dt_ * scalar_type( e.a( stage, j ) ), *explicit_rates_[j], previous );
                if ( i.a( stage, j ) != 0 )
                    operations_->add_lin_comb( dt_ * scalar_type( i.a( stage, j ) ), *implicit_rates_[j], previous );
            }
            const auto stage_time =
                time_ + dt_ * scalar_type( stiffly_accurate_ && stage + 1 == e.size() ? 1 : e.c( stage ) );
            if ( !std::isfinite( stage_time ) || !valid( previous ) )
            {
                status_ = single_step_status::failed_nonfinite;
                return false;
            }
            if constexpr ( nmfd::time_steppers::detail::has_set_time<explicit_operator_type, scalar_type>::value )
                explicit_part.set_time( stage_time );
            implicit_part.set_time( stage_time );
            operations_->assign( previous, *work_ );
            if ( i.a( stage, stage ) != 0 )
            {
                const auto alpha = scalar_type( 1 ) / ( dt_ * scalar_type( i.a( stage, stage ) ) );
                if ( !std::isfinite( alpha ) )
                {
                    status_ = single_step_status::failed_nonfinite;
                    return false;
                }
                const auto shift = nmfd::time_steppers::detail::balanced_affine_shift( alpha, params_.reference_time );
                implicit_part.set_affine_shift( shift.a, shift.b );
                if ( !solver_->solve( *work_ ) )
                {
                    status_ = single_step_status::failed_solver;
                    return false;
                }
                if ( !valid( *work_ ) )
                    return false;
                if ( needs_rate( i, stage ) )
                {
                    operations_->assign_lin_comb(
                        scalar_type( 1 ), *work_, scalar_type( -1 ), previous, *implicit_rates_[stage]
                    );
                    operations_->scale( alpha, *implicit_rates_[stage] );
                    if ( !valid( *implicit_rates_[stage] ) )
                        return false;
                }
            }
            else if constexpr ( nmfd::time_steppers::detail::has_rate_evaluation<
                                    implicit_operator_type, vector_type>::value )
            {
                if ( needs_rate( i, stage ) && !evaluate_rate( implicit_part, *work_, *implicit_rates_[stage] ) )
                    return false;
            }
            if ( needs_rate( e, stage ) && !evaluate_rate( explicit_part, *work_, *explicit_rates_[stage] ) )
                return false;
        }
        if ( !stiffly_accurate_ )
        {
            operations_->assign( in, *work_ );
            for ( std::size_t j = 0; j < e.size(); ++j )
            {
                if ( e.b( j ) != 0 )
                    operations_->add_lin_comb( dt_ * scalar_type( e.b( j ) ), *explicit_rates_[j], *work_ );
                if ( i.b( j ) != 0 )
                    operations_->add_lin_comb( dt_ * scalar_type( i.b( j ) ), *implicit_rates_[j], *work_ );
            }
        }
        if ( !valid( *work_ ) )
            return false;
        if ( estimate_error_ )
        {
            // Combine signed state defects before applying the controller's norm/scaling.
            operations_->assign_scalar( scalar_type( 0 ), *error_ );
            for ( std::size_t j = 0; j < e.size(); ++j )
            {
                if ( e.error_b( j ) != 0 )
                    operations_->add_lin_comb( dt_ * scalar_type( e.error_b( j ) ), *explicit_rates_[j], *error_ );
                if ( i.error_b( j ) != 0 )
                    operations_->add_lin_comb( dt_ * scalar_type( i.error_b( j ) ), *implicit_rates_[j], *error_ );
            }
            if ( !valid( *error_ ) )
                return false;
        }
        return true;
    }
};
}
#endif
