#ifndef NMFD_TIME_STEPPERS_ADAPTATION_GLOBALIZATION_H
#define NMFD_TIME_STEPPERS_ADAPTATION_GLOBALIZATION_H

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <nmfd/time_steppers/detail/status.h>

namespace nmfd
{
namespace time_steppers
{
namespace integration
{
// Pseudo-time relaxation, not physical-time accuracy control. MeritMeasure(state)
// evaluates an unscaled steady residual with fixed physical parameters/time.
template <class VectorOperations, class MeritMeasure>
class time_step_adaptation_globalization
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    struct params
    {
        scalar_type initial_step         = scalar_type( .01 );
        scalar_type minimum_step         = scalar_type( 1e-14 );
        scalar_type maximum_step         = scalar_type( 1e6 );
        scalar_type residual_power       = scalar_type( 1 );
        scalar_type minimum_factor       = scalar_type( .5 );
        scalar_type maximum_factor       = scalar_type( 2 );
        scalar_type rejection_factor     = scalar_type( .5 );
        scalar_type maximum_merit_growth = scalar_type( 0 ); // Zero disables growth-based rejection.
    };

    explicit time_step_adaptation_globalization( std::shared_ptr<MeritMeasure> merit, const params &p = {} )
        : merit_( std::move( merit ) ), params_( p )
    {
        if ( !merit_ )
            throw std::invalid_argument( "Globalization requires a steady-residual merit measure" );
        const scalar_type values[] = { p.initial_step,   p.minimum_step,   p.maximum_step,    p.residual_power,
                                       p.minimum_factor, p.maximum_factor, p.rejection_factor };
        for ( const auto value : values )
            if ( !std::isfinite( value ) || value <= 0 )
                throw std::invalid_argument( "Globalization parameters must be positive and finite" );
        if ( p.minimum_step > p.maximum_step || p.minimum_factor > 1 || p.maximum_factor < 1 ||
             p.rejection_factor >= 1 || !std::isfinite( p.maximum_merit_growth ) || p.maximum_merit_growth < 0 ||
             ( p.maximum_merit_growth > 0 && p.maximum_merit_growth < 1 ) )
            throw std::invalid_argument( "Invalid globalization step, factor or merit bounds" );
        reset();
    }

    scalar_type initialize( scalar_type time, const vector_type &, scalar_type dt = scalar_type( -1 ) )
    {
        if ( !std::isfinite( time ) || !std::isfinite( dt ) || ( dt != scalar_type( -1 ) && dt <= 0 ) )
            throw std::invalid_argument( "Invalid globalization initialization time or step" );
        reset();
        if ( dt > 0 )
            dt_ = std::clamp( dt, params_.minimum_step, params_.maximum_step );
        return dt_;
    }

    void reset()
    {
        dt_       = std::clamp( params_.initial_step, params_.minimum_step, params_.maximum_step );
        has_base_ = pending_ = rejected_ = false;
    }

    scalar_type get_dt() const
    {
        return dt_;
    }

    bool requires_error_estimate() const
    {
        return false;
    }

    // Assess only finite, successfully solved candidates. An embedded error, if
    // supplied by a caller, is irrelevant to steady-residual globalization.
    adaptation_status assess(
        scalar_type time, scalar_type dt_used, const vector_type &in, const vector_type &out, scalar_type &dt_next,
        unsigned int = 0, const vector_type * = nullptr
    )
    {
        pending_ = false;
        dt_next  = dt_;
        if ( !std::isfinite( time ) || !std::isfinite( dt_used ) || dt_used <= 0 || !std::isfinite( time + dt_used ) ||
             time + dt_used == time )
            return adaptation_status::failed;
        if ( !has_base_ || time != base_time_ )
        {
            base_merit_ = ( *merit_ )( in );
            base_time_  = time;
            has_base_   = true;
        }
        if ( !valid_merit( base_merit_ ) )
            return adaptation_status::failed;
        trial_merit_ = ( *merit_ )( out );
        if ( !valid_merit( trial_merit_ ) ||
             ( params_.maximum_merit_growth > 0 && trial_merit_ > base_merit_ &&
               ( base_merit_ == 0 || trial_merit_ / params_.maximum_merit_growth > base_merit_ ) ) )
        {
            const auto decision = reject_step( dt_used );
            dt_next             = dt_;
            return decision;
        }

        scalar_type factor = 1;
        if ( trial_merit_ == 0 )
        {
            if ( base_merit_ > 0 )
                factor = params_.maximum_factor;
        }
        else if ( base_merit_ == 0 )
            factor = params_.minimum_factor;
        else
        {
            // Bound logarithms before exponentiation to avoid overflowing ratios.
            factor = std::exp(
                std::clamp(
                    params_.residual_power * ( std::log( base_merit_ ) - std::log( trial_merit_ ) ),
                    std::log( params_.minimum_factor ), std::log( params_.maximum_factor )
                )
            );
        }
        factor = std::clamp( factor, params_.minimum_factor, params_.maximum_factor );
        if ( rejected_ )
            factor = std::min( scalar_type( 1 ), factor );
        dt_ = std::clamp(
            std::min( dt_used, params_.maximum_step / factor ) * factor, params_.minimum_step, params_.maximum_step
        );
        dt_next       = dt_;
        pending_time_ = time + dt_used;
        pending_step_ = dt_used;
        pending_      = true;
        return adaptation_status::accepted;
    }

    // Promote the cached merit only for an unchanged committed endpoint. A
    // modified endpoint is reassessed lazily; no candidate vector is retained.
    void update( adaptation_status outcome, scalar_type time, scalar_type h, const vector_type & )
    {
        if ( outcome == adaptation_status::accepted && pending_ && time == pending_time_ && std::isfinite( h ) &&
             h > 0 )
        {
            base_merit_ = trial_merit_;
            base_time_  = time;
            has_base_   = true;
        }
        else if ( outcome == adaptation_status::accepted || outcome == adaptation_status::accepted_modified )
        {
            has_base_ = false;
            if ( std::isfinite( h ) && h > 0 )
                dt_ = std::clamp( std::min( dt_, h ), params_.minimum_step, params_.maximum_step );
        }
        else if ( outcome == adaptation_status::failed && pending_ )
            dt_ = std::min( dt_, pending_step_ );
        pending_ = false;
        if ( outcome != adaptation_status::rejected )
            rejected_ = false;
    }

    // Numerical stage failure and merit rejection share the same bounded retry.
    adaptation_status reject_step( scalar_type dt_used )
    {
        pending_        = false;
        rejected_       = true;
        const auto next = dt_used * params_.rejection_factor;
        if ( !std::isfinite( dt_used ) || dt_used <= 0 || next < params_.minimum_step || next >= dt_used )
            return adaptation_status::failed;
        dt_ = std::min( next, params_.maximum_step );
        return adaptation_status::rejected;
    }

private:
    std::shared_ptr<MeritMeasure> merit_;
    params                        params_;
    scalar_type                   dt_ = 0, base_merit_ = 0, trial_merit_ = 0, base_time_ = 0;
    scalar_type                   pending_time_ = 0, pending_step_ = 0;
    bool                          has_base_ = false, pending_ = false, rejected_ = false;

    static bool valid_merit( scalar_type value )
    {
        return std::isfinite( value ) && value >= 0;
    }
};
}
}
}

#endif
