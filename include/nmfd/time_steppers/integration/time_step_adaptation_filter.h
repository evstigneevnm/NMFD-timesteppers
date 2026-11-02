#ifndef NMFD_TIME_STEPPERS_ADAPTATION_FILTER_H
#define NMFD_TIME_STEPPERS_ADAPTATION_FILTER_H

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <nmfd/time_steppers/detail/scipy_error_mapping.h>
#include <nmfd/time_steppers/detail/status.h>

namespace nmfd
{
namespace time_steppers
{
namespace integration
{
template <class VectorOperations>
class time_step_adaptation_filter
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    static_assert( std::is_floating_point_v<scalar_type>, "Time adaptation requires a real floating-point scalar" );

    struct params
    {
        scalar_type relative_tolerance = scalar_type( 1e-8 );
        scalar_type absolute_tolerance = scalar_type( 1e-10 );
        scalar_type initial_step       = scalar_type( .01 );
        scalar_type minimum_step       = scalar_type( 1e-14 );
        scalar_type maximum_step       = scalar_type( 1 );
        std::string controller         = "PID";
    };

    time_step_adaptation_filter( std::shared_ptr<VectorOperations> operations, const params &p = {} )
        : operations_( std::move( operations ) ), params_( p ), coefficients_( select_filter( p.controller ) )
    {
        if ( !operations_ )
            throw std::invalid_argument( "Adaptation requires vector operations" );
        const scalar_type values[] = {
            p.relative_tolerance, p.absolute_tolerance, p.initial_step, p.minimum_step, p.maximum_step
        };
        for ( const auto value : values )
            if ( !std::isfinite( value ) || value <= 0 )
                throw std::invalid_argument( "Adaptive-step parameters must be positive and finite" );
        if ( p.minimum_step > p.maximum_step || operations_->get_vector_size() == 0 )
            throw std::invalid_argument( "Inverted adaptive-step bounds or empty vector space" );
        params_.relative_tolerance =
            std::max( p.relative_tolerance, scalar_type( 100 ) * std::numeric_limits<scalar_type>::epsilon() );
        reset();
    }

    scalar_type initialize( scalar_type time, const vector_type &, scalar_type dt = scalar_type( -1 ) )
    {
        if ( !std::isfinite( time ) || !std::isfinite( dt ) || ( dt != scalar_type( -1 ) && dt <= 0 ) )
            throw std::invalid_argument( "Invalid adaptation initialization time or step" );
        reset();
        if ( dt > 0 )
            dt_ = std::clamp( dt, params_.minimum_step, params_.maximum_step );
        return dt_;
    }

    void reset()
    {
        dt_           = std::clamp( params_.initial_step, params_.minimum_step, params_.maximum_step );
        history_size_ = 0;
        pending_ = rejected_ = false;
    }

    scalar_type get_dt() const
    {
        return dt_;
    }

    scalar_type relative_tolerance() const
    {
        return params_.relative_tolerance;
    }

    bool requires_error_estimate() const
    {
        return true;
    }

    adaptation_status assess(
        scalar_type time, scalar_type dt_used, const vector_type &in, const vector_type &out, scalar_type &dt_next,
        unsigned int error_order = 0, const vector_type *error = nullptr
    )
    {
        pending_ = false;
        dt_next  = dt_;
        if ( !std::isfinite( time ) || !std::isfinite( dt_used ) || dt_used == 0 || !error || !error_order ||
             !std::isfinite( time + dt_used ) )
            return adaptation_status::failed;
        if ( history_size_ && ( time != committed_time_ || error_order != order_ || ( dt_used > 0 ) != forward_ ) )
            history_size_ = 0;

        const detail::scipy_error_square_mapping<scalar_type> mapping{
            params_.absolute_tolerance, params_.relative_tolerance
        };
        const auto square = operations_->transform_reduce_sum( mapping, *error, in, out );
        const auto ratio  = std::sqrt( square / static_cast<scalar_type>( operations_->get_vector_size() ) );
        if ( !std::isfinite( ratio ) || ratio >= 1 )
        {
            // Rejected trials never enter the filter. Retry with bounded I control.
            const auto factor =
                !std::isfinite( ratio )
                    ? scalar_type( .2 )
                    : std::max(
                          scalar_type( .2 ), scalar_type( .9 ) * std::pow( ratio, -scalar_type( 1 ) / error_order )
                      );
            const auto decision = reject_with_step( std::abs( dt_used ) * factor );
            dt_next             = dt_;
            return decision;
        }

        trial_ = {
            std::log( std::max( ratio, std::numeric_limits<scalar_type>::min() ) ), std::log( std::abs( dt_used ) )
        };
        auto growth = ratio == 0 ? scalar_type( 10 ) : filtered_growth( error_order );
        if ( rejected_ )
            growth = std::min( scalar_type( 1 ), growth );
        dt_           = std::clamp( std::abs( dt_used ) * growth, params_.minimum_step, params_.maximum_step );
        dt_next       = dt_;
        pending_      = true;
        pending_time_ = time + dt_used;
        order_        = error_order;
        forward_      = dt_used > 0;
        return adaptation_status::accepted;
    }

    void update( adaptation_status outcome, scalar_type time, scalar_type h, const vector_type & )
    {
        if ( outcome == adaptation_status::accepted && pending_ && time == pending_time_ && std::isfinite( h ) &&
             h != 0 && ( h > 0 ) == forward_ )
        {
            history_[1]     = history_[0];
            history_[0]     = trial_;
            history_size_   = std::min( 2u, history_size_ + 1 );
            committed_time_ = time;
        }
        else if ( outcome == adaptation_status::accepted || outcome == adaptation_status::accepted_modified )
        {
            // The assessed error no longer describes the externally committed endpoint.
            history_size_ = 0;
            if ( std::isfinite( h ) && h != 0 )
                dt_ = std::clamp( std::min( dt_, std::abs( h ) ), params_.minimum_step, params_.maximum_step );
        }
        pending_ = false;
        if ( outcome != adaptation_status::rejected )
            rejected_ = false;
    }

    adaptation_status reject_step( scalar_type dt_used )
    {
        return reject_with_step( std::abs( dt_used ) * scalar_type( .2 ) );
    }

private:
    struct coefficients
    {
        scalar_type  alpha, beta, gamma, a, b;
        unsigned int history;
    };
    struct sample
    {
        scalar_type log_error = 0, log_step = 0;
    };
    std::shared_ptr<VectorOperations> operations_;
    params                            params_;
    coefficients                      coefficients_;
    std::array<sample, 2>             history_{};
    sample                            trial_;
    scalar_type                       dt_ = 0, pending_time_ = 0, committed_time_ = 0;
    unsigned int                      history_size_ = 0, order_ = 0;
    bool                              pending_ = false, rejected_ = false, forward_ = true;

    // Kennedy & Carpenter, NASA/TM-2016-219173, Eq. 93 and Table 8.
    // Error coefficients are divided by the supplied state-error exponent, not the primary order.
    static coefficients select_filter( const std::string &name )
    {
        using T = scalar_type;
        if ( name == "I" )
            return { 1, 0, 0, 0, 0, 0 };
        if ( name == "H211" )
            return { T( 1 ) / 4, -T( 1 ) / 4, 0, -T( 1 ) / 4, 0, 1 };
        if ( name == "PC" )
            return { 2, 1, 0, 1, 0, 1 };
        if ( name == "PID" )
            return { T( 1 ) / 18, -T( 1 ) / 9, T( 1 ) / 18, 0, 0, 2 };
        if ( name == "H312" )
            return { T( 1 ) / 8, -T( 1 ) / 4, T( 1 ) / 8, -T( 3 ) / 8, -T( 1 ) / 8, 2 };
        if ( name == "PPID" )
            return { T( 6 ) / 20, -T( 1 ) / 20, -T( 5 ) / 20, 1, 0, 2 };
        if ( name == "H321" )
            return { T( 1 ) / 3, -T( 1 ) / 18, -T( 5 ) / 18, T( 5 ) / 6, T( 1 ) / 6, 2 };
        throw std::invalid_argument( "Unknown time-step filter: " + name );
    }

    scalar_type filtered_growth( unsigned int error_order ) const
    {
        // Until enough committed samples exist, use I control without fabricated history.
        const auto c = history_size_ < coefficients_.history ? coefficients{ 1, 0, 0, 0, 0, 0 } : coefficients_;
        auto       log_growth = std::log( scalar_type( .9 ) ) - c.alpha * trial_.log_error / error_order;
        if ( c.history > 0 )
            log_growth +=
                c.beta * history_[0].log_error / error_order + c.a * ( trial_.log_step - history_[0].log_step );
        if ( c.history > 1 )
            log_growth +=
                -c.gamma * history_[1].log_error / error_order + c.b * ( history_[0].log_step - history_[1].log_step );
        // Log-domain evaluation avoids intermediate powers and step-ratio overflow.
        return std::exp( std::clamp( log_growth, std::log( scalar_type( .2 ) ), std::log( scalar_type( 10 ) ) ) );
    }

    adaptation_status reject_with_step( scalar_type next )
    {
        pending_  = false;
        rejected_ = true;
        if ( !std::isfinite( next ) || next < params_.minimum_step )
            return adaptation_status::failed;
        dt_ = std::min( next, params_.maximum_step );
        return adaptation_status::rejected;
    }
};
}
}
}
#endif
