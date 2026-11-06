#ifndef NMFD_TIME_STEPPERS_BDF_HISTORY_H
#define NMFD_TIME_STEPPERS_BDF_HISTORY_H

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/time_steppers/bdf/coefficients.h>

namespace nmfd::time_steppers::bdf
{
// Committed Newton differences D_k=h^k*[u_n,...,u_{n-k}], scaled by
// the last signed step. Scalar step history never leaves the host.
template <class VectorOperations>
class divided_difference_history
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    explicit divided_difference_history( const VectorOperations &operations ) : operations_( operations )
    {
    }
    void initialize( bool second )
    {
        initialize( second ? 2u : 1u );
    }
    void initialize( unsigned int capacity )
    {
        if ( capacity < 1 || capacity > 5 || capacity_ )
            throw std::invalid_argument( "Invalid BDF history capacity or repeated initialization" );
        capacity_ = capacity;
        for ( unsigned int i = 0; i < capacity; ++i )
            allocate( differences_[i] );
        if ( capacity > 2 )
            allocate( scratch_ );
    }
    void reset()
    {
        count_ = 0;
        steps_.fill( 0 );
    }
    unsigned int size() const
    {
        return count_;
    }
    scalar_type step() const
    {
        return steps_[0];
    }

    bool usable( scalar_type h, scalar_type minimum_ratio, scalar_type maximum_ratio ) const
    {
        if ( !count_ || std::signbit( h ) != std::signbit( step() ) )
            return false;
        const auto ratio    = h / step();
        const auto roundoff = scalar_type( 8 ) * std::numeric_limits<scalar_type>::epsilon();
        return std::isfinite( ratio ) && ratio >= minimum_ratio * ( 1 - roundoff ) &&
               ratio <= maximum_ratio * ( 1 + roundoff );
    }

    // Restrict higher orders to slowly changing recent grids. This is not a
    // stability-limit detector for weakly damped oscillatory modes.
    bool smooth( unsigned int order, scalar_type h ) const
    {
        scalar_type newer = h;
        for ( unsigned int i = 0; i < order; ++i )
        {
            const auto ratio = newer / steps_[i];
            if ( !std::isfinite( ratio ) || ratio < scalar_type( .8 ) ||
                 ratio > scalar_type( 1.2 ) * ( 1 + scalar_type( 16 ) * std::numeric_limits<scalar_type>::epsilon() ) )
                return false;
            newer = steps_[i];
        }
        return true;
    }
    formula<scalar_type> coefficients( unsigned int order, scalar_type h ) const
    {
        if ( order > count_ )
            throw std::logic_error( "Insufficient BDF history" );
        return make_formula( order, h, steps_ );
    }
    void reference( scalar_type weight, const vector_type &in, vector_type &out ) const
    {
        operations_.assign_lin_comb( scalar_type( 1 ), in, weight, **differences_[0], out );
    }
    void reference( const formula<scalar_type> &c, const vector_type &in, vector_type &out ) const
    {
        operations_.assign( in, out );
        for ( unsigned int i = 0; i + 1 < c.order; ++i )
            operations_.add_lin_comb( c.reference[i], **differences_[i], out );
    }
    void predict( const formula<scalar_type> &c, const vector_type &in, vector_type &out ) const
    {
        operations_.assign( in, out );
        for ( unsigned int i = 0; i < c.order; ++i )
            operations_.add_lin_comb( c.predictor[i], **differences_[i], out );
    }

    // Candidate-corrected polynomial defect in state units:
    // e=(candidate-P_q)/(1+alpha*span), with span to the oldest node.
    void estimate(
        unsigned int order, scalar_type h, const vector_type &in, const vector_type &candidate, vector_type &error
    ) const
    {
        if ( order > 2 )
        {
            const auto c = coefficients( order, h );
            operations_.assign_lin_comb( scalar_type( 1 ), candidate, scalar_type( -1 ), in, error );
            for ( unsigned int i = 0; i < order; ++i )
                operations_.add_lin_comb( -c.predictor[i], **differences_[i], error );
            operations_.scale( c.error_weight, error );
            return;
        }
        const auto r = h / step();
        operations_.assign_lin_comb( scalar_type( 1 ), candidate, scalar_type( -1 ), in, error );
        operations_.add_lin_comb( -r, **differences_[0], error );
        operations_.scale( r / ( 1 + r ), error );
        if ( order == 1 )
            operations_.scale( ( 1 + r ) / ( 1 + 2 * r ), error );
        else
        {
            const auto s      = steps_[1] / step();
            const auto weight = ( 1 + r ) * ( 1 + r ) / ( ( 1 + 2 * r ) * ( 1 + r + s ) + r * ( 1 + r ) );
            operations_.add_lin_comb( -r * r, **differences_[1], error );
            operations_.scale( weight, error );
        }
    }

    // O(qN) update. Rotate scratch through the differences, consuming each old
    // D_k before reusing its storage. One/two-difference histories need no scratch.
    void commit( const vector_type &in, const vector_type &out, scalar_type h, bool restart )
    {
        if ( restart )
            reset();
        if ( capacity_ > 2 )
        {
            operations_.assign_lin_comb( scalar_type( 1 ), out, scalar_type( -1 ), in, **scratch_ );
            const auto  valid = std::min( capacity_, count_ + 1 );
            scalar_type span = h, power = 1;
            for ( unsigned int i = 0; i < valid; ++i )
            {
                if ( i + 1 < valid )
                {
                    span += steps_[i];
                    power *= h / step();
                    operations_.assign_lin_comb(
                        h / span, **scratch_, -power * h / span, **differences_[i], **differences_[i]
                    );
                }
                scratch_.swap( differences_[i] );
            }
        }
        else
        {
            if ( capacity_ == 2 && count_ )
            {
                const auto ratio = h / step();
                operations_.assign_lin_comb( scalar_type( 1 ), out, scalar_type( -1 ), in, **differences_[1] );
                operations_.add_lin_comb( -ratio, **differences_[0], **differences_[1] );
                operations_.scale( ratio / ( 1 + ratio ), **differences_[1] );
            }
            operations_.assign_lin_comb( scalar_type( 1 ), out, scalar_type( -1 ), in, **differences_[0] );
        }
        for ( unsigned int i = 4; i > 0; --i )
            steps_[i] = steps_[i - 1];
        steps_[0] = h;
        count_    = std::min( capacity_, count_ + 1 );
    }

private:
    using buffer = nmfd::detail::vector_wrap<VectorOperations, false, false>;
    const VectorOperations                &operations_;
    std::array<std::unique_ptr<buffer>, 5> differences_;
    std::unique_ptr<buffer>                scratch_;
    std::array<scalar_type, 5>             steps_{};
    unsigned int                           count_ = 0, capacity_ = 0;
    void                                   allocate( std::unique_ptr<buffer> &slot )
    {
        slot = std::make_unique<buffer>( operations_ );
        slot->init();
        slot->start_use();
    }
};
}
#endif
