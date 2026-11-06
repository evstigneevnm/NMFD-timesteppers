#ifndef NMFD_TIME_STEPPERS_BDF_ORDER_POLICY_H
#define NMFD_TIME_STEPPERS_BDF_ORDER_POLICY_H

#include <algorithm>

namespace nmfd::time_steppers::bdf
{
// Error weighting and timestep proposals remain the controller's responsibility.
// Only a committed step may promote order; failures influence the current retry.
class order_policy
{
public:
    void reset()
    {
        requested_   = 1;
        clean_steps_ = 0;
    }
    unsigned int select( unsigned int history, unsigned int failures, bool restart ) const
    {
        if ( restart || failures >= 4 )
            return 1;
        const auto order = failures >= 2 ? std::max( 1u, requested_ - 1 ) : requested_;
        return std::min( order, std::max( 1u, history ) );
    }
    void commit( unsigned int order, unsigned int maximum, bool clean, bool restart )
    {
        if ( restart || !clean || order != requested_ )
            clean_steps_ = 0;
        requested_ = order;
        if ( clean && order < maximum && ++clean_steps_ >= std::max( 3u, order + 1 ) )
        {
            requested_   = order + 1;
            clean_steps_ = 0;
        }
    }

private:
    unsigned int requested_ = 1, clean_steps_ = 0;
};
}
#endif
