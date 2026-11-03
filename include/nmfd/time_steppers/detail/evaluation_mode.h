#ifndef NMFD_TIME_STEPPERS_EVALUATION_MODE_H
#define NMFD_TIME_STEPPERS_EVALUATION_MODE_H

namespace nmfd
{
namespace time_steppers
{
// The optional three-argument problem.apply() produces a physical, consistent
// derivative. The existing two-argument overload remains the stage residual.
enum class evaluation_mode
{
    rate
};
}
}
#endif
