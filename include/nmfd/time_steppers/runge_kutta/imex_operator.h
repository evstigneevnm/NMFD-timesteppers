#ifndef NMFD_TIME_STEPPERS_IMEX_OPERATOR_H
#define NMFD_TIME_STEPPERS_IMEX_OPERATOR_H

#include <memory>
#include <stdexcept>
#include <utility>

namespace nmfd::time_steppers::runge_kutta
{
// Assembly retains separate partition contexts; neither handle is rebound per stage.
template <class ExplicitOperator, class ImplicitOperator>
struct imex_operator
{
    using explicit_operator_type = ExplicitOperator;
    using implicit_operator_type = ImplicitOperator;

    const std::shared_ptr<ExplicitOperator> explicit_part;
    const std::shared_ptr<ImplicitOperator> implicit_part;

    imex_operator( std::shared_ptr<ExplicitOperator> explicit_op, std::shared_ptr<ImplicitOperator> implicit_op )
        : explicit_part( std::move( explicit_op ) ), implicit_part( std::move( implicit_op ) )
    {
        if ( !explicit_part || !implicit_part )
            throw std::invalid_argument( "IMEX requires both partition operators" );
    }
};
}
#endif
