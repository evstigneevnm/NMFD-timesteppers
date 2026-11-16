#ifndef NMFD_TIME_STEPPERS_DETAIL_APPLY_OPERATOR_H
#define NMFD_TIME_STEPPERS_DETAIL_APPLY_OPERATOR_H

#include <type_traits>
#include <utility>

namespace nmfd::time_steppers::detail
{
template <class Operator, class Vector, class = void>
struct operator_apply_traits : std::false_type
{
};

template <class Operator, class Vector>
struct operator_apply_traits<
    Operator, Vector, std::void_t<decltype( std::declval<Operator &>().apply( std::declval<Vector &>() ) )>>
    : std::bool_constant<
          std::is_same_v<decltype( std::declval<Operator &>().apply( std::declval<Vector &>() ) ), void> ||
          std::is_same_v<decltype( std::declval<Operator &>().apply( std::declval<Vector &>() ) ), bool>>
{
    using result_type = decltype( std::declval<Operator &>().apply( std::declval<Vector &>() ) );
};

// Dispatch only the return type; Boolean success is still a runtime result.
// The unsupported branch avoids secondary errors for a missing apply method.
template <class Operator, class Vector>
bool apply_operator( Operator &op, Vector &value )
{
    using traits = operator_apply_traits<Operator, Vector>;
    static_assert( traits::value, "Operator apply(vector&) must return void or bool" );
    if constexpr ( traits::value )
    {
        if constexpr ( std::is_same_v<typename traits::result_type, bool> )
            return op.apply( value );
        else
        {
            op.apply( value );
            return true;
        }
    }
    return false;
}
}
#endif
