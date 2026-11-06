#ifndef NMFD_TIME_STEPPERS_PROBLEM_TRAITS_H
#define NMFD_TIME_STEPPERS_PROBLEM_TRAITS_H

#include <type_traits>
#include <utility>
#include <memory>

namespace nmfd::time_steppers::detail
{
template <class Problem, class T, class = void>
struct has_set_time : std::false_type
{
};

template <class Problem, class T>
struct has_set_time<Problem, T, std::void_t<decltype( std::declval<Problem &>().set_time( std::declval<T>() ) )>>
    : std::true_type
{
};

template <class Problem, class Vector, class = void>
struct has_rate_evaluation : std::false_type
{
};

template <class Problem, class Vector>
struct has_rate_evaluation<
    Problem, Vector,
    std::void_t<decltype( std::declval<Problem &>()
                              .evaluate_rate( std::declval<const Vector &>(), std::declval<Vector &>() ) )>>
    : std::is_same<
          decltype( std::declval<Problem &>()
                        .evaluate_rate( std::declval<const Vector &>(), std::declval<Vector &>() ) ),
          bool>
{
};

template <class Adaptation, class Vector, class T, class = void>
struct has_embedded_assessment : std::false_type
{
};

template <class Adaptation, class Vector, class T>
struct has_embedded_assessment<
    Adaptation, Vector, T,
    std::void_t<decltype( std::declval<Adaptation &>().assess(
        std::declval<T>(), std::declval<T>(), std::declval<const Vector &>(), std::declval<const Vector &>(),
        std::declval<T &>(), std::declval<unsigned int>(), std::declval<const Vector *>()
    ) )>> : std::true_type
{
};

template <class Problem, class Vector, class T, class = void>
struct has_implicit_stage_interface : std::false_type
{
};

template <class Problem, class Vector, class T>
struct has_implicit_stage_interface<
    Problem, Vector, T,
    std::void_t<
        typename Problem::jacobi_operator_type, decltype( std::declval<Problem &>().set_time( std::declval<T>() ) ),
        decltype( std::declval<Problem &>().set_affine_shift( std::declval<T>(), std::declval<T>() ) ),
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
#endif
