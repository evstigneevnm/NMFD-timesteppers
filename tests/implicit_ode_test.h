#ifndef NMFD_TIME_STEPPERS_TESTS_IMPLICIT_ODE_H
#define NMFD_TIME_STEPPERS_TESTS_IMPLICIT_ODE_H
#include "implicit_adaptive_test.h"

namespace nmfd
{
namespace time_steppers
{
namespace tests
{
template <int Size, class Mapping>
__DEVICE_TAG__ void identity_mass_residual(
    const Mapping &mapping, const stage_scaling &scaling, const vector_type &previous, const vector_type &rhs
)
{
    mapping( 0 );
    for ( int i = 0; i < Size; ++i )
        mapping.out( i ) =
            scaling.mass * ( mapping.in(i)-previous( i ) ) - scaling.rhs * ( mapping.out( i ) + rhs( i ) );
}

template <class Equation>
void test_ode(
    const Equation &equation, const scfd::static_vec::vec<double, Equation::size> &initial_values,
    const scfd::static_vec::vec<double, Equation::size> &reference, double end, double dt
)
{
    using problem_type = stage_problem<Equation>;
    using types        = newton_types<problem_type>;
    auto operations    = std::make_shared<operations_type>( Equation::size );
    auto problem       = std::make_shared<problem_type>( operations, equation );
    check_jacobian( operations, *problem, initial_values );
    vector_wrap_type initial( *operations ), result( *operations );
    initial.start_use();
    result.start_use();
    operations->set( initial_values.d, *initial, Equation::size );
    for ( const char *method : { "IE", "SDIRK2(1)2" } )
    {
        double errors[2]{};
        for ( int level = 0; level < 2; ++level )
        {
            auto step = make_step( operations, problem, dt / ( level + 1 ), method );
            integration::time_integrator<operations_type, typename types::step_type> integrator( operations, step );
            integrator.set_time_interval( 0, end );
            integrator.apply( *initial, *result );
            require(
                integrator.get_status() == integration_status::completed && integrator.get_final_time() == end,
                "Implicit ODE integration failed"
            );
            typename problem_type::state_type actual;
            operations->get( *result, actual.d, Equation::size );
            for ( int i = 0; i < Equation::size; ++i )
                errors[level] = std::max( errors[level], std::abs( actual[i] - reference[i] ) );
            integrator.apply( *initial, *result );
            require( integrator.get_status() == integration_status::completed, "Implicit ODE restart failed" );
            typename problem_type::state_type repeated;
            operations->get( *result, repeated.d, Equation::size );
            for ( int i = 0; i < Equation::size; ++i )
                require( repeated[i] == actual[i], "Implicit ODE restart changed trajectory" );
        }
        require( errors[1] < std::max( 2e-10, .7 * errors[0] ), "Implicit ODE refinement failed" );
        double reference_scale = 1;
        for ( int i = 0; i < Equation::size; ++i )
            reference_scale = std::max( reference_scale, std::abs( reference[i] ) );
        require( errors[1] < .005 * reference_scale, "Implicit ODE reference mismatch" );
        std::cout << method << " reference errors " << errors[0] << " -> " << errors[1] << '\n';
    }
}
}
}
}
#endif
