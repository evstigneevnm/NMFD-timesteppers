#ifndef NMFD_TIME_STEPPERS_TESTS_IMPLICIT_RATE_H
#define NMFD_TIME_STEPPERS_TESTS_IMPLICIT_RATE_H

#include "implicit_test_support.h"

namespace nmfd
{
namespace time_steppers
{
namespace tests
{
// Opt-in test facade: retain the ordinary residual overload and use the same
// equation's physical rate kernel. No mass solve is performed by the stepper.
template <class Equation>
class rate_problem : public stage_problem<Equation>
{
    using base_type = stage_problem<Equation>;

public:
    using base_type::apply;
    using base_type::base_type;
    unsigned int        rate_calls = 0;
    std::vector<double> rate_times, implicit_times;

    void set_affine_shift( double alpha )
    {
        implicit_times.push_back( this->time );
        base_type::set_affine_shift( alpha );
    }
    struct rate_mapping
    {
        Equation            equation;
        double              time;
        vector_type         state, out;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t ) const
        {
            auto rate = out;
            equation.rate( time, state, rate );
        }
    };
    bool apply( const vector_type &state, vector_type &rate, evaluation_mode )
    {
        ++rate_calls;
        rate_times.push_back( this->time );
        execute_small_mapping( rate_mapping{ this->equation, this->time, state, rate } );
        return true;
    }
};

// The scalar mass and RHS are single-sourced in the manufactured ODE model.
template <class Equation>
struct rate_scalar_equation : scalar_time_equation<Equation>
{
    using scalar_time_equation<Equation>::scalar_time_equation;
    __DEVICE_TAG__ void rate( double t, const vector_type &state, vector_type &out ) const
    {
        out( 0 ) = this->rhs( t, state( 0 ) ) / this->mass( t );
    }
};
}
}
}
#endif
