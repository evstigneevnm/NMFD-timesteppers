#ifndef NMFD_TIME_STEPPERS_TESTS_ROSENBROCK_TIME_PROBLEM_H
#define NMFD_TIME_STEPPERS_TESTS_ROSENBROCK_TIME_PROBLEM_H

#include "implicit_rosenbrock_test.h"

namespace rosenbrock_tests
{
// Manufactured identity-mass system u'=j(t)*u+(2-j(t))*exp(2*t), u=exp(2*t).
// j=-lambda provides stiff linear relaxation; j=t tests a time-varying Jacobian.
// The partial derivative holds u fixed: f_t=j'*u+(4-2*j-j')*exp(2*t).
// All components evolve independently, including guarded 8192-component tests.
template <class Operations = operations_type>
struct time_problem : problem<decay, Operations>
{
    using base        = problem<decay, Operations>;
    using vector_type = typename Operations::vector_type;
    using base::base;
    bool         varying = true;
    double       lambda = 50, prepared_time = 0, first_rate_time = 0, last_rate_time = 0, derivative_time = 0;
    unsigned int derivatives = 0, fail_derivative_on = 0, invalid_derivative_on = 0, throw_rate_on = 0;
    bool         throw_derivative = false;

    struct mapping
    {
        double              time, lambda;
        bool                varying, derivative;
        vector_type         state, out;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
        {
            const double j  = varying ? time : -lambda;
            const double dj = varying ? 1 : 0;
            out( i )        = derivative ? dj * state( i ) + ( 4 - 2 * j - dj ) * exp( 2 * time )
                                         : j * state( i ) + ( 2 - j ) * exp( 2 * time );
        }
    };
    bool evaluate_rate( const vector_type &state, vector_type &out )
    {
        ++this->rates;
        if ( this->rates == 1 )
            first_rate_time = this->time;
        last_rate_time = this->time;
        if ( this->throw_rate || this->rates == throw_rate_on )
            throw std::runtime_error( "Injected time-dependent rate exception" );
        if ( this->rates == this->fail_rate_on || this->rates == this->invalid_rate_on )
        {
            this->ops->assign_scalar(
                this->rates == this->invalid_rate_on ? std::numeric_limits<double>::infinity() : 99., out
            );
            return this->rates != this->fail_rate_on;
        }
        typename Operations::for_each_type{}(
            mapping{ this->time, lambda, varying, false, state, out }, this->ops->size()
        );
        return true;
    }
    bool evaluate_time_derivative( const vector_type &state, vector_type &out )
    {
        ++derivatives;
        derivative_time = this->time;
        if ( throw_derivative )
            throw std::runtime_error( "Injected time derivative exception" );
        if ( derivatives == fail_derivative_on || derivatives == invalid_derivative_on )
        {
            this->ops->assign_scalar(
                derivatives == invalid_derivative_on ? std::numeric_limits<double>::infinity() : 99., out
            );
            return derivatives != fail_derivative_on;
        }
        typename Operations::for_each_type{}(
            mapping{ this->time, lambda, varying, true, state, out }, this->ops->size()
        );
        return true;
    }
    void set_linearization_point( const vector_type &state )
    {
        prepared_time      = this->time;
        this->model.lambda = varying ? -this->time : lambda;
        base::set_linearization_point( state );
    }
};
}
#endif
