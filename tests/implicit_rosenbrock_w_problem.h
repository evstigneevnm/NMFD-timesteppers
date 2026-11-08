#ifndef NMFD_TIME_STEPPERS_TESTS_ROSENBROCK_W_PROBLEM_H
#define NMFD_TIME_STEPPERS_TESTS_ROSENBROCK_W_PROBLEM_H

#include "implicit_rosenbrock_time_problem.h"

namespace rosenbrock_tests
{
// Keep the physical rate unchanged; only the supplied stage operator uses
// B=factor*J+offset*I. Its inverse is used solely by the analytical test solver.
template <class Model>
struct approximate_model : Model
{
    double factor = 1, offset = 0;
    template <class X>
    __DEVICE_TAG__ double jacobian( std::ptrdiff_t i, const X &u, const X &v ) const
    {
        return factor * Model::jacobian( i, u, v ) + offset * v( i );
    }
    template <class X>
    __DEVICE_TAG__ double inverse( std::ptrdiff_t i, const X &u, const X &v, double a, double b ) const
    {
        return Model::inverse( i, u, v, a - b * offset, b * factor );
    }
};

template <class Model, class Operations = operations_type>
struct w_problem : problem<approximate_model<Model>, Operations>
{
    using base = problem<approximate_model<Model>, Operations>;
    using base::base;
    bool lagged = false, prepared = false;
    void set_linearization_point( const typename Operations::vector_type &state )
    {
        if ( !lagged || !prepared )
            base::set_linearization_point( state );
        else
        {
            ++this->linearizations;
            this->jacobi->a     = this->a;
            this->jacobi->b     = this->b;
            this->jacobi->model = this->model;
        }
        prepared = true;
    }
};

// Manufactured exp(2t) solution retains its exact physical f_t, but supplies
// B=0.3*J-0.4*I. Rate calls must not overwrite the frozen approximate operator.
template <class Operations = operations_type>
struct w_time_problem : time_problem<Operations>
{
    using base = time_problem<Operations>;
    using base::base;
    void set_linearization_point( const typename Operations::vector_type &state )
    {
        base::set_linearization_point( state );
        this->jacobi->model.lambda = .3 * this->jacobi->model.lambda + .4;
    }
};

// The physical coupled decay has J=[-2,1;1,-2]. B=[-1,.4;-.25,-3]
// does not commute with J, exposing mixed W order conditions hidden by scalars.
struct noncommuting_decay : coupled_decay
{
    template <class X>
    __DEVICE_TAG__ double jacobian( std::ptrdiff_t i, const X &, const X &v ) const
    {
        return ( i % 2 ? -3 : -1 ) * v( i ) + ( i % 2 ? -.25 : .4 ) * v( i ^ 1 );
    }
    template <class X>
    __DEVICE_TAG__ double inverse( std::ptrdiff_t i, const X &, const X &v, double a, double b ) const
    {
        const double d0 = a + b, d1 = a + 3 * b;
        return ( ( i % 2 ? d0 : d1 ) * v( i ) + b * ( i % 2 ? -.25 : .4 ) * v( i ^ 1 ) ) / ( d0 * d1 + .1 * b * b );
    }
};
}
#endif
