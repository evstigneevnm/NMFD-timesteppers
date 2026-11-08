#ifndef NMFD_TIME_STEPPERS_TESTS_ROSENBROCK_MASS_PROBLEM_H
#define NMFD_TIME_STEPPERS_TESTS_ROSENBROCK_MASS_PROBLEM_H

#include "implicit_rosenbrock_w_problem.h"

namespace rosenbrock_tests
{
// Constant invertible 2x2 blocks, repeated over backend vectors. Choosing
// f=M*q preserves the exact decay/logistic solutions while testing coupled M.
struct mass_matrix
{
    double                m00 = 2, m01 = .7, m10 = -.3, m11 = 1.5;
    __DEVICE_TAG__ double multiply( std::ptrdiff_t i, double x, double y ) const
    {
        return i % 2 ? m10 * x + m11 * y : m00 * x + m01 * y;
    }
    template <class Vector>
    struct view;
};
template <class Vector>
struct mass_matrix::view
{
    Vector                vector;
    mass_matrix           mass;
    bool                  inverse;
    __DEVICE_TAG__ double operator()( std::ptrdiff_t i ) const
    {
        if ( !inverse )
            return vector( i );
        const auto first = i - i % 2;
        return ( i % 2 ? -mass.m10 * vector( first ) + mass.m00 * vector( first + 1 )
                       : mass.m11 * vector( first ) - mass.m01 * vector( first + 1 ) ) /
               ( mass.m00 * mass.m11 - mass.m01 * mass.m10 );
    }
};

// Only the existing residual contract is provided: deliberately no rate or
// mass-action callback. The prepared operator owns a snapshot of its data.
template <class Model, class Operations = operations_type>
struct mass_problem
{
    using vector_type = typename Operations::vector_type;
    using wrap        = nmfd::detail::vector_wrap<Operations>;
    struct jacobian
    {
        std::shared_ptr<Operations> ops;
        wrap                        point;
        Model                       model;
        mass_matrix                 mass;
        double                      a = 0, b = 1;
        explicit jacobian( std::shared_ptr<Operations> o ) : ops( std::move( o ) ), point( *ops )
        {
            point.start_use();
        }
        struct mapping
        {
            Model               model;
            mass_matrix         mass;
            double              a, b;
            vector_type         point, in, out;
            bool                inverse;
            __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
            {
                if ( inverse )
                    out( i ) = model.inverse(
                        i, typename mass_matrix::view<vector_type>{ point, mass, false },
                        typename mass_matrix::view<vector_type>{ in, mass, true }, a, b
                    );
                else
                {
                    const auto j = i - i % 2;
                    out( i )     = mass.multiply(
                        i, a * in(j)-b * model.jacobian( j, point, in ),
                        a * in( j + 1 ) - b * model.jacobian( j + 1, point, in )
                    );
                }
            }
        };
        void apply( const vector_type &in, vector_type &out ) const
        {
            typename Operations::for_each_type{}( mapping{ model, mass, a, b, *point, in, out, false }, ops->size() );
        }
        void solve( const vector_type &in, vector_type &out ) const
        {
            typename Operations::for_each_type{}( mapping{ model, mass, a, b, *point, in, out, true }, ops->size() );
        }
    };
    using jacobi_operator_type = jacobian;
    std::shared_ptr<Operations> ops;
    wrap                        previous, rhs;
    std::shared_ptr<jacobian>   jacobi;
    Model                       model;
    mass_matrix                 mass;
    double                      a = 0, b = 1, time = 0;
    unsigned int                linearizations = 0, residuals = 0, invalid_on = 0, throw_on = 0;
    bool                        null_jacobian = false;
    explicit mass_problem( std::shared_ptr<Operations> o, Model m = {}, mass_matrix matrix = {} )
        : ops( std::move( o ) ), previous( *ops ), rhs( *ops ), jacobi( std::make_shared<jacobian>( ops ) ), model( m ),
          mass( matrix )
    {
        previous.start_use();
        rhs.start_use();
    }
    void set_time( double t )
    {
        time = t;
    }
    void set_affine_shift( double mass, double force )
    {
        a = mass;
        b = force;
    }
    vector_type &previous_state()
    {
        return *previous;
    }
    const vector_type &previous_state() const
    {
        return *previous;
    }
    vector_type &known_rhs()
    {
        return *rhs;
    }
    const vector_type &known_rhs() const
    {
        return *rhs;
    }
    struct mapping
    {
        Model               model;
        mass_matrix         mass;
        double              a, b;
        vector_type         in, previous, rhs, out;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
        {
            const auto j = i - i % 2;
            out( i )     = a * mass.multiply( i, in(j)-previous( j ), in( j + 1 ) - previous( j + 1 ) ) -
                       b * ( mass.multiply( i, model.rate( j, in ), model.rate( j + 1, in ) ) + rhs( i ) );
        }
    };
    void apply( const vector_type &in, vector_type &out )
    {
        ++residuals;
        if ( residuals == throw_on )
            throw std::runtime_error( "Injected mass residual exception" );
        if ( residuals == invalid_on )
            ops->assign_scalar( std::numeric_limits<double>::infinity(), out );
        else
            typename Operations::for_each_type{}( mapping{ model, mass, a, b, in, *previous, *rhs, out }, ops->size() );
    }
    void set_linearization_point( const vector_type &in )
    {
        ++linearizations;
        jacobi->model = model;
        jacobi->mass  = mass;
        jacobi->a     = a;
        jacobi->b     = b;
        ops->assign( in, *jacobi->point );
    }
    std::shared_ptr<const jacobian> get_jacobi_operator() const
    {
        return null_jacobian ? nullptr : jacobi;
    }
};
}
#endif
