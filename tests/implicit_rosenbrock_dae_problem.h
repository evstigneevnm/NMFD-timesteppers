#ifndef NMFD_TIME_STEPPERS_TESTS_ROSENBROCK_DAE_PROBLEM_H
#define NMFD_TIME_STEPPERS_TESTS_ROSENBROCK_DAE_PROBLEM_H

#include "implicit_rosenbrock_test.h"

namespace rosenbrock_tests
{
// Repeated autonomous index-1 blocks with singular M=diag(mass,0):
// linear:    mass*x'=mass*(-2*x+z),     0=z-x;
// nonlinear: mass*x'=mass*(-x-z+x*x),  0=z-x*x.
// From (1,1), exact x=exp(-t), z=exp(-t) or exp(-2*t).
// g_z=1 stays nonsingular. Only differential Jacobian rows may be approximated.
template <class Operations = operations_type>
struct dae_problem
{
    using vector_type = typename Operations::vector_type;
    using wrap        = nmfd::detail::vector_wrap<Operations>;
    struct jacobian
    {
        std::shared_ptr<Operations> ops;
        wrap                        point;
        double                      a = 0, b = 1, mass = 2, factor = 1;
        bool                        nonlinear = true;
        explicit jacobian( std::shared_ptr<Operations> o ) : ops( std::move( o ) ), point( *ops )
        {
            point.start_use();
        }
        struct mapping
        {
            vector_type         point, in, out;
            double              a, b, mass, factor;
            bool                nonlinear, inverse;
            __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
            {
                const auto   j = i - i % 2;
                const double d = nonlinear ? 2 * point( j ) : 1;
                const double A = mass * ( a + b * factor * ( nonlinear ? 1 - d : 2 ) );
                const double B = mass * b * factor * ( nonlinear ? 1 : -1 );
                // Solve the full shifted 2x2 system, never the singular mass.
                if ( inverse )
                {
                    const double x = ( in(j)-B * in( j + 1 ) ) / ( A + B * d );
                    out( i )       = i % 2 ? in( j + 1 ) + d * x : x;
                }
                else
                    out( i ) = i % 2 ? in( j + 1 ) - d * in( j ) : A * in( j ) + B * in( j + 1 );
            }
        };
        void apply( const vector_type &in, vector_type &out ) const
        {
            typename Operations::for_each_type{}(
                mapping{ *point, in, out, a, b, mass, factor, nonlinear, false }, ops->size()
            );
        }
        void solve( const vector_type &in, vector_type &out ) const
        {
            typename Operations::for_each_type{}(
                mapping{ *point, in, out, a, b, mass, factor, nonlinear, true }, ops->size()
            );
        }
    };
    using jacobi_operator_type = jacobian;
    std::shared_ptr<Operations> ops;
    wrap                        previous, rhs;
    std::shared_ptr<jacobian>   jacobi;
    double                      a = 0, b = 1, mass = 2, factor = 1, time = 0;
    bool                        nonlinear = true;
    unsigned int                residuals = 0, linearizations = 0, invalid_on = 0, throw_on = 0;
    explicit dae_problem( std::shared_ptr<Operations> o )
        : ops( std::move( o ) ), previous( *ops ), rhs( *ops ), jacobi( std::make_shared<jacobian>( ops ) )
    {
        previous.start_use();
        rhs.start_use();
    }
    void set_time( double t )
    {
        time = t;
    }
    void set_affine_shift( double m, double f )
    {
        a = m;
        b = f;
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
        vector_type         in, previous, rhs, out;
        double              a, b, mass;
        bool                nonlinear;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
        {
            const auto   j = i - i % 2;
            const double x = in( j ), z = in( j + 1 );
            out( i ) = i % 2 ? z - ( nonlinear ? x * x : x ) - rhs( i )
                             : a * mass * ( x - previous( j ) ) -
                                   b * ( mass * ( nonlinear ? -x - z + x * x : -2 * x + z ) + rhs( i ) );
        }
    };
    void apply( const vector_type &in, vector_type &out )
    {
        if ( ++residuals == throw_on )
            throw std::runtime_error( "Injected DAE residual exception" );
        if ( residuals == invalid_on )
            ops->assign_scalar( std::numeric_limits<double>::infinity(), out );
        else
            typename Operations::for_each_type{}(
                mapping{ in, *previous, *rhs, out, a, b, mass, nonlinear }, ops->size()
            );
    }
    void set_linearization_point( const vector_type &in )
    {
        ++linearizations;
        jacobi->a         = a;
        jacobi->b         = b;
        jacobi->mass      = mass;
        jacobi->factor    = factor;
        jacobi->nonlinear = nonlinear;
        ops->assign( in, *jacobi->point );
    }
    std::shared_ptr<const jacobian> get_jacobi_operator() const
    {
        return jacobi;
    }
    struct constraint_mapping
    {
        vector_type         in, out;
        bool                nonlinear;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
        {
            const auto j = i - i % 2;
            out( i )     = i % 2 ? in( i ) - ( nonlinear ? in(j)*in( j ) : in( j ) ) : 0;
        }
    };
    double constraint_norm( const vector_type &in, vector_type &scratch ) const
    {
        typename Operations::for_each_type{}( constraint_mapping{ in, scratch, nonlinear }, ops->size() );
        return ops->norm( scratch );
    }
};

// Example problem-owned policy, not an additional timestepper callback. State
// error and unscaled constraint tolerance are independently enforced through
// the existing controller boundary. Initialization never silently projects.
template <class Controller, class Problem>
struct dae_control : Controller
{
    using vector_type = typename Problem::vector_type;
    std::shared_ptr<Problem> problem;
    typename Problem::wrap   scratch;
    double                   constraint_tolerance  = 1e-7;
    unsigned int             constraint_rejections = 0;
    dae_control( std::shared_ptr<Problem> p, const typename Controller::params &params )
        : Controller( p->ops, params ), problem( std::move( p ) ), scratch( *problem->ops )
    {
        scratch.start_use();
    }
    double initialize( double t, const vector_type &in, double dt = -1 )
    {
        if ( problem->constraint_norm( in, *scratch ) > 1e-10 )
            throw std::invalid_argument( "Inconsistent index-1 initial state" );
        return Controller::initialize( t, in, dt );
    }
    adaptation_status assess(
        double t, double dt, const vector_type &in, const vector_type &out, double &next, unsigned int order = 0,
        const vector_type *error = nullptr
    )
    {
        const auto norm = problem->constraint_norm( out, *scratch );
        if ( !std::isfinite( norm ) || norm > constraint_tolerance )
        {
            ++constraint_rejections;
            const auto status = Controller::reject_step( dt );
            next              = Controller::get_dt();
            return status;
        }
        return Controller::assess( t, dt, in, out, next, order, error );
    }
};

template <class Step>
auto dae_params()
{
    typename Step::params p;
    p.method    = "ROS34PW2";
    p.mass_form = p.index1_dae = true;
    return p;
}
}
#endif
