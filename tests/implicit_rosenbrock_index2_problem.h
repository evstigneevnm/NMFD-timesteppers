#ifndef NMFD_TIME_STEPPERS_TESTS_ROSENBROCK_INDEX2_PROBLEM_H
#define NMFD_TIME_STEPPERS_TESTS_ROSENBROCK_INDEX2_PROBLEM_H

#include "implicit_rosenbrock_test.h"

namespace rosenbrock_tests
{
enum class pressure_gauge
{
    mean_zero,
    pinned,
    missing
};

// Autonomous Stokes-like blocks: u0'=-u0-d, u1'=-2*u1+d,
// g=u0-u1=0, d=p0-p1. A pressure gauge removes the common pressure nullspace.
// Exact u0=u1=exp(-3*t/2), d=u0/2; M=diag(1,1,0,0).
//
// Moving blocks add a differential clock T'=1 and use
// u0'=-u0-d+sin(T), u1'=-2*u1+d+exp(-T)-cos(T)-3*sin(T),
// g=u0-u1-sin(T)=0. Exact u0=exp(-T), u1=exp(-T)-sin(T), d=sin(T).
// The clock column includes all force/constraint derivatives. No q_t callback
// or inverse of the singular mass is supplied. Pressure enters linearly;
// this is not a validation of general nonlinear multiplier dependence.
template <bool Moving, class Operations = rosenbrock_tests::operations_type, bool Rates = false>
struct index2_problem
{
    using operations_type                    = Operations;
    using vector_type                        = typename Operations::vector_type;
    using wrap                               = nmfd::detail::vector_wrap<Operations>;
    static constexpr unsigned int block_size = Moving ? 5 : 4;

    struct jacobian
    {
        std::shared_ptr<Operations> ops;
        wrap                        point;
        double                      a = 0, b = 1;
        pressure_gauge              gauge = pressure_gauge::mean_zero;
        explicit jacobian( std::shared_ptr<Operations> o ) : ops( std::move( o ) ), point( *ops )
        {
            point.start_use();
        }
        struct mapping
        {
            vector_type         point, in, out;
            double              a, b;
            pressure_gauge      gauge;
            bool                inverse;
            __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
            {
                const auto   row = i % block_size, j = i - row;
                const double t     = Moving ? point( j + block_size - 1 ) : 0;
                const double f0t   = Moving ? cos( t ) : 0;
                const double f1t   = Moving ? -exp( -t ) + sin( t ) - 3 * cos( t ) : 0;
                const double gt    = Moving ? cos( t ) : 0;
                const double clock = Moving ? in( j + block_size - 1 ) : 0;
                if ( inverse )
                {
                    const double dt = Moving ? clock / a : 0;
                    const double r0 = in( j ) + b * f0t * dt, r1 = in( j + 1 ) + b * f1t * dt;
                    const double r2 = in( j + 2 ) + gt * dt;
                    const double u1 = ( r0 + r1 - ( a + b ) * r2 ) / ( 2 * a + 3 * b ), u0 = u1 + r2;
                    const double d = ( r0 - ( a + b ) * u0 ) / b;
                    out( i )       = row == 0                          ? u0
                                     : row == 1                        ? u1
                                     : row == 4                        ? dt
                                     : gauge == pressure_gauge::pinned ? ( row == 2 ? in( j + 3 ) : in( j + 3 ) - d )
                                                                       : ( in( j + 3 ) + ( row == 2 ? d : -d ) ) / 2;
                }
                else
                {
                    const double d = in( j + 2 ) - in( j + 3 );
                    out( i )       = row == 0   ? ( a + b ) * in( j ) + b * d - b * f0t * clock
                                     : row == 1 ? ( a + 2 * b ) * in( j + 1 ) - b * d - b * f1t * clock
                                     : row == 2 ? in(j)-in( j + 1 ) - gt * clock
                                     : row == 4 ? a * clock
                                     : gauge == pressure_gauge::missing ? 0
                                     : gauge == pressure_gauge::pinned  ? in( j + 2 )
                                                                        : in( j + 2 ) + in( j + 3 );
                }
            }
        };
        void apply( const vector_type &in, vector_type &out ) const
        {
            typename Operations::for_each_type{}( mapping{ *point, in, out, a, b, gauge, false }, ops->size() );
        }
        bool solve( const vector_type &in, vector_type &out ) const
        {
            // Explicit setup failure: a compatible singular Krylov solve alone
            // cannot certify pressure uniqueness when the gauge is missing.
            if ( gauge == pressure_gauge::missing || a == 0 || b == 0 || 2 * a + 3 * b == 0 )
                return false;
            typename Operations::for_each_type{}( mapping{ *point, in, out, a, b, gauge, true }, ops->size() );
            return true;
        }
    };
    using jacobi_operator_type = jacobian;
    std::shared_ptr<Operations>                         ops;
    wrap                                                previous, rhs;
    nmfd::detail::vector_wrap<Operations, false, false> rate_scratch;
    std::shared_ptr<jacobian>                           jacobi;
    pressure_gauge                                      gauge = pressure_gauge::mean_zero;
    double                                              a = 0, b = 1, time = 0, rate_consistency_tolerance = 1e-5;
    unsigned int residuals = 0, linearizations = 0, rates = 0, invalid_on = 0, throw_on = 0;
    unsigned int fail_rate_on = 0, invalid_rate_on = 0, throw_rate_on = 0;

    explicit index2_problem( std::shared_ptr<Operations> o )
        : ops( std::move( o ) ), previous( *ops ), rhs( *ops ), rate_scratch( *ops ),
          jacobi( std::make_shared<jacobian>( ops ) )
    {
        if ( !ops->size() || ops->size() % block_size )
            throw std::invalid_argument( "Index-2 test requires complete blocks" );
        previous.start_use();
        rhs.start_use();
        if constexpr ( Rates )
        {
            rate_scratch.init();
            rate_scratch.start_use();
        }
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
        double              a, b;
        pressure_gauge      gauge;
        bool                rate;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
        {
            const auto   row = i % block_size, j = i - row;
            const double t = Moving ? in( j + block_size - 1 ) : 0;
            const double x = in( j ), y = in( j + 1 ), d = in( j + 2 ) - in( j + 3 );
            const double f0 = -x - d + ( Moving ? sin( t ) : 0 );
            const double f1 = -2 * y + d + ( Moving ? exp( -t ) - cos( t ) - 3 * sin( t ) : 0 );
            if ( rate )
            {
                // Differentiate the hidden constraint to determine pressure rate.
                const double dp = ( -f0 + 2 * f1 + ( Moving ? exp( -t ) + 4 * cos( t ) : 0 ) ) / 2;
                out( i )        = row == 0                          ? f0
                                  : row == 1                        ? f1
                                  : row == 4                        ? 1
                                  : gauge == pressure_gauge::pinned ? ( row == 2 ? 0 : -dp )
                                                                    : ( row == 2 ? dp : -dp ) / 2;
            }
            else
                out( i ) =
                    row == 0   ? a * ( x - previous( j ) ) - b * ( f0 + rhs( j ) )
                    : row == 1 ? a * ( y - previous( j + 1 ) ) - b * ( f1 + rhs( j + 1 ) )
                    : row == 2 ? x - y - ( Moving ? sin( t ) : 0 ) - rhs( j + 2 )
                    : row == 4 ? a * ( t - previous( j + 4 ) ) - b * ( 1 + rhs( j + 4 ) )
                    : gauge == pressure_gauge::missing
                        ? 0
                        : ( gauge == pressure_gauge::pinned ? in( j + 2 ) : in( j + 2 ) + in( j + 3 ) ) - rhs( j + 3 );
        }
    };
    void apply( const vector_type &in, vector_type &out )
    {
        if ( ++residuals == throw_on )
            throw std::runtime_error( "Injected index-2 residual exception" );
        if ( residuals == invalid_on )
            ops->assign_scalar( std::numeric_limits<double>::infinity(), out );
        else
            typename Operations::for_each_type{}(
                mapping{ in, *previous, *rhs, out, a, b, gauge, false }, ops->size()
            );
    }
    void set_linearization_point( const vector_type &in )
    {
        ++linearizations;
        jacobi->a     = a;
        jacobi->b     = b;
        jacobi->gauge = gauge;
        ops->assign( in, *jacobi->point );
    }
    std::shared_ptr<const jacobian> get_jacobi_operator() const
    {
        return jacobi;
    }

    struct consistency_mapping
    {
        vector_type         in, out;
        pressure_gauge      gauge;
        bool                hidden;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
        {
            const auto   row = i % block_size, j = i - row;
            const double t = Moving ? in( j + block_size - 1 ) : 0;
            // f0-f1-cos(T)=0 gives d=(-u0+2*u1-exp(-T)+4*sin(T))/2.
            const double target = ( -in( j ) + 2 * in( j + 1 ) + ( Moving ? -exp( -t ) + 4 * sin( t ) : 0 ) ) / 2;
            out( i )            = row == 0 && hidden ? in( j + 2 ) - in( j + 3 ) - target
                                  : row == 2         ? in(j)-in( j + 1 ) - ( Moving ? sin( t ) : 0 )
                                  : row == 3 ? ( gauge == pressure_gauge::pinned ? in( j + 2 ) : in( j + 2 ) + in( j + 3 ) )
                                             : 0;
        }
    };
    double constraint_norm( const vector_type &in, vector_type &scratch, bool hidden = true ) const
    {
        typename Operations::for_each_type{}( consistency_mapping{ in, scratch, gauge, hidden }, ops->size() );
        return ops->norm( scratch );
    }
    template <bool Enabled = Rates, std::enable_if_t<Enabled, int> = 0>
    bool evaluate_rate( const vector_type &in, vector_type &out )
    {
        if ( ++rates == throw_rate_on )
            throw std::runtime_error( "Injected consistent-rate exception" );
        if ( rates == fail_rate_on || gauge == pressure_gauge::missing )
            return false;
        const double defect = constraint_norm( in, *rate_scratch );
        if ( !std::isfinite( defect ) || defect > rate_consistency_tolerance )
            return false;
        if ( rates == invalid_rate_on )
            ops->assign_scalar( std::numeric_limits<double>::infinity(), out );
        else
            typename Operations::for_each_type{}( mapping{ in, {}, {}, out, 0, 1, gauge, true }, ops->size() );
        return true;
    }

    struct state_mapping
    {
        vector_type         in, out;
        double              time;
        pressure_gauge      gauge;
        bool                initialize;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t block ) const
        {
            const auto   j = block * block_size;
            const double t = initialize ? time : Moving ? in( j + block_size - 1 ) : 0;
            const double x = initialize ? exp( -( Moving ? 1 : 1.5 ) * t ) : in( j );
            const double y = x - ( Moving ? sin( t ) : 0 );
            const double d = Moving ? ( -x + 2 * y - exp( -t ) + 4 * sin( t ) ) / 2 : x / 2;
            for ( unsigned int row = 0; row < block_size; ++row )
                out( j + row ) = row == 0                          ? x
                                 : row == 1                        ? y
                                 : row == 4                        ? t
                                 : gauge == pressure_gauge::pinned ? ( row == 2 ? 0 : -d )
                                                                   : ( row == 2 ? d : -d ) / 2;
        }
    };
    void initialize_state( double t, vector_type &out ) const
    {
        typename Operations::for_each_type{}( state_mapping{ {}, out, t, gauge, true }, ops->size() / block_size );
    }
    void reconstruct( const vector_type &in, vector_type &out ) const
    {
        // Load each complete block before writing it, including in-place use.
        typename Operations::for_each_type{}( state_mapping{ in, out, 0, gauge, false }, ops->size() / block_size );
    }
    static double exact( unsigned int i, double t, pressure_gauge g = pressure_gauge::mean_zero )
    {
        const double x = std::exp( -( Moving ? 1 : 1.5 ) * t ), d = Moving ? std::sin( t ) : x / 2;
        return i == 0                        ? x
               : i == 1                      ? x - ( Moving ? std::sin( t ) : 0 )
               : i == 4                      ? t
               : g == pressure_gauge::pinned ? ( i == 2 ? 0 : -d )
                                             : ( i == 2 ? d : -d ) / 2;
    }
};

template <class Problem>
struct index2_solver
{
    using vector_type = typename Problem::vector_type;
    std::shared_ptr<const typename Problem::jacobi_operator_type> jacobi;
    unsigned int calls = 0, bindings = 0, fail_on = 0, invalid_on = 0, throw_on = 0;
    std::shared_ptr<typename Problem::operations_type> ops;
    explicit index2_solver( std::shared_ptr<typename Problem::operations_type> o ) : ops( std::move( o ) )
    {
    }
    void set_operator( std::shared_ptr<const typename Problem::jacobi_operator_type> j )
    {
        ++bindings;
        jacobi = std::move( j );
    }
    bool solve( const vector_type &in, vector_type &out )
    {
        if ( ++calls == throw_on )
            throw std::runtime_error( "Injected index-2 solve exception" );
        if ( calls == fail_on )
            return false;
        if ( calls == invalid_on )
        {
            ops->assign_scalar( std::numeric_limits<double>::infinity(), out );
            return true;
        }
        return jacobi->solve( in, out );
    }
};

template <class Controller, class Problem>
struct index2_control : Controller
{
    using vector_type = typename Problem::vector_type;
    std::shared_ptr<Problem> problem;
    typename Problem::wrap   scratch;
    double                   constraint_tolerance  = 1e-6;
    unsigned int             constraint_rejections = 0;
    template <class... Args>
    index2_control( std::shared_ptr<Problem> p, Args &&...args )
        : Controller( std::forward<Args>( args )... ), problem( std::move( p ) ), scratch( *problem->ops )
    {
        scratch.start_use();
    }
    double initialize( double t, const vector_type &in, double dt = -1 )
    {
        const auto defect = problem->constraint_norm( in, *scratch );
        if ( problem->gauge == pressure_gauge::missing || !std::isfinite( defect ) || defect > 1e-10 )
            throw std::invalid_argument( "Inconsistent index-2 initial state or missing pressure gauge" );
        return Controller::initialize( t, in, dt );
    }
    adaptation_status assess(
        double t, double dt, const vector_type &in, const vector_type &out, double &next, unsigned int order = 0,
        const vector_type *error = nullptr
    )
    {
        const auto defect = problem->constraint_norm( out, *scratch );
        if ( !std::isfinite( defect ) || defect > constraint_tolerance )
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
auto index2_params()
{
    typename Step::params p;
    p.method    = "ROSI2P2";
    p.mass_form = p.index2_dae = true;
    return p;
}
}
#endif
