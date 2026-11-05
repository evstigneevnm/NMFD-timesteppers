#include "implicit_imex_test.h"

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;

// Noncommuting partitions sum to diag(-1,-2), so x=exp(-t), y=exp(-2t).
// Unlike a scalar split, these catch the ordering of mixed explicit/implicit terms.
struct coupled_partition
{
    static constexpr int size     = 2;
    bool                 implicit = false;
    __DEVICE_TAG__ auto  matrix() const
    {
        return implicit ? scfd::static_mat::mat<double, 2, 2>{ -1.4, -.7, .3, -2.2 }
                        : scfd::static_mat::mat<double, 2, 2>{ .4, .7, -.3, .2 };
    }
    __DEVICE_TAG__ void rate( double, const vector_type &x, vector_type &out ) const
    {
        const auto   b = matrix();
        const double u = x( 0 ), v = x( 1 );
        out( 0 ) = b( 0, 0 ) * u + b( 0, 1 ) * v;
        out( 1 ) = b( 1, 0 ) * u + b( 1, 1 ) * v;
    }
    __DEVICE_TAG__ void residual(
        double t, const stage_scaling &s, const vector_type &x, const vector_type &previous, const vector_type &rhs,
        vector_type &out
    ) const
    {
        rate( t, x, out );
        for ( int j = 0; j < 2; ++j )
            out( j ) = s.mass * ( x( j ) - previous( j ) ) - s.rhs * ( out( j ) + rhs( j ) );
    }
    auto jacobian(
        double, const stage_scaling &s, const scfd::static_vec::vec<double, 2> &,
        const scfd::static_vec::vec<double, 2> &
    ) const
    {
        auto result = matrix();
        for ( int j = 0; j < 2; ++j )
            for ( int k = 0; k < 2; ++k )
                result( j, k ) = ( j == k ? s.mass : 0 ) - s.rhs * result( j, k );
        return result;
    }
};

// f_E=.3*u^2+cos(t), f_I=-u-.3*u^2-cos(t); total u'=-u.
// Nonlinearity and time-dependent cancellation require both partitions at the same stage state/time.
struct nonlinear_partition
{
    static constexpr int  size     = 1;
    bool                  implicit = false;
    __DEVICE_TAG__ double value( double t, double u ) const
    {
        const double e = .3 * u * u + std::cos( t );
        return implicit ? -u - e : e;
    }
    __DEVICE_TAG__ void rate( double t, const vector_type &x, vector_type &out ) const
    {
        out( 0 ) = value( t, x( 0 ) );
    }
    __DEVICE_TAG__ void residual(
        double t, const stage_scaling &s, const vector_type &x, const vector_type &previous, const vector_type &rhs,
        vector_type &out
    ) const
    {
        out( 0 ) = s.mass * ( x( 0 ) - previous( 0 ) ) - s.rhs * ( value( t, x( 0 ) ) + rhs( 0 ) );
    }
    auto jacobian(
        double, const stage_scaling &s, const scfd::static_vec::vec<double, 1> &x,
        const scfd::static_vec::vec<double, 1> &
    ) const
    {
        const double derivative = implicit ? -1 - .6 * x[0] : .6 * x[0];
        return scfd::static_mat::mat<double, 1, 1>{ s.mass - s.rhs * derivative };
    }
};

template <class Equation>
void test_order( Equation e, Equation i, const char *description )
{
    using partition    = rate_problem<Equation>;
    using problem_type = imex_problem<partition, partition>;
    auto ops           = std::make_shared<operations_type>( Equation::size );
    auto problem =
        std::make_shared<problem_type>( std::make_shared<partition>( ops, e ), std::make_shared<partition>( ops, i ) );
    vector_wrap_type initial( *ops ), out( *ops );
    initial.start_use();
    out.start_use();
    for ( const char *method : imex_methods )
    {
        double errors[3]{};
        for ( int level = 0; level < 3; ++level )
        {
            const double                                  start = .3, end = 1.1;
            scfd::static_vec::vec<double, Equation::size> exact, result;
            for ( int j = 0; j < Equation::size; ++j )
                exact[j] = std::exp( -( j + 1 ) * start );
            ops->set( exact.d, *initial );
            auto step = make_imex_step( ops, problem, .1 / ( 1 << level ), method );
            integration::time_integrator<operations_type, imex_step<problem_type>> integrate( ops, step );
            integrate.set_time_interval( start, end );
            integrate.apply( *initial, *out );
            require(
                integrate.get_status() == integration_status::completed && integrate.get_final_time() == end,
                "IMEX endpoint or convergence failure"
            );
            ops->get( *out, result.d );
            for ( int j = 0; j < Equation::size; ++j )
                errors[level] = std::max( errors[level], std::abs( result[j] - std::exp( -( j + 1 ) * end ) ) );
            integrate.apply( *initial, *out );
            scfd::static_vec::vec<double, Equation::size> repeated;
            ops->get( *out, repeated.d );
            for ( int j = 0; j < Equation::size; ++j )
                require( std::abs( repeated[j] - result[j] ) < 2e-13, "IMEX restart retained obsolete stage data" );
        }
        const auto order    = runge_kutta::make_imex_butcher_table( method ).order;
        const auto observed = std::log2( errors[1] / errors[2] );
        std::cout << description << ' ' << method << " errors=" << errors[0] << ',' << errors[1] << ',' << errors[2]
                  << " order=" << observed << '\n';
        // A particular split can cancel its leading truncation term.
        require( observed > order - .2, "IMEX mixed/limiting order mismatch" );
    }
}

// Stiff tracking u'=-kappa*(u-sin(t))+cos(t), exact u=sin(t), split into
// the time-dependent explicit forcing and the stiff linear implicit decay.
struct stiff_partition : imex_linear_equation
{
    bool                implicit = false;
    __DEVICE_TAG__ void rate( double t, const vector_type &x, vector_type &out ) const
    {
        out( 0 ) = implicit ? lambda * x( 0 ) : -lambda * std::sin( t ) + std::cos( t );
    }
};

void test_timing_and_stiffness()
{
    using explicit_type      = rate_problem<stiff_partition>;
    using implicit_type      = rate_problem<imex_linear_equation>;
    using problem_type       = imex_problem<explicit_type, implicit_type>;
    auto             ops     = std::make_shared<operations_type>( 1 );
    auto             e       = std::make_shared<explicit_type>( ops, stiff_partition{ { -30 }, false } );
    auto             i       = std::make_shared<implicit_type>( ops, imex_linear_equation{ -30 } );
    auto             problem = std::make_shared<problem_type>( e, i );
    vector_wrap_type initial( *ops ), out( *ops );
    initial.start_use();
    out.start_use();
    ops->assign_scalar( std::sin( .3 ), *initial );
    for ( const char *method : imex_methods )
    {
        double errors[2]{};
        for ( int level = 0; level < 2; ++level )
        {
            const double dt   = .04 / ( level + 1 );
            auto         step = make_imex_step( ops, problem, dt, method );
            step->set_time( .3 );
            step->set_target_time( .3 + dt );
            e->rate_times.clear();
            i->rate_times.clear();
            i->implicit_times.clear();
            const auto requests = problem->requests;
            step->apply( *initial, *out );
            require(
                step->get_status() == single_step_status::converged && problem->requests == requests,
                "IMEX fetched partition handles during a stage or failed the stage"
            );
            for ( const auto time : e->rate_times )
            {
                bool found = false;
                for ( std::size_t j = 0; j < step->table().explicit_table.size(); ++j )
                    found = found || std::abs( time - ( .3 + dt * step->table().explicit_table.c( j ) ) ) < 1e-14;
                require( found, "Explicit partition used a stale stage time" );
            }
            for ( std::size_t j = 0; j < step->table().implicit_table.size(); ++j )
                if ( step->table().implicit_table.a( j, j ) != 0 )
                {
                    require( !i->implicit_times.empty(), "Implicit stage was skipped" );
                    const auto time = i->implicit_times.front();
                    i->implicit_times.erase( i->implicit_times.begin() );
                    require(
                        std::abs( time - ( .3 + dt * step->table().implicit_table.c( j ) ) ) < 1e-14,
                        "Implicit partition used a stale stage time"
                    );
                }
            step->finalize( adaptation_status::accepted, .3 + step->get_dt(), *out );
            integration::time_integrator<operations_type, imex_step<problem_type>> integrate( ops, step );
            integrate.set_time_interval( .3, .7 );
            integrate.apply( *initial, *out );
            require( integrate.get_status() == integration_status::completed, "Stiff IMEX tracking failed" );
            errors[level] = std::abs( ops->get_value_at_point( 0, *out ) - std::sin( .7 ) );
        }
        require( errors[1] < errors[0] && errors[1] < .03, "Stiff IMEX tracking failed refinement" );
    }
}

int main()
try
{
    backend_type::init_device();
    test_order( imex_linear_equation{ -1 }, imex_linear_equation{ 0 }, "Pure explicit" );
    test_order( imex_linear_equation{ 0 }, imex_linear_equation{ -1 }, "Pure implicit" );
    test_order( coupled_partition{ false }, coupled_partition{ true }, "Noncommuting split" );
    test_order( nonlinear_partition{ false }, nonlinear_partition{ true }, "Nonlinear timed split" );
    test_timing_and_stiffness();
    std::cout << "Fixed IMEX orders, partition limits, coupling, stage times and restart: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
