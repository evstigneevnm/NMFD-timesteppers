#include "implicit_imex_test.h"

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;

// M(t,u)=[[2+u0^2,.25],[.25,3+t^2]] is symmetric positive definite.
// Each physical RHS is M*g_partition; g_E+g_I=(-u0,-2*u1), giving
// u=(exp(-t),exp(-2*t)). The explicit provider solves the coupled mass system;
// the shifted implicit Jacobian includes DM[u]*v, not only M-J_f.
struct mass_partition
{
    static constexpr int size     = 2;
    bool                 implicit = false, state_dependent = true;
    __DEVICE_TAG__ auto  mass( double t, double u ) const
    {
        return scfd::static_mat::mat<double, 2, 2>{ 2 + ( state_dependent ? u * u : 0 ), .25, .25, 3 + t * t };
    }
    __DEVICE_TAG__ auto rate_matrix() const
    {
        return implicit ? scfd::static_mat::mat<double, 2, 2>{ -1.4, -.2, .3, -2.5 }
                        : scfd::static_mat::mat<double, 2, 2>{ .4, .2, -.3, .5 };
    }
    __DEVICE_TAG__ void rate( double t, const vector_type &x, vector_type &out ) const
    {
        const auto   m = mass( t, x( 0 ) ), b = rate_matrix();
        const double g0          = b( 0, 0 ) * x( 0 ) + b( 0, 1 ) * x( 1 );
        const double g1          = b( 1, 0 ) * x( 0 ) + b( 1, 1 ) * x( 1 );
        const double f0          = m( 0, 0 ) * g0 + m( 0, 1 ) * g1;
        const double f1          = m( 1, 0 ) * g0 + m( 1, 1 ) * g1;
        const double determinant = m( 0, 0 ) * m( 1, 1 ) - m( 0, 1 ) * m( 1, 0 );
        out( 0 )                 = ( m( 1, 1 ) * f0 - m( 0, 1 ) * f1 ) / determinant;
        out( 1 )                 = ( m( 0, 0 ) * f1 - m( 1, 0 ) * f0 ) / determinant;
    }
    __DEVICE_TAG__ void residual(
        double t, const stage_scaling &s, const vector_type &x, const vector_type &previous, const vector_type &rhs,
        vector_type &out
    ) const
    {
        const auto   m = mass( t, x( 0 ) ), b = rate_matrix();
        const double d0 = s.mass * ( x( 0 ) - previous( 0 ) ) - s.rhs * ( b( 0, 0 ) * x( 0 ) + b( 0, 1 ) * x( 1 ) );
        const double d1 = s.mass * ( x( 1 ) - previous( 1 ) ) - s.rhs * ( b( 1, 0 ) * x( 0 ) + b( 1, 1 ) * x( 1 ) );
        out( 0 )        = m( 0, 0 ) * d0 + m( 0, 1 ) * d1 - s.rhs * rhs( 0 );
        out( 1 )        = m( 1, 0 ) * d0 + m( 1, 1 ) * d1 - s.rhs * rhs( 1 );
    }
    auto jacobian(
        double t, const stage_scaling &s, const scfd::static_vec::vec<double, 2> &x,
        const scfd::static_vec::vec<double, 2> &previous
    ) const
    {
        const auto m = mass( t, x[0] ), b = rate_matrix();
        auto       result = scfd::static_mat::mat<double, 2, 2>::make_zero();
        for ( int j = 0; j < 2; ++j )
            for ( int k = 0; k < 2; ++k )
                for ( int l = 0; l < 2; ++l )
                    result( j, k ) += m( j, l ) * ( ( l == k ? s.mass : 0 ) - s.rhs * b( l, k ) );
        if ( state_dependent )
            result( 0, 0 ) +=
                2 * x[0] * ( s.mass * ( x[0] - previous[0] ) - s.rhs * ( b( 0, 0 ) * x[0] + b( 0, 1 ) * x[1] ) );
        return result;
    }
};

void test_mass( bool state_dependent )
{
    using partition    = rate_problem<mass_partition>;
    using problem_type = imex_problem<partition, partition>;
    auto ops           = std::make_shared<operations_type>( 2 );
    auto e             = std::make_shared<partition>( ops, mass_partition{ false, state_dependent } );
    auto i             = std::make_shared<partition>( ops, mass_partition{ true, state_dependent } );
    auto problem       = std::make_shared<problem_type>( e, i );
    for ( const double time : { .25, .6 } )
        for ( const double alpha : { -.5, 2., 1e6 } )
            check_jacobian( ops, *i, { .7, .4 }, time, alpha );
    vector_wrap_type initial( *ops ), out( *ops );
    initial.start_use();
    out.start_use();
    for ( const char *method : imex_methods )
    {
        double errors[2]{};
        for ( int level = 0; level < 2; ++level )
        {
            const scfd::static_vec::vec<double, 2> x{ std::exp( -.25 ), std::exp( -.5 ) };
            ops->set( x.d, *initial );
            auto step = make_imex_step( ops, problem, .05 / ( level + 1 ), method );
            integration::time_integrator<operations_type, imex_step<problem_type>> integrate( ops, step );
            integrate.set_time_interval( .25, .85 );
            integrate.apply( *initial, *out );
            require( integrate.get_status() == integration_status::completed, "IMEX coupled mass integration failed" );
            scfd::static_vec::vec<double, 2> actual;
            ops->get( *out, actual.d );
            errors[level] =
                std::max( std::abs( actual[0] - std::exp( -.85 ) ), std::abs( actual[1] - std::exp( -1.7 ) ) );
        }
        const double observed = std::log2( errors[0] / errors[1] );
        const auto   order    = runge_kutta::make_imex_butcher_table( method ).order;
        std::cout << ( state_dependent ? "State/time mass " : "Time mass " ) << method << " order=" << observed << '\n';
        require( observed > order - .2 && observed < order + .3, "IMEX mass solve or mass Jacobian changed the order" );
    }
    require( e->rate_calls > 0 && i->rate_calls > 0, "Mass solves were not delegated to both partition providers" );
}

int main()
try
{
    backend_type::init_device();
    test_mass( false );
    test_mass( true );
    std::cout << "Fixed IMEX coupled time/state-dependent mass, Jacobians and refinement: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
