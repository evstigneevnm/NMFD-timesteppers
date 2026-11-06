#include "implicit_imex_test.h"
#include "implicit_adaptive_test.h"

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

template <class Controller>
void test_adaptive_mass( bool state_dependent, const char *label, typename Controller::params settings = {} )
{
    using partition    = rate_problem<mass_partition>;
    using problem_type = imex_problem<partition, partition>;
    using control_type = observed_controller<Controller>;
    using step_type    = imex_step<problem_type, control_type>;
    auto ops           = std::make_shared<operations_type>( 2 );
    auto problem       = std::make_shared<problem_type>(
        std::make_shared<partition>( ops, mass_partition{ false, state_dependent } ),
        std::make_shared<partition>( ops, mass_partition{ true, state_dependent } )
    );
    vector_wrap_type initial( *ops ), out( *ops );
    initial.start_use();
    out.start_use();
    const scfd::static_vec::vec<double, 2> start{ std::exp( -.25 ), std::exp( -.5 ) };
    ops->set( start.d, *initial );
    double      errors[2]{};
    std::size_t steps[2]{};
    for ( int level = 0; level < 2; ++level )
    {
        auto p               = settings;
        p.relative_tolerance = level ? 2e-5 : 5e-3;
        p.absolute_tolerance = p.relative_tolerance * .01;
        p.initial_step = p.maximum_step = .6;
        p.minimum_step                  = 1e-8;
        auto control                    = std::make_shared<control_type>( ops, p );
        control->expected_order         = 3;
        auto step                       = std::make_shared<step_type>(
            ops, problem, control, newton_types<partition>::make_solver( ops ),
            typename step_type::params{ "IMEX_ARK324L2SA" }
        );
        integration::time_integrator<operations_type, step_type> flow( ops, step );
        flow.set_time_interval( .25, .85 );
        scfd::static_vec::vec<double, 2> first;
        for ( int run = 0; run < 2; ++run )
        {
            flow.apply( *initial, *out );
            require(
                flow.get_status() == integration_status::completed && flow.get_final_time() == .85 &&
                    control->commits == flow.get_steps() &&
                    control->assessments == control->commits + control->error_rejections,
                "Adaptive IMEX mass solve or acceptance accounting failed"
            );
            scfd::static_vec::vec<double, 2> actual;
            ops->get( *out, actual.d, 2 );
            if ( run )
                require(
                    flow.get_steps() == steps[level] && std::abs( actual[0] - first[0] ) < 1e-13 &&
                        std::abs( actual[1] - first[1] ) < 1e-13,
                    "Adaptive IMEX mass restart changed its trajectory"
                );
            first        = actual;
            steps[level] = flow.get_steps();
            errors[level] =
                std::max( std::abs( actual[0] - std::exp( -.85 ) ), std::abs( actual[1] - std::exp( -1.7 ) ) );
        }
        if ( level )
            require( control->error_rejections > 0, "Adaptive IMEX mass test did not reject an inaccurate step" );
    }
    std::cout << label << " state_mass=" << state_dependent << " errors=" << errors[0] << " -> " << errors[1] << '\n';
    require(
        errors[1] < std::max( 5e-9, .3 * errors[0] ) && errors[1] < 1e-4 && steps[1] > steps[0],
        "Adaptive IMEX mass accuracy did not improve"
    );
}

int main()
try
{
    backend_type::init_device();
    test_mass( false );
    test_mass( true );
    for ( const bool state_dependent : { false, true } )
    {
        test_adaptive_mass<integration::time_step_adaptation_matlab<operations_type>>( state_dependent, "MATLAB" );
        test_adaptive_mass<integration::time_step_adaptation_scipy<operations_type>>( state_dependent, "SciPy" );
        using filter = integration::time_step_adaptation_filter<operations_type>;
        filter::params p;
        p.controller = "H211";
        test_adaptive_mass<filter>( state_dependent, "H211", p );
    }
    std::cout << "Fixed IMEX coupled time/state-dependent mass, Jacobians and refinement: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
