#include "implicit_rosenbrock_test.h"

using namespace rosenbrock_tests;

template <class Model, bool Krylov = false>
double integrate( Model model, double h, double end, double initial, double reference_time = 1 )
{
    using P     = problem<Model>;
    using L     = std::conditional_t<Krylov, krylov_solver<P>, direct_solver<P>>;
    using S     = rosenbrock::time_step<operations_type, P, adaptation_type, L>;
    auto ops    = std::make_shared<operations_type>( 1 );
    auto p      = std::make_shared<P>( ops, model );
    auto solver = [&]() {
        if constexpr ( Krylov )
            return make_krylov<L>( ops );
        else
            return std::make_shared<L>( ops );
    }();
    typename S::params parameters;
    parameters.reference_time = reference_time;
    auto step                 = std::make_shared<S>(
        ops, p, std::make_shared<adaptation_type>( adaptation_type::params{ h } ), solver, parameters
    );
    vector_wrap_type u( *ops ), out( *ops );
    u.start_use();
    out.start_use();
    ops->assign_scalar( initial, *u );
    integration::time_integrator<operations_type, S> run( ops, step );
    run.set_time_interval( 0, end );
    run.apply( *u, *out );
    check( run.get_status() == integration_status::completed, "ROS2 integration failed" );
    check( p->rates == 2 * p->linearizations && p->residuals == 0, "ROS2 used Newton or relinearized a stage" );
    if constexpr ( !Krylov )
        check(
            solver->bindings == p->linearizations && solver->calls == 2 * solver->bindings,
            "ROS2 did not reuse the shifted operator within a step"
        );
    return ops->get_value_at_point( 0, *out );
}

void test_orders()
{
    for ( bool nonlinear : { false, true } )
    {
        const double exact = nonlinear ? 1 / ( 1 + 3 * std::exp( -1. ) ) : std::exp( -1. );
        auto         error = [&]( double h ) {
            return std::abs(
                ( nonlinear ? integrate( logistic{}, h, 1, .25 ) : integrate( decay{}, h, 1, 1 ) ) - exact
            );
        };
        const double e1 = error( .02 ), e2 = error( .01 ), e3 = error( .005 );
        check(
            e1 / e2 > 3.7 && e1 / e2 < 4.3 && e2 / e3 > 3.7 && e2 / e3 < 4.3,
            "ROS2 analytical second-order convergence failed"
        );
    }
    check(
        std::abs( integrate<decay, true>( {}, .01, 1, 1 ) - integrate( decay{}, .01, 1, 1 ) ) < 1e-12,
        "ROS2 GMRES and supplied direct inverse disagree"
    );
    check(
        std::abs( integrate( decay{}, .001, -.1, 1 ) - std::exp( .1 ) ) < 1e-6,
        "ROS2 backward decay integration is inaccurate"
    );
    check(
        integrate( logistic{}, .01, 1, 1 ) == 1 && integrate( logistic{}, .01, 1, 0 ) == 0,
        "ROS2 did not preserve exact equilibria"
    );
}

void test_stiff_decay()
{
    // u'=-lambda*u, h=1. The known ROS2 stability function approaches zero
    // for lambda*h -> infinity, without reducing the fixed timestep.
    const rosenbrock::coefficients c;
    double                         previous = 1;
    for ( double lambda : { 1., 10., 1e3, 1e6 } )
    {
        const double z        = -lambda;
        const double expected = ( 1 + ( 1 - 2 * c.gamma ) * z ) / std::pow( 1 - c.gamma * z, 2 );
        const double result   = integrate( decay{ lambda }, 1, 1, 1 );
        check( std::abs( result - expected ) < 2e-14, "ROS2 stiff stability function mismatch" );
        check( std::abs( result ) <= previous, "ROS2 stiff damping did not improve" );
        previous = std::abs( result );
    }
    check( previous < 1e-6, "ROS2 stiff limit is not damped" );
}

void test_balancing()
{
    // Constant forcing is exact; these shifts exercise both tiny and huge h.
    for ( double h : { 1e-100, 1e-14, 1e-3, 1., 1e3, 1e14, 1e100, 1e300 } )
    {
        const double value = integrate( constant_rate{}, h, h, 0 );
        check( std::abs( value / h - 1 ) < 3e-15, "ROS2 right-hand side balancing lost the time scaling" );
    }
    for ( double reference : { 1e-6, 1., 1e6 } )
        check(
            std::abs( integrate( decay{}, .01, 1, 1, reference ) - integrate( decay{}, .01, 1, 1 ) ) < 1e-13,
            "ROS2 reference-time balancing changed the numerical solution"
        );
}

void test_coupled()
{
    using P   = problem<coupled_decay>;
    using L   = krylov_solver<P>;
    using S   = rosenbrock::time_step<operations_type, P, adaptation_type, L>;
    auto ops  = std::make_shared<operations_type>( 2 );
    auto p    = std::make_shared<P>( ops );
    auto step = std::make_shared<S>(
        ops, p, std::make_shared<adaptation_type>( adaptation_type::params{ .001 } ), make_krylov<L>( ops )
    );
    vector_wrap_type u( *ops ), out( *ops );
    u.start_use();
    out.start_use();
    ops->assign_scalar( 0., *u );
    ops->set_value_at_point( 1., 0, *u );
    integration::time_integrator<operations_type, S> run( ops, step );
    run.set_time_interval( 0, .1 );
    run.apply( *u, *out );
    check( run.get_status() == integration_status::completed, "ROS2 coupled GMRES integration failed" );
    for ( int i = 0; i < 2; ++i )
    {
        const double exact = ( std::exp( -.1 ) + ( i ? -1 : 1 ) * std::exp( -.3 ) ) / 2;
        check( std::abs( ops->get_value_at_point( i, *out ) - exact ) < 2e-6, "ROS2 coupled flow is inaccurate" );
    }
}

int main()
try
{
    backend_type::init_device();
    test_orders();
    test_stiff_decay();
    test_balancing();
    test_coupled();
    std::cout << "Fixed-step ROS2 order, nonlinear/coupled flows, stability and scaling: " << checks
              << " checks PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
