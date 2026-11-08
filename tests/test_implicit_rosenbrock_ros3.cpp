#include "implicit_rosenbrock_time_problem.h"

using namespace rosenbrock_tests;

void test_tables()
{
    const rosenbrock::coefficients legacy;
    const rosenbrock::method_table ros2, ros3( "ROS3" );
    check(
        ros2.stage_count() == 2 && ros2.order() == 2 && ros2.embedded_order() == 1 && ros2.error_order() == 2,
        "ROS2 table metadata changed"
    );
    check(
        ros2.gamma == legacy.gamma && ros2.a( 1, 0 ) == legacy.a21 && ros2.c( 1, 0 ) == legacy.c21 &&
            ros2.m( 0 ) == legacy.m1 && ros2.m( 1 ) == legacy.m2 && ros2.e( 0 ) == legacy.e1 &&
            ros2.e( 1 ) == legacy.e2 && ros2.time_derivative( 0 ) == legacy.gamma1 &&
            ros2.time_derivative( 1 ) == legacy.gamma2,
        "Method-neutral table changed ROS2 coefficients"
    );
    check(
        ros3.stage_count() == 3 && ros3.order() == 3 && ros3.embedded_order() == 2 && ros3.error_order() == 3,
        "ROS3 table metadata is incorrect"
    );
    for ( const auto *table : { &ros2, &ros3 } )
        for ( std::size_t i = 0; i < table->stage_count(); ++i )
        {
            check( table->a( i, i ) == 0 && table->c( i, i ) == 0, "Stage coupling is not strictly lower triangular" );
            if ( table->reuse_previous_rate( i ) )
            {
                check( i && table->stage_time( i ) == table->stage_time( i - 1 ), "Reused rate has a different time" );
                for ( std::size_t j = 0; j < i; ++j )
                    check( table->a( i, j ) == table->a( i - 1, j ), "Reused rate has a different stage state" );
            }
        }
    check(
        !ros3.reuse_previous_rate( 0 ) && !ros3.reuse_previous_rate( 1 ) && ros3.reuse_previous_rate( 2 ),
        "ROS3 lost its shared second/third-stage rate"
    );
    for ( unsigned int operation = 0; operation < 4; ++operation )
    {
        bool failed = false;
        try
        {
            if ( operation == 0 )
                ros3.a( 3, 0 );
            if ( operation == 1 )
                ros3.c( 0, 3 );
            if ( operation == 2 )
                ros3.m( 3 );
            if ( operation == 3 )
                ros3.reuse_previous_rate( 3 );
        }
        catch ( const std::out_of_range & )
        {
            failed = true;
        }
        check( failed, "ROS3 coefficient access accepted an invalid stage" );
    }
}

// Exponential decay and logistic growth have exact solutions; both must show
// third-order global convergence with one frozen Jacobian and two rates per step.
template <class Model>
double integrate( Model model, double h, double end, double initial, double reference_time = 1 )
{
    using P                   = problem<Model>;
    using L                   = direct_solver<P>;
    using S                   = rosenbrock::time_step<operations_type, P, adaptation_type, L>;
    auto               ops    = std::make_shared<operations_type>( 1 );
    auto               p      = std::make_shared<P>( ops, model );
    auto               linear = std::make_shared<L>( ops );
    typename S::params settings;
    settings.method         = "ROS3";
    settings.reference_time = reference_time;
    auto step               = std::make_shared<S>(
        ops, p, std::make_shared<adaptation_type>( adaptation_type::params{ h } ), linear, settings
    );
    vector_wrap_type in( *ops ), out( *ops );
    in.start_use();
    out.start_use();
    ops->assign_scalar( initial, *in );
    integration::time_integrator<operations_type, S> run( ops, step );
    run.set_time_interval( 0, end );
    run.apply( *in, *out );
    check(
        run.get_status() == integration_status::completed && run.get_final_time() == end, "ROS3 integration failed"
    );
    check(
        p->rates == 2 * p->linearizations && p->residuals == 0 && linear->bindings == p->linearizations &&
            linear->calls == 3 * linear->bindings,
        "ROS3 did not reuse its rate and shifted operator"
    );
    return ops->get_value_at_point( 0, *out );
}

void test_orders_and_scaling()
{
    for ( bool nonlinear : { false, true } )
    {
        const double exact = nonlinear ? 1 / ( 1 + 3 * std::exp( -1. ) ) : std::exp( -1. );
        auto         error = [&]( double h ) {
            return std::abs(
                ( nonlinear ? integrate( logistic{}, h, 1, .25 ) : integrate( decay{}, h, 1, 1 ) ) - exact
            );
        };
        const double e1 = error( .04 ), e2 = error( .02 ), e3 = error( .01 );
        std::cout << "ROS3 nonlinear=" << nonlinear << " errors=" << e1 << ',' << e2 << ',' << e3 << '\n';
        check( e1 / e2 > 7.4 && e1 / e2 < 8.6 && e2 / e3 > 7.4 && e2 / e3 < 8.6, "ROS3 lost third-order convergence" );
    }
    check( std::abs( integrate( decay{}, .01, -.1, 1 ) - std::exp( .1 ) ) < 1e-8, "ROS3 backward flow is inaccurate" );
    check(
        integrate( logistic{}, .01, .1, 1 ) == 1 && integrate( logistic{}, .01, .1, 0 ) == 0,
        "ROS3 changed exact equilibria"
    );
    for ( double h : { 1e-100, 1e-14, 1e-3, 1., 1e3, 1e14, 1e100, 1e300 } )
        check(
            std::abs( integrate( constant_rate{}, h, h, 0 ) / h - 1 ) < 5e-15, "ROS3 balancing lost its time scaling"
        );
    const double reference = integrate( decay{}, .01, 1, 1 );
    for ( double scale : { 1e-6, 1., 1e6 } )
        check(
            std::abs( integrate( decay{}, .01, 1, 1, scale ) - reference ) < 1e-13,
            "ROS3 reference time changed the solution"
        );

    // The independent rational stability function tends to zero on the negative
    // real axis. No timestep reduction is allowed in this fixed-step test.
    const double g        = rosenbrock::method_table( "ROS3" ).gamma;
    double       previous = 1;
    for ( double lambda : { 1., 10., 1e3, 1e8 } )
    {
        const double z      = -lambda;
        const double exact  = ( 1 + ( 1 - 3 * g ) * z + ( .5 - 3 * g + 3 * g * g ) * z * z ) / std::pow( 1 - g * z, 3 );
        const double result = integrate( decay{ lambda }, 1, 1, 1 );
        check(
            std::abs( result - exact ) < 3e-14 && std::abs( result ) <= previous,
            "ROS3 stiff stability function mismatch"
        );
        previous = std::abs( result );
    }
    check( previous < 4e-8, "ROS3 stiff limit is not damped" );
}

// x'=-2x+y, y'=x-2y has eigenvalues -1,-3. Check both exact modal
// components and agreement between injected GMRES and a supplied direct inverse.
template <bool Krylov>
double coupled()
{
    using P     = problem<coupled_decay>;
    using L     = std::conditional_t<Krylov, krylov_solver<P>, direct_solver<P>>;
    using S     = rosenbrock::time_step<operations_type, P, adaptation_type, L>;
    auto ops    = std::make_shared<operations_type>( 2 );
    auto linear = [&] {
        if constexpr ( Krylov )
            return make_krylov<L>( ops );
        else
            return std::make_shared<L>( ops );
    }();
    auto step = std::make_shared<S>(
        ops, std::make_shared<P>( ops ), std::make_shared<adaptation_type>( adaptation_type::params{ .01 } ), linear,
        typename S::params{ "ROS3" }
    );
    vector_wrap_type in( *ops ), out( *ops );
    in.start_use();
    out.start_use();
    ops->assign_scalar( 0., *in );
    ops->set_value_at_point( 1., 0, *in );
    integration::time_integrator<operations_type, S> run( ops, step );
    run.set_time_interval( 0, .1 );
    run.apply( *in, *out );
    check( run.get_status() == integration_status::completed, "ROS3 coupled flow failed" );
    for ( int i = 0; i < 2; ++i )
    {
        const double exact = ( std::exp( -.1 ) + ( i ? -1 : 1 ) * std::exp( -.3 ) ) / 2;
        check(
            std::abs( ops->get_value_at_point( i, *out ) - exact ) < 1e-7, "ROS3 coupled modal solution is inaccurate"
        );
    }
    return ops->get_value_at_point( 0, *out );
}

// u'=j(t)u+(2-j(t))*exp(2t) has u=exp(2t). Varying j tests the
// partial time derivative and frozen Jacobian; negative constant j adds stiffness.
double time_error( bool varying, double start, double end, double h )
{
    using P          = time_problem<>;
    using L          = direct_solver<P>;
    using S          = rosenbrock::time_step<operations_type, P, adaptation_type, L>;
    auto ops         = std::make_shared<operations_type>( 1 );
    auto p           = std::make_shared<P>( ops );
    p->varying       = varying;
    auto      linear = std::make_shared<L>( ops );
    S::params settings;
    settings.method     = "ROS3";
    settings.autonomous = false;
    auto step           = std::make_shared<S>(
        ops, p, std::make_shared<adaptation_type>( adaptation_type::params{ h } ), linear, settings
    );
    vector_wrap_type in( *ops ), out( *ops );
    in.start_use();
    out.start_use();
    ops->assign_scalar( std::exp( 2 * start ), *in );
    integration::time_integrator<operations_type, S> run( ops, step );
    run.set_time_interval( start, end );
    run.apply( *in, *out );
    check(
        run.get_status() == integration_status::completed && run.get_final_time() == end,
        "Non-autonomous ROS3 missed its target"
    );
    check(
        p->rates == 2 * run.get_steps() && p->derivatives == run.get_steps() && linear->bindings == run.get_steps() &&
            linear->calls == 3 * run.get_steps(),
        "ROS3 repeated a shared rate or base preparation"
    );
    check(
        p->time == p->prepared_time && p->derivative_time == p->prepared_time &&
            std::abs( p->last_rate_time - ( p->prepared_time + step->table().gamma * step->get_dt() ) ) < 2e-16,
        "ROS3 used the wrong shared stage time or did not restore base time"
    );
    return std::abs( ops->get_value_at_point( 0, *out ) - std::exp( 2 * end ) );
}

int main()
try
{
    backend_type::init_device();
    test_tables();
    test_orders_and_scaling();
    check( std::abs( coupled<false>() - coupled<true>() ) < 1e-12, "ROS3 GMRES/direct inverse disagreement" );
    for ( bool varying : { false, true } )
        for ( bool backward : { false, true } )
        {
            const double start = .37, end = backward ? .17 : .57, h = varying ? .02 : .001;
            const double e1 = time_error( varying, start, end, h ), e2 = time_error( varying, start, end, h / 2 );
            std::cout << "ROS3 time varying=" << varying << " backward=" << backward << " errors=" << e1 << ',' << e2
                      << '\n';
            check( e1 / e2 > 7.2 && e1 / e2 < 8.8, "Non-autonomous ROS3 lost third-order accuracy" );
        }
    std::cout << "ROS3 coefficients, third-order flows, stiff decay, timing and scaling: " << checks
              << " checks PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
