#include "implicit_rosenbrock_index2_problem.h"
#include "implicit_rosenbrock_time_problem.h"

using namespace rosenbrock_tests;

// Table 3.2 of Rang/Angermann, in the original untransformed convention.
// Check the index-2/PDE conditions independently of the runtime recurrence.
using row    = std::array<double, 4>;
using matrix = std::array<row, 4>;
row multiply( const matrix &a, const row &x )
{
    row y{};
    for ( unsigned int i = 0; i < 4; ++i )
        for ( unsigned int j = 0; j < 4; ++j )
            y[i] += a[i][j] * x[j];
    return y;
}
double dot( const row &a, const row &b )
{
    double s = 0;
    for ( unsigned int i = 0; i < 4; ++i )
        s += a[i] * b[i];
    return s;
}
matrix inverse_lower( const matrix &a )
{
    matrix result{};
    for ( unsigned int j = 0; j < 4; ++j )
        for ( unsigned int i = 0; i < 4; ++i )
        {
            double value = i == j ? 1 : 0;
            for ( unsigned int k = 0; k < i; ++k )
                value -= a[i][k] * result[k][j];
            result[i][j] = value / a[i][i];
        }
    return result;
}
void coefficients()
{
    const rosenbrock::method_table t( "ROSI2P2" );
    const matrix                   A = {
        { { 0, 0, 0, 0 },
                            { .5, 0, 0, 0 },
                            { -.51983699657507165, 1.5198369965750715, 0, 0 },
                            { -.51983699657507165, 1.5198369965750715, 0, 0 } }
    };
    matrix G = {
        { { t.gamma, 0, 0, 0 },
          { -.5, t.gamma, 0, 0 },
          { -.40164172503011392, 1.1742718526976650, t.gamma, 0 },
          { 1.1865036632417383, -1.5198369965750715, -.10253318817512568, t.gamma } }
    };
    const row b  = { 2. / 3, 0, -.10253318817512568, t.gamma };
    const row bh = { -.95742384859111473, 2.9148476971822297, .5, -1.4574238485911146 };
    matrix    B  = G;
    for ( unsigned int i = 0; i < 4; ++i )
        for ( unsigned int j = 0; j < 4; ++j )
            B[i][j] += A[i][j];
    const auto ig = inverse_lower( G ), ib = inverse_lower( B );
    const row  one = { 1, 1, 1, 1 }, c = multiply( A, one ), beta = multiply( B, one ), d = multiply( G, one );
    row        square{};
    for ( unsigned int i = 0; i < 4; ++i )
    {
        square[i] = c[i] * c[i];
        check( std::abs( t.stage_time( i ) - c[i] ) < 5e-15, "ROSI2P2 changed original stage times" );
        check( std::abs( t.time_derivative( i ) - d[i] ) < 5e-15, "ROSI2P2 changed original partial-time weights" );
        double m = 0, e = 0;
        for ( unsigned int k = 0; k < 4; ++k )
        {
            m += b[k] * ig[k][i];
            e += ( b[k] - bh[k] ) * ig[k][i];
        }
        check(
            std::abs( t.m( i ) - m ) < 2e-14 && std::abs( t.e( i ) - e ) < 2e-14, "ROSI2P2 weight conversion failed"
        );
        for ( unsigned int j = 0; j < i; ++j )
        {
            double a = 0;
            for ( unsigned int k = 0; k < 4; ++k )
                a += A[i][k] * ig[k][j];
            check(
                std::abs( t.a( i, j ) - a ) < 2e-14 && std::abs( t.c( i, j ) + ig[i][j] ) < 2e-14,
                "ROSI2P2 lower-table conversion failed"
            );
        }
        check( std::abs( B[3][i] - b[i] ) < 3e-15, "ROSI2P2 lost stiff accuracy" );
    }
    check(
        std::abs( dot( b, one ) - 1 ) < 2e-14 && std::abs( dot( b, beta ) - .5 ) < 2e-14 &&
            std::abs( dot( b, multiply( B, beta ) ) - 1. / 6 ) < 2e-14 && std::abs( dot( b, square ) - 1. / 3 ) < 2e-14,
        "ROSI2P2 ODE order conditions failed"
    );
    check(
        std::abs( dot( bh, one ) - 1 ) < 2e-14 && std::abs( dot( bh, beta ) - .5 ) < 2e-14,
        "ROSI2P2 embedded ODE order conditions failed"
    );
    const auto w = multiply( ib, square ), v = multiply( ib, w );
    auto       av = multiply( A, v );
    for ( unsigned int i = 0; i < 4; ++i )
        av[i] *= c[i];
    check( std::abs( dot( b, w ) - 1 ) < 5e-13 && std::abs( dot( b, v ) - 2 ) < 5e-13, "Index-2 E3/F3a failed" );
    row dae_embedded{};
    for ( unsigned int j = 0; j < 4; ++j )
        for ( unsigned int i = 0; i < 4; ++i )
            dae_embedded[j] += ( t.m( i ) - t.index2_e( i ) ) * G[i][j];
    check(
        std::abs( dot( dae_embedded, one ) - 1 ) < 3e-14 && std::abs( dot( dae_embedded, beta ) - .5 ) < 3e-14 &&
            std::abs( dot( dae_embedded, w ) - 1 ) < 3e-13 && std::abs( dot( dae_embedded, v ) - 2 ) < 3e-13,
        "DAE embedded pair violates ODE/constraint moments"
    );
    check(
        std::abs( dot( bh, w ) - 1 ) > 1 && std::abs( dot( bh, v ) - 2 ) > 1,
        "Published ODE embedded pair no longer isolates the invalid DAE estimator"
    );
    check(
        std::abs( dot( b, av ) - 2. / 3 ) < 5e-13 && std::abs( dot( b, multiply( ib, av ) ) - 2 ) < 5e-13,
        "Index-2 F3b/F3c failed"
    );
    auto defect = multiply( B, beta );
    for ( unsigned int i = 0; i < 4; ++i )
        defect[i] = 2 * defect[i] - square[i];
    for ( unsigned int power = 1; power <= 2; ++power )
    {
        defect = multiply( B, defect );
        check( std::abs( dot( b, defect ) ) < 2e-14, "ROSI2P2 PDE conditions failed" );
    }
    check(
        t.order() == 3 && t.embedded_order() == 2 && t.supports_constant_mass_index2() &&
            !t.supports_approximate_jacobian() && t.requires_consistent_dense_rates(),
        "Index-2 capability metadata failed"
    );
    for ( const char *method : { "ROS2", "ROS3", "ROS34PW2" } )
        check(
            !rosenbrock::method_table( method ).supports_constant_mass_index2(), "ODE/index-1 table enabled index-2"
        );
}

// Four-variable gauged index-2 system from the support header. Velocity and
// pressure must converge separately, independent of the chosen pressure gauge.
template <bool Krylov = false>
std::array<double, 2> trajectory( double h, pressure_gauge gauge )
{
    using P     = index2_problem<false>;
    using L     = std::conditional_t<Krylov, krylov_solver<P>, index2_solver<P>>;
    using C     = index2_control<adaptation_type, P>;
    using S     = rosenbrock::time_step<operations_type, P, C, L>;
    auto ops    = std::make_shared<operations_type>( 4 );
    auto p      = std::make_shared<P>( ops );
    p->gauge    = gauge;
    auto solver = [&] {
        if constexpr ( Krylov )
            return make_krylov<L>( ops );
        else
            return std::make_shared<L>( ops );
    }();
    auto             cp = std::make_shared<C>( p, adaptation_type::params{ std::abs( h ) } );
    S                step( ops, p, cp, solver, index2_params<S>() );
    vector_wrap_type in( *ops ), out( *ops ), scratch( *ops );
    in.start_use();
    out.start_use();
    scratch.start_use();
    const double start = .37, end = start + std::copysign( .4, h );
    p->initialize_state( start, *in );
    std::array<double, 2> errors{};
    const auto            count = std::lround( .4 / std::abs( h ) );
    for ( long n = 0; n < count; ++n )
    {
        const double time = start + n * h;
        step.set_time( time );
        step.set_target_time( end );
        step.apply( *in, *out );
        check( step.get_status() == single_step_status::converged, "Gauged index-2 step failed" );
        const double next = step.get_dt() == end - time ? end : time + step.get_dt();
        for ( unsigned int i = 0; i < 4; ++i )
            errors[i / 2] =
                std::max( errors[i / 2], std::abs( ops->get_value_at_point( i, *out ) - P::exact( i, next, gauge ) ) );
        check( p->constraint_norm( *out, *scratch ) < 2e-12, "Autonomous constraint/hidden equation/gauge drifted" );
        step.finalize( adaptation_status::accepted, next, *out );
        ops->assign( *out, *in );
    }
    check(
        p->residuals == 4 * count && p->linearizations == count && p->rates == 0,
        "Index-2 stepping requested rates or relinearized"
    );
    return errors;
}

// Non-autonomous ODE execution is independent of the clock-augmented DAE
// formulation. Check its primary/dense orders and the last-stage rate reuse.
std::array<double, 2> ordinary_ode( double h )
{
    using P       = time_problem<>;
    using S       = rosenbrock::time_step<operations_type, P, adaptation_type, direct_solver<P>, true>;
    auto      ops = std::make_shared<operations_type>( 1 );
    auto      p   = std::make_shared<P>( ops );
    auto      c   = std::make_shared<adaptation_type>( adaptation_type::params{ h } );
    auto      l   = std::make_shared<direct_solver<P>>( ops );
    S::params sp;
    sp.method     = "ROSI2P2";
    sp.autonomous = false;
    S                step( ops, p, c, l, sp );
    vector_wrap_type in( *ops ), out( *ops ), sample( *ops );
    in.start_use();
    out.start_use();
    sample.start_use();
    ops->assign_scalar( std::exp( .74 ), *in );
    const auto            count = std::lround( .4 / h );
    std::array<double, 2> errors{};
    for ( long n = 0; n < count; ++n )
    {
        const double t = .37 + n * h;
        step.set_time( t );
        step.set_target_time( .77 );
        step.apply( *in, *out );
        check( step.get_status() == single_step_status::converged, "ROSI2P2 ordinary ODE failed" );
        const double next = step.get_dt() == .77 - t ? .77 : t + step.get_dt();
        errors[0] = std::max( errors[0], std::abs( ops->get_value_at_point( 0, *out ) - std::exp( 2 * next ) ) );
        auto         view = step.get_continuous_integration();
        const double time = view.evaluate( .43, *sample );
        errors[1] = std::max( errors[1], std::abs( ops->get_value_at_point( 0, *sample ) - std::exp( 2 * time ) ) );
        check(
            p->prepared_time == t && p->derivative_time == t && p->time == t,
            "ROSI2P2 ODE time derivative was reinterpreted"
        );
        step.finalize( adaptation_status::accepted, next, *out );
        ops->assign( *out, *in );
    }
    check(
        p->rates == 5 * count && p->derivatives == count && l->calls == 4 * count,
        "ROSI2P2 ODE stage/dense counts changed"
    );
    return errors;
}

int main()
try
{
    backend_type::init_device();
    coefficients();
    const auto ode_a = ordinary_ode( .04 ), ode_b = ordinary_ode( .02 ), ode_c = ordinary_ode( .01 );
    for ( unsigned int i = 0; i < 2; ++i )
        check( ode_a[i] / ode_b[i] > 6. && ode_b[i] / ode_c[i] > 6., "ROSI2P2 ordinary primary/dense order lost" );
    for ( auto gauge : { pressure_gauge::mean_zero, pressure_gauge::pinned } )
        for ( double sign : { 1., -1. } )
        {
            const auto a = trajectory( sign * .04, gauge ), b = trajectory( sign * .02, gauge ),
                       c = trajectory( sign * .01, gauge );
            std::cout << "index2 gauge=" << int( gauge ) << " sign=" << sign << " errors=" << a[0] << ',' << b[0] << ','
                      << c[0] << " pressure=" << a[1] << ',' << b[1] << ',' << c[1] << '\n';
            for ( unsigned int i = 0; i < 2; ++i )
                check( a[i] / b[i] > 6.5 && b[i] / c[i] > 6.5, "Index-2 velocity/pressure order lost" );
            const auto krylov = trajectory<true>( sign * .02, gauge );
            for ( unsigned int i = 0; i < 2; ++i )
                check( std::abs( krylov[i] - b[i] ) < 2e-11, "Coupled GMRES/direct discrepancy" );
        }
    std::cout << "PASS: Rosenbrock index-2 coefficients, gauged velocities/pressures (" << checks << " checks)\n";
}
catch ( const std::exception &e )
{
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
