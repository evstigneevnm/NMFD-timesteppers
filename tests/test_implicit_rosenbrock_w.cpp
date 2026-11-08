#include "implicit_rosenbrock_w_problem.h"
#include <array>

using namespace rosenbrock_tests;
using row    = std::array<double, 4>;
using matrix = std::array<row, 4>;

// Original, untransformed Rang/Angermann RA34PW2 coefficients, independently
// tabulated in PETSc. Tests below do not use the engine's increment recurrence.
constexpr matrix original_a = {
    { { 0, 0, 0, 0 },
      { .87173304301691801, 0, 0, 0 },
      { .84457060015369423, -.11299064236484185, 0, 0 },
      { 0, 0, 1, 0 } }
};
constexpr matrix original_g = {
    { { .435866521508459, 0, 0, 0 },
      { -.87173304301691801, .435866521508459, 0, 0 },
      { -.90338057013044082, .054180672388095326, .435866521508459, 0 },
      { .24212380706095346, -1.2232505839045147, .54526025533510214, .435866521508459 } }
};
constexpr row original_b   = { .24212380706095346, -1.2232505839045147, 1.545260255335102, .435866521508459 };
constexpr row original_low = { .37810903145819369, -.096042292212423178, .5, .2179332607542295 };

double dot( const row &x, const row &y )
{
    double value = 0;
    for ( std::size_t i = 0; i < 4; ++i )
        value += x[i] * y[i];
    return value;
}
row multiply( const matrix &m, const row &x )
{
    return { dot( m[0], x ), dot( m[1], x ), dot( m[2], x ), dot( m[3], x ) };
}

void test_table_and_w_conditions()
{
    const rosenbrock::method_table table( "ROS34PW2" );
    check(
        table.stage_count() == 4 && table.order() == 3 && table.embedded_order() == 2 && table.error_order() == 3 &&
            table.supports_approximate_jacobian() && table.dense_output_order() == 3,
        "W table capabilities are incorrect"
    );
    matrix g{}, a{};
    row    b{}, low{};
    // Recover Gamma by triangular substitution from -c=lower(Gamma^-1).
    for ( std::size_t i = 0; i < 4; ++i )
        for ( std::size_t j = 0; j <= i; ++j )
        {
            double value = i == j ? 1 : 0;
            for ( std::size_t k = 0; k < i; ++k )
                value += table.c( i, k ) * g[k][j];
            g[i][j] = table.gamma * value;
        }
    for ( std::size_t i = 0; i < 4; ++i )
    {
        for ( std::size_t j = 0; j < 4; ++j )
        {
            for ( std::size_t k = 0; k < 4; ++k )
                a[i][j] += table.a( i, k ) * g[k][j];
            b[i] += table.m( j ) * g[j][i];
            low[i] += ( table.m( j ) - table.e( j ) ) * g[j][i];
            check(
                std::abs( g[i][j] - original_g[i][j] ) < 3e-14 && std::abs( a[i][j] - original_a[i][j] ) < 3e-14,
                "Transformed W table differs from the published table"
            );
        }
        check(
            std::abs( b[i] - original_b[i] ) < 3e-14 && std::abs( low[i] - original_low[i] ) < 3e-14,
            "W endpoint or embedded weights use the wrong transformation"
        );
        check( !table.reuse_previous_rate( i ), "W table reused a distinct physical rate" );
    }
    const row one = { 1, 1, 1, 1 }, times = multiply( a, one ), gamma_times = multiply( g, one );
    row       squared{};
    for ( std::size_t i = 0; i < 4; ++i )
    {
        squared[i] = times[i] * times[i];
        check(
            std::abs( table.stage_time( i ) - times[i] ) < 3e-14 &&
                std::abs( table.time_derivative( i ) - gamma_times[i] ) < 3e-14,
            "W time augmentation is inconsistent with its table"
        );
    }
    // Colored-tree conditions separate J and the arbitrary approximation B:
    // F, JF, BF, JJF, JBF, BJF, BBF and F''(F,F), respectively.
    const row primary = { dot( b, one ), dot( b, times ), dot( b, gamma_times ), dot( b, multiply( a, times ) ) };
    const row exact   = { 1, .5, 0, 1. / 6 };
    for ( std::size_t i = 0; i < 4; ++i )
        check( std::abs( primary[i] - exact[i] ) < 3e-14, "W primary order condition failed" );
    for ( double value :
          { dot( b, multiply( a, gamma_times ) ), dot( b, multiply( g, times ) ),
            dot( b, multiply( g, gamma_times ) ) } )
        check( std::abs( value ) < 3e-14, "Approximate-Jacobian colored-tree condition failed" );
    check( std::abs( dot( b, squared ) - 1. / 3 ) < 3e-14, "W nonlinear branching condition failed" );
    check(
        std::abs( dot( low, one ) - 1 ) < 3e-14 && std::abs( dot( low, times ) - .5 ) < 3e-14 &&
            std::abs( dot( low, gamma_times ) ) < 3e-14,
        "W embedded order conditions failed"
    );
    for ( bool coupling : { false, true } )
    {
        bool rejected = false;
        try
        {
            if ( coupling )
                table.a( 0, 4 );
            else
                table.m( 4 );
        }
        catch ( const std::out_of_range & )
        {
            rejected = true;
        }
        check( rejected, "W table accepted an invalid stage index" );
    }
    const auto weights = rosenbrock::continuous_weights( table, .5 );
    check(
        std::all_of( weights.begin(), weights.end(), []( double value ) { return value == 0; } ),
        "W secant correction is inconsistent at the midpoint"
    );
}

template <class P, bool Krylov = false>
double integrate(
    std::shared_ptr<operations_type> ops, std::shared_ptr<P> p, double h, double start, double end, double initial,
    bool autonomous = true, double reference_time = 1
)
{
    using L     = std::conditional_t<Krylov, krylov_solver<P>, direct_solver<P>>;
    using S     = rosenbrock::time_step<operations_type, P, adaptation_type, L>;
    auto linear = [&] {
        if constexpr ( Krylov )
            return make_krylov<L>( ops );
        else
            return std::make_shared<L>( ops );
    }();
    typename S::params settings;
    settings.method         = "ROS34PW2";
    settings.autonomous     = autonomous;
    settings.reference_time = reference_time;
    auto step               = std::make_shared<S>(
        ops, p, std::make_shared<adaptation_type>( adaptation_type::params{ h } ), linear, settings
    );
    vector_wrap_type in( *ops ), out( *ops );
    in.start_use();
    out.start_use();
    ops->assign_scalar( 0., *in );
    ops->set_value_at_point( initial, 0, *in );
    integration::time_integrator<operations_type, S> run( ops, step );
    run.set_time_interval( start, end );
    run.apply( *in, *out );
    check(
        run.get_status() == integration_status::completed && run.get_final_time() == end,
        "W trajectory did not reach its target"
    );
    check(
        p->rates == 4 * run.get_steps() && p->linearizations == run.get_steps() && p->residuals == 0,
        "W stepping repeated preparation or skipped a physical rate"
    );
    if constexpr ( !Krylov )
        check(
            linear->bindings == run.get_steps() && linear->calls == 4 * linear->bindings,
            "W stepping did not use four solves and one binding per attempt"
        );
    return ops->get_value_at_point( 0, *out );
}

// Nonlinear logistic growth must retain order three even with an O(1) error
// in B, B=0, or a Jacobian frozen at the initial state for the whole trajectory.
void test_approximate_orders()
{
    for ( unsigned int approximation = 0; approximation < 4; ++approximation )
    {
        auto error = [&]( double h ) {
            auto ops        = std::make_shared<operations_type>( 1 );
            auto p          = std::make_shared<w_problem<logistic>>( ops );
            p->model.factor = approximation == 1 ? .3 : approximation == 2 ? 0 : 1;
            p->model.offset = approximation == 1 ? -.4 : 0;
            p->lagged       = approximation == 3;
            return std::abs( integrate( ops, p, h, 0, 1, .25 ) - 1 / ( 1 + 3 * std::exp( -1. ) ) );
        };
        const double e1 = error( .04 ), e2 = error( .02 ), e3 = error( .01 );
        std::cout << "ROS34PW2 approximation=" << approximation << " errors=" << e1 << ',' << e2 << ',' << e3 << '\n';
        check(
            e1 / e2 > 7.1 && e1 / e2 < 9 && e2 / e3 > 7.1 && e2 / e3 < 9,
            "W method lost third order with an approximate Jacobian"
        );
    }
    // Noncommuting 2x2 J/B catches errors which scalar or diagonal tests miss.
    for ( bool krylov : { false, true } )
    {
        auto error = [&]( double h ) {
            auto         ops   = std::make_shared<operations_type>( 2 );
            auto         p     = std::make_shared<problem<noncommuting_decay>>( ops );
            const double value = krylov ? integrate<decltype( p )::element_type, true>( ops, p, h, 0, .4, 1 )
                                        : integrate( ops, p, h, 0, .4, 1 );
            return std::abs( value - ( std::exp( -.4 ) + std::exp( -1.2 ) ) / 2 );
        };
        const double e1 = error( .02 ), e2 = error( .01 );
        check( e1 / e2 > 7 && e1 / e2 < 9, "W method failed noncommuting coupled order test" );
    }
}

// u'=j(t)u+(2-j(t))*exp(2t), exact exp(2t), has a known partial f_t.
// B differs from J independently of h; test both time directions and stiffness.
void test_time_and_scaling()
{
    for ( bool varying : { false, true } )
        for ( bool backward : { false, true } )
        {
            auto error = [&]( double h ) {
                auto ops           = std::make_shared<operations_type>( 1 );
                auto p             = std::make_shared<w_time_problem<>>( ops );
                p->varying         = varying;
                p->lambda          = 5;
                const double start = .37, end = backward ? .17 : .57;
                const double value = integrate( ops, p, h, start, end, std::exp( 2 * start ), false );
                check(
                    p->derivatives == p->linearizations && p->time == p->prepared_time,
                    "W time derivative was repeated or base time was not restored"
                );
                check(
                    std::abs( p->last_rate_time - ( p->prepared_time + ( backward ? -1 : 1 ) * h ) ) < 1e-13,
                    "W final stage did not use its endpoint time"
                );
                return std::abs( value - std::exp( 2 * end ) );
            };
            const double e1 = error( .02 ), e2 = error( .01 );
            std::cout << "ROS34PW2 time varying=" << varying << " backward=" << backward << " errors=" << e1 << ','
                      << e2 << '\n';
            check( e1 / e2 > 6.5 && e1 / e2 < 9.5, "Non-autonomous W method lost third order" );
        }
    for ( double h : { 1e-100, 1e-14, .01, 1., 1e14, 1e100, 1e300 } )
    {
        auto ops = std::make_shared<operations_type>( 1 );
        check(
            std::abs( integrate( ops, std::make_shared<problem<constant_rate>>( ops ), h, 0, h, 0 ) / h - 1 ) < 1e-14,
            "W balancing changed a constant physical rate"
        );
    }
    double previous = 1;
    for ( double lambda : { 1., 10., 1e3, 1e8 } )
    {
        auto ops = std::make_shared<operations_type>( 1 );
        // Independent original-stage solve for u'=-lambda*u with B=J.
        row k{};
        for ( std::size_t i = 0; i < 4; ++i )
            k[i] =
                -lambda * ( 1 + dot( original_a[i], k ) + dot( original_g[i], k ) ) / ( 1 + lambda * original_g[i][i] );
        const double expected = 1 + dot( original_b, k );
        const double value    = integrate( ops, std::make_shared<problem<decay>>( ops, decay{ lambda } ), 1, 0, 1, 1 );
        check(
            std::abs( value - expected ) < 2e-13 && std::abs( value ) <= previous,
            "W method disagrees with its exact-Jacobian stability function"
        );
        previous = std::abs( value );
    }
    check( previous < 1e-6, "W exact-Jacobian stiff limit is not damped" );
}

int main()
try
{
    backend_type::init_device();
    test_table_and_w_conditions();
    test_approximate_orders();
    test_time_and_scaling();
    std::cout << "ROS34PW2 table, W order conditions, approximate/noncommuting flows and timing: " << checks
              << " checks PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
