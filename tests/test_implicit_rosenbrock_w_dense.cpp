#include "implicit_rosenbrock_w_problem.h"
#include <array>

using namespace rosenbrock_tests;
using row    = std::array<double, 6>;
using matrix = std::array<row, 6>;

// Original four-stage RA34PW2 table, independent of the engine's transformed
// recurrence. Dense stages evaluate at the base and at the primary endpoint.
struct extended_table
{
    matrix a = {
        { { 0, 0, 0, 0 },
          { .87173304301691801, 0, 0, 0 },
          { .84457060015369423, -.11299064236484185, 0, 0 },
          { 0, 0, 1, 0 } }
    };
    matrix g = {
        { { .435866521508459, 0, 0, 0 },
          { -.87173304301691801, .435866521508459, 0, 0 },
          { -.90338057013044082, .054180672388095326, .435866521508459, 0 },
          { .24212380706095346, -1.2232505839045147, .54526025533510214, .435866521508459 } }
    };
    row b = { .24212380706095346, -1.2232505839045147, 1.545260255335102, .435866521508459 };
    extended_table()
    {
        for ( std::size_t i = 0; i < 2; ++i )
        {
            const auto &w = rosenbrock::ros34pw2_dense_coupling( i );
            for ( std::size_t j = 0; j < 4; ++j )
            {
                a[5][j] = b[j];
                for ( std::size_t k = 0; k < 4; ++k )
                    g[4 + i][j] += w[k] * g[k][j];
            }
            g[4 + i][4 + i] = g[0][0];
        }
    }
};
double dot( const row &a, const row &b )
{
    double value = 0;
    for ( std::size_t i = 0; i < a.size(); ++i )
        value += a[i] * b[i];
    return value;
}
row multiply( const matrix &a, const row &x )
{
    row out{};
    for ( std::size_t i = 0; i < out.size(); ++i )
        out[i] = dot( a[i], x );
    return out;
}

void test_w_conditions()
{
    const extended_table t;
    const row            one = { 1, 1, 1, 1, 1, 1 }, alpha = multiply( t.a, one ), gamma = multiply( t.g, one );
    row                  squared{};
    for ( std::size_t i = 0; i < 6; ++i )
        squared[i] = alpha[i] * alpha[i];
    const std::array<row, 8> terms = { one,
                                       alpha,
                                       gamma,
                                       multiply( t.a, alpha ),
                                       multiply( t.a, gamma ),
                                       multiply( t.g, alpha ),
                                       multiply( t.g, gamma ),
                                       squared };
    for ( std::size_t i = 4; i < 6; ++i )
        check(
            std::abs( gamma[i] ) < 3e-14 && std::abs( terms[5][i] ) < 3e-14 && std::abs( terms[6][i] ) < 3e-14,
            "W dense slopes retain spurious B terms"
        );
    const rosenbrock::method_table table( "ROS34PW2" );
    for ( double theta : { 0., .01, .1, .25, .5, .75, .9, .99, 1. } )
    {
        const double h01 = theta * theta * ( 3 - 2 * theta ), h10 = theta * ( 1 - theta ) * ( 1 - theta ),
                     h11 = -theta * theta * ( 1 - theta );
        row weights{};
        for ( std::size_t i = 0; i < 4; ++i )
            weights[i] = h01 * t.b[i];
        weights[4] = h10;
        weights[5] = h11;
        // F, JF, BF, JJF, JBF, BJF, BBF, F''(F,F), keeping B distinct from J.
        const std::array<double, 8> expected = { theta, theta * theta / 2,        0, theta * theta * theta / 6, 0, 0,
                                                 0,     theta * theta * theta / 3 };
        for ( std::size_t i = 0; i < terms.size(); ++i )
            check(
                std::abs( dot( weights, terms[i] ) - expected[i] ) < 3e-13,
                "Dense extension violates an independent W order condition"
            );
        const auto transformed = rosenbrock::continuous_weights( table, theta );
        for ( std::size_t i = 0; i < 4; ++i )
            check(
                std::abs( transformed[i] + theta * table.m( i ) - h01 * table.m( i ) ) < 3e-14,
                "W dense secant weights use the wrong increment convention"
            );
    }
    bool invalid = false;
    try
    {
        rosenbrock::ros34pw2_dense_coupling( 2 );
    }
    catch ( const std::out_of_range & )
    {
        invalid = true;
    }
    check( invalid, "Dense stage coefficients accepted an invalid stage" );
}

// Sample each pending step, checking exact endpoint copies and equality to
// dense-disabled stepping on the same grid. Exact solutions supply every component.
template <class P, class Exact>
double trajectory(
    std::shared_ptr<operations_type> ops, std::shared_ptr<P> p, std::shared_ptr<P> pp, double start, double end,
    double h, Exact exact, bool autonomous
)
{
    using L     = direct_solver<P>;
    using S     = rosenbrock::time_step<operations_type, P, adaptation_type, L, true>;
    using Plain = rosenbrock::time_step<operations_type, P, adaptation_type, L>;
    typename S::params sp;
    sp.method     = "ROS34PW2";
    sp.autonomous = autonomous;
    typename Plain::params ps;
    ps.method     = sp.method;
    ps.autonomous = autonomous;
    auto  linear  = std::make_shared<L>( ops );
    S     step( ops, p, std::make_shared<adaptation_type>( adaptation_type::params{ h } ), linear, sp );
    Plain plain(
        ops, pp, std::make_shared<adaptation_type>( adaptation_type::params{ h } ), std::make_shared<L>( ops ), ps
    );
    vector_wrap_type state( *ops ), out( *ops ), reference( *ops ), sample( *ops );
    for ( auto *v : { &state, &out, &reference, &sample } )
        v->start_use();
    for ( std::ptrdiff_t i = 0; i < ops->size(); ++i )
        ops->set_value_at_point( exact( start, i ), i, *state );
    const auto count = static_cast<unsigned int>( std::lround( std::abs( end - start ) / h ) );
    double     error = 0;
    for ( unsigned int i = 0; i < count; ++i )
    {
        const double time = start + ( end - start ) * i / count,
                     next = i + 1 == count ? end : start + ( end - start ) * ( i + 1 ) / count;
        step.set_time( time );
        step.set_target_time( next );
        plain.set_time( time );
        plain.set_target_time( next );
        step.apply( *state, *out );
        plain.apply( *state, *reference );
        check(
            step.get_status() == single_step_status::converged && plain.get_status() == single_step_status::converged,
            "W dense trajectory failed"
        );
        for ( std::ptrdiff_t j = 0; j < ops->size(); ++j )
            check(
                ops->get_value_at_point( j, *out ) == ops->get_value_at_point( j, *reference ),
                "W dense preparation changed the primary endpoint"
            );
        const auto rates = p->rates, solves = linear->calls;
        const auto view = step.get_continuous_integration();
        for ( double theta : { 0., .125, .25, .5, .75, .875, 1. } )
        {
            const auto t = view.evaluate( theta, *sample );
            for ( std::ptrdiff_t j = 0; j < ops->size(); ++j )
            {
                const double value = ops->get_value_at_point( j, *sample );
                error              = std::max( error, std::abs( value - exact( t, j ) ) );
                if ( theta == 0 || theta == 1 )
                    check(
                        value == ops->get_value_at_point( j, theta == 0 ? *state : *out ),
                        "W dense endpoint is not an exact copy"
                    );
            }
        }
        check(
            p->rates == rates && linear->calls == solves && p->time == time,
            "W sampling called the problem/solver or changed its base time"
        );
        step.finalize( adaptation_status::accepted, next, *out );
        plain.finalize( adaptation_status::accepted, next, *reference );
        ops->assign( *out, *state );
    }
    check(
        step.dense_output_order() == 3 && plain.dense_output_order() == 0 && p->rates == 5 * count &&
            linear->calls == 6 * count && linear->bindings == count && p->linearizations == count,
        "W dense output performed unexpected work"
    );
    return error;
}

// Logistic growth tests nonlinear order with exact, O(1)-perturbed, zero and
// lagged B. The frozen approximation is independent of the requested timestep.
void test_orders()
{
    for ( unsigned int approximation = 0; approximation < 4; ++approximation )
    {
        auto error = [&]( double h, double end ) {
            auto ops = std::make_shared<operations_type>( 1 );
            auto p = std::make_shared<w_problem<logistic>>( ops ), pp = std::make_shared<w_problem<logistic>>( ops );
            p->model.factor = pp->model.factor = approximation == 1 ? .3 : approximation == 2 ? 0 : 1;
            p->model.offset = pp->model.offset = approximation == 1 ? -.4 : 0;
            p->lagged = pp->lagged = approximation == 3;
            return trajectory(
                ops, p, pp, 0, end, h, []( double t, std::ptrdiff_t ) { return 1 / ( 1 + 3 * std::exp( -t ) ); }, true
            );
        };
        const double e1 = error( .04, 1 ), e2 = error( .02, 1 ), e3 = error( .01, 1 );
        std::cout << "W dense approximation=" << approximation << " errors=" << e1 << ',' << e2 << ',' << e3 << '\n';
        check( e1 / e2 > 7 && e1 / e2 < 9.3 && e2 / e3 > 7 && e2 / e3 < 9.3, "W dense output lost global order three" );
        const double local1 = error( .04, .04 ), local2 = error( .02, .02 );
        check( local1 / local2 > 12 && local1 / local2 < 20, "W dense output lost its O(h^4) one-step accuracy" );
    }
    // x'=-2x+y, y'=x-2y, with a noncommuting B. Test both exact modes.
    auto coupled_error = []( double h ) {
        auto ops = std::make_shared<operations_type>( 2 );
        return trajectory(
            ops, std::make_shared<problem<noncommuting_decay>>( ops ),
            std::make_shared<problem<noncommuting_decay>>( ops ), 0, .4, h,
            []( double t, std::ptrdiff_t i ) { return ( std::exp( -t ) + ( i ? -1 : 1 ) * std::exp( -3 * t ) ) / 2; },
            true
        );
    };
    const double ratio = coupled_error( .02 ) / coupled_error( .01 );
    check( ratio > 7 && ratio < 9.3, "Noncommuting W dense output lost order three" );
    // u'=j(t)u+(2-j(t))*exp(2t) has a known non-autonomous exact solution.
    for ( bool varying : { false, true } )
        for ( bool backward : { false, true } )
        {
            auto error = [&]( double h ) {
                auto ops = std::make_shared<operations_type>( 1 );
                auto p = std::make_shared<w_time_problem<>>( ops ), pp = std::make_shared<w_time_problem<>>( ops );
                p->varying = pp->varying = varying;
                p->lambda = pp->lambda = 5;
                return trajectory(
                    ops, p, pp, .37, backward ? .17 : .57, h,
                    []( double t, std::ptrdiff_t ) { return std::exp( 2 * t ); }, false
                );
            };
            const double e1 = error( .02 ), e2 = error( .01 );
            std::cout << "W dense time varying=" << varying << " backward=" << backward << " errors=" << e1 << ',' << e2
                      << '\n';
            check( e1 / e2 > 6.5 && e1 / e2 < 9.5, "Non-autonomous W dense output lost order three" );
        }
}

// Compare against six independent original-stage scalar equations. Large
// negative lambda*h checks bounded samples, not accuracy in unresolved layers
// or unconditional stability for an inaccurate Jacobian approximation.
void test_stiff_limits()
{
    const extended_table table;
    for ( double factor : { .5, 1., 2. } )
        for ( double lambda : { 1., 10., 1e3, 1e12 } )
        {
            using P         = w_problem<decay>;
            using L         = direct_solver<P>;
            using S         = rosenbrock::time_step<operations_type, P, adaptation_type, L, true>;
            auto ops        = std::make_shared<operations_type>( 1 );
            auto p          = std::make_shared<P>( ops );
            p->model.lambda = lambda;
            p->model.factor = factor;
            S step(
                ops, p, std::make_shared<adaptation_type>( adaptation_type::params{ 1 } ), std::make_shared<L>( ops ),
                S::params{ "ROS34PW2" }
            );
            vector_wrap_type state( *ops ), out( *ops ), sample( *ops );
            state.start_use();
            out.start_use();
            sample.start_use();
            ops->assign_scalar( 1., *state );
            step.apply( *state, *out );
            check( step.get_status() == single_step_status::converged, "Stiff W dense preparation failed" );
            row k{};
            for ( std::size_t i = 0; i < 6; ++i )
                k[i] = ( -lambda * ( 1 + dot( table.a[i], k ) ) - factor * lambda * dot( table.g[i], k ) ) /
                       ( 1 + factor * lambda * table.g[i][i] );
            const double endpoint = 1 + dot( table.b, k );
            const auto   view     = step.get_continuous_integration();
            for ( double theta : { 0., .1, .25, .5, .75, .9, 1. } )
            {
                view.evaluate( theta, *sample );
                const double value    = ops->get_value_at_point( 0, *sample );
                const double expected = ( 1 - 3 * theta * theta + 2 * theta * theta * theta ) +
                                        ( 3 * theta * theta - 2 * theta * theta * theta ) * endpoint +
                                        theta * ( 1 - theta ) * ( 1 - theta ) * k[4] -
                                        theta * theta * ( 1 - theta ) * k[5];
                check(
                    std::isfinite( value ) && std::abs( value ) < 20 && std::abs( value - expected ) < 3e-12,
                    "Stiff W dense output differs from the independent rational extension"
                );
            }
            step.finalize( adaptation_status::accepted, 1, *out );
        }
}

// u'=1, u(0)=0 isolates the signed timestep and balanced shift scaling.
// Exact interior values remain theta*h even at extreme representable steps.
void test_scaling()
{
    using P              = problem<constant_rate>;
    using L              = direct_solver<P>;
    using S              = rosenbrock::time_step<operations_type, P, adaptation_type, L, true>;
    auto             ops = std::make_shared<operations_type>( 1 );
    vector_wrap_type in( *ops ), out( *ops ), sample( *ops );
    in.start_use();
    out.start_use();
    sample.start_use();
    ops->assign_scalar( 0., *in );
    for ( double h : { 1e-100, 1e-14, .1, 1., 1e14, 1e100, 1e300 } )
        for ( double reference_time : { 1e-6, 1., 1e6 } )
        {
            S::params settings;
            settings.method         = "ROS34PW2";
            settings.reference_time = reference_time;
            S step(
                ops, std::make_shared<P>( ops ), std::make_shared<adaptation_type>( adaptation_type::params{ h } ),
                std::make_shared<L>( ops ), settings
            );
            step.set_target_time( h );
            step.apply( *in, *out );
            check( step.get_status() == single_step_status::converged, "Scaled W dense preparation failed" );
            const auto view = step.get_continuous_integration();
            for ( double theta : { .125, .5, .875 } )
            {
                view.evaluate( theta, *sample );
                check(
                    std::abs( ops->get_value_at_point( 0, *sample ) / h - theta ) < 3e-14,
                    "W continuous extension lost timestep/reference-time scaling"
                );
            }
            step.finalize( adaptation_status::accepted, h, *out );
        }
}

int main()
try
{
    backend_type::init_device();
    test_w_conditions();
    test_orders();
    test_stiff_limits();
    test_scaling();
    std::cout << "W-compatible dense conditions, approximate/noncommuting orders, endpoints and stiff limits: "
              << checks << " checks PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
