#include "implicit_rosenbrock_time_problem.h"

using namespace rosenbrock_tests;

// The four exact-Jacobian elementary differentials are F, F'F, F'^2 F,
// F''(F,F). Time augmentation covers the corresponding partial-time terms.
void test_order_conditions()
{
    for ( const char *method : { "ROS2", "ROS3" } )
    {
        const rosenbrock::method_table       table( method );
        const unsigned int                   count = table.stage_count() == 2 ? 2 : 4;
        std::array<std::array<double, 4>, 4> terms{};
        for ( unsigned int i = 0; i < count; ++i )
        {
            double time = 0, a_e = 0, c_d = 0, c_e = 0, c_p = 0, c_q = 0;
            for ( unsigned int j = 0; j < i; ++j )
            {
                const double a = i == 3 ? table.m( j ) : table.a( i, j );
                const double c = i == 3 ? -table.m( j ) : table.c( i, j );
                time += a * terms[0][j];
                a_e += a * terms[1][j];
                c_d += c * terms[0][j];
                c_e += c * terms[1][j];
                c_p += c * terms[2][j];
                c_q += c * terms[3][j];
            }
            terms[0][i] = table.gamma * ( 1 + c_d );
            terms[1][i] = table.gamma * ( terms[0][i] + time + c_e );
            terms[2][i] = table.gamma * ( terms[1][i] + a_e + c_p );
            terms[3][i] = table.gamma * ( time * time / 2 + c_q );
            if ( i < table.stage_count() )
                check(
                    std::abs( time - table.stage_time( i ) ) < 2e-14 &&
                        std::abs( terms[0][i] - table.time_derivative( i ) ) < 2e-14,
                    "Rosenbrock time augmentation is inconsistent"
                );
        }
        for ( double theta : { 0., .01, .25, .5, .9, 1. } )
        {
            auto weights = rosenbrock::continuous_weights( table, theta );
            for ( unsigned int i = 0; i < table.stage_count(); ++i )
                weights[i] += theta * table.m( i );
            const std::array<double, 4> expected{
                theta, theta * theta / 2, theta * theta * theta / 6, theta * theta * theta / 6
            };
            for ( unsigned int differential = 0; differential < ( count == 2 ? 2u : 4u ); ++differential )
            {
                double coefficient = 0;
                for ( unsigned int i = 0; i < count; ++i )
                    coefficient += weights[i] * terms[differential][i];
                check(
                    std::abs( coefficient - expected[differential] ) < 2e-13,
                    "Dense extension violates a rooted-tree order condition"
                );
            }
        }
    }
}

// Integrate and sample each pending step before commitment. Comparing against a
// dense-disabled stepper checks that extension preparation never changes endpoints.
template <class P, class Exact>
double trajectory(
    const char *method, std::shared_ptr<operations_type> ops, std::shared_ptr<P> p, std::shared_ptr<P> plain_problem,
    double start, double end, double h, Exact exact, bool autonomous
)
{
    using L     = direct_solver<P>;
    using S     = rosenbrock::time_step<operations_type, P, adaptation_type, L, true>;
    using Plain = rosenbrock::time_step<operations_type, P, adaptation_type, L>;
    typename S::params settings;
    settings.method     = method;
    settings.autonomous = autonomous;
    typename Plain::params ps;
    ps.method     = method;
    ps.autonomous = autonomous;
    auto  linear  = std::make_shared<L>( ops );
    S     step( ops, p, std::make_shared<adaptation_type>( adaptation_type::params{ h } ), linear, settings );
    Plain plain(
        ops, plain_problem, std::make_shared<adaptation_type>( adaptation_type::params{ h } ),
        std::make_shared<L>( ops ), ps
    );
    vector_wrap_type state( *ops ), out( *ops ), reference( *ops ), sample( *ops );
    state.start_use();
    out.start_use();
    reference.start_use();
    sample.start_use();
    ops->assign_scalar( exact( start ), *state );
    double             error = 0;
    unsigned int       steps = 0;
    const unsigned int count = std::lround( std::abs( end - start ) / h );
    for ( unsigned int i = 0; i < count; ++i )
    {
        const double t    = start + ( end - start ) * i / count,
                     next = i + 1 == count ? end : start + ( end - start ) * ( i + 1 ) / count;
        step.set_time( t );
        step.set_target_time( next );
        plain.set_time( t );
        plain.set_target_time( next );
        step.apply( *state, *out );
        plain.apply( *state, *reference );
        check(
            step.get_status() == single_step_status::converged && plain.get_status() == single_step_status::converged,
            "Rosenbrock dense trajectory failed"
        );
        check(
            ops->get_value_at_point( 0, *out ) == ops->get_value_at_point( 0, *reference ),
            "Dense preparation changed the numerical endpoint"
        );
        const auto view  = step.get_continuous_integration();
        const auto rates = p->rates, solves = linear->calls;
        for ( double theta : { 0., .25, .5, .75, 1. } )
        {
            const double time = view.evaluate( theta, *sample ), value = ops->get_value_at_point( 0, *sample );
            error = std::max( error, std::abs( value - exact( time ) ) );
            if ( theta == 0 || theta == 1 )
                check(
                    value == ops->get_value_at_point( 0, theta == 0 ? *state : *out ),
                    "Dense extension changed an endpoint"
                );
        }
        check(
            p->rates == rates && linear->calls == solves && p->time == t,
            "Sampling invoked the problem/solver or changed time"
        );
        step.finalize( adaptation_status::accepted, next, *out );
        plain.finalize( adaptation_status::accepted, next, *reference );
        ops->assign( *out, *state );
        ++steps;
    }
    const bool ros3 = step.table().order() == 3;
    check(
        step.dense_output_order() == step.table().order() && plain.dense_output_order() == 0 &&
            p->rates == ( ros3 ? 3u : 2u ) * steps && linear->calls == ( ros3 ? 4u : 2u ) * steps &&
            linear->bindings == steps,
        "Dense output performed unexpected rate evaluations, solves or rebindings"
    );
    return error;
}

template <class Model, class Exact>
double autonomous_error( const char *method, Model model, double h, Exact exact )
{
    auto ops = std::make_shared<operations_type>( 1 );
    return trajectory(
        method, ops, std::make_shared<problem<Model>>( ops, model ), std::make_shared<problem<Model>>( ops, model ), 0,
        .2, h, exact, true
    );
}

void test_orders()
{
    for ( const char *method : { "ROS2", "ROS3" } )
        for ( unsigned int model = 0; model < 4; ++model )
        {
            auto error = [&]( double h ) {
                if ( model == 0 )
                    return autonomous_error( method, decay{}, h, []( double t ) { return std::exp( -t ); } );
                if ( model == 1 )
                    return autonomous_error( method, logistic{}, h, []( double t ) {
                        return 1 / ( 1 + 3 * std::exp( -t ) );
                    } );
                auto ops = std::make_shared<operations_type>( 1 );
                auto p = std::make_shared<time_problem<>>( ops ), plain = std::make_shared<time_problem<>>( ops );
                p->varying = plain->varying = model == 2;
                p->lambda = plain->lambda = 10;
                return trajectory(
                    method, ops, p, plain, .37, model == 2 ? .57 : .17, h, []( double t ) { return std::exp( 2 * t ); },
                    false
                );
            };
            const double       h = model == 3 ? .0025 : .02, e1 = error( h ), e2 = error( h / 2 );
            const unsigned int order = rosenbrock::method_table( method ).order();
            std::cout << method << " dense model=" << model << " errors=" << e1 << ',' << e2
                      << " order=" << std::log2( e1 / e2 ) << '\n';
            check(
                e1 / e2 > .8 * ( 1u << order ) && e1 / e2 < 1.2 * ( 1u << order ),
                "Dense output lost its reported continuous order"
            );
        }
}

// Large negative lambda*h tests the bounded rational extension, not a claim of
// uniform interpolation accuracy inside an unresolved initial boundary layer.
void test_stiff_decay()
{
    for ( const char *method : { "ROS2", "ROS3" } )
        for ( double lambda : { 1., 10., 1e3, 1e12 } )
        {
            using P  = problem<decay>;
            using L  = direct_solver<P>;
            using S  = rosenbrock::time_step<operations_type, P, adaptation_type, L, true>;
            auto ops = std::make_shared<operations_type>( 1 );
            auto p   = std::make_shared<P>( ops, decay{ lambda } );
            S    step(
                ops, p, std::make_shared<adaptation_type>( adaptation_type::params{ 1 } ), std::make_shared<L>( ops ),
                typename S::params{ method }
            );
            vector_wrap_type state( *ops ), out( *ops );
            state.start_use();
            out.start_use();
            ops->assign_scalar( 1., *state );
            step.apply( *state, *out );
            check( step.get_status() == single_step_status::converged, "Stiff dense preparation failed" );
            const auto view = step.get_continuous_integration();
            for ( double theta : { .1, .25, .5, .75, .9 } )
            {
                view.evaluate( theta, *out );
                const double value = ops->get_value_at_point( 0, *out );
                check(
                    std::isfinite( value ) && std::abs( value ) < 2, "Dense stiff-decay samples grew with lambda*h"
                );
                if ( lambda == 1e12 )
                {
                    const auto   w = rosenbrock::continuous_weights( step.table(), theta );
                    const double limit =
                        step.table().order() == 2
                            ? 1 - theta - theta * ( 1 - theta ) / ( 2 * step.table().gamma * step.table().gamma )
                            : 1 - theta - w[0];
                    check( std::abs( value - limit ) < 1e-9, "Dense stiff-limit polynomial mismatch" );
                }
            }
            step.finalize( adaptation_status::accepted, 1, *out );
        }
}

int main()
try
{
    backend_type::init_device();
    test_order_conditions();
    test_orders();
    test_stiff_decay();
    std::cout << "Rosenbrock native dense orders, exact endpoints, stiff limits and callback-free sampling: " << checks
              << " checks PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
