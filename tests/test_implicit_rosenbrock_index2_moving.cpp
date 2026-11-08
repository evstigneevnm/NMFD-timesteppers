#include "implicit_rosenbrock_index2_problem.h"
#include <nmfd/time_steppers/integration/time_step_adaptation_scipy.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_matlab.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_filter.h>

using namespace rosenbrock_tests;
using P = index2_problem<true>;

// Clock-augmented moving Stokes constraint from the support header. The exact
// clock derivative and all clock Jacobian columns are part of the autonomous
// problem; evaluate_time_derivative is deliberately absent.
std::array<double, 4> trajectory( double h, pressure_gauge gauge )
{
    using S  = rosenbrock::time_step<operations_type, P, adaptation_type, index2_solver<P>>;
    auto ops = std::make_shared<operations_type>( 5 );
    auto p   = std::make_shared<P>( ops );
    p->gauge = gauge;
    S step(
        ops, p, std::make_shared<adaptation_type>( adaptation_type::params{ std::abs( h ) } ),
        std::make_shared<index2_solver<P>>( ops ), index2_params<S>()
    );
    vector_wrap_type in( *ops ), out( *ops ), scratch( *ops );
    in.start_use();
    out.start_use();
    scratch.start_use();
    const double start = .37, end = start + std::copysign( .4, h );
    p->initialize_state( start, *in );
    std::array<double, 4> error{};
    const auto            count = std::lround( .4 / std::abs( h ) );
    for ( long n = 0; n < count; ++n )
    {
        const double time = start + n * h;
        step.set_time( time );
        step.set_target_time( end );
        step.apply( *in, *out );
        check( step.get_status() == single_step_status::converged, "Moving index-2 step failed" );
        const double next = step.get_dt() == end - time ? end : time + step.get_dt();
        for ( unsigned int i = 0; i < 4; ++i )
            error[i / 2] =
                std::max( error[i / 2], std::abs( ops->get_value_at_point( i, *out ) - P::exact( i, next, gauge ) ) );
        check( std::abs( ops->get_value_at_point( 4, *out ) - next ) < 2e-14, "Moving DAE clock lost physical time" );
        error[2] = std::max( error[2], p->constraint_norm( *out, *scratch, false ) );
        error[3] = std::max( error[3], p->constraint_norm( *out, *scratch, true ) );
        check( p->time == time && p->jacobi->a == p->a, "Moving stages changed frozen time or shift" );
        step.finalize( adaptation_status::accepted, next, *out );
        ops->assign( *out, *in );
    }
    check( p->residuals == 4 * count && p->rates == 0, "Moving stages attempted a singular-mass rate solve" );
    return error;
}

void jacobian()
{
    auto             ops = std::make_shared<operations_type>( 5 );
    auto             p   = std::make_shared<P>( ops );
    vector_wrap_type state( *ops ), direction( *ops ), plus( *ops ), minus( *ops ), fp( *ops ), fm( *ops ),
        product( *ops );
    for ( auto *v : { &state, &direction, &plus, &minus, &fp, &fm, &product } )
        v->start_use();
    p->initialize_state( .37, *state );
    ops->assign( *state, p->previous_state() );
    ops->assign_scalar( 0., p->known_rhs() );
    ops->assign_scalar( .3, *direction );
    for ( auto gauge : { pressure_gauge::mean_zero, pressure_gauge::pinned } )
        for ( double b : { 1e-6, .02, 1. } )
        {
            p->gauge = gauge;
            p->set_affine_shift( 1, b );
            p->set_linearization_point( *state );
            const double eps = 2e-6;
            ops->assign_lin_comb( 1., *state, eps, *direction, *plus );
            ops->assign_lin_comb( 1., *state, -eps, *direction, *minus );
            p->apply( *plus, *fp );
            p->apply( *minus, *fm );
            ops->assign_lin_comb( 1 / ( 2 * eps ), *fp, -1 / ( 2 * eps ), *fm, *plus );
            p->jacobi->apply( *direction, *product );
            ops->add_lin_comb( -1., *product, *plus );
            check( ops->norm( *plus ) < 3e-10, "Moving-constraint/force clock Jacobian columns are wrong" );
            check( p->jacobi->solve( *direction, *fp ), "Complete shifted index-2 solve failed" );
            p->jacobi->apply( *fp, *product );
            ops->add_lin_comb( -1., *direction, *product );
            check( ops->norm( *product ) < 3e-10, "Complete shifted index-2 true residual is too large" );
            check(
                ops->get_value_at_point( 4, *p->jacobi->point ) == ops->get_value_at_point( 4, *state ),
                "Trial residual corrupted the frozen clock"
            );
        }
    p->gauge = pressure_gauge::missing;
    p->set_linearization_point( *state );
    check( !p->jacobi->solve( *direction, *fp ), "Missing gauge silently produced a unique direct inverse" );
}

struct capture_estimate : adaptation_type
{
    std::shared_ptr<operations_type> ops;
    vector_wrap_type                 error;
    unsigned int                     error_order = 0;
    capture_estimate( std::shared_ptr<operations_type> o, double h )
        : adaptation_type( params{ h } ), ops( std::move( o ) ), error( *ops )
    {
        error.start_use();
    }
    bool requires_error_estimate() const
    {
        return true;
    }
    adaptation_status assess(
        double, double, const vector_type &, const vector_type &, double &, unsigned int order = 0,
        const vector_type *estimate = nullptr
    )
    {
        check( estimate != nullptr, "Index-2 embedded estimate missing" );
        error_order = order;
        ops->assign( *estimate, *error );
        return adaptation_status::accepted;
    }
};
void estimates()
{
    using S              = rosenbrock::time_step<operations_type, P, capture_estimate, index2_solver<P>>;
    auto             ops = std::make_shared<operations_type>( 5 );
    auto             p   = std::make_shared<P>( ops );
    vector_wrap_type in( *ops ), out( *ops );
    in.start_use();
    out.start_use();
    p->initialize_state( .37, *in );
    std::array<double, 2> previous{};
    for ( double h : { .04, .02, .01 } )
    {
        auto c = std::make_shared<capture_estimate>( ops, h );
        S    step( ops, p, c, std::make_shared<index2_solver<P>>( ops ), index2_params<S>() );
        step.set_time( .37 );
        step.set_target_time( .37 + h );
        step.apply( *in, *out );
        check( step.get_status() == single_step_status::converged, "Moving embedded step failed" );
        check( c->error_order == 2, "Index-2 controller used the ODE exponent for pressure defects" );
        std::array<double, 2> error{};
        for ( unsigned int i = 0; i < 4; ++i )
            error[i / 2] = std::max( error[i / 2], std::abs( ops->get_value_at_point( i, *c->error ) ) );
        std::cout << "index2 embedded h=" << h << " velocity=" << error[0] << " pressure=" << error[1]
                  << " controller order=" << c->error_order << '\n';
        if ( previous[0] )
            check( previous[0] / error[0] > 6., "Index-2 differential estimator lost local order three" );
        if ( previous[1] )
            check( previous[1] / error[1] > 3., "Index-2 pressure estimator lost local order two" );
        previous = error;
    }
}

template <class Controller>
std::array<double, 2> adaptive( double tolerance, const char *filter = nullptr )
{
    using C                         = index2_control<Controller, P>;
    using S                         = rosenbrock::time_step<operations_type, P, C, index2_solver<P>>;
    auto                        ops = std::make_shared<operations_type>( 5 );
    auto                        p   = std::make_shared<P>( ops );
    typename Controller::params cp;
    cp.relative_tolerance = tolerance;
    cp.absolute_tolerance = tolerance * .01;
    cp.initial_step = cp.maximum_step = .2;
    if constexpr ( std::is_same_v<Controller, integration::time_step_adaptation_filter<operations_type>> )
        cp.controller = filter;
    auto c                  = std::make_shared<C>( p, ops, cp );
    c->constraint_tolerance = tolerance * .1;
    auto step = std::make_shared<S>( ops, p, c, std::make_shared<index2_solver<P>>( ops ), index2_params<S>() );
    integration::time_integrator<operations_type, S> run( ops, step );
    run.set_time_interval( .37, .77 );
    vector_wrap_type in( *ops ), out( *ops ), scratch( *ops );
    in.start_use();
    out.start_use();
    scratch.start_use();
    p->initialize_state( .37, *in );
    std::array<double, 2> error{};
    double                first = 0;
    std::size_t           steps = 0;
    for ( unsigned int repeat = 0; repeat < 2; ++repeat )
    {
        run.apply( *in, *out );
        check( run.get_status() == integration_status::completed, "Adaptive moving index-2 solve failed" );
        check(
            p->constraint_norm( *out, *scratch ) <= c->constraint_tolerance,
            "Adaptive index-2 hidden constraint ignored"
        );
        const double result = ops->get_value_at_point( 0, *out );
        if ( !repeat )
        {
            first = result;
            steps = run.get_steps();
        }
        else
            check( result == first && steps == run.get_steps(), "Adaptive index-2 restart retained old clock/history" );
        for ( unsigned int i = 0; i < 4; ++i )
            error[i / 2] =
                std::max( error[i / 2], std::abs( ops->get_value_at_point( i, *out ) - P::exact( i, .77 ) ) );
    }
    check( c->constraint_rejections > 0, "Independent index-2 consistency gate was not exercised" );
    return error;
}

int main()
try
{
    backend_type::init_device();
    static_assert( !detail::has_time_derivative_evaluation<P, vector_type>::value, "Clock must not be a q_t callback" );
    jacobian();
    estimates();
    for ( auto gauge : { pressure_gauge::mean_zero, pressure_gauge::pinned } )
        for ( double sign : { 1., -1. } )
        {
            const auto a = trajectory( sign * .04, gauge ), b = trajectory( sign * .02, gauge ),
                       c = trajectory( sign * .01, gauge );
            std::cout << "moving gauge=" << int( gauge ) << " sign=" << sign << " velocity=" << a[0] << ',' << b[0]
                      << ',' << c[0] << " pressure=" << a[1] << ',' << b[1] << ',' << c[1] << " constraints=" << a[2]
                      << ',' << b[2] << ',' << c[2] << " hidden=" << a[3] << ',' << b[3] << ',' << c[3] << '\n';
            check( a[0] / b[0] > 6. && b[0] / c[0] > 6., "Moving velocity order lost" );
            // ROSI2P2 has third-order differential accuracy here, but raw
            // time-dependent multipliers are second order. Reconstruction of
            // pressure from the differentiated constraint is tested separately.
            for ( unsigned int i : { 1u, 3u } )
                check( a[i] / b[i] > 3.5 && b[i] / c[i] > 3.5, "Moving raw pressure/hidden-defect order lost" );
            check( a[2] < 2e-13 && b[2] < 2e-13 && c[2] < 2e-13, "Moving primary constraint or gauge drifted" );
        }
    auto refine = []( auto tag, const char *filter = nullptr ) {
        using C           = typename decltype( tag )::type;
        const auto coarse = adaptive<C>( 1e-3, filter ), fine = adaptive<C>( 1e-5, filter );
        for ( unsigned int i = 0; i < 2; ++i )
            check( fine[i] < coarse[i] * .15 && fine[i] < 2e-5, "Adaptive index-2 refinement failed" );
    };
    refine( std::common_type<integration::time_step_adaptation_scipy<operations_type>>{} );
    refine( std::common_type<integration::time_step_adaptation_matlab<operations_type>>{} );
    for ( const char *filter : { "I", "H211", "PC", "PID", "H312", "PPID", "H321" } )
        refine( std::common_type<integration::time_step_adaptation_filter<operations_type>>{}, filter );
    std::cout << "PASS: Rosenbrock moving constraints, clock Jacobian, adaptation (" << checks << " checks)\n";
}
catch ( const std::exception &e )
{
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
