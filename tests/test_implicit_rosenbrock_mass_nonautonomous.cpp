#include "implicit_rosenbrock_mass_time_problem.h"
#include <nmfd/time_steppers/integration/time_step_adaptation_filter.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_matlab.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_scipy.h>

using namespace rosenbrock_tests;

struct initial_time_state
{
    vector_type         out;
    double              time;
    __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
    {
        out( i ) = exp( ( i % 2 + 1 ) * time );
    }
};

// Compare M*q(t,u) residual stages with an independently assembled q(t,u)
// rate path. Exact g(t)=(exp(t),exp(2*t)) detects partial/total-time confusion.
template <bool Dense>
std::pair<double, double> fixed( const char *method, manufactured_time_model model, double h, bool backward = false )
{
    using P                = mass_time_problem<>;
    using Q                = rate_time_problem<>;
    using S                = rosenbrock::time_step<operations_type, P, adaptation_type, direct_solver<P>, Dense>;
    using R                = rosenbrock::time_step<operations_type, Q, adaptation_type, direct_solver<Q>, Dense>;
    auto               ops = std::make_shared<operations_type>( 2 );
    auto               p   = std::make_shared<P>( ops, model );
    auto               q   = std::make_shared<Q>( ops, model );
    typename S::params sp;
    sp.method     = method;
    sp.autonomous = false;
    sp.mass_form  = true;
    typename R::params rp;
    rp.method     = method;
    rp.autonomous = false;
    auto solver   = std::make_shared<direct_solver<P>>( ops );
    S    step( ops, p, std::make_shared<adaptation_type>( adaptation_type::params{ h } ), solver, sp );
    R    reference(
        ops, q, std::make_shared<adaptation_type>( adaptation_type::params{ h } ),
        std::make_shared<direct_solver<Q>>( ops ), rp
    );
    vector_wrap_type in( *ops ), u( *ops ), v( *ops ), sample( *ops ), other( *ops );
    for ( auto *x : { &in, &u, &v, &sample, &other } )
        x->start_use();
    const double       start = .37, duration = model.lambda > 1 ? .02 : .2;
    const double       end   = start + ( backward ? -duration : duration );
    const unsigned int count = std::lround( duration / h );
    operations_type::for_each_type{}( initial_time_state{ *u, start }, 2 );
    ops->assign( *u, *v );
    double dense_error = 0;
    for ( unsigned int n = 0; n < count; ++n )
    {
        const double t = start + ( backward ? -1. : 1. ) * n * h;
        step.set_time( t );
        reference.set_time( t );
        step.set_target_time( end );
        reference.set_target_time( end );
        ops->assign( *u, *in );
        step.apply( *in, *u );
        ops->assign( *v, *in );
        reference.apply( *in, *v );
        if ( step.get_status() != single_step_status::converged ||
             reference.get_status() != single_step_status::converged )
            std::cerr << method << " lambda=" << model.lambda << " h=" << h << " t=" << t
                      << " mass status=" << int( step.get_status() ) << " rate status=" << int( reference.get_status() )
                      << '\n';
        check(
            step.get_status() == single_step_status::converged &&
                reference.get_status() == single_step_status::converged,
            "Non-autonomous mass-form fixed step failed"
        );
        ops->assign_lin_comb( 1., *u, -1., *v, *other );
        check( ops->norm( *other ) < 3e-11, "Non-autonomous mass/rate stage disagreement" );
        check(
            p->time == t && p->prepared_time == t && p->derivative_time == t && p->jacobi->model.time == t &&
                p->derivatives == n + 1 && solver->bindings == n + 1,
            "Mass-form stage changed the base time, derivative or frozen Jacobian"
        );
        for ( std::size_t i = 0; i < step.table().stage_count(); ++i )
        {
            const double c        = step.table().stage_time( i );
            const double expected = c == 1 && step.get_dt() == end - t ? end : t + c * step.get_dt();
            check( p->residual_times[i] == expected, "Mass-form residual used the wrong stage time" );
        }
        if constexpr ( Dense )
        {
            auto view = step.get_continuous_integration();
            auto ref  = reference.get_continuous_integration();
            for ( double theta : { .25, .5, .75 } )
            {
                if ( theta != .5 && n != 0 && n + 1 != count )
                    continue;
                const double time = view.evaluate( theta, *sample );
                ref.evaluate( theta, *other );
                ops->add_lin_comb( -1., *sample, *other );
                check( ops->norm( *other ) < 5e-11, "Non-autonomous mass/rate dense disagreement" );
                for ( unsigned int i = 0; i < 2; ++i )
                    dense_error = std::max(
                        dense_error, std::abs( ops->get_value_at_point( i, *sample ) - std::exp( ( i + 1 ) * time ) )
                    );
            }
            const auto   stages   = step.table().stage_count();
            const double endpoint = step.get_dt() == end - t ? end : t + step.get_dt();
            if ( stages == 3 )
                check( p->residual_times[3] == endpoint, "ROS3 mass dense correction missed endpoint time" );
            if ( stages == 4 )
                check(
                    p->residual_times[4] == t && p->residual_times[5] == endpoint,
                    "W mass dense slopes missed base/endpoint times"
                );
        }
        const double endpoint = step.get_dt() == end - t ? end : t + step.get_dt();
        step.finalize( adaptation_status::accepted, endpoint, *u );
        reference.finalize( adaptation_status::accepted, endpoint, *v );
    }
    double error = 0;
    for ( unsigned int i = 0; i < 2; ++i )
        error = std::max( error, std::abs( ops->get_value_at_point( i, *u ) - std::exp( ( i + 1 ) * end ) ) );
    const auto stages = step.table().stage_count();
    const auto extra  = Dense ? ( stages == 4 ? 2 : stages == 3 ? 1 : 0 ) : 0;
    check( p->residuals == ( stages + extra ) * count, "Mass time path added unexpected residual evaluations" );
    return { error, dense_error };
}

void orders( const char *method, manufactured_time_model model, bool backward )
{
    // The stiff eigenvalues reach about -150; use lambda*h small enough for
    // classical order, independently of stiff-limit stability/accuracy tests.
    const double h = model.lambda > 1 ? .0005 : .04;
    const auto   a = fixed<true>( method, model, h, backward ), b = fixed<true>( method, model, h / 2, backward );
    const auto   c       = fixed<true>( method, model, h / 4, backward );
    const double minimum = std::pow( 2., rosenbrock::method_table( method ).order() - .45 );
    if ( a.first / b.first <= minimum || b.first / c.first <= minimum || a.second / b.second <= minimum ||
         b.second / c.second <= minimum )
        std::cerr << method << " lambda=" << model.lambda << " varying=" << model.varying << " backward=" << backward
                  << " factor=" << model.factor << " errors=" << a.first << ',' << b.first << ',' << c.first
                  << " dense=" << a.second << ',' << b.second << ',' << c.second << '\n';
    check( a.first / b.first > minimum && b.first / c.first > minimum, "Non-autonomous mass primary order lost" );
    check(
        a.second / b.second > minimum && b.second / c.second > minimum, "Non-autonomous mass continuous order lost"
    );
    const auto plain = fixed<false>( method, model, h, backward );
    check( std::abs( plain.first - a.first ) < 1e-13, "Mass time dense preparation changed the primary solution" );
}

// Analytic polynomial forcing checks h^2 underflow/overflow without overflow
// in the true solution. No evaluate_rate callback is supplied by this problem.
void extreme_steps( const char *method )
{
    using P              = mass_time_problem<operations_type, polynomial_time_model>;
    using S              = rosenbrock::time_step<operations_type, P, adaptation_type, direct_solver<P>, true>;
    auto             ops = std::make_shared<operations_type>( 2 );
    vector_wrap_type in( *ops ), out( *ops ), sample( *ops );
    in.start_use();
    out.start_use();
    sample.start_use();
    ops->assign_scalar( 0., *in );
    for ( double h : { 1e-200, 1e200 } )
    {
        polynomial_time_model model;
        model.slope        = h < 1 ? 1e300 : 1e-300;
        const double exact = ( model.slope * h ) * h / 2;
        auto         p     = std::make_shared<P>( ops, model );
        S::params    sp;
        sp.method     = method;
        sp.mass_form  = true;
        sp.autonomous = false;
        S step(
            ops, p, std::make_shared<adaptation_type>( adaptation_type::params{ h } ),
            std::make_shared<direct_solver<P>>( ops ), sp
        );
        step.set_target_time( h );
        step.apply( *in, *out );
        check(
            step.get_status() == single_step_status::converged, "Mass time defect overflowed or underflowed prematurely"
        );
        check(
            std::abs( ops->get_value_at_point( 0, *out ) / exact - 1 ) < 2e-12,
            "Mass time defect lost extreme-step accuracy"
        );
        auto view = step.get_continuous_integration();
        view.evaluate( .5, *sample );
        check(
            std::abs( ops->get_value_at_point( 0, *sample ) / ( exact * .25 ) - 1 ) < 3e-12,
            "Mass time dense output lost extreme-step accuracy"
        );
    }
}

template <class C>
void adaptive( const char *method, const char *filter = nullptr )
{
    using P              = mass_time_problem<>;
    using S              = rosenbrock::time_step<operations_type, P, C, direct_solver<P>, true>;
    auto             ops = std::make_shared<operations_type>( 2 );
    auto             p   = std::make_shared<P>( ops );
    vector_wrap_type in( *ops ), out( *ops );
    in.start_use();
    out.start_use();
    operations_type::for_each_type{}( initial_time_state{ *in, .37 }, 2 );
    double      errors[2]{};
    std::size_t steps[2]{};
    for ( unsigned int level = 0; level < 2; ++level )
    {
        typename C::params cp;
        cp.initial_step = cp.maximum_step = .2;
        cp.relative_tolerance             = level ? 1e-6 : 1e-3;
        cp.absolute_tolerance             = cp.relative_tolerance * .01;
        if constexpr ( std::is_same_v<C, integration::time_step_adaptation_filter<operations_type>> )
            cp.controller = filter;
        typename S::params sp;
        sp.method     = method;
        sp.mass_form  = true;
        sp.autonomous = false;
        auto step     = std::make_shared<S>(
            ops, p, std::make_shared<C>( ops, cp ), std::make_shared<direct_solver<P>>( ops ), sp
        );
        integration::time_integrator<operations_type, S> run( ops, step );
        run.set_time_interval( .37, .77 );
        double first = 0;
        for ( unsigned int restart = 0; restart < 2; ++restart )
        {
            run.apply( *in, *out );
            check(
                run.get_status() == integration_status::completed && run.get_final_time() == .77,
                "Adaptive non-autonomous mass integration failed"
            );
            const double result = ops->get_value_at_point( 1, *out );
            if ( !restart )
            {
                first        = result;
                steps[level] = run.get_steps();
            }
            else
                check( result == first && steps[level] == run.get_steps(), "Mass time restart retained state/history" );
            errors[level] = std::abs( result - std::exp( 1.54 ) );
        }
    }
    check(
        errors[1] < errors[0] * .1 && errors[1] < 5e-5 && steps[1] > steps[0], "Adaptive mass time refinement failed"
    );
}

int main()
try
{
    backend_type::init_device();
    static_assert( !detail::has_rate_evaluation<mass_time_problem<>, vector_type>::value );
    for ( const char *method : { "ROS2", "ROS3", "ROS34PW2" } )
    {
        for ( bool varying : { false, true } )
        {
            manufactured_time_model model;
            model.varying = varying;
            orders( method, model, false );
            orders( method, model, true );
        }
        manufactured_time_model stiff;
        stiff.lambda = 50;
        orders( method, stiff, false );
        // Unresolved stiff decay is a bounded tracking/refinement check, not
        // an assertion of stiffness-uniform primary or interpolation order.
        stiff.lambda      = 1e4;
        const auto coarse = fixed<true>( method, stiff, .01 ), fine = fixed<true>( method, stiff, .005 );
        check(
            fine.first < coarse.first && fine.first < 1e-3 && fine.second < coarse.second && fine.second < 3e-3,
            "Mass time stiff tracking/refinement failed"
        );
        extreme_steps( method );
        adaptive<integration::time_step_adaptation_matlab<operations_type>>( method );
        adaptive<integration::time_step_adaptation_scipy<operations_type>>( method );
        for ( const char *filter : { "I", "H211", "PC", "PID", "H312", "PPID", "H321" } )
            adaptive<integration::time_step_adaptation_filter<operations_type>>( method, filter );
    }
    manufactured_time_model approximate;
    approximate.factor = .4;
    approximate.offset = .2;
    orders( "ROS34PW2", approximate, false );
    orders( "ROS34PW2", approximate, true );
    std::cout << "PASS: non-autonomous Rosenbrock constant mass (" << checks << " checks)\n";
}
catch ( const std::exception &e )
{
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
