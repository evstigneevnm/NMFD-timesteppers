#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#ifdef TEST_VECTOR_BACKEND_CUDA
#    include <scfd/backend/cuda.h>
using backend_type = scfd::backend::cuda<std::ptrdiff_t>;
#elif defined( TEST_VECTOR_BACKEND_OMP )
#    include <scfd/backend/omp.h>
using backend_type = scfd::backend::omp<std::ptrdiff_t>;
#else
#    include <scfd/backend/serial_cpu.h>
using backend_type = scfd::backend::serial_cpu<std::ptrdiff_t>;
#endif
#include <nmfd/operations/scfd_vector_space.h>
#include <nmfd/time_steppers/integration/time_integrator.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_filter.h>
#include <nmfd/time_steppers/runge_kutta/explicit_time_step.h>

using namespace nmfd::time_steppers;
constexpr const char *presets[] = { "I", "H211", "PC", "PID", "H312", "PPID", "H321" };

void require( bool condition, const char *message )
{
    if ( !condition )
        throw std::runtime_error( message );
}

template <class T>
struct fixture
{
    using operations_type                                 = nmfd::operations::scfd_vector_space<backend_type, T>;
    using controller_type                                 = integration::time_step_adaptation_filter<operations_type>;
    std::shared_ptr<operations_type>           operations = std::make_shared<operations_type>( 3 );
    nmfd::detail::vector_wrap<operations_type> state{ *operations }, error{ *operations };
    fixture()
    {
        state.start_use();
        error.start_use();
        operations->assign_scalar( T( 0 ), *state );
    }
    typename controller_type::params parameters( const char *preset = "PID" ) const
    {
        typename controller_type::params p;
        p.absolute_tolerance = 1;
        p.minimum_step       = T( 1e-6 );
        p.maximum_step       = 10;
        p.controller         = preset;
        return p;
    }
    adaptation_status assess( controller_type &control, T time, T h, T ratio, T &next, unsigned int order = 2 )
    {
        operations->assign_scalar( ratio, *error );
        return control.assess( time, h, *state, *state, next, order, &*error );
    }
    void close( T actual, T expected ) const
    {
        require(
            std::abs( actual - expected ) <
                T( 128 ) * std::numeric_limits<T>::epsilon() * std::max( T( 1 ), std::abs( expected ) ),
            "Filter proposal differs from independent recurrence"
        );
    }
};

// Direct powers provide an independent oracle for the log-domain implementation.
template <class T>
void test_presets()
{
    fixture<T>        f;
    const long double coefficients[][5] = {
        { 1, 0, 0, 0, 0 },
        { .25L, -.25L, 0, -.25L, 0 },
        { 2, 1, 0, 1, 0 },
        { 1.L / 18, -1.L / 9, 1.L / 18, 0, 0 },
        { .125L, -.25L, .125L, -.375L, -.125L },
        { .3L, -.05L, -.25L, 1, 0 },
        { 1.L / 3, -1.L / 18, -5.L / 18, 5.L / 6, 1.L / 6 }
    };
    const unsigned int required[] = { 0, 1, 1, 2, 2, 2, 2 };
    const T            errors[]   = { T( .6 ), T( .3 ), T( .8 ), T( .4 ), T( .7 ) };
    const T            steps[]    = { T( .07 ), T( .05 ), T( .09 ), T( .03 ), T( .08 ) };
    for ( unsigned int preset = 0; preset < 7; ++preset )
        for ( unsigned int order : { 2u, 3u, 5u } )
        {
            typename fixture<T>::controller_type control( f.operations, f.parameters( presets[preset] ) );
            T                                    time = 0, next = 0;
            for ( unsigned int n = 0; n < 5; ++n )
            {
                require(
                    f.assess( control, time, steps[n], errors[n], next, order ) == adaptation_status::accepted,
                    "Filter rejected a sub-tolerance trial"
                );
                const auto *c      = coefficients[n < required[preset] ? 0 : preset];
                auto        factor = .9L * std::pow( static_cast<long double>( errors[n] ), -c[0] / order );
                if ( n >= required[preset] && required[preset] > 0 )
                    factor *= std::pow( static_cast<long double>( errors[n - 1] ), c[1] / order ) *
                              std::pow( static_cast<long double>( steps[n] ) / steps[n - 1], c[3] );
                if ( n >= required[preset] && required[preset] > 1 )
                    factor *= std::pow( static_cast<long double>( errors[n - 2] ), -c[2] / order ) *
                              std::pow( static_cast<long double>( steps[n - 1] ) / steps[n - 2], c[4] );
                f.close( next, T( steps[n] * std::clamp( factor, .2L, 10.L ) ) );
                time += steps[n];
                control.update( adaptation_status::accepted, time, steps[n], *f.state );
            }
        }
}

template <class T>
void test_history_and_failures()
{
    fixture<T> f;
    using controller_type = typename fixture<T>::controller_type;
    const auto p          = f.parameters();
    for ( const char *name : presets )
    {
        controller_type control( f.operations, f.parameters( name ) ), reference( f.operations, f.parameters( name ) );
        T               time = 0, next = 0, expected = 0;
        for ( T error : { T( .3 ), T( .8 ) } )
        {
            f.assess( control, time, T( .1 ), error, next );
            f.assess( reference, time, T( .1 ), error, expected );
            time += T( .1 );
            control.update( adaptation_status::accepted, time, T( .1 ), *f.state );
            reference.update( adaptation_status::accepted, time, T( .1 ), *f.state );
        }
        // Abandoned assessments, failed solves, and error rejections must not enter history.
        f.assess( control, time, T( .08 ), T( .2 ), next );
        control.update( adaptation_status::failed, time, T( .08 ), *f.state );
        f.assess( control, time, T( .04 ), T( .5 ), next );
        f.assess( reference, time, T( .04 ), T( .5 ), expected );
        f.close( next, expected );
        require(
            f.assess( control, time, T( .1 ), T( 4 ), next ) == adaptation_status::rejected,
            "Filter failed to reject a large error"
        );
        control.update( adaptation_status::rejected, time, T( .1 ), *f.state );
        reference.reject_step( T( .1 ) );
        f.assess( control, time, T( .02 ), T( .5 ), next );
        f.assess( reference, time, T( .02 ), T( .5 ), expected );
        f.close( next, expected );
        require( next <= T( .02 ), "Filter grew immediately after rejection" );
        control.update( adaptation_status::accepted_modified, time + T( .01 ), T( .01 ), *f.state );
        require( control.get_dt() <= T( .01 ), "Modified endpoint did not bound the next step" );
        f.assess( control, time + T( .01 ), T( .01 ), T( .5 ), next );
        f.close( next, T( .01 ) * T( .9 ) * std::sqrt( T( 2 ) ) );
        control.reset();
        f.close( control.get_dt(), p.initial_step );
    }

    controller_type control( f.operations, f.parameters( "H211" ) );
    T               next   = 0;
    auto            warmup = [&]() {
        control.reset();
        f.assess( control, T( 0 ), T( .1 ), T( .4 ), next );
        control.update( adaptation_status::accepted, T( .1 ), T( .1 ), *f.state );
    };
    for ( int change = 0; change < 3; ++change )
    {
        warmup();
        const T            time = change == 0 ? T( .5 ) : T( .1 ), h = change == 1 ? -T( .01 ) : T( .01 );
        const unsigned int order = change == 2 ? 5 : 2;
        f.assess( control, time, h, T( .5 ), next, order );
        f.close( next, T( .01 ) * T( .9 ) * std::pow( T( .5 ), -T( 1 ) / order ) );
    }
    warmup();
    control.initialize( T( 3 ), *f.state, T( .2 ) );
    f.close( control.get_dt(), T( .2 ) );
    for ( T ratio :
          { T( 0 ), std::numeric_limits<T>::min(), T( 1 ), std::numeric_limits<T>::infinity(),
            std::numeric_limits<T>::quiet_NaN() } )
    {
        control.reset();
        const auto result = f.assess( control, T( 0 ), T( .1 ), ratio, next );
        require( std::isfinite( next ) && next > 0, "Degenerate errors produced an invalid proposal" );
        require(
            result == ( ratio < 1 ? adaptation_status::accepted : adaptation_status::rejected ),
            "Incorrect filter error boundary"
        );
    }
    for ( int field = 0; field < 3; ++field )
    {
        f.operations->assign_scalar( T( 0 ), *f.state );
        f.operations->assign_scalar( T( .1 ), *f.error );
        f.operations->assign_scalar( std::numeric_limits<T>::infinity(), field == 2 ? *f.error : *f.state );
        control.reset();
        require(
            control.assess( 0, T( .1 ), *f.state, *f.state, next, 2, &*f.error ) == adaptation_status::rejected,
            "Nonfinite filter input was accepted"
        );
    }
    f.operations->assign_scalar( T( 0 ), *f.state );
    require(
        control.assess( 0, T( .1 ), *f.state, *f.state, next ) == adaptation_status::failed,
        "Missing estimate was accepted"
    );
    require( control.reject_step( p.minimum_step ) == adaptation_status::failed, "Filter retried below minimum step" );
    require( control.reject_step( T( 0 ) ) == adaptation_status::failed, "Zero failed step was retried" );
    control.reset();
    require(
        f.assess( control, T( 0 ), p.minimum_step / 2, T( .5 ), next ) == adaptation_status::accepted,
        "Accurate clipped final step below the minimum was rejected"
    );
    for ( int field = 0; field < 7; ++field )
    {
        auto bad = p;
        if ( field == 0 )
            bad.relative_tolerance = -1;
        if ( field == 1 )
            bad.absolute_tolerance = std::numeric_limits<T>::infinity();
        if ( field == 2 )
            bad.initial_step = std::numeric_limits<T>::quiet_NaN();
        if ( field == 3 )
            bad.minimum_step = 0;
        if ( field == 4 )
            bad.maximum_step = -1;
        if ( field == 5 )
            bad.minimum_step = bad.maximum_step * 2;
        if ( field == 6 )
            bad.controller = "unknown";
        bool threw = false;
        try
        {
            controller_type invalid( f.operations, bad );
        }
        catch ( const std::invalid_argument & )
        {
            threw = true;
        }
        require( threw, "Invalid filter parameters were accepted" );
    }
}

using operations_type = nmfd::operations::scfd_vector_space<backend_type, double>;

template <class T>
void test_scaling_and_metadata()
{
    fixture<T> f;
    using controller_type = typename fixture<T>::controller_type;
    nmfd::detail::vector_wrap<typename fixture<T>::operations_type> candidate( *f.operations );
    candidate.start_use();
    auto p               = f.parameters( "I" );
    p.absolute_tolerance = T( .02 );
    p.relative_tolerance = T( .01 );
    controller_type control( f.operations, p );
    const T         x[] = { 0, -4, T( .25 ) }, y[] = { 0, 1, -8 }, ratios[] = { T( -.25 ), T( .5 ), T( -.75 ) };
    T               e[3], sum = 0, next = 0;
    for ( unsigned int i = 0; i < 3; ++i )
    {
        e[i] = ratios[i] *
               ( p.absolute_tolerance + p.relative_tolerance * std::max( std::abs( x[i] ), std::abs( y[i] ) ) );
        sum += ratios[i] * ratios[i];
    }
    f.operations->set( x, *f.state );
    f.operations->set( y, *candidate );
    f.operations->set( e, *f.error );
    require(
        control.assess( 0, T( .1 ), *f.state, *candidate, next, 2, &*f.error ) == adaptation_status::accepted,
        "Additive RMS filter rejected a small state error"
    );
    f.close( next, T( .1 ) * T( .9 ) * std::pow( std::sqrt( sum / 3 ), T( -.5 ) ) );
    for ( unsigned int i = 0; i < 3; ++i )
    {
        require(
            f.operations->get_value_at_point( i, *f.state ) == x[i] &&
                f.operations->get_value_at_point( i, *candidate ) == y[i] &&
                f.operations->get_value_at_point( i, *f.error ) == e[i],
            "Filter changed supplied vectors"
        );
    }
    f.operations->assign_scalar( std::numeric_limits<T>::quiet_NaN(), *candidate );
    control.reset();
    require(
        control.assess( 0, T( .1 ), *f.state, *candidate, next, 2, &*f.error ) == adaptation_status::rejected,
        "Filter accepted a nonfinite candidate"
    );
    for ( unsigned int field = 0; field < 4; ++field )
    {
        const T time = field == 0 ? std::numeric_limits<T>::infinity() : 0;
        const T h    = field == 1 ? T( 0 ) : field == 2 ? std::numeric_limits<T>::quiet_NaN() : T( .1 );
        require(
            control.assess( time, h, *f.state, *f.state, next, field == 3 ? 0 : 2, &*f.error ) ==
                adaptation_status::failed,
            "Filter accepted invalid assessment metadata"
        );
    }
    for ( int field = 0; field < 3; ++field )
    {
        bool threw = false;
        try
        {
            control.initialize(
                field == 0 ? std::numeric_limits<T>::infinity() : 0, *f.state,
                field == 1   ? T( 0 )
                : field == 2 ? std::numeric_limits<T>::quiet_NaN()
                             : T( -1 )
            );
        }
        catch ( const std::invalid_argument & )
        {
            threw = true;
        }
        require( threw, "Filter accepted invalid initialization" );
    }
    bool threw = false;
    try
    {
        controller_type invalid( nullptr );
    }
    catch ( const std::invalid_argument & )
    {
        threw = true;
    }
    require( threw, "Filter accepted missing operations" );
}

// u'=u+t, u(0)=1, exact 2*exp(t)-t-1: stage-time dependent, nonstiff growth.
struct forced_growth
{
    std::shared_ptr<operations_type> operations;
    double                           time = 0;
    void                             set_time( double t )
    {
        time = t;
    }
    void apply( const operations_type::vector_type &in, operations_type::vector_type &out )
    {
        operations->assign( in, out );
        operations->add_mul_scalar( time, 1., out );
    }
};

void test_integration()
{
    using controller_type = integration::time_step_adaptation_filter<operations_type>;
    using step_type       = runge_kutta::explicit_time_step<operations_type, forced_growth, controller_type>;
    auto                                       ops     = std::make_shared<operations_type>( 1 );
    auto                                       problem = std::make_shared<forced_growth>( forced_growth{ ops } );
    nmfd::detail::vector_wrap<operations_type> initial( *ops ), result( *ops );
    initial.start_use();
    result.start_use();
    for ( const char *name : presets )
        for ( const char *method : { "RK23", "RK45" } )
        {
            controller_type::params p;
            p.controller         = name;
            p.relative_tolerance = 1e-9;
            p.absolute_tolerance = 1e-11;
            p.initial_step       = 1;
            auto control         = std::make_shared<controller_type>( ops, p );
            auto step            = std::make_shared<step_type>( ops, problem, control, step_type::params{ method } );
            integration::time_integrator<operations_type, step_type> integrator( ops, step );
            double                                                   first       = 0;
            std::size_t                                              first_steps = 0;
            ops->assign_scalar( 1., *initial );
            integrator.set_time_interval( 0, .5 );
            for ( int run = 0; run < 2; ++run )
            {
                integrator.apply( *initial, *result );
                const auto value = ops->get_value_at_point( 0, *result );
                require(
                    integrator.get_status() == integration_status::completed && integrator.get_final_time() == .5 &&
                        std::abs( value - ( 2 * std::exp( .5 ) - 1.5 ) ) < 2e-8,
                    "Filter-controlled RK analytical mismatch"
                );
                if ( run )
                    require(
                        value == first && integrator.get_steps() == first_steps,
                        "Filter history leaked across trajectories"
                    );
                first       = value;
                first_steps = integrator.get_steps();
            }
            ops->assign_scalar( 2 * std::exp( .5 ) - 1.5, *initial );
            integrator.set_time_interval( .5, 0 );
            integrator.apply( *initial, *result );
            require(
                integrator.get_status() == integration_status::completed &&
                    std::abs( ops->get_value_at_point( 0, *result ) - 1 ) < 2e-8,
                "Filter-controlled RK backward mismatch"
            );
        }
    // DOP853 requires its dual estimator; never silently interpret E5 as its full error.
    step_type unsupported( ops, problem, std::make_shared<controller_type>( ops ), step_type::params{ "DOP853" } );
    ops->assign_scalar( 42., *result );
    unsupported.apply( *initial, *result );
    require(
        unsupported.get_status() == single_step_status::error_estimate_unavailable &&
            ops->get_value_at_point( 0, *result ) == 42.,
        "Filter silently accepted the unsupported DOP853 estimator"
    );
}

int main()
try
{
    backend_type::init_device();
    test_presets<float>();
    test_presets<double>();
    test_history_and_failures<float>();
    test_history_and_failures<double>();
    test_scaling_and_metadata<float>();
    test_scaling_and_metadata<double>();
    test_integration();
    std::cout << "Seven filter recurrences, committed history, bounds, recovery, restart and explicit RK: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
