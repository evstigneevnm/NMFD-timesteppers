#include "implicit_rosenbrock_time_problem.h"
#include <nmfd/time_steppers/integration/time_step_adaptation_filter.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_matlab.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_scipy.h>

using namespace rosenbrock_tests;
using P = time_problem<>;
struct L : direct_solver<P>
{
    using direct_solver<P>::direct_solver;
    unsigned int throw_on = 0;
    bool         solve( const vector_type &in, vector_type &out )
    {
        if ( calls + 1 == throw_on )
            throw std::runtime_error( "Injected dense-only solve exception" );
        return direct_solver<P>::solve( in, out );
    }
};
using S = rosenbrock::time_step<operations_type, P, adaptation_type, L, true>;

template <class Action>
void throws( Action action, const char *message )
{
    bool failed = false;
    try
    {
        action();
    }
    catch ( const std::exception & )
    {
        failed = true;
    }
    check( failed, message );
}

// The manufactured exponential has a known value at every shortened endpoint.
// It tests borrowed-view lifetime and rollback independently of orbit equations.
void test_views()
{
    for ( const char *method : { "ROS2", "ROS3" } )
    {
        auto      ops    = std::make_shared<operations_type>( 1 );
        auto      p      = std::make_shared<P>( ops );
        auto      linear = std::make_shared<L>( ops );
        S::params settings;
        settings.method     = method;
        settings.autonomous = false;
        S step( ops, p, std::make_shared<adaptation_type>( adaptation_type::params{ .01 } ), linear, settings );
        vector_wrap_type state( *ops ), sample( *ops );
        state.start_use();
        sample.start_use();
        ops->assign_scalar( std::exp( .74 ), *state );
        step.set_time( .37 );
        step.set_target_time( .375 );
        throws( [&] { step.get_continuous_integration(); }, "Idle step exposed a dense view" );
        step.apply( *state, *state );
        check( step.get_status() == single_step_status::converged, "In-place dense step failed" );
        auto view = step.get_continuous_integration();
        check(
            view.evaluate( 0., *sample ) == .37 && ops->get_value_at_point( 0, *sample ) == std::exp( .74 ),
            "In-place step lost its dense input snapshot"
        );
        check(
            view.evaluate( 1., *sample ) == .375 &&
                ops->get_value_at_point( 0, *sample ) == ops->get_value_at_point( 0, *state ),
            "Clipped dense endpoint is inconsistent"
        );
        for ( double theta :
              { -.1, 1.1, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() } )
            throws( [&] { view.evaluate( theta, *sample ); }, "Dense view accepted an invalid coordinate" );
        throws( [&] { step.set_time( 1 ); }, "Pending dense step allowed time mutation" );
        const double shortened = view.evaluate( .5, *state );
        step.finalize( adaptation_status::accepted_modified, shortened, *state );
        throws( [&] { view.evaluate( .5, *sample ); }, "Modified commitment left its old dense view valid" );
        ops->assign( *state, *sample );
        step.set_time( shortened );
        step.apply( *state, *state );
        check(
            step.get_status() == single_step_status::converged && p->prepared_time == shortened,
            "Shortened endpoint did not reprepare the base state"
        );
        auto second = step.get_continuous_integration();
        throws( [&] { view.evaluate( .5, *sample ); }, "A new candidate revived an expired dense view" );
        step.finalize( adaptation_status::rejected, shortened, *sample );
        ops->assign( *sample, *state );
        throws( [&] { second.evaluate( .5, *sample ); }, "Rejected candidate retained dense output" );
        step.apply( *state, *state );
        auto third = step.get_continuous_integration();
        step.reset();
        throws( [&] { third.evaluate( .5, *sample ); }, "Reset retained dense output" );
    }
}

struct retry_controller : adaptation_type
{
    double       next     = .01;
    unsigned int failures = 0, commits = 0;
    double       get_dt() const
    {
        return next;
    }
    adaptation_status reject_step( double h )
    {
        ++failures;
        next = std::abs( h ) / 2;
        return adaptation_status::rejected;
    }
    void update( adaptation_status status, double, double, const vector_type & )
    {
        commits += status != adaptation_status::rejected;
    }
};

void test_preparation_failures()
{
    for ( bool solve : { false, true } )
        for ( unsigned int failure = 0; failure < 3; ++failure )
        {
            auto ops    = std::make_shared<operations_type>( 1 );
            auto p      = std::make_shared<P>( ops );
            auto linear = std::make_shared<L>( ops );
            if ( solve )
            {
                linear->fail_on    = failure == 0 ? 4 : 0;
                linear->invalid_on = failure == 1 ? 4 : 0;
                linear->throw_on   = failure == 2 ? 4 : 0;
            }
            else
            {
                p->fail_rate_on    = failure == 0 ? 3 : 0;
                p->invalid_rate_on = failure == 1 ? 3 : 0;
                p->throw_rate_on   = failure == 2 ? 3 : 0;
            }
            S::params settings;
            settings.method     = "ROS3";
            settings.autonomous = false;
            S step( ops, p, std::make_shared<adaptation_type>( adaptation_type::params{ .01 } ), linear, settings );
            vector_wrap_type state( *ops ), out( *ops );
            state.start_use();
            out.start_use();
            ops->assign_scalar( std::exp( .74 ), *state );
            ops->assign_scalar( 42., *out );
            step.set_time( .37 );
            bool threw = false;
            try
            {
                step.apply( *state, *out );
            }
            catch ( const std::runtime_error & )
            {
                threw = true;
            }
            check(
                threw == ( failure == 2 ) && ops->get_value_at_point( 0, *out ) == 42 && p->time == .37,
                "Dense preparation failure changed output or failed to restore time"
            );
            if ( !threw )
                check(
                    step.get_status() ==
                        ( failure == 1 ? single_step_status::failed_nonfinite : single_step_status::failed_solver ),
                    "Dense preparation lost its failure status"
                );
            throws( [&] { step.get_continuous_integration(); }, "Failed dense preparation exposed a view" );
        }
    for ( bool solve : { false, true } )
    {
        auto ops        = std::make_shared<operations_type>( 1 );
        auto p          = std::make_shared<P>( ops );
        auto linear     = std::make_shared<L>( ops );
        linear->fail_on = solve ? 4 : 0;
        p->fail_rate_on = solve ? 0 : 3;
        using Step      = rosenbrock::time_step<operations_type, P, retry_controller, L, true>;
        Step::params settings;
        settings.method          = "ROS3";
        settings.autonomous      = false;
        auto             control = std::make_shared<retry_controller>();
        Step             step( ops, p, control, linear, settings );
        vector_wrap_type state( *ops ), out( *ops );
        state.start_use();
        out.start_use();
        ops->assign_scalar( 1., *state );
        step.apply( *state, *out );
        check(
            step.get_status() == single_step_status::converged && step.get_attempts() == 2 && control->failures == 1 &&
                control->commits == 0 && linear->bindings == 2,
            "Dense preparation retry did not rebuild from unchanged input"
        );
        step.finalize( adaptation_status::accepted, step.get_dt(), *out );
        check( control->commits == 1, "Dense preparation committed a failed attempt" );
        settings.maximum_attempts = 1;
        linear->fail_on           = linear->calls + 4;
        Step limited( ops, p, control, linear, settings );
        ops->assign_scalar( 42., *out );
        limited.apply( *state, *out );
        check(
            limited.get_status() == single_step_status::attempt_limit_reached &&
                ops->get_value_at_point( 0, *out ) == 42,
            "Exhausted dense preparation retry published a candidate"
        );
    }
}

// Error-rejected candidates must not pay for dense-only rate/solve work or
// publish a view. Every existing accuracy controller follows the same contract.
template <class Controller>
void test_error_acceptance( const char *filter = nullptr )
{
    for ( const char *method : { "ROS2", "ROS3" } )
    {
        auto                        ops    = std::make_shared<operations_type>( 1 );
        auto                        p      = std::make_shared<P>( ops );
        auto                        linear = std::make_shared<L>( ops );
        typename Controller::params cp;
        cp.relative_tolerance = 1e-6;
        cp.absolute_tolerance = 1e-8;
        cp.initial_step = cp.maximum_step = .2;
        if constexpr ( std::is_same_v<Controller, integration::time_step_adaptation_filter<operations_type>> )
            cp.controller = filter;
        using Step = rosenbrock::time_step<operations_type, P, Controller, L, true>;
        typename Step::params sp;
        sp.method     = method;
        sp.autonomous = false;
        Step             step( ops, p, std::make_shared<Controller>( ops, cp ), linear, sp );
        vector_wrap_type state( *ops ), out( *ops );
        state.start_use();
        out.start_use();
        ops->assign_scalar( std::exp( .74 ), *state );
        step.set_time( .37 );
        step.set_target_time( .57 );
        step.apply( *state, *out );
        const unsigned int attempts = step.get_attempts(), stages = step.table().stage_count(), extra = stages == 3;
        check(
            step.get_status() == single_step_status::converged && attempts > 1 && p->rates == 2 * attempts + extra &&
                linear->calls == stages * attempts + extra && linear->bindings == attempts &&
                p->derivatives == attempts,
            "Error-rejected candidates performed dense-only work or changed controller behavior"
        );
        const auto view = step.get_continuous_integration();
        view.evaluate( .5, *out );
        check(
            std::abs( ops->get_value_at_point( 0, *out ) - std::exp( 2 * ( .37 + .5 * step.get_dt() ) ) ) < 2e-5,
            "Adaptive dense midpoint is inaccurate"
        );
        view.evaluate( 1., *out );
        step.finalize( adaptation_status::accepted, .37 + step.get_dt(), *out );
    }
}

int main()
try
{
    backend_type::init_device();
    test_views();
    test_preparation_failures();
    test_error_acceptance<integration::time_step_adaptation_matlab<operations_type>>();
    test_error_acceptance<integration::time_step_adaptation_scipy<operations_type>>();
    for ( const char *filter : { "I", "H211", "PC", "PID", "H312", "PPID", "H321" } )
        test_error_acceptance<integration::time_step_adaptation_filter<operations_type>>( filter );
    std::cout << "Rosenbrock dense views, in-place steps, external endpoints and preparation rollback: " << checks
              << " checks PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
