#include "implicit_rosenbrock_w_problem.h"
#include <nmfd/time_steppers/integration/time_step_adaptation_filter.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_matlab.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_scipy.h>

using namespace rosenbrock_tests;
using P = w_time_problem<>;
struct L : direct_solver<P>
{
    using direct_solver<P>::direct_solver;
    unsigned int throw_on = 0, overflow_on = 0;
    bool         solve( const vector_type &in, vector_type &out )
    {
        if ( calls + 1 == throw_on )
            throw std::runtime_error( "Injected W dense solve exception" );
        if ( calls + 1 == overflow_on )
        {
            ++calls;
            ops->assign_scalar( std::numeric_limits<double>::max(), out );
            return true;
        }
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
struct fixture
{
    std::shared_ptr<operations_type> ops    = std::make_shared<operations_type>( 1 );
    std::shared_ptr<P>               p      = std::make_shared<P>( ops );
    std::shared_ptr<L>               linear = std::make_shared<L>( ops );
    vector_wrap_type                 state{ *ops }, out{ *ops }, sample{ *ops };
    S::params                        settings;
    fixture()
    {
        settings.method     = "ROS34PW2";
        settings.autonomous = false;
        for ( auto *v : { &state, &out, &sample } )
            v->start_use();
        ops->assign_scalar( std::exp( .74 ), *state );
        ops->assign_scalar( 42., *out );
    }
    double value( const vector_type &x )
    {
        return ops->get_value_at_point( 0, x );
    }
};

// The manufactured exp(2t) trajectory checks shortened commits and view
// lifetime without relying on any periodic-orbit or event-location machinery.
void test_views()
{
    fixture f;
    S step( f.ops, f.p, std::make_shared<adaptation_type>( adaptation_type::params{ .01 } ), f.linear, f.settings );
    step.set_time( .37 );
    step.set_target_time( .375 );
    throws( [&] { step.get_continuous_integration(); }, "Idle W step exposed a dense view" );
    step.apply( *f.state, *f.state );
    check( step.get_status() == single_step_status::converged, "In-place W dense step failed" );
    auto view = step.get_continuous_integration();
    check(
        view.evaluate( 0., *f.sample ) == .37 && f.value( *f.sample ) == std::exp( .74 ),
        "W dense input snapshot was overwritten"
    );
    check(
        view.evaluate( 1., *f.sample ) == .375 && f.value( *f.sample ) == f.value( *f.state ),
        "Clipped W endpoint is inconsistent"
    );
    for ( double theta :
          { -.1, 1.1, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() } )
        throws( [&] { view.evaluate( theta, *f.sample ); }, "W view accepted an invalid coordinate" );
    throws( [&] { step.set_time( 1 ); }, "Pending W view allowed time mutation" );
    const auto shortened = view.evaluate( .5, *f.state );
    step.finalize( adaptation_status::accepted_modified, shortened, *f.state );
    throws( [&] { view.evaluate( .5, *f.sample ); }, "Modified W commit retained its old view" );
    step.set_time( shortened );
    f.ops->assign( *f.state, *f.sample );
    step.apply( *f.state, *f.state );
    check(
        step.get_status() == single_step_status::converged && f.p->prepared_time == shortened,
        "W shortening did not reprepare the base"
    );
    auto second = step.get_continuous_integration();
    throws( [&] { view.evaluate( .5, *f.sample ); }, "A new W attempt revived an old view" );
    step.finalize( adaptation_status::rejected, shortened, *f.sample );
    f.ops->assign( *f.sample, *f.state );
    throws( [&] { second.evaluate( .5, *f.sample ); }, "Rejected W candidate retained its view" );
    step.apply( *f.state, *f.state );
    auto third = step.get_continuous_integration();
    step.reset();
    throws( [&] { third.evaluate( .5, *f.sample ); }, "W reset retained a borrowed view" );
}

void test_failures()
{
    // Both dense solves and the endpoint physical rate fail independently.
    // A finite but overflowing solve result exercises slope reconstruction too.
    for ( unsigned int provider : { 5u, 6u, 0u } )
        for ( unsigned int failure = 0; failure < ( provider ? 4u : 3u ); ++failure )
            for ( bool inplace : { false, true } )
            {
                fixture f;
                if ( provider )
                {
                    f.linear->fail_on     = failure == 0 ? provider : 0;
                    f.linear->invalid_on  = failure == 1 ? provider : 0;
                    f.linear->throw_on    = failure == 2 ? provider : 0;
                    f.linear->overflow_on = failure == 3 ? provider : 0;
                }
                else
                {
                    f.p->fail_rate_on    = failure == 0 ? 5 : 0;
                    f.p->invalid_rate_on = failure == 1 ? 5 : 0;
                    f.p->throw_rate_on   = failure == 2 ? 5 : 0;
                }
                S step(
                    f.ops, f.p, std::make_shared<adaptation_type>( adaptation_type::params{ .01 } ), f.linear,
                    f.settings
                );
                step.set_time( .37 );
                auto        &out    = inplace ? *f.state : *f.out;
                const double before = f.value( out );
                bool         threw  = false;
                try
                {
                    step.apply( *f.state, out );
                }
                catch ( const std::runtime_error & )
                {
                    threw = true;
                }
                check(
                    threw == ( failure == 2 ) && f.value( out ) == before && f.p->time == .37,
                    "W dense failure changed committed state, output or base time"
                );
                if ( !threw )
                    check(
                        step.get_status() == ( failure == 1 || failure == 3 ? single_step_status::failed_nonfinite
                                                                            : single_step_status::failed_solver ),
                        "W dense failure lost its status"
                    );
                throws( [&] { step.get_continuous_integration(); }, "Failed W preparation exposed a view" );
                f.linear->fail_on = f.linear->invalid_on = f.linear->throw_on = f.linear->overflow_on = 0;
                f.p->fail_rate_on = f.p->invalid_rate_on = f.p->throw_rate_on = 0;
                const auto calls = f.linear->calls, rates = f.p->rates;
                step.apply( *f.state, out );
                check(
                    step.get_status() == single_step_status::converged && f.linear->calls == calls + 6 &&
                        f.p->rates == rates + 5 && f.linear->bindings == 2,
                    "W retry reused failed dense data or cached physical rates"
                );
                step.finalize( adaptation_status::accepted, .38, out );
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
void test_retries()
{
    using Step = rosenbrock::time_step<operations_type, P, retry_controller, L, true>;
    for ( unsigned int provider : { 0u, 5u, 6u } )
    {
        fixture f;
        auto    control   = std::make_shared<retry_controller>();
        f.linear->fail_on = provider;
        f.p->fail_rate_on = provider ? 0 : 5;
        Step::params sp;
        sp.method     = "ROS34PW2";
        sp.autonomous = false;
        Step step( f.ops, f.p, control, f.linear, sp );
        step.set_time( .37 );
        step.set_target_time( .39 );
        step.apply( *f.state, *f.out );
        check(
            step.get_status() == single_step_status::converged && step.get_attempts() == 2 && control->failures == 1 &&
                control->commits == 0 && f.linear->bindings == 2 && f.p->derivatives == 2,
            "W dense retry failed to rebuild the shifted system"
        );
        check(
            std::abs( f.value( *f.out ) - std::exp( 2 * ( .37 + step.get_dt() ) ) ) < 1e-8,
            "W retry used an incorrect base state"
        );
        step.finalize( adaptation_status::accepted, .37 + step.get_dt(), *f.out );
        check( control->commits == 1, "W dense retry committed a failed attempt" );
        sp.maximum_attempts = 1;
        f.linear->fail_on   = f.linear->calls + 5;
        Step limited( f.ops, f.p, control, f.linear, sp );
        f.ops->assign_scalar( 42., *f.out );
        limited.apply( *f.state, *f.out );
        check(
            limited.get_status() == single_step_status::attempt_limit_reached && f.value( *f.out ) == 42,
            "W dense budget exhaustion published a candidate"
        );
    }
}

// Each rejected accuracy trial performs only the original four solves/rates.
// Only the finally accepted candidate prepares the two additional slope stages.
template <class Controller>
void test_error_acceptance( const char *filter = nullptr )
{
    fixture                     f;
    typename Controller::params cp;
    cp.relative_tolerance = 1e-6;
    cp.absolute_tolerance = 1e-8;
    cp.initial_step = cp.maximum_step = .2;
    if constexpr ( std::is_same_v<Controller, integration::time_step_adaptation_filter<operations_type>> )
        cp.controller = filter;
    using Step = rosenbrock::time_step<operations_type, P, Controller, L, true>;
    typename Step::params sp;
    sp.method     = "ROS34PW2";
    sp.autonomous = false;
    Step step( f.ops, f.p, std::make_shared<Controller>( f.ops, cp ), f.linear, sp );
    step.set_time( .37 );
    step.set_target_time( .57 );
    step.apply( *f.state, *f.out );
    const auto attempts = step.get_attempts();
    check(
        step.get_status() == single_step_status::converged && attempts > 1 && f.linear->calls == 4 * attempts + 2 &&
            f.p->rates == 4 * attempts + 1 && f.linear->bindings == attempts && f.p->derivatives == attempts,
        "Error-rejected W attempts performed dense-only work"
    );
    step.get_continuous_integration().evaluate( .5, *f.sample );
    check(
        std::abs( f.value( *f.sample ) - std::exp( 2 * ( .37 + step.get_dt() / 2 ) ) ) < 1e-6,
        "Adaptive W dense sample is inaccurate"
    );
    step.finalize( adaptation_status::accepted, .37 + step.get_dt(), *f.out );
}

int main()
try
{
    backend_type::init_device();
    test_views();
    test_failures();
    test_retries();
    test_error_acceptance<integration::time_step_adaptation_matlab<operations_type>>();
    test_error_acceptance<integration::time_step_adaptation_scipy<operations_type>>();
    for ( const char *filter : { "I", "H211", "PC", "PID", "H312", "PPID", "H321" } )
        test_error_acceptance<integration::time_step_adaptation_filter<operations_type>>( filter );
    std::cout << "W dense borrowed views, slope/rate failures, rollback and accepted-only preparation: " << checks
              << " checks PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
