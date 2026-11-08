#include "implicit_rosenbrock_time_problem.h"
#include <nmfd/time_steppers/integration/time_step_adaptation_scipy.h>

using namespace rosenbrock_tests;

// Manufactured u=exp(2t) exercises the shared time-dependent rate. Failures
// after that rate is cached must not leak a partial candidate or poison retries.
using P = time_problem<>;
using L = direct_solver<P>;
using S = rosenbrock::time_step<operations_type, P, adaptation_type, L>;

struct fixture
{
    std::shared_ptr<operations_type> ops    = std::make_shared<operations_type>( 1 );
    std::shared_ptr<P>               p      = std::make_shared<P>( ops );
    std::shared_ptr<L>               linear = std::make_shared<L>( ops );
    vector_wrap_type                 in{ *ops }, out{ *ops };
    S::params                        settings;
    fixture()
    {
        settings.method     = "ROS3";
        settings.autonomous = false;
        in.start_use();
        out.start_use();
        ops->assign_scalar( std::exp( .74 ), *in );
        ops->assign_scalar( 42., *out );
    }
    double value( const vector_type &v )
    {
        return ops->get_value_at_point( 0, v );
    }
};

void test_failed_stages()
{
    for ( unsigned int stage : { 1u, 2u, 3u } )
        for ( bool invalid : { false, true } )
            for ( bool inplace : { false, true } )
            {
                fixture f;
                if ( invalid )
                    f.linear->invalid_on = stage;
                else
                    f.linear->fail_on = stage;
                S step(
                    f.ops, f.p, std::make_shared<adaptation_type>( adaptation_type::params{ .01 } ), f.linear,
                    f.settings
                );
                step.set_time( .37 );
                auto        &out    = inplace ? *f.in : *f.out;
                const double before = f.value( out );
                step.apply( *f.in, out );
                check(
                    step.get_status() ==
                            ( invalid ? single_step_status::failed_nonfinite : single_step_status::failed_solver ) &&
                        f.value( out ) == before && f.p->time == .37 && f.p->linearizations == 1,
                    "Failed ROS3 stage changed output, base time or committed input"
                );
                f.linear->fail_on = f.linear->invalid_on = 0;
                const auto rates                         = f.p->rates;
                step.apply( *f.in, out );
                check(
                    step.get_status() == single_step_status::converged && f.linear->bindings == 2 &&
                        f.p->rates == rates + 2 && std::abs( f.value( out ) - std::exp( .76 ) ) < 1e-7,
                    "ROS3 recovery reused a failed stage or cached rate"
                );
                step.finalize( adaptation_status::accepted, .38, out );
            }
    // Each rate and f_t provider can refuse, return nonfinite data, or throw.
    for ( unsigned int provider : { 0u, 1u, 2u } )
        for ( unsigned int failure : { 0u, 1u, 2u } )
        {
            fixture f;
            if ( !provider )
            {
                f.p->fail_derivative_on    = failure == 0;
                f.p->invalid_derivative_on = failure == 1;
                f.p->throw_derivative      = failure == 2;
            }
            else
            {
                f.p->fail_rate_on    = failure == 0 ? provider : 0;
                f.p->invalid_rate_on = failure == 1 ? provider : 0;
                f.p->throw_rate_on   = failure == 2 ? provider : 0;
            }
            S step( f.ops, f.p, std::make_shared<adaptation_type>(), f.linear, f.settings );
            step.set_time( .37 );
            bool threw = false;
            try
            {
                step.apply( *f.in, *f.out );
            }
            catch ( const std::runtime_error & )
            {
                threw = true;
            }
            check(
                threw == ( failure == 2 ) && f.value( *f.out ) == 42 && f.p->time == .37,
                "ROS3 provider failure was not transactional"
            );
            if ( !threw )
                check(
                    step.get_status() ==
                        ( failure == 1 ? single_step_status::failed_nonfinite : single_step_status::failed_solver ),
                    "ROS3 lost the provider failure status"
                );
        }
}

void test_adaptive_recovery()
{
    using C    = integration::time_step_adaptation_scipy<operations_type>;
    using Step = rosenbrock::time_step<operations_type, P, C, L>;
    for ( unsigned int stage : { 1u, 2u, 3u } )
    {
        fixture f;
        f.linear->fail_on = stage;
        C::params cp;
        cp.relative_tolerance = 1e-5;
        cp.absolute_tolerance = 1e-7;
        cp.initial_step = cp.maximum_step = .02;
        auto         control              = std::make_shared<C>( f.ops, cp );
        Step::params sp;
        sp.method     = "ROS3";
        sp.autonomous = false;
        Step step( f.ops, f.p, control, f.linear, sp );
        step.set_time( .37 );
        step.set_target_time( .39 );
        step.apply( *f.in, *f.out );
        check(
            step.get_status() == single_step_status::converged && step.get_attempts() >= 2 &&
                f.p->derivatives == step.get_attempts() && f.p->linearizations == step.get_attempts() &&
                std::abs( f.value( *f.out ) - std::exp( 2 * ( .37 + step.get_dt() ) ) ) < 1e-7,
            "Adaptive ROS3 did not rebuild its base operator and shared rate after a failed stage"
        );
        const auto rates = f.p->rates;
        step.finalize( adaptation_status::rejected, .37, *f.in );
        step.apply( *f.in, *f.out );
        check(
            step.get_status() == single_step_status::converged && f.p->rates >= rates + 2,
            "External rejection retained a ROS3 rate"
        );
        const double shortened = step.get_dt() / 2;
        f.ops->assign_scalar( std::exp( 2 * ( .37 + shortened ) ), *f.out );
        step.finalize( adaptation_status::accepted_modified, .37 + shortened, *f.out );
        step.set_time( .37 + shortened );
        f.p->varying = false;
        f.p->lambda  = 10;
        step.apply( *f.out, *f.out );
        check(
            step.get_status() == single_step_status::converged && f.p->jacobi->model.lambda == 10 &&
                std::abs( f.value( f.p->previous_state() ) - std::exp( 2 * ( .37 + shortened ) ) ) < 1e-15,
            "ROS3 retained a stale parameter, base state or linearization after endpoint modification"
        );
        step.finalize( adaptation_status::accepted, .37 + shortened + step.get_dt(), *f.out );
        step.reset();
        step.set_time( .37 );
        step.set_target_time( .39 );
        step.apply( *f.in, *f.out );
        check( step.get_status() == single_step_status::converged, "ROS3 reset did not release pending/cached data" );
        step.finalize( adaptation_status::accepted, .37 + step.get_dt(), *f.out );
    }
    fixture   f;
    C::params cp;
    cp.relative_tolerance = 1e-14;
    cp.absolute_tolerance = 1e-16;
    cp.initial_step = cp.minimum_step = cp.maximum_step = .1;
    Step::params sp;
    sp.method     = "ROS3";
    sp.autonomous = false;
    Step step( f.ops, f.p, std::make_shared<C>( f.ops, cp ), f.linear, sp );
    step.apply( *f.in, *f.out );
    check(
        step.get_status() == single_step_status::failed_minimum_dt && f.value( *f.out ) == 42,
        "ROS3 minimum-step failure published an unaccepted candidate"
    );
    cp.minimum_step     = 1e-12;
    sp.maximum_attempts = 1;
    Step limited( f.ops, f.p, std::make_shared<C>( f.ops, cp ), f.linear, sp );
    limited.apply( *f.in, *f.out );
    check(
        limited.get_status() == single_step_status::attempt_limit_reached && f.value( *f.out ) == 42,
        "ROS3 retry-budget exhaustion published a candidate"
    );
}

int main()
try
{
    backend_type::init_device();
    test_failed_stages();
    test_adaptive_recovery();
    std::cout << "ROS3 three-stage failures, shared-rate retries, rollback, rebinding and reset: " << checks
              << " checks PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
