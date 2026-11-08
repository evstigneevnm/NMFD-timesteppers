#include "implicit_rosenbrock_mass_problem.h"

using namespace rosenbrock_tests;

// Coupled decay M*u'=M*[-2*x+y,x-2*y], initially u=(1,1), so u=exp(-t).
// Deliberately corrupt stage/dense solves and residuals: no failed attempt may
// publish output. Every retry must reprepare the operator and reference buffers.
template <bool Dense>
void lifecycle( const char *method )
{
    using P                    = mass_problem<coupled_decay>;
    using L                    = direct_solver<P>;
    using S                    = rosenbrock::time_step<operations_type, P, adaptation_type, L, Dense>;
    auto               ops     = std::make_shared<operations_type>( 2 );
    auto               p       = std::make_shared<P>( ops );
    auto               linear  = std::make_shared<L>( ops );
    auto               control = std::make_shared<adaptation_type>( adaptation_type::params{ .02 } );
    typename S::params sp;
    sp.method               = method;
    sp.mass_form            = true;
    const auto       stages = rosenbrock::method_table( method ).stage_count();
    const auto       solves = stages + ( Dense ? ( stages == 4 ? 2 : stages == 3 ? 1 : 0 ) : 0 );
    vector_wrap_type in( *ops ), out( *ops ), sample( *ops );
    in.start_use();
    out.start_use();
    sample.start_use();
    ops->assign_scalar( 1., *in );
    for ( std::size_t failure = 1; failure <= solves; ++failure )
        for ( unsigned int kind = 0; kind < 4; ++kind )
        {
            p->residuals = p->linearizations = linear->calls = linear->bindings = 0;
            p->invalid_on = p->throw_on = linear->fail_on = linear->invalid_on = 0;
            if ( kind == 0 )
                linear->fail_on = failure;
            if ( kind == 1 )
                linear->invalid_on = failure;
            if ( kind == 2 )
                p->invalid_on = failure;
            if ( kind == 3 )
                p->throw_on = failure;
            S step( ops, p, control, linear, sp );
            ops->assign_scalar( 77., *out );
            bool threw = false;
            try
            {
                step.apply( *in, *out );
            }
            catch ( const std::runtime_error & )
            {
                threw = true;
            }
            check( threw == ( kind == 3 ), "Mass-form exception handling changed" );
            check(
                ops->get_value_at_point( 0, *out ) == 77. && ops->get_value_at_point( 0, *in ) == 1.,
                "Failed mass-form attempt published or modified a state"
            );
            if constexpr ( Dense )
            {
                bool expired = false;
                try
                {
                    step.get_continuous_integration();
                }
                catch ( const std::logic_error & )
                {
                    expired = true;
                }
                check( expired, "Failed mass-form attempt published a dense view" );
            }
            p->invalid_on = p->throw_on = linear->fail_on = linear->invalid_on = 0;
            step.apply( *in, *out );
            check(
                step.get_status() == single_step_status::converged && p->linearizations == 2,
                "Mass-form recovery did not reprepare its operator"
            );
            step.finalize( adaptation_status::accepted_modified, .01, *out );
            p->mass = { 3, -.4, .2, 2 };
            step.set_time( .01 );
            step.set_target_time( .03 );
            step.apply( *in, *out );
            check(
                step.get_status() == single_step_status::converged && p->jacobi->mass.m00 == 3,
                "Mass-form rebinding retained an old mass operator"
            );
            if constexpr ( Dense )
            {
                auto view = step.get_continuous_integration();
                view.evaluate( .5, *sample );
                step.finalize( adaptation_status::rejected, .01, *in );
                bool expired = false;
                try
                {
                    view.evaluate( .5, *sample );
                }
                catch ( const std::logic_error & )
                {
                    expired = true;
                }
                check( expired, "Rejected mass-form dense view remained valid" );
            }
            else
                step.finalize( adaptation_status::rejected, .01, *in );
            step.reset();
            step.set_time( 0 );
            step.apply( *in, *out );
            check( step.get_status() == single_step_status::converged, "Mass-form reset failed" );
            step.finalize( adaptation_status::accepted, .02, *out );
        }
    S inplace( ops, p, control, linear, sp );
    inplace.apply( *in, *in );
    check(
        inplace.get_status() == single_step_status::converged &&
            std::abs( ops->get_value_at_point( 0, *in ) - std::exp( -.02 ) ) < 2e-5,
        "In-place mass-form step corrupted its input before publication"
    );
    if constexpr ( Dense )
    {
        auto view = inplace.get_continuous_integration();
        view.evaluate( 0, *sample );
        check( ops->get_value_at_point( 0, *sample ) == 1., "Mass-form dense start lost its in-place snapshot" );
        view.evaluate( 1, *sample );
        check(
            ops->get_value_at_point( 0, *sample ) == ops->get_value_at_point( 0, *in ),
            "Mass-form dense endpoint differs from the published state"
        );
    }
    inplace.finalize( adaptation_status::accepted, .02, *in );
    sp.autonomous = false;
    bool refused  = false;
    try
    {
        S invalid( ops, p, control, linear, sp );
    }
    catch ( const std::invalid_argument & )
    {
        refused = true;
    }
    check( refused, "Mass form accepted non-autonomous execution without a partial-time callback" );
    sp.autonomous = true;
    sp.mass_form  = false;
    refused       = false;
    try
    {
        S invalid( ops, p, control, linear, sp );
    }
    catch ( const std::invalid_argument & )
    {
        refused = true;
    }
    check( refused, "Rate form silently accepted a problem without evaluate_rate" );
}

int main()
try
{
    backend_type::init_device();
    for ( const char *method : { "ROS2", "ROS3", "ROS34PW2" } )
    {
        lifecycle<false>( method );
        lifecycle<true>( method );
    }
    std::cout << "PASS: Rosenbrock mass lifecycle (" << checks << " checks)\n";
}
catch ( const std::exception &e )
{
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
