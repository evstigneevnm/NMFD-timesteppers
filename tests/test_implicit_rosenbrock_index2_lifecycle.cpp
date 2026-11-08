#include "implicit_rosenbrock_index2_problem.h"

using namespace rosenbrock_tests;

template <class Exception = std::exception, class Action>
void refuses( Action action, const char *message )
{
    bool refused = false;
    try
    {
        action();
    }
    catch ( const Exception & )
    {
        refused = true;
    }
    check( refused, message );
}

void selection_and_initialization()
{
    using P     = index2_problem<true>;
    using C     = index2_control<adaptation_type, P>;
    using L     = index2_solver<P>;
    using S     = rosenbrock::time_step<operations_type, P, C, L>;
    using Dense = rosenbrock::time_step<operations_type, P, C, L, true>;
    static_assert(
        !detail::has_rate_evaluation<P, vector_type>::value, "Ordinary index-2 stepping must not require rates"
    );
    auto ops = std::make_shared<operations_type>( 5 );
    auto p   = std::make_shared<P>( ops );
    auto c   = std::make_shared<C>( p, adaptation_type::params{ .01 } );
    auto l   = std::make_shared<L>( ops );
    for ( const char *method : { "ROS2", "ROS3", "ROS34PW2" } )
    {
        auto sp   = index2_params<S>();
        sp.method = method;
        refuses<std::invalid_argument>(
            [&] { S invalid( ops, p, c, l, sp ); }, "ODE/index-1 table accepted index-2 mode"
        );
    }
    for ( unsigned int option = 0; option < 3; ++option )
    {
        auto sp = index2_params<S>();
        if ( option == 0 )
            sp.mass_form = false;
        if ( option == 1 )
            sp.autonomous = false;
        if ( option == 2 )
            sp.index1_dae = true;
        refuses<std::invalid_argument>( [&] { S invalid( ops, p, c, l, sp ); }, "Invalid index-2 selection accepted" );
    }
    refuses<std::invalid_argument>(
        [&] { Dense invalid( ops, p, c, l, index2_params<Dense>() ); },
        "Dense index-2 silently invented a singular mass inverse"
    );
    vector_wrap_type in( *ops ), out( *ops );
    in.start_use();
    out.start_use();
    for ( unsigned int component : { 0u, 2u, 4u } )
    {
        p->initialize_state( .37, *in );
        ops->set_value_at_point( ops->get_value_at_point( component, *in ) + .1, component, *in );
        ops->assign_scalar( 42., *out );
        S step( ops, p, c, l, index2_params<S>() );
        step.set_time( .37 );
        refuses<std::invalid_argument>(
            [&] { step.apply( *in, *out ); }, "Inconsistent state/pressure/clock was silently projected"
        );
        check(
            ops->get_value_at_point( 0, *out ) == 42 && l->calls == 0, "Inconsistent initialization published or solved"
        );
    }
    p->gauge = pressure_gauge::missing;
    p->initialize_state( .37, *in );
    S missing( ops, p, c, l, index2_params<S>() );
    refuses<std::invalid_argument>( [&] { missing.apply( *in, *out ); }, "Missing pressure gauge accepted" );
}

// Every stage failure and dense-only rate failure must leave the input,
// output and frozen base state unchanged; no view may escape a failed attempt.
template <bool Dense>
void failures()
{
    using P = index2_problem<true, operations_type, Dense>;
    using L = index2_solver<P>;
    using S = rosenbrock::time_step<operations_type, P, adaptation_type, L, Dense>;
    for ( unsigned int source = 0; source < ( Dense ? 3u : 2u ); ++source )
        for ( unsigned int kind = 0; kind < 3; ++kind )
            for ( unsigned int at = 1; at <= ( source == 2 ? 2u : 4u ); ++at )
            {
                auto ops    = std::make_shared<operations_type>( 5 );
                auto p      = std::make_shared<P>( ops );
                auto solver = std::make_shared<L>( ops );
                if ( source == 0 )
                {
                    solver->fail_on    = kind == 0 ? at : 0;
                    solver->invalid_on = kind == 1 ? at : 0;
                    solver->throw_on   = kind == 2 ? at : 0;
                }
                else if ( source == 1 )
                {
                    if ( kind == 0 )
                        continue; // apply() returns void: finite residuals do not signal failure.
                    p->invalid_on = kind == 1 ? at : 0;
                    p->throw_on   = kind == 2 ? at : 0;
                }
                else
                {
                    p->fail_rate_on    = kind == 0 ? at : 0;
                    p->invalid_rate_on = kind == 1 ? at : 0;
                    p->throw_rate_on   = kind == 2 ? at : 0;
                }
                S step(
                    ops, p, std::make_shared<adaptation_type>( adaptation_type::params{ .01 } ), solver,
                    index2_params<S>()
                );
                vector_wrap_type in( *ops ), out( *ops ), snapshot( *ops ), scratch( *ops );
                for ( auto *v : { &in, &out, &snapshot, &scratch } )
                    v->start_use();
                p->initialize_state( .37, *in );
                ops->assign( *in, *snapshot );
                ops->assign_scalar( 42., *out );
                step.set_time( .37 );
                step.set_target_time( .38 );
                bool threw = false;
                try
                {
                    step.apply( *in, *out );
                }
                catch ( const std::runtime_error & )
                {
                    threw = true;
                }
                check( threw == ( kind == 2 ), "Index-2 numerical/exception failure classification changed" );
                check(
                    ops->get_value_at_point( 0, *out ) == 42 && p->time == .37,
                    "Failed index-2 candidate changed output/time"
                );
                ops->assign_lin_comb( 1., *in, -1., *snapshot, *scratch );
                check( ops->norm( *scratch ) == 0, "Failed index-2 attempt modified input" );
                ops->assign_lin_comb( 1., *p->jacobi->point, -1., *snapshot, *scratch );
                check( ops->norm( *scratch ) == 0, "Index-2 trial modified the frozen operator point" );
                if ( !threw )
                    check(
                        step.get_status() ==
                            ( kind == 1 ? single_step_status::failed_nonfinite : single_step_status::failed_solver ),
                        "Index-2 failure status lost"
                    );
                if constexpr ( Dense )
                    refuses<std::logic_error>(
                        [&] { step.get_continuous_integration(); }, "Failed index-2 attempt exposed dense output"
                    );
            }
}

struct retry_control : adaptation_type
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
    void update( adaptation_status outcome, double, double, const vector_type & )
    {
        commits += outcome != adaptation_status::rejected;
    }
};

void retries_and_views()
{
    using P = index2_problem<true, operations_type, true>;
    using L = index2_solver<P>;
    using S = rosenbrock::time_step<operations_type, P, retry_control, L, true>;
    for ( bool dense_failure : { false, true } )
    {
        auto ops = std::make_shared<operations_type>( 5 );
        auto p   = std::make_shared<P>( ops );
        auto l   = std::make_shared<L>( ops );
        auto c   = std::make_shared<retry_control>();
        if ( dense_failure )
            p->fail_rate_on = 2;
        else
            l->fail_on = 3;
        S                step( ops, p, c, l, index2_params<S>() );
        vector_wrap_type in( *ops ), out( *ops ), sample( *ops );
        in.start_use();
        out.start_use();
        sample.start_use();
        p->initialize_state( .37, *in );
        step.set_time( .37 );
        step.set_target_time( .4 );
        step.apply( *in, *out );
        check(
            step.get_status() == single_step_status::converged && step.get_attempts() == 2 && l->bindings == 2 &&
                c->failures == 1 && c->commits == 0,
            "Index-2 retry did not rebuild an uncommitted candidate"
        );
        auto view = step.get_continuous_integration();
        for ( double theta : { -.1, 1.1, std::numeric_limits<double>::quiet_NaN() } )
            refuses<std::invalid_argument>(
                [&] { view.evaluate( theta, *sample ); }, "Index-2 view accepted invalid theta"
            );
        refuses<std::logic_error>( [&] { step.set_time( 1 ); }, "Pending index-2 step allowed time mutation" );
        step.finalize( adaptation_status::rejected, .37, *in );
        refuses<std::logic_error>( [&] { view.evaluate( .5, *sample ); }, "Rejected index-2 view remained live" );
        step.apply( *in, *out );
        auto second = step.get_continuous_integration();
        step.reset();
        refuses<std::logic_error>( [&] { second.evaluate( .5, *sample ); }, "Reset index-2 view remained live" );
        step.apply( *in, *out );
        const auto third = step.get_continuous_integration();
        step.finalize( adaptation_status::accepted, .37 + step.get_dt(), *out );
        refuses<std::logic_error>( [&] { third.evaluate( .5, *sample ); }, "Accepted index-2 view remained live" );
        auto limited_params             = index2_params<S>();
        limited_params.maximum_attempts = 1;
        p->fail_rate_on                 = p->rates + 2;
        S limited( ops, p, c, l, limited_params );
        limited.set_time( .37 );
        ops->assign_scalar( 42., *out );
        limited.apply( *in, *out );
        check(
            limited.get_status() == single_step_status::attempt_limit_reached &&
                ops->get_value_at_point( 0, *out ) == 42,
            "Exhausted index-2 retries published a candidate"
        );
    }
    // The optional provider can explicitly refuse an endpoint with an
    // excessive hidden defect; dense preparation must not silently project it.
    auto ops                      = std::make_shared<operations_type>( 5 );
    auto p                        = std::make_shared<P>( ops );
    p->rate_consistency_tolerance = 1e-12;
    using Fixed                   = rosenbrock::time_step<operations_type, P, adaptation_type, L, true>;
    Fixed step(
        ops, p, std::make_shared<adaptation_type>( adaptation_type::params{ .01 } ), std::make_shared<L>( ops ),
        index2_params<Fixed>()
    );
    vector_wrap_type in( *ops ), out( *ops );
    in.start_use();
    out.start_use();
    p->initialize_state( .37, *in );
    ops->assign_scalar( 42., *out );
    step.set_time( .37 );
    step.apply( *in, *out );
    check(
        step.get_status() == single_step_status::failed_solver && ops->get_value_at_point( 0, *out ) == 42,
        "Inconsistent endpoint was accepted for dense rates"
    );
}

int main()
try
{
    backend_type::init_device();
    selection_and_initialization();
    failures<false>();
    failures<true>();
    retries_and_views();
    std::cout << "PASS: Rosenbrock index-2 initialization, failures, rollback and views (" << checks << " checks)\n";
}
catch ( const std::exception &e )
{
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
