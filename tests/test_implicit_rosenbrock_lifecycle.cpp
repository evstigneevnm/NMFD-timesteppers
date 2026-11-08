#include "implicit_rosenbrock_test.h"

using namespace rosenbrock_tests;

// u'=1 with identity mass has exact increments for every signed h. Numerical
// failures and external endpoint changes can therefore be tested independently
// of discretization error, without solving a different nonlinear problem.
using P = problem<constant_rate>;
using L = direct_solver<P>;

struct controller : adaptation_type
{
    double       next = .1, committed_time = 0, committed_dt = 0;
    unsigned int failures = 0, assessments = 0, commits = 0;
    bool         needs_error = false, reject_candidate = false;
    void         reset()
    {
        next     = .1;
        failures = assessments = commits = 0;
    }
    double initialize( double, const vector_type & )
    {
        return next;
    }
    double get_dt() const
    {
        return next;
    }
    bool requires_error_estimate() const
    {
        return needs_error;
    }
    adaptation_status reject_step( double h )
    {
        ++failures;
        next = std::abs( h ) / 2;
        return adaptation_status::rejected;
    }
    adaptation_status assess( double, double h, const vector_type &, const vector_type &, double &next_dt )
    {
        ++assessments;
        next_dt = next;
        if ( reject_candidate )
        {
            next = std::abs( h ) / 2;
            return adaptation_status::rejected;
        }
        return adaptation_status::accepted;
    }
    void update( adaptation_status outcome, double t, double h, const vector_type & )
    {
        if ( outcome != adaptation_status::rejected )
        {
            ++commits;
            committed_time = t;
            committed_dt   = h;
        }
    }
};
using S = rosenbrock::time_step<operations_type, P, controller, L>;

struct fixture
{
    std::shared_ptr<operations_type> ops        = std::make_shared<operations_type>( 1 );
    std::shared_ptr<P>               p          = std::make_shared<P>( ops );
    std::shared_ptr<L>               solver     = std::make_shared<L>( ops );
    std::shared_ptr<controller>      adaptation = std::make_shared<controller>();
    vector_wrap_type                 u{ *ops }, out{ *ops };
    fixture()
    {
        u.start_use();
        out.start_use();
        ops->assign_scalar( 0., *u );
        ops->assign_scalar( 42., *out );
    }
    double value() const
    {
        return ops->get_value_at_point( 0, *out );
    }
};

template <class F>
void throws( F f, const char *message )
{
    bool failed = false;
    try
    {
        f();
    }
    catch ( const std::exception & )
    {
        failed = true;
    }
    check( failed, message );
}

void test_configuration()
{
    fixture f;
    for ( const char *method : { "unknown", "ROS4", "" } )
        throws( [&] { S s( f.ops, f.p, f.adaptation, f.solver, S::params{ method } ); }, "Unknown ROS table accepted" );
    for ( double scale :
          { 0., -1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() } )
        throws(
            [&] {
                S::params p;
                p.reference_time = scale;
                S s( f.ops, f.p, f.adaptation, f.solver, p );
            },
            "Invalid ROS reference time accepted"
        );
    throws(
        [&] {
            S::params p;
            p.maximum_attempts = 0;
            S s( f.ops, f.p, f.adaptation, f.solver, p );
        },
        "Empty ROS budget accepted"
    );
    throws(
        [&] {
            S::params p;
            p.autonomous = false;
            S s( f.ops, f.p, f.adaptation, f.solver, p );
        },
        "Non-autonomous ROS request accepted"
    );
    throws(
        [&] { rosenbrock::time_step<operations_type, P, controller, L, true> s( f.ops, f.p, f.adaptation, f.solver ); },
        "Unvalidated ROS dense output accepted"
    );
    throws( [&] { S s( nullptr, f.p, f.adaptation, f.solver ); }, "Null ROS operations accepted" );
    throws( [&] { S s( f.ops, nullptr, f.adaptation, f.solver ); }, "Null ROS problem accepted" );
    throws( [&] { S s( f.ops, f.p, nullptr, f.solver ); }, "Null ROS controller accepted" );
    throws( [&] { S s( f.ops, f.p, f.adaptation, nullptr ); }, "Null ROS solver accepted" );
    S s( f.ops, f.p, f.adaptation, f.solver );
    f.adaptation->needs_error = true;
    s.apply( *f.u, *f.out );
    check(
        s.get_status() == single_step_status::error_estimate_unavailable && s.get_attempts() == 0 && f.value() == 42,
        "ROS error-estimate refusal changed output or attempted a solve"
    );
    f.adaptation->needs_error = false;
    f.p->null_jacobian        = true;
    throws( [&] { s.apply( *f.u, *f.out ); }, "Null ROS Jacobian accepted" );
    check( f.value() == 42, "ROS binding exception published output" );
}

void test_failures()
{
    for ( unsigned int stage : { 1u, 2u } )
    {
        fixture f;
        f.solver->fail_on = stage;
        S s( f.ops, f.p, f.adaptation, f.solver );
        s.apply( *f.u, *f.out );
        check(
            s.get_status() == single_step_status::converged && s.get_attempts() == 2 && f.adaptation->failures == 1 &&
                f.solver->bindings == 2 && std::abs( f.value() - .05 ) < 1e-15,
            "ROS linear-stage retry did not rebuild from the unchanged input"
        );
        s.finalize( adaptation_status::accepted, .05, *f.out );
    }
    for ( unsigned int stage : { 1u, 2u } )
        for ( bool invalid : { false, true } )
        {
            fixture f;
            using Fixed = rosenbrock::time_step<operations_type, P, adaptation_type, L>;
            if ( invalid )
                f.solver->invalid_on = stage;
            else
                f.solver->fail_on = stage;
            Fixed s( f.ops, f.p, std::make_shared<adaptation_type>(), f.solver );
            s.apply( *f.u, *f.out );
            check(
                s.get_status() ==
                        ( invalid ? single_step_status::failed_nonfinite : single_step_status::failed_solver ) &&
                    f.value() == 42 && f.p->linearizations == 1,
                "Failed ROS stage leaked its partial increment"
            );
            s.reset();
            s.apply( *f.u, *f.out );
            check( s.get_status() == single_step_status::converged, "ROS did not recover after a failed solve" );
            s.finalize( adaptation_status::accepted, s.get_dt(), *f.out );
        }
    for ( unsigned int stage : { 1u, 2u } )
        for ( bool invalid : { false, true } )
        {
            fixture f;
            if ( invalid )
                f.p->invalid_rate_on = stage;
            else
                f.p->fail_rate_on = stage;
            S s( f.ops, f.p, f.adaptation, f.solver );
            s.apply( *f.u, *f.out );
            check(
                s.get_status() == single_step_status::converged && s.get_attempts() == 2 &&
                    f.adaptation->failures == 1 && f.p->linearizations == 2 && std::abs( f.value() - .05 ) < 1e-15,
                "ROS rate retry used a contaminated initial state or stale shift"
            );
            check( f.solver->bindings == 2, "ROS rate failure did not rebind the retried operator" );
            s.finalize( adaptation_status::accepted, .05, *f.out );
        }
    for ( bool rate_exception : { false, true } )
    {
        fixture f;
        f.p->throw_rate  = rate_exception;
        f.solver->throws = !rate_exception;
        S s( f.ops, f.p, f.adaptation, f.solver );
        throws( [&] { s.apply( *f.u, *f.out ); }, "ROS swallowed a backend/configuration exception" );
        check( f.value() == 42, "ROS exception changed output" );
    }
    {
        fixture f;
        f.solver->fail_on = 2;
        using Fixed       = rosenbrock::time_step<operations_type, P, adaptation_type, L>;
        Fixed s( f.ops, f.p, std::make_shared<adaptation_type>(), f.solver );
        s.apply( *f.u, *f.u );
        check(
            s.get_status() == single_step_status::failed_solver && f.ops->get_value_at_point( 0, *f.u ) == 0,
            "ROS in-place failure corrupted its input"
        );
    }
    fixture f;
    f.adaptation->reject_candidate = true;
    S s( f.ops, f.p, f.adaptation, f.solver, S::params{ "ROS2", 3 } );
    s.apply( *f.u, *f.out );
    check(
        s.get_status() == single_step_status::attempt_limit_reached && s.get_attempts() == 3 && f.value() == 42,
        "ROS candidate rejection ignored its budget or changed output"
    );
}

void test_commit_and_restart()
{
    fixture f;
    S       s( f.ops, f.p, f.adaptation, f.solver );
    s.apply( *f.u, *f.out );
    check(
        s.get_status() == single_step_status::converged && std::abs( f.value() - .1 ) < 1e-15 && !f.adaptation->commits,
        "ROS apply committed state prematurely"
    );
    throws( [&] { s.apply( *f.u, *f.out ); }, "ROS overwrote a pending candidate" );
    throws( [&] { s.set_time( 1 ); }, "ROS changed time with a pending candidate" );
    throws( [&] { s.set_target_time( 2 ); }, "ROS changed target with a pending candidate" );
    throws( [&] { s.finalize( adaptation_status::accepted, .2, *f.out ); }, "ROS accepted an out-of-step endpoint" );
    check( f.adaptation->commits == 0, "Invalid ROS finalize updated controller" );
    f.ops->assign_scalar( .035, *f.out );
    s.finalize( adaptation_status::accepted_modified, .035, *f.out );
    check(
        f.adaptation->committed_time == .035 && f.adaptation->committed_dt == .035,
        "ROS controller missed the shortened endpoint"
    );
    s.set_time( .035 );
    s.set_target_time( .1 );
    s.apply( *f.out, *f.out );
    check(
        std::abs( f.value() - .1 ) < 1e-15 && f.p->linearizations == 2,
        "ROS reused stale data after external modification"
    );
    s.finalize( adaptation_status::accepted, .1, *f.out );
    throws( [&] { s.finalize( adaptation_status::accepted, .1, *f.out ); }, "ROS finalized twice" );
    s.reset();
    s.set_time( 0 );
    s.set_target_time( .1 );
    s.apply( *f.u, *f.out );
    s.finalize( adaptation_status::rejected, 0, *f.u );
    s.apply( *f.u, *f.out );
    check( std::abs( f.value() - .1 ) < 1e-15, "ROS external rejection corrupted the next attempt" );
    s.reset();
    check(
        s.get_status() == single_step_status::idle && s.get_dt() == 0 && s.get_attempts() == 0,
        "ROS reset retained trajectory state"
    );
    s.set_time( 1 );
    s.set_target_time( .97 );
    f.ops->assign_scalar( 1., *f.u );
    s.apply( *f.u, *f.out );
    check(
        std::abs( f.value() - .97 ) < 1e-15 && std::abs( s.get_dt() + .03 ) < 1e-15,
        "ROS backward clipping is incorrect"
    );
    s.finalize( adaptation_status::accepted, .97, *f.out );
    for ( double bad : { std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() } )
    {
        throws( [&] { s.set_time( bad ); }, "ROS accepted a nonfinite start" );
        throws( [&] { s.set_target_time( bad ); }, "ROS accepted a nonfinite target" );
    }
    s.set_time( 1 );
    s.set_target_time( 1 );
    s.apply( *f.u, *f.out );
    check( s.get_status() == single_step_status::step_size_underflow, "ROS attempted a zero-length step" );
    f.ops->assign_scalar( std::numeric_limits<double>::infinity(), *f.u );
    s.apply( *f.u, *f.out );
    check(
        s.get_status() == single_step_status::failed_nonfinite && s.get_attempts() == 0,
        "ROS accepted a nonfinite input"
    );
}

void test_ownership()
{
    fixture                   f;
    std::weak_ptr<P>          problem_lifetime    = f.p;
    std::weak_ptr<L>          solver_lifetime     = f.solver;
    std::weak_ptr<controller> controller_lifetime = f.adaptation;
    auto                      s                   = std::make_unique<S>( f.ops, f.p, f.adaptation, f.solver );
    f.p.reset();
    f.solver.reset();
    f.adaptation.reset();
    check(
        !problem_lifetime.expired() && !solver_lifetime.expired() && !controller_lifetime.expired(),
        "ROS did not retain supplied dependencies"
    );
    s->apply( *f.u, *f.out );
    s.reset();
    check(
        problem_lifetime.expired() && solver_lifetime.expired() && controller_lifetime.expired(),
        "ROS introduced an ownership cycle"
    );
}

int main()
try
{
    backend_type::init_device();
    test_configuration();
    test_failures();
    test_commit_and_restart();
    test_ownership();
    std::cout << "ROS2 failure rollback, lifecycle, retry, restart and ownership: " << checks << " checks PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
