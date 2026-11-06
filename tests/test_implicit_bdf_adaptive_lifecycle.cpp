#include "implicit_bdf_adaptive_test.h"
#include <limits>

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;
namespace model    = nmfd::time_steppers::tests::bdf_test;
using problem_type = model::problem_type;
using types        = newton_types<problem_type>;

// u'=t-u from implicit_bdf_test.h. The controller deliberately prescribes its
// grid; real Newton solves reveal every reference, restart and history update.
struct prescribed_controller
{
    std::shared_ptr<operations_type> ops;
    double                           next = .1, error = 0;
    unsigned int                     assessments = 0, commits = 0, exponent = 0, orders_seen = 0;
    bool                             reject = false, numerical_retry = false;
    void                             reset()
    {
        next        = .1;
        assessments = commits = 0;
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
        return true;
    }
    adaptation_status assess(
        double, double h, const vector_type &, const vector_type &, double &dt_next, unsigned int order = 0,
        const vector_type *estimate = nullptr
    )
    {
        require( estimate && order >= 2 && order <= 6, "Missing BDF state-error estimate" );
        orders_seen |= 1u << ( order - 1 );
        error    = ops->get_value_at_point( 0, *estimate );
        exponent = order;
        ++assessments;
        if ( reject )
        {
            reject = false;
            next   = std::abs( h ) / 2;
            return adaptation_status::rejected;
        }
        dt_next = next;
        return adaptation_status::accepted;
    }
    void update( adaptation_status outcome, double, double, const vector_type & )
    {
        commits += outcome == adaptation_status::accepted || outcome == adaptation_status::accepted_modified;
    }
    adaptation_status reject_step( double )
    {
        return numerical_retry ? adaptation_status::rejected : adaptation_status::failed;
    }
};

struct controlled_solver
{
    std::shared_ptr<operations_type>    ops;
    std::shared_ptr<types::solver_type> solver;
    unsigned int                        bindings = 0, calls = 0, fail_on = 0, invalid_on = 0, failures_left = 0;
    void                                set_nonlinear_operator( std::shared_ptr<problem_type> problem )
    {
        ++bindings;
        solver->set_nonlinear_operator( problem );
    }
    bool solve( vector_type &state )
    {
        ++calls;
        if ( calls == fail_on || failures_left )
        {
            if ( failures_left )
                --failures_left;
            ops->assign_scalar( 99., state );
            return false;
        }
        const bool ok = solver->solve( state );
        if ( calls == invalid_on )
            ops->assign_scalar( std::numeric_limits<double>::infinity(), state );
        return ok;
    }
};
using step_type = bdf::implicit_time_step<operations_type, problem_type, prescribed_controller, controlled_solver>;

void test_startup_and_rollback()
{
    auto ops     = std::make_shared<operations_type>( 1 );
    auto problem = std::make_shared<problem_type>( ops );
    auto control = std::make_shared<prescribed_controller>( prescribed_controller{ ops } );
    auto solver  = std::make_shared<controlled_solver>( controlled_solver{ ops, types::make_solver( ops ) } );
    step_type::params p;
    p.adaptive = true;
    step_type        step( ops, problem, control, solver, p );
    vector_wrap_type state( *ops ), out( *ops );
    state.start_use();
    out.start_use();
    ops->assign_scalar( 1., *state );
    for ( unsigned int stage : { 1u, 2u, 3u } )
    {
        solver->fail_on = solver->calls + stage;
        ops->assign_scalar( 42., *out );
        step.apply( *state, *out );
        require(
            step.get_status() == single_step_status::failed_solver && control->assessments == 0 &&
                ops->get_value_at_point( 0, *out ) == 42,
            "Failed BDF startup stage was assessed or published"
        );
    }
    solver->fail_on = 0;
    for ( unsigned int stage : { 1u, 2u, 3u } )
    {
        solver->invalid_on = solver->calls + stage;
        step.apply( *state, *out );
        require(
            step.get_status() == single_step_status::failed_nonfinite && control->assessments == 0 &&
                ops->get_value_at_point( 0, *out ) == 42,
            "Nonfinite BDF startup stage was assessed or published"
        );
    }
    solver->invalid_on = 0;
    auto before        = solver->calls;
    step.apply( *state, *out );
    const double first = ( 1 + .1 * .1 ) / 1.1;
    const double half  = ( 1 + .05 * .05 ) / 1.05;
    const double fine  = ( half + .05 * .1 ) / 1.05;
    require(
        step.get_status() == single_step_status::converged && solver->calls == before + 3 && step.get_order() == 1 &&
            std::abs( control->error - 2 * ( first - fine ) ) < 5e-10 &&
            std::abs( ops->get_value_at_point( 0, *out ) - first ) < 2e-10,
        "BDF startup did not estimate or commit the full backward-Euler candidate"
    );
    require(
        problem->time == .1 && problem->scaling().mass == 1 && problem->scaling().rhs == .1,
        "BDF startup left its half-step operator prepared"
    );
    step.finalize( adaptation_status::accepted, .1, *out );
    ops->assign( *out, *state );
    step.set_time( .1 );
    before = solver->calls;
    step.apply( *state, *out );
    const double second = ( first + .1 * .2 ) / 1.1;
    require(
        step.get_order() == 1 && solver->calls == before + 1 && control->error != 0,
        "BDF invented a third difference or repeated startup unnecessarily"
    );
    step.finalize( adaptation_status::accepted, .2, *out );
    ops->assign( *out, *state );
    step.set_time( .2 );

    for ( const double h : { .15, .075, .15 } )
    {
        control->next = h;
        const auto c  = bdf::make_coefficients( 2, h, .1 );
        // Solve alpha*(u-w)=t-u directly for the reference, not by reusing history.
        const double w      = second + c.history_weight * ( second - first );
        const double direct = ( c.alpha * w + .2 + h ) / ( c.alpha + 1 );
        step.apply( *state, *out );
        require(
            step.get_status() == single_step_status::converged && step.get_order() == 2 && control->exponent == 3 &&
                std::abs( ops->get_value_at_point( 0, *out ) - direct ) < 2e-10,
            "Variable-step BDF2 or rejected-commit history mismatch"
        );
        step.finalize( adaptation_status::failed, .2, *state );
    }
    control->next   = .1;
    control->reject = true;
    step.apply( *state, *out );
    require(
        step.get_status() == single_step_status::converged && step.get_attempts() == 2 && step.get_order() == 2 &&
            step.get_dt() == .05,
        "BDF error rejection discarded usable history"
    );
    step.finalize( adaptation_status::failed, .2, *state );
    control->next = .1;
    before        = solver->calls;
    step.apply( *state, *state );
    const double third = ( 2 * second - .5 * first + .1 * .3 ) / 1.6;
    require(
        solver->calls == before + 1 && std::abs( ops->get_value_at_point( 0, *state ) - third ) < 2e-10,
        "In-place BDF2 did not retain its pending input"
    );
    step.finalize( adaptation_status::accepted, .2 + step.get_dt(), *state );
    double time = .2 + step.get_dt();
    step.set_time( time );
    step.apply( *state, *state );
    const double fourth = ( 2 * third - .5 * second + .1 * ( time + .1 ) ) / 1.6;
    require(
        std::abs( ops->get_value_at_point( 0, *state ) - fourth ) < 2e-10,
        "BDF promoted the candidate instead of its input as history"
    );
    // Commit an externally reconstructed, shortened endpoint, then restart.
    time += .025;
    ops->assign_scalar( 2., *state );
    step.finalize( adaptation_status::accepted_modified, time, *state );
    step.set_time( time );
    before = solver->calls;
    step.apply( *state, *out );
    require(
        step.get_order() == 1 && solver->calls == before + 3, "Modified BDF endpoint did not restart its estimator"
    );
    step.finalize( adaptation_status::accepted, time + step.get_dt(), *out );
    time += step.get_dt();
    ops->assign( *out, *state );
    step.set_time( time );
    control->next = 10;
    step.apply( *state, *out );
    require( step.get_dt() <= .2 && step.get_order() == 1, "BDF did not bound growth or startup order" );
    step.finalize( adaptation_status::failed, time, *state );
    control->next = .001;
    before        = solver->calls;
    step.apply( *state, *out );
    require( step.get_order() == 1 && solver->calls == before + 3, "Extreme BDF shrinkage reused unreliable history" );
    step.finalize( adaptation_status::failed, time, *state );
    control->next = .1;
    step.set_target_time( time - 1 );
    before = solver->calls;
    step.apply( *state, *out );
    require(
        step.get_order() == 1 && solver->calls == before + 3 && step.get_dt() < 0,
        "BDF direction reversal did not restart"
    );
    step.finalize( adaptation_status::failed, time, *state );
    step.reset();
    require(
        step.get_status() == single_step_status::idle && solver->bindings == 1,
        "BDF reset changed solver binding or retained status"
    );
}

void test_automatic_order()
{
    auto ops = std::make_shared<operations_type>( 1 );
    for ( unsigned int cap = 1; cap <= 5; ++cap )
    {
        auto problem  = std::make_shared<problem_type>( ops );
        auto control  = std::make_shared<prescribed_controller>( prescribed_controller{ ops } );
        control->next = .01;
        auto solver   = std::make_shared<controlled_solver>( controlled_solver{ ops, types::make_solver( ops ) } );
        step_type::params p{ "BDF" };
        p.adaptive      = true;
        p.maximum_order = cap;
        if ( cap == 1 )
        {
            for ( unsigned int invalid : { 0u, 6u } )
            {
                p.maximum_order = invalid;
                bool failed     = false;
                try
                {
                    step_type bad( ops, problem, control, solver, p );
                }
                catch ( const std::invalid_argument & )
                {
                    failed = true;
                }
                require( failed, "Automatic BDF accepted an invalid order cap" );
            }
            p.maximum_order = cap;
            p.adaptive      = false;
            bool failed     = false;
            try
            {
                step_type bad( ops, problem, control, solver, p );
            }
            catch ( const std::invalid_argument & )
            {
                failed = true;
            }
            require( failed, "Automatic BDF accepted uncontrolled startup" );
            p.adaptive = true;
        }
        step_type        step( ops, problem, control, solver, p );
        vector_wrap_type state( *ops ), out( *ops );
        state.start_use();
        out.start_use();
        ops->assign_scalar( 1., *state );
        double time = 0;
        for ( unsigned int i = 0; i < 36; ++i )
        {
            step.set_time( time );
            step.apply( *state, *out );
            require(
                step.get_status() == single_step_status::converged && step.get_order() <= cap,
                "Automatic BDF exceeded its order cap or failed"
            );
            if ( i % 7 == 0 )
            {
                const auto order = step.get_order();
                const auto value = ops->get_value_at_point( 0, *out );
                step.finalize( adaptation_status::failed, time, *state );
                step.apply( *state, *out );
                require(
                    step.get_order() == order && std::abs( ops->get_value_at_point( 0, *out ) - value ) < 2e-12,
                    "Failed commit promoted BDF order or changed history"
                );
            }
            time += step.get_dt();
            step.finalize( adaptation_status::accepted, time, *out );
            ops->assign( *out, *state );
        }
        const auto expected = ( 1u << ( cap + 1 ) ) - 2;
        require(
            ( control->orders_seen & expected ) == expected && step.get_order() == cap,
            "Automatic BDF did not visit every permitted order"
        );
        if ( cap == 5 )
        {
            step.set_time( time );
            control->numerical_retry = true;
            solver->failures_left    = 2;
            step.apply( *state, *out );
            require(
                step.get_status() == single_step_status::converged && step.get_attempts() == 3 && step.get_order() == 4,
                "Repeated Newton failure did not lower automatic BDF order"
            );
            step.finalize( adaptation_status::failed, time, *state );
            solver->failures_left = 4;
            step.apply( *state, *out );
            require(
                step.get_status() == single_step_status::converged && step.get_attempts() == 5 && step.get_order() == 1,
                "Repeated failures did not fall back to BDF1"
            );
            step.finalize( adaptation_status::failed, time, *state );
            control->numerical_retry = false;
            step.apply( *state, *state );
            require( step.get_order() == 5, "Failed commit lost the committed high-order policy" );
            time += .0025;
            ops->assign_scalar( 2., *state );
            step.finalize( adaptation_status::accepted_modified, time, *state );
            step.set_time( time );
            const auto before = solver->calls;
            step.apply( *state, *out );
            require(
                step.get_order() == 1 && solver->calls == before + 3,
                "Modified high-order endpoint retained history or skipped controlled startup"
            );
            step.finalize( adaptation_status::failed, time, *state );
        }
    }
}

struct counting_operations : operations_type
{
    using operations_type::operations_type;
    mutable unsigned int allocations = 0, live = 0, fail_activation = 0;
    void                 start_use_vector( vector_type &x ) const override
    {
        const bool allocate = x.is_free();
        operations_type::start_use_vector( x );
        if ( allocate )
        {
            ++allocations;
            ++live;
            if ( allocations == fail_activation )
                throw std::runtime_error( "Injected adaptive BDF allocation failure" );
        }
    }
    void free_vector( vector_type &x ) const override
    {
        if ( !x.is_free() )
            --live;
        operations_type::free_vector( x );
    }
};

void test_storage()
{
    auto ops = std::make_shared<counting_operations>( 1 );
    {
        auto problem = std::make_shared<problem_type>( ops );
        auto control = std::make_shared<prescribed_controller>( prescribed_controller{ ops } );
        auto solver  = types::make_solver( ops );
        using counted_step =
            bdf::implicit_time_step<counting_operations, problem_type, prescribed_controller, types::solver_type>;
        typename counted_step::params p;
        p.adaptive = true;
        vector_wrap_type state( *ops );
        state.start_use();
        ops->assign_scalar( 1., *state );
        // Warm up Newton's lazy Krylov workspace before counting engine storage.
        {
            counted_step warmup( ops, problem, control, solver, p );
            warmup.apply( *state, *state );
            warmup.finalize( adaptation_status::accepted, warmup.get_dt(), *state );
        }
        const auto baseline = ops->live;
        p.method            = "BDF";
        for ( unsigned int failure = 1; failure <= 9; ++failure )
        {
            ops->fail_activation = ops->allocations + failure;
            bool failed          = false;
            try
            {
                counted_step invalid( ops, problem, control, solver, p );
            }
            catch ( const std::runtime_error & )
            {
                failed = true;
            }
            require( failed && ops->live == baseline, "Partial adaptive BDF construction leaked workspace" );
        }
        ops->fail_activation = 0;
        for ( const char *method : { "BDF1", "BDF2", "BDF" } )
        {
            p.method = method;
            counted_step step( ops, problem, control, solver, p );
            require(
                ops->live == baseline + ( p.method == "BDF"    ? 9
                                          : p.method == "BDF2" ? 5
                                                               : 4 ),
                "Unexpected adaptive BDF workspace budget"
            );
            const auto allocations = ops->allocations;
            double     time        = 0;
            for ( int i = 0; i < 5; ++i )
            {
                step.set_time( time );
                step.apply( *state, *state );
                require( step.get_status() == single_step_status::converged, "Counted adaptive BDF failed" );
                time += step.get_dt();
                step.finalize( adaptation_status::accepted, time, *state );
            }
            require( ops->allocations == allocations, "Adaptive BDF allocated during stepping" );
        }
        require( ops->live == baseline, "Adaptive BDF leaked its history" );
        std::weak_ptr<problem_type>       weak_problem = problem;
        std::weak_ptr<types::solver_type> weak_solver  = solver;
        auto                              step = std::make_shared<counted_step>( ops, problem, control, solver, p );
        problem.reset();
        solver.reset();
        control.reset();
        require( !weak_problem.expired() && !weak_solver.expired(), "Adaptive BDF lost dependency ownership" );
        step.reset();
        require( weak_problem.expired() && weak_solver.expired(), "Adaptive BDF ownership cycle" );
    }
    require( ops->live == 0, "Adaptive BDF assembly leaked vectors" );
}

int main()
try
{
    backend_type::init_device();
    test_startup_and_rollback();
    test_automatic_order();
    test_storage();
    std::cout
        << "Adaptive BDF startup failures, variable-grid rollback, in-place commits and estimator restart: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
