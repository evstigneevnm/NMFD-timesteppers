#include "implicit_bdf_test.h"
#include <limits>

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;
namespace model    = nmfd::time_steppers::tests::bdf_test;
using problem_type = model::problem_type;
using types        = newton_types<problem_type>;

// The scalar forced-decay problem distinguishes old, current and modified
// states. Injected failures must not turn any candidate into committed history.
struct controlled_solver
{
    std::shared_ptr<operations_type>    operations;
    std::shared_ptr<types::solver_type> solver;
    unsigned int                        bindings = 0, calls = 0, failures = 0;
    bool                                invalid = false, throws = false;
    void                                set_nonlinear_operator( std::shared_ptr<problem_type> problem )
    {
        ++bindings;
        solver->set_nonlinear_operator( problem );
    }
    bool solve( vector_type &state )
    {
        ++calls;
        if ( throws )
            throw std::invalid_argument( "Injected BDF configuration error" );
        if ( failures )
        {
            --failures;
            operations->assign_scalar( 99., state );
            return false;
        }
        const auto result = solver->solve( state );
        if ( invalid )
            operations->assign_scalar( std::numeric_limits<double>::infinity(), state );
        return result;
    }
};

struct controller : adaptation_type
{
    double            next     = .1;
    unsigned int      failures = 0, assessments = 0, commits = 0;
    bool              retry = false, reject = false, needs_error = false;
    adaptation_status last_commit = adaptation_status::failed;
    void              reset()
    {
        next = .1;
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
        if ( !retry )
            return adaptation_status::failed;
        next = std::abs( h ) / 2;
        return adaptation_status::rejected;
    }
    adaptation_status assess( double, double h, const vector_type &, const vector_type &, double &next_dt )
    {
        ++assessments;
        if ( reject )
        {
            reject = false;
            next   = std::abs( h ) / 2;
            return adaptation_status::rejected;
        }
        next_dt = next;
        return adaptation_status::accepted;
    }
    void update( adaptation_status outcome, double, double, const vector_type & )
    {
        last_commit = outcome;
        if ( outcome == adaptation_status::accepted || outcome == adaptation_status::accepted_modified )
            ++commits;
    }
};

using step_type = bdf::implicit_time_step<operations_type, problem_type, controller, controlled_solver>;

void test_transaction()
{
    auto      ops        = std::make_shared<operations_type>( 1 );
    auto      problem    = std::make_shared<problem_type>( ops );
    auto      solver     = std::make_shared<controlled_solver>( controlled_solver{ ops, types::make_solver( ops ) } );
    auto      adaptation = std::make_shared<controller>();
    step_type step( ops, problem, adaptation, solver );
    vector_wrap_type state( *ops ), out( *ops );
    state.start_use();
    out.start_use();
    ops->assign_scalar( 1., *state );
    require( step.get_status() == single_step_status::idle, "BDF construction is not idle" );
    step.apply( *state, *out );
    const double first = ( 1 + .1 * .1 ) / 1.1;
    require( step.get_order() == 1 && adaptation->commits == 0, "BDF startup committed before finalize" );
    for ( int operation = 0; operation < 3; ++operation )
    {
        bool rejected = false;
        try
        {
            if ( operation == 0 )
                step.apply( *state, *out );
            if ( operation == 1 )
                step.set_time( .1 );
            if ( operation == 2 )
                step.set_target_time( 2 );
        }
        catch ( const std::logic_error & )
        {
            rejected = true;
        }
        require( rejected, "BDF overwrote a pending candidate" );
    }
    step.finalize( adaptation_status::accepted, .1, *out );
    ops->assign( *out, *state );
    step.set_time( .1 );
    ops->assign_scalar( 42., *out );
    solver->failures       = 1;
    const auto assessments = adaptation->assessments;
    step.apply( *state, *out );
    require(
        step.get_status() == single_step_status::failed_solver && step.get_order() == 2 &&
            ops->get_value_at_point( 0, *out ) == 42 && adaptation->assessments == assessments,
        "Failed BDF solve published or assessed a candidate"
    );
    const double second = ( 2 * first - .5 + .1 * .2 ) / 1.6;
    for ( const auto outcome : { adaptation_status::rejected, adaptation_status::failed, adaptation_status::accepted } )
    {
        step.apply( *state, *out );
        require(
            step.get_order() == 2 && std::abs( ops->get_value_at_point( 0, *out ) - second ) < 2e-10,
            "BDF rejected/failed commit advanced its history"
        );
        step.finalize(
            outcome, outcome == adaptation_status::accepted ? .2 : .1,
            outcome == adaptation_status::accepted ? *out : *state
        );
    }
    require( adaptation->commits == 2 && solver->bindings == 1, "BDF commit or binding count mismatch" );
    step.set_time( .2 );
    ops->assign( *out, *state );
    solver->invalid = true;
    ops->assign_scalar( 42., *out );
    step.apply( *state, *out );
    require(
        step.get_status() == single_step_status::failed_nonfinite && ops->get_value_at_point( 0, *out ) == 42,
        "BDF published a nonfinite solve"
    );
    solver->invalid = false;
    solver->throws  = true;
    bool propagated = false;
    try
    {
        step.apply( *state, *out );
    }
    catch ( const std::invalid_argument & )
    {
        propagated = true;
    }
    require( propagated && ops->get_value_at_point( 0, *out ) == 42, "BDF swallowed a configuration exception" );
    solver->throws          = false;
    adaptation->needs_error = true;
    const auto calls        = solver->calls;
    step.apply( *state, *out );
    require(
        step.get_status() == single_step_status::error_estimate_unavailable && solver->calls == calls,
        "BDF invented an error estimator"
    );
}

void test_restart_rules()
{
    auto      ops        = std::make_shared<operations_type>( 1 );
    auto      problem    = std::make_shared<problem_type>( ops );
    auto      solver     = std::make_shared<controlled_solver>( controlled_solver{ ops, types::make_solver( ops ) } );
    auto      adaptation = std::make_shared<controller>();
    step_type step( ops, problem, adaptation, solver );
    vector_wrap_type state( *ops );
    state.start_use();
    ops->assign_scalar( 1., *state );
    double time = 0, older = 0;
    auto   advance = [&]( double target, unsigned int order, adaptation_status outcome = adaptation_status::accepted ) {
        const auto current = ops->get_value_at_point( 0, *state );
        step.set_time( time );
        step.set_target_time( target );
        step.apply( *state, *state );
        const auto   h     = step.get_dt();
        const double alpha = ( order == 2 ? 1.5 : 1. ) / h;
        const double w     = order == 2 ? ( 4 * current - older ) / 3 : current;
        require(
            step.get_status() == single_step_status::converged && step.get_order() == order &&
                std::abs( ops->get_value_at_point( 0, *state ) - ( alpha * w + time + h ) / ( alpha + 1 ) ) < 2e-10,
            "BDF in-place history or restart formula mismatch"
        );
        older = current;
        time += h;
        if ( outcome == adaptation_status::accepted_modified )
            ops->assign_scalar( 2., *state );
        step.finalize( outcome, time, *state );
    };
    advance( 1, 1 );
    advance( 1, 2 );
    advance( 1, 2, adaptation_status::accepted_modified );
    advance( 1, 1 );
    advance( 1, 2 );
    // A shortened external commit is treated as modified even if the caller
    // forgets the label. Its state is externally reconstructed and consistent.
    step.set_time( time );
    step.apply( *state, *state );
    time += .025;
    ops->assign_scalar( 2., *state );
    step.finalize( adaptation_status::accepted, time, *state );
    require( adaptation->last_commit == adaptation_status::accepted_modified, "BDF missed a shortened commit" );
    advance( 1, 1 );
    advance( 1, 2 );
    // Step clipping uses BDF1, then a new equal-step pair permits BDF2 again.
    advance( time + .04, 1 );
    adaptation->next = .04;
    advance( 2, 2 );
    adaptation->next = .08;
    advance( 2, 1 );
    advance( 2, 2 );
    // Jumping time resets the controller; reversing direction resets BDF order.
    time = 3;
    advance( 4, 1 );
    advance( 2, 1 );
    advance( 2, 2 );
    step.reset();
    require(
        step.get_status() == single_step_status::idle && step.get_attempts() == 0,
        "BDF reset retained trajectory status"
    );
    advance( 4, 1 );
}

void test_retries()
{
    auto      ops        = std::make_shared<operations_type>( 1 );
    auto      problem    = std::make_shared<problem_type>( ops );
    auto      solver     = std::make_shared<controlled_solver>( controlled_solver{ ops, types::make_solver( ops ) } );
    auto      adaptation = std::make_shared<controller>();
    step_type step( ops, problem, adaptation, solver, { "BDF2", 3 } );
    vector_wrap_type state( *ops ), out( *ops );
    state.start_use();
    out.start_use();
    ops->assign_scalar( 1., *state );
    step.apply( *state, *out );
    step.finalize( adaptation_status::accepted, .1, *out );
    ops->assign( *out, *state );
    step.set_time( .1 );
    adaptation->retry = true;
    solver->failures  = 1;
    step.apply( *state, *out );
    const auto u = ops->get_value_at_point( 0, *state );
    require(
        step.get_status() == single_step_status::converged && step.get_attempts() == 2 && step.get_order() == 1 &&
            step.get_dt() == .05 && std::abs( ops->get_value_at_point( 0, *out ) - ( u + .05 * .15 ) / 1.05 ) < 2e-10,
        "BDF retry used constant-step coefficients after changing h"
    );
    step.finalize( adaptation_status::failed, .1, *state );
    adaptation->next   = .1;
    adaptation->reject = true;
    step.apply( *state, *out );
    require(
        step.get_status() == single_step_status::converged && step.get_attempts() == 2 && step.get_order() == 1,
        "BDF controller rejection was not transactional"
    );
    step.finalize( adaptation_status::failed, .1, *state );
    solver->failures = 3;
    ops->assign_scalar( 42., *out );
    step.apply( *state, *out );
    require(
        step.get_status() == single_step_status::attempt_limit_reached && ops->get_value_at_point( 0, *out ) == 42,
        "BDF attempt budget corrupted output"
    );
}

// Count only timestepper workspace; the actual Newton solver has its own RAII.
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
                throw std::runtime_error( "Injected BDF allocation failure" );
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
        auto solver  = types::make_solver( ops );
        using counted_step =
            bdf::implicit_time_step<counting_operations, problem_type, adaptation_type, types::solver_type>;
        auto adaptation = std::make_shared<adaptation_type>();
        {
            counted_step     warmup( ops, problem, adaptation, solver, { "BDF1" } );
            vector_wrap_type state( *ops );
            state.start_use();
            ops->assign_scalar( 1., *state );
            warmup.apply( *state, *state );
            require( warmup.get_status() == single_step_status::converged, "BDF storage warmup failed" );
            warmup.finalize( adaptation_status::accepted, warmup.get_dt(), *state );
        }
        const auto baseline = ops->live;
        for ( unsigned int failure = 1; failure <= 3; ++failure )
        {
            ops->fail_activation = ops->allocations + failure;
            bool failed          = false;
            try
            {
                counted_step invalid( ops, problem, adaptation, solver );
            }
            catch ( const std::runtime_error & )
            {
                failed = true;
            }
            require( failed && ops->live == baseline, "Partial BDF construction leaked a vector" );
        }
        ops->fail_activation = 0;
        for ( const char *method : { "BDF1", "BDF2" } )
        {
            counted_step step( ops, problem, adaptation, solver, { method } );
            require(
                ops->live == baseline + ( std::string( method ) == "BDF2" ? 3 : 1 ), "BDF retained unexpected workspace"
            );
            vector_wrap_type state( *ops );
            state.start_use();
            ops->assign_scalar( 1., *state );
            step.apply( *state, *state );
            step.finalize( adaptation_status::accepted, step.get_dt(), *state );
            const auto allocations = ops->allocations;
            for ( int i = 0; i < 3; ++i )
            {
                step.set_time( ( i + 1 ) * .01 );
                step.apply( *state, *state );
                step.finalize( adaptation_status::accepted, ( i + 2 ) * .01, *state );
            }
            require( ops->allocations == allocations, "BDF allocated inside stepping" );
        }
        require( ops->live == baseline, "BDF destruction leaked workspace" );
        std::weak_ptr<problem_type>       weak_problem = problem;
        std::weak_ptr<types::solver_type> weak_solver  = solver;
        auto                              step = std::make_shared<counted_step>( ops, problem, adaptation, solver );
        problem.reset();
        solver.reset();
        adaptation.reset();
        require( !weak_problem.expired() && !weak_solver.expired(), "BDF did not retain its dependencies" );
        step.reset();
        require( weak_problem.expired() && weak_solver.expired(), "BDF dependency ownership cycle" );
    }
    require( ops->live == 0, "BDF assembly leaked vectors" );
}

struct section_manager
{
    std::shared_ptr<operations_type> ops;
    std::shared_ptr<step_type>       step;
    double                          *end     = nullptr;
    unsigned int                     calls   = 0;
    bool                             invalid = false;
    void                             set_time_interval( const double &start, double &next )
    {
        end = &next;
        if ( start == 0 )
            calls = 0;
    }
    bool apply( const vector_type &, vector_type &out )
    {
        require( step->get_order() == ( calls < 2 ? 1u : 2u ), "External shortening did not restart BDF" );
        if ( invalid )
            ops->assign_scalar( std::numeric_limits<double>::quiet_NaN(), out );
        else if ( calls++ == 0 )
        {
            *end = .035;
            ops->assign_scalar( model::forced_decay::exact( *end ), out );
            return true;
        }
        return false;
    }
    integration_status get_status() const
    {
        return integration_status::running;
    }
};

void test_external_manager()
{
    auto ops     = std::make_shared<operations_type>( 1 );
    auto problem = std::make_shared<problem_type>( ops );
    auto solver  = std::make_shared<controlled_solver>( controlled_solver{ ops, types::make_solver( ops ) } );
    auto step    = std::make_shared<step_type>( ops, problem, std::make_shared<controller>(), solver );
    auto manager = std::make_shared<section_manager>( section_manager{ ops, step } );
    integration::time_integrator<operations_type, step_type, section_manager> flow( ops, step, {}, manager );
    flow.set_time_interval( 0, .235 );
    vector_wrap_type initial( *ops ), out( *ops );
    initial.start_use();
    out.start_use();
    ops->assign_scalar( 1., *initial );
    for ( int run = 0; run < 2; ++run )
    {
        flow.apply( *initial, *out );
        require(
            flow.get_status() == integration_status::completed && manager->calls == 3 &&
                std::abs( ops->get_value_at_point( 0, *out ) - model::forced_decay::exact( .235 ) ) < .02,
            "BDF external-manager integration or repeat failed"
        );
    }
    manager->invalid = true;
    flow.apply( *initial, *out );
    require(
        flow.get_status() == integration_status::external_operation_failure && ops->get_value_at_point( 0, *out ) == 1,
        "BDF integrator committed a failed external endpoint"
    );
}

void test_validation()
{
    auto ops        = std::make_shared<operations_type>( 1 );
    auto problem    = std::make_shared<problem_type>( ops );
    auto solver     = std::make_shared<controlled_solver>( controlled_solver{ ops, types::make_solver( ops ) } );
    auto adaptation = std::make_shared<controller>();
    for ( const step_type::params p :
          { step_type::params{ "BDF3" }, step_type::params{ "BDF2", 0 }, step_type::params{ "BDF2", 1, 0. } } )
    {
        bool rejected = false;
        try
        {
            step_type invalid( ops, problem, adaptation, solver, p );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        require( rejected, "BDF accepted invalid configuration" );
    }
    for ( int missing = 0; missing < 4; ++missing )
    {
        bool rejected = false;
        try
        {
            step_type invalid(
                missing == 0 ? nullptr : ops, missing == 1 ? nullptr : problem, missing == 2 ? nullptr : adaptation,
                missing == 3 ? nullptr : solver
            );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        require( rejected, "BDF accepted a missing dependency" );
    }
    step_type        step( ops, problem, adaptation, solver );
    vector_wrap_type state( *ops ), out( *ops );
    state.start_use();
    out.start_use();
    ops->assign_scalar( 1., *state );
    ops->assign_scalar( 42., *out );
    for ( const double invalid :
          { -1., 0., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() } )
    {
        adaptation->next = invalid;
        step.apply( *state, *out );
        require(
            step.get_status() == single_step_status::step_size_underflow && ops->get_value_at_point( 0, *out ) == 42,
            "Invalid BDF step changed output"
        );
    }
    ops->assign_scalar( std::numeric_limits<double>::quiet_NaN(), *state );
    step.apply( *state, *out );
    require( step.get_status() == single_step_status::failed_nonfinite, "BDF accepted invalid initial data" );
    for ( const double invalid : { std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() } )
    {
        bool rejected = false;
        try
        {
            step.set_time( invalid );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        require( rejected, "BDF accepted nonfinite time" );
        rejected = false;
        try
        {
            step.set_target_time( invalid );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        require( rejected, "BDF accepted nonfinite target" );
    }
}

struct refusing_linear_solver
{
    void set_operator( std::shared_ptr<const problem_type::jacobi_operator_type> )
    {
    }
    bool solve( const vector_type &, vector_type & )
    {
        return false;
    }
};

void test_linear_failure()
{
    using iteration   = nmfd::solvers::newton_iteration<operations_type, problem_type, refusing_linear_solver>;
    using solver_type = nmfd::solvers::nonlinear_solver<operations_type, scfd::utils::log_std, problem_type, iteration>;
    using failed_step = bdf::implicit_time_step<operations_type, problem_type, adaptation_type, solver_type>;
    auto ops          = std::make_shared<operations_type>( 1 );
    auto problem      = std::make_shared<problem_type>( ops );
    auto solver       = std::make_shared<solver_type>(
        ops, nullptr, std::make_shared<iteration>( ops, std::make_shared<refusing_linear_solver>() )
    );
    failed_step      step( ops, problem, std::make_shared<adaptation_type>(), solver );
    vector_wrap_type state( *ops ), out( *ops );
    state.start_use();
    out.start_use();
    ops->assign_scalar( 1., *state );
    ops->assign_scalar( 42., *out );
    step.apply( *state, *out );
    require(
        step.get_status() == single_step_status::failed_solver &&
            solver->convergence_strategy()->get_result_status() == 6 && ops->get_value_at_point( 0, *out ) == 42,
        "Linear failure did not propagate through NMFD Newton to BDF"
    );
}

int main()
try
{
    backend_type::init_device();
    test_transaction();
    test_restart_rules();
    test_retries();
    test_storage();
    test_external_manager();
    test_validation();
    test_linear_failure();
    std::cout << "BDF committed history, in-place stepping, failures, retries, modified endpoints and RAII: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
