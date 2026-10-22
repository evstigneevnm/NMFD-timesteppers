#include "implicit_test_support.h"
#include <limits>

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;

// Constant rate: u'=1, u(0)=0; exact solution u=t.
// Nonstiff autonomous scalar ODE with identity mass, reproduced exactly in exact arithmetic.
// Isolates solver failures, retries, rollback, external endpoint changes, restart, and ownership.
struct constant_rate
{
    static constexpr int size = 1;
    __DEVICE_TAG__ void  residual(
         double, double a, const vector_type &x, const vector_type &previous, const vector_type &rhs, vector_type &out
     ) const
    {
        out( 0 ) = a * ( x( 0 ) - previous( 0 ) ) - 1 - rhs( 0 );
    }
    auto jacobian(
        double, double a, const scfd::static_vec::vec<double, 1> &, const scfd::static_vec::vec<double, 1> &
    ) const
    {
        return scfd::static_mat::mat<double, 1, 1>{ a };
    }
};

using problem_type = stage_problem<constant_rate>;
using types        = newton_types<problem_type>;

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

void test_linear_failure( std::shared_ptr<operations_type> operations, std::shared_ptr<problem_type> problem )
{
    using iteration_type = nmfd::solvers::newton_iteration<operations_type, problem_type, refusing_linear_solver>;
    using solver_type =
        nmfd::solvers::nonlinear_solver<operations_type, scfd::utils::log_std, problem_type, iteration_type>;
    using step_type     = runge_kutta::implicit_time_step<operations_type, problem_type, adaptation_type, solver_type>;
    auto      iteration = std::make_shared<iteration_type>( operations, std::make_shared<refusing_linear_solver>() );
    auto      solver    = std::make_shared<solver_type>( operations, nullptr, iteration );
    step_type step( operations, problem, std::make_shared<adaptation_type>(), solver );
    vector_wrap_type initial( *operations ), result( *operations );
    initial.start_use();
    result.start_use();
    operations->assign_scalar( 0., *initial );
    operations->assign_scalar( 42., *result );
    step.apply( *initial, *result );
    require(
        step.get_status() == single_step_status::failed_solver &&
            solver->convergence_strategy()->get_result_status() == 6 &&
            operations->get_value_at_point( 0, *result ) == 42,
        "Linear failure did not propagate through real Newton to DIRK"
    );
}

struct controlled_solver
{
    std::shared_ptr<operations_type>    operations;
    std::shared_ptr<types::solver_type> solver;
    unsigned int                        calls = 0, fail_on = 0, invalid_on = 0, bindings = 0;
    bool                                configuration_error = false;
    void                                set_nonlinear_operator( std::shared_ptr<problem_type> problem )
    {
        ++bindings;
        solver->set_nonlinear_operator( problem );
    }
    bool solve( vector_type &state )
    {
        ++calls;
        if ( configuration_error )
            throw std::invalid_argument( "Injected configuration error" );
        if ( calls == fail_on )
        {
            operations->assign_scalar( 99., state );
            return false;
        }
        const bool ok = solver->solve( state );
        if ( calls == invalid_on )
            operations->assign_scalar( std::numeric_limits<double>::infinity(), state );
        return ok;
    }
};

struct retry_adaptation : adaptation_type
{
    double       next     = .1;
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
    adaptation_status reject_step( double dt )
    {
        ++failures;
        next = std::abs( dt ) / 2;
        return adaptation_status::rejected;
    }
    adaptation_status assess( double, double dt, const vector_type &, const vector_type &, double &dt_next )
    {
        ++assessments;
        dt_next = next;
        if ( reject_candidate )
        {
            next = std::abs( dt ) / 2;
            return adaptation_status::rejected;
        }
        return adaptation_status::accepted;
    }
    void update( adaptation_status outcome, double, double, const vector_type & )
    {
        if ( outcome == adaptation_status::accepted || outcome == adaptation_status::accepted_modified )
            ++commits;
    }
};

struct section_manager
{
    std::shared_ptr<operations_type> operations;
    double                           start = 0, *end = nullptr;
    bool                             invalid = false, modified = false;
    void                             set_time_interval( const double &t, double &next )
    {
        start = t;
        end   = &next;
        if ( t == 0 )
            modified = false;
    }
    bool apply( const vector_type &in, vector_type &out )
    {
        if ( invalid )
            operations->assign_scalar( std::numeric_limits<double>::quiet_NaN(), out );
        else if ( !modified )
        {
            *end = .035;
            operations->assign( in, out );
            operations->add_mul_scalar( *end - start, 1., out );
            modified = true;
            return true;
        }
        return false;
    }
    integration_status get_status() const
    {
        return integration_status::running;
    }
};

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
                throw std::runtime_error( "Injected activation failure" );
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
    auto operations = std::make_shared<counting_operations>( 1 );
    {
        auto problem    = std::make_shared<problem_type>( operations );
        auto solver     = types::make_solver( operations );
        auto adaptation = std::make_shared<adaptation_type>( adaptation_type::params{ .1 } );
        using step_type =
            runge_kutta::implicit_time_step<counting_operations, problem_type, adaptation_type, types::solver_type>;
        const auto baseline         = operations->live;
        operations->fail_activation = operations->allocations + 2;
        bool failed                 = false;
        try
        {
            step_type invalid( operations, problem, adaptation, solver, step_type::params{ "SDIRK2(1)2" } );
        }
        catch ( const std::runtime_error & )
        {
            failed = true;
        }
        require( failed && operations->live == baseline, "DIRK partial construction leaked workspace" );
        operations->fail_activation = 0;
        auto step =
            std::make_shared<step_type>( operations, problem, adaptation, solver, step_type::params{ "SDIRK2(1)2" } );
        require( operations->live == baseline + 2, "Stiffly accurate DIRK retained an unnecessary endpoint rate" );
        vector_wrap_type initial( *operations ), result( *operations );
        initial.start_use();
        result.start_use();
        operations->assign_scalar( 0., *initial );
        integration::time_integrator<operations_type, step_type> integrator( operations, step );
        integrator.set_time_interval( 0, .3 );
        integrator.apply( *initial, *result );
        require( integrator.get_status() == integration_status::completed, "DIRK storage warmup failed" );
        const auto allocations = operations->allocations;
        integrator.apply( *initial, *result );
        require(
            integrator.get_status() == integration_status::completed && operations->allocations == allocations,
            "DIRK allocated trajectory vectors during integration"
        );
    }
    require( operations->live == 0, "DIRK/Newton workspace was not released" );
}

int main()
try
{
    backend_type::init_device();
    auto operations = std::make_shared<operations_type>( 1 );
    auto problem    = std::make_shared<problem_type>( operations );
    auto solver =
        std::make_shared<controlled_solver>( controlled_solver{ operations, types::make_solver( operations ) } );
    auto adaptation = std::make_shared<adaptation_type>( adaptation_type::params{ .1 } );
    using step_type =
        runge_kutta::implicit_time_step<operations_type, problem_type, adaptation_type, controlled_solver>;
    auto step =
        std::make_shared<step_type>( operations, problem, adaptation, solver, step_type::params{ "SDIRK2(1)2" } );
    vector_wrap_type initial( *operations ), result( *operations );
    initial.start_use();
    result.start_use();
    operations->assign_scalar( 0., *initial );
    operations->assign_scalar( 42., *result );
    require( step->get_status() == single_step_status::idle, "New DIRK must be idle" );
    step->set_time( 0 );
    step->set_target_time( .3 );
    solver->fail_on = 2;
    step->apply( *initial, *result );
    require(
        step->get_status() == single_step_status::failed_solver && solver->calls == 2 &&
            operations->get_value_at_point( 0, *result ) == 42 && operations->get_value_at_point( 0, *initial ) == 0,
        "Failed later stage overwrote caller state"
    );
    solver->fail_on = 0;
    step->apply( *initial, *result );
    require( step->get_status() == single_step_status::converged, "DIRK recovery failed" );
    bool pending_refused = false;
    try
    {
        step->set_time( .1 );
    }
    catch ( const std::logic_error & )
    {
        pending_refused = true;
    }
    require( pending_refused, "Pending DIRK allowed reconfiguration" );
    step->finalize( adaptation_status::accepted, .1, *result );
    require( step->get_status() == single_step_status::converged, "Finalization lost numerical status" );
    require( solver->bindings == 1, "DIRK rebound the nonlinear solver inside the numerical loop" );

    step->reset();
    require( step->get_status() == single_step_status::idle, "Reset DIRK must be idle" );
    solver->invalid_on = solver->calls + 1;
    operations->assign_scalar( 42., *result );
    step->apply( *initial, *result );
    require(
        step->get_status() == single_step_status::failed_nonfinite &&
            operations->get_value_at_point( 0, *result ) == 42,
        "Nonfinite stage was published"
    );
    solver->invalid_on          = 0;
    solver->configuration_error = true;
    bool propagated             = false;
    try
    {
        step->apply( *initial, *result );
    }
    catch ( const std::invalid_argument & )
    {
        propagated = true;
    }
    require(
        propagated && operations->get_value_at_point( 0, *result ) == 42,
        "DIRK swallowed a configuration exception or changed caller state"
    );
    solver->configuration_error = false;

    auto retry = std::make_shared<retry_adaptation>();
    using retry_step_type =
        runge_kutta::implicit_time_step<operations_type, problem_type, retry_adaptation, controlled_solver>;
    auto retry_step =
        std::make_shared<retry_step_type>( operations, problem, retry, solver, retry_step_type::params{ "IE", 3 } );
    solver->fail_on = solver->calls + 1;
    retry_step->set_time( 0 );
    retry_step->set_target_time( .3 );
    retry_step->apply( *initial, *result );
    require(
        retry_step->get_status() == single_step_status::converged && retry_step->get_attempts() == 2 &&
            retry->failures == 1 && retry->assessments == 1 && retry->commits == 0 && retry_step->get_dt() == .05,
        "Failed numerical attempt entered assessment or bypassed retry"
    );
    retry_step->finalize( adaptation_status::accepted, .05, *result );
    require( retry->commits == 1, "Acceptance was not recorded at commitment" );
    retry_step->reset();
    retry->reject_candidate = true;
    operations->assign_scalar( 42., *result );
    retry_step->apply( *initial, *result );
    require(
        retry_step->get_status() == single_step_status::attempt_limit_reached && retry_step->get_attempts() == 3 &&
            operations->get_value_at_point( 0, *result ) == 42,
        "DIRK rejection budget or rollback failed"
    );
    retry_step->reset();
    retry->needs_error = true;
    retry_step->apply( *initial, *result );
    require(
        retry_step->get_status() == single_step_status::error_estimate_unavailable,
        "Fixed DIRK silently supplied an embedded error estimate"
    );
    retry->needs_error      = false;
    retry->reject_candidate = false;
    solver->fail_on         = 0;

    auto manager = std::make_shared<section_manager>( section_manager{ operations } );
    integration::time_integrator<operations_type, step_type, section_manager> integrator(
        operations, step, {}, manager
    );
    integrator.set_time_interval( 0, .3 );
    for ( int run = 0; run < 2; ++run )
    {
        integrator.apply( *initial, *result );
        require(
            integrator.get_status() == integration_status::completed && integrator.get_final_time() == .3 &&
                std::abs( operations->get_value_at_point( 0, *result ) - .3 ) < 1e-12,
            "Modified DIRK endpoint or repeated trajectory failed"
        );
    }
    manager->invalid = true;
    integrator.apply( *initial, *result );
    require(
        integrator.get_status() == integration_status::external_operation_failure &&
            operations->get_value_at_point( 0, *result ) == 0 && integrator.get_final_time() == 0,
        "Invalid external endpoint replaced committed state"
    );
    manager->invalid = false;
    solver->fail_on  = solver->calls + 1;
    integrator.apply( *initial, *result );
    require(
        integrator.get_status() == integration_status::step_failure &&
            operations->get_value_at_point( 0, *result ) == 0,
        "Failed integration changed committed state"
    );
    solver->fail_on = 0;
    operations->assign_scalar( 0., *result );
    integrator.apply( *result, *result );
    require(
        integrator.get_status() == integration_status::completed &&
            std::abs( operations->get_value_at_point( 0, *result ) - .3 ) < 1e-12,
        "Aliased DIRK integration failed"
    );

    std::weak_ptr<problem_type>       weak_problem;
    std::weak_ptr<types::solver_type> weak_solver;
    {
        auto owned_problem = std::make_shared<problem_type>( operations );
        auto owned_solver  = types::make_solver( operations );
        weak_problem       = owned_problem;
        weak_solver        = owned_solver;
        auto owned_step    = std::make_shared<types::step_type>( operations, owned_problem, adaptation, owned_solver );
        owned_problem.reset();
        owned_solver.reset();
        require( !weak_problem.expired() && !weak_solver.expired(), "DIRK did not retain dependencies" );
    }
    require( weak_problem.expired() && weak_solver.expired(), "DIRK introduced an ownership cycle" );
    for ( int missing = 0; missing < 4; ++missing )
    {
        bool refused = false;
        try
        {
            auto invalid = std::make_shared<types::step_type>(
                missing == 0 ? nullptr : operations, missing == 1 ? nullptr : problem,
                missing == 2 ? nullptr : adaptation, missing == 3 ? nullptr : types::make_solver( operations )
            );
        }
        catch ( const std::invalid_argument & )
        {
            refused = true;
        }
        require( refused, "DIRK accepted a null retained dependency" );
    }
    test_storage();
    test_linear_failure( operations, problem );
    std::cout << "DIRK failures, bounded retries, commit/rollback, restart, external endpoints and ownership: PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
