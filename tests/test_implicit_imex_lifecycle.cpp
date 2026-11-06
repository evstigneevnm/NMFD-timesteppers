#include "implicit_imex_adaptive_test.h"
#include <limits>

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;

// Split u'=.25*u-1.25*u=-u. Failure injection tests the same candidate/commit
// boundary as real Newton, without changing the partition equations or tolerances.
struct checked_partition : rate_problem<imex_linear_equation>
{
    using base = rate_problem<imex_linear_equation>;
    std::shared_ptr<operations_type> ops;
    unsigned int                     calls = 0, fail_on = 0;
    int                              mode = 0; // 0: false; 1: nonfinite output; 2: backend/configuration exception.
    bool                             always_fail = false;
    checked_partition( std::shared_ptr<operations_type> o, double lambda )
        : base( o, imex_linear_equation{ lambda } ), ops( std::move( o ) )
    {
    }
    bool evaluate_rate( const vector_type &state, vector_type &out )
    {
        ++calls;
        if ( always_fail || calls == fail_on )
        {
            if ( mode == 2 )
                throw std::runtime_error( "Injected partition mass-solve error" );
            if ( mode == 0 )
                return false;
            ops->assign_scalar( std::numeric_limits<double>::quiet_NaN(), out );
            return true;
        }
        return base::evaluate_rate( state, out );
    }
};
using problem_type = imex_problem<checked_partition, checked_partition>;
using types        = newton_types<checked_partition>;

struct controlled_solver
{
    std::shared_ptr<operations_type>    ops;
    std::shared_ptr<types::solver_type> solver;
    unsigned int                        calls = 0, fail_on = 0, bindings = 0;
    int                                 mode          = 0;
    bool                                throw_binding = false;
    void                                set_nonlinear_operator( std::shared_ptr<checked_partition> problem )
    {
        if ( throw_binding )
            throw std::runtime_error( "Injected solver binding error" );
        ++bindings;
        solver->set_nonlinear_operator( std::move( problem ) );
    }
    bool solve( vector_type &state )
    {
        ++calls;
        if ( calls == fail_on )
        {
            if ( mode == 2 )
                throw std::runtime_error( "Injected backend error" );
            ops->assign_scalar( mode == 1 ? std::numeric_limits<double>::infinity() : 99., state );
            return mode == 1;
        }
        return solver->solve( state );
    }
};

struct retry_controller : adaptation_type
{
    double       next     = .1;
    unsigned int failures = 0, assessments = 0, commits = 0;
    bool         needs_error = false, reject_candidates = false;
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
        if ( reject_candidates )
        {
            next = std::abs( dt ) / 2;
            return adaptation_status::rejected;
        }
        return adaptation_status::accepted;
    }
    void update( adaptation_status status, double, double, const vector_type & )
    {
        if ( status == adaptation_status::accepted || status == adaptation_status::accepted_modified )
            ++commits;
    }
};

auto make_problem( std::shared_ptr<operations_type> ops )
{
    return std::make_shared<problem_type>(
        std::make_shared<checked_partition>( ops, .25 ), std::make_shared<checked_partition>( ops, -1.25 )
    );
}

void test_failures()
{
    auto             ops = std::make_shared<operations_type>( 1 );
    vector_wrap_type initial( *ops ), out( *ops ), reference( *ops );
    initial.start_use();
    out.start_use();
    reference.start_use();
    ops->assign_scalar( 1., *initial );
    for ( int source = 0; source < 4; ++source )
        for ( int mode = 0; mode < 3; ++mode )
        {
            auto  problem = make_problem( ops );
            auto  solver  = std::make_shared<controlled_solver>( controlled_solver{ ops, types::make_solver( ops ) } );
            auto  control = std::make_shared<retry_controller>();
            auto &e       = *problem->partitions->explicit_part;
            auto &i       = *problem->partitions->implicit_part;
            e.mode = i.mode = solver->mode = mode;
            if ( source < 2 )
                e.fail_on = source + 1;
            if ( source == 2 )
                i.fail_on = 1;
            if ( source == 3 )
                solver->fail_on = 1;
            using step_type = imex_step<problem_type, retry_controller, controlled_solver>;
            step_type step( ops, problem, control, solver, step_type::params{ "IMEX_HEUN_TR2" } );
            ops->assign_scalar( 77., *out );
            bool threw = false;
            try
            {
                step.apply( *initial, *out );
            }
            catch ( const std::runtime_error & )
            {
                threw = true;
            }
            require( ops->get_value_at_point( 0, *initial ) == 1, "Failed IMEX stage modified its input" );
            if ( mode == 2 )
            {
                require(
                    threw && ops->get_value_at_point( 0, *out ) == 77 && control->assessments == 0,
                    "IMEX swallowed a backend exception or assessed its candidate"
                );
                continue;
            }
            require(
                !threw && step.get_status() == single_step_status::converged && step.get_attempts() == 2 &&
                    control->failures == 1 && control->assessments == 1 && control->commits == 0 &&
                    step.get_dt() == .05 && solver->bindings == 1 && problem->requests == 1,
                "IMEX rate/Newton failure did not reject and retry correctly"
            );
            auto fresh_problem = make_problem( ops );
            auto fresh         = make_imex_step( ops, fresh_problem, .05, "IMEX_HEUN_TR2" );
            fresh->apply( *initial, *reference );
            require(
                fresh->get_status() == single_step_status::converged &&
                    std::abs( ops->get_value_at_point( 0, *out ) - ops->get_value_at_point( 0, *reference ) ) < 1e-13,
                "IMEX retry used a failed stage or displacement"
            );
            step.finalize( adaptation_status::accepted, .05, *out );
            fresh->finalize( adaptation_status::accepted, .05, *reference );
            require( control->commits == 1, "IMEX committed before finalization" );
        }
    for ( const int mode : { 0, 1 } )
    {
        auto  problem = make_problem( ops );
        auto &e       = *problem->partitions->explicit_part;
        e.fail_on     = 1;
        e.mode        = mode;
        auto failed   = make_imex_step( ops, problem, .1, "IMEX_EULER" );
        ops->assign_scalar( 77., *out );
        failed->apply( *initial, *out );
        require(
            failed->get_status() ==
                    ( mode == 0 ? single_step_status::failed_solver : single_step_status::failed_nonfinite ) &&
                ops->get_value_at_point( 0, *out ) == 77,
            "Terminal IMEX rate failure published a candidate"
        );
        failed->apply( *initial, *initial );
        require(
            failed->get_status() == single_step_status::converged, "IMEX could not recover after terminal failure"
        );
        failed->finalize( adaptation_status::accepted, .1, *initial );
        ops->assign_scalar( 1., *initial );
    }
    auto problem                  = make_problem( ops );
    auto controller               = std::make_shared<retry_controller>();
    controller->reject_candidates = true;
    using step_type               = imex_step<problem_type, retry_controller>;
    step_type exhausted( ops, problem, controller, types::make_solver( ops ), step_type::params{ "IMEX_EULER", 2 } );
    ops->assign_scalar( 77., *out );
    exhausted.apply( *initial, *out );
    require(
        exhausted.get_status() == single_step_status::attempt_limit_reached && exhausted.get_attempts() == 2 &&
            controller->assessments == 2 && ops->get_value_at_point( 0, *out ) == 77,
        "IMEX escaped a bounded controller-rejection budget"
    );
    controller->reject_candidates                   = false;
    problem->partitions->explicit_part->always_fail = true;
    exhausted.reset();
    exhausted.apply( *initial, *out );
    require(
        exhausted.get_status() == single_step_status::attempt_limit_reached && controller->failures == 2 &&
            controller->assessments == 0 && ops->get_value_at_point( 0, *out ) == 77,
        "IMEX escaped a bounded numerical-failure budget"
    );
    controller->needs_error = true;
    const auto calls        = problem->partitions->explicit_part->calls;
    exhausted.apply( *initial, *out );
    require(
        exhausted.get_status() == single_step_status::error_estimate_unavailable && exhausted.get_attempts() == 0 &&
            problem->partitions->explicit_part->calls == calls && ops->get_value_at_point( 0, *out ) == 77,
        "IMEX invented an embedded estimate"
    );
}

struct autonomous_rate
{
    std::shared_ptr<operations_type> ops;
    unsigned int                     calls = 0;
    bool                             evaluate_rate( const vector_type &x, vector_type &out )
    {
        ++calls;
        ops->assign_lin_comb( .25, x, out );
        return true;
    }
};

void test_optional_interfaces()
{
    using implicit_type       = stage_problem<imex_linear_equation>; // Deliberately no evaluate_rate().
    using problem             = imex_problem<autonomous_rate, implicit_type>;
    auto             ops      = std::make_shared<operations_type>( 1 );
    auto             e        = std::make_shared<autonomous_rate>( autonomous_rate{ ops } );
    auto             i        = std::make_shared<implicit_type>( ops, imex_linear_equation{ -1.25 } );
    auto             assembly = std::make_shared<problem>( e, i );
    vector_wrap_type initial( *ops ), out( *ops );
    initial.start_use();
    out.start_use();
    ops->assign_scalar( 1., *initial );
    for ( const char *method : { "IMEX_EULER", "IMEX_ARS222" } )
    {
        auto step = make_imex_step( ops, assembly, .1, method );
        e->calls  = 0;
        step->apply( *initial, *out );
        require(
            step->get_status() == single_step_status::converged && step->dense_output_order() == 0 &&
                e->calls + 1 == step->table().explicit_table.size(),
            "IMEX required an unused rate or evaluated its unused endpoint"
        );
        step->finalize( adaptation_status::accepted, .1, *out );
    }
    bool rejected = false;
    try
    {
        auto invalid = make_imex_step( ops, assembly, .1, "IMEX_HEUN_TR2" );
    }
    catch ( const std::invalid_argument & )
    {
        rejected = true;
    }
    require( rejected, "IMEX accepted a consumed implicit rate without a provider" );
    using dense_type = runge_kutta::imex_time_step<
        operations_type, problem, adaptation_type, newton_types<implicit_type>::solver_type, true>;
    rejected = false;
    try
    {
        dense_type invalid(
            ops, assembly, std::make_shared<adaptation_type>(), newton_types<implicit_type>::make_solver( ops )
        );
    }
    catch ( const std::invalid_argument & )
    {
        rejected = true;
    }
    require( rejected, "IMEX reused an unvalidated partition interpolant" );
}

struct counting_operations : operations_type
{
    using operations_type::operations_type;
    mutable unsigned int allocations = 0, live = 0;
    void                 start_use_vector( vector_type &x ) const override
    {
        if ( x.is_free() )
        {
            ++allocations;
            ++live;
        }
        operations_type::start_use_vector( x );
    }
    void free_vector( vector_type &x ) const override
    {
        if ( !x.is_free() )
            --live;
        operations_type::free_vector( x );
    }
};

void test_ownership_and_lifecycle()
{
    auto ops              = std::make_shared<counting_operations>( 1 );
    auto problem          = make_problem( ops );
    auto bundle           = problem->partitions;
    auto solver           = std::make_shared<controlled_solver>( controlled_solver{ ops, types::make_solver( ops ) } );
    using step_type       = imex_step<problem_type, adaptation_type, controlled_solver>;
    const auto live       = ops->live;
    solver->throw_binding = true;
    bool rejected         = false;
    try
    {
        step_type invalid( ops, problem, std::make_shared<adaptation_type>(), solver );
    }
    catch ( const std::runtime_error & )
    {
        rejected = true;
    }
    require( rejected && ops->live == live, "Failed IMEX construction leaked backend workspace" );
    solver->throw_binding                                                = false;
    std::weak_ptr<problem_type>                     application_lifetime = problem;
    std::weak_ptr<problem_type::imex_operator_type> partition_lifetime   = bundle;
    auto                                            step                 = std::make_shared<step_type>(
        ops, problem, std::make_shared<adaptation_type>( adaptation_type::params{ .1 } ), solver
    );
    require( problem->requests == 2, "IMEX did not acquire its bundle once per construction" );
    problem.reset();
    bundle.reset();
    require(
        application_lifetime.expired() && !partition_lifetime.expired(),
        "IMEX retained a factory or lost its partitions"
    );
    vector_wrap_type initial( *ops ), out( *ops );
    initial.start_use();
    out.start_use();
    ops->assign_scalar( 1., *initial );
    step->apply( *initial, *out );
    require(
        step->get_status() == single_step_status::converged && solver->bindings == 1, "IMEX rebound Newton per stage"
    );
    rejected = false;
    try
    {
        step->apply( *initial, *out );
    }
    catch ( const std::logic_error & )
    {
        rejected = true;
    }
    require( rejected, "IMEX overwrote an unfinalized candidate" );
    rejected = false;
    try
    {
        step->set_time( .1 );
    }
    catch ( const std::logic_error & )
    {
        rejected = true;
    }
    require( rejected, "IMEX changed a pending step time" );
    rejected = false;
    try
    {
        step->finalize( adaptation_status::accepted, .2, *out );
    }
    catch ( const std::invalid_argument & )
    {
        rejected = true;
    }
    require( rejected, "IMEX accepted an endpoint outside its pending interval" );
    step->finalize( adaptation_status::accepted_modified, .04, *initial );
    const auto allocations = ops->allocations;
    for ( int n = 0; n < 3; ++n )
    {
        step->reset();
        step->set_time( .04 );
        step->apply( *initial, *out );
        require(
            step->get_status() == single_step_status::converged && ops->allocations == allocations,
            "IMEX reset or repeated steps allocated trajectory workspace"
        );
        step->finalize( adaptation_status::failed, .04, *initial );
    }
    step.reset();
    solver.reset();
    require( partition_lifetime.expired(), "IMEX ownership cycle retained partition data" );
}

struct modified_endpoint
{
    std::shared_ptr<operations_type> ops;
    double                          *end     = nullptr;
    bool                             changed = false;
    void                             set_time_interval( const double &, double &time )
    {
        end = &time;
    }
    bool apply( const vector_type &, vector_type &out )
    {
        if ( changed )
            return false;
        *end = .04;
        ops->assign_scalar( 2., out );
        changed = true;
        return true;
    }
    integration_status get_status() const
    {
        return integration_status::running;
    }
};

void test_modified_endpoint_and_backward()
{
    auto ops      = std::make_shared<operations_type>( 1 );
    auto problem  = make_problem( ops );
    auto step     = make_imex_step( ops, problem, .1, "IMEX_HEUN_TR2" );
    auto external = std::make_shared<modified_endpoint>( modified_endpoint{ ops } );
    integration::time_integrator<operations_type, imex_step<problem_type>, modified_endpoint> flow(
        ops, step, {}, external
    );
    vector_wrap_type initial( *ops ), out( *ops ), reference( *ops );
    initial.start_use();
    out.start_use();
    reference.start_use();
    ops->assign_scalar( 1., *initial );
    flow.set_time_interval( 0, .3 );
    flow.apply( *initial, *out );
    require( flow.get_status() == integration_status::completed, "Modified IMEX endpoint failed" );
    auto fresh = make_imex_step( ops, problem, .1, "IMEX_HEUN_TR2" );
    integration::time_integrator<operations_type, imex_step<problem_type>> verify( ops, fresh );
    verify.set_time_interval( .04, .3 );
    ops->assign_scalar( 2., *initial );
    verify.apply( *initial, *reference );
    require(
        verify.get_status() == integration_status::completed &&
            std::abs( ops->get_value_at_point( 0, *out ) - ops->get_value_at_point( 0, *reference ) ) < 1e-13,
        "Modified IMEX endpoint reused an obsolete rate"
    );
    for ( const double reference_time : { .25, 1., 4. } )
        for ( const double h : { -.1, 1e-6, .1, 2. } )
        {
            using step_type = imex_step<problem_type>;
            step_type backward(
                ops, problem, std::make_shared<adaptation_type>( adaptation_type::params{ std::abs( h ) } ),
                types::make_solver( ops ), step_type::params{ "IMEX_EULER", 32, reference_time }
            );
            ops->assign_scalar( 1., *initial );
            backward.set_time( .3 );
            backward.set_target_time( .3 + h );
            backward.apply( *initial, *out );
            const double dt = backward.get_dt();
            require(
                backward.get_status() == single_step_status::converged &&
                    std::abs( ops->get_value_at_point( 0, *out ) - ( 1 + .25 * dt ) / ( 1 + 1.25 * dt ) ) < 1e-10,
                "IMEX backward/reference-time scaling changed its stage equation"
            );
            backward.finalize( adaptation_status::accepted, .3 + dt, *out );
        }
}

// A successful numerical retry must rebuild BOTH rate arrays and the error vector.
void test_adaptive_recovery()
{
    using controller     = observed_controller<integration::time_step_adaptation_scipy<operations_type>>;
    using step_type      = imex_step<problem_type, controller, controlled_solver>;
    auto             ops = std::make_shared<operations_type>( 1 );
    vector_wrap_type initial( *ops ), out( *ops ), reference( *ops );
    initial.start_use();
    out.start_use();
    reference.start_use();
    ops->assign_scalar( 1., *initial );
    for ( int source = 0; source < 4; ++source )
        for ( int mode = 0; mode < 2; ++mode )
        {
            auto  problem = make_problem( ops );
            auto  solver  = std::make_shared<controlled_solver>( controlled_solver{ ops, types::make_solver( ops ) } );
            auto &e       = *problem->partitions->explicit_part;
            auto &i       = *problem->partitions->implicit_part;
            e.mode = i.mode = solver->mode = mode;
            if ( source < 2 )
                e.fail_on = source + 1;
            if ( source == 2 )
                i.fail_on = 1;
            if ( source == 3 )
                solver->fail_on = 1;
            auto p         = controller_parameters<controller>( 1e-5 );
            p.initial_step = p.maximum_step = .3;
            auto control                    = std::make_shared<controller>( ops, p );
            control->expected_order         = 3;
            step_type step( ops, problem, control, solver, step_type::params{ "IMEX_ARK324L2SA" } );
            step.apply( *initial, *out );
            require(
                step.get_status() == single_step_status::converged && control->numerical_rejections == 1 &&
                    step.get_attempts() == control->error_rejections + 2 &&
                    control->assessments == control->error_rejections + 1 && control->commits == 0,
                "Adaptive IMEX assessed a failed stage or lost bounded recovery"
            );
            auto fresh = make_imex_step( ops, make_problem( ops ), step.get_dt(), "IMEX_ARK324L2SA" );
            fresh->apply( *initial, *reference );
            require(
                fresh->get_status() == single_step_status::converged &&
                    std::abs( ops->get_value_at_point( 0, *out ) - ops->get_value_at_point( 0, *reference ) ) < 1e-12,
                "Adaptive IMEX recovery retained a failed candidate or derivative"
            );
            step.finalize( adaptation_status::accepted, step.get_dt(), *out );
            fresh->finalize( adaptation_status::accepted, fresh->get_dt(), *reference );
        }
}

void test_embedding_capabilities()
{
    const auto g = 1.L - std::sqrt( 2.L ) / 2, k = 1 - 1 / ( 2 * g );
    // The ARS222 primary ignores the initial implicit rate and final explicit
    // rate. A first-order embedding consumes both, so adaptation must retain them.
    const runge_kutta::imex_butcher_table pair(
        { { { 0, 0, 0 }, { g, 0, 0 }, { k, 1 - k, 0 } }, { k, 1 - k, 0 }, 2, {}, { .5L, 0, .5L }, 1 },
        { { { 0, 0, 0 }, { 0, g, 0 }, { 0, 1 - g, g } }, { 0, 1 - g, g }, 2, {}, { .5L, 0, .5L }, 1 }, 2, 1
    );
    using no_rate         = stage_problem<imex_linear_equation>;
    using no_rate_problem = imex_problem<autonomous_rate, no_rate>;
    using no_rate_step    = imex_step<no_rate_problem, imex_error_capture>;
    auto ops              = std::make_shared<operations_type>( 1 );
    auto problem          = std::make_shared<no_rate_problem>(
        std::make_shared<autonomous_rate>( autonomous_rate{ ops } ),
        std::make_shared<no_rate>( ops, imex_linear_equation{ -1.25 } )
    );
    vector_wrap_type initial( *ops ), out( *ops );
    initial.start_use();
    out.start_use();
    ops->assign_scalar( 1., *initial );
    auto control     = std::make_shared<imex_error_capture>( ops, .1 );
    control->enabled = false;
    no_rate_step fixed( ops, problem, control, newton_types<no_rate>::make_solver( ops ), pair );
    fixed.apply( *initial, *out );
    require(
        fixed.get_status() == single_step_status::converged && problem->partitions->explicit_part->calls == 2,
        "Fixed IMEX allocated/evaluated embedding-only rates"
    );
    fixed.finalize( adaptation_status::accepted, .1, *out );
    control->enabled = true;
    bool rejected    = false;
    try
    {
        no_rate_step adaptive( ops, problem, control, newton_types<no_rate>::make_solver( ops ), pair );
    }
    catch ( const std::invalid_argument & )
    {
        rejected = true;
    }
    require( rejected, "IMEX embedding silently skipped a required implicit rate" );

    auto complete   = make_problem( ops );
    auto capture    = std::make_shared<imex_error_capture>( ops, .1 );
    using step_type = imex_step<problem_type, imex_error_capture>;
    step_type adaptive( ops, complete, capture, types::make_solver( ops ), pair );
    adaptive.apply( *initial, *out );
    require(
        adaptive.get_status() == single_step_status::converged && capture->exponent == 2 &&
            complete->partitions->explicit_part->calls == 3 && complete->partitions->implicit_part->calls == 1,
        "IMEX did not compute its embedding-only endpoint/zero-diagonal rates"
    );
    adaptive.finalize( adaptation_status::accepted, .1, *out );

    auto                                  ark = runge_kutta::make_imex_butcher_table( "IMEX_ARK324L2SA" );
    const runge_kutta::imex_butcher_table unpaired( ark.explicit_table, ark.implicit_table, 3 );
    auto                                  unavailable = make_problem( ops );
    capture                                           = std::make_shared<imex_error_capture>( ops, .1 );
    step_type strict( ops, unavailable, capture, types::make_solver( ops ), unpaired );
    ops->assign_scalar( 77., *out );
    strict.apply( *initial, *out );
    require(
        strict.get_status() == single_step_status::error_estimate_unavailable && capture->assessments == 0 &&
            unavailable->partitions->explicit_part->calls == 0 && ops->get_value_at_point( 0, *out ) == 77,
        "Partition embeddings were incorrectly treated as a validated combined estimate"
    );
}

void test_adaptive_workspace()
{
    auto ops                   = std::make_shared<counting_operations>( 1 );
    auto problem               = make_problem( ops );
    auto solver                = types::make_solver( ops );
    using step_type            = imex_step<problem_type, imex_error_capture>;
    auto control               = std::make_shared<imex_error_capture>( ops, .1 );
    control->enabled           = false;
    unsigned int fixed_vectors = 0;
    const auto   before        = ops->live;
    {
        step_type fixed( ops, problem, control, solver, step_type::params{ "IMEX_ARK324L2SA" } );
        fixed_vectors = ops->live - before;
    }
    control->enabled = true;
    step_type adaptive( ops, problem, control, solver, step_type::params{ "IMEX_ARK324L2SA" } );
    require( ops->live - before == fixed_vectors + 1, "Adaptive IMEX did not allocate exactly one state-error vector" );
    vector_wrap_type initial( *ops ), out( *ops );
    initial.start_use();
    out.start_use();
    ops->assign_scalar( 1., *initial );
    adaptive.apply( *initial, *out );
    require( adaptive.get_status() == single_step_status::converged, "Adaptive workspace warmup failed" );
    adaptive.finalize( adaptation_status::accepted, .1, *out );
    const auto allocations = ops->allocations;
    for ( int run = 0; run < 3; ++run )
    {
        adaptive.reset();
        adaptive.apply( *initial, *out );
        require(
            adaptive.get_status() == single_step_status::converged && ops->allocations == allocations,
            "Adaptive IMEX allocated vector workspace during restart or repeated solves"
        );
        adaptive.finalize( adaptation_status::accepted, .1, *out );
    }
    auto throwing           = std::make_shared<controlled_solver>( controlled_solver{ ops, solver } );
    throwing->throw_binding = true;
    const auto live         = ops->live;
    bool       rejected     = false;
    try
    {
        imex_step<problem_type, imex_error_capture, controlled_solver> invalid(
            ops, problem, control, throwing, { "IMEX_ARK324L2SA" }
        );
    }
    catch ( const std::runtime_error & )
    {
        rejected = true;
    }
    require( rejected && ops->live == live, "Failed adaptive construction leaked error/rate storage" );
}

void test_modified_adaptive_endpoint()
{
    using filter            = integration::time_step_adaptation_filter<operations_type>;
    using controller        = observed_controller<filter>;
    using step_type         = imex_step<problem_type, controller>;
    auto ops                = std::make_shared<operations_type>( 1 );
    auto problem            = make_problem( ops );
    auto p                  = controller_parameters<filter>( 1e-5 );
    p.controller            = "H211";
    auto control            = std::make_shared<controller>( ops, p );
    control->expected_order = 3;
    step_type        step( ops, problem, control, types::make_solver( ops ), step_type::params{ "IMEX_ARK324L2SA" } );
    vector_wrap_type initial( *ops ), out( *ops ), reference( *ops );
    initial.start_use();
    out.start_use();
    reference.start_use();
    ops->assign_scalar( 1., *initial );
    step.apply( *initial, *out );
    require( step.get_status() == single_step_status::converged, "Adaptive shortened-step setup failed" );
    const auto shortened = step.get_dt() / 2;
    ops->assign_scalar( 2., *initial );
    step.finalize( adaptation_status::accepted_modified, shortened, *initial );
    p.initial_step                = control->get_dt();
    auto fresh_control            = std::make_shared<controller>( ops, p );
    fresh_control->expected_order = 3;
    step_type fresh(
        ops, make_problem( ops ), fresh_control, types::make_solver( ops ), step_type::params{ "IMEX_ARK324L2SA" }
    );
    step.set_time( shortened );
    fresh.set_time( shortened );
    step.apply( *initial, *out );
    fresh.apply( *initial, *reference );
    require(
        step.get_status() == single_step_status::converged && fresh.get_status() == single_step_status::converged &&
            step.get_dt() == fresh.get_dt() &&
            std::abs( ops->get_value_at_point( 0, *out ) - ops->get_value_at_point( 0, *reference ) ) < 1e-13,
        "Modified IMEX endpoint retained invalid controller/rate history"
    );
    step.finalize( adaptation_status::accepted, shortened + step.get_dt(), *out );
    fresh.finalize( adaptation_status::accepted, shortened + fresh.get_dt(), *reference );
}

int main()
try
{
    backend_type::init_device();
    test_failures();
    test_optional_interfaces();
    test_ownership_and_lifecycle();
    test_modified_endpoint_and_backward();
    test_adaptive_recovery();
    test_embedding_capabilities();
    test_adaptive_workspace();
    test_modified_adaptive_endpoint();
    std::cout << "Fixed IMEX bounded failures, optional rates, commitment, ownership and workspace reuse: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
