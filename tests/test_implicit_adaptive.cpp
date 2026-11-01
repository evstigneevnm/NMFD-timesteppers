#include "implicit_adaptive_test.h"
#include "nonautonomous_test_problems.h"
#include <limits>

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;

// u'=-u, u(0)=1, exact exp(-t): regular autonomous decay with identity mass.
// Its scalar stage solves give independent primary and embedded reference values.
struct decay
{
    __DEVICE_TAG__ double mass( double ) const
    {
        return 1;
    }
    __DEVICE_TAG__ double rhs( double, double u ) const
    {
        return -u;
    }
    __DEVICE_TAG__ double rhs_derivative( double, double ) const
    {
        return -1;
    }
    double exact( double t ) const
    {
        return std::exp( -t );
    }
};

// u'=-u^2, u(0)=1, exact 1/(1+t): nonlinear decay, regular on t>-1.
struct quadratic_decay : decay
{
    __DEVICE_TAG__ double rhs( double, double u ) const
    {
        return -u * u;
    }
    __DEVICE_TAG__ double rhs_derivative( double, double u ) const
    {
        return -2 * u;
    }
    double exact( double t ) const
    {
        return 1 / ( 1 + t );
    }
};

// (1+u^2)*u'=-(1+u^2)*u, exact exp(-t). Positive state-dependent mass;
// the nonlinear stage Jacobian includes its derivative, not only a frozen mass.
struct state_mass_decay
{
    static constexpr int size = 1;
    __DEVICE_TAG__ void  residual(
         double, const stage_scaling &scaling, const vector_type &x, const vector_type &previous, const vector_type &rhs,
         vector_type &out
     ) const
    {
        const auto u = x( 0 );
        out( 0 ) = ( 1 + u * u ) * ( scaling.mass * ( u - previous( 0 ) ) + scaling.rhs * u ) - scaling.rhs * rhs( 0 );
    }
    auto jacobian(
        double, const stage_scaling &scaling, const scfd::static_vec::vec<double, 1> &x,
        const scfd::static_vec::vec<double, 1> &previous
    ) const
    {
        const auto u = x[0];
        return scfd::static_mat::mat<double, 1, 1>{
            scaling.mass * ( 1 + u * u + 2 * u * ( u - previous[0] ) ) + scaling.rhs * ( 1 + 3 * u * u )
        };
    }
    double exact( double t ) const
    {
        return std::exp( -t );
    }
};

// x'=-x, 0=z-sin(t)*x: a non-autonomous index-1 DAE with M=diag(1,0).
// Exact (x,z)=(exp(-t),sin(t)*exp(-t)); the stiffly accurate primary endpoint
// must remain on the moving constraint, regardless of its embedded companion.
struct moving_constraint
{
    static constexpr int size = 2;
    __DEVICE_TAG__ void  residual(
         double t, const stage_scaling &scaling, const vector_type &x, const vector_type &previous,
         const vector_type &rhs, vector_type &out
     ) const
    {
        out( 0 ) = scaling.mass * ( x( 0 ) - previous( 0 ) ) + scaling.rhs * ( x( 0 ) - rhs( 0 ) );
        out( 1 ) = x( 1 ) - std::sin( t ) * x( 0 ) - rhs( 1 );
    }
    auto jacobian(
        double t, const stage_scaling &scaling, const scfd::static_vec::vec<double, 2> &,
        const scfd::static_vec::vec<double, 2> &
    ) const
    {
        return scfd::static_mat::mat<double, 2, 2>{ scaling.mass + scaling.rhs, 0, -std::sin( t ), 1 };
    }
};

using problem_type = stage_problem<scalar_time_equation<decay>>;
using types        = newton_types<problem_type>;

struct error_recorder : adaptation_type
{
    std::shared_ptr<operations_type> operations;
    double                           error = 0;
    unsigned int                     order = 0;
    error_recorder( std::shared_ptr<operations_type> ops, double h )
        : adaptation_type( params{ h } ), operations( std::move( ops ) )
    {
    }
    bool requires_error_estimate() const
    {
        return true;
    }
    adaptation_status assess(
        double, double, const vector_type &, const vector_type &, double &next, unsigned int exponent = 0,
        const vector_type *estimate = nullptr
    )
    {
        require( estimate != nullptr, "DIRK did not supply its embedded error" );
        error = operations->get_value_at_point( 0, *estimate );
        order = exponent;
        next  = get_dt();
        return adaptation_status::accepted;
    }
};

void test_estimators()
{
    auto             operations = std::make_shared<operations_type>( 1 );
    auto             problem    = std::make_shared<problem_type>( operations );
    vector_wrap_type initial( *operations ), result( *operations );
    initial.start_use();
    result.start_use();
    operations->assign_scalar( 1., *initial );
    const runge_kutta::butcher_table general( { { .25L, 0 }, { .25L, .5L } }, { .5L, .5L }, 2, {}, { 1, 0 }, 1 );
    const runge_kutta::butcher_table no_endpoint_error(
        { { .25L, 0, 0 }, { .25L, .25L, 0 }, { .5L, .25L, .25L } }, { .5L, .25L, .25L }, 2, {}, { .75L, 0, .25L }, 1
    );
    for ( const auto &table :
          { runge_kutta::make_butcher_table( "SDIRK2(1)2" ), runge_kutta::make_butcher_table( "SDIRK3(1)3" ), general,
            no_endpoint_error } )
    {
        double primary_errors[3]{}, embedded_errors[3]{}, estimates[3]{};
        for ( int resolution = 0; resolution < 3; ++resolution )
        {
            const auto h          = .2 / ( 1 << resolution );
            auto       adaptation = std::make_shared<error_recorder>( operations, h );
            using step_type =
                runge_kutta::implicit_time_step<operations_type, problem_type, error_recorder, types::solver_type>;
            step_type step( operations, problem, adaptation, types::make_solver( operations ), table );
            step.set_target_time( h );
            step.apply( *initial, *result );
            require( step.get_status() == single_step_status::converged, "Embedded DIRK step failed" );
            std::vector<long double> rates( table.size() );
            long double              primary = 1, embedded = 1;
            for ( std::size_t i = 0; i < table.size(); ++i )
            {
                long double previous = 1;
                for ( std::size_t j = 0; j < i; ++j )
                    previous += h * table.a( i, j ) * rates[j];
                rates[i] = -previous / ( 1 + h * table.a( i, i ) );
                primary += h * table.b( i ) * rates[i];
                embedded += h * table.embedded_b( i ) * rates[i];
            }
            const auto value = operations->get_value_at_point( 0, *result );
            require( std::abs( value - primary ) < 2e-10, "DIRK published the embedded rather than primary state" );
            require(
                std::abs( adaptation->error - ( primary - embedded ) ) < 2e-10,
                "DIRK state error omitted a stage or multiplied by the timestep twice"
            );
            require( adaptation->order == table.embedded_order() + 1, "DIRK estimator exponent is incorrect" );
            primary_errors[resolution]  = std::abs( value - std::exp( -h ) );
            embedded_errors[resolution] = std::abs( value - adaptation->error - std::exp( -h ) );
            estimates[resolution]       = std::abs( adaptation->error );
            step.finalize( adaptation_status::accepted, h, *result );
        }
        require(
            std::log2( primary_errors[1] / primary_errors[2] ) > table.order() + .65,
            "DIRK primary local order changed during adaptive execution"
        );
        require(
            std::abs( std::log2( embedded_errors[1] / embedded_errors[2] ) - table.error_order() ) < .3,
            "DIRK embedded local order mismatch"
        );
        require(
            std::abs( std::log2( estimates[1] / estimates[2] ) - table.error_order() ) < .3,
            "DIRK state-error scaling mismatch"
        );
    }
}

template <class Controller, class Equation>
void test_convergence( Equation equation, const char *label, const char *method = "SDIRK2(1)2" )
{
    using problem      = stage_problem<Equation>;
    using solver_types = newton_types<problem>;
    using adaptation   = observed_controller<Controller>;
    using step_type =
        runge_kutta::implicit_time_step<operations_type, problem, adaptation, typename solver_types::solver_type>;
    auto             operations = std::make_shared<operations_type>( 1 );
    auto             model      = std::make_shared<problem>( operations, equation );
    vector_wrap_type initial( *operations ), result( *operations );
    initial.start_use();
    result.start_use();
    constexpr double start = .25, end = .65;
    operations->assign_scalar( equation.exact( start ), *initial );
    double      errors[2]{};
    std::size_t steps[2]{};
    for ( int refinement = 0; refinement < 2; ++refinement )
    {
        auto control =
            std::make_shared<adaptation>( operations, controller_parameters<Controller>( refinement ? 2e-5 : 2e-3 ) );
        auto step = std::make_shared<step_type>(
            operations, model, control, solver_types::make_solver( operations ), typename step_type::params{ method }
        );
        integration::time_integrator<operations_type, step_type> integrator( operations, step );
        integrator.set_time_interval( start, end );
        double first_value = 0;
        for ( int run = 0; run < 2; ++run )
        {
            integrator.apply( *initial, *result );
            require(
                integrator.get_status() == integration_status::completed && integrator.get_final_time() == end,
                "Adaptive DIRK failed to reach the exact target"
            );
            const auto value = operations->get_value_at_point( 0, *result );
            require( control->commits == integrator.get_steps(), "DIRK committed a rejected candidate" );
            if ( !run )
            {
                first_value       = value;
                steps[refinement] = integrator.get_steps();
            }
            else
                require(
                    value == first_value && steps[refinement] == integrator.get_steps(),
                    "Adaptive DIRK retained state from a previous trajectory"
                );
            errors[refinement] = std::abs( value - equation.exact( end ) );
        }
        if ( refinement )
            require( control->error_rejections > 0, "Adaptive DIRK did not exercise embedded-error rejection" );
    }
    require(
        errors[1] < .25 * errors[0] && steps[1] > steps[0], "DIRK accuracy did not improve with tighter tolerances"
    );
    std::cout << label << " " << method << ": errors " << errors[0] << " -> " << errors[1] << ", steps " << steps[0]
              << " -> " << steps[1] << '\n';
}

struct controlled_solver
{
    std::shared_ptr<operations_type>    operations;
    std::shared_ptr<types::solver_type> solver;
    unsigned int                        calls = 0, fail_on = 0, invalid_on = 0;
    bool                                fail_all = false;
    void                                set_nonlinear_operator( std::shared_ptr<problem_type> problem )
    {
        solver->set_nonlinear_operator( std::move( problem ) );
    }
    bool solve( vector_type &state )
    {
        ++calls;
        if ( fail_all || calls == fail_on )
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

template <class Controller>
void test_recovery()
{
    using adaptation = observed_controller<Controller>;
    using step_type  = runge_kutta::implicit_time_step<operations_type, problem_type, adaptation, controlled_solver>;
    auto operations  = std::make_shared<operations_type>( 1 );
    auto problem     = std::make_shared<problem_type>( operations );
    auto solver =
        std::make_shared<controlled_solver>( controlled_solver{ operations, types::make_solver( operations ) } );
    vector_wrap_type initial( *operations ), result( *operations );
    initial.start_use();
    result.start_use();
    operations->assign_scalar( 1., *initial );
    auto p = controller_parameters<Controller>( .1 );
    for ( bool invalid : { false, true } )
    {
        auto      control = std::make_shared<adaptation>( operations, p );
        step_type step( operations, problem, control, solver, typename step_type::params{ "SDIRK2(1)2", 8 } );
        solver->fail_on    = invalid ? 0 : solver->calls + 2;
        solver->invalid_on = invalid ? solver->calls + 1 : 0;
        operations->assign_scalar( 42., *result );
        step.apply( *initial, *result );
        require(
            step.get_status() == single_step_status::converged && step.get_attempts() == 2 &&
                control->numerical_rejections == 1 && control->assessments == 1 && !control->commits,
            "Invalid DIRK stage entered error assessment or bypassed recovery"
        );
        auto             clean_control = std::make_shared<adaptation_type>( adaptation_type::params{ step.get_dt() } );
        types::step_type clean(
            operations, problem, clean_control, types::make_solver( operations ),
            types::step_type::params{ "SDIRK2(1)2" }
        );
        vector_wrap_type reference( *operations );
        reference.start_use();
        clean.apply( *initial, *reference );
        require(
            std::abs( operations->get_value_at_point( 0, *result ) - operations->get_value_at_point( 0, *reference ) ) <
                    1e-12 &&
                operations->get_value_at_point( 0, *initial ) == 1,
            "Rejected DIRK stages contaminated the successful retry"
        );
        clean.finalize( adaptation_status::accepted, clean.get_dt(), *reference );
        const auto shortened = step.get_dt() / 2;
        operations->assign_scalar( std::exp( -shortened ), *result );
        step.finalize( adaptation_status::accepted_modified, shortened, *result );
        require(
            control->commits == 1 && control->get_dt() <= shortened,
            "Modified DIRK commitment did not limit the next step"
        );
    }
    solver->fail_on = solver->invalid_on = 0;
    for ( int failure = 0; failure < 3; ++failure )
    {
        p              = controller_parameters<Controller>( 1e-10 );
        p.initial_step = .1;
        if ( failure != 2 )
            p.minimum_step = .1;
        solver->fail_all  = failure != 0;
        auto      control = std::make_shared<adaptation>( operations, p );
        step_type step( operations, problem, control, solver, typename step_type::params{ "SDIRK2(1)2", 2 } );
        operations->assign_scalar( 42., *result );
        step.apply( *initial, *result );
        const auto expected = failure == 0   ? single_step_status::failed_minimum_dt
                              : failure == 1 ? single_step_status::failed_solver
                                             : single_step_status::attempt_limit_reached;
        require(
            step.get_status() == expected && !control->commits && operations->get_value_at_point( 0, *result ) == 42 &&
                operations->get_value_at_point( 0, *initial ) == 1,
            "Adaptive DIRK terminal failure published an uncommitted state"
        );
        require(
            failure == 0 ? control->assessments == 1 : control->assessments == 0,
            "Failed Newton solve was assessed as an embedded candidate"
        );
        if ( failure == 2 )
            require( step.get_attempts() == 2, "Adaptive DIRK exceeded its attempt budget" );
    }
    solver->fail_all = false;
    auto control     = std::make_shared<adaptation>( operations, controller_parameters<Controller>() );
    for ( const char *method : { "IE", "IM" } )
    {
        step_type  step( operations, problem, control, solver, typename step_type::params{ method } );
        const auto calls = solver->calls;
        step.apply( *initial, *result );
        require(
            step.get_status() == single_step_status::error_estimate_unavailable && solver->calls == calls,
            "Adaptive DIRK silently invented an estimator"
        );
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
                throw std::runtime_error( "Injected adaptive DIRK activation failure" );
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
        auto problem     = std::make_shared<problem_type>( operations );
        auto solver      = types::make_solver( operations );
        using adaptation = integration::time_step_adaptation_scipy<operations_type>;
        using adaptive_step =
            runge_kutta::implicit_time_step<counting_operations, problem_type, adaptation, types::solver_type>;
        using constant_step =
            runge_kutta::implicit_time_step<counting_operations, problem_type, adaptation_type, types::solver_type>;
        auto       controller = std::make_shared<adaptation>( operations, controller_parameters<adaptation>( .01 ) );
        const auto baseline   = operations->live;
        for ( unsigned int failure = 1; failure <= 4; ++failure )
        {
            operations->fail_activation = operations->allocations + failure;
            bool failed                 = false;
            try
            {
                adaptive_step step( operations, problem, controller, solver, adaptive_step::params{ "SDIRK2(1)2" } );
            }
            catch ( const std::runtime_error & )
            {
                failed = true;
            }
            require( failed && operations->live == baseline, "Adaptive DIRK construction leaked workspace" );
        }
        operations->fail_activation = 0;
        {
            constant_step step(
                operations, problem, std::make_shared<adaptation_type>(), solver, constant_step::params{ "SDIRK2(1)2" }
            );
            require( operations->live == baseline + 2, "Constant DIRK allocated error or endpoint-rate storage" );
        }
        auto step = std::make_shared<adaptive_step>(
            operations, problem, controller, solver, adaptive_step::params{ "SDIRK2(1)2" }
        );
        require( operations->live == baseline + 4, "Adaptive DIRK workspace count is incorrect" );
        vector_wrap_type initial( *operations ), result( *operations );
        initial.start_use();
        result.start_use();
        operations->assign_scalar( 1., *initial );
        integration::time_integrator<operations_type, adaptive_step> integrator( operations, step );
        integrator.set_time_interval( 0, .4 );
        integrator.apply( *initial, *result );
        require( integrator.get_status() == integration_status::completed, "Adaptive DIRK storage warmup failed" );
        const auto allocations = operations->allocations;
        integrator.apply( *initial, *result );
        require(
            integrator.get_status() == integration_status::completed && operations->allocations == allocations,
            "Adaptive DIRK allocated trajectory vectors during integration"
        );
    }
    require( !operations->live, "Adaptive DIRK workspace was not released" );
}

struct constraint_monitor
{
    std::shared_ptr<operations_type> operations;
    double                           time  = 0;
    unsigned int                     calls = 0;
    void                             set_time_interval( const double &, double &end )
    {
        time = end;
    }
    bool apply( const vector_type &, vector_type &out )
    {
        double state[2];
        operations->get( out, state, 2 );
        require(
            std::abs( state[1] - std::sin( time ) * state[0] ) < 2e-9,
            "Adaptive DIRK committed an inconsistent DAE endpoint"
        );
        ++calls;
        return false;
    }
    integration_status get_status() const
    {
        return integration_status::running;
    }
};

template <class Controller>
void test_constraint_and_backward()
{
    using problem      = stage_problem<moving_constraint>;
    using solver_types = newton_types<problem>;
    using step_type =
        runge_kutta::implicit_time_step<operations_type, problem, Controller, typename solver_types::solver_type>;
    auto operations = std::make_shared<operations_type>( 2 );
    auto model      = std::make_shared<problem>( operations );
    auto controller = std::make_shared<Controller>( operations, controller_parameters<Controller>( 1e-4 ) );
    auto step       = std::make_shared<step_type>(
        operations, model, controller, solver_types::make_solver( operations ),
        typename step_type::params{ "SDIRK2(1)2" }
    );
    auto monitor = std::make_shared<constraint_monitor>( constraint_monitor{ operations } );
    integration::time_integrator<operations_type, step_type, constraint_monitor> integrator(
        operations, step, {}, monitor
    );
    vector_wrap_type initial( *operations ), result( *operations );
    initial.start_use();
    result.start_use();
    for ( bool backwards : { false, true } )
    {
        const auto   start = backwards ? .55 : .25, end = backwards ? .25 : .55;
        const double state[] = { std::exp( -start ), std::sin( start ) * std::exp( -start ) };
        operations->set( state, *initial, 2 );
        monitor->calls = 0;
        integrator.set_time_interval( start, end );
        integrator.apply( *initial, *result );
        require(
            integrator.get_status() == integration_status::completed && integrator.get_final_time() == end &&
                monitor->calls == integrator.get_steps(),
            "Adaptive DIRK constrained traversal failed"
        );
        require(
            std::abs( operations->get_value_at_point( 0, *result ) - std::exp( -end ) ) < 2e-5,
            "Adaptive DIRK constrained/backward accuracy failed"
        );
    }
}

template <class Controller>
void test_controller( const char *name )
{
    test_convergence<Controller>( scalar_time_equation<decay>{}, name );
    test_convergence<Controller>( scalar_time_equation<decay>{}, name, "SDIRK3(1)3" );
    test_convergence<Controller>( scalar_time_equation<quadratic_decay>{}, "nonlinear decay" );
    test_convergence<Controller>( scalar_time_equation<time_mass_decay<double>>{}, "time-dependent mass" );
    test_convergence<Controller>( state_mass_decay{}, "state-dependent mass" );
    test_convergence<Controller>(
        scalar_time_equation<stiff_tracking<double>>{ stiff_tracking<double>{ 100 } }, "stiff tracking"
    );
    test_recovery<Controller>();
    test_constraint_and_backward<Controller>();
}

int main()
try
{
    backend_type::init_device();
    test_estimators();
    test_controller<integration::time_step_adaptation_matlab<operations_type>>( "MATLAB" );
    test_controller<integration::time_step_adaptation_scipy<operations_type>>( "SciPy" );
    test_storage();
    std::cout << "Adaptive DIRK estimators, accuracy, recovery, rollback and storage: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
