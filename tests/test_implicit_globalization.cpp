#include "implicit_test_support.h"
#include <limits>
#include <nmfd/time_steppers/integration/time_step_adaptation_globalization.h>

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;

// Scalar controller probes need no vector storage or solver dependency.
struct scalar_operations
{
    using scalar_type = double;
    using vector_type = double;
};
struct scalar_merit
{
    unsigned int calls = 0;
    double       operator()( double value )
    {
        ++calls;
        return value;
    }
};
using scalar_control = integration::time_step_adaptation_globalization<scalar_operations, scalar_merit>;

void test_policy()
{
    auto                   merit = std::make_shared<scalar_merit>();
    scalar_control::params p;
    p.initial_step = 1;
    p.minimum_step = .125;
    p.maximum_step = 8;
    scalar_control control( merit, p );
    double         next = 0;
    require( !control.requires_error_estimate(), "Globalization requested a local truncation error" );
    control.initialize( 0, 8 );
    require(
        control.assess( 0, 1, 8, 4, next ) == adaptation_status::accepted && std::abs( next - 2 ) < 1e-14,
        "Residual-ratio growth is incorrect"
    );
    require( merit->calls == 2, "Initial globalization assessment repeated a merit evaluation" );
    control.update( adaptation_status::accepted, 1, 1, 4 );
    require(
        control.assess( 1, 2, 4, 1, next ) == adaptation_status::accepted && next == 4 && merit->calls == 3,
        "Committed merit was not cached or growth was not bounded"
    );
    control.update( adaptation_status::failed, 1, 0, 4 );
    require( control.get_dt() == 2, "Failed commitment retained candidate step growth" );
    require(
        control.assess( 1, 2, 4, 2, next ) == adaptation_status::accepted && next == 4,
        "Failed commitment promoted the candidate merit"
    );
    control.update( adaptation_status::failed, 1, 0, 4 );
    require(
        control.reject_step( 2 ) == adaptation_status::rejected && control.get_dt() == 1,
        "Numerical failure did not shrink the retry"
    );
    control.update( adaptation_status::rejected, 1, 2, 4 );
    require(
        control.assess( 1, 1, 4, 2, next ) == adaptation_status::accepted && next == 1,
        "A rejected trial corrupted the baseline or permitted immediate growth"
    );
    control.update( adaptation_status::accepted_modified, 1.5, .5, 16 );
    require( control.get_dt() == .5, "Modified pseudo-time endpoint did not limit the next proposal" );
    const auto calls = merit->calls;
    require(
        control.assess( 1.5, .5, 16, 8, next ) == adaptation_status::accepted && std::abs( next - 1 ) < 1e-14 &&
            merit->calls == calls + 2,
        "Modified endpoint reused an obsolete merit"
    );
    control.update( adaptation_status::accepted, 2, .5, 8 );
    require(
        control.assess( 10, 1, 64, 32, next ) == adaptation_status::accepted && std::abs( next - 2 ) < 1e-14,
        "A pseudo-time discontinuity reused the old merit"
    );

    const auto growth_calls = merit->calls;
    control.reset();
    require(
        control.assess( 0, 1, 10, 12, next ) == adaptation_status::accepted && std::abs( next - 5. / 6 ) < 1e-14,
        "Default policy rejected finite merit growth or failed to shrink the next step"
    );
    control.update( adaptation_status::accepted, 1, 1, 12 );
    require(
        control.assess( 1, next, 12, 6, next ) == adaptation_status::accepted && std::abs( next - 5. / 3 ) < 1e-14 &&
            merit->calls == growth_calls + 3,
        "Accepted merit growth was not committed as the next baseline"
    );

    p.maximum_merit_growth = 1;
    scalar_control monotone( merit, p );
    require(
        monotone.assess( 0, 1, 1, 2, next ) == adaptation_status::rejected && next == .5,
        "Explicit monotone merit guard was ignored"
    );
    p.maximum_merit_growth = 4;
    scalar_control permissive( merit, p );
    require(
        permissive.assess( 0, 1, 1, 4, next ) == adaptation_status::accepted && next == .5,
        "Configured merit growth did not use bounded step shrinkage"
    );
    permissive.reset();
    require(
        permissive.assess( 0, 1, 1, 5, next ) == adaptation_status::rejected,
        "Explicit merit growth guard accepted excessive growth"
    );
    p.residual_power = .5;
    scalar_control fractional( merit, p );
    require(
        fractional.assess( 0, 1, 8, 4, next ) == adaptation_status::accepted &&
            std::abs( next - std::sqrt( 2. ) ) < 1e-14,
        "Configured residual-ratio exponent was ignored"
    );
    for ( const double value : { 0., 1e-300, 1e300 } )
    {
        control.reset();
        require(
            control.assess( 0, 8, value, 0, next ) == adaptation_status::accepted && next == 8,
            "Zero merit or extreme residual ratios broke bounded adaptation"
        );
    }
    control.reset();
    require(
        control.assess( 0, 1, 1e300, 1e-300, next ) == adaptation_status::accepted && next == 2,
        "Extreme nonzero merits overflowed the residual-ratio proposal"
    );
    control.reset();
    require(
        control.assess( 0, 1, 0, 1, next ) == adaptation_status::accepted && next == .5,
        "Zero-to-nonzero finite merit did not use bounded next-step shrinkage"
    );
    monotone.reset();
    require(
        monotone.assess( 0, 1, 0, 1, next ) == adaptation_status::rejected,
        "Explicit monotone guard allowed leaving zero merit"
    );
    control.reset();
    require(
        control.assess( 0, 1, 1e-300, 1e300, next ) == adaptation_status::accepted && next == .5,
        "Extreme merit growth overflowed the default proposal or rejected a finite candidate"
    );
    for ( const double invalid :
          { -1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() } )
    {
        control.reset();
        require(
            control.assess( 0, 1, invalid, 0, next ) == adaptation_status::failed,
            "Invalid committed merit was accepted"
        );
        control.reset();
        require(
            control.assess( 0, 1, 1, invalid, next ) == adaptation_status::rejected && next == .5,
            "Invalid trial merit did not request a smaller step"
        );
    }
    control.reset();
    require(
        control.reject_step( .125 ) == adaptation_status::failed && control.get_dt() == 1,
        "Globalization retried below the minimum step"
    );
    for ( const double h :
          { 0., -1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() } )
    {
        require(
            control.assess( 0, h, 1, .5, next ) == adaptation_status::failed &&
                control.reject_step( h ) == adaptation_status::failed && control.get_dt() == 1,
            "Invalid/backward pseudo-time step changed controller state"
        );
    }
    control.initialize( 0, 1, 100 );
    require( control.get_dt() == 8, "Initial pseudo-time step was not bounded" );
    control.reset();
    require(
        control.get_dt() == 1 && control.assess( 0, 1, 8, 4, next ) == adaptation_status::accepted &&
            std::abs( next - 2 ) < 1e-14,
        "Reset retained globalization history"
    );
    require(
        control.assess( 1e300, 1, 1, .5, next ) == adaptation_status::failed,
        "Unrepresentable pseudo-time advancement was accepted"
    );
}

void test_configuration()
{
    auto   merit                                   = std::make_shared<scalar_merit>();
    double scalar_control::params::*const fields[] = {
        &scalar_control::params::initial_step,    &scalar_control::params::minimum_step,
        &scalar_control::params::maximum_step,    &scalar_control::params::residual_power,
        &scalar_control::params::minimum_factor,  &scalar_control::params::maximum_factor,
        &scalar_control::params::rejection_factor
    };
    for ( const auto field : fields )
        for ( const double value :
              { 0., -1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() } )
        {
            scalar_control::params p;
            p.*field      = value;
            bool rejected = false;
            try
            {
                scalar_control invalid( merit, p );
            }
            catch ( const std::invalid_argument & )
            {
                rejected = true;
            }
            require( rejected, "Globalization accepted invalid parameters" );
        }
    for ( const double value :
          { -1., .5, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() } )
    {
        scalar_control::params p;
        p.maximum_merit_growth = value;
        bool rejected          = false;
        try
        {
            scalar_control invalid( merit, p );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        require( rejected, "Globalization accepted an invalid optional merit guard" );
    }
    for ( int invalid = 0; invalid < 5; ++invalid )
    {
        scalar_control::params p;
        if ( invalid == 0 )
            p.minimum_step = p.maximum_step * 2;
        if ( invalid == 1 )
            p.minimum_factor = 2;
        if ( invalid == 2 )
            p.maximum_factor = .5;
        if ( invalid == 3 )
            p.rejection_factor = 1;
        if ( invalid == 4 )
            p.maximum_merit_growth = .5;
        bool rejected = false;
        try
        {
            scalar_control control( merit, p );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        require( rejected, "Globalization accepted inverted/noncontracting bounds" );
    }
    bool rejected = false;
    try
    {
        scalar_control control( nullptr );
    }
    catch ( const std::invalid_argument & )
    {
        rejected = true;
    }
    require( rejected, "Globalization accepted an absent merit measure" );
    scalar_control control( merit );
    for ( const double value :
          { 0., -2., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() } )
    {
        rejected = false;
        try
        {
            control.initialize( 0, 1, value );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        require( rejected, "Globalization accepted an invalid initial step" );
    }
    std::weak_ptr<scalar_merit> lifetime = merit;
    merit.reset();
    require( !lifetime.expired(), "Globalization did not retain its merit measure" );
}

// Autonomous relaxation u'=1-u^3 has the unique real stable equilibrium u=1.
// Its nonlinear stage Jacobian is analytic; growing pseudo-time steps approach
// the steady Newton equation, with no physical-time error estimate involved.
struct cubic_relaxation
{
    __DEVICE_TAG__ double mass( double ) const
    {
        return 1;
    }
    __DEVICE_TAG__ double rhs( double, double u ) const
    {
        return 1 - u * u * u;
    }
    __DEVICE_TAG__ double rhs_derivative( double, double u ) const
    {
        return -3 * u * u;
    }
};

// Singular-mass DAE: u'=1-u, 0=z-2u. The steady state is (1,2).
// Differential rows use balanced stage scaling; the algebraic row remains
// unscaled. No mass inverse or unconstrained interpolation is required.
struct constrained_relaxation
{
    static constexpr int size = 2;
    __DEVICE_TAG__ void  residual(
         double, const stage_scaling &s, const vector_type &x, const vector_type &previous, const vector_type &rhs,
         vector_type &out
     ) const
    {
        out( 0 ) = s.mass * ( x( 0 ) - previous( 0 ) ) - s.rhs * ( 1 - x( 0 ) + rhs( 0 ) );
        out( 1 ) = x( 1 ) - 2 * x( 0 );
    }
    auto jacobian(
        double, const stage_scaling &s, const scfd::static_vec::vec<double, 2> &,
        const scfd::static_vec::vec<double, 2> &
    ) const
    {
        return scfd::static_mat::mat<double, 2, 2>{ s.mass + s.rhs, 0, -2, 1 };
    }
};

struct physical_merit
{
    std::shared_ptr<operations_type> operations;
    bool                             constrained = false;
    unsigned int                     calls = 0, invalid_on = 0;
    double                           operator()( const vector_type &state )
    {
        ++calls;
        if ( calls == invalid_on )
            return std::numeric_limits<double>::infinity();
        // Only scalar samples cross the backend boundary in these tiny tests.
        const auto u = operations->get_value_at_point( 0, state );
        return constrained
                   ? std::max( std::abs( 1 - u ), std::abs( operations->get_value_at_point( 1, state ) - 2 * u ) )
                   : std::abs( 1 - u * u * u );
    }
};
using controller = integration::time_step_adaptation_globalization<operations_type, physical_merit>;

struct steady_stop
{
    std::shared_ptr<physical_merit> merit;
    integration_status              status = integration_status::running;
    void                            set_time_interval( const double &, double & )
    {
    }
    bool apply( const vector_type &, vector_type &state )
    {
        status = ( *merit )( state ) <= 1e-8 ? integration_status::stopped_by_external_operation
                                             : integration_status::running;
        return false;
    }
    integration_status get_status() const
    {
        return status;
    }
};

template <class Equation>
void test_equilibrium( bool constrained )
{
    using problem_type = stage_problem<Equation>;
    using types        = newton_types<problem_type>;
    using step_type =
        runge_kutta::implicit_time_step<operations_type, problem_type, controller, typename types::solver_type>;
    auto             operations = std::make_shared<operations_type>( Equation::size );
    auto             problem    = std::make_shared<problem_type>( operations );
    auto             merit      = std::make_shared<physical_merit>( physical_merit{ operations, constrained } );
    auto             control    = std::make_shared<controller>( merit );
    vector_wrap_type initial( *operations ), result( *operations ), reference( *operations );
    initial.start_use();
    result.start_use();
    reference.start_use();
    const double values[] = { 2, 4 };
    operations->set( values, *initial, Equation::size );
    for ( const char *method : { "IE", "SDIRK2(1)2", "SDIRK3(1)3" } )
    {
        auto step = std::make_shared<step_type>(
            operations, problem, control, types::make_solver( operations ), typename step_type::params{ method }
        );
        auto stop = std::make_shared<steady_stop>( steady_stop{ merit } );
        integration::time_integrator<operations_type, step_type, steady_stop> integrator(
            operations, step, { 200 }, stop
        );
        integrator.set_time_interval( 0, 1e6 );
        double      first_value = 0;
        std::size_t first_steps = 0;
        for ( int run = 0; run < 2; ++run )
        {
            integrator.apply( *initial, *result );
            require(
                integrator.get_status() == integration_status::stopped_by_external_operation &&
                    ( *merit )( *result ) <= 1e-8 && control->get_dt() > 1,
                "Pseudo-time DIRK did not converge toward the known steady state"
            );
            const auto value = operations->get_value_at_point( 0, *result );
            require(
                std::abs( value - 1 ) < 1e-8 && operations->get_value_at_point( 0, *initial ) == 2,
                "Globalization changed the equilibrium or initial state"
            );
            if ( !run )
            {
                first_value = value;
                first_steps = integrator.get_steps();
            }
            else
                require(
                    value == first_value && integrator.get_steps() == first_steps,
                    "Repeated relaxation retained controller history"
                );
        }
        std::cout << method << ( constrained ? " DAE" : " cubic ODE" ) << ": " << first_steps << " steps, merit "
                  << ( *merit )( *result ) << '\n';
    }

    // Controller selection must not alter the stage equations or fixed-h trajectory.
    controller::params p;
    p.initial_step = p.minimum_step = p.maximum_step = .125;
    auto bounded_control                             = std::make_shared<controller>( merit, p );
    auto bounded_step =
        std::make_shared<step_type>( operations, problem, bounded_control, types::make_solver( operations ) );
    integration::time_integrator<operations_type, step_type> bounded( operations, bounded_step );
    auto                                                     fixed_step = make_step( operations, problem, .125, "IE" );
    integration::time_integrator<operations_type, typename types::step_type> fixed( operations, fixed_step );
    bounded.set_time_interval( 0, .5 );
    fixed.set_time_interval( 0, .5 );
    bounded.apply( *initial, *result );
    fixed.apply( *initial, *reference );
    require(
        bounded.get_status() == integration_status::completed && fixed.get_status() == integration_status::completed &&
            bounded.get_steps() == fixed.get_steps(),
        "Controller selection changed fixed-step traversal"
    );
    for ( int i = 0; i < Equation::size; ++i )
        require(
            operations->get_value_at_point( i, *result ) == operations->get_value_at_point( i, *reference ),
            "Globalization changed an ordinary fixed-step trajectory"
        );
}

using relaxation_problem = stage_problem<scalar_time_equation<cubic_relaxation>>;
using relaxation_types   = newton_types<relaxation_problem>;
struct failing_solver
{
    std::shared_ptr<operations_type>               operations;
    std::shared_ptr<relaxation_types::solver_type> solver;
    unsigned int                                   calls = 0, failures = 0;
    void                                           set_nonlinear_operator( std::shared_ptr<relaxation_problem> problem )
    {
        solver->set_nonlinear_operator( std::move( problem ) );
    }
    bool solve( vector_type &state )
    {
        ++calls;
        if ( failures )
        {
            --failures;
            operations->assign_scalar( 99., state );
            return false;
        }
        return solver->solve( state );
    }
};

void test_recovery()
{
    using step_type = runge_kutta::implicit_time_step<operations_type, relaxation_problem, controller, failing_solver>;
    auto operations = std::make_shared<operations_type>( 1 );
    auto problem    = std::make_shared<relaxation_problem>( operations );
    auto merit      = std::make_shared<physical_merit>( physical_merit{ operations } );
    auto solver =
        std::make_shared<failing_solver>( failing_solver{ operations, relaxation_types::make_solver( operations ) } );
    vector_wrap_type initial( *operations ), result( *operations ), reference( *operations );
    initial.start_use();
    result.start_use();
    reference.start_use();
    operations->assign_scalar( 2., *initial );
    controller::params p;
    p.initial_step = .1;
    for ( int scenario = 0; scenario < 5; ++scenario )
    {
        p.minimum_step    = scenario == 2 || scenario == 4 ? .1 : 1e-14;
        auto      control = std::make_shared<controller>( merit, p );
        step_type step( operations, problem, control, solver, step_type::params{ "IE", 3 } );
        merit->calls      = 0;
        merit->invalid_on = scenario == 1 || scenario == 4 ? 2 : 0;
        solver->calls     = 0;
        solver->failures  = scenario == 1 || scenario == 4 ? 0 : scenario == 0 ? 1 : 10;
        operations->assign_scalar( 42., *result );
        step.apply( *initial, *result );
        if ( scenario < 2 )
        {
            require(
                step.get_status() == single_step_status::converged && step.get_attempts() == 2 &&
                    step.get_dt() == .05 && solver->calls == 2 && control->get_dt() <= .05,
                "Globalization failed to retry a numerical/merit failure"
            );
            require(
                merit->calls == ( scenario == 0 ? 2u : 3u ),
                "Failed stage was assessed or baseline was reevaluated on retry"
            );
            auto clean = make_step( operations, problem, .05, "IE" );
            clean->apply( *initial, *reference );
            require(
                clean->get_status() == single_step_status::converged &&
                    operations->get_value_at_point( 0, *result ) == operations->get_value_at_point( 0, *reference ),
                "Rejected pseudo-time stages contaminated the retry"
            );
            clean->finalize( adaptation_status::accepted, .05, *reference );
            step.finalize( adaptation_status::accepted, .05, *result );
        }
        else
            require(
                step.get_status() == ( scenario == 2   ? single_step_status::failed_solver
                                       : scenario == 3 ? single_step_status::attempt_limit_reached
                                                       : single_step_status::failed_minimum_dt ) &&
                    merit->calls == ( scenario == 4 ? 2u : 0u ) && operations->get_value_at_point( 0, *result ) == 42,
                "Terminal pseudo-time failure published a candidate or assessed a failed stage"
            );
        require( operations->get_value_at_point( 0, *initial ) == 2, "Pseudo-time failure modified committed input" );
    }
}

int main()
try
{
    backend_type::init_device();
    test_policy();
    test_configuration();
    test_equilibrium<scalar_time_equation<cubic_relaxation>>( false );
    test_equilibrium<constrained_relaxation>( true );
    test_recovery();
    std::cout << "Pseudo-time globalization, steady ODE/DAE, recovery and rollback: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
