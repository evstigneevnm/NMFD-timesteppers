#include "implicit_rate_test.h"
#include "implicit_adaptive_test.h"
#include "nonautonomous_test_problems.h"

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;

// u'=t*u, u(.25)=exp(.25^2/2): exact exp(t^2/2). This non-autonomous
// manufactured solution checks both primary and embedded orders independently.
// A second case, (1+t)*u'=-u with exact 1/(1+t), checks adaptive mass solves.
using equation_type = rate_scalar_equation<coefficient_growth<double>>;
using problem_type  = rate_problem<equation_type>;
using types         = newton_types<problem_type>;

struct captured_error : adaptation_type
{
    std::shared_ptr<operations_type> operations;
    double                           error    = 0;
    unsigned int                     exponent = 0;
    captured_error( std::shared_ptr<operations_type> ops, double h )
        : adaptation_type( params{ h } ), operations( std::move( ops ) )
    {
    }
    bool requires_error_estimate() const
    {
        return true;
    }
    adaptation_status assess(
        double, double, const vector_type &, const vector_type &, double &next, unsigned int order = 0,
        const vector_type *estimate = nullptr
    )
    {
        require( estimate, "Embedded estimate is missing" );
        error    = operations->get_value_at_point( 0, *estimate );
        exponent = order;
        next     = get_dt();
        return adaptation_status::accepted;
    }
};

void test_orders()
{
    auto             operations = std::make_shared<operations_type>( 1 );
    auto             problem    = std::make_shared<problem_type>( operations );
    vector_wrap_type initial( *operations ), out( *operations );
    initial.start_use();
    out.start_use();
    const auto exact = []( double t ) { return std::exp( t * t / 2 ); };
    operations->assign_scalar( exact( .25 ), *initial );
    for ( const char *method : { "CN", "ESDIRK2(1)3", "ARK324L2SA" } )
    {
        operations->assign_scalar( exact( .25 ), *initial );
        double errors[2]{};
        for ( int level = 0; level < 2; ++level )
        {
            auto step = make_step( operations, problem, .1 / ( 1 << level ), method );
            integration::time_integrator<operations_type, types::step_type> integrator( operations, step );
            integrator.set_time_interval( .25, 1.25 );
            integrator.apply( *initial, *out );
            require( integrator.get_status() == integration_status::completed, "ESDIRK order run failed" );
            errors[level] = std::abs( operations->get_value_at_point( 0, *out ) - exact( 1.25 ) );
        }
        const auto observed = std::log2( errors[0] / errors[1] );
        std::cout << method << " global order=" << observed << ", errors=" << errors[0] << " -> " << errors[1] << '\n';
        require(
            std::abs( observed - runge_kutta::make_butcher_table( method ).order() ) < .3,
            "ESDIRK primary global order mismatch"
        );
        if ( std::string( method ) == "CN" )
            continue;

        using step_type =
            runge_kutta::implicit_time_step<operations_type, problem_type, captured_error, types::solver_type>;
        double       primary[3]{}, embedded[3]{};
        const double local_start = 1;
        operations->assign_scalar( exact( local_start ), *initial );
        for ( int level = 0; level < 3; ++level )
        {
            const double h       = .2 / ( 1 << level );
            auto         control = std::make_shared<captured_error>( operations, h );
            step_type    step(
                operations, problem, control, types::make_solver( operations ), step_type::params{ method }
            );
            step.set_time( local_start );
            step.set_target_time( local_start + h );
            step.apply( *initial, *out );
            require(
                step.get_status() == single_step_status::converged && control->exponent == step.table().error_order(),
                "Embedded ESDIRK stage or controller exponent failed"
            );
            const auto value = operations->get_value_at_point( 0, *out );
            primary[level]   = std::abs( value - exact( local_start + h ) );
            embedded[level]  = std::abs( value - control->error - exact( local_start + h ) );
            step.finalize( adaptation_status::accepted, local_start + h, *out );
        }
        const auto primary_order  = std::log2( primary[1] / primary[2] );
        const auto embedded_order = std::log2( embedded[1] / embedded[2] );
        const auto table          = runge_kutta::make_butcher_table( method );
        std::cout << method << " local orders=" << primary_order << ", " << embedded_order << '\n';
        require(
            std::abs( primary_order - ( table.order() + 1 ) ) < .5 &&
                std::abs( embedded_order - ( table.embedded_order() + 1 ) ) < .5,
            "ESDIRK primary/embedded local orders mismatch"
        );
    }
}

template <class Controller, class Equation>
void test_controller( const char *label, const Equation &equation, typename Controller::params settings = {} )
{
    using equation_type = rate_scalar_equation<Equation>;
    using problem_type  = rate_problem<equation_type>;
    using solver_types  = newton_types<problem_type>;
    using adaptation    = observed_controller<Controller>;
    using step_type =
        runge_kutta::implicit_time_step<operations_type, problem_type, adaptation, typename solver_types::solver_type>;
    auto             operations = std::make_shared<operations_type>( 1 );
    auto             problem    = std::make_shared<problem_type>( operations, equation_type{ equation } );
    vector_wrap_type initial( *operations ), out( *operations );
    initial.start_use();
    out.start_use();
    operations->assign_scalar( equation.exact( .25 ), *initial );
    for ( const char *method : { "ESDIRK2(1)3", "ARK324L2SA" } )
    {
        double      errors[2]{};
        std::size_t steps[2]{};
        for ( int level = 0; level < 2; ++level )
        {
            auto p               = settings;
            p.relative_tolerance = level ? 2e-5 : 5e-3;
            p.absolute_tolerance = p.relative_tolerance * .01;
            p.initial_step = p.maximum_step = 1;
            p.minimum_step                  = 1e-8;
            auto control                    = std::make_shared<adaptation>( operations, p );
            control->expected_order         = runge_kutta::make_butcher_table( method ).error_order();
            auto step                       = std::make_shared<step_type>(
                operations, problem, control, solver_types::make_solver( operations ),
                typename step_type::params{ method }
            );
            integration::time_integrator<operations_type, step_type> integrator( operations, step, { 10000 } );
            integrator.set_time_interval( .25, 1.25 );
            double first = 0;
            for ( int run = 0; run < 2; ++run )
            {
                integrator.apply( *initial, *out );
                require(
                    integrator.get_status() == integration_status::completed && integrator.get_final_time() == 1.25,
                    "Adaptive ESDIRK did not reach its target"
                );
                require(
                    control->commits == integrator.get_steps() &&
                        control->assessments == control->commits + control->error_rejections,
                    "Adaptive ESDIRK committed an unassessed/rejected candidate"
                );
                const auto actual = operations->get_value_at_point( 0, *out );
                if ( run )
                    require(
                        integrator.get_steps() == steps[level] && std::abs( actual - first ) < 1e-13,
                        "Adaptive ESDIRK restart retained history"
                    );
                first         = actual;
                steps[level]  = integrator.get_steps();
                errors[level] = std::abs( actual - equation.exact( 1.25 ) );
                require(
                    operations->get_value_at_point( 0, *initial ) == equation.exact( .25 ),
                    "Adaptive ESDIRK changed its input"
                );
            }
            if ( level )
                require( control->error_rejections > 0, "Adaptive ESDIRK did not exercise rejection" );
        }
        std::cout << label << ' ' << method << " errors=" << errors[0] << " -> " << errors[1] << ", steps=" << steps[0]
                  << " -> " << steps[1] << '\n';
        require(
            errors[1] < std::max( 5e-9, .3 * errors[0] ) && errors[1] < 2e-4 && steps[1] > steps[0],
            "Adaptive ESDIRK accuracy did not improve"
        );
    }
}

// Stiff tracking u'=-kappa*(u-sin(t))+cos(t), exact sin(t). Check accuracy
// and refinement, not uniform classical order in the stiff limit.
void test_stiff_tracking()
{
    using equation_type         = rate_scalar_equation<stiff_tracking<double>>;
    using problem_type          = rate_problem<equation_type>;
    using types                 = newton_types<problem_type>;
    auto             operations = std::make_shared<operations_type>( 1 );
    vector_wrap_type initial( *operations ), out( *operations );
    initial.start_use();
    out.start_use();
    operations->assign_scalar( std::sin( .25 ), *initial );
    for ( const double kappa : { 100., 10000. } )
    {
        auto problem = std::make_shared<problem_type>( operations, equation_type{ stiff_tracking<double>{ kappa } } );
        for ( const char *method : { "CN", "ESDIRK2(1)3", "ARK324L2SA" } )
        {
            double errors[2]{};
            for ( int level = 0; level < 2; ++level )
            {
                auto step = make_step( operations, problem, .02 / ( 1 << level ), method );
                integration::time_integrator<operations_type, types::step_type> integrator( operations, step );
                integrator.set_time_interval( .25, .65 );
                integrator.apply( *initial, *out );
                require( integrator.get_status() == integration_status::completed, "Stiff ESDIRK tracking failed" );
                errors[level] = std::abs( operations->get_value_at_point( 0, *out ) - std::sin( .65 ) );
            }
            std::cout << method << " kappa=" << kappa << " tracking errors=" << errors[0] << " -> " << errors[1]
                      << '\n';
            require(
                errors[1] < std::max( 5e-9, .8 * errors[0] ) && errors[1] < 1e-4, "Stiff ESDIRK refinement failed"
            );
        }
    }
}

int main()
try
{
    backend_type::init_device();
    test_orders();
    test_stiff_tracking();
    using matlab = integration::time_step_adaptation_matlab<operations_type>;
    using scipy  = integration::time_step_adaptation_scipy<operations_type>;
    using filter = integration::time_step_adaptation_filter<operations_type>;
    test_controller<matlab>( "MATLAB growth", coefficient_growth<double>{} );
    test_controller<scipy>( "SciPy growth", coefficient_growth<double>{} );
    for ( const char *name : { "I", "H211", "PC", "PID", "H312", "PPID", "H321" } )
    {
        filter::params p;
        p.controller = name;
        test_controller<filter>( name, coefficient_growth<double>{}, p );
    }
    test_controller<matlab>( "MATLAB time-dependent mass", time_mass_decay<double>{} );
    test_controller<scipy>( "SciPy time-dependent mass", time_mass_decay<double>{} );
    filter::params p;
    p.controller = "PID";
    test_controller<filter>( "PID time-dependent mass", time_mass_decay<double>{}, p );
    std::cout << "ESDIRK primary/embedded orders, MATLAB/SciPy/filter adaptation and restart: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
