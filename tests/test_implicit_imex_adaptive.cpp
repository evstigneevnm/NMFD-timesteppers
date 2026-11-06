#include "implicit_imex_adaptive_test.h"

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;

// Split u'=lambda_E*u+lambda_I*u=-u. Independently assemble scalar RK stages
// to check the signed combined defect, primary local order four and embedded
// local order three, including pure-explicit/pure-implicit limiting cases.
void test_combined_estimator()
{
    using partition      = rate_problem<imex_linear_equation>;
    using problem_type   = imex_problem<partition, partition>;
    using step_type      = imex_step<problem_type, imex_error_capture>;
    auto             ops = std::make_shared<operations_type>( 1 );
    vector_wrap_type initial( *ops ), out( *ops );
    initial.start_use();
    out.start_use();
    ops->assign_scalar( 1., *initial );
    for ( const double le : { .5, 0., -1. } )
    {
        const double li      = -1 - le;
        auto         problem = std::make_shared<problem_type>(
            std::make_shared<partition>( ops, imex_linear_equation{ le } ),
            std::make_shared<partition>( ops, imex_linear_equation{ li } )
        );
        double primary[3]{}, embedded[3]{};
        for ( int level = 0; level < 3; ++level )
        {
            const double h       = .2 / ( 1 << level );
            auto         control = std::make_shared<imex_error_capture>( ops, h );
            step_type    step(
                ops, problem, control, newton_types<partition>::make_solver( ops ),
                step_type::params{ "IMEX_ARK324L2SA" }
            );
            step.set_time( .3 );
            step.set_target_time( .3 + h );
            step.apply( *initial, *out );
            const auto &pair = step.table();
            require(
                step.get_status() == single_step_status::converged && pair.is_embedded() &&
                    pair.embedded_order() == 2 && control->exponent == 3,
                "IMEX lost its combined estimator metadata or controller exponent"
            );
            long double stages[4]{}, expected = 1, expected_embedded = 1;
            for ( std::size_t j = 0; j < 4; ++j )
            {
                long double reference = 1;
                for ( std::size_t k = 0; k < j; ++k )
                    reference +=
                        h * ( le * pair.explicit_table.a( j, k ) + li * pair.implicit_table.a( j, k ) ) * stages[k];
                stages[j] = reference / ( 1 - h * li * pair.implicit_table.a( j, j ) );
                expected += h * ( le * pair.explicit_table.b( j ) + li * pair.implicit_table.b( j ) ) * stages[j];
                expected_embedded +=
                    h * ( le * pair.explicit_table.embedded_b( j ) + li * pair.implicit_table.embedded_b( j ) ) *
                    stages[j];
            }
            const double value = ops->get_value_at_point( 0, *out );
            require(
                std::abs( value - expected ) < 2e-12 &&
                    std::abs( control->error - ( expected - expected_embedded ) ) < 2e-12,
                "IMEX controller did not receive the signed sum of both partition defects"
            );
            primary[level]  = std::abs( value - std::exp( -step.get_dt() ) );
            embedded[level] = std::abs( value - control->error - std::exp( -step.get_dt() ) );
            step.finalize( adaptation_status::accepted, .3 + step.get_dt(), *out );
        }
        const double primary_order  = std::log2( primary[1] / primary[2] );
        const double embedded_order = std::log2( embedded[1] / embedded[2] );
        std::cout << "ARK IMEX lambda_E=" << le << " local orders=" << primary_order << ',' << embedded_order << '\n';
        require(
            std::abs( primary_order - 4 ) < .4 && std::abs( embedded_order - 3 ) < .4,
            "Combined IMEX primary/embedded local order mismatch"
        );
    }
}

// f_E=.25*u^2+cos(t), f_I=-u-f_E. Total u'=-u and exact exp(-t).
// Nonlinear/time-dependent cancellation tests the combined estimate, not each partition in isolation.
struct nonlinear_rhs
{
    bool                  implicit = false;
    __DEVICE_TAG__ double mass( double ) const
    {
        return 1;
    }
    __DEVICE_TAG__ double rhs( double t, double u ) const
    {
        const double e = .25 * u * u + std::cos( t );
        return implicit ? -u - e : e;
    }
    __DEVICE_TAG__ double rhs_derivative( double, double u ) const
    {
        return implicit ? -1 - .5 * u : .5 * u;
    }
};

template <class Controller, class Equation>
void test_controller(
    const char *label, Equation e, Equation i, typename Controller::params settings = {}, bool backward = false
)
{
    using partition    = rate_problem<Equation>;
    using problem_type = imex_problem<partition, partition>;
    using control_type = observed_controller<Controller>;
    using step_type    = imex_step<problem_type, control_type>;
    auto ops           = std::make_shared<operations_type>( 1 );
    auto problem =
        std::make_shared<problem_type>( std::make_shared<partition>( ops, e ), std::make_shared<partition>( ops, i ) );
    vector_wrap_type initial( *ops ), out( *ops );
    initial.start_use();
    out.start_use();
    const double start = backward ? 1.25 : .25, end = backward ? .25 : 1.25;
    ops->assign_scalar( std::exp( -start ), *initial );
    double      errors[2]{};
    std::size_t steps[2]{};
    for ( int level = 0; level < 2; ++level )
    {
        auto p               = settings;
        p.relative_tolerance = level ? 2e-5 : 5e-3;
        p.absolute_tolerance = p.relative_tolerance * .01;
        p.initial_step = p.maximum_step = 1;
        p.minimum_step                  = 1e-8;
        auto control                    = std::make_shared<control_type>( ops, p );
        control->expected_order         = 3;
        auto step                       = std::make_shared<step_type>(
            ops, problem, control, newton_types<partition>::make_solver( ops ),
            typename step_type::params{ "IMEX_ARK324L2SA" }
        );
        integration::time_integrator<operations_type, step_type> flow( ops, step );
        flow.set_time_interval( start, end );
        double       first            = 0;
        unsigned int first_rejections = 0;
        for ( int run = 0; run < 2; ++run )
        {
            flow.apply( *initial, *out );
            require(
                flow.get_status() == integration_status::completed && flow.get_final_time() == end,
                "Adaptive IMEX failed to reach its clipped endpoint"
            );
            require(
                control->commits == flow.get_steps() &&
                    control->assessments == control->commits + control->error_rejections,
                "Adaptive IMEX committed a rejected or unassessed candidate"
            );
            const auto actual = ops->get_value_at_point( 0, *out );
            require( ops->get_value_at_point( 0, *initial ) == std::exp( -start ), "Adaptive IMEX changed its input" );
            if ( run )
                require(
                    flow.get_steps() == steps[level] && control->error_rejections == first_rejections &&
                        std::abs( actual - first ) < 1e-13,
                    "Adaptive IMEX restart retained controller or method history"
                );
            first            = actual;
            first_rejections = control->error_rejections;
            errors[level]    = std::abs( actual - std::exp( -end ) );
            steps[level]     = flow.get_steps();
        }
        if ( level )
            require( control->error_rejections > 0, "Adaptive IMEX did not exercise LTE rejection" );
    }
    std::cout << label << " direction=" << ( backward ? -1 : 1 ) << " errors=" << errors[0] << " -> " << errors[1]
              << " steps=" << steps[0] << " -> " << steps[1] << '\n';
    require(
        errors[1] < std::max( 5e-9, .3 * errors[0] ) && errors[1] < 2e-4 && steps[1] > steps[0],
        "Adaptive IMEX accuracy did not improve with tighter tolerances"
    );
}

// Stiff tracking: f_E=kappa*sin(t)+cos(t), f_I=-kappa*u; exact sin(t).
// Check adaptive accuracy in the stiff regime without assuming uniform classical order.
struct tracking_rhs
{
    double                kappa = 100;
    __DEVICE_TAG__ double mass( double ) const
    {
        return 1;
    }
    __DEVICE_TAG__ double rhs( double t, double ) const
    {
        return kappa * std::sin( t ) + std::cos( t );
    }
    __DEVICE_TAG__ double rhs_derivative( double, double ) const
    {
        return 0;
    }
};

void test_stiff_tracking()
{
    using explicit_type  = rate_problem<rate_scalar_equation<tracking_rhs>>;
    using implicit_type  = rate_problem<imex_linear_equation>;
    using problem_type   = imex_problem<explicit_type, implicit_type>;
    using controller     = observed_controller<integration::time_step_adaptation_scipy<operations_type>>;
    using step_type      = imex_step<problem_type, controller>;
    auto             ops = std::make_shared<operations_type>( 1 );
    vector_wrap_type initial( *ops ), out( *ops );
    initial.start_use();
    out.start_use();
    ops->assign_scalar( std::sin( .25 ), *initial );
    for ( const double kappa : { 100., 10000. } )
    {
        auto problem = std::make_shared<problem_type>(
            std::make_shared<explicit_type>( ops, rate_scalar_equation<tracking_rhs>{ tracking_rhs{ kappa } } ),
            std::make_shared<implicit_type>( ops, imex_linear_equation{ -kappa } )
        );
        double errors[2]{};
        for ( int level = 0; level < 2; ++level )
        {
            auto p         = controller_parameters<controller>( level ? 1e-5 : 1e-3 );
            p.initial_step = p.maximum_step = .4;
            auto control                    = std::make_shared<controller>( ops, p );
            control->expected_order         = 3;
            auto step                       = std::make_shared<step_type>(
                ops, problem, control, newton_types<implicit_type>::make_solver( ops ),
                step_type::params{ "IMEX_ARK324L2SA" }
            );
            integration::time_integrator<operations_type, step_type> flow( ops, step );
            flow.set_time_interval( .25, .65 );
            flow.apply( *initial, *out );
            require( flow.get_status() == integration_status::completed, "Stiff adaptive IMEX tracking failed" );
            errors[level] = std::abs( ops->get_value_at_point( 0, *out ) - std::sin( .65 ) );
        }
        std::cout << "Stiff IMEX kappa=" << kappa << " errors=" << errors[0] << " -> " << errors[1] << '\n';
        require(
            errors[1] < std::max( 1e-8, .5 * errors[0] ) && errors[1] < 1e-4,
            "Stiff adaptive IMEX failed accuracy/refinement"
        );
    }
}

int main()
try
{
    backend_type::init_device();
    test_combined_estimator();
    using matlab = integration::time_step_adaptation_matlab<operations_type>;
    using scipy  = integration::time_step_adaptation_scipy<operations_type>;
    using filter = integration::time_step_adaptation_filter<operations_type>;
    test_controller<matlab>( "MATLAB", imex_linear_equation{ .5 }, imex_linear_equation{ -1.5 } );
    test_controller<scipy>( "SciPy", imex_linear_equation{ .5 }, imex_linear_equation{ -1.5 } );
    for ( const char *name : { "I", "H211", "PC", "PID", "H312", "PPID", "H321" } )
    {
        filter::params p;
        p.controller = name;
        test_controller<filter>( name, imex_linear_equation{ .5 }, imex_linear_equation{ -1.5 }, p );
    }
    using nonlinear_equation = rate_scalar_equation<nonlinear_rhs>;
    test_controller<matlab>(
        "Nonlinear MATLAB", nonlinear_equation{ nonlinear_rhs{ false } }, nonlinear_equation{ nonlinear_rhs{ true } }
    );
    test_controller<scipy>(
        "Nonlinear SciPy", nonlinear_equation{ nonlinear_rhs{ false } }, nonlinear_equation{ nonlinear_rhs{ true } }
    );
    test_controller<scipy>( "Backward SciPy", imex_linear_equation{ .5 }, imex_linear_equation{ -1.5 }, {}, true );
    filter::params p;
    p.controller = "PID";
    test_controller<filter>( "Backward PID", imex_linear_equation{ .5 }, imex_linear_equation{ -1.5 }, p, true );
    test_stiff_tracking();
    std::cout << "Adaptive IMEX combined estimator, orders, controllers, stiff/timed coupling and restart: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
