#include "implicit_rosenbrock_mass_problem.h"
#include <nmfd/time_steppers/integration/time_step_adaptation_filter.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_matlab.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_scipy.h>

using namespace rosenbrock_tests;

// Independent logistic components: M*u'=M*u*(1-u), u(0)=(1/4,1/4).
// The coupled, nonsymmetric M does not change the exact solution. Error
// refinement and deterministic restarts must hold for every controller family.
template <class Controller>
void trajectory( const char *method, const char *filter = nullptr )
{
    using P                 = mass_problem<logistic>;
    using L                 = direct_solver<P>;
    using S                 = rosenbrock::time_step<operations_type, P, Controller, L, true>;
    auto             ops    = std::make_shared<operations_type>( 2 );
    auto             p      = std::make_shared<P>( ops );
    auto             linear = std::make_shared<L>( ops );
    vector_wrap_type in( *ops ), out( *ops );
    in.start_use();
    out.start_use();
    ops->assign_scalar( .25, *in );
    double      errors[2]{};
    std::size_t counts[2]{};
    for ( unsigned int level = 0; level < 2; ++level )
    {
        typename Controller::params cp;
        cp.relative_tolerance = level ? 1e-5 : 1e-3;
        cp.absolute_tolerance = cp.relative_tolerance * .01;
        cp.initial_step = cp.maximum_step = .5;
        if constexpr ( std::is_same_v<Controller, integration::time_step_adaptation_filter<operations_type>> )
            cp.controller = filter;
        typename S::params sp;
        sp.method    = method;
        sp.mass_form = true;
        auto step    = std::make_shared<S>( ops, p, std::make_shared<Controller>( ops, cp ), linear, sp );
        integration::time_integrator<operations_type, S> run( ops, step );
        run.set_time_interval( .37, 1.37 );
        double first = 0;
        for ( unsigned int restart = 0; restart < 2; ++restart )
        {
            run.apply( *in, *out );
            check(
                run.get_status() == integration_status::completed && run.get_final_time() == 1.37,
                "Adaptive mass-form integration failed"
            );
            const double result = ops->get_value_at_point( 0, *out );
            if ( !restart )
            {
                first         = result;
                counts[level] = run.get_steps();
            }
            else
                check(
                    result == first && counts[level] == run.get_steps(), "Mass-form restart retained trajectory history"
                );
            errors[level] = std::abs( result - 1 / ( 1 + 3 * std::exp( -1. ) ) );
            check( ops->get_value_at_point( 0, *in ) == .25, "Mass-form adaptation changed the committed input" );
        }
    }
    check(
        errors[1] < errors[0] * .25 && errors[1] < 1e-4 && counts[1] > counts[0],
        "Mass-form tolerance refinement failed"
    );
}

// A numerical failure, including dense-only work, must shrink and retry from
// the same input. The successful retry must bind a freshly shifted operator.
void numerical_retry( const char *method )
{
    using P              = mass_problem<logistic>;
    using L              = direct_solver<P>;
    using C              = integration::time_step_adaptation_scipy<operations_type>;
    using S              = rosenbrock::time_step<operations_type, P, C, L, true>;
    auto             ops = std::make_shared<operations_type>( 2 );
    vector_wrap_type in( *ops ), out( *ops );
    in.start_use();
    out.start_use();
    ops->assign_scalar( .25, *in );
    for ( bool invalid_residual : { false, true } )
    {
        auto      p      = std::make_shared<P>( ops );
        auto      solver = std::make_shared<L>( ops );
        C::params cp;
        cp.relative_tolerance = cp.absolute_tolerance = .1;
        cp.initial_step = cp.maximum_step = .05;
        S::params sp;
        sp.method    = method;
        sp.mass_form = true;
        if ( invalid_residual )
            p->invalid_on = rosenbrock::method_table( method ).stage_count() == 2
                                ? 2
                                : rosenbrock::method_table( method ).stage_count() + 1;
        else
            solver->fail_on = 1;
        S step( ops, p, std::make_shared<C>( ops, cp ), solver, sp );
        step.apply( *in, *out );
        check(
            step.get_status() == single_step_status::converged && step.get_attempts() > 1 && step.get_dt() < .05,
            "Mass-form numerical failure did not shrink/retry"
        );
        check(
            p->linearizations == step.get_attempts() && solver->bindings == step.get_attempts(),
            "Mass-form retry reused a failed shift"
        );
        check(
            ops->get_value_at_point( 0, *in ) == .25 && ops->get_value_at_point( 0, *out ) > .25,
            "Mass-form retry lost its original input"
        );
    }
}

int main()
try
{
    backend_type::init_device();
    for ( const char *method : { "ROS2", "ROS3", "ROS34PW2" } )
    {
        numerical_retry( method );
        trajectory<integration::time_step_adaptation_matlab<operations_type>>( method );
        trajectory<integration::time_step_adaptation_scipy<operations_type>>( method );
        for ( const char *filter : { "I", "H211", "PC", "PID", "H312", "PPID", "H321" } )
            trajectory<integration::time_step_adaptation_filter<operations_type>>( method, filter );
    }
    std::cout << "PASS: adaptive Rosenbrock constant mass (" << checks << " checks)\n";
}
catch ( const std::exception &e )
{
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
