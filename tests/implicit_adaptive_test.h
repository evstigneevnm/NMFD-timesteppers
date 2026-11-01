#ifndef NMFD_TIME_STEPPERS_TESTS_IMPLICIT_ADAPTIVE_H
#define NMFD_TIME_STEPPERS_TESTS_IMPLICIT_ADAPTIVE_H

#include "implicit_test_support.h"
#include <nmfd/time_steppers/integration/time_step_adaptation_matlab.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_scipy.h>

namespace nmfd
{
namespace time_steppers
{
namespace tests
{
template <class Controller>
struct observed_controller : Controller
{
    using Controller::Controller;
    unsigned int assessments = 0, error_rejections = 0, numerical_rejections = 0, commits = 0;

    void reset()
    {
        Controller::reset();
        assessments = error_rejections = numerical_rejections = commits = 0;
    }
    adaptation_status assess(
        double t, double h, const vector_type &in, const vector_type &out, double &next, unsigned int order = 0,
        const vector_type *error = nullptr
    )
    {
        require( error && order == 2, "DIRK assessment lost its state error or scaling exponent" );
        ++assessments;
        const auto result = Controller::assess( t, h, in, out, next, order, error );
        error_rejections += result == adaptation_status::rejected;
        return result;
    }
    adaptation_status reject_step( double h )
    {
        ++numerical_rejections;
        return Controller::reject_step( h );
    }
    void update( adaptation_status outcome, double t, double h, const vector_type &state )
    {
        commits += outcome == adaptation_status::accepted || outcome == adaptation_status::accepted_modified;
        Controller::update( outcome, t, h, state );
    }
};

template <class Controller>
typename Controller::params controller_parameters( double tolerance = 1e-4 )
{
    typename Controller::params p;
    p.relative_tolerance = tolerance;
    p.absolute_tolerance = tolerance * .01;
    p.initial_step       = .3;
    p.minimum_step       = 1e-8;
    p.maximum_step       = .4;
    return p;
}

// Compare finite-time endpoints, not chaotic long-time trajectories or formal DAE order.
// External monitors independently check algebraic constraints on every accepted step.
template <class Controller, class Equation, class Monitor>
double test_adaptive_trajectory(
    std::shared_ptr<operations_type> operations, const Equation &equation,
    const scfd::static_vec::vec<double, Equation::size> &initial_values,
    const scfd::static_vec::vec<double, Equation::size> &reference, double start, double end, const char *label,
    const char *controller_name, const char *method, std::shared_ptr<Monitor> monitor
)
{
    using problem_type = stage_problem<Equation>;
    using types        = newton_types<problem_type>;
    using adaptation   = observed_controller<Controller>;
    using step_type =
        runge_kutta::implicit_time_step<operations_type, problem_type, adaptation, typename types::solver_type>;
    auto             problem = std::make_shared<problem_type>( operations, equation );
    vector_wrap_type initial( *operations ), result( *operations );
    initial.start_use();
    result.start_use();
    operations->set( initial_values.d, *initial, Equation::size );
    double      errors[2]{}, reference_scale = 1;
    std::size_t steps[2]{};
    for ( int i = 0; i < Equation::size; ++i )
        reference_scale = std::max( reference_scale, std::abs( reference[i] ) );
    for ( int level = 0; level < 2; ++level )
    {
        auto p         = controller_parameters<Controller>( level ? 2e-5 : 2e-3 );
        p.initial_step = p.maximum_step = std::abs( end - start );
        auto control                    = std::make_shared<adaptation>( operations, p );
        auto step                       = std::make_shared<step_type>(
            operations, problem, control, types::make_solver( operations ), typename step_type::params{ method }
        );
        integration::time_integrator<operations_type, step_type, Monitor> integrator(
            operations, step, { 10000 }, monitor
        );
        integrator.set_time_interval( start, end );
        typename problem_type::state_type first;
        unsigned int first_error_rejections = 0, first_numerical_rejections = 0, first_linearizations = 0;
        for ( int run = 0; run < 2; ++run )
        {
            unsigned int monitor_calls = 0;
            if constexpr ( !std::is_same_v<Monitor, detail::no_external_operations> )
                monitor_calls = monitor->calls;
            const auto linearizations = problem->linearizations;
            integrator.apply( *initial, *result );
            if ( integrator.get_status() != integration_status::completed )
                std::cerr << label << ' ' << controller_name << ' ' << method << " tol=" << p.relative_tolerance
                          << " stopped at " << integrator.get_final_time() << " dt=" << step->get_dt()
                          << " step_status=" << static_cast<int>( step->get_status() )
                          << " numerical_rejections=" << control->numerical_rejections << '\n';
            require(
                integrator.get_status() == integration_status::completed && integrator.get_final_time() == end,
                "Adaptive system failed to reach its exact target"
            );
            require(
                control->commits == integrator.get_steps() &&
                    control->assessments == control->commits + control->error_rejections,
                "Adaptive system committed a rejected or unassessed candidate"
            );
            if constexpr ( !std::is_same_v<Monitor, detail::no_external_operations> )
                require(
                    monitor->calls - monitor_calls == integrator.get_steps(),
                    "DAE monitor missed an accepted endpoint or saw a rejected trial"
                );
            typename problem_type::state_type actual, unchanged;
            operations->get( *result, actual.d, Equation::size );
            operations->get( *initial, unchanged.d, Equation::size );
            for ( int i = 0; i < Equation::size; ++i )
            {
                require( unchanged[i] == initial_values[i], "Adaptive system changed its initial state" );
                require( std::isfinite( actual[i] ), "Adaptive system published a nonfinite endpoint" );
                errors[level] = std::max( errors[level], std::abs( actual[i] - reference[i] ) );
                if ( run )
                    require( actual[i] == first[i], "Adaptive system restart changed its trajectory" );
            }
            if ( !run )
            {
                first                      = actual;
                steps[level]               = integrator.get_steps();
                first_error_rejections     = control->error_rejections;
                first_numerical_rejections = control->numerical_rejections;
                first_linearizations       = problem->linearizations - linearizations;
            }
            else
                require(
                    steps[level] == integrator.get_steps() && first_error_rejections == control->error_rejections &&
                        first_numerical_rejections == control->numerical_rejections &&
                        first_linearizations == problem->linearizations - linearizations,
                    "Adaptive system restart retained controller or solver history"
                );
        }
        if ( level )
            require( control->error_rejections > 0, "Adaptive system did not exercise embedded-error rejection" );
        std::cout << label << ' ' << controller_name << ' ' << method << " tol=" << p.relative_tolerance
                  << " error=" << errors[level] << " steps=" << steps[level]
                  << " error_rejections=" << control->error_rejections
                  << " numerical_rejections=" << control->numerical_rejections
                  << " linearizations=" << first_linearizations << '\n';
    }
    // Tight endpoint errors can reach the nonlinear-solve floor, especially after stiff decay.
    require(
        errors[1] < std::max( 5e-9, .3 * errors[0] ) && steps[1] > steps[0],
        "Adaptive system accuracy did not improve with tighter tolerances"
    );
    require( errors[1] < 2e-5 * reference_scale, "Adaptive system reference mismatch" );
    return errors[1];
}

template <class Equation, class Monitor = detail::no_external_operations>
double test_adaptive_system(
    const Equation &equation, const scfd::static_vec::vec<double, Equation::size> &initial,
    const scfd::static_vec::vec<double, Equation::size> &reference, double start, double end, const char *label,
    std::shared_ptr<operations_type> operations = nullptr, std::shared_ptr<Monitor> monitor = nullptr
)
{
    if ( !operations )
        operations = std::make_shared<operations_type>( Equation::size );
    double maximum_error = 0;
    for ( const char *method : { "SDIRK2(1)2", "SDIRK3(1)3" } )
    {
        maximum_error = std::max(
            maximum_error, test_adaptive_trajectory<integration::time_step_adaptation_matlab<operations_type>>(
                               operations, equation, initial, reference, start, end, label, "MATLAB", method, monitor
                           )
        );
        maximum_error = std::max(
            maximum_error, test_adaptive_trajectory<integration::time_step_adaptation_scipy<operations_type>>(
                               operations, equation, initial, reference, start, end, label, "SciPy", method, monitor
                           )
        );
    }
    return maximum_error;
}
}
}
}
#endif
