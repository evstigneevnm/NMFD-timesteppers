#ifndef NMFD_TIME_STEPPERS_TESTS_IMPLICIT_BDF_ADAPTIVE_H
#define NMFD_TIME_STEPPERS_TESTS_IMPLICIT_BDF_ADAPTIVE_H

#include "implicit_bdf_test.h"
#include "implicit_adaptive_test.h"

namespace nmfd::time_steppers::tests::bdf_test
{
template <class Controller>
struct observed_bdf_controller : observed_controller<Controller>
{
    using base = observed_controller<Controller>;
    using base::base;
    unsigned int orders = 0, committed_orders = 0, pending_order = 0;
    void         reset()
    {
        base::reset();
        orders = committed_orders = pending_order = 0;
    }
    adaptation_status assess(
        double t, double h, const vector_type &in, const vector_type &out, double &next, unsigned int order = 0,
        const vector_type *error = nullptr
    )
    {
        require( order >= 2 && order <= 6, "BDF lost its local-error scaling exponent" );
        orders |= 1u << order;
        pending_order        = order - 1;
        this->expected_order = order;
        return base::assess( t, h, in, out, next, order, error );
    }
    void update( adaptation_status outcome, double time, double h, const vector_type &state )
    {
        if ( outcome == adaptation_status::accepted || outcome == adaptation_status::accepted_modified )
            committed_orders |= 1u << pending_order;
        base::update( outcome, time, h, state );
    }
};

template <class Controller, class Equation, class Monitor = detail::no_external_operations>
double adaptive_trajectory(
    const Equation &equation, const scfd::static_vec::vec<double, Equation::size> &initial,
    const scfd::static_vec::vec<double, Equation::size> &reference, double start, double end, const char *method,
    const char *controller_name, double tolerance, std::shared_ptr<operations_type> ops,
    std::shared_ptr<Monitor> monitor = nullptr, unsigned int maximum_order = 5
)
{
    using problem_type = stage_problem<Equation>;
    using control_type = observed_bdf_controller<Controller>;
    using step_type    = bdf::implicit_time_step<
           operations_type, problem_type, control_type, typename newton_types<problem_type>::solver_type>;
    auto problem          = std::make_shared<problem_type>( ops, equation );
    auto settings         = controller_parameters<Controller>( tolerance );
    settings.initial_step = settings.maximum_step = std::abs( end - start );
    if constexpr ( std::is_same_v<Controller, integration::time_step_adaptation_filter<operations_type>> )
        settings.controller = controller_name;
    auto                       control = std::make_shared<control_type>( ops, settings );
    typename step_type::params p{ method };
    p.adaptive      = true;
    p.maximum_order = maximum_order;
    auto step = std::make_shared<step_type>( ops, problem, control, newton_types<problem_type>::make_solver( ops ), p );
    integration::time_integrator<operations_type, step_type, Monitor> flow( ops, step, { 20000 }, monitor );
    flow.set_time_interval( start, end );
    vector_wrap_type state( *ops ), out( *ops );
    state.start_use();
    out.start_use();
    ops->set( initial.d, *state, Equation::size );
    double                            error      = 0;
    std::size_t                       steps      = 0;
    unsigned int                      rejections = 0;
    typename problem_type::state_type first;
    for ( int run = 0; run < 2; ++run )
    {
        flow.apply( *state, *out );
        if ( flow.get_status() != integration_status::completed )
            std::cerr << method << ' ' << controller_name << " tolerance=" << tolerance
                      << " time=" << flow.get_final_time() << " h=" << step->get_dt()
                      << " status=" << int( step->get_status() ) << '\n';
        require(
            flow.get_status() == integration_status::completed && flow.get_final_time() == end,
            "Adaptive BDF failed to reach its target"
        );
        require(
            control->assessments == control->commits + control->error_rejections &&
                control->commits == flow.get_steps(),
            "Adaptive BDF assessment/commit mismatch"
        );
        typename problem_type::state_type actual;
        ops->get( *out, actual.d, Equation::size );
        for ( int i = 0; i < Equation::size; ++i )
        {
            error = std::max( error, std::abs( actual[i] - reference[i] ) );
            if ( run )
                require( std::abs( actual[i] - first[i] ) < 2e-12, "Adaptive BDF restart changed its endpoint" );
        }
        if ( !run )
        {
            first      = actual;
            steps      = flow.get_steps();
            rejections = control->error_rejections;
        }
        else
            require(
                steps == flow.get_steps() && rejections == control->error_rejections,
                "Adaptive BDF restart retained controller history"
            );
    }
    require( control->orders & ( 1u << 2 ), "Adaptive BDF did not assess startup error" );
    if ( std::string( method ) == "BDF2" )
        require( control->orders & ( 1u << 3 ), "Adaptive BDF2 did not leave startup" );
    std::cout << method << ' ' << controller_name << " tol=" << tolerance << " error=" << error << " steps=" << steps
              << " rejections=" << rejections << '\n';
    return error;
}
}
#endif
