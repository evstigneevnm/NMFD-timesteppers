#include "implicit_bdf_dense_test.h"
#include <array>

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;
namespace model = nmfd::time_steppers::tests::bdf_test;

// Exact polynomial histories isolate interpolation from startup/Newton errors.
// Nonuniform, signed intervals test the scaled Newton basis for orders 1-5.
template <class Function>
void seed(
    bdf::divided_difference_history<operations_type> &history, const operations_type &ops, vector_type &old,
    vector_type &current, unsigned int order, const std::array<double, 5> &steps, Function value
)
{
    double t = 0;
    for ( unsigned int i = 0; i < order; ++i )
        t -= steps[i];
    ops.assign_scalar( value( t ), old );
    for ( unsigned int i = order; i > 0; --i )
    {
        t = i == 1 ? 0 : t + steps[i - 1];
        ops.assign_scalar( value( t ), current );
        history.commit( old, current, steps[i - 1], i == order );
        ops.assign( current, old );
    }
}

void test_interpolants()
{
    auto             ops = std::make_shared<operations_type>( 1 );
    vector_wrap_type old( *ops ), current( *ops ), candidate( *ops ), sample( *ops );
    old.start_use();
    current.start_use();
    candidate.start_use();
    sample.start_use();
    for ( unsigned int q = 1; q <= 5; ++q )
        for ( double sign : { -1., 1. } )
            for ( double ratio : { .5, 1., 1.2 } )
            {
                const double                h = sign * .08 * ratio;
                const std::array<double, 5> steps{ sign * .08, sign * .075, sign * .09, sign * .07, sign * .085 };
                bdf::divided_difference_history<operations_type> history( *ops );
                history.initialize( q );
                for ( unsigned int degree = 0; degree <= q; ++degree )
                {
                    auto value = [degree]( double t ) { return 1 + std::pow( t, int( degree ) ); };
                    seed( history, *ops, *old, *current, q, steps, value );
                    ops->assign_scalar( value( h ), *candidate );
                    for ( double theta : { 0., .23, .67, 1., .4, .23 } )
                    {
                        history.interpolate( q, theta, h, *current, *candidate, *sample );
                        require(
                            std::abs( ops->get_value_at_point( 0, *sample ) - value( theta * h ) ) < 2e-12,
                            "BDF dense polynomial reproduction mismatch"
                        );
                    }
                }
            }
    // u'=-u: exact-node interpolation has local error O(h^(q+1)).
    for ( unsigned int q = 1; q <= 5; ++q )
        for ( double sign : { -1., 1. } )
        {
            double errors[2]{};
            for ( int level = 0; level < 2; ++level )
            {
                const double                                     h = sign * .2 / ( 1 << level );
                std::array<double, 5>                            steps{ h, h * .95, h * 1.1, h * .9, h };
                bdf::divided_difference_history<operations_type> history( *ops );
                history.initialize( q );
                seed( history, *ops, *old, *current, q, steps, []( double t ) { return std::exp( -t ); } );
                ops->assign_scalar( std::exp( -h ), *candidate );
                history.interpolate( q, .37, h, *current, *candidate, *sample );
                errors[level] = std::abs( ops->get_value_at_point( 0, *sample ) - std::exp( -.37 * h ) );
            }
            const double observed = std::log2( errors[0] / errors[1] );
            require( observed > q + .6 && observed < q + 1.4, "BDF interpolation local order mismatch" );
        }
    model::throws<std::invalid_argument>( [] { bdf::make_interpolation( 6u, .5, .1 ); }, "Invalid dense order" );
    model::throws<std::invalid_argument>( [] { bdf::make_interpolation( 2u, .5, .1 ); }, "Missing dense history" );
}

// u'=t-u, u=t-1+2*exp(-t). Dense-enabled/disabled trajectories must be
// identical, including cold startup, variable grids and automatic order changes.
template <bool Dense>
double trajectory( unsigned int cap, double direction, bool variable, unsigned int &orders )
{
    using step_type = bdf::implicit_time_step<
        operations_type, model::problem_type, model::dense_controller, newton_types<model::problem_type>::solver_type,
        Dense>;
    auto                       ops     = std::make_shared<operations_type>( 1 );
    auto                       problem = std::make_shared<model::problem_type>( ops );
    auto                       control = std::make_shared<model::dense_controller>( .01 );
    typename step_type::params p{ "BDF" };
    p.adaptive      = true;
    p.maximum_order = cap;
    step_type        step( ops, problem, control, newton_types<model::problem_type>::make_solver( ops ), p );
    vector_wrap_type state( *ops ), sample( *ops );
    state.start_use();
    sample.start_use();
    double time = .25;
    ops->assign_scalar( model::forced_decay::exact( time ), *state );
    step.set_target_time( time + direction * 2 );
    for ( int i = 0; i < 60; ++i )
    {
        control->next = .01 * ( variable ? 1 + .08 * std::sin( i * .4 ) : 1 );
        step.set_time( time );
        const double before = ops->get_value_at_point( 0, *state );
        step.apply( *state, *state );
        require( step.get_status() == single_step_status::converged, "Dense BDF trajectory failed" );
        orders |= 1u << step.get_order();
        const double endpoint = ops->get_value_at_point( 0, *state );
        if constexpr ( Dense )
        {
            require(
                step.dense_output_order() == step.get_order(), "Dense BDF reports proposed rather than used order"
            );
            const auto count = problem->linearizations, assessments = control->assessments;
            const auto view = step.get_continuous_integration();
            view.evaluate( 0, *sample );
            require( ops->get_value_at_point( 0, *sample ) == before, "Dense BDF start not an exact copy" );
            view.evaluate( 1, *sample );
            require( ops->get_value_at_point( 0, *sample ) == endpoint, "Dense BDF endpoint not an exact copy" );
            for ( double theta : { .73, .19, .73 } )
            {
                const auto t = view.evaluate( theta, *sample );
                require(
                    std::abs( ops->get_value_at_point( 0, *sample ) - model::forced_decay::exact( t ) ) < .012,
                    "Dense BDF trajectory interior error"
                );
            }
            require(
                problem->linearizations == count && control->assessments == assessments,
                "Dense sampling evaluated Newton or adaptation"
            );
        }
        time += step.get_dt();
        step.finalize( adaptation_status::accepted, time, *state );
    }
    return ops->get_value_at_point( 0, *state );
}

void test_tolerance_refinement()
{
    using controller     = integration::time_step_adaptation_scipy<operations_type>;
    using step_type      = model::dense_step<model::problem_type, controller>;
    auto             ops = std::make_shared<operations_type>( 1 );
    vector_wrap_type state( *ops ), sample( *ops );
    state.start_use();
    sample.start_use();
    for ( double direction : { -1., 1. } )
    {
        double errors[2]{};
        for ( int level = 0; level < 2; ++level )
        {
            auto               problem = std::make_shared<model::problem_type>( ops );
            controller::params cp;
            cp.relative_tolerance = level ? 1e-6 : 1e-3;
            cp.absolute_tolerance = cp.relative_tolerance / 100;
            cp.initial_step = cp.maximum_step = .2;
            typename step_type::params p{ "BDF" };
            p.adaptive = true;
            step_type step(
                ops, problem, std::make_shared<controller>( ops, cp ),
                newton_types<model::problem_type>::make_solver( ops ), p
            );
            double time = .25, end = time + direction;
            ops->assign_scalar( model::forced_decay::exact( time ), *state );
            step.set_target_time( end );
            unsigned int steps = 0;
            while ( time != end && ++steps < 10000 )
            {
                step.set_time( time );
                step.apply( *state, *state );
                require( step.get_status() == single_step_status::converged, "Adaptive dense BDF failed" );
                auto view = step.get_continuous_integration();
                for ( double theta : { .23, .71 } )
                {
                    auto t        = view.evaluate( theta, *sample );
                    errors[level] = std::max(
                        errors[level],
                        std::abs( ops->get_value_at_point( 0, *sample ) - model::forced_decay::exact( t ) )
                    );
                }
                time = view.evaluate( 1, *sample );
                step.finalize( adaptation_status::accepted, time, *state );
            }
            require( time == end, "Adaptive dense BDF budget exhausted" );
        }
        require( errors[1] < .1 * errors[0] && errors[1] < .0003, "Dense BDF tolerance refinement failed" );
    }
}

int main()
try
{
    backend_type::init_device();
    test_interpolants();
    test_tolerance_refinement();
    for ( unsigned int cap = 1; cap <= 5; ++cap )
        for ( double direction : { -1., 1. } )
            for ( bool variable : { false, true } )
            {
                unsigned int plain = 0, dense = 0;
                const auto   expected = trajectory<false>( cap, direction, variable, plain );
                const auto   actual   = trajectory<true>( cap, direction, variable, dense );
                require(
                    expected == actual && plain == dense && ( dense & ( 1u << cap ) ),
                    "Dense output changed BDF trajectory, order selection or history"
                );
            }
    std::cout << "BDF dense orders 1-5, signed/nonuniform grids and unchanged/refined trajectories: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
