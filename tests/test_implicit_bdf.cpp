#include "implicit_bdf_test.h"

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;
namespace model = nmfd::time_steppers::tests::bdf_test;

// Smooth forced decay from implicit_bdf_test.h checks both formulas directly,
// first/second-order refinement, signed shifts, and nonzero-time restarts.
void test_formulas()
{
    auto             operations = std::make_shared<operations_type>( 1 );
    auto             problem    = std::make_shared<model::problem_type>( operations );
    vector_wrap_type state( *operations ), out( *operations );
    state.start_use();
    out.start_use();
    for ( const char *method : { "BDF1", "BDF2" } )
        for ( const double reference_time : { .25, 1., 4. } )
            for ( const double h : { -2., -.1, 1e-6, .1, 2. } )
            {
                using step_type = model::step_type<model::problem_type>;
                step_type step(
                    operations, problem, std::make_shared<adaptation_type>( adaptation_type::params{ std::abs( h ) } ),
                    newton_types<model::problem_type>::make_solver( operations ), { method, 32, reference_time }
                );
                double time = .3, u = .7, older = u;
                operations->assign_scalar( u, *state );
                operations->assign_scalar( 123., problem->known_rhs() );
                for ( int i = 0; i < 3; ++i )
                {
                    step.set_time( time );
                    step.set_target_time( time + h );
                    step.apply( *state, *out );
                    const auto         dt       = step.get_dt();
                    const unsigned int order    = i > 0 && std::string( method ) == "BDF2" ? 2 : 1;
                    const double       alpha    = ( order == 2 ? 1.5 : 1. ) / dt;
                    const double       w        = order == 2 ? ( 4 * u - older ) / 3 : u;
                    const double       expected = ( alpha * w + time + dt ) / ( alpha + 1 );
                    const auto         shift    = balanced_stage_scaling( alpha, reference_time );
                    require(
                        step.get_status() == single_step_status::converged && step.get_order() == order,
                        "BDF formula selected the wrong order or failed"
                    );
                    require(
                        std::abs( operations->get_value_at_point( 0, *out ) - expected ) < 2e-9,
                        "BDF scalar recurrence mismatch"
                    );
                    require(
                        problem->time == time + dt && problem->scaling().mass == shift.mass &&
                            problem->scaling().rhs == shift.rhs && operations->norm_l2( problem->known_rhs() ) == 0,
                        "BDF changed its endpoint time, balanced shift or zero RHS contract"
                    );
                    step.finalize( adaptation_status::accepted, time + dt, *out );
                    older = u;
                    u     = expected;
                    time += dt;
                    operations->assign( *out, *state );
                }
                require( step.dense_output_order() == 0, "BDF advertised unimplemented dense output" );
            }
}

void test_order()
{
    auto             operations = std::make_shared<operations_type>( 1 );
    auto             problem    = std::make_shared<model::problem_type>( operations );
    vector_wrap_type state( *operations ), out( *operations );
    state.start_use();
    out.start_use();
    for ( const char *method : { "BDF1", "BDF2" } )
        for ( const double start : { 0., .25, 1.25 } )
        {
            double errors[2]{};
            for ( int level = 0; level < 2; ++level )
            {
                const double end  = start == 1.25 ? .25 : start + 1;
                auto         step = model::make_step( operations, problem, .025 / ( level + 1 ), method );
                integration::time_integrator<operations_type, typename model::step_type<model::problem_type>> flow(
                    operations, step
                );
                flow.set_time_interval( start, end );
                operations->assign_scalar( model::forced_decay::exact( start ), *state );
                for ( int run = 0; run < 2; ++run )
                {
                    flow.apply( *state, *out );
                    require(
                        flow.get_status() == integration_status::completed && flow.get_final_time() == end,
                        "BDF integration or trajectory reset failed"
                    );
                    const auto error =
                        std::abs( operations->get_value_at_point( 0, *out ) - model::forced_decay::exact( end ) );
                    if ( run == 0 )
                        errors[level] = error;
                    else
                        require( error == errors[level], "BDF repeated trajectories retained stale history" );
                }
            }
            const double observed = std::log2( errors[0] / errors[1] );
            const int    expected = std::string( method ) == "BDF2" ? 2 : 1;
            std::cout << method << " start=" << start << " order=" << observed << '\n';
            require( observed > expected - .15 && observed < expected + .15, "BDF analytical order mismatch" );
        }
}

// u'=-1000*u is strongly damped. Steps much larger than its decay time must
// remain bounded and decay; transient sign changes in BDF2 are permitted.
struct stiff_decay_equation
{
    __DEVICE_TAG__ double mass( double ) const
    {
        return 1;
    }
    __DEVICE_TAG__ double rhs( double, double u ) const
    {
        return -1000 * u;
    }
    __DEVICE_TAG__ double rhs_derivative( double, double ) const
    {
        return -1000;
    }
};

void test_stiff_decay()
{
    using problem_type          = stage_problem<scalar_time_equation<stiff_decay_equation>>;
    auto             operations = std::make_shared<operations_type>( 1 );
    auto             problem    = std::make_shared<problem_type>( operations );
    vector_wrap_type state( *operations );
    state.start_use();
    for ( const char *method : { "BDF1", "BDF2" } )
    {
        auto step = model::make_step( operations, problem, .1, method );
        integration::time_integrator<operations_type, model::step_type<problem_type>> flow( operations, step );
        flow.set_time_interval( 0, 1 );
        operations->assign_scalar( 1., *state );
        flow.apply( *state, *state );
        require(
            flow.get_status() == integration_status::completed && operations->norm_l2( *state ) < 2e-9,
            "Stiff BDF trajectory did not damp the fast mode"
        );
    }
}

int main()
try
{
    backend_type::init_device();
    test_formulas();
    test_order();
    test_stiff_decay();
    std::cout << "BDF1/BDF2 formulas, startup, non-autonomous order, backward solves and stiff decay: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
