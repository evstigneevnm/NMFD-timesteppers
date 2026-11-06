#include "implicit_imex_dense_test.h"
#include "ode_benchmark_imex.h"

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;

// Split u'=lambda_E*u+lambda_I*u=-u, exact exp(-t). Sample every accepted
// step to check global interior order, both limiting cases and backward time.
void test_linear( double lambda_e, bool backward )
{
    using partition    = rate_problem<imex_linear_equation>;
    using problem_type = imex_problem<partition, partition>;
    auto ops           = std::make_shared<operations_type>( 1 );
    auto problem       = std::make_shared<problem_type>(
        std::make_shared<partition>( ops, imex_linear_equation{ lambda_e } ),
        std::make_shared<partition>( ops, imex_linear_equation{ -1 - lambda_e } )
    );
    vector_wrap_type state( *ops ), out( *ops ), sample( *ops );
    state.start_use();
    out.start_use();
    sample.start_use();
    double errors[3]{};
    for ( int level = 0; level < 3; ++level )
    {
        auto         step  = make_imex_dense_step( ops, problem, .1 / ( 1 << level ) );
        const double start = backward ? .85 : .25, end = backward ? .25 : .85;
        ops->assign_scalar( std::exp( -start ), *state );
        step->set_target_time( end );
        double time = start;
        for ( int n = 0; n < 100 && ( backward ? time > end : time < end ); ++n )
        {
            step->set_time( time );
            step->apply( *state, *out );
            require(
                step->get_status() == single_step_status::converged && step->dense_output_order() == 3,
                "IMEX dense linear solve failed"
            );
            const auto view = step->get_continuous_integration();
            view.evaluate( 0, *sample );
            require(
                ops->get_value_at_point( 0, *sample ) == ops->get_value_at_point( 0, *state ), "Dense initial endpoint"
            );
            for ( const double theta : { .13, .37, .71, 1. } )
            {
                const auto t = view.evaluate( theta, *sample );
                errors[level] =
                    std::max( errors[level], std::abs( ops->get_value_at_point( 0, *sample ) - std::exp( -t ) ) );
            }
            require(
                ops->get_value_at_point( 0, *sample ) == ops->get_value_at_point( 0, *out ), "Dense final endpoint"
            );
            time = view.evaluate( 1, *sample );
            ops->assign( *out, *state );
            step->finalize( adaptation_status::accepted, time, *state );
        }
        require( time == end, "Dense backward/forward clipping missed target" );
    }
    const auto order = std::log2( errors[1] / errors[2] );
    std::cout << "Dense linear lambda_E=" << lambda_e << " backward=" << backward << " order=" << order << '\n';
    require( order > 2.7 && order < 3.4 && errors[2] < 2e-6, "IMEX dense interior order mismatch" );
}

// Nonlinear, non-autonomous cancellation: f_E=.25*u^2+cos(t), f_I=-u-f_E.
// The full flow is exp(-t), but neither partition can provide its dense output alone.
struct nonlinear_split
{
    bool                  implicit = false;
    __DEVICE_TAG__ double mass( double ) const
    {
        return 1;
    }
    __DEVICE_TAG__ double rhs( double t, double u ) const
    {
        const auto e = .25 * u * u + std::cos( t );
        return implicit ? -u - e : e;
    }
    __DEVICE_TAG__ double rhs_derivative( double, double u ) const
    {
        return implicit ? -1 - .5 * u : .5 * u;
    }
};

void test_adaptive_nonlinear()
{
    using equation     = rate_scalar_equation<nonlinear_split>;
    using partition    = rate_problem<equation>;
    using problem_type = imex_problem<partition, partition>;
    using controller   = observed_controller<integration::time_step_adaptation_scipy<operations_type>>;
    using step_type    = imex_dense_step<problem_type, controller>;
    auto ops           = std::make_shared<operations_type>( 1 );
    auto problem       = std::make_shared<problem_type>(
        std::make_shared<partition>( ops, equation{ nonlinear_split{ false } } ),
        std::make_shared<partition>( ops, equation{ nonlinear_split{ true } } )
    );
    vector_wrap_type state( *ops ), out( *ops ), sample( *ops );
    state.start_use();
    out.start_use();
    sample.start_use();
    double errors[2]{};
    for ( int level = 0; level < 2; ++level )
    {
        auto p                  = controller_parameters<controller>( level ? 1e-6 : 1e-3 );
        auto control            = std::make_shared<controller>( ops, p );
        control->expected_order = 3;
        step_type step(
            ops, problem, control, newton_types<partition>::make_solver( ops ), step_type::params{ "IMEX_ARK324L2SA" }
        );
        double first_error = 0;
        for ( int run = 0; run < 2; ++run )
        {
            step.reset();
            step.set_target_time( 1.25 );
            ops->assign_scalar( std::exp( -.25 ), *state );
            double time = .25, error = 0;
            for ( int n = 0; n < 1000 && time < 1.25; ++n )
            {
                step.set_time( time );
                step.apply( *state, *out );
                require( step.get_status() == single_step_status::converged, "Adaptive nonlinear IMEX dense solve" );
                const auto view = step.get_continuous_integration();
                for ( const auto theta : { .19, .53, .87 } )
                {
                    const auto t = view.evaluate( theta, *sample );
                    error = std::max( error, std::abs( ops->get_value_at_point( 0, *sample ) - std::exp( -t ) ) );
                }
                time = view.evaluate( 1, *sample );
                ops->assign( *out, *state );
                step.finalize( adaptation_status::accepted, time, *state );
            }
            require( time == 1.25 && control->commits > 0, "Adaptive dense integration missed endpoint" );
            if ( run )
                require( std::abs( error - first_error ) < 1e-12, "Dense reset retained old rates" );
            first_error = errors[level] = error;
        }
    }
    require( errors[1] < .1 * errors[0] && errors[1] < 2e-5, "Adaptive nonlinear dense refinement" );
}

// Manufactured mass systems retain their original physical equations: coupled
// constant mass, positive state-dependent mass, and explicitly time-dependent mass.
template <class Equation>
void test_mass( Equation equation )
{
    using problem_type = imex_problem<benchmark_explicit_partition<Equation>, benchmark_implicit_partition<Equation>>;
    auto ops           = std::make_shared<operations_type>( Equation::size );
    auto problem       = std::make_shared<problem_type>(
        std::make_shared<benchmark_explicit_partition<Equation>>( benchmark_explicit_partition<Equation>{ equation } ),
        std::make_shared<benchmark_implicit_partition<Equation>>( ops, equation )
    );
    vector_wrap_type state( *ops ), out( *ops ), sample( *ops );
    state.start_use();
    out.start_use();
    sample.start_use();
    double errors[3]{};
    for ( int level = 0; level < 3; ++level )
    {
        auto step    = make_imex_dense_step( ops, problem, .1 / ( 1 << level ) );
        auto initial = exact_state<Equation>( .25 );
        ops->set( initial.d, *state, Equation::size );
        step->set_target_time( .85 );
        double time = .25;
        for ( int n = 0; n < 100 && time < .85; ++n )
        {
            step->set_time( time );
            step->apply( *state, *out );
            require( step->get_status() == single_step_status::converged, "Dense mass stage failed" );
            const auto view = step->get_continuous_integration();
            for ( const auto theta : { .13, .37, .71, 1. } )
            {
                const auto          t        = view.evaluate( theta, *sample );
                const auto          expected = exact_state<Equation>( t );
                decltype( initial ) actual;
                ops->get( *sample, actual.d, Equation::size );
                for ( int j = 0; j < Equation::size; ++j )
                    errors[level] = std::max( errors[level], std::abs( actual[j] - expected[j] ) );
            }
            time = view.evaluate( 1, *sample );
            ops->assign( *out, *state );
            step->finalize( adaptation_status::accepted, time, *state );
        }
        require( time == .85, "Dense mass integration missed target" );
    }
    const auto order = std::log2( errors[1] / errors[2] );
    std::cout << "Dense mass size=" << Equation::size << " order=" << order << '\n';
    require( order > 2.6 && order < 3.4 && errors[2] < 3e-6, "Mass dense extension order mismatch" );
}

int main()
try
{
    backend_type::init_device();
    for ( const double lambda : { .25, 0., -1. } )
    {
        test_linear( lambda, false );
        test_linear( lambda, true );
    }
    test_adaptive_nonlinear();
    test_mass( coupled_mass{} );
    test_mass( variable_mass{} );
    test_mass( time_dependent_mass{} );
    std::cout << "Combined IMEX dense orders, partition limits, nonlinear/time/mass systems and reset: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
