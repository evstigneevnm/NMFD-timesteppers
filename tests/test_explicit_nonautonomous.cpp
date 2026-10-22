#include "nonautonomous_test_problems.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#ifdef TEST_VECTOR_BACKEND_CUDA
#    include <scfd/backend/cuda.h>
using backend_type = scfd::backend::cuda<std::ptrdiff_t>;
#elif defined( TEST_VECTOR_BACKEND_OMP )
#    include <scfd/backend/omp.h>
using backend_type = scfd::backend::omp<std::ptrdiff_t>;
#else
#    include <scfd/backend/serial_cpu.h>
using backend_type = scfd::backend::serial_cpu<std::ptrdiff_t>;
#endif
#include <nmfd/operations/scfd_vector_space.h>
#include <nmfd/time_steppers/runge_kutta/explicit_time_step.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_constant.h>
#include <nmfd/time_steppers/integration/time_integrator.h>

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;
using operations_type = nmfd::operations::scfd_vector_space<backend_type, double>;
using vector_type     = operations_type::vector_type;
using adaptation_type = integration::time_step_adaptation_constant<operations_type>;

// Non-autonomous scalar benchmarks with exact trajectories:
// u'=t*u -> exp(t^2/2); (1+t)*u'=-u -> 1/(1+t);
// u'=-kappa*(u-sin(t))+cos(t) -> sin(t).
// Explicit problems return the mass-inverted rate. Shifted initial times use the exact initial state.
// Tests audit stage times, refinement and restart. Stiff tracking uses only stability-resolving steps.
template <class Equation>
struct rate_problem
{
    Equation            equation;
    double              time = 0;
    std::vector<double> stage_times;
    void                set_time( double t )
    {
        time = t;
        stage_times.push_back( t );
    }
    struct mapping
    {
        Equation            equation;
        double              time;
        vector_type         in, out;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t ) const
        {
            out( 0 ) = equation.rhs( time, in( 0 ) ) / equation.mass( time );
        }
    };
    void apply( const vector_type &in, vector_type &out ) const
    {
        const mapping map{ equation, time, in, out };
        if constexpr ( operations_type::memory_type::is_host_visible )
            map( 0 );
        else
        {
            operations_type::for_each_type execute;
            execute( map, 1 );
            execute.wait();
        }
    }
};

void require( bool condition, const char *message )
{
    if ( !condition )
        throw std::runtime_error( message );
}

template <class Equation>
void test_equation( const Equation &equation, double coarse_step = .1 )
{
    using problem_type = rate_problem<Equation>;
    using step_type    = runge_kutta::explicit_time_step<operations_type, problem_type, adaptation_type>;
    auto                                       operations = std::make_shared<operations_type>( 1 );
    auto                                       problem    = std::make_shared<problem_type>( problem_type{ equation } );
    nmfd::detail::vector_wrap<operations_type> initial( *operations ), result( *operations );
    initial.start_use();
    result.start_use();
    for ( const double time : { .25, .6, 1.25 } )
    {
        const auto derivative =
            ( equation.rhs( time, .7 + 1e-5 ) - equation.rhs( time, .7 - 1e-5 ) ) / ( 2e-5 * equation.mass( time ) );
        require(
            std::abs( derivative - equation.rhs_derivative( time, .7 ) / equation.mass( time ) ) <
                1e-7 * std::max( 1., std::abs( derivative ) ),
            "Explicit rate derivative mismatch"
        );
    }
    for ( const char *method : { "EE", "HE", "RK23", "RK33SSP", "RK43SSP", "RK4", "RK64SSP", "RK45", "DOP853" } )
    {
        const auto table = runge_kutta::make_butcher_table( method );
        double     errors[3]{};
        for ( int level = 0; level < 3; ++level )
        {
            const double dt         = coarse_step / ( 1 << level );
            auto         adaptation = std::make_shared<adaptation_type>( adaptation_type::params{ dt } );
            auto         step =
                std::make_shared<step_type>( operations, problem, adaptation, typename step_type::params{ method } );
            operations->assign_scalar( equation.exact( .25 ), *initial );
            problem->stage_times.clear();
            step->set_time( .25 );
            step->set_target_time( .25 + dt );
            step->apply( *initial, *result );
            require(
                step->get_status() == single_step_status::converged && problem->stage_times.size() == table.size(),
                "Explicit non-autonomous stage audit failed"
            );
            for ( std::size_t i = 0; i < table.size(); ++i )
                require(
                    std::abs( problem->stage_times[i] - ( .25 + step->get_dt() * double( table.c( i ) ) ) ) < 1e-14,
                    "Explicit RHS used the wrong stage time"
                );
            step->finalize( adaptation_status::accepted, .25 + step->get_dt(), *result );
            integration::time_integrator<operations_type, step_type> integrator( operations, step );
            integrator.set_time_interval( .25, 1.25 );
            integrator.apply( *initial, *result );
            require(
                integrator.get_status() == integration_status::completed && integrator.get_final_time() == 1.25,
                "Explicit shifted-time integration failed"
            );
            const auto actual = operations->get_value_at_point( 0, *result );
            errors[level]     = std::abs( actual - equation.exact( 1.25 ) );
            integrator.apply( *initial, *result );
            require(
                integrator.get_status() == integration_status::completed &&
                    operations->get_value_at_point( 0, *result ) == actual,
                "Explicit shifted-time restart failed"
            );
        }
        require( errors[2] < std::max( 2e-11, .8 * errors[1] ), "Explicit non-autonomous refinement failed" );
        require(
            errors[2] < .03 * std::max( 1., std::abs( equation.exact( 1.25 ) ) ),
            "Explicit non-autonomous analytical solution mismatch"
        );
        if ( errors[2] > 2e-11 )
            require(
                std::log2( errors[1] / errors[2] ) > table.order() - .4, "Explicit non-autonomous order mismatch"
            );
        std::cout << method << " shifted-time errors " << errors[0] << " -> " << errors[1] << " -> " << errors[2]
                  << '\n';
    }
}

int main()
try
{
    backend_type::init_device();
    std::cout << "Time-dependent coefficient: u'=t*u\n";
    test_equation( coefficient_growth<double>{} );
    std::cout << "Time-dependent mass: (1+t)*u'=-u\n";
    test_equation( time_mass_decay<double>{} );
    std::cout << "Tracking: kappa=1\n";
    test_equation( stiff_tracking<double>{ 1 } );
    // h*kappa<=0.5 for every explicit method, including Euler.
    std::cout << "Tracking: kappa=100, h*kappa<=0.5\n";
    test_equation( stiff_tracking<double>{ 100 }, .005 );
    std::cout << "Explicit time-dependent coefficients/mass, stage times, tracking and shifted restart: PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
