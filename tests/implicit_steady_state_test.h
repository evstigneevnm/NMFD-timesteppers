#ifndef NMFD_TIME_STEPPERS_TESTS_IMPLICIT_STEADY_STATE_H
#define NMFD_TIME_STEPPERS_TESTS_IMPLICIT_STEADY_STATE_H

#include "implicit_ode_test.h"
#include <limits>
#include <nmfd/time_steppers/integration/time_step_adaptation_globalization.h>

namespace nmfd
{
namespace time_steppers
{
namespace tests
{
// Evaluate the physical RHS, not the shift-scaled stage residual. Scratch and
// reductions stay on the selected backend; only the scalar norm is returned.
template <class Problem>
class rhs_merit
{
public:
    rhs_merit( std::shared_ptr<operations_type> operations, std::shared_ptr<Problem> problem )
        : operations_( std::move( operations ) ), problem_( std::move( problem ) ), residual_( *operations_ )
    {
        residual_.start_use();
    }
    double operator()( const vector_type &state )
    {
        if ( !problem_->equation.evaluate_rate( state, *residual_ ) )
            return std::numeric_limits<double>::infinity();
        return operations_->norm_l2( *residual_ );
    }

private:
    std::shared_ptr<operations_type> operations_;
    std::shared_ptr<Problem>         problem_;
    vector_wrap_type                 residual_;
};

template <class Merit>
struct steady_monitor
{
    std::shared_ptr<Merit> merit;
    double                 tolerance, previous;
    unsigned int           increases = 0;
    integration_status     status    = integration_status::running;
    void                   set_time_interval( const double &, double & )
    {
    }
    bool apply( const vector_type &, vector_type &state )
    {
        const auto value = ( *merit )( state );
        require( std::isfinite( value ), "Steady monitor received a nonfinite physical residual" );
        increases += value > previous * ( 1 + 1e-6 );
        previous = value;
        status   = value <= tolerance ? integration_status::stopped_by_external_operation : integration_status::running;
        return false;
    }
    integration_status get_status() const
    {
        return status;
    }
};

// Stopping belongs to external management. The large pseudo-time ceiling permits
// stiff unstable modes to be damped by large implicit steps, with a bounded budget.
template <class Step, class Merit>
std::size_t run_to_steady(
    std::shared_ptr<operations_type> operations, std::shared_ptr<Step> step, std::shared_ptr<Merit> merit,
    const vector_type &initial, vector_type &out, double tolerance
)
{
    using monitor_type = steady_monitor<Merit>;
    auto monitor       = std::make_shared<monitor_type>( monitor_type{ merit, tolerance, ( *merit )( initial ) } );
    integration::time_integrator<operations_type, Step, monitor_type> integrator( operations, step, { 2000 }, monitor );
    integrator.set_time_interval( 0, 1e12 );
    integrator.apply( initial, out );
    if ( integrator.get_status() != integration_status::stopped_by_external_operation )
        std::cerr << "Steady run stopped at " << integrator.get_final_time() << ", steps=" << integrator.get_steps()
                  << ", merit=" << ( *merit )( out ) << ", step status=" << int( step->get_status() ) << '\n';
    require(
        integrator.get_status() == integration_status::stopped_by_external_operation && ( *merit )( out ) <= tolerance,
        "Relaxation did not reach its independently checked steady tolerance"
    );
    std::cout << "steps=" << integrator.get_steps() << ", merit increases=" << monitor->increases
              << ", final merit=" << ( *merit )( out ) << '\n';
    return integrator.get_steps();
}

// Shared test assembly only: equations, equilibria and pseudo-step choices belong
// to the individual system files. Numerical solver tolerances are not relaxed.
template <class Equation>
void test_unstable_equilibrium(
    const Equation &equation, const scfd::static_vec::vec<double, Equation::size> &initial_values,
    const scfd::static_vec::vec<double, Equation::size> &equilibrium, double initial_step, const char *label,
    double steady_tolerance = 1e-8
)
{
    using problem_type = stage_problem<Equation>;
    using types        = newton_types<problem_type>;
    using merit_type   = rhs_merit<problem_type>;
    using controller   = integration::time_step_adaptation_globalization<operations_type, merit_type>;
    using step_type =
        runge_kutta::implicit_time_step<operations_type, problem_type, controller, typename types::solver_type>;
    auto             operations = std::make_shared<operations_type>( Equation::size );
    auto             problem    = std::make_shared<problem_type>( operations, equation );
    auto             merit      = std::make_shared<merit_type>( operations, problem );
    vector_wrap_type initial( *operations ), result( *operations );
    initial.start_use();
    result.start_use();
    operations->set( equilibrium.d, *result, Equation::size );
    require( ( *merit )( *result ) < 1e-10, "Analytical equilibrium does not satisfy the physical equation" );
    operations->set( initial_values.d, *initial, Equation::size );
    require( ( *merit )( *initial ) > 1e-6, "Unstable-equilibrium test starts already converged" );
    check_jacobian( operations, *problem, initial_values );
    for ( const char *method : { "IE", "SDIRK2(1)2", "SDIRK3(1)3" } )
    {
        typename controller::params p;
        p.initial_step = initial_step;
        auto control   = std::make_shared<controller>( merit, p );
        auto step      = std::make_shared<step_type>(
            operations, problem, control, types::make_solver( operations ), typename step_type::params{ method }
        );
        scfd::static_vec::vec<double, Equation::size> first;
        std::size_t                                   first_steps = 0;
        for ( int run = 0; run < 2; ++run )
        {
            std::cout << label << ' ' << method << ( run ? " restart: " : ": " );
            const auto steps = run_to_steady( operations, step, merit, *initial, *result, steady_tolerance );
            scfd::static_vec::vec<double, Equation::size> actual, unchanged;
            operations->get( *result, actual.d, Equation::size );
            operations->get( *initial, unchanged.d, Equation::size );
            double maximum_error = 0;
            for ( int i = 0; i < Equation::size; ++i )
            {
                maximum_error = std::max( maximum_error, std::abs( actual[i] - equilibrium[i] ) );
                require( std::abs( actual[i] - equilibrium[i] ) < 1e-6, "Relaxation reached the wrong equilibrium" );
                require( unchanged[i] == initial_values[i], "Steady relaxation modified its initial state" );
                if ( run )
                    require( std::abs( actual[i] - first[i] ) < 1e-10, "Steady restart changed its endpoint" );
            }
            if ( run )
                require( steps == first_steps, "Steady restart retained controller or solver history" );
            else
                std::cout << "maximum component error=" << maximum_error << '\n';
            first       = actual;
            first_steps = steps;
        }
    }
}
}
}
}
#endif
