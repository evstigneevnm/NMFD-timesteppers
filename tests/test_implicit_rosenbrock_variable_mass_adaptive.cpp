#include "implicit_rosenbrock_variable_mass_problem.h"
#include <nmfd/time_steppers/integration/time_step_adaptation_filter.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_matlab.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_scipy.h>

using namespace rosenbrock_tests;

// Refine tolerances for the distinct state/time/coupled mass systems. Mass
// solves remain problem-owned; only the combined state defect reaches the
// ordinary controllers. Independent trajectories must restart identically.
template <class Model, class Controller>
void refinement( const char *method, const char *filter = nullptr )
{
    using P              = variable_mass_problem<Model>;
    using S              = rosenbrock::time_step<operations_type, P, Controller, direct_solver<P>, true>;
    auto             ops = std::make_shared<operations_type>( 2 );
    auto             p   = std::make_shared<P>( ops );
    vector_wrap_type in( *ops ), out( *ops );
    in.start_use();
    out.start_use();
    operations_type::for_each_type{}( variable_mass_initial<Model, vector_type>{ .37, *in }, 2 );
    // Preserve backend-produced values: CPU/GPU exp() may differ by one ulp.
    const std::array<double, 2> initial{ ops->get_value_at_point( 0, *in ), ops->get_value_at_point( 1, *in ) };
    double                      errors[2]{};
    for ( unsigned int level = 0; level < 2; ++level )
    {
        typename Controller::params cp;
        cp.relative_tolerance = level ? 1e-5 : 1e-3;
        cp.absolute_tolerance = cp.relative_tolerance * .01;
        cp.initial_step = cp.maximum_step = .2;
        if constexpr ( std::is_same_v<Controller, integration::time_step_adaptation_filter<operations_type>> )
            cp.controller = filter;
        typename S::params sp;
        sp.method     = method;
        sp.autonomous = variable_mass_autonomous<Model>;
        auto step     = std::make_shared<S>(
            ops, p, std::make_shared<Controller>( ops, cp ), std::make_shared<direct_solver<P>>( ops ), sp
        );
        integration::time_integrator<operations_type, S> run( ops, step );
        run.set_time_interval( .37, .77 );
        std::array<double, 2> first{};
        std::size_t           count = 0;
        for ( unsigned int restart = 0; restart < 2; ++restart )
        {
            run.apply( *in, *out );
            check(
                run.get_status() == integration_status::completed && run.get_final_time() == .77,
                "Adaptive variable-mass trajectory failed"
            );
            if ( !restart )
                count = run.get_steps();
            else
                check( count == run.get_steps(), "Variable-mass restart changed its accepted mesh" );
            for ( unsigned int i = 0; i < 2; ++i )
            {
                const double value = ops->get_value_at_point( i, *out );
                if ( !restart )
                    first[i] = value;
                else
                    check( value == first[i], "Variable-mass restart retained stale data" );
                errors[level] =
                    std::max( errors[level], std::abs( value - variable_mass_exact<Model>( i, .77, .37 ) ) );
                check(
                    ops->get_value_at_point( i, *in ) == initial[i],
                    "Variable-mass integration modified its initial state"
                );
            }
        }
    }
    check( errors[1] < .4 * errors[0] && errors[1] < 3e-4, "Variable-mass tolerance refinement failed" );
}

int main()
try
{
    backend_type::init_device();
    using Scipy  = integration::time_step_adaptation_scipy<operations_type>;
    using Matlab = integration::time_step_adaptation_matlab<operations_type>;
    using Filter = integration::time_step_adaptation_filter<operations_type>;
    for ( const char *method : { "ROS2", "ROS3", "ROS34PW2" } )
    {
        refinement<state_mass_rate_model, Scipy>( method );
        refinement<time_mass_rate_model, Scipy>( method );
        refinement<coupled_variable_mass_model, Scipy>( method );
    }
    refinement<state_mass_rate_model, Matlab>( "ROS34PW2" );
    refinement<time_mass_rate_model, Matlab>( "ROS34PW2" );
    refinement<coupled_variable_mass_model, Matlab>( "ROS34PW2" );
    for ( const char *filter : { "I", "H211", "PC", "PID", "H312", "PPID", "H321" } )
    {
        refinement<state_mass_rate_model, Filter>( "ROS34PW2", filter );
        refinement<coupled_variable_mass_model, Filter>( "ROS34PW2", filter );
    }
    std::cout << "PASS: adaptive Rosenbrock variable mass (" << checks << " checks)\n";
}
catch ( const std::exception &e )
{
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
