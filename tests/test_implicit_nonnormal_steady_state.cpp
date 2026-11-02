#include "implicit_steady_state_test.h"

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;

// u'=A*u, A=[-1,10;0,-2], u(0)=(-6,-1/2). Both eigenvalues are negative,
// but f(0)=(1,1) and f^T*A*f=7>0: the steady residual initially increases.
// Exact solution: u1=-11*exp(-t)+5*exp(-2*t), u2=-exp(-2*t)/2.
// A mandatory merit decrease rejects arbitrarily small valid initial steps.
struct nonnormal_decay
{
    static constexpr int size = 2;
    struct mapping
    {
        vector_type         in, out;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t ) const
        {
            out( 0 ) = -in( 0 ) + 10 * in( 1 );
            out( 1 ) = -2 * in( 1 );
        }
    };
    void apply( const vector_type &in, vector_type &out ) const
    {
        execute_small_mapping( mapping{ in, out } );
    }
    __DEVICE_TAG__ void residual(
        double, const stage_scaling &scaling, const vector_type &state, const vector_type &previous,
        const vector_type &rhs, vector_type &out
    ) const
    {
        identity_mass_residual<2>( mapping{ state, out }, scaling, previous, rhs );
    }
    auto jacobian(
        double, const stage_scaling &s, const scfd::static_vec::vec<double, 2> &,
        const scfd::static_vec::vec<double, 2> &
    ) const
    {
        return scfd::static_mat::mat<double, 2, 2>{ s.mass + s.rhs, -10 * s.rhs, 0, s.mass + 2 * s.rhs };
    }
};

void test_nonnormal_decay()
{
    using problem_type = stage_problem<nonnormal_decay>;
    using types        = newton_types<problem_type>;
    using merit_type   = rhs_merit<problem_type>;
    using controller   = integration::time_step_adaptation_globalization<operations_type, merit_type>;
    using step_type    = runge_kutta::implicit_time_step<operations_type, problem_type, controller, types::solver_type>;
    auto             operations = std::make_shared<operations_type>( 2 );
    auto             problem    = std::make_shared<problem_type>( operations );
    auto             merit      = std::make_shared<merit_type>( operations, problem );
    vector_wrap_type initial( *operations ), result( *operations );
    initial.start_use();
    result.start_use();
    const double values[] = { -6, -.5 };
    operations->set( values, *initial, 2 );
    check_jacobian( operations, *problem, { -6, -.5 } );
    const auto         baseline = ( *merit )( *initial );
    controller::params p;
    p.initial_step = .05;
    p.minimum_step = .001;
    auto control   = std::make_shared<controller>( merit, p );
    auto step      = std::make_shared<step_type>( operations, problem, control, types::make_solver( operations ) );
    step->apply( *initial, *result );
    require(
        step->get_status() == single_step_status::converged && step->get_attempts() == 1 &&
            ( *merit )( *result ) > baseline && control->get_dt() < step->get_dt(),
        "Default pseudo-time policy rejected the necessary nonnormal residual increase"
    );
    step->finalize( adaptation_status::accepted, step->get_dt(), *result );

    p.maximum_merit_growth = 1;
    auto      guarded      = std::make_shared<controller>( merit, p );
    step_type guarded_step( operations, problem, guarded, types::make_solver( operations ) );
    operations->assign_scalar( 42., *result );
    guarded_step.apply( *initial, *result );
    require(
        guarded_step.get_status() == single_step_status::failed_minimum_dt &&
            operations->get_value_at_point( 0, *result ) == 42,
        "Monotone guard regression no longer exposes nonnormal stagnation/rollback"
    );

    for ( const char *method : { "IE", "SDIRK2(1)2", "SDIRK3(1)3" } )
    {
        p.maximum_merit_growth = 0;
        auto relaxed           = std::make_shared<controller>( merit, p );
        auto method_step       = std::make_shared<step_type>(
            operations, problem, relaxed, types::make_solver( operations ), step_type::params{ method }
        );
        std::cout << "Nonnormal " << method << ": ";
        const auto steps = run_to_steady( operations, method_step, merit, *initial, *result, 1e-8 );
        require( operations->norm_l2( *result ) < 1e-7, "Nonnormal relaxation reached the wrong equilibrium" );
        const auto first = operations->get_value_at_point( 0, *result );
        require(
            run_to_steady( operations, method_step, merit, *initial, *result, 1e-8 ) == steps &&
                std::abs( operations->get_value_at_point( 0, *result ) - first ) < 1e-13,
            "Nonnormal relaxation restart retained the merit history"
        );
    }

    using monitor_type = steady_monitor<merit_type>;
    auto monitor       = std::make_shared<monitor_type>( monitor_type{ merit, 1e-8, baseline } );
    integration::time_integrator<operations_type, step_type, monitor_type> limited( operations, step, { 1 }, monitor );
    limited.set_time_interval( 0, 1000 );
    limited.apply( *initial, *result );
    require(
        limited.get_status() == integration_status::attempt_limit_reached && ( *merit )( *result ) > 1e-8,
        "An exhausted relaxation budget was reported as steady convergence"
    );
}

int main()
try
{
    backend_type::init_device();
    test_nonnormal_decay();
    std::cout << "Nonnormal steady relaxation: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
