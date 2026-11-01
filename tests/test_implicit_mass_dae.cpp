#include "implicit_adaptive_test.h"
#include "nonautonomous_test_problems.h"

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;

// Trajectories use intervals of length 0.4, including a shifted restart for the moving constraint.

// coupled_mass: M*u'=-M*diag(1,2)*u, M=[[2,1],[1,3]], u(0)=(1,1).
// Linear autonomous system with a constant symmetric positive-definite, non-diagonal mass.
// Exact solution: u=(exp(-t),exp(-2*t)); verifies coupled stages without mass inversion.
struct coupled_mass
{
    static constexpr int size = 2;
    __DEVICE_TAG__ void  residual(
         double, const stage_scaling &scaling, const vector_type &x, const vector_type &previous, const vector_type &rhs,
         vector_type &out
     ) const
    {
        const auto d0 = x( 0 ) - previous( 0 ), d1 = x( 1 ) - previous( 1 );
        out( 0 ) = scaling.mass * ( 2 * d0 + d1 ) + scaling.rhs * ( 2 * x( 0 ) + 2 * x( 1 ) - rhs( 0 ) );
        out( 1 ) = scaling.mass * ( d0 + 3 * d1 ) + scaling.rhs * ( x( 0 ) + 6 * x( 1 ) - rhs( 1 ) );
    }
    auto jacobian(
        double, const stage_scaling &scaling, const scfd::static_vec::vec<double, 2> &,
        const scfd::static_vec::vec<double, 2> &
    ) const
    {
        return scfd::static_mat::mat<double, 2, 2>{
            2 * scaling.mass + 2 * scaling.rhs, scaling.mass + 2 * scaling.rhs, scaling.mass + scaling.rhs,
            3 * scaling.mass + 6 * scaling.rhs
        };
    }
};

// variable_mass: (1+u^2)*u'=-(1+u^2)*u, u(0)=1.
// The mass is state-dependent and strictly positive; the exact trajectory is u=exp(-t).
// The stage residual is nonlinear, and its Jacobian must include the mass derivative.
struct variable_mass
{
    static constexpr int size = 1;
    __DEVICE_TAG__ void  residual(
         double, const stage_scaling &scaling, const vector_type &x, const vector_type &previous, const vector_type &rhs,
         vector_type &out
     ) const
    {
        const auto u = x( 0 );
        out( 0 ) = ( 1 + u * u ) * ( scaling.mass * ( u - previous( 0 ) ) + scaling.rhs * u ) - scaling.rhs * rhs( 0 );
    }
    auto jacobian(
        double, const stage_scaling &scaling, const scfd::static_vec::vec<double, 1> &x,
        const scfd::static_vec::vec<double, 1> &previous
    ) const
    {
        return scfd::static_mat::mat<double, 1, 1>{
            scaling.mass * ( 1 + x[0] * x[0] + 2 * x[0] * ( x[0] - previous[0] ) ) +
            scaling.rhs * ( 1 + 3 * x[0] * x[0] )
        };
    }
};

// (1+t)*u'=-u, u(0)=1; exact u=1/(1+t). Positive, explicitly time-dependent mass.
// The balanced stage Jacobian is a*(1+t)+b, evaluated at the current stage time.
using time_dependent_mass = scalar_time_equation<time_mass_decay<double>>;

// coupled_dae: x'=-x-z+x^2, 0=z-x^2, with initial state (1,1).
// Nonlinear autonomous index-1 DAE with singular mass M=diag(1,0).
// Exact solution: (x,z)=(exp(-t),exp(-2*t)); checks the constraint at accepted endpoints.
struct coupled_dae
{
    static constexpr int size = 2;
    __DEVICE_TAG__ void  residual(
         double, const stage_scaling &scaling, const vector_type &x, const vector_type &previous, const vector_type &rhs,
         vector_type &out
     ) const
    {
        out( 0 ) =
            scaling.mass * ( x( 0 ) - previous( 0 ) ) + scaling.rhs * ( x( 0 ) + x( 1 ) - x( 0 ) * x( 0 ) - rhs( 0 ) );
        out( 1 ) = x( 1 ) - x( 0 ) * x( 0 ) - rhs( 1 );
    }
    auto jacobian(
        double, const stage_scaling &scaling, const scfd::static_vec::vec<double, 2> &x,
        const scfd::static_vec::vec<double, 2> &
    ) const
    {
        return scfd::static_mat::mat<double, 2, 2>{
            scaling.mass + scaling.rhs * ( 1 - 2 * x[0] ), scaling.rhs, -2 * x[0], 1
        };
    }
};

// x'=-x-z+sin(t)*x, 0=z-sin(t)*x, with consistent initial state (1,0).
// Non-autonomous index-1 DAE, M=diag(1,0); the constraint manifold moves with time.
// Exact solution (x,z)=(exp(-t),exp(-t)*sin(t)); verifies stage times and endpoint consistency.
struct moving_constraint_dae
{
    static constexpr int size = 2;
    __DEVICE_TAG__ void  residual(
         double t, const stage_scaling &scaling, const vector_type &x, const vector_type &previous,
         const vector_type &rhs, vector_type &out
     ) const
    {
        out( 0 ) = scaling.mass * ( x( 0 ) - previous( 0 ) ) +
                   scaling.rhs * ( ( 1 - std::sin( t ) ) * x( 0 ) + x( 1 ) - rhs( 0 ) );
        out( 1 ) = x( 1 ) - std::sin( t ) * x( 0 ) - rhs( 1 );
    }
    auto jacobian(
        double t, const stage_scaling &scaling, const scfd::static_vec::vec<double, 2> &,
        const scfd::static_vec::vec<double, 2> &
    ) const
    {
        return scfd::static_mat::mat<double, 2, 2>{
            scaling.mass + scaling.rhs * ( 1 - std::sin( t ) ), scaling.rhs, -std::sin( t ), 1
        };
    }
};

// stokes_dae: u0'+u0+p=sin(t), u1'+2*u1+p=sin(t)-exp(-t), 0=u0+u1.
// Linear non-autonomous index-2 DAE with M=diag(1,1,0); p is a pressure-like multiplier.
// Initial state (1,-1,0); exact solution (u0,u1,p)=(exp(-t),-exp(-t),sin(t)).
// Tests a synthetic monolithic velocity-pressure saddle system.
struct stokes_dae
{
    static constexpr int size = 3;
    __DEVICE_TAG__ void  residual(
         double t, const stage_scaling &scaling, const vector_type &x, const vector_type &previous,
         const vector_type &rhs, vector_type &out
     ) const
    {
        out( 0 ) =
            scaling.mass * ( x( 0 ) - previous( 0 ) ) + scaling.rhs * ( x( 0 ) + x( 2 ) - std::sin( t ) - rhs( 0 ) );
        out( 1 ) = scaling.mass * ( x( 1 ) - previous( 1 ) ) +
                   scaling.rhs * ( 2 * x( 1 ) + x( 2 ) + std::exp( -t ) - std::sin( t ) - rhs( 1 ) );
        out( 2 ) = x( 0 ) + x( 1 ) - rhs( 2 );
    }
    auto jacobian(
        double, const stage_scaling &scaling, const scfd::static_vec::vec<double, 3> &,
        const scfd::static_vec::vec<double, 3> &
    ) const
    {
        return scfd::static_mat::mat<double, 3, 3>{
            scaling.mass + scaling.rhs, 0, scaling.rhs, 0, scaling.mass + 2 * scaling.rhs, scaling.rhs, 1, 1, 0
        };
    }
};

template <class Equation>
struct constraint_monitor
{
    std::shared_ptr<operations_type> operations;
    unsigned int                     calls = 0;
    double                           time  = 0;
    void                             set_time_interval( const double &, double &end )
    {
        time = end;
    }
    bool apply( const vector_type &, vector_type &out )
    {
        scfd::static_vec::vec<double, Equation::size> x;
        operations->get( out, x.d, Equation::size );
        require( std::abs( constraint( x, time ) ) < 2e-9, "DAE constraint violated at accepted endpoint" );
        ++calls;
        return false;
    }
    static double constraint( const scfd::static_vec::vec<double, Equation::size> &x, double time )
    {
        if constexpr ( std::is_same_v<Equation, stokes_dae> )
            return x[0] + x[1];
        else if constexpr ( std::is_same_v<Equation, moving_constraint_dae> )
            return x[1] - std::sin( time ) * x[0];
        else
            return x[1] - x[0] * x[0];
    }
    integration_status get_status() const
    {
        return integration_status::running;
    }
};

template <class Equation>
scfd::static_vec::vec<double, Equation::size> exact_state( double time )
{
    scfd::static_vec::vec<double, Equation::size> state;
    for ( int i = 0; i < Equation::size; ++i )
    {
        state[i] = std::exp( -time * ( i + 1 ) );
        if constexpr ( std::is_same_v<Equation, stokes_dae> )
            state[i] = i == 0 ? std::exp( -time ) : i == 1 ? -std::exp( -time ) : std::sin( time );
        else if constexpr ( std::is_same_v<Equation, time_dependent_mass> )
            state[i] = 1 / ( 1 + time );
        else if constexpr ( std::is_same_v<Equation, moving_constraint_dae> )
            state[i] = i == 0 ? std::exp( -time ) : std::exp( -time ) * std::sin( time );
    }
    return state;
}

// Deliberately inconsistent points must retain their full constraint residual,
// even when the differential rows have been strongly scaled. Then check actual
// small/large and backward IE stages from consistent initial data.
template <class Equation>
void test_dae_scaling(
    std::shared_ptr<operations_type> operations, const scfd::static_vec::vec<double, Equation::size> &point
)
{
    using problem_type = stage_problem<Equation>;
    using monitor_type = constraint_monitor<Equation>;
    vector_wrap_type state( *operations ), result( *operations );
    state.start_use();
    result.start_use();
    for ( const double reference_time : { .25, 4. } )
    {
        problem_type problem( operations, {}, reference_time );
        operations->assign_scalar( .2, problem.previous_state() );
        operations->assign_scalar( .1, problem.known_rhs() );
        operations->set( point.d, *state, Equation::size );
        problem.set_time( .4 );
        for ( const double alpha : { -1e12, -1e-12, 0., 1e-12, 1e12 } )
        {
            problem.set_affine_shift( alpha );
            problem.apply( *state, *result );
            const auto actual = operations->get_value_at_point( Equation::size - 1, *result );
            require(
                std::abs( actual - ( monitor_type::constraint( point, .4 ) - .1 ) ) < 2e-15,
                "Balanced scaling hid an algebraic constraint or changed its known RHS"
            );
        }
    }
    auto       problem = std::make_shared<problem_type>( operations );
    const auto initial = exact_state<Equation>( .25 );
    operations->set( initial.d, *state, Equation::size );
    for ( const double h : { -1e-6, 1e-6, 10. } )
    {
        auto   step = make_step( operations, problem, std::abs( h ), "IE" );
        double end  = .25 + h;
        step->set_time( .25 );
        step->set_target_time( end );
        step->apply( *state, *result );
        require( step->get_status() == single_step_status::converged, "Balanced small/large-step DAE solve failed" );
        monitor_type monitor{ operations };
        monitor.set_time_interval( .25, end );
        monitor.apply( *state, *result );
        step->finalize( adaptation_status::accepted, end, *result );
    }
}

template <class Equation>
void test_mass( const scfd::static_vec::vec<double, Equation::size> &point, const char *label, double start = 0 )
{
    using problem_type = stage_problem<Equation>;
    using types        = newton_types<problem_type>;
    using monitor_type = constraint_monitor<Equation>;
    auto operations    = std::make_shared<operations_type>( Equation::size );
    auto problem       = std::make_shared<problem_type>( operations );
    for ( const double time : { 0., .4, .9 } )
        for ( const double alpha : { -1e6, -.01, 0., .01, 1e6 } )
            check_jacobian( operations, *problem, point, time, alpha );
    vector_wrap_type initial( *operations ), result( *operations );
    initial.start_use();
    result.start_use();
    const auto initial_values = exact_state<Equation>( start );
    const auto reference      = exact_state<Equation>( start + .4 );
    operations->set( initial_values.d, *initial, Equation::size );
    for ( const char *method : { "IE", "SDIRK2(1)2" } )
    {
        double errors[2]{};
        for ( int level = 0; level < 2; ++level )
        {
            auto step    = make_step( operations, problem, .04 / ( level + 1 ), method );
            auto monitor = std::make_shared<monitor_type>( monitor_type{ operations } );
            if constexpr ( std::is_same_v<Equation, coupled_dae> || std::is_same_v<Equation, stokes_dae> ||
                           std::is_same_v<Equation, moving_constraint_dae> )
            {
                integration::time_integrator<operations_type, typename types::step_type, monitor_type> integrator(
                    operations, step, {}, monitor
                );
                integrator.set_time_interval( start, start + .4 );
                for ( int run = 0; run < 2; ++run )
                {
                    integrator.apply( *initial, *result );
                    require(
                        integrator.get_status() == integration_status::completed && monitor->calls > 0,
                        "Coupled DAE integration or restart failed"
                    );
                }
            }
            else
            {
                integration::time_integrator<operations_type, typename types::step_type> integrator( operations, step );
                integrator.set_time_interval( start, start + .4 );
                for ( int run = 0; run < 2; ++run )
                {
                    integrator.apply( *initial, *result );
                    require(
                        integrator.get_status() == integration_status::completed, "Mass integration or restart failed"
                    );
                }
            }
            typename problem_type::state_type actual;
            operations->get( *result, actual.d, Equation::size );
            for ( int i = 0; i < Equation::size; ++i )
                errors[level] = std::max( errors[level], std::abs( actual[i] - reference[i] ) );
        }
        require( errors[1] < .6 * errors[0], "Mass/DAE refinement did not improve accuracy" );
    }
    if constexpr ( std::is_same_v<Equation, coupled_dae> || std::is_same_v<Equation, stokes_dae> ||
                   std::is_same_v<Equation, moving_constraint_dae> )
    {
        test_dae_scaling<Equation>( operations, point );
        test_adaptive_system(
            Equation{}, initial_values, reference, start, start + .4, label, operations,
            std::make_shared<monitor_type>( monitor_type{ operations } )
        );
    }
    else
        test_adaptive_system( Equation{}, initial_values, reference, start, start + .4, label, operations );
}

int main()
try
{
    backend_type::init_device();
    test_mass<coupled_mass>( { .7, .4 }, "Coupled mass" );
    test_mass<variable_mass>( { .7 }, "State-dependent mass" );
    test_mass<time_dependent_mass>( { .7 }, "Time-dependent mass" );
    test_mass<coupled_dae>( { .7, .4 }, "Nonlinear index-1 DAE" );
    test_mass<moving_constraint_dae>( { .7, .4 }, "Moving index-1 DAE" );
    test_mass<moving_constraint_dae>( { .7, .4 }, "Shifted moving index-1 DAE", .25 );
    test_mass<stokes_dae>( { .7, -.7, .2 }, "Stokes index-2 DAE" );
    std::cout << "DIRK coupled/state/time-dependent mass, moving constraints, index-1 and Stokes index-2 DAEs: PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
