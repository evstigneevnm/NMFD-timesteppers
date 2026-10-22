#include "implicit_test_support.h"
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
         double, double alpha, const vector_type &x, const vector_type &previous, const vector_type &rhs,
         vector_type &out
     ) const
    {
        const auto d0 = x( 0 ) - previous( 0 ), d1 = x( 1 ) - previous( 1 );
        out( 0 ) = alpha * ( 2 * d0 + d1 ) + 2 * x( 0 ) + 2 * x( 1 ) - rhs( 0 );
        out( 1 ) = alpha * ( d0 + 3 * d1 ) + x( 0 ) + 6 * x( 1 ) - rhs( 1 );
    }
    auto jacobian(
        double, double a, const scfd::static_vec::vec<double, 2> &, const scfd::static_vec::vec<double, 2> &
    ) const
    {
        return scfd::static_mat::mat<double, 2, 2>{ 2 * a + 2, a + 2, a + 1, 3 * a + 6 };
    }
};

// variable_mass: (1+u^2)*u'=-(1+u^2)*u, u(0)=1.
// The mass is state-dependent and strictly positive; the exact trajectory is u=exp(-t).
// The stage residual is nonlinear, and its Jacobian must include the mass derivative.
struct variable_mass
{
    static constexpr int size = 1;
    __DEVICE_TAG__ void  residual(
         double, double alpha, const vector_type &x, const vector_type &previous, const vector_type &rhs,
         vector_type &out
     ) const
    {
        const auto u = x( 0 );
        out( 0 )     = ( 1 + u * u ) * ( alpha * ( u - previous( 0 ) ) + u ) - rhs( 0 );
    }
    auto jacobian(
        double, double a, const scfd::static_vec::vec<double, 1> &x, const scfd::static_vec::vec<double, 1> &previous
    ) const
    {
        return scfd::static_mat::mat<double, 1, 1>{
            a * ( 1 + x[0] * x[0] ) + 2 * a * x[0] * ( x[0] - previous[0] ) + 1 + 3 * x[0] * x[0]
        };
    }
};

// (1+t)*u'=-u, u(0)=1; exact u=1/(1+t). Positive, explicitly time-dependent mass.
// The full stage Jacobian is alpha*(1+t)+1, evaluated at the current stage time.
using time_dependent_mass = scalar_time_equation<time_mass_decay<double>>;

// coupled_dae: x'=-x-z+x^2, 0=z-x^2, with initial state (1,1).
// Nonlinear autonomous index-1 DAE with singular mass M=diag(1,0).
// Exact solution: (x,z)=(exp(-t),exp(-2*t)); checks the constraint at accepted endpoints.
struct coupled_dae
{
    static constexpr int size = 2;
    __DEVICE_TAG__ void  residual(
         double, double alpha, const vector_type &x, const vector_type &previous, const vector_type &rhs,
         vector_type &out
     ) const
    {
        out( 0 ) = alpha * ( x( 0 ) - previous( 0 ) ) + x( 0 ) + x( 1 ) - x( 0 ) * x( 0 ) - rhs( 0 );
        out( 1 ) = x( 1 ) - x( 0 ) * x( 0 ) - rhs( 1 );
    }
    auto jacobian(
        double, double a, const scfd::static_vec::vec<double, 2> &x, const scfd::static_vec::vec<double, 2> &
    ) const
    {
        return scfd::static_mat::mat<double, 2, 2>{ a + 1 - 2 * x[0], 1, -2 * x[0], 1 };
    }
};

// x'=-x-z+sin(t)*x, 0=z-sin(t)*x, with consistent initial state (1,0).
// Non-autonomous index-1 DAE, M=diag(1,0); the constraint manifold moves with time.
// Exact solution (x,z)=(exp(-t),exp(-t)*sin(t)); verifies stage times and endpoint consistency.
struct moving_constraint_dae
{
    static constexpr int size = 2;
    __DEVICE_TAG__ void  residual(
         double t, double alpha, const vector_type &x, const vector_type &previous, const vector_type &rhs,
         vector_type &out
     ) const
    {
        out( 0 ) = alpha * ( x( 0 ) - previous( 0 ) ) + ( 1 - std::sin( t ) ) * x( 0 ) + x( 1 ) - rhs( 0 );
        out( 1 ) = x( 1 ) - std::sin( t ) * x( 0 ) - rhs( 1 );
    }
    auto jacobian(
        double t, double alpha, const scfd::static_vec::vec<double, 2> &, const scfd::static_vec::vec<double, 2> &
    ) const
    {
        return scfd::static_mat::mat<double, 2, 2>{ alpha + 1 - std::sin( t ), 1, -std::sin( t ), 1 };
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
         double t, double alpha, const vector_type &x, const vector_type &previous, const vector_type &rhs,
         vector_type &out
     ) const
    {
        out( 0 ) = alpha * ( x( 0 ) - previous( 0 ) ) + x( 0 ) + x( 2 ) - std::sin( t ) - rhs( 0 );
        out( 1 ) = alpha * ( x( 1 ) - previous( 1 ) ) + 2 * x( 1 ) + x( 2 ) + std::exp( -t ) - std::sin( t ) - rhs( 1 );
        out( 2 ) = x( 0 ) + x( 1 ) - rhs( 2 );
    }
    auto jacobian(
        double, double a, const scfd::static_vec::vec<double, 3> &, const scfd::static_vec::vec<double, 3> &
    ) const
    {
        return scfd::static_mat::mat<double, 3, 3>{ a + 1, 0, 1, 0, a + 2, 1, 1, 1, 0 };
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
        double constraint = 0;
        if constexpr ( std::is_same_v<Equation, stokes_dae> )
            constraint = x[0] + x[1];
        else if constexpr ( std::is_same_v<Equation, moving_constraint_dae> )
            constraint = x[1] - std::sin( time ) * x[0];
        else
            constraint = x[1] - x[0] * x[0];
        require( std::abs( constraint ) < 2e-9, "DAE constraint violated at accepted endpoint" );
        ++calls;
        return false;
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

template <class Equation>
void test_mass( const scfd::static_vec::vec<double, Equation::size> &point, double start = 0 )
{
    using problem_type = stage_problem<Equation>;
    using types        = newton_types<problem_type>;
    using monitor_type = constraint_monitor<Equation>;
    auto operations    = std::make_shared<operations_type>( Equation::size );
    auto problem       = std::make_shared<problem_type>( operations );
    for ( const double time : { 0., .4, .9 } )
        check_jacobian( operations, *problem, point, time );
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
}

int main()
try
{
    backend_type::init_device();
    test_mass<coupled_mass>( { .7, .4 } );
    test_mass<variable_mass>( { .7 } );
    test_mass<time_dependent_mass>( { .7 } );
    test_mass<coupled_dae>( { .7, .4 } );
    test_mass<moving_constraint_dae>( { .7, .4 } );
    test_mass<moving_constraint_dae>( { .7, .4 }, .25 );
    test_mass<stokes_dae>( { .7, -.7, .2 } );
    std::cout << "DIRK coupled/state/time-dependent mass, moving constraints, index-1 and Stokes index-2 DAEs: PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
