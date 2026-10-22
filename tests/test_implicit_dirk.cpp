#include "implicit_test_support.h"
#include <limits>

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;

// Forced decay: u'=-u+t, u(0)=1; exact solution u=t-1+2*exp(-t).
// Linear, non-autonomous ODE with identity mass and an exponentially decaying homogeneous mode.
// Verifies stage times, convergence orders, unequal DIRK diagonals, and endpoint handling.
// Also checks backward integration and rejection of unsupported tables.
struct forced_decay
{
    static constexpr int size = 1;
    __DEVICE_TAG__ void  residual(
         double t, double alpha, const vector_type &x, const vector_type &previous, const vector_type &rhs,
         vector_type &out
     ) const
    {
        out( 0 ) = alpha * ( x( 0 ) - previous( 0 ) ) + x( 0 ) - t - rhs( 0 );
    }
    auto jacobian(
        double, double alpha, const scfd::static_vec::vec<double, 1> &, const scfd::static_vec::vec<double, 1> &
    ) const
    {
        return scfd::static_mat::mat<double, 1, 1>{ alpha + 1 };
    }
};

int main()
try
{
    backend_type::init_device();
    using problem_type          = stage_problem<forced_decay>;
    using types                 = newton_types<problem_type>;
    auto             operations = std::make_shared<operations_type>( 1 );
    auto             problem    = std::make_shared<problem_type>( operations );
    vector_wrap_type initial( *operations ), result( *operations );
    initial.start_use();
    result.start_use();
    operations->assign_scalar( 1., *initial );
    check_jacobian( operations, *problem, { .7 } );
    const runge_kutta::butcher_table unequal_stiff( { { .25L, 0 }, { 2.L / 3, 1.L / 3 } }, { 2.L / 3, 1.L / 3 }, 2 );
    const runge_kutta::butcher_table unequal_general( { { .25L, 0 }, { .25L, .5L } }, { .5L, .5L }, 2 );
    const auto                       tables = {
        runge_kutta::make_butcher_table( "IE" ),
        runge_kutta::make_butcher_table( "IM" ),
        runge_kutta::make_butcher_table( "SDIRK2(1)2" ),
        runge_kutta::make_butcher_table( "SDIRK3(1)3" ),
        unequal_stiff,
        unequal_general
    };
    for ( const auto &table : tables )
    {
        double errors[2]{};
        for ( int resolution = 0; resolution < 2; ++resolution )
        {
            auto adaptation = std::make_shared<adaptation_type>( adaptation_type::params{ .1 / ( resolution + 1 ) } );
            auto solver     = types::make_solver( operations );
            auto step       = std::make_shared<types::step_type>( operations, problem, adaptation, solver, table );
            integration::time_integrator<operations_type, types::step_type> integrator( operations, step );
            integrator.set_time_interval( 0, .8 );
            integrator.apply( *initial, *result );
            if ( integrator.get_status() != integration_status::completed )
                std::cerr << "t=" << integrator.get_final_time() << " dt=" << step->get_dt()
                          << " step=" << int( step->get_status() )
                          << " Newton=" << solver->convergence_strategy()->get_result_status() << '\n';
            require( integrator.get_status() == integration_status::completed, "DIRK integration failed" );
            require( integrator.get_final_time() == .8, "DIRK endpoint was not clipped exactly" );
            errors[resolution] =
                std::abs( operations->get_value_at_point( 0, *result ) - ( -.2 + 2 * std::exp( -.8 ) ) );
            integrator.apply( *initial, *result );
            require( integrator.get_status() == integration_status::completed, "DIRK repeated solve failed" );
        }
        const auto order = std::log2( errors[0] / errors[1] );
        require( order > table.order() - .15 && order < table.order() + .2, "DIRK analytical order mismatch" );
        std::cout << "DIRK order " << table.order() << ": observed " << order << '\n';
    }
    auto                                                            step = make_step( operations, problem, .03 );
    integration::time_integrator<operations_type, types::step_type> integrator( operations, step );
    integrator.set_time_interval( .8, 0 );
    operations->assign_scalar( -.2 + 2 * std::exp( -.8 ), *initial );
    integrator.apply( *initial, *result );
    require(
        integrator.get_status() == integration_status::completed &&
            std::abs( operations->get_value_at_point( 0, *result ) - 1 ) < 3e-4,
        "DIRK backward integration failed"
    );

    for ( const char *name : { "EE", "CN", "ESDIRK2(1)3" } )
    {
        bool rejected = false;
        try
        {
            auto invalid = make_step( operations, problem, .1, name );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        require( rejected, "DIRK accepted a zero-diagonal stage" );
    }
    bool rejected = false;
    try
    {
        runge_kutta::butcher_table coupled( { { .25L, .25L }, { .25L, .25L } }, { .5L, .5L }, 2 );
        auto                       invalid = std::make_shared<types::step_type>(
            operations, problem, std::make_shared<adaptation_type>(), types::make_solver( operations ), coupled
        );
    }
    catch ( const std::invalid_argument & )
    {
        rejected = true;
    }
    require( rejected, "DIRK accepted a coupled IRK table" );
    std::cout << "DIRK non-autonomous order, unequal diagonals, midpoint, backward and table validation: PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
