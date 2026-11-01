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
         double t, const stage_scaling &scaling, const vector_type &x, const vector_type &previous,
         const vector_type &rhs, vector_type &out
     ) const
    {
        out( 0 ) = scaling.mass * ( x( 0 ) - previous( 0 ) ) + scaling.rhs * ( x( 0 ) - t - rhs( 0 ) );
    }
    auto jacobian(
        double, const stage_scaling &scaling, const scfd::static_vec::vec<double, 1> &,
        const scfd::static_vec::vec<double, 1> &
    ) const
    {
        return scfd::static_mat::mat<double, 1, 1>{ scaling.mass + scaling.rhs };
    }
};

// Exercise both scaling limits without changing the underlying stage equation.
void test_balanced_scaling()
{
    auto             operations = std::make_shared<operations_type>( 1 );
    vector_wrap_type state( *operations ), residual( *operations );
    state.start_use();
    residual.start_use();
    operations->assign_scalar( .7, *state );
    for ( const double reference_time : { .25, 1., 4. } )
    {
        stage_problem<forced_decay> problem( operations, {}, reference_time );
        problem.set_time( .4 );
        operations->assign_scalar( .2, problem.previous_state() );
        operations->assign_scalar( .1, problem.known_rhs() );
        for ( const double alpha : { -1e12, -7., -1., -1e-12, 0., 1e-12, 1., 7., 1e12 } )
        {
            problem.set_affine_shift( alpha );
            const auto scaling = problem.scaling();
            const auto scale   = std::max( std::abs( static_cast<long double>( alpha ) ), 1.L / reference_time );
            require(
                std::abs( scaling.mass ) <= 1 && scaling.rhs > 0 && scaling.rhs <= reference_time &&
                    std::abs( scaling.mass - alpha / scale ) < 2e-15 &&
                    std::abs( scaling.rhs - 1 / scale ) < 2e-15 * reference_time,
                "Stage scaling is unbounded or changes the reference timescale"
            );
            problem.apply( *state, *residual );
            const long double physical = static_cast<long double>( alpha ) * ( .7L - .2L ) + .7L - .4L - .1L;
            require(
                std::abs( operations->get_value_at_point( 0, *residual ) - physical / scale ) < 2e-15,
                "Balanced residual changed the equation or omitted the known RHS"
            );
            require(
                problem.scaling().mass == scaling.mass && problem.scaling().rhs == scaling.rhs,
                "Residual trial changed frozen stage scaling"
            );
            check_jacobian( operations, problem, { .7 }, .4, alpha );
        }
        const auto before = problem.scaling();
        for ( const auto invalid :
              { std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() } )
        {
            bool rejected = false;
            try
            {
                problem.set_affine_shift( invalid );
            }
            catch ( const std::invalid_argument & )
            {
                rejected = true;
            }
            require(
                rejected && problem.alpha == 1e12 && problem.scaling().mass == before.mass &&
                    problem.scaling().rhs == before.rhs,
                "Invalid shift changed prepared stage scaling"
            );
        }
    }
    for ( const auto invalid :
          { 0., -1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::denorm_min() } )
    {
        bool rejected = false;
        try
        {
            stage_problem<forced_decay> problem( operations, {}, invalid );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        require( rejected, "Stage scaling accepted an invalid reference time" );
    }

    // Frozen-time pseudo-transient stages approach f+r=0 as h grows. Check that
    // physical balance, not the h-dependent scaled norm, measures progress.
    using problem_type = stage_problem<forced_decay>;
    auto problem       = std::make_shared<problem_type>( operations );
    auto solver        = newton_types<problem_type>::make_solver( operations );
    solver->set_nonlinear_operator( problem );
    problem->set_time( .4 );
    operations->assign_scalar( .1, problem->known_rhs() );
    operations->assign_scalar( 1., *state );
    double physical_error = .5;
    for ( const double alpha : { 1., 1e-3, 1e-6, 0. } )
    {
        operations->assign( *state, problem->previous_state() );
        problem->set_affine_shift( alpha );
        require( solver->solve( *state ), "Large-step or steady-limit Newton stage failed" );
        const auto error = std::abs( operations->get_value_at_point( 0, *state ) - .5 );
        require( error < physical_error, "Pseudo-transient stage did not reduce the physical residual" );
        physical_error = error;
    }
    require( physical_error < 2e-10, "Steady-limit stage did not solve the physical equation" );
}

int main()
try
{
    backend_type::init_device();
    test_balanced_scaling();
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
