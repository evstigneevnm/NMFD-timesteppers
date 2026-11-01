#include "implicit_test_support.h"
#include "nonautonomous_test_problems.h"

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;

// u'=t*u, exact exp(t^2/2): the balanced Jacobian a-b*t must follow the stage time.
// (1+t)*u'=-u, exact 1/(1+t): verifies an explicitly time-dependent positive mass.
// u'=-kappa*(u-sin(t))+cos(t), exact sin(t): tests stiff non-autonomous forcing.
// Runs start at t=.25 with exact data. Large kappa tests bounded error and refinement,
// not uniform classical order in the stiff limit, where order reduction can occur.
template <class Equation>
void test_equation( const Equation &equation, bool stiff = false )
{
    using stage_equation = scalar_time_equation<Equation>;
    using problem_type   = stage_problem<stage_equation>;
    using types          = newton_types<problem_type>;
    auto operations      = std::make_shared<operations_type>( 1 );
    auto problem         = std::make_shared<problem_type>( operations, stage_equation{ equation } );
    for ( const double time : { .25, .6, 1.25 } )
        for ( const double alpha : { 7., 19. } )
            check_jacobian( operations, *problem, { .7 }, time, alpha );
    vector_wrap_type initial( *operations ), result( *operations );
    initial.start_use();
    result.start_use();
    operations->assign_scalar( equation.exact( .25 ), *initial );
    for ( const char *method : { "IE", "IM", "SDIRK2(1)2", "SDIRK3(1)3" } )
    {
        double errors[3]{};
        for ( int level = 0; level < 3; ++level )
        {
            auto step = make_step( operations, problem, .1 / ( 1 << level ), method );
            integration::time_integrator<operations_type, typename types::step_type> integrator( operations, step );
            integrator.set_time_interval( .25, 1.25 );
            integrator.apply( *initial, *result );
            require(
                integrator.get_status() == integration_status::completed && integrator.get_final_time() == 1.25,
                "Implicit shifted-time integration failed"
            );
            const auto actual = operations->get_value_at_point( 0, *result );
            errors[level]     = std::abs( actual - equation.exact( 1.25 ) );
            integrator.apply( *initial, *result );
            require(
                integrator.get_status() == integration_status::completed &&
                    operations->get_value_at_point( 0, *result ) == actual,
                "Implicit shifted-time restart failed"
            );
        }
        require( errors[2] < std::max( 2e-10, .85 * errors[1] ), "Implicit non-autonomous refinement failed" );
        require(
            errors[2] < .03 * std::max( 1., std::abs( equation.exact( 1.25 ) ) ),
            "Implicit non-autonomous analytical solution mismatch"
        );
        if ( !stiff && errors[2] > 2e-10 )
        {
            const auto order = runge_kutta::make_butcher_table( method ).order();
            require( std::log2( errors[1] / errors[2] ) > order - .3, "Implicit non-autonomous order mismatch" );
        }
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
    std::cout << "Stiff tracking: kappa=100\n";
    test_equation( stiff_tracking<double>{ 100 }, true );
    std::cout << "Stiff tracking: kappa=10000\n";
    test_equation( stiff_tracking<double>{ 10000 }, true );
    std::cout
        << "Implicit time-dependent coefficients/mass, stiff tracking, timed Jacobians and shifted restart: PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
