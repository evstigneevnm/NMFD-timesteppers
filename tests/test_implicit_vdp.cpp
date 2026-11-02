#include "implicit_vdp_problem.h"

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;

int main()
try
{
    backend_type::init_device();
    test_ode( vdp_equation{}, { 2, 0 }, { 1.5081442369756097, -.78021807462969694 }, 1, .01 );
    test_adaptive_system(
        vdp_equation{}, { 2, 0 }, { 1.5081442369756097, -.78021807462969694 }, 0, 1, "Van der Pol mu=1"
    );
    vdp_equation stiff;
    stiff.mu = 1000;
    // SciPy Radau and DOP853 independently agree to 1e-15 at t=.02.
    test_ode( stiff, { 2, 0 }, { 1.9999868888196612, -.00066667382728169422 }, .02, .002 );
    const auto stiff_error = test_adaptive_system(
        stiff, { 2, 0 }, { 1.9999868888196612, -.00066667382728169422 }, 0, .02, "Van der Pol mu=1000"
    );
    require( stiff_error < 5e-9, "Stiff Van der Pol lost its small velocity component" );
    std::cout << "Implicit Van der Pol, stiff mu=1000, Jacobian, reference refinement and restart: PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
