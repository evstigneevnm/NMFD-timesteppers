#include "implicit_bdf_adaptive_test.h"

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;
namespace model = nmfd::time_steppers::tests::bdf_test;

// u'=t-u, u=t-1+2*exp(-t). The nonlinear solve uses NMFD Newton/GMRES;
// complete cold-start trajectories include the low-order startup contribution.
template <class Controller>
void test_controller( const char *name )
{
    auto ops = std::make_shared<operations_type>( 1 );
    for ( const double direction : { 1., -1. } )
    {
        const double start = .25, end = start + direction * 2;
        double       errors[2]{};
        for ( int level = 0; level < 2; ++level )
            errors[level] = model::adaptive_trajectory<Controller>(
                scalar_time_equation<model::forced_decay>{}, { model::forced_decay::exact( start ) },
                { model::forced_decay::exact( end ) }, start, end, "BDF", name, level ? 2e-7 : 2e-3, ops
            );
        require( errors[1] < .2 * errors[0] && errors[1] < .0002, "Automatic BDF startup/refinement error" );
    }
}

int main()
try
{
    backend_type::init_device();
    test_controller<integration::time_step_adaptation_matlab<operations_type>>( "MATLAB" );
    test_controller<integration::time_step_adaptation_scipy<operations_type>>( "SciPy" );
    for ( const char *name : { "I", "H211", "PC", "PID", "H312", "PPID", "H321" } )
        test_controller<integration::time_step_adaptation_filter<operations_type>>( name );
    std::cout << "Automatic BDF all controllers, cold startup, tolerance refinement and repeated trajectories: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
