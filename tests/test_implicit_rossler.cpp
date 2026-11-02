#include "implicit_rossler_problem.h"

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;

int main()
try
{
    backend_type::init_device();
    test_ode(
        rossler_equation{}, { 2, 0, 0 }, { .98415292572457735, 1.8475471904515657, .045002515579204969 }, 1, .01
    );
    test_adaptive_system(
        rossler_equation{}, { 2, 0, 0 }, { .98415292572457735, 1.8475471904515657, .045002515579204969 }, 0, 1,
        "Rossler"
    );
    std::cout << "Implicit Rossler, analytical Jacobian, reference refinement and restart: PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
