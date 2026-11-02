#include "implicit_vdp_problem.h"
#include "implicit_steady_state_test.h"

using namespace nmfd::time_steppers::tests;

// Van der Pol: x'=v, v'=mu*(1-x^2)*v-x. The unique equilibrium is (0,0).
// Its eigenvalues solve lambda^2-mu*lambda+1=0: an unstable focus for mu=1,
// an unstable node with rates about .001 and 1000 for mu=1000.
// Backward Euler damps a positive real rate only when h*lambda>2. Thus the
// stiff root search needs very large pseudo-steps, unlike accurate integration
// toward the physical limit cycle. Both residual and state error are checked.
// The stiff Jacobian's inverse amplifies residuals by about 1000, so that case
// needs a tighter physical steady tolerance to resolve the state within 1e-6.
// 5e-10 stays above the unchanged Newton absolute tolerance (2e-10).
int main()
try
{
    backend_type::init_device();
    for ( const double mu : { 1., 1000. } )
    {
        vdp_equation equation;
        equation.mu = mu;
        test_unstable_equilibrium(
            equation, { .01, .005 }, { 0, 0 }, mu == 1 ? 10 : 1e6, mu == 1 ? "Van der Pol mu=1" : "Van der Pol mu=1000",
            mu == 1 ? 1e-8 : 5e-10
        );
    }
    std::cout << "Van der Pol unstable steady states: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
