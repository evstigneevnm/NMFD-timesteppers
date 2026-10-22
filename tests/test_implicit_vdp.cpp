#include "implicit_ode_test.h"
#include "explicit_vdp_problem.h"

using namespace nmfd::time_steppers::tests;

// Van der Pol: x'=v, v'=mu*(1-x^2)*v-x; initial state (2,0), identity mass.
// Autonomous nonlinear oscillator; mu=1 is tested over [0,1].
// The stiff case mu=1000 is tested over [0,0.02], covering fast relaxation and early slow motion.
// References are independently computed; the stiff reference is cross-checked with Radau/DOP853.
// Checks Jacobians, refinement, and restart, not accuracy over a complete relaxation cycle.
struct vdp_equation : explicit_vdp_problem<operations_type>
{
    static constexpr int size = 2;
    __DEVICE_TAG__ void  residual(
         double, double alpha, const vector_type &x, const vector_type &previous, const vector_type &rhs,
         vector_type &out
     ) const
    {
        identity_mass_residual<2>( mapping{ x, out, mu }, alpha, previous, rhs );
    }
    auto jacobian(
        double, double alpha, const scfd::static_vec::vec<double, 2> &x, const scfd::static_vec::vec<double, 2> &
    ) const
    {
        return scfd::static_mat::mat<double, 2, 2>{
            alpha, -1, 1 + 2 * mu * x[0] * x[1], alpha - mu * ( 1 - x[0] * x[0] )
        };
    }
};

int main()
try
{
    backend_type::init_device();
    test_ode( vdp_equation{}, { 2, 0 }, { 1.5081442369756097, -.78021807462969694 }, 1, .01 );
    vdp_equation stiff;
    stiff.mu = 1000;
    // SciPy Radau and DOP853 independently agree to 1e-15 at t=.02.
    test_ode( stiff, { 2, 0 }, { 1.9999868888196612, -.00066667382728169422 }, .02, .002 );
    std::cout << "Implicit Van der Pol, stiff mu=1000, Jacobian, reference refinement and restart: PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
