#include "implicit_bdf_adaptive_test.h"

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;
namespace model = nmfd::time_steppers::tests::bdf_test;

// u'=-kappa*(u-cos(t))-sin(t), u=cos(t). A strongly damped mode and
// smooth forcing test stiff tracking rather than accepting a vanishing solution.
struct stiff_tracking_equation
{
    double                kappa;
    __DEVICE_TAG__ double mass( double ) const
    {
        return 1;
    }
    __DEVICE_TAG__ double rhs( double t, double u ) const
    {
        return -kappa * ( u - std::cos( t ) ) - std::sin( t );
    }
    __DEVICE_TAG__ double rhs_derivative( double, double ) const
    {
        return -kappa;
    }
};

// x'=-d*x-w*y, y'=w*x-d*y; eigenvalues -d +/- i*w.
// The known decaying rotation exposes errors in weakly damped oscillatory modes,
// where BDF3-5 are not A-stable. It does not establish unconditional stability.
struct damped_rotation
{
    static constexpr int size    = 2;
    double               damping = .02, frequency = 20;
    __DEVICE_TAG__ void  residual(
         double, const stage_scaling &s, const vector_type &x, const vector_type &old, const vector_type &rhs,
         vector_type &out
     ) const
    {
        out( 0 ) = s.mass * ( x( 0 ) - old( 0 ) ) + s.rhs * ( damping * x( 0 ) + frequency * x( 1 ) - rhs( 0 ) );
        out( 1 ) = s.mass * ( x( 1 ) - old( 1 ) ) + s.rhs * ( damping * x( 1 ) - frequency * x( 0 ) - rhs( 1 ) );
    }
    auto jacobian(
        double, const stage_scaling &s, const scfd::static_vec::vec<double, 2> &,
        const scfd::static_vec::vec<double, 2> &
    ) const
    {
        return scfd::static_mat::mat<double, 2, 2>{
            s.mass + s.rhs * damping, s.rhs * frequency, -s.rhs * frequency, s.mass + s.rhs * damping
        };
    }
    scfd::static_vec::vec<double, 2> exact( double t ) const
    {
        return {
            std::exp( -damping * t ) * std::cos( frequency * t ), std::exp( -damping * t ) * std::sin( frequency * t )
        };
    }
};

int main()
try
{
    backend_type::init_device();
    using controller = integration::time_step_adaptation_scipy<operations_type>;
    auto scalar_ops  = std::make_shared<operations_type>( 1 );
    for ( const double stiffness : { 100., 10000. } )
    {
        const auto error = model::adaptive_trajectory<controller>(
            scalar_time_equation<stiff_tracking_equation>{ stiff_tracking_equation{ stiffness } }, { 1. },
            { std::cos( 2. ) }, 0, 2, "BDF", "Stiff tracking", 1e-7, scalar_ops
        );
        require( error < 2e-6, "Automatic BDF failed stiff tracking" );
    }
    auto            ops = std::make_shared<operations_type>( 2 );
    damped_rotation equation;
    for ( unsigned int cap : { 3u, 4u, 5u } )
    {
        const auto error = model::adaptive_trajectory<controller>(
            equation, equation.exact( .25 ), equation.exact( 1.25 ), .25, 1.25, "BDF", "Damped rotation", 1e-8, ops,
            std::shared_ptr<detail::no_external_operations>{}, cap
        );
        require( error < .0002, "Automatic BDF weakly damped oscillation error" );
    }
    std::cout << "Automatic BDF stiff tracking, weakly damped rotations and order caps: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
