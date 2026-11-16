#include "implicit_rosenbrock_projection_problem.h"
using namespace projection_tests;
using O = space_type;
using P = rosenbrock_projection_problem<O>;
using L = projection_linear_solver<O>;

// Compare every endpoint, embedded defect and dense sample with independently
// reduced equations. Test exact, zero, scaled, lagged and noncommuting W
// Jacobians; all compatible approximations have P*B*(I-P)=0.
double trajectory( const char *method, double h, bool nonlinear, int approximation = 0 )
{
    using C  = rosenbrock_recording_control<O>;
    using S  = rosenbrock_projected_step<O, C, true>;
    using R  = rosenbrock::time_step<O, P, C, L, true>;
    auto ops = std::make_shared<O>( 3 );
    auto p = std::make_shared<P>( ops ), r = std::make_shared<P>( ops, true );
    p->nonlinear = r->nonlinear = nonlinear;
    p->factor = r->factor = approximation == 1 ? 0 : approximation == 2 ? .5 : 1;
    p->noncommuting = r->noncommuting = approximation == 3;
    p->lagged = r->lagged = approximation == 4;
    auto               c = std::make_shared<C>( ops, std::abs( h ) ), rc = std::make_shared<C>( ops, std::abs( h ) );
    auto               linear = std::make_shared<L>( ops ), rl = std::make_shared<L>( ops );
    auto               projection = std::make_shared<projector<O>>( ops );
    typename S::params sp{ method };
    sp.autonomous = false;
    typename R::params rp{ method };
    rp.autonomous = false;
    S         step( ops, p, c, linear, sp, projection );
    R         reference( ops, r, rc, rl, rp );
    buffer<O> state( *ops ), ref( *ops ), out( *ops ), expected( *ops ), analytic( *ops ), sample( *ops ),
        rsample( *ops ), scratch( *ops );
    const double start = .3, end = start + std::copysign( .4, h );
    double       t = start, error = 0;
    rosenbrock_exact( *ops, *state, t, nonlinear );
    ops->assign( *state, *ref );
    for ( int j = 0; j < int( std::lround( .4 / std::abs( h ) ) ); ++j )
    {
        step.set_time( t );
        step.set_target_time( end );
        reference.set_time( t );
        reference.set_target_time( end );
        step.apply( *state, *out );
        reference.apply( *ref, *expected );
        check(
            step.get_status() == single_step_status::converged &&
                reference.get_status() == single_step_status::converged,
            "Projected/reduced Rosenbrock failed"
        );
        check(
            difference( *ops, *out, *expected, *scratch ) < 2e-12, "Projected Rosenbrock differs from reduced equations"
        );
        check( difference( *ops, *c->error, *rc->error, *scratch ) < 2e-12, "Projected embedded estimate differs" );
        check( c->exponent == step.table().error_order(), "Projected estimator changed its exponent" );
        check( constraint( *ops, *out, *scratch ) < 2e-13, "Rosenbrock endpoint left projector range" );
        const auto solves = linear->calls, rates = p->rates, calls = projection->calls;
        auto       view = step.get_continuous_integration();
        auto       rv   = reference.get_continuous_integration();
        for ( double theta : { 0., .2, .6, 1. } )
        {
            const double time = view.evaluate( theta, *sample );
            rv.evaluate( theta, *rsample );
            check(
                difference( *ops, *sample, *rsample, *scratch ) < 3e-12,
                "Projected dense formula differs from reduced equations"
            );
            check( constraint( *ops, *sample, *scratch ) < 3e-13, "Rosenbrock dense sample left projector range" );
            rosenbrock_exact( *ops, *analytic, time, nonlinear );
            error = std::max( error, difference( *ops, *sample, *analytic, *scratch ) );
        }
        check(
            solves == linear->calls && rates == p->rates && calls == projection->calls,
            "Dense sampling performed a solve/rate/projection"
        );
        check( projection->calls == linear->calls, "A successful Rosenbrock solve escaped projection" );
        check( p->time == t, "Projection or dense preparation changed frozen base time" );
        const double next = step.get_dt() == end - t ? end : t + step.get_dt();
        step.finalize( adaptation_status::accepted, next, *out );
        reference.finalize( adaptation_status::accepted, next, *expected );
        ops->assign( *out, *state );
        ops->assign( *expected, *ref );
        t = next;
    }
    return error;
}

double adaptive( const char *method, double tolerance )
{
    using C         = adaptive_control<O>;
    using S         = rosenbrock_projected_step<O, C, true>;
    auto ops        = std::make_shared<O>( 3 );
    auto p          = std::make_shared<P>( ops );
    p->nonlinear    = true;
    p->noncommuting = std::string( method ) == "ROS34PW2";
    typename C::params cp;
    cp.initial_step       = .1;
    cp.maximum_step       = .2;
    cp.relative_tolerance = tolerance;
    cp.absolute_tolerance = tolerance * .01;
    typename S::params sp{ method };
    sp.autonomous        = false;
    auto      projection = std::make_shared<projector<O>>( ops );
    S         step( ops, p, std::make_shared<C>( ops, cp ), std::make_shared<L>( ops ), sp, projection );
    buffer<O> state( *ops ), expected( *ops ), scratch( *ops );
    double    t = .3;
    rosenbrock_exact( *ops, *state, t, true );
    for ( int j = 0; t != 1.3 && j < 10000; ++j )
    {
        step.set_time( t );
        step.set_target_time( 1.3 );
        step.apply( *state, *state );
        check( step.get_status() == single_step_status::converged, "Adaptive projected Rosenbrock failed" );
        t = step.get_dt() == 1.3 - t ? 1.3 : t + step.get_dt();
        step.finalize( adaptation_status::accepted, t, *state );
    }
    check( t == 1.3, "Adaptive projected Rosenbrock budget exhausted" );
    rosenbrock_exact( *ops, *expected, t, true );
    return difference( *ops, *state, *expected, *scratch );
}

// Deliberately violate P*B*(I-P)=0 while keeping the physical RHS compatible.
// The resulting discrepancy establishes why arbitrary W matrices are outside
// this projection contract; the engine does not run an O(N^2) matrix audit.
void incompatible_approximation()
{
    using S  = rosenbrock_projected_step<O>;
    using R  = rosenbrock::time_step<O, P, fixed_control<O>, L>;
    auto ops = std::make_shared<O>( 3 );
    auto p = std::make_shared<P>( ops ), r = std::make_shared<P>( ops, true );
    p->normal_feedback = r->normal_feedback = 3;
    auto               projection           = std::make_shared<projector<O>>( ops );
    typename S::params sp{ "ROS34PW2" };
    sp.autonomous = false;
    typename R::params rp{ "ROS34PW2" };
    rp.autonomous = false;
    S step(
        ops, p, std::make_shared<fixed_control<O>>( fixed_control<O>::params{ .1 } ), std::make_shared<L>( ops ), sp,
        projection
    );
    R ref(
        ops, r, std::make_shared<fixed_control<O>>( fixed_control<O>::params{ .1 } ), std::make_shared<L>( ops ), rp
    );
    buffer<O> state( *ops ), out( *ops ), expected( *ops ), scratch( *ops );
    exact( *ops, *state );
    step.apply( *state, *out );
    ref.apply( *state, *expected );
    check(
        step.get_status() == single_step_status::converged && ref.get_status() == single_step_status::converged,
        "Incompatible W counterexample did not solve its shifted systems"
    );
    check(
        difference( *ops, *out, *expected, *scratch ) > 1e-7, "Incompatible W approximation counterexample disappeared"
    );
}
int main()
try
{
    backend_type::init_device();
    for ( const char *method : { "ROS2", "ROS3", "ROS34PW2" } )
    {
        for ( bool nonlinear : { false, true } )
        {
            const double order = std::string( method ) == "ROS2" ? 2 : 3;
            check(
                trajectory( method, .04, nonlinear ) / trajectory( method, .02, nonlinear ) >
                    std::pow( 2., order - .4 ),
                "Projected primary/dense Rosenbrock order"
            );
            trajectory( method, -.02, nonlinear );
        }
        check( adaptive( method, 1e-7 ) < .3 * adaptive( method, 1e-4 ), "Projected Rosenbrock tolerance refinement" );
    }
    for ( int approximation = 1; approximation <= 4; ++approximation )
    {
        check(
            trajectory( "ROS34PW2", .04, true, approximation ) / trajectory( "ROS34PW2", .02, true, approximation ) >
                std::pow( 2., 2.6 ),
            "Projected W approximation lost order"
        );
    }
    incompatible_approximation();
    std::cout << checks << " Rosenbrock projection accuracy checks passed\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
