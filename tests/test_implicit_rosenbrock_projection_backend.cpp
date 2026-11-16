#include "implicit_rosenbrock_projection_problem.h"
#include "projection_transfer_guard.h"
using namespace projection_tests;

// 8192 independent constrained 3-component systems. Guard all rate/Jacobian,
// GMRES, increment projection, estimation and interpolation work against bulk
// transfers, and require fixed backend workspace after the first step.
template <bool Dense>
void run( const char *method, bool adaptive )
{
    using O                       = counted_space<guarded_backend>;
    using C                       = adaptive_control<O>;
    using S                       = rosenbrock_projected_step<O, C, Dense>;
    auto ops                      = std::make_shared<O>( 3 * 8192 );
    auto p                        = std::make_shared<rosenbrock_projection_problem<O>>( ops );
    p->noncommuting               = std::string( method ) == "ROS34PW2";
    auto               linear     = std::make_shared<projection_linear_solver<O>>( ops );
    auto               projection = std::make_shared<projector<O>>( ops );
    typename C::params cp;
    cp.initial_step       = .01;
    cp.maximum_step       = adaptive ? .04 : .01;
    cp.relative_tolerance = adaptive ? 1e-6 : 1.;
    cp.absolute_tolerance = adaptive ? 1e-9 : 1.;
    typename S::params sp{ method };
    sp.autonomous = false;
    S         step( ops, p, std::make_shared<C>( ops, cp ), linear, sp, projection );
    buffer<O> state( *ops ), out( *ops ), expected( *ops ), sample( *ops ), scratch( *ops );
    double    t = .3;
    exact( *ops, *state, t );
    std::size_t allocations = 0, steps = 0;
    while ( t != .7 && steps < 10000 )
    {
        step.set_time( t );
        step.set_target_time( .7 );
        step.apply( *state, *out );
        check( step.get_status() == single_step_status::converged, "Guarded Rosenbrock projection failed" );
        check( constraint( *ops, *out, *scratch ) < 2e-12, "Large Rosenbrock constraint drift" );
        if constexpr ( Dense )
        {
            auto       view  = step.get_continuous_integration();
            const auto calls = projection->calls;
            for ( double theta : { 0., .3, .7, 1. } )
            {
                view.evaluate( theta, *sample );
                check( constraint( *ops, *sample, *scratch ) < 3e-12, "Large Rosenbrock dense constraint" );
            }
            check( calls == projection->calls, "Rosenbrock sampling invoked projection" );
        }
        t = step.get_dt() == .7 - t ? .7 : t + step.get_dt();
        step.finalize( adaptation_status::accepted, t, *out );
        ops->assign( *out, *state );
        if ( steps == 0 )
            allocations = guarded_memory::allocation_count;
        check(
            allocations == guarded_memory::allocation_count, "Rosenbrock projection allocated vectors after warmup"
        );
        ++steps;
    }
    check( t == .7, "Guarded Rosenbrock budget exhausted" );
    check( projection->calls == linear->calls, "Guarded stage/dense solve escaped projection" );
    exact( *ops, *expected, t );
    difference( *ops, *state, *expected, *scratch );
    const double error = ops->norm_l2( *scratch );
    std::cout << method << " dense=" << Dense << " adaptive=" << adaptive << " RMS error=" << error << '\n';
    check( error < ( adaptive ? 1e-5 : 1e-4 ), "Guarded Rosenbrock physical accuracy" );
    bool blocked = false;
    try
    {
        ops->get( *state, static_cast<double *>( nullptr ) );
    }
    catch ( const std::runtime_error & )
    {
        blocked = true;
    }
    check( blocked, "Rosenbrock bulk download guard disconnected" );
    blocked = false;
    try
    {
        ops->set( static_cast<const double *>( nullptr ), *state );
    }
    catch ( const std::runtime_error & )
    {
        blocked = true;
    }
    check( blocked, "Rosenbrock bulk upload guard disconnected" );
}
int main()
try
{
    backend_type::init_device();
    for ( const char *method : { "ROS2", "ROS3", "ROS34PW2" } )
    {
        run<false>( method, false );
        run<true>( method, true );
    }
    check( guarded_memory::allocations.empty(), "Rosenbrock projection leaked backend storage" );
    std::cout << checks << " Rosenbrock projection backend checks passed; bulk host bytes=0\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
