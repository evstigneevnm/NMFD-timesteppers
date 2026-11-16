#include "implicit_bdf_projection_test.h"
#include "projection_transfer_guard.h"
using namespace projection_tests;

// 24576 components; grow automatic BDF through order five without reallocating
// history. All solves, projection, error estimates, interpolation and checks
// remain on the selected backend. Only scalar reductions leave it.
template <bool Adaptive, bool Dense>
void run( const char *method )
{
    using O                       = counted_space<guarded_backend>;
    using C                       = std::conditional_t<Adaptive, bdf_accepting_control<O>, fixed_control<O>>;
    using S                       = bdf_projected_step<O, C, Dense>;
    auto               ops        = std::make_shared<O>( 3 * 8192 );
    auto               p          = std::make_shared<implicit_problem<O>>( ops );
    auto               projection = std::make_shared<projector<O>>( ops );
    auto               control    = std::make_shared<C>( typename C::params{ .01 } );
    typename S::params params{ method };
    params.adaptive = Adaptive;
    S         step( ops, p, control, implicit_newton<O>::make( ops ), params, projection );
    buffer<O> state( *ops ), out( *ops ), reference( *ops ), scratch( *ops ), sample( *ops );
    double    t = .3;
    exact( *ops, *state, t );
    unsigned int highest     = 0;
    std::size_t  allocations = 0;
    for ( int j = 0; j < 60; ++j )
    {
        step.set_time( t );
        step.set_target_time( 2. );
        step.apply( *state, *out );
        check( step.get_status() == single_step_status::converged, "Guarded projected BDF failed" );
        highest = std::max( highest, step.get_order() );
        check( constraint( *ops, *out, *scratch ) < 2e-12, "Large BDF constraint drift" );
        if constexpr ( Dense )
        {
            auto       view  = step.get_continuous_integration();
            const auto calls = projection->calls;
            for ( double theta : { 0., .25, .7, 1. } )
            {
                view.evaluate( theta, *sample );
                check( constraint( *ops, *sample, *scratch ) < 3e-12, "Large BDF dense constraint" );
            }
            check( projection->calls == calls, "Dense BDF sampling invoked projector" );
        }
        t += step.get_dt();
        step.finalize( adaptation_status::accepted, t, *out );
        ops->assign( *out, *state );
        if ( j == 0 )
            allocations = guarded_memory::allocation_count;
        check(
            allocations == guarded_memory::allocation_count,
            "BDF projection allocated vectors during history/order growth"
        );
    }
    check(
        highest == ( std::string( method ) == "BDF"    ? 5u
                     : std::string( method ) == "BDF2" ? 2u
                                                       : 1u ),
        "Guarded BDF failed order progression"
    );
    check( projection->calls == 60 + ( Adaptive ? 2 : 0 ), "BDF startup/normal projection count" );
    exact( *ops, *reference, t );
    difference( *ops, *state, *reference, *scratch );
    check( ops->norm_l2( *scratch ) < .003, "Guarded BDF physical error" );
    bool blocked = false;
    try
    {
        ops->get( *state, static_cast<double *>( nullptr ) );
    }
    catch ( const std::runtime_error & )
    {
        blocked = true;
    }
    check( blocked, "BDF projection download guard disconnected" );
    blocked = false;
    try
    {
        ops->set( static_cast<const double *>( nullptr ), *state );
    }
    catch ( const std::runtime_error & )
    {
        blocked = true;
    }
    check( blocked, "BDF projection upload guard disconnected" );
}
int main()
try
{
    backend_type::init_device();
    run<false, false>( "BDF1" );
    run<false, true>( "BDF2" );
    run<true, false>( "BDF" );
    run<true, true>( "BDF" );
    check( guarded_memory::allocations.empty(), "BDF projection workspace leaked" );
    std::cout << checks << " BDF projection backend checks passed\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
