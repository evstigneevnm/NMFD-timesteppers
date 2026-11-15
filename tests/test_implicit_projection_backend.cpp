#include "implicit_projection_problem.h"
#include "projection_transfer_guard.h"

using namespace projection_tests;

// 24576 unknowns with coupled block dynamics. Newton/GMRES, projection, error
// estimation and dense sampling must use fixed backend storage, not host arrays.
template <bool Dense, bool Adaptive>
void run( const char *method )
{
    using O      = counted_space<guarded_backend>;
    using C      = std::conditional_t<Adaptive, adaptive_control<O>, fixed_control<O>>;
    using S      = implicit_projected_step<O, C, Dense>;
    auto ops     = std::make_shared<O>( 3 * 8192 );
    auto p       = std::make_shared<implicit_problem<O>>( ops );
    auto project = std::make_shared<projector<O>>( ops );
    auto control = [&]() {
        if constexpr ( Adaptive )
        {
            typename C::params cp;
            cp.initial_step = cp.maximum_step = .02;
            cp.relative_tolerance             = 1e-6;
            cp.absolute_tolerance             = 1e-8;
            return std::make_shared<C>( ops, cp );
        }
        else
            return std::make_shared<C>( typename C::params{ .02 } );
    }();
    S         step( ops, p, control, implicit_newton<O>::make( ops ), typename S::params{ method }, project );
    buffer<O> state( *ops ), out( *ops ), expected( *ops ), scratch( *ops ), sample( *ops );
    exact( *ops, *state );
    auto execute = [&]() {
        step.reset();
        step.set_time( 0 );
        step.set_target_time( .02 );
        step.apply( *state, *out );
        check( step.get_status() == single_step_status::converged, "Guarded projected DIRK failed" );
        check( constraint( *ops, *out, *scratch ) < 1e-13, "Large DIRK endpoint constraint" );
        exact( *ops, *expected, step.get_dt() );
        difference( *ops, *out, *expected, *scratch );
        check( ops->norm_l2( *scratch ) < 1e-5, "Large DIRK accuracy" );
        if constexpr ( Dense )
        {
            auto view  = step.get_continuous_integration();
            auto calls = project->calls;
            for ( double theta : { 0., .2, .7, 1. } )
            {
                view.evaluate( theta, *sample );
                check( constraint( *ops, *sample, *scratch ) < 1e-13, "Large DIRK dense constraint" );
            }
            check( project->calls == calls, "DIRK dense sampling performed a solve" );
        }
        step.finalize( adaptation_status::accepted, step.get_dt(), *out );
    };
    execute();
    const auto allocations = guarded_memory::allocation_count;
    for ( int i = 0; i < 3; ++i )
        execute();
    check( guarded_memory::allocation_count == allocations, "DIRK allocated vectors after warmup" );
    bool blocked = false;
    try
    {
        ops->get( *state, static_cast<double *>( nullptr ) );
    }
    catch ( const std::runtime_error & )
    {
        blocked = true;
    }
    check( blocked, "DIRK transfer guard disconnected" );
}
int main()
try
{
    backend_type::init_device();
    for ( const char *method : { "ESDIRK2(1)3", "SDIRK3(1)3", "ARK324L2SA" } )
    {
        run<false, false>( method );
        run<true, false>( method );
        run<false, true>( method );
        run<true, true>( method );
        check( guarded_memory::allocations.empty(), "DIRK projection workspace leaked" );
    }
    std::cout << checks << " implicit projection backend checks passed\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
