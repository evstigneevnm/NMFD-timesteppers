#include "implicit_imex_projection_problem.h"
#include "projection_transfer_guard.h"

using namespace projection_tests;

// 8192 nonlinear constrained blocks: partitions, Newton/GMRES, projection,
// error estimation, dense output and verification all remain backend-resident.
template <bool Dense, bool Adaptive>
void run( const char *method )
{
    using O      = counted_space<guarded_backend>;
    using P      = imex_projection_problem<O>;
    using I      = typename P::implicit_type;
    using C      = std::conditional_t<Adaptive, adaptive_control<O>, fixed_control<O>>;
    using S      = imex_projected_step<O, C, Dense>;
    auto ops     = std::make_shared<O>( 3 * 8192 );
    auto p       = std::make_shared<P>( ops );
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
    S         step( ops, p, control, implicit_newton<O, I>::make( ops ), typename S::params{ method }, project );
    buffer<O> state( *ops ), out( *ops ), expected( *ops ), sample( *ops ), scratch( *ops );
    imex_exact( *ops, *state );
    auto execute = [&]() {
        step.reset();
        step.set_time( 0 );
        step.set_target_time( .02 );
        step.apply( *state, *out );
        check( step.get_status() == single_step_status::converged, "Guarded projected IMEX failed" );
        check( constraint( *ops, *out, *scratch ) < 1e-13, "Large IMEX endpoint constraint" );
        imex_exact( *ops, *expected, step.get_dt() );
        difference( *ops, *out, *expected, *scratch );
        check( ops->norm_l2( *scratch ) < 1e-5, "Large IMEX accuracy" );
        if constexpr ( Dense )
        {
            auto       view  = step.get_continuous_integration();
            const auto calls = project->calls;
            for ( double theta : { 0., .2, .7, 1. } )
            {
                view.evaluate( theta, *sample );
                check( constraint( *ops, *sample, *scratch ) < 1e-13, "Large IMEX dense constraint" );
            }
            check( project->calls == calls, "Dense IMEX sampling invoked projector" );
        }
        step.finalize( adaptation_status::accepted, step.get_dt(), *out );
    };
    execute();
    const auto count = guarded_memory::allocation_count;
    for ( int j = 0; j < 3; ++j )
        execute();
    check( guarded_memory::allocation_count == count, "IMEX allocated vectors after warmup" );
    check(
        project->calls == p->bundle->implicit_part->shift_calls + p->bundle->implicit_part->rate_calls +
                              p->bundle->explicit_part->rate_calls,
        "Large IMEX stage escaped projection"
    );
    bool blocked = false;
    try
    {
        ops->get( *state, static_cast<double *>( nullptr ) );
    }
    catch ( const std::runtime_error & )
    {
        blocked = true;
    }
    check( blocked, "IMEX download guard disconnected" );
    blocked = false;
    try
    {
        ops->set( static_cast<const double *>( nullptr ), *state );
    }
    catch ( const std::runtime_error & )
    {
        blocked = true;
    }
    check( blocked, "IMEX upload guard disconnected" );
}
int main()
try
{
    backend_type::init_device();
    run<false, false>( "IMEX_ARS222" );
    run<false, false>( "IMEX_ARK324L2SA" );
    run<true, false>( "IMEX_ARK324L2SA" );
    run<false, true>( "IMEX_ARK324L2SA" );
    run<true, true>( "IMEX_ARK324L2SA" );
    check( guarded_memory::allocations.empty(), "IMEX projection workspace leaked" );
    std::cout << checks << " IMEX projection backend checks passed\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
