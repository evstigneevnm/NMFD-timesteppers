#include "implicit_projection_problem.h"

using namespace projection_tests;

// Compare every DIRK/ESDIRK stage with an independent reduced ODE, including
// zero-diagonal stages, embedded error control and dense-only rate evaluations.
template <bool Dense, bool Adaptive>
void compare( const char *method, double start, double dt )
{
    using O         = space_type;
    using C         = std::conditional_t<Adaptive, adaptive_control<O>, fixed_control<O>>;
    using S         = implicit_projected_step<O, C, Dense>;
    using R         = implicit_reference_step<O, C, Dense>;
    auto ops        = std::make_shared<O>( 3 );
    auto p          = std::make_shared<implicit_problem<O>>( ops );
    auto r          = std::make_shared<implicit_problem<O>>( ops, true );
    p->record_times = true;
    auto project    = std::make_shared<projector<O>>( ops );
    auto control    = [&]() {
        if constexpr ( Adaptive )
        {
            typename C::params cp;
            cp.initial_step       = std::abs( dt );
            cp.maximum_step       = std::abs( dt );
            cp.relative_tolerance = 1e-6;
            cp.absolute_tolerance = 1e-8;
            return std::make_shared<C>( ops, cp );
        }
        else
            return std::make_shared<C>( typename C::params{ std::abs( dt ) } );
    };
    S         step( ops, p, control(), implicit_newton<O>::make( ops ), typename S::params{ method }, project );
    R         reference( ops, r, control(), implicit_newton<O>::make( ops ), typename R::params{ method } );
    buffer<O> state( *ops ), out( *ops ), expected( *ops ), sample( *ops ), ref_sample( *ops ), scratch( *ops );
    exact( *ops, *state, start );
    step.set_time( start );
    reference.set_time( start );
    step.set_target_time( start + dt );
    reference.set_target_time( start + dt );
    step.apply( *state, *out );
    reference.apply( *state, *expected );
    check( step.get_status() == single_step_status::converged, "Projected DIRK failed" );
    check( reference.get_status() == single_step_status::converged, "Reduced DIRK failed" );
    check( std::abs( step.get_dt() - reference.get_dt() ) < 2e-11, "Projection changed combined error control" );
    check( difference( *ops, *out, *expected, *scratch ) < 2e-11, "Post-stage projection differs from reduced DIRK" );
    check( constraint( *ops, *out, *scratch ) < 1e-13, "Projected DIRK endpoint constraint" );
    std::size_t implicit_stages = 0;
    for ( std::size_t i = 0; i < step.table().size(); ++i )
        if ( step.table().a( i, i ) != 0 )
            ++implicit_stages;
    check( project->calls == p->stage_times.size() + p->rate_calls, "A DIRK stage escaped projection" );
    check( p->stage_times.size() >= implicit_stages, "Missing implicit stage solve" );
    if constexpr ( !Adaptive )
    {
        std::size_t j = 0;
        for ( std::size_t i = 0; i < step.table().size(); ++i )
            if ( step.table().a( i, i ) != 0 )
                check(
                    std::abs( p->stage_times[j++] - ( start + dt * step.table().c( i ) ) ) < 1e-14,
                    "Wrong projected stage time"
                );
    }
    if constexpr ( Dense )
    {
        auto       view  = step.get_continuous_integration();
        auto       ref   = reference.get_continuous_integration();
        const auto calls = project->calls;
        for ( double theta : { 0., .2, .6, 1. } )
        {
            view.evaluate( theta, *sample );
            ref.evaluate( theta, *ref_sample );
            check( difference( *ops, *sample, *ref_sample, *scratch ) < 3e-11, "Projected dense polynomial mismatch" );
            check( constraint( *ops, *sample, *scratch ) < 1e-13, "Projected dense constraint" );
        }
        check( project->calls == calls, "Sampling invoked the pressure projector" );
    }
    step.finalize( adaptation_status::accepted, start + step.get_dt(), *out );
    reference.finalize( adaptation_status::accepted, start + reference.get_dt(), *expected );
}

double endpoint_error( const char *method, double dt, bool incompatible = false )
{
    using O      = space_type;
    using S      = implicit_projected_step<O>;
    auto ops     = std::make_shared<O>( 3 );
    auto p       = std::make_shared<implicit_problem<O>>( ops );
    p->coupling  = incompatible ? 1 : 0;
    auto control = std::make_shared<fixed_control<O>>( fixed_control<O>::params{ dt } );
    S    step(
        ops, p, control, implicit_newton<O>::make( ops ), S::params{ method }, std::make_shared<projector<O>>( ops )
    );
    buffer<O> state( *ops ), expected( *ops ), scratch( *ops );
    exact( *ops, *state );
    const int count = int( std::round( .4 / dt ) );
    for ( int i = 0; i < count; ++i )
    {
        step.set_time( i * dt );
        step.set_target_time( .4 );
        step.apply( *state, *state );
        check( step.get_status() == single_step_status::converged, "DIRK order integration failed" );
        step.finalize( adaptation_status::accepted, std::min( .4, i * dt + step.get_dt() ), *state );
    }
    exact( *ops, *expected, .4 );
    return difference( *ops, *state, *expected, *scratch );
}

int main()
try
{
    backend_type::init_device();
    for ( const char *method : { "IE", "IM", "CN", "SDIRK2(1)2", "ESDIRK2(1)3", "SDIRK3(1)3", "ARK324L2SA" } )
    {
        compare<false, false>( method, .3, .04 );
        compare<true, false>( method, .3, -.04 );
        const auto table = runge_kutta::make_butcher_table( method );
        if ( table.is_embedded() )
        {
            compare<false, true>( method, .3, .04 );
            compare<true, true>( method, .3, .04 );
        }
        const double coarse = endpoint_error( method, .04 ), fine = endpoint_error( method, .02 );
        check( coarse / fine > std::pow( 2., table.order() - .35 ), "Projected DIRK lost primary order" );
    }
    // A successful projection is NOT proof of consistency for arbitrary f.
    // Normal-to-tangential coupling produces an O(h) defect even with CN.
    check(
        endpoint_error( "CN", .04, true ) > 20 * endpoint_error( "CN", .04 ),
        "Noncommuting counterexample was not exposed"
    );
    std::cout << checks << " implicit projection accuracy checks passed\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
