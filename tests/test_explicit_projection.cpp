#include "explicit_projection_problem.h"

using namespace projection_tests;

// Compare every explicit primary/embedded/dense formula against an independently
// assembled reduced RHS. No geometric state normalization or singular-mass
// projection is inferred from this fixed linear, homogeneous constraint.
template <bool Dense>
void check_method( const char *method )
{
    using S                = projected_step<space_type, fixed_control<space_type>, Dense>;
    using R                = reference_step<space_type, fixed_control<space_type>, Dense>;
    auto ops               = std::make_shared<space_type>( 6 );
    auto p                 = std::make_shared<problem<space_type>>( ops );
    auto reduced           = std::make_shared<problem<space_type>>( ops, true );
    auto projection        = std::make_shared<projector<space_type>>( ops );
    auto control           = std::make_shared<fixed_control<space_type>>( fixed_control<space_type>::params{ .125 } );
    auto reference_control = std::make_shared<fixed_control<space_type>>( fixed_control<space_type>::params{ .125 } );
    S    step( ops, p, control, typename S::params{ method }, projection );
    R    reference( ops, reduced, reference_control, typename R::params{ method } );
    buffer<space_type> state( *ops ), expected( *ops ), sample( *ops ), reference_sample( *ops ), scratch( *ops );
    exact( *ops, *state );
    exact( *ops, *expected );
    p->record_times = reduced->record_times = true;
    for ( double start : { 0., .125 } )
    {
        step.set_time( start );
        reference.set_time( start );
        step.apply( *state, *state );
        reference.apply( *expected, *expected );
        check( step.get_status() == single_step_status::converged, "Projected RK failed" );
        check( reference.get_status() == single_step_status::converged, "Reduced reference RK failed" );
        check( difference( *ops, *state, *expected, *scratch ) < 2e-14, "Projected stages differ from reduced RHS" );
        check( constraint( *ops, *state, *scratch ) < 2e-15, "Projected endpoint left the constraint plane" );
        if ( step.table().is_embedded() )
            check(
                difference( *ops, step.error_estimate(), reference.error_estimate(), *scratch ) < 2e-14,
                "Embedded defect uses unprojected stages"
            );
        if constexpr ( Dense )
        {
            auto       view           = step.get_continuous_integration();
            auto       reference_view = reference.get_continuous_integration();
            const auto calls          = projection->calls;
            for ( double theta : { 0., .2, .75, 1. } )
            {
                view.evaluate( theta, *sample );
                reference_view.evaluate( theta, *reference_sample );
                check(
                    difference( *ops, *sample, *reference_sample, *scratch ) < 2e-14,
                    "Dense output uses unprojected stages"
                );
                check( constraint( *ops, *sample, *scratch ) < 2e-15, "Dense sample left constraint plane" );
            }
            check( projection->calls == calls, "Dense sampling invoked the projector" );
        }
        step.finalize( adaptation_status::accepted, start + .125, *state );
        reference.finalize( adaptation_status::accepted, start + .125, *expected );
    }
    check( p->calls == projection->calls && p->calls == reduced->calls, "Not every evaluated rate was projected" );
    check( p->times == reduced->times, "Projection changed stage times" );
    const auto stages = Dense && step.table().has_dense_output() ? step.table().dense_outout_stage_count()
                                                                 : step.table().size() + ( Dense ? 1 : 0 );
    check( projection->calls == 2 * stages, "Unexpected number of stage projections" );
}

double endpoint_error( const char *method, double h )
{
    using S                       = projected_step<space_type>;
    auto               ops        = std::make_shared<space_type>( 3 );
    auto               p          = std::make_shared<problem<space_type>>( ops );
    auto               projection = std::make_shared<projector<space_type>>( ops );
    auto               control = std::make_shared<fixed_control<space_type>>( fixed_control<space_type>::params{ h } );
    auto               step    = std::make_shared<S>( ops, p, control, S::params{ method }, projection );
    buffer<space_type> state( *ops ), expected( *ops ), scratch( *ops );
    exact( *ops, *state );
    exact( *ops, *expected, 1 );
    integration::time_integrator<space_type, S> run( ops, step );
    run.apply( *state, *state );
    check( run.get_status() == integration_status::completed, "Projected trajectory did not finish" );
    return difference( *ops, *state, *expected, *scratch );
}

// Compare adaptive trajectories and both DOP853 defect components at every
// accepted candidate, so projection cannot silently alter controller decisions.
template <bool Dense>
void check_adaptive( const char *method, double direction )
{
    using C                       = adaptive_control<space_type>;
    using S                       = projected_step<space_type, C, Dense>;
    using R                       = reference_step<space_type, C, Dense>;
    auto               ops        = std::make_shared<space_type>( 3 );
    auto               p          = std::make_shared<problem<space_type>>( ops );
    auto               reduced    = std::make_shared<problem<space_type>>( ops, true );
    auto               projection = std::make_shared<projector<space_type>>( ops );
    typename C::params cp;
    cp.initial_step = cp.maximum_step    = .25;
    cp.relative_tolerance                = 1e-8;
    cp.absolute_tolerance                = 1e-10;
    auto               control           = std::make_shared<C>( ops, cp );
    auto               reference_control = std::make_shared<C>( ops, cp );
    S                  step( ops, p, control, typename S::params{ method, 32, true }, projection );
    R                  reference( ops, reduced, reference_control, typename R::params{ method, 32, true } );
    buffer<space_type> state( *ops ), expected( *ops ), scratch( *ops ), sample( *ops ), reference_sample( *ops );
    exact( *ops, *state, .37 );
    exact( *ops, *expected, .37 );
    double       time  = .37;
    const double end   = time + direction * .5;
    std::size_t  steps = 0;
    while ( time != end && ++steps < 1000 )
    {
        step.set_time( time );
        reference.set_time( time );
        step.set_target_time( end );
        reference.set_target_time( end );
        step.apply( *state, *state );
        reference.apply( *expected, *expected );
        check(
            step.get_status() == single_step_status::converged &&
                reference.get_status() == single_step_status::converged,
            "Adaptive projected RK failed"
        );
        check( std::abs( step.get_dt() - reference.get_dt() ) < 1e-10, "Projected adaptive schedule differs" );
        check( difference( *ops, *state, *expected, *scratch ) < 1e-10, "Adaptive projected endpoint differs" );
        for ( unsigned int component = 0; component < ( std::string( method ) == "DOP853" ? 2u : 1u ); ++component )
            check(
                difference( *ops, step.error_estimate( component ), reference.error_estimate( component ), *scratch ) <
                    2e-14,
                "Adaptive projected defect differs"
            );
        if constexpr ( Dense )
        {
            step.get_continuous_integration().evaluate( .4, *sample );
            reference.get_continuous_integration().evaluate( .4, *reference_sample );
            check( difference( *ops, *sample, *reference_sample, *scratch ) < 1e-10, "Adaptive dense result differs" );
        }
        time = step.get_dt() == end - time ? end : time + step.get_dt();
        step.finalize( adaptation_status::accepted, time, *state );
        reference.finalize( adaptation_status::accepted, time, *expected );
    }
    check( time == end, "Adaptive projected integration exhausted budget" );
    check( projection->calls == p->calls, "Adaptive rate escaped projection" );
    exact( *ops, *expected, end );
    check( difference( *ops, *state, *expected, *scratch ) < 2e-7, "Adaptive projected physical accuracy" );
}

int main()
try
{
    backend_type::init_device();
    for ( const char *method : { "EE", "HE", "RK23", "RK33SSP", "RK43SSP", "RK4", "RK64SSP", "RK45", "DOP853" } )
    {
        check_method<false>( method );
        check_method<true>( method );
    }
    for ( const char *method : { "EE", "RK4", "RK45", "DOP853" } )
    {
        const auto coarse = endpoint_error( method, .5 ), fine = endpoint_error( method, .25 );
        const auto order = runge_kutta::make_butcher_table( method ).order();
        check( coarse > std::pow( 2., order - .6 ) * fine, "Stage projection reduced the RK order" );
    }
    for ( const char *method : { "RK45", "DOP853" } )
        for ( double direction : { -1., 1. } )
        {
            check_adaptive<false>( method, direction );
            check_adaptive<true>( method, direction );
        }
    std::cout << checks << " explicit projection accuracy checks passed\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
