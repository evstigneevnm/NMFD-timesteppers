#include "implicit_imex_projection_problem.h"

using namespace projection_tests;

template <bool Dense, bool Adaptive>
void compare( const char *method, double start, double dt, double implicit_fraction = .6 )
{
    using O                                = space_type;
    using P                                = imex_projection_problem<O>;
    using I                                = typename P::implicit_type;
    using C                                = std::conditional_t<Adaptive, adaptive_control<O>, fixed_control<O>>;
    using S                                = imex_projected_step<O, C, Dense>;
    using R                                = imex_reference_step<O, C, Dense>;
    auto ops                               = std::make_shared<O>( 3 );
    auto p                                 = std::make_shared<P>( ops, false, implicit_fraction );
    auto reduced                           = std::make_shared<P>( ops, true, implicit_fraction );
    auto project                           = std::make_shared<projector<O>>( ops );
    p->bundle->implicit_part->record_times = p->bundle->explicit_part->record_times = true;
    auto control                                                                    = [&]() {
        if constexpr ( Adaptive )
        {
            typename C::params cp;
            cp.initial_step = cp.maximum_step = std::abs( dt );
            cp.relative_tolerance             = 1e-7;
            cp.absolute_tolerance             = 1e-9;
            return std::make_shared<C>( ops, cp );
        }
        else
            return std::make_shared<C>( typename C::params{ std::abs( dt ) } );
    };
    S         step( ops, p, control(), implicit_newton<O, I>::make( ops ), typename S::params{ method }, project );
    R         reference( ops, reduced, control(), implicit_newton<O, I>::make( ops ), typename R::params{ method } );
    buffer<O> state( *ops ), out( *ops ), expected( *ops ), sample( *ops ), ref_sample( *ops ), scratch( *ops );
    imex_exact( *ops, *state, start );
    step.set_time( start );
    reference.set_time( start );
    step.set_target_time( start + dt );
    reference.set_target_time( start + dt );
    step.apply( *state, *out );
    reference.apply( *state, *expected );
    check( step.get_status() == single_step_status::converged, "Projected IMEX failed" );
    check( reference.get_status() == single_step_status::converged, "Reduced IMEX failed" );
    check( std::abs( step.get_dt() - reference.get_dt() ) < 2e-10, "Projected combined estimator differs" );
    check(
        difference( *ops, *out, *expected, *scratch ) < 2e-11,
        "Projected IMEX differs from independent reduced equation"
    );
    check( constraint( *ops, *out, *scratch ) < 1e-13, "IMEX endpoint constraint" );
    const auto &i = *p->bundle->implicit_part;
    const auto &e = *p->bundle->explicit_part;
    check( project->calls == i.shift_calls + i.rate_calls + e.rate_calls, "IMEX stage escaped projection" );
    check( p->bindings == 1 && reduced->bindings == 1, "IMEX rebound partition bundle during stages" );
    if constexpr ( !Adaptive )
    {
        std::size_t solve = 0;
        for ( std::size_t stage = 0; stage < step.table().implicit_table.size(); ++stage )
            if ( step.table().implicit_table.a( stage, stage ) != 0 )
                check(
                    std::abs( i.stage_times[solve++] - ( start + dt * step.table().implicit_table.c( stage ) ) ) <
                        1e-14,
                    "Projected implicit stage time"
                );
        const auto check_rate_times = [&]( const auto &table, const auto &actual, bool zero_diagonal_only ) {
            std::vector<double> expected_times;
            for ( std::size_t stage = 0; stage < table.size(); ++stage )
            {
                if ( zero_diagonal_only && table.a( stage, stage ) != 0 )
                    continue;
                bool consumed = table.b( stage ) != 0;
                for ( std::size_t row = stage + 1; row < table.size(); ++row )
                    consumed = consumed || table.a( row, stage ) != 0;
                if ( consumed )
                    expected_times.push_back( start + dt * table.c( stage ) );
            }
            if constexpr ( Dense )
                expected_times.push_back( start + dt );
            check( actual.size() == expected_times.size(), "Unexpected projected partition rate calls" );
            for ( std::size_t j = 0; j < actual.size(); ++j )
                check( std::abs( actual[j] - expected_times[j] ) < 1e-14, "Wrong projected partition rate time" );
        };
        check_rate_times( step.table().explicit_table, e.rate_times, false );
        check_rate_times( step.table().implicit_table, i.rate_times, true );
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
            check( difference( *ops, *sample, *ref_sample, *scratch ) < 3e-11, "Projected IMEX dense mismatch" );
            check( constraint( *ops, *sample, *scratch ) < 1e-13, "Projected IMEX dense constraint" );
        }
        check( project->calls == calls, "IMEX sampling invoked projector" );
    }
    step.finalize( adaptation_status::accepted, start + step.get_dt(), *out );
    reference.finalize( adaptation_status::accepted, start + reference.get_dt(), *expected );
}

double endpoint_error( const char *method, double dt, double weight )
{
    using O      = space_type;
    using P      = imex_projection_problem<O>;
    using I      = typename P::implicit_type;
    using S      = imex_projected_step<O>;
    auto ops     = std::make_shared<O>( 3 );
    auto p       = std::make_shared<P>( ops, false, weight );
    auto control = std::make_shared<fixed_control<O>>( fixed_control<O>::params{ dt } );
    S    step(
        ops, p, control, implicit_newton<O, I>::make( ops ), S::params{ method }, std::make_shared<projector<O>>( ops )
    );
    buffer<O> state( *ops ), expected( *ops ), scratch( *ops );
    imex_exact( *ops, *state );
    const int count = int( std::round( .4 / dt ) );
    for ( int j = 0; j < count; ++j )
    {
        const double t = j * dt;
        step.set_time( t );
        step.set_target_time( .4 );
        step.apply( *state, *state );
        check( step.get_status() == single_step_status::converged, "Projected IMEX refinement failed" );
        step.finalize( adaptation_status::accepted, std::min( .4, t + step.get_dt() ), *state );
    }
    imex_exact( *ops, *expected, .4 );
    return difference( *ops, *state, *expected, *scratch );
}

int main()
try
{
    backend_type::init_device();
    for ( const char *method : { "IMEX_EULER", "IMEX_HEUN_TR2", "IMEX_ARS233", "IMEX_ARS222", "IMEX_ARK324L2SA" } )
    {
        const auto table = runge_kutta::make_imex_butcher_table( method );
        for ( double weight : { 0., .6, 1. } )
        {
            compare<false, false>( method, .3, .04, weight );
            compare<false, false>( method, .3, -.04, weight );
            const double coarse = endpoint_error( method, .04, weight ), fine = endpoint_error( method, .02, weight );
            check(
                coarse / fine > std::pow( 2., table.order - .4 ),
                "Projected IMEX lost order, including pure partition limits"
            );
        }
    }
    compare<true, false>( "IMEX_ARK324L2SA", .3, .04 );
    compare<true, false>( "IMEX_ARK324L2SA", .3, -.04 );
    compare<false, true>( "IMEX_ARK324L2SA", .3, .08 );
    compare<true, true>( "IMEX_ARK324L2SA", .3, .08 );
    std::cout << checks << " IMEX projection accuracy checks passed\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
