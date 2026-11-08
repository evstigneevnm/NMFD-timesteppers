#include "implicit_rosenbrock_index2_problem.h"

using namespace rosenbrock_tests;

// Cubic Hermite interpolation uses problem-supplied full endpoint rates.
// Differential components are third order. Moving raw pressure remains second
// order; reconstruct it using the differentiated constraint, not interpolation
// of an arbitrary algebraic variable. Both gauges are independently checked.
template <bool Moving>
std::array<double, 3> trajectory( double h, pressure_gauge gauge )
{
    using P     = index2_problem<Moving, operations_type, true>;
    using L     = index2_solver<P>;
    using S     = rosenbrock::time_step<operations_type, P, adaptation_type, L, true>;
    using Plain = rosenbrock::time_step<operations_type, P, adaptation_type, L>;
    auto ops    = std::make_shared<operations_type>( P::block_size );
    auto p = std::make_shared<P>( ops ), q = std::make_shared<P>( ops );
    p->gauge = q->gauge = gauge;
    // Coarse raw multiplier defects are O(h^2). The rate provider refuses
    // states outside its explicit consistency budget; it never projects input.
    p->rate_consistency_tolerance = 2e-4;
    auto solver                   = std::make_shared<L>( ops );
    S    step(
        ops, p, std::make_shared<adaptation_type>( adaptation_type::params{ std::abs( h ) } ), solver,
        index2_params<S>()
    );
    Plain plain(
        ops, q, std::make_shared<adaptation_type>( adaptation_type::params{ std::abs( h ) } ),
        std::make_shared<L>( ops ), index2_params<Plain>()
    );
    vector_wrap_type in( *ops ), out( *ops ), other( *ops ), sample( *ops ), projected( *ops ), scratch( *ops );
    for ( auto *v : { &in, &out, &other, &sample, &projected, &scratch } )
        v->start_use();
    const double start = .37, end = start + std::copysign( .4, h );
    p->initialize_state( start, *in );
    std::array<double, 3> error{};
    const auto            count = std::lround( .4 / std::abs( h ) );
    for ( long n = 0; n < count; ++n )
    {
        const double t = start + n * h;
        step.set_time( t );
        step.set_target_time( end );
        step.apply( *in, *out );
        plain.set_time( t );
        plain.set_target_time( end );
        plain.apply( *in, *other );
        check(
            step.get_status() == single_step_status::converged && plain.get_status() == single_step_status::converged,
            "Consistent-rate Hermite preparation failed"
        );
        ops->assign_lin_comb( 1., *out, -1., *other, *scratch );
        check( ops->norm( *scratch ) == 0, "Consistent dense preparation changed the numerical endpoint" );
        const auto rates = p->rates, solves = solver->calls;
        auto       view = step.get_continuous_integration();
        for ( double theta : { 0., .17, .43, .77, 1. } )
        {
            const double time = view.evaluate( theta, *sample );
            if ( theta == 0 || theta == 1 )
            {
                ops->assign_lin_comb( 1., *sample, -1., theta == 0 ? *in : *out, *scratch );
                check( ops->norm( *scratch ) == 0, "Index-2 dense output did not copy its endpoint exactly" );
            }
            p->reconstruct( *sample, *projected );
            check(
                p->constraint_norm( *projected, *scratch ) < 2e-13,
                "Dense reconstruction violated constraint/gauge/hidden equation"
            );
            for ( unsigned int i = 0; i < 4; ++i )
            {
                const double exact = P::exact( i, time, gauge );
                error[i / 2] = std::max( error[i / 2], std::abs( ops->get_value_at_point( i, *sample ) - exact ) );
                if ( i >= 2 )
                    error[2] = std::max( error[2], std::abs( ops->get_value_at_point( i, *projected ) - exact ) );
            }
            if constexpr ( Moving )
                check(
                    std::abs( ops->get_value_at_point( 4, *projected ) - time ) < 2e-14, "Dense clock lost sample time"
                );
        }
        check( p->rates == rates && solver->calls == solves, "Dense sampling performed a rate or shifted solve" );
        const double next = step.get_dt() == end - t ? end : t + step.get_dt();
        step.finalize( adaptation_status::accepted, next, *out );
        plain.finalize( adaptation_status::accepted, next, *other );
        ops->assign( *out, *in );
    }
    check(
        p->rates == 2 * count && solver->calls == 4 * count && solver->bindings == count,
        "Consistent Hermite extension used W slope solves or extra bindings"
    );
    check(
        step.dense_output_order() == 3 && plain.dense_output_order() == 0, "Consistent Hermite order diagnostic changed"
    );
    return error;
}

// A scalar section u0=exp(-event_time) is located in a pending moving-DAE
// step. Reconstruct before committing its shortened endpoint; the next step
// must freeze exactly that state. This is test-owned event logic, not a new
// timestepper callback or a dependency on periodic_orbit.
void shortened_events()
{
    using P = index2_problem<true, operations_type, true>;
    using S = rosenbrock::time_step<operations_type, P, adaptation_type, index2_solver<P>, true>;
    for ( double direction : { 1., -1. } )
    {
        auto ops                      = std::make_shared<operations_type>( 5 );
        auto p                        = std::make_shared<P>( ops );
        p->rate_consistency_tolerance = 2e-4;
        auto solver                   = std::make_shared<index2_solver<P>>( ops );
        S    step(
            ops, p, std::make_shared<adaptation_type>( adaptation_type::params{ .02 } ), solver, index2_params<S>()
        );
        vector_wrap_type in( *ops ), out( *ops ), sample( *ops ), scratch( *ops ), snapshot( *ops );
        for ( auto *v : { &in, &out, &sample, &scratch, &snapshot } )
            v->start_use();
        const double start = direction > 0 ? .37 : .57, event = start + direction * .01,
                     target = start + direction * .1;
        p->initialize_state( start, *in );
        step.set_time( start );
        step.set_target_time( target );
        step.apply( *in, *out );
        check( step.get_status() == single_step_status::converged, "Event step failed" );
        const auto   view  = step.get_continuous_integration();
        const double level = std::exp( -event ), left_value = ops->get_value_at_point( 0, *in ) - level;
        double       lo = 0, hi = 1;
        for ( unsigned int iteration = 0; iteration < 48; ++iteration )
        {
            const double theta = ( lo + hi ) / 2;
            view.evaluate( theta, *sample );
            if ( ( ops->get_value_at_point( 0, *sample ) - level ) * left_value > 0 )
                lo = theta;
            else
                hi = theta;
        }
        const double time = view.evaluate( ( lo + hi ) / 2, *out );
        p->reconstruct( *out, *out );
        check(
            std::abs( time - event ) < 2e-7 && p->constraint_norm( *out, *scratch ) < 2e-13,
            "Shortened constrained event has inaccurate time or state"
        );
        check( p->evaluate_rate( *out, *sample ), "Reconstructed event did not admit a consistent full rate" );
        ops->assign( *out, *snapshot );
        step.finalize( adaptation_status::accepted_modified, time, *out );
        bool expired = false;
        try
        {
            view.evaluate( .5, *sample );
        }
        catch ( const std::logic_error & )
        {
            expired = true;
        }
        check( expired, "Event shortening retained an old dense view" );
        step.set_time( time );
        step.apply( *out, *out );
        check( step.get_status() == single_step_status::converged, "Event restart failed" );
        ops->assign_lin_comb( 1., *snapshot, -1., *p->jacobi->point, *scratch );
        check( ops->norm( *scratch ) == 0, "Event restart froze stale pre-projection data" );
        step.finalize( adaptation_status::accepted, time + step.get_dt(), *out );
    }
}

int main()
try
{
    backend_type::init_device();
    for ( auto gauge : { pressure_gauge::mean_zero, pressure_gauge::pinned } )
        for ( double sign : { 1., -1. } )
        {
            auto check_orders = [&]( auto moving ) {
                constexpr bool Moving = decltype( moving )::value;
                const auto     a = trajectory<Moving>( sign * .04, gauge ), b = trajectory<Moving>( sign * .02, gauge ),
                           c = trajectory<Moving>( sign * .01, gauge );
                std::cout << "index2 dense moving=" << Moving << " sign=" << sign << " gauge=" << int( gauge )
                          << " velocity=" << a[0] << ',' << b[0] << ',' << c[0] << " raw pressure=" << a[1] << ','
                          << b[1] << ',' << c[1] << " reconstructed pressure=" << a[2] << ',' << b[2] << ',' << c[2]
                          << '\n';
                for ( unsigned int i : { 0u, 2u } )
                    check( a[i] / b[i] > 6. && b[i] / c[i] > 6., "Differential/reconstructed dense order lost" );
                const double minimum = Moving ? 3.5 : 6.;
                check( a[1] / b[1] > minimum && b[1] / c[1] > minimum, "Raw dense pressure order lost" );
            };
            check_orders( std::false_type{} );
            check_orders( std::true_type{} );
        }
    shortened_events();
    std::cout << "PASS: Rosenbrock index-2 dense reconstruction and shortened events (" << checks << " checks)\n";
}
catch ( const std::exception &e )
{
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
