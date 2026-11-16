#include "implicit_bdf_projection_test.h"

using namespace projection_tests;

// On x+y=0: x'=-x+z, z'=-2z. Raw normal forcing is discarded.
// Compare fixed-grid trajectories to an independently reduced problem. Adaptive
// controllers may choose different meshes after roundoff-sized error changes;
// compare their common final state, not their individual timestep proposals.
template <bool Adaptive, bool Dense>
double compare( const char *method, double direction, double tolerance = 1e-5, unsigned int cap = 5 )
{
    using O         = space_type;
    using C         = std::conditional_t<Adaptive, adaptive_control<O>, fixed_control<O>>;
    using S         = bdf_projected_step<O, C, Dense>;
    using R         = bdf_reference_step<O, C, Dense>;
    auto ops        = std::make_shared<O>( 3 );
    auto p          = std::make_shared<implicit_problem<O>>( ops );
    auto reduced    = std::make_shared<implicit_problem<O>>( ops, true );
    auto project    = std::make_shared<projector<O>>( ops );
    p->record_times = true;
    auto control    = [&]() {
        if constexpr ( Adaptive )
        {
            typename C::params c;
            c.initial_step       = .02;
            c.maximum_step       = .04;
            c.relative_tolerance = tolerance;
            c.absolute_tolerance = tolerance * .01;
            return std::make_shared<C>( ops, c );
        }
        else
            return std::make_shared<C>( typename C::params{ .01 } );
    };
    typename S::params sp{ method };
    sp.adaptive      = Adaptive;
    sp.maximum_order = cap;
    typename R::params rp{ method };
    rp.adaptive      = Adaptive;
    rp.maximum_order = cap;
    S         step( ops, p, control(), implicit_newton<O>::make( ops ), sp, project );
    R         reference( ops, reduced, control(), implicit_newton<O>::make( ops ), rp );
    buffer<O> state( *ops ), ref( *ops ), out( *ops ), expected( *ops ), sample( *ops ), ref_sample( *ops ),
        scratch( *ops );
    double       time   = .3;
    const double target = time + direction * .4;
    exact( *ops, *state, time );
    ops->assign( *state, *ref );
    std::size_t  count   = 0;
    unsigned int highest = 0;
    while ( time != target && count < 20000 )
    {
        step.set_time( time );
        step.set_target_time( target );
        reference.set_time( time );
        reference.set_target_time( target );
        step.apply( *state, *out );
        check( step.get_status() == single_step_status::converged, "Projected BDF solve failed" );
        if constexpr ( !Adaptive )
        {
            reference.apply( *ref, *expected );
            check( reference.get_status() == single_step_status::converged, "Reduced BDF solve failed" );
            check( step.get_order() == reference.get_order(), "Projection changed fixed BDF order" );
            check( step.get_dt() == reference.get_dt(), "Projection changed fixed BDF timestep" );
            check(
                difference( *ops, *out, *expected, *scratch ) < 2e-8, "Projected BDF differs from reduced equation"
            );
        }
        check( constraint( *ops, *out, *scratch ) < 2e-12, "Projected BDF endpoint constraint" );
        highest = std::max( highest, step.get_order() );
        if constexpr ( Dense )
        {
            auto       view  = step.get_continuous_integration();
            const auto calls = project->calls;
            for ( double theta : { 0., .2, .6, 1. } )
            {
                view.evaluate( theta, *sample );
                if constexpr ( !Adaptive )
                {
                    reference.get_continuous_integration().evaluate( theta, *ref_sample );
                    check( difference( *ops, *sample, *ref_sample, *scratch ) < 3e-8, "Projected BDF dense mismatch" );
                }
                check( constraint( *ops, *sample, *scratch ) < 3e-12, "Projected BDF dense constraint" );
            }
            check( project->calls == calls, "BDF sampling invoked projector" );
        }
        const double next     = step.get_dt() == target - time ? target : time + step.get_dt();
        const double ref_next = reference.get_dt() == target - time ? target : time + reference.get_dt();
        step.finalize( adaptation_status::accepted, next, *out );
        if constexpr ( !Adaptive )
            reference.finalize( adaptation_status::accepted, ref_next, *expected );
        ops->assign( *out, *state );
        if constexpr ( !Adaptive )
            ops->assign( *expected, *ref );
        time = next;
        ++count;
    }
    check( time == target, "BDF projection trajectory budget exhausted" );
    check( project->calls == p->stage_times.size(), "BDF solve escaped projection" );
    if constexpr ( Adaptive )
    {
        time = .3;
        for ( std::size_t k = 0; time != target && k < 20000; ++k )
        {
            reference.set_time( time );
            reference.set_target_time( target );
            reference.apply( *ref, *expected );
            check( reference.get_status() == single_step_status::converged, "Reduced adaptive BDF failed" );
            time = reference.get_dt() == target - time ? target : time + reference.get_dt();
            reference.finalize( adaptation_status::accepted, time, *expected );
            ops->assign( *expected, *ref );
        }
        check( time == target, "Reduced BDF budget exhausted" );
        check(
            difference( *ops, *state, *ref, *scratch ) < 20 * tolerance,
            "Independently adapted projected/reduced trajectories disagree"
        );
    }
    if ( std::string( method ) == "BDF2" )
        check( highest == 2, "Projection incorrectly restarted BDF2 after every commit" );
    if ( std::string( method ) == "BDF" )
        check( highest >= std::min( cap, 3u ), "Projected adaptive BDF never increased order" );
    exact( *ops, *expected, target );
    return difference( *ops, *state, *expected, *scratch );
}

double fixed_error( const char *method, double dt )
{
    using O         = space_type;
    using S         = bdf_projected_step<O>;
    auto ops        = std::make_shared<O>( 3 );
    auto p          = std::make_shared<implicit_problem<O>>( ops );
    auto projection = std::make_shared<projector<O>>( ops );
    S    step(
        ops, p, std::make_shared<fixed_control<O>>( fixed_control<O>::params{ dt } ), implicit_newton<O>::make( ops ),
        S::params{ method }, projection
    );
    buffer<O> state( *ops ), reference( *ops ), scratch( *ops );
    exact( *ops, *state );
    double t = 0;
    for ( int i = 0; i < int( std::round( .4 / dt ) ); ++i )
    {
        step.set_time( t );
        step.set_target_time( .4 );
        step.apply( *state, *state );
        check( step.get_status() == single_step_status::converged, "BDF projection refinement failed" );
        t = step.get_dt() == .4 - t ? .4 : t + step.get_dt();
        step.finalize( adaptation_status::accepted, t, *state );
    }
    exact( *ops, *reference, t );
    return difference( *ops, *state, *reference, *scratch );
}

// Exact prehistory isolates formal q=3..5 from cold-start accuracy. Use the
// canonical history and stage problem, without adding a public seeding API.
void formal_orders()
{
    using O     = space_type;
    auto ops    = std::make_shared<O>( 3 );
    auto p      = std::make_shared<implicit_problem<O>>( ops );
    auto solver = implicit_newton<O>::make( ops );
    solver->set_nonlinear_operator( p );
    projector<O> projection( ops );
    buffer<O>    state( *ops ), old( *ops ), work( *ops ), reference( *ops ), scratch( *ops );
    for ( unsigned int q = 3; q <= 5; ++q )
        for ( double sign : { -1., 1. } )
        {
            double errors[2]{};
            for ( int level = 0; level < 2; ++level )
            {
                const int                          count = 12 * ( 1 << level );
                const double                       h     = sign * .4 / count;
                bdf::divided_difference_history<O> history( *ops );
                history.initialize( q );
                exact( *ops, *old, -int( q ) * h );
                for ( unsigned int k = q; k > 0; --k )
                {
                    exact( *ops, *state, -int( k - 1 ) * h );
                    history.commit( *old, *state, h, k == q );
                    ops->assign( *state, *old );
                }
                for ( int k = 0; k < count; ++k )
                {
                    auto c = history.coefficients( q, h );
                    history.reference( c, *state, p->previous_state() );
                    ops->assign_scalar( 0., p->known_rhs() );
                    p->set_time( ( k + 1 ) * h );
                    auto shift = detail::balanced_affine_shift( c.alpha, 1. );
                    p->set_affine_shift( shift.a, shift.b );
                    ops->assign( *state, *work );
                    check(
                        solver->solve( *work ) && projection.apply( *work ), "Projected high-order formula solve failed"
                    );
                    history.commit( *state, *work, h, false );
                    ops->assign( *work, *state );
                }
                exact( *ops, *reference, sign * .4 );
                errors[level] = difference( *ops, *state, *reference, *scratch );
            }
            check( std::log2( errors[0] / errors[1] ) > q - .4, "Projection lost formal high-order BDF accuracy" );
        }
}

int main()
try
{
    backend_type::init_device();
    for ( const char *method : { "BDF1", "BDF2" } )
    {
        compare<false, false>( method, 1 );
        compare<false, true>( method, -1 );
        compare<true, true>( method, 1 );
        const double p = std::string( method ) == "BDF1" ? 1 : 2;
        check(
            fixed_error( method, .02 ) / fixed_error( method, .01 ) > std::pow( 2., p - .35 ),
            "Projected BDF1/2 order regression"
        );
    }
    for ( unsigned int cap = 1; cap <= 5; ++cap )
        compare<true, true>( "BDF", 1, 1e-5, cap );
    const double coarse = compare<true, false>( "BDF", -1, 1e-4 ), fine = compare<true, true>( "BDF", -1, 1e-6 );
    check( fine < coarse * .5, "Projected automatic BDF tolerance refinement failed" );
    formal_orders();
    std::cout << checks << " BDF projection accuracy checks passed\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
