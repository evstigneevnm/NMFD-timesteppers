#include "implicit_bdf_projection_test.h"
#include "implicit_imex_projection_problem.h"
#include "implicit_rosenbrock_projection_problem.h"
#include "projection_transfer_guard.h"
using namespace projection_tests;

enum class family
{
    explicit_rk,
    dirk,
    imex,
    bdf,
    rosenbrock
};

// The established constrained three-component models differ by method family,
// but bool and void projectors perform exactly the same P(x,y,z). Compare whole
// trajectories, embedded defects, dense output and publication behavior.
template <family Family, class Space>
struct assembly
{
    using C    = rosenbrock_recording_control<Space>;
    using IMEX = imex_projection_problem<Space>;
    using P    = std::conditional_t<
           Family == family::explicit_rk, problem<Space>,
           std::conditional_t<
               Family == family::imex, IMEX,
               std::conditional_t<
                   Family == family::rosenbrock, rosenbrock_projection_problem<Space>, implicit_problem<Space>>>>;
    using I = std::conditional_t<
        Family == family::imex, typename IMEX::implicit_type,
        std::conditional_t<Family == family::explicit_rk, implicit_problem<Space>, P>>;
    using L = std::conditional_t<
        Family == family::rosenbrock, projection_linear_solver<Space, I>, typename implicit_newton<Space, I>::solver>;
    template <class Operator>
    using step_type = std::conditional_t<
        Family == family::explicit_rk, runge_kutta::explicit_time_step<Space, P, C, true, Operator>,
        std::conditional_t<
            Family == family::dirk, runge_kutta::implicit_time_step<Space, P, C, L, true, Operator>,
            std::conditional_t<
                Family == family::imex, runge_kutta::imex_time_step<Space, P, C, L, true, Operator>,
                std::conditional_t<
                    Family == family::bdf, bdf::implicit_time_step<Space, P, C, L, true, Operator>,
                    rosenbrock::time_step<Space, P, C, L, true, Operator>>>>>;
    std::shared_ptr<Space> ops;
    std::shared_ptr<P>     p;
    std::shared_ptr<C>     control;
    std::shared_ptr<L>     solver;
    explicit assembly( std::shared_ptr<Space> operations )
        : ops( std::move( operations ) ), p( std::make_shared<P>( ops ) ), control( std::make_shared<C>( ops ) )
    {
        if constexpr ( Family == family::rosenbrock )
            solver = std::make_shared<L>( ops );
        else if constexpr ( Family != family::explicit_rk )
            solver = implicit_newton<Space, I>::make( ops );
    }
    template <class Operator>
    auto make( std::shared_ptr<Operator> op )
    {
        using S = step_type<Operator>;
        typename S::params sp;
        sp.maximum_attempts = 3;
        if constexpr ( Family == family::explicit_rk )
            sp.method = "RK45";
        if constexpr ( Family == family::dirk )
            sp.method = "ESDIRK2(1)3";
        if constexpr ( Family == family::imex )
            sp.method = "IMEX_ARK324L2SA";
        if constexpr ( Family == family::bdf )
        {
            sp.method   = "BDF2";
            sp.adaptive = true;
        }
        if constexpr ( Family == family::rosenbrock )
        {
            sp.method     = "ROS34PW2";
            sp.autonomous = false;
        }
        if constexpr ( Family == family::explicit_rk )
            return std::make_unique<S>( ops, p, control, sp, op );
        else
            return std::make_unique<S>( ops, p, control, solver, sp, op );
    }
};

template <family Family, class Space>
std::size_t compare( std::shared_ptr<Space> ops )
{
    assembly<Family, Space> flagged( ops ), direct( ops );
    auto                    boolean = std::make_shared<projector<Space>>( ops );
    auto                    mapping = std::make_shared<void_projector<Space>>( ops );
    const auto              start   = ops->allocations;
    auto                    a       = flagged.make( boolean );
    const auto              storage = ops->allocations - start;
    const auto              before  = ops->allocations;
    auto                    b       = direct.make( mapping );
    check( ops->allocations - before == storage, "Void projection added timestepper vectors" );
    buffer<Space> state( *ops ), reference( *ops ), out( *ops ), expected( *ops ), sample( *ops ), other( *ops ),
        scratch( *ops );
    exact( *ops, *state, .3 );
    ops->assign( *state, *reference );
    double      time        = .3;
    std::size_t allocations = 0, first_calls = 0;
    for ( int k = 0; k < 6; ++k )
    {
        a->set_time( time );
        b->set_time( time );
        a->set_target_time( 1. );
        b->set_target_time( 1. );
        a->apply( *state, *out );
        b->apply( *reference, *expected );
        check(
            a->get_status() == single_step_status::converged && b->get_status() == single_step_status::converged,
            "Bool/void projection trajectory failed"
        );
        check( difference( *ops, *out, *expected, *scratch ) < 1e-11, "Void projection changed numerical endpoint" );
        check(
            difference( *ops, *flagged.control->error, *direct.control->error, *scratch ) < 1e-11,
            "Void projection changed embedded defect"
        );
        check(
            boolean->calls == mapping->calls && mapping->calls > 0, "Projection invoked a different number of times"
        );
        const auto calls = mapping->calls;
        auto       av    = a->get_continuous_integration();
        auto       bv    = b->get_continuous_integration();
        for ( double theta : { 0., .2, .7, 1. } )
        {
            av.evaluate( theta, *sample );
            bv.evaluate( theta, *other );
            check( difference( *ops, *sample, *other, *scratch ) < 1e-11, "Void projection changed dense output" );
            check( constraint( *ops, *other, *scratch ) < 1e-11, "Void projection lost dense constraint" );
        }
        check( calls == mapping->calls, "Dense sampling invoked void projector" );
        time += a->get_dt();
        a->finalize( adaptation_status::accepted, time, *out );
        b->finalize( adaptation_status::accepted, time, *expected );
        ops->assign( *out, *state );
        ops->assign( *expected, *reference );
        if ( k == 0 )
        {
            allocations = ops->allocations;
            first_calls = mapping->calls;
        }
        check( allocations == ops->allocations, "Projection allocated backend vectors after warmup" );
    }
    if constexpr ( Family == family::bdf )
        check( b->get_order() == 2, "Void projection restarted BDF history" );
    return first_calls;
}

template <family Family>
void failure( std::size_t location, bool throws, bool alias, bool retry = false )
{
    using Space                 = space_type;
    auto                    ops = std::make_shared<Space>( 3 );
    assembly<Family, Space> context( ops );
    context.control->retry = retry;
    auto mapping           = std::make_shared<void_projector<Space>>( ops );
    if ( throws )
        mapping->throw_on = location;
    else
        mapping->invalid_on = location;
    auto          step = context.make( mapping );
    buffer<Space> state( *ops ), initial( *ops ), out( *ops ), sentinel( *ops ), scratch( *ops );
    exact( *ops, *state, .3 );
    ops->assign( *state, *initial );
    ops->assign_scalar( 7., *out );
    ops->assign( *out, *sentinel );
    step->set_time( .3 );
    step->set_target_time( .5 );
    bool caught = false;
    try
    {
        step->apply( *state, alias ? *state : *out );
    }
    catch ( const std::runtime_error & )
    {
        caught = true;
    }
    check( caught == throws, "Void projection exception was swallowed" );
    if ( !retry )
    {
        if ( !caught )
            check(
                step->get_status() == single_step_status::failed_nonfinite,
                "Void projection lost nonfinite failure status"
            );
        check( difference( *ops, *state, *initial, *scratch ) == 0, "Void projection failure changed input" );
        check( difference( *ops, *out, *sentinel, *scratch ) == 0, "Void projection failure published output" );
        bool refused = false;
        try
        {
            step->get_continuous_integration();
        }
        catch ( const std::logic_error & )
        {
            refused = true;
        }
        check( refused, "Void projection failure published dense data" );
        mapping->invalid_on = mapping->throw_on = 0;
        step->apply( *state, *out );
    }
    else
        check( step->get_attempts() == 2 && step->get_dt() == .01, "Void nonfinite projection did not shrink/retry" );
    check( step->get_status() == single_step_status::converged, "Void projection did not recover" );
    auto         view = step->get_continuous_integration();
    const double next = view.evaluate( .5, *scratch );
    step->finalize( adaptation_status::accepted_modified, next, *scratch );
    bool expired = false;
    try
    {
        view.evaluate( .5, *out );
    }
    catch ( const std::logic_error & )
    {
        expired = true;
    }
    check( expired, "Void projection retained stale dense data" );
}

template <family Family>
void run()
{
    std::cout << "Bool/void projection family=" << int( Family ) << '\n';
    auto       ops   = std::make_shared<space_type>( 3 );
    const auto count = compare<Family>( ops );
    for ( auto location : { std::size_t( 1 ), std::size_t( 2 ), count } )
    {
        for ( bool throws : { false, true } )
            for ( bool alias : { false, true } )
                failure<Family>( location, throws, alias );
        failure<Family>( location, false, false, true );
    }
    assembly<Family, space_type> context( ops );
    bool                         refused = false;
    try
    {
        context.template make<void_projector<space_type>>( nullptr );
    }
    catch ( const std::invalid_argument & )
    {
        refused = true;
    }
    check( refused, "Stepper accepted a null void projector" );
    std::weak_ptr<void_projector<space_type>> weak;
    {
        auto mapping = std::make_shared<void_projector<space_type>>( ops );
        weak         = mapping;
        auto step    = context.make( mapping );
        mapping.reset();
        check( !weak.expired(), "Stepper lost void projector ownership" );
    }
    check( weak.expired(), "Void projector ownership cycle" );
    // Large mapping/estimation/dense work stays on the selected backend.
    compare<Family>( std::make_shared<counted_space<guarded_backend>>( 3 * 8192 ) );
}
int main()
try
{
    backend_type::init_device();
    run<family::explicit_rk>();
    run<family::dirk>();
    run<family::imex>();
    run<family::bdf>();
    run<family::rosenbrock>();
    check( guarded_memory::allocations.empty(), "Void projection leaked backend workspace" );
    std::cout << checks << " bool/void operator projection checks passed; guarded bulk host bytes=0\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
