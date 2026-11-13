#include "explicit_projection_problem.h"
#include <cstdint>
#include <map>

using namespace projection_tests;

// 8192 independent constrained blocks (24576 unknowns). All RHS, projection,
// error control, dense sampling and verification reductions stay on the backend.
// Both direct transfers and bulk get/set paths are guarded after initialization.
struct guarded_memory : backend_type::memory_type
{
    using base = backend_type::memory_type;
    inline static std::map<std::uintptr_t, std::size_t> allocations;
    inline static std::size_t                           allocation_count = 0;
    static void                                         malloc( void **pointer, std::size_t bytes )
    {
        base::malloc( pointer, bytes );
        ++allocation_count;
        if ( bytes )
            allocations.emplace( reinterpret_cast<std::uintptr_t>( *pointer ), bytes );
    }
    static void free( void *pointer )
    {
        allocations.erase( reinterpret_cast<std::uintptr_t>( pointer ) );
        base::free( pointer );
    }
    static bool owns( const void *pointer, std::size_t bytes )
    {
        const auto address    = reinterpret_cast<std::uintptr_t>( pointer );
        auto       allocation = allocations.upper_bound( address );
        if ( allocation == allocations.begin() )
            return false;
        --allocation;
        const auto offset = address - allocation->first;
        return offset <= allocation->second && bytes <= allocation->second - offset;
    }
    static void copy_to_host( std::size_t bytes, const void *src, void *dst )
    {
        if ( bytes > sizeof( double ) )
            throw std::runtime_error( "Stage projection bulk download" );
        base::copy_to_host( bytes, src, dst );
    }
    static void copy_from_host( std::size_t bytes, const void *src, void *dst )
    {
        if ( bytes > sizeof( double ) )
            throw std::runtime_error( "Stage projection bulk upload" );
        base::copy_from_host( bytes, src, dst );
    }
};
struct guarded_backend : backend_type
{
    using memory_type = guarded_memory;
    struct copy_type
    {
        template <class Ordinal, class T>
        void operator()( Ordinal count, const T *src, T *dst ) const
        {
            const auto bytes = std::size_t( count ) * sizeof( T );
            if ( count > 1 && ( !guarded_memory::owns( src, bytes ) || !guarded_memory::owns( dst, bytes ) ) )
                throw std::runtime_error( "Stage projection bulk get/set" );
            typename backend_type::copy_type{}( count, src, dst );
        }
    };
};

template <bool Dense, bool Adaptive>
void check_backend( const char *method )
{
    using O         = counted_space<guarded_backend>;
    using C         = std::conditional_t<Adaptive, adaptive_control<O>, fixed_control<O>>;
    using S         = projected_step<O, C, Dense>;
    auto ops        = std::make_shared<O>( 3 * 8192 );
    auto p          = std::make_shared<problem<O>>( ops );
    auto projection = std::make_shared<projector<O>>( ops );
    auto control    = [&]() {
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
    buffer<O> state( *ops ), out( *ops ), sample( *ops ), expected( *ops ), scratch( *ops );
    exact( *ops, *state );
    S    step( ops, p, control, typename S::params{ method, 32, true }, projection );
    auto run = [&]() {
        step.reset();
        step.set_time( 0 );
        step.set_target_time( .02 );
        step.apply( *state, *out );
        check( step.get_status() == single_step_status::converged, "Guarded projection failed" );
        check( constraint( *ops, *out, *scratch ) < 1e-13, "Large projected constraint defect" );
        exact( *ops, *expected, step.get_dt() );
        check( difference( *ops, *out, *expected, *scratch ) < 1e-7, "Large projected endpoint accuracy" );
        if constexpr ( Dense )
        {
            auto view = step.get_continuous_integration();
            for ( double theta : { 0., .2, .7, 1. } )
            {
                view.evaluate( theta, *sample );
                check( constraint( *ops, *sample, *scratch ) < 1e-13, "Large projected dense constraint" );
            }
        }
        step.finalize( adaptation_status::accepted, step.get_dt(), *out );
    };
    run();
    const auto allocated = guarded_memory::allocation_count;
    for ( unsigned int repeat = 0; repeat < 3; ++repeat )
        run();
    check( guarded_memory::allocation_count == allocated, "Projection allocated backend vectors after warmup" );
    check( projection->calls == p->calls, "Large stage rate escaped projection" );
    bool blocked = false;
    try
    {
        ops->get( *state, static_cast<double *>( nullptr ) );
    }
    catch ( const std::runtime_error & )
    {
        blocked = true;
    }
    check( blocked, "Projection bulk get guard is disconnected" );
    blocked = false;
    try
    {
        ops->set( static_cast<const double *>( nullptr ), *state );
    }
    catch ( const std::runtime_error & )
    {
        blocked = true;
    }
    check( blocked, "Projection bulk set guard is disconnected" );
}

int main()
try
{
    backend_type::init_device();
    for ( const char *method : { "RK45", "DOP853" } )
    {
        check_backend<false, false>( method );
        check_backend<true, false>( method );
        check_backend<false, true>( method );
        check_backend<true, true>( method );
        check( guarded_memory::allocations.empty(), "Projection workspace leaked" );
    }
    for ( bool upload : { false, true } )
    {
        bool blocked = false;
        try
        {
            if ( upload )
                guarded_memory::copy_from_host( 24, nullptr, nullptr );
            else
                guarded_memory::copy_to_host( 24, nullptr, nullptr );
        }
        catch ( const std::runtime_error & )
        {
            blocked = true;
        }
        check( blocked, "Projection raw transfer guard is disconnected" );
    }
    std::cout << checks << " explicit projection backend checks passed\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
