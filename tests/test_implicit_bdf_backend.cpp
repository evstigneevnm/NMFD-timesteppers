#include "implicit_bdf_test.h"
#include <nmfd/time_steppers/integration/time_step_adaptation_scipy.h>
#include <cstdint>
#include <map>

using namespace nmfd::time_steppers;
using nmfd::time_steppers::tests::require;

// u'=-u for 8192 independent components, u(0)=1. Matrix-free Newton/GMRES
// exercises real backend algebra. Transfers larger than one scalar are forbidden,
// and the copy functor separately rejects bulk get/set through host pointers.
struct guarded_memory : backend_type::memory_type
{
    using parent = backend_type::memory_type;
    inline static std::map<std::uintptr_t, std::size_t> allocations;
    inline static std::size_t                           allocation_count = 0;
    inline static std::size_t                           transfer_count   = 0;
    static void                                         malloc( void **pointer, std::size_t bytes )
    {
        parent::malloc( pointer, bytes );
        ++allocation_count;
        if ( bytes )
            allocations.emplace( reinterpret_cast<std::uintptr_t>( *pointer ), bytes );
    }
    static void free( void *pointer )
    {
        allocations.erase( reinterpret_cast<std::uintptr_t>( pointer ) );
        parent::free( pointer );
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
        ++transfer_count;
        if ( bytes > sizeof( double ) )
            throw std::runtime_error( "BDF vector-sized host download" );
        parent::copy_to_host( bytes, src, dst );
    }
    static void copy_from_host( std::size_t bytes, const void *src, void *dst )
    {
        ++transfer_count;
        if ( bytes > sizeof( double ) )
            throw std::runtime_error( "BDF vector-sized device upload" );
        parent::copy_from_host( bytes, src, dst );
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
                throw std::runtime_error( "BDF bulk get/set transfer" );
            typename backend_type::copy_type{}( count, src, dst );
        }
    };
};
using space_type  = nmfd::operations::scfd_vector_space<guarded_backend, double>;
using state_type  = space_type::vector_type;
using buffer_type = nmfd::detail::vector_wrap<space_type>;

struct problem_type
{
    using vector_type = state_type;
    struct jacobian
    {
        std::shared_ptr<space_type> ops;
        double                      factor = 0;
        void                        apply( const state_type &in, state_type &out ) const
        {
            ops->assign_lin_comb( factor, in, out );
        }
    };
    using jacobi_operator_type = jacobian;
    std::shared_ptr<space_type> ops;
    buffer_type                 previous, rhs;
    std::shared_ptr<jacobian>   jacobi;
    double                      a = 0, b = 1;
    explicit problem_type( std::shared_ptr<space_type> operations )
        : ops( std::move( operations ) ), previous( *ops ), rhs( *ops ),
          jacobi( std::make_shared<jacobian>( jacobian{ ops } ) )
    {
        previous.start_use();
        rhs.start_use();
    }
    void set_time( double )
    {
    }
    void set_affine_shift( double mass, double force )
    {
        a = mass;
        b = force;
    }
    state_type &previous_state()
    {
        return *previous;
    }
    state_type &known_rhs()
    {
        return *rhs;
    }
    const state_type &previous_state() const
    {
        return *previous;
    }
    const state_type &known_rhs() const
    {
        return *rhs;
    }
    void apply( const state_type &in, state_type &out ) const
    {
        ops->assign_lin_comb( a + b, in, -a, *previous, out );
        ops->add_lin_comb( -b, *rhs, out );
    }
    void set_linearization_point( const state_type & )
    {
        jacobi->factor = a + b;
    }
    std::shared_ptr<const jacobian> get_jacobi_operator() const
    {
        return jacobi;
    }
};

struct adaptive_controller : integration::time_step_adaptation_scipy<space_type>
{
    using base = integration::time_step_adaptation_scipy<space_type>;
    using base::base;
    unsigned int maximum_order = 0;
    void         reset()
    {
        base::reset();
        maximum_order = 0;
    }
    adaptation_status assess(
        double t, double h, const state_type &in, const state_type &out, double &next, unsigned int exponent = 0,
        const state_type *error = nullptr
    )
    {
        maximum_order = std::max( maximum_order, exponent - 1 );
        return base::assess( t, h, in, out, next, exponent, error );
    }
};

int main()
try
{
    backend_type::init_device();
    using log           = scfd::utils::log_std;
    using monitor       = nmfd::solvers::monitor_krylov<space_type, log>;
    using linear        = nmfd::solvers::gmres<space_type, monitor, log, problem_type::jacobian>;
    using iteration     = nmfd::solvers::newton_iteration<space_type, problem_type, linear>;
    using solver_type   = nmfd::solvers::nonlinear_solver<space_type, log, problem_type, iteration>;
    using strategy      = nmfd::solvers::default_convergence_strategy<space_type, log, problem_type>;
    using control_type  = integration::time_step_adaptation_constant<space_type>;
    using step_type     = bdf::implicit_time_step<space_type, problem_type, control_type, solver_type>;
    auto        ops     = std::make_shared<space_type>( 8192 );
    auto        problem = std::make_shared<problem_type>( ops );
    buffer_type state( *ops );
    state.start_use();
    bool blocked = false;
    try
    {
        ops->get( *state, static_cast<double *>( nullptr ) );
    }
    catch ( const std::runtime_error & )
    {
        blocked = true;
    }
    require( blocked, "BDF bulk-copy guard is not connected" );
    auto make_solver = [&]() {
        linear::params lp;
        lp.basis_size      = 4;
        lp.batch_size      = 1;
        lp.monitor.abs_tol = 1e-12;
        lp.monitor.rel_tol = 1e-11;
        strategy::params np;
        np.abs_tol         = 1e-10;
        np.rel_tol         = 1e-12;
        auto linear_solver = std::make_shared<linear>( ops, nullptr, lp );
        return std::make_shared<solver_type>(
            ops, nullptr, std::make_shared<iteration>( ops, linear_solver ),
            std::make_shared<strategy>( ops, nullptr, np )
        );
    };
    for ( const char *method : { "BDF1", "BDF2", "BDF" } )
    {
        if ( std::string( method ) != "BDF" )
        {
            auto step = std::make_shared<step_type>(
                ops, problem, std::make_shared<control_type>( control_type::params{ .02 } ), make_solver(),
                step_type::params{ method }
            );
            integration::time_integrator<space_type, step_type> flow( ops, step );
            flow.set_time_interval( 0, .2 );
            for ( int run = 0; run < 2; ++run )
            {
                ops->assign_scalar( 1., *state );
                flow.apply( *state, *state );
                require( flow.get_status() == integration_status::completed, "Matrix-free BDF integration failed" );
                ops->add_mul_scalar( -std::exp( -.2 ), 1., *state );
                require(
                    ops->norm_l2( *state ) / std::sqrt( double( ops->size() ) ) < .002,
                    "Matrix-free BDF trajectory error"
                );
            }
        }
        using adaptive_control = adaptive_controller;
        using adaptive_step    = bdf::implicit_time_step<space_type, problem_type, adaptive_control, solver_type>;
        adaptive_control::params cp;
        cp.relative_tolerance = 1e-6;
        cp.absolute_tolerance = 1e-8;
        cp.initial_step = cp.maximum_step = .2;
        typename adaptive_step::params p{ method };
        p.adaptive    = true;
        auto control  = std::make_shared<adaptive_control>( ops, cp );
        auto adaptive = std::make_shared<adaptive_step>( ops, problem, control, make_solver(), p );
        integration::time_integrator<space_type, adaptive_step> adaptive_flow( ops, adaptive );
        adaptive_flow.set_time_interval( 0, .2 );
        for ( int run = 0; run < 2; ++run )
        {
            ops->assign_scalar( 1., *state );
            adaptive_flow.apply( *state, *state );
            require(
                adaptive_flow.get_status() == integration_status::completed,
                "Matrix-free adaptive BDF integration failed"
            );
            if ( std::string( method ) == "BDF" )
                require( control->maximum_order >= 3, "Guarded BDF did not exercise higher-order backend history" );
            ops->add_mul_scalar( -std::exp( -.2 ), 1., *state );
            require(
                ops->norm_l2( *state ) / std::sqrt( double( ops->size() ) ) < .0003,
                "Matrix-free adaptive BDF trajectory error"
            );
        }
    }
    // Sample higher-order history on the selected backend, without new vector
    // allocations or even scalar host transfers during interpolation itself.
    using dense_step = bdf::implicit_time_step<space_type, problem_type, adaptive_controller, solver_type, true>;
    adaptive_controller::params cp;
    cp.relative_tolerance = 1e-6;
    cp.absolute_tolerance = 1e-8;
    cp.initial_step = cp.maximum_step = .02;
    dense_step::params p{ "BDF" };
    p.adaptive = true;
    dense_step  dense( ops, problem, std::make_shared<adaptive_controller>( ops, cp ), make_solver(), p );
    buffer_type sample( *ops ), repeat( *ops );
    sample.start_use();
    repeat.start_use();
    ops->assign_scalar( 1., *state );
    dense.set_target_time( 1 );
    double       time          = 0;
    unsigned int maximum_order = 1;
    for ( int i = 0; i < 50; ++i )
    {
        dense.set_time( time );
        dense.apply( *state, *state );
        require( dense.get_status() == single_step_status::converged, "Guarded dense BDF failed" );
        maximum_order        = std::max( maximum_order, dense.get_order() );
        const auto allocated = guarded_memory::allocation_count;
        const auto transfers = guarded_memory::transfer_count;
        const auto view      = dense.get_continuous_integration();
        for ( double theta : { 0., .19, .67, 1., .19 } )
            view.evaluate( theta, *sample );
        view.evaluate( .19, *repeat );
        require( allocated == guarded_memory::allocation_count, "Dense BDF allocated backend memory" );
        require( transfers == guarded_memory::transfer_count, "Dense BDF sampling transferred host data" );
        ops->add_lin_comb( -1., *sample, *repeat );
        require( ops->norm( *repeat ) == 0, "Guarded repeated BDF sample changed" );
        time = view.evaluate( 1, *sample );
        dense.finalize( adaptation_status::accepted, time, *state );
    }
    require( maximum_order >= 3, "Guarded dense BDF did not exercise high-order interpolation" );
    std::cout << "Matrix-free BDF, 8192 unknowns and blocked vector-sized host transfers: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
