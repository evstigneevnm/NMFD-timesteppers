#include "implicit_test_support.h"
#include <nmfd/time_steppers/runge_kutta/imex_time_step.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_scipy.h>
#include <cstdint>
#include <map>

using namespace nmfd::time_steppers;
using nmfd::time_steppers::tests::require;

// An 8192-component diagonal ODE: u'=.25*u-1.25*u=-u, u(0)=1.
// It exercises real Newton/GMRES without assembling a matrix or accessing state
// on the host. Both public copy paths reject vector-sized transfers on every backend.
struct guarded_memory : backend_type::memory_type
{
    using parent = backend_type::memory_type;
    inline static std::map<std::uintptr_t, std::size_t> allocations;
    static void                                         malloc( void **pointer, std::size_t bytes )
    {
        parent::malloc( pointer, bytes );
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
        if ( bytes > sizeof( double ) )
            throw std::runtime_error( "IMEX vector-sized host download" );
        parent::copy_to_host( bytes, src, dst );
    }
    static void copy_from_host( std::size_t bytes, const void *src, void *dst )
    {
        if ( bytes > sizeof( double ) )
            throw std::runtime_error( "IMEX vector-sized device upload" );
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
                throw std::runtime_error( "IMEX bulk get/set transfer" );
            typename backend_type::copy_type{}( count, src, dst );
        }
    };
};
using space_type  = nmfd::operations::scfd_vector_space<guarded_backend, double>;
using state_type  = space_type::vector_type;
using buffer_type = nmfd::detail::vector_wrap<space_type>;

struct partition
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
    double                      lambda, a = 0, b = 1;
    partition( std::shared_ptr<space_type> operations, double rate )
        : ops( std::move( operations ) ), previous( *ops ), rhs( *ops ),
          jacobi( std::make_shared<jacobian>( jacobian{ ops } ) ), lambda( rate )
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
        ops->assign_lin_comb( a - b * lambda, in, -a, *previous, out );
        ops->add_lin_comb( -b, *rhs, out );
    }
    bool evaluate_rate( const state_type &in, state_type &out ) const
    {
        ops->assign_lin_comb( lambda, in, out );
        return true;
    }
    void set_linearization_point( const state_type & )
    {
        jacobi->factor = a - b * lambda;
    }
    std::shared_ptr<const jacobian> get_jacobi_operator() const
    {
        return jacobi;
    }
};
struct problem
{
    using bundle_type = runge_kutta::imex_operator<partition, partition>;
    std::shared_ptr<bundle_type> bundle;
    std::shared_ptr<bundle_type> get_imex_operator() const
    {
        return bundle;
    }
};

int main()
try
{
    backend_type::init_device();
    using log            = scfd::utils::log_std;
    using monitor        = nmfd::solvers::monitor_krylov<space_type, log>;
    using linear         = nmfd::solvers::gmres<space_type, monitor, log, partition::jacobian>;
    using iteration      = nmfd::solvers::newton_iteration<space_type, partition, linear>;
    using solver         = nmfd::solvers::nonlinear_solver<space_type, log, partition, iteration>;
    using strategy       = nmfd::solvers::default_convergence_strategy<space_type, log, partition>;
    using controller     = integration::time_step_adaptation_constant<space_type>;
    using step_type      = runge_kutta::imex_time_step<space_type, problem, controller, solver>;
    auto        ops      = std::make_shared<space_type>( 8192 );
    auto        assembly = std::make_shared<problem>( problem{ std::make_shared<problem::bundle_type>(
        std::make_shared<partition>( ops, .25 ), std::make_shared<partition>( ops, -1.25 )
    ) } );
    buffer_type initial( *ops ), out( *ops );
    initial.start_use();
    out.start_use();
    ops->assign_scalar( 1., *initial );
    bool rejected = false;
    try
    {
        ops->get( *initial, static_cast<double *>( nullptr ) );
    }
    catch ( const std::runtime_error & )
    {
        rejected = true;
    }
    require( rejected, "IMEX bulk-copy guard was not connected to vector operations" );
    for ( const char *method : { "IMEX_EULER", "IMEX_HEUN_TR2", "IMEX_ARS222", "IMEX_ARS233" } )
    {
        linear::params lp;
        lp.basis_size        = 4;
        lp.batch_size        = 1;
        lp.monitor.abs_tol   = 1e-12;
        lp.monitor.rel_tol   = 1e-11;
        auto             lin = std::make_shared<linear>( ops, nullptr, lp );
        strategy::params np;
        np.abs_tol  = 1e-10;
        np.rel_tol  = 1e-12;
        auto newton = std::make_shared<solver>(
            ops, nullptr, std::make_shared<iteration>( ops, lin ), std::make_shared<strategy>( ops, nullptr, np )
        );
        auto step = std::make_shared<step_type>(
            ops, assembly, std::make_shared<controller>( controller::params{ .02 } ), newton,
            step_type::params{ method }
        );
        integration::time_integrator<space_type, step_type> flow( ops, step );
        flow.set_time_interval( 0, .2 );
        for ( int run = 0; run < 2; ++run )
        {
            flow.apply( *initial, *out );
            require( flow.get_status() == integration_status::completed, "Backend-local IMEX solve failed" );
            ops->add_mul_scalar( -std::exp( -.2 ), 1., *out );
            require(
                ops->norm_l2( *out ) / std::sqrt( double( ops->size() ) ) < .005, "Backend-local IMEX trajectory error"
            );
        }
    }
    using adaptive_controller = integration::time_step_adaptation_scipy<space_type>;
    using adaptive_step       = runge_kutta::imex_time_step<space_type, problem, adaptive_controller, solver>;
    linear::params lp;
    lp.basis_size        = 4;
    lp.batch_size        = 1;
    lp.monitor.abs_tol   = 1e-12;
    lp.monitor.rel_tol   = 1e-11;
    auto             lin = std::make_shared<linear>( ops, nullptr, lp );
    strategy::params np;
    np.abs_tol  = 1e-10;
    np.rel_tol  = 1e-12;
    auto newton = std::make_shared<solver>(
        ops, nullptr, std::make_shared<iteration>( ops, lin ), std::make_shared<strategy>( ops, nullptr, np )
    );
    adaptive_controller::params p;
    p.relative_tolerance = 1e-7;
    p.absolute_tolerance = 1e-9;
    p.initial_step = p.maximum_step = .2;
    auto adaptive                   = std::make_shared<adaptive_step>(
        ops, assembly, std::make_shared<adaptive_controller>( ops, p ), newton,
        adaptive_step::params{ "IMEX_ARK324L2SA" }
    );
    integration::time_integrator<space_type, adaptive_step> flow( ops, adaptive );
    flow.set_time_interval( 0, .2 );
    for ( int run = 0; run < 2; ++run )
    {
        flow.apply( *initial, *out );
        require(
            flow.get_status() == integration_status::completed && flow.get_steps() > 1,
            "Backend-local adaptive IMEX did not control its error"
        );
        ops->add_mul_scalar( -std::exp( -.2 ), 1., *out );
        require(
            ops->norm_l2( *out ) / std::sqrt( double( ops->size() ) ) < 1e-6, "Backend-local adaptive IMEX accuracy"
        );
    }
    std::cout << "Fixed/adaptive matrix-free IMEX, 8192 unknowns and blocked vector-sized host transfers: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
