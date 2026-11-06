#include "implicit_bdf_test.h"
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
            throw std::runtime_error( "BDF vector-sized host download" );
        parent::copy_to_host( bytes, src, dst );
    }
    static void copy_from_host( std::size_t bytes, const void *src, void *dst )
    {
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
    for ( const char *method : { "BDF1", "BDF2" } )
    {
        linear::params lp;
        lp.basis_size      = 4;
        lp.batch_size      = 1;
        lp.monitor.abs_tol = 1e-12;
        lp.monitor.rel_tol = 1e-11;
        strategy::params np;
        np.abs_tol         = 1e-10;
        np.rel_tol         = 1e-12;
        auto linear_solver = std::make_shared<linear>( ops, nullptr, lp );
        auto solver        = std::make_shared<solver_type>(
            ops, nullptr, std::make_shared<iteration>( ops, linear_solver ),
            std::make_shared<strategy>( ops, nullptr, np )
        );
        auto step = std::make_shared<step_type>(
            ops, problem, std::make_shared<control_type>( control_type::params{ .02 } ), solver,
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
                ops->norm_l2( *state ) / std::sqrt( double( ops->size() ) ) < .002, "Matrix-free BDF trajectory error"
            );
        }
    }
    std::cout << "Matrix-free BDF, 8192 unknowns and blocked vector-sized host transfers: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
