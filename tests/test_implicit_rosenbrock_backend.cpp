#include "implicit_rosenbrock_mass_time_problem.h"
#include "implicit_rosenbrock_dae_problem.h"
#include "implicit_rosenbrock_variable_mass_problem.h"
#include "implicit_rosenbrock_index2_problem.h"
#include <nmfd/time_steppers/integration/time_step_adaptation_scipy.h>
#include <cstdint>
#include <map>

using namespace rosenbrock_tests;

// u'=-u for 8192 components, u(0)=1. All rate/Jacobian work and GMRES
// algebra stay on the selected backend. Both bulk transfer paths are guarded;
// scalar reductions remain allowed. Repeated steps must not allocate vectors.
struct guarded_memory : backend_type::memory_type
{
    using parent = backend_type::memory_type;
    inline static std::map<std::uintptr_t, std::size_t> allocations;
    inline static std::size_t                           allocation_count = 0;
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
        if ( bytes > sizeof( double ) )
            throw std::runtime_error( "ROS2 vector-sized host download" );
        parent::copy_to_host( bytes, src, dst );
    }
    static void copy_from_host( std::size_t bytes, const void *src, void *dst )
    {
        if ( bytes > sizeof( double ) )
            throw std::runtime_error( "ROS2 vector-sized device upload" );
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
                throw std::runtime_error( "ROS2 bulk get/set transfer" );
            typename backend_type::copy_type{}( count, src, dst );
        }
    };
};
struct space : nmfd::operations::scfd_vector_space<guarded_backend, double>
{
    using base = nmfd::operations::scfd_vector_space<guarded_backend, double>;
    using base::base;
    mutable std::size_t activations = 0, fail_activation = 0;
    void                start_use_vector( vector_type &v ) const override
    {
        base::start_use_vector( v );
        if ( ++activations == fail_activation )
            throw std::runtime_error( "Injected ROS2 workspace failure" );
    }
};

int main()
try
{
    backend_type::init_device();
    using P                                  = problem<decay, space>;
    using L                                  = krylov_solver<P, space>;
    using C                                  = integration::time_step_adaptation_constant<space>;
    using S                                  = rosenbrock::time_step<space, P, C, L>;
    auto                             ops     = std::make_shared<space>( 8192 );
    auto                             p       = std::make_shared<P>( ops );
    auto                             linear  = make_krylov<L>( ops );
    auto                             control = std::make_shared<C>( C::params{ .02 } );
    nmfd::detail::vector_wrap<space> state( *ops ), out( *ops );
    state.start_use();
    out.start_use();
    ops->assign_scalar( 1., *state );
    bool blocked = false;
    try
    {
        ops->get( *state, static_cast<double *>( nullptr ) );
    }
    catch ( const std::runtime_error & )
    {
        blocked = true;
    }
    check( blocked, "ROS2 bulk transfer guard is disconnected" );
    blocked = false;
    try
    {
        guarded_memory::copy_to_host( 8192 * sizeof( double ), nullptr, nullptr );
    }
    catch ( const std::runtime_error & )
    {
        blocked = true;
    }
    check( blocked, "ROS2 raw download guard is disconnected" );
    for ( const char *method : { "ROS2", "ROS3", "ROS34PW2" } )
    {
        const rosenbrock::method_table table( method );
        const auto                     workspace = table.stage_count() + 2;
        const auto                     baseline  = guarded_memory::allocations.size();
        for ( unsigned int i = 1; i <= workspace; ++i )
        {
            ops->fail_activation = ops->activations + i;
            bool failed          = false;
            try
            {
                S invalid( ops, p, control, linear, S::params{ method } );
            }
            catch ( const std::runtime_error & )
            {
                failed = true;
            }
            check(
                failed && guarded_memory::allocations.size() == baseline, "Partial ROS2 construction leaked workspace"
            );
        }
        ops->fail_activation = 0;
        {
            ops->assign_scalar( 1., *state );
            S step( ops, p, control, linear, S::params{ method } );
            check(
                guarded_memory::allocations.size() == baseline + workspace,
                "Rosenbrock retained unnecessary workspace vectors"
            );
            step.apply( *state, *out );
            check( step.get_status() == single_step_status::converged, "Matrix-free ROS2 warmup failed" );
            step.finalize( adaptation_status::accepted, .02, *out );
            const auto allocations = guarded_memory::allocation_count;
            for ( unsigned int i = 0; i < 3; ++i )
            {
                step.reset();
                step.set_time( 0 );
                step.set_target_time( .02 );
                step.apply( *state, *out );
                check( step.get_status() == single_step_status::converged, "Repeated matrix-free ROS2 solve failed" );
                step.finalize( adaptation_status::accepted, .02, *out );
            }
            check( allocations == guarded_memory::allocation_count, "ROS2 allocated backend vectors after warmup" );
            const double g = table.gamma, z = -.02;
            const double exact =
                table.order() == 2 ? ( 1 + ( 1 - 2 * g ) * z ) / std::pow( 1 - g * z, 2 )
                : table.stage_count() == 3
                    ? ( 1 + ( 1 - 3 * g ) * z + ( .5 - 3 * g + 3 * g * g ) * z * z ) / std::pow( 1 - g * z, 3 )
                    : ( 1 + ( 1 - 4 * g ) * z + ( .5 - 4 * g + 6 * g * g ) * z * z +
                        ( 1. / 6 - 2 * g + 6 * g * g - 4 * g * g * g ) * z * z * z ) /
                          std::pow( 1 - g * z, 4 );
            ops->add_mul_scalar( -exact, 1., *out );
            check( ops->norm( *out ) < 1e-10, "Matrix-free ROS2 output disagrees with its stability function" );
        }
    }
    auto adaptive_mode = [&]( auto selected_problem, bool nonautonomous, const char *method, auto dense,
                              bool mass_form = false ) {
        using Problem = typename decltype( selected_problem )::element_type;
        using Linear  = krylov_solver<Problem, space>;
        using Control = integration::time_step_adaptation_scipy<space>;
        using Step    = rosenbrock::time_step<space, Problem, Control, Linear, decltype( dense )::value>;
        typename Control::params cp;
        cp.relative_tolerance = 1e-4;
        cp.absolute_tolerance = 1e-6;
        cp.initial_step = cp.maximum_step = .02;
        auto                  control     = std::make_shared<Control>( ops, cp );
        auto                  solver      = make_krylov<Linear>( ops );
        typename Step::params sp;
        sp.method            = method;
        sp.autonomous        = !nonautonomous;
        sp.mass_form         = mass_form;
        const auto stages    = rosenbrock::method_table( method ).stage_count();
        const auto workspace = stages + 2 + ( nonautonomous ? 1 : 0 ) + ( dense ? stages - 1 : 0 );
        const auto before    = guarded_memory::allocations.size();
        for ( unsigned int i = 1; i <= workspace; ++i )
        {
            ops->fail_activation = ops->activations + i;
            bool failed          = false;
            try
            {
                Step invalid( ops, selected_problem, control, solver, sp );
            }
            catch ( const std::runtime_error & )
            {
                failed = true;
            }
            check(
                failed && guarded_memory::allocations.size() == before,
                "Partial adaptive/non-autonomous ROS2 construction leaked workspace"
            );
        }
        ops->fail_activation = 0;
        Step step( ops, selected_problem, control, solver, sp );
        check(
            guarded_memory::allocations.size() == before + workspace,
            "Adaptive ROS2 allocated an unnecessary error or time-derivative vector"
        );
        ops->assign_scalar( std::exp( .74 ), *state );
        step.set_time( .37 );
        step.set_target_time( .39 );
        step.apply( *state, *out );
        check( step.get_status() == single_step_status::converged, "Guarded adaptive ROS2 warmup failed" );
        if constexpr ( dense )
        {
            auto view = step.get_continuous_integration();
            for ( double theta : { 0., .25, .5, .75, 1. } )
                view.evaluate( theta, *out );
        }
        step.finalize( adaptation_status::accepted, .37 + step.get_dt(), *out );
        const auto allocated = guarded_memory::allocation_count;
        for ( unsigned int repeat = 0; repeat < 3; ++repeat )
        {
            step.reset();
            step.set_time( .37 );
            step.set_target_time( .39 );
            step.apply( *state, *out );
            check( step.get_status() == single_step_status::converged, "Guarded adaptive ROS2 restart failed" );
            if constexpr ( dense )
            {
                auto view = step.get_continuous_integration();
                for ( double theta : { 0., .25, .5, .75, 1. } )
                    view.evaluate( theta, *out );
            }
            step.finalize( adaptation_status::accepted, .37 + step.get_dt(), *out );
        }
        check( guarded_memory::allocation_count == allocated, "Adaptive ROS2 allocated backend memory after warmup" );
    };
    auto time = std::make_shared<time_problem<space>>( ops );
    for ( const char *method : { "ROS2", "ROS3" } )
    {
        adaptive_mode( p, false, method, std::false_type{} );
        adaptive_mode( time, true, method, std::false_type{} );
        adaptive_mode( p, false, method, std::true_type{} );
        adaptive_mode( time, true, method, std::true_type{} );
    }
    auto approximate          = std::make_shared<w_problem<decay, space>>( ops );
    approximate->model.factor = .3;
    auto approximate_time     = std::make_shared<w_time_problem<space>>( ops );
    adaptive_mode( approximate, false, "ROS34PW2", std::false_type{} );
    adaptive_mode( approximate_time, true, "ROS34PW2", std::false_type{} );
    adaptive_mode( approximate, false, "ROS34PW2", std::true_type{} );
    adaptive_mode( approximate_time, true, "ROS34PW2", std::true_type{} );
    auto mass = std::make_shared<mass_problem<noncommuting_decay, space>>( ops );
    for ( const char *method : { "ROS2", "ROS3", "ROS34PW2" } )
    {
        auto timed_mass = std::make_shared<mass_time_problem<space>>( ops );
        if ( std::string( method ) == "ROS34PW2" )
        {
            timed_mass->model.factor = .4;
            timed_mass->model.offset = .2;
        }
        adaptive_mode( timed_mass, true, method, std::false_type{}, true );
        adaptive_mode( timed_mass, true, method, std::true_type{}, true );
        // ROS2/ROS3 require the exact state Jacobian; only W uses this B.
        if ( std::string( method ) == "ROS34PW2" )
        {
            adaptive_mode( mass, false, method, std::false_type{}, true );
            adaptive_mode( mass, false, method, std::true_type{}, true );
        }
        else
        {
            auto exact_mass = std::make_shared<mass_problem<coupled_decay, space>>( ops );
            adaptive_mode( exact_mass, false, method, std::false_type{}, true );
            adaptive_mode( exact_mass, false, method, std::true_type{}, true );
        }
    }
    check( mass->residuals > 0, "Mass-form transfer guard did not exercise residual stages" );
    auto variable   = std::make_shared<variable_mass_problem<coupled_variable_mass_model, space>>( ops );
    auto state_mass = std::make_shared<variable_mass_problem<state_mass_rate_model, space>>( ops );
    for ( const char *method : { "ROS2", "ROS3", "ROS34PW2" } )
    {
        variable->model.factor = std::string( method ) == "ROS34PW2" ? .5 : 1;
        adaptive_mode( variable, true, method, std::false_type{} );
        adaptive_mode( variable, true, method, std::true_type{} );
        adaptive_mode( state_mass, false, method, std::false_type{} );
        adaptive_mode( state_mass, false, method, std::true_type{} );
    }
    check(
        variable->rates > 0 && variable->derivatives > 0 && variable->residuals == 0,
        "Variable-mass guard did not exercise problem-owned rates/derivatives"
    );
    auto dae_mode = [&]( auto adaptive ) {
        using Problem = dae_problem<space>;
        using Linear  = krylov_solver<Problem, space>;
        using Base    = integration::time_step_adaptation_scipy<space>;
        using Control = std::conditional_t<decltype( adaptive )::value, dae_control<Base, Problem>, C>;
        using Step    = rosenbrock::time_step<space, Problem, Control, Linear>;
        auto p        = std::make_shared<Problem>( ops );
        p->factor     = .5;
        auto solver   = make_krylov<Linear>( ops );
        auto control  = [&] {
            if constexpr ( adaptive )
            {
                typename Base::params cp;
                cp.relative_tolerance = cp.absolute_tolerance = 1e-4;
                cp.initial_step = cp.maximum_step = .02;
                auto checked                      = std::make_shared<Control>( p, cp );
                checked->constraint_tolerance     = 1e-7;
                return checked;
            }
            else
                return std::make_shared<Control>( typename Control::params{ .02 } );
        }();
        const auto baseline = guarded_memory::allocations.size();
        for ( unsigned int i = 1; i <= 6; ++i )
        {
            ops->fail_activation = ops->activations + i;
            bool failed          = false;
            try
            {
                Step invalid( ops, p, control, solver, dae_params<Step>() );
            }
            catch ( const std::runtime_error & )
            {
                failed = true;
            }
            check(
                failed && guarded_memory::allocations.size() == baseline, "Partial DAE construction leaked workspace"
            );
        }
        ops->fail_activation = 0;
        Step step( ops, p, control, solver, dae_params<Step>() );
        check( guarded_memory::allocations.size() == baseline + 6, "Index-1 mode allocated additional engine vectors" );
        ops->assign_scalar( 1., *state );
        step.set_target_time( .02 );
        step.apply( *state, *out );
        check( step.get_status() == single_step_status::converged, "Guarded DAE warmup failed" );
        step.finalize( adaptation_status::accepted, step.get_dt(), *out );
        const auto allocated = guarded_memory::allocation_count;
        for ( unsigned int repeat = 0; repeat < 3; ++repeat )
        {
            step.reset();
            step.apply( *state, *out );
            check( step.get_status() == single_step_status::converged, "Guarded DAE repeated solve failed" );
            check(
                p->constraint_norm( *out, p->known_rhs() ) < ( adaptive ? 1e-7 : 1e-4 ),
                "Guarded DAE constraint defect is unbounded"
            );
            step.finalize( adaptation_status::accepted, step.get_dt(), *out );
        }
        check( allocated == guarded_memory::allocation_count, "Index-1 solve allocated workspace after warmup" );
    };
    dae_mode( std::false_type{} );
    dae_mode( std::true_type{} );
    auto index2_mode = [&]( auto moving, auto dense, auto adaptive ) {
        using Problem = index2_problem<decltype( moving )::value, space, decltype( dense )::value>;
        using Linear  = krylov_solver<Problem, space>;
        using Base    = std::conditional_t<
               decltype( adaptive )::value, integration::time_step_adaptation_scipy<space>,
               integration::time_step_adaptation_constant<space>>;
        using Control   = index2_control<Base, Problem>;
        using Step      = rosenbrock::time_step<space, Problem, Control, Linear, decltype( dense )::value>;
        auto operations = std::make_shared<space>( Problem::block_size * 1638 );
        auto problem    = std::make_shared<Problem>( operations );
        problem->rate_consistency_tolerance = 1e-3;
        typename Linear::params lp;
        lp.basis_size            = 6;
        lp.batch_size            = 1;
        lp.monitor.rel_tol       = 1e-12;
        lp.monitor.abs_tol       = 0;
        lp.monitor.max_iters_num = 60;
        auto linear              = std::make_shared<Linear>( operations, nullptr, lp );
        auto control             = [&] {
            if constexpr ( adaptive )
            {
                typename Base::params cp;
                cp.relative_tolerance = 1e-4;
                cp.absolute_tolerance = 1e-6;
                cp.initial_step = cp.maximum_step = .01;
                return std::make_shared<Control>( problem, operations, cp );
            }
            else
                return std::make_shared<Control>( problem, typename Base::params{ .01 } );
        }();
        control->constraint_tolerance = 1e-3;
        nmfd::detail::vector_wrap<space> input( *operations ), output( *operations ), sample( *operations );
        input.start_use();
        output.start_use();
        sample.start_use();
        const auto baseline  = guarded_memory::allocations.size();
        const auto workspace = dense ? 9u : 6u;
        for ( unsigned int i = 1; i <= workspace; ++i )
        {
            operations->fail_activation = operations->activations + i;
            bool failed                 = false;
            try
            {
                Step invalid( operations, problem, control, linear, index2_params<Step>() );
            }
            catch ( const std::runtime_error & )
            {
                failed = true;
            }
            check(
                failed && guarded_memory::allocations.size() == baseline,
                "Partial index-2 construction leaked backend storage"
            );
        }
        operations->fail_activation = 0;
        Step step( operations, problem, control, linear, index2_params<Step>() );
        check(
            guarded_memory::allocations.size() == baseline + workspace, "Index-2 engine allocated unexpected workspace"
        );
        std::size_t allocated = 0;
        for ( unsigned int repeat = 0; repeat < 4; ++repeat )
        {
            step.reset();
            problem->initialize_state( .37, *input );
            step.set_time( .37 );
            step.set_target_time( .38 );
            step.apply( *input, *output );
            check( step.get_status() == single_step_status::converged, "Guarded full-system index-2 solve failed" );
            if constexpr ( dense )
            {
                auto view = step.get_continuous_integration();
                for ( double theta : { 0., .25, .5, .75, 1. } )
                {
                    view.evaluate( theta, *sample );
                    problem->reconstruct( *sample, *sample );
                    check(
                        problem->constraint_norm( *sample, problem->known_rhs() ) < 1e-10,
                        "Backend constrained reconstruction failed"
                    );
                }
            }
            step.finalize( adaptation_status::accepted, .37 + step.get_dt(), *output );
            if ( !repeat )
                allocated = guarded_memory::allocation_count;
            else
                check( allocated == guarded_memory::allocation_count, "Index-2 allocated vectors after warmup" );
        }
        check( problem->rates == ( dense ? 8u : 0u ), "Disabled index-2 dense rates were evaluated" );
    };
    auto index2_variants = [&]( auto moving ) {
        index2_mode( moving, std::false_type{}, std::false_type{} );
        index2_mode( moving, std::true_type{}, std::false_type{} );
        index2_mode( moving, std::false_type{}, std::true_type{} );
        index2_mode( moving, std::true_type{}, std::true_type{} );
    };
    index2_variants( std::false_type{} );
    index2_variants( std::true_type{} );
    check( time->derivatives > 0 && time->time == .37, "Guarded ROS2 omitted f_t or changed base time" );
    std::cout << "Fixed/adaptive ROS2/ROS3/ROS34PW2/ROSI2P2, fixed workspace and blocked vector-sized host transfers: "
              << checks << " checks PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
