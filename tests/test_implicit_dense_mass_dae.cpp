#include "implicit_dense_test.h"
#include "implicit_mass_dae_problems.h"

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;

// Nonsingular mass cases reuse M=[[2,1],[1,3]], M(u)=1+u^2, and M(t)=1+t.
// Exact solutions are (exp(-t),exp(-2*t)), exp(-t), and 1/(1+t).
// Interior refinement must reflect the continuous-extension order, not just endpoints.
template <class Equation>
double exact_component( double t, int i )
{
    if constexpr ( std::is_base_of_v<time_mass_decay<double>, Equation> )
        return 1 / ( 1 + t );
    return std::exp( -( i + 1 ) * t );
}

template <class Equation>
void test_mass( const char *label )
{
    auto             operations = std::make_shared<operations_type>( Equation::size );
    auto             problem    = std::make_shared<rate_problem<Equation>>( operations );
    vector_wrap_type initial( *operations ), out( *operations ), sample( *operations );
    initial.start_use();
    out.start_use();
    sample.start_use();
    for ( const auto *method : dense_methods )
        for ( const double direction : { 1., -1. } )
        {
            double errors[3]{};
            for ( int level = 0; level < 3; ++level )
            {
                scfd::static_vec::vec<double, Equation::size> start;
                for ( int i = 0; i < Equation::size; ++i )
                    start[i] = exact_component<Equation>( .35, i );
                operations->set( start.d, *initial, Equation::size );
                const auto h    = .16 / ( 1 << level );
                auto       step = make_dense_step( operations, problem, h, method );
                step->set_time( .35 );
                step->set_target_time( .35 + direction * h );
                step->apply( *initial, *out );
                require( step->get_status() == single_step_status::converged, "Mass dense step failed" );
                const auto view = step->get_continuous_integration();
                for ( const auto theta : { .23, .67 } )
                {
                    const auto t = view.evaluate( theta, *sample );
                    for ( int i = 0; i < Equation::size; ++i )
                        errors[level] = std::max(
                            errors[level],
                            std::abs( operations->get_value_at_point( i, *sample ) - exact_component<Equation>( t, i ) )
                        );
                }
                step->finalize( adaptation_status::accepted, view.evaluate( 1, *sample ), *out );
            }
            const auto order = runge_kutta::make_butcher_table( method ).dense_order();
            std::cout << label << ' ' << method << " direction=" << direction
                      << " dense local order=" << std::log2( errors[1] / errors[2] ) << '\n';
            require(
                errors[0] / errors[1] > std::pow( 2., order + .2 ) &&
                    errors[1] / errors[2] > std::pow( 2., order + .2 ),
                "Mass dense interior order mismatch"
            );
        }
}

// x'=-x-z+x^2, 0=z-x^2. The singular mass has no ordinary inverse.
// The problem supplies z'=2*x*x' and reconstructs z=x^2 at sampled times.
struct nonlinear_constraint : coupled_dae
{
    __DEVICE_TAG__ void rate( double, const vector_type &x, vector_type &out ) const
    {
        const auto dx = -x( 0 ) - x( 1 ) + x( 0 ) * x( 0 );
        out( 0 )      = dx;
        out( 1 )      = 2 * x( 0 ) * dx;
    }
    __DEVICE_TAG__ static double algebraic( double, double x )
    {
        return x * x;
    }
};

// x'=-x-z+sin(t)*x, 0=z-sin(t)*x. A moving index-1 constraint.
// Consistent stage rates and reconstructed samples are separate responsibilities.
struct moving_constraint : moving_constraint_dae
{
    __DEVICE_TAG__ static double algebraic( double t, double x )
    {
        return std::sin( t ) * x;
    }
};

template <class Equation>
struct reconstructing_problem : rate_problem<Equation>
{
    using rate_problem<Equation>::rate_problem;
    bool refuse_reconstruction = false;
    struct mapping
    {
        double              time;
        vector_type         state;
        __DEVICE_TAG__ void operator()( std::ptrdiff_t ) const
        {
            state( 1 ) = Equation::algebraic( time, state( 0 ) );
        }
    };
    // Application-owned consistency operation, not a timestepper callback.
    bool reconstruct( double time, vector_type &state ) const
    {
        if ( refuse_reconstruction )
            return false;
        execute_small_mapping( mapping{ time, state } );
        return true;
    }
};

template <class Equation>
void test_constraint( const char *label )
{
    auto             operations = std::make_shared<operations_type>( 2 );
    auto             problem    = std::make_shared<reconstructing_problem<Equation>>( operations );
    vector_wrap_type initial( *operations ), out( *operations ), sample( *operations );
    initial.start_use();
    out.start_use();
    sample.start_use();
    double largest_raw_constraint = 0;
    // Stiff accuracy preserves the endpoint constraint; IM is deliberately excluded.
    for ( const auto *method : { "IE", "CN", "SDIRK2(1)2", "ESDIRK2(1)3", "SDIRK3(1)3", "ARK324L2SA" } )
        for ( const double direction : { 1., -1. } )
        {
            double errors[3]{};
            for ( int level = 0; level < 3; ++level )
            {
                const scfd::static_vec::vec<double, 2> start{
                    std::exp( -.35 ), Equation::algebraic( .35, std::exp( -.35 ) )
                };
                operations->set( start.d, *initial, 2 );
                const auto h    = .16 / ( 1 << level );
                auto       step = make_dense_step( operations, problem, h, method );
                step->set_time( .35 );
                step->set_target_time( .35 + direction * h );
                step->apply( *initial, *out );
                require( step->get_status() == single_step_status::converged, "DAE dense step failed" );
                const auto view = step->get_continuous_integration();
                for ( const auto theta : { .23, .67 } )
                {
                    const auto t           = view.evaluate( theta, *sample );
                    const auto x           = operations->get_value_at_point( 0, *sample );
                    largest_raw_constraint = std::max(
                        largest_raw_constraint,
                        std::abs( operations->get_value_at_point( 1, *sample ) - Equation::algebraic( t, x ) )
                    );
                    require( problem->reconstruct( t, *sample ), "DAE sample reconstruction failed" );
                    require(
                        std::abs( operations->get_value_at_point( 1, *sample ) - Equation::algebraic( t, x ) ) < 1e-14,
                        "Reconstructed dense sample violates its constraint"
                    );
                    errors[level] = std::max( errors[level], std::abs( x - std::exp( -t ) ) );
                    errors[level] = std::max(
                        errors[level],
                        std::abs(
                            operations->get_value_at_point( 1, *sample ) - Equation::algebraic( t, std::exp( -t ) )
                        )
                    );
                }
                const auto end = view.evaluate( 1, *sample );
                require(
                    std::abs(
                        operations->get_value_at_point( 1, *sample ) -
                        Equation::algebraic( end, operations->get_value_at_point( 0, *sample ) )
                    ) < 2e-9,
                    "Dense endpoint lost the solved DAE constraint"
                );
                step->finalize( adaptation_status::accepted, end, *out );
            }
            std::cout << label << ' ' << method << " direction=" << direction << " reconstructed errors=" << errors[0]
                      << " -> " << errors[1] << " -> " << errors[2] << '\n';
            require(
                errors[1] < .7 * errors[0] && errors[2] < .7 * errors[1] && errors[2] < 2e-3,
                "Reconstructed DAE samples did not converge under refinement"
            );
        }
    require(
        largest_raw_constraint > 1e-7, "DAE test did not distinguish raw interpolation from consistent reconstruction"
    );
}

using sampled_problem = reconstructing_problem<moving_constraint>;
using sampled_step    = dense_step_type<sampled_problem>;
struct constrained_manager
{
    std::shared_ptr<sampled_problem> problem;
    std::shared_ptr<sampled_step>    step;
    double                          *end    = nullptr;
    integration_status               status = integration_status::running;
    void                             set_time_interval( const double &, double &next )
    {
        end    = &next;
        status = integration_status::running;
    }
    bool apply( const vector_type &, vector_type &out )
    {
        *end   = step->get_continuous_integration().evaluate( .4, out );
        status = problem->reconstruct( *end, out ) ? integration_status::stopped_by_external_operation
                                                   : integration_status::external_operation_failure;
        return true;
    }
    integration_status get_status() const
    {
        return status;
    }
};

void test_constrained_commit()
{
    auto operations = std::make_shared<operations_type>( 2 );
    auto problem    = std::make_shared<sampled_problem>( operations );
    auto step       = make_dense_step( operations, problem, .1, "SDIRK3(1)3" );
    auto external   = std::make_shared<constrained_manager>( constrained_manager{ problem, step } );
    integration::time_integrator<operations_type, sampled_step, constrained_manager> integrator(
        operations, step, {}, external
    );
    vector_wrap_type initial( *operations ), out( *operations );
    initial.start_use();
    out.start_use();
    const scfd::static_vec::vec<double, 2> start{ std::exp( -.35 ), std::sin( .35 ) * std::exp( -.35 ) };
    operations->set( start.d, *initial, 2 );
    integrator.set_time_interval( .35, .75 );
    for ( int repeat = 0; repeat < 2; ++repeat )
    {
        integrator.apply( *initial, *out );
        const auto time = integrator.get_final_time();
        require(
            integrator.get_status() == integration_status::stopped_by_external_operation &&
                integrator.get_steps() == 1 && std::abs( time - .39 ) < 1e-14 &&
                std::abs(
                    operations->get_value_at_point( 1, *out ) -
                    std::sin( time ) * operations->get_value_at_point( 0, *out )
                ) < 1e-14,
            "Constrained external sample was not committed consistently"
        );
    }
    problem->refuse_reconstruction = true;
    integrator.apply( *initial, *out );
    require(
        integrator.get_status() == integration_status::external_operation_failure && integrator.get_steps() == 0 &&
            integrator.get_final_time() == .35,
        "Failed DAE reconstruction was committed"
    );
    for ( int i = 0; i < 2; ++i )
        require(
            operations->get_value_at_point( i, *out ) == start[i] &&
                operations->get_value_at_point( i, *initial ) == start[i],
            "DAE reconstruction failure changed committed state"
        );
    require_exception<std::logic_error>(
        [&] { step->get_continuous_integration(); }, "Failed reconstruction retained a pending dense view"
    );
}

int main()
try
{
    backend_type::init_device();
    test_mass<coupled_mass>( "Coupled mass" );
    test_mass<variable_mass>( "State-dependent mass" );
    test_mass<rate_scalar_equation<time_mass_decay<double>>>( "Time-dependent mass" );
    test_constraint<nonlinear_constraint>( "Nonlinear index-1 DAE" );
    test_constraint<moving_constraint>( "Moving index-1 DAE" );
    test_constrained_commit();
    std::cout << "Implicit dense mass, application-owned DAE reconstruction and failed sample rollback: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
