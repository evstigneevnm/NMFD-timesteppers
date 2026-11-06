#include "ode_benchmark_imex.h"

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;

// Each visual benchmark retains its original M*u'=f. Verify that the two
// problem-owned rates sum to M^{-1}f and that the shifted implicit Jacobian
// includes the correct split and mass derivatives at several states/times.
template <class Equation>
void check_split( const Equation &equation )
{
    using implicit_equation                    = benchmark_implicit_equation<Equation>;
    using state_type                           = typename benchmark_split<Equation>::state_type;
    auto                                   ops = std::make_shared<operations_type>( Equation::size );
    stage_problem<Equation>                full( ops, equation );
    stage_problem<implicit_equation>       implicit( ops, implicit_equation( equation ) );
    benchmark_explicit_partition<Equation> explicit_part{ equation };
    benchmark_implicit_partition<Equation> implicit_part( ops, equation );
    vector_wrap_type state( *ops ), full_rhs( *ops ), explicit_rate( *ops ), implicit_rate( *ops );
    state.start_use();
    full_rhs.start_use();
    explicit_rate.start_use();
    implicit_rate.start_use();
    for ( double time : { .25, .7, 1.3 } )
        for ( double value : { -.4, .8 } )
        {
            state_type point, rhs;
            for ( int j = 0; j < Equation::size; ++j )
                point[j] = value * ( j + 1 );
            ops->set( point.d, *state, Equation::size );
            ops->assign( *state, full.previous_state() );
            ops->assign_scalar( 0., full.known_rhs() );
            full.set_time( time );
            full.set_affine_shift( 0, 1 );
            full.apply( *state, *full_rhs );
            ops->get( *full_rhs, rhs.d, Equation::size );
            for ( int j = 0; j < Equation::size; ++j )
                rhs[j] = -rhs[j];
            const auto expected = benchmark_mass_solve( equation, time, point, rhs );
            explicit_part.set_time( time );
            implicit_part.set_time( time );
            require(
                explicit_part.evaluate_rate( *state, *explicit_rate ) &&
                    implicit_part.evaluate_rate( *state, *implicit_rate ),
                "Benchmark partition rate failed"
            );
            ops->add_lin_comb( 1., *implicit_rate, *explicit_rate );
            ops->get( *explicit_rate, rhs.d, Equation::size );
            for ( int j = 0; j < Equation::size; ++j )
                require(
                    std::abs( rhs[j] - expected[j] ) < 1e-11 * ( 1 + std::abs( expected[j] ) ),
                    "IMEX benchmark changed the unsplit equation or mass solve"
                );
            check_jacobian( ops, implicit, point, time );
        }
}

int main()
try
{
    backend_type::init_device();
    lorenz_equation lorenz;
    lorenz.epsilon = .05;
    lorenz.delta   = .2;
    check_split( lorenz );
    check_split( rossler_equation{} );
    check_split( vdp_equation{} );
    vdp_equation stiff_vdp;
    stiff_vdp.mu = 1000;
    check_split( stiff_vdp );
    check_split( coupled_mass{} );
    check_split( variable_mass{} );
    check_split( time_dependent_mass{} );
    check_split( scalar_time_equation<coefficient_growth<double>>{} );
    check_split( scalar_time_equation<stiff_tracking<double>>{ stiff_tracking<double>{ 10000 } } );
    std::cout << "Visual IMEX benchmark partitions, mass solves and shifted Jacobians: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
