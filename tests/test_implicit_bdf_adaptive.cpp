#include "implicit_bdf_adaptive_test.h"

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;
namespace model = nmfd::time_steppers::tests::bdf_test;

// Exact polynomial histories distinguish the LTE constant from its exponent.
// For u=t^2 (BDF1) and u=t^3 (BDF2), u'=2t and u'=3t^2 respectively;
// both have identity mass and no stiffness. No nonlinear reference is needed.
void test_polynomial_estimates()
{
    auto             ops = std::make_shared<operations_type>( 1 );
    vector_wrap_type old( *ops ), current( *ops ), candidate( *ops ), error( *ops );
    old.start_use();
    current.start_use();
    candidate.start_use();
    error.start_use();
    for ( const double direction : { -1., 1. } )
        for ( const double ratio : { .2, .5, 1., 2. } )
        {
            const double previous_h = direction * .1, h = previous_h * ratio, older_h = previous_h * .7;
            for ( unsigned int order : { 1u, 2u } )
            {
                const auto polynomial = [order]( double t ) { return order == 1 ? t * t : t * t * t; };
                bdf::divided_difference_history<operations_type> history( *ops );
                history.initialize( true );
                ops->assign_scalar( polynomial( -previous_h - older_h ), *old );
                ops->assign_scalar( polynomial( -previous_h ), *current );
                history.commit( *old, *current, older_h, true );
                ops->assign( *current, *old );
                ops->assign_scalar( 0., *current );
                history.commit( *old, *current, previous_h, false );
                const auto c         = bdf::make_coefficients( order, h, previous_h );
                const auto rhs       = order == 1 ? 2 * h : 3 * h * h;
                double     reference = 0;
                if ( order == 2 )
                {
                    history.reference( c.history_weight, *current, *candidate );
                    reference = ops->get_value_at_point( 0, *candidate );
                }
                const auto value = reference + rhs / c.alpha;
                ops->assign_scalar( value, *candidate );
                history.estimate( order, h, *current, *candidate, *error );
                require(
                    std::abs( ops->get_value_at_point( 0, *error ) - ( value - polynomial( h ) ) ) < 2e-14,
                    "BDF polynomial LTE constant or variable-grid scaling mismatch"
                );
                require(
                    history.usable( h, .2, 2. ) && !history.usable( -h, .2, 2. ), "BDF history accepted a reversed grid"
                );
            }
        }
}

template <class Controller>
void test_controller( const char *name )
{
    auto ops       = std::make_shared<operations_type>( 1 );
    using equation = scalar_time_equation<model::forced_decay>;
    for ( const char *method : { "BDF1", "BDF2" } )
        for ( const double direction : { 1., -1. } )
        {
            const double start = .25, end = start + direction * .5;
            double       errors[2]{};
            for ( int level = 0; level < 2; ++level )
                errors[level] = model::adaptive_trajectory<Controller>(
                    equation{}, { model::forced_decay::exact( start ) }, { model::forced_decay::exact( end ) }, start,
                    end, method, name, level ? 2e-6 : 2e-3, ops
                );
            // Local O(h^(q+1)) control produces global O(h^q) accuracy,
            // not an endpoint error bounded by the local tolerance itself.
            const double limit = std::string( method ) == "BDF1" ? .002 : .0003;
            require(
                errors[1] < .15 * errors[0] && errors[1] < limit,
                "Adaptive BDF tolerance refinement did not improve the solution"
            );
        }
}

int main()
try
{
    backend_type::init_device();
    test_polynomial_estimates();
    test_controller<integration::time_step_adaptation_matlab<operations_type>>( "MATLAB" );
    test_controller<integration::time_step_adaptation_scipy<operations_type>>( "SciPy" );
    for ( const char *name : { "I", "H211", "PC", "PID", "H312", "PPID", "H321" } )
        test_controller<integration::time_step_adaptation_filter<operations_type>>( name );
    std::cout << "Adaptive BDF coefficients, polynomial LTE, all controllers, forward/backward and restart: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
