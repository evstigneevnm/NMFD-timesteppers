#include "implicit_bdf_test.h"
#include <array>

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;
using history_type = bdf::divided_difference_history<operations_type>;

// Polynomial ODEs u'=p'(t), u=p(t), isolate exact differentiation and the
// leading local-error constant on nonuniform grids. All masses are identity.
template <class Function>
void seed(
    history_type &history, operations_type &ops, vector_type &old, vector_type &current, unsigned int order,
    const std::array<double, 5> &steps, Function value
)
{
    double t = 0;
    for ( unsigned int i = 0; i < order; ++i )
        t -= steps[i];
    ops.assign_scalar( value( t ), old );
    for ( unsigned int i = order; i > 0; --i )
    {
        t = i == 1 ? 0 : t + steps[i - 1];
        ops.assign_scalar( value( t ), current );
        history.commit( old, current, steps[i - 1], i == order );
        ops.assign( current, old );
    }
}

void test_polynomials()
{
    auto             ops = std::make_shared<operations_type>( 1 );
    vector_wrap_type old( *ops ), current( *ops ), work( *ops ), error( *ops );
    old.start_use();
    current.start_use();
    work.start_use();
    error.start_use();
    const double harmonic[] = { 0, 1, 1.5, 11. / 6, 25. / 12, 137. / 60 };
    for ( unsigned int q = 1; q <= 5; ++q )
        for ( const double sign : { -1., 1. } )
            for ( const double ratio : { .5, 1., 1.2 } )
            {
                const double          h = sign * .08 * ratio;
                std::array<double, 5> steps{ sign * .08, sign * .075, sign * .09, sign * .07, sign * .085 };
                history_type          history( *ops );
                history.initialize( q );
                for ( unsigned int degree = 0; degree <= q + 1; ++degree )
                {
                    auto value = [degree]( double t ) { return std::pow( t, int( degree ) ); };
                    seed( history, *ops, *old, *current, q, steps, value );
                    const auto c = history.coefficients( q, h );
                    history.reference( c, *current, *work );
                    const double reference  = ops->get_value_at_point( 0, *work );
                    const double derivative = degree ? degree * std::pow( h, int( degree ) - 1 ) : 0;
                    const double candidate  = reference + derivative / c.alpha;
                    if ( degree <= q )
                    {
                        require(
                            std::abs( candidate - value( h ) ) < 2e-13, "BDF polynomial differentiation mismatch"
                        );
                        history.predict( c, *current, *work );
                        require(
                            std::abs( ops->get_value_at_point( 0, *work ) - value( h ) ) < 2e-13,
                            "BDF polynomial predictor mismatch"
                        );
                    }
                    else
                    {
                        ops->assign_scalar( candidate, *work );
                        history.estimate( q, h, *current, *work, *error );
                        require(
                            std::abs( ops->get_value_at_point( 0, *error ) - ( candidate - value( h ) ) ) < 2e-13,
                            "BDF variable-grid LTE constant mismatch"
                        );
                    }
                }
                steps.fill( h );
                require(
                    std::abs( bdf::make_formula( q, h, steps ).alpha * h - harmonic[q] ) < 2e-14,
                    "BDF constant-grid leading coefficient mismatch"
                );
            }
}

// u'=-u, u(0)=1. Exact negative-time history isolates formal BDF order from
// startup error; this is not a claim that a backward-Euler starter has order 5.
void test_formal_orders()
{
    auto             ops = std::make_shared<operations_type>( 1 );
    vector_wrap_type old( *ops ), current( *ops ), work( *ops );
    old.start_use();
    current.start_use();
    work.start_use();
    for ( unsigned int q = 3; q <= 5; ++q )
        for ( const double sign : { -1., 1. } )
        {
            double errors[2]{};
            for ( int level = 0; level < 2; ++level )
            {
                const int             count = 25 * ( level + 1 );
                const double          h     = sign / count;
                std::array<double, 5> steps;
                steps.fill( h );
                history_type history( *ops );
                history.initialize( q );
                seed( history, *ops, *old, *current, q, steps, []( double t ) { return std::exp( -t ); } );
                for ( int i = 0; i < count; ++i )
                {
                    const auto c = history.coefficients( q, h );
                    history.reference( c, *current, *work );
                    ops->scale( c.alpha / ( c.alpha + 1 ), *work );
                    history.commit( *current, *work, h, false );
                    ops->assign( *work, *current );
                }
                errors[level] = std::abs( ops->get_value_at_point( 0, *current ) - std::exp( -sign ) );
            }
            const auto observed = std::log2( errors[0] / errors[1] );
            std::cout << "BDF" << q << " direction=" << sign << " observed order=" << observed << '\n';
            require( observed > q - .25 && observed < q + .25, "BDF3-5 formal order mismatch" );
        }
}

int main()
try
{
    backend_type::init_device();
    test_polynomials();
    test_formal_orders();
    std::cout << "BDF1-5 nonuniform coefficients, predictors, LTE constants and isolated formal orders: PASS\n";
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
