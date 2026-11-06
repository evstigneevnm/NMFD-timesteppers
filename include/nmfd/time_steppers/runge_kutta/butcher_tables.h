#ifndef NMFD_TIME_STEPPERS_BUTCHER_TABLES_H
#define NMFD_TIME_STEPPERS_BUTCHER_TABLES_H

#include <string>
#include <nmfd/time_steppers/runge_kutta/butcher_table.h>
#include <nmfd/time_steppers/runge_kutta/dop853_coefficients.h>

namespace nmfd
{
namespace time_steppers
{
namespace runge_kutta
{
namespace detail
{
// Integrate a linear rate with the RK endpoint and the stiffly accurate final rate.
inline butcher_table::matrix_type quadratic_dense_coefficients( const butcher_table::vector_type &b )
{
    butcher_table::matrix_type coefficients;
    for ( const auto weight : b )
        coefficients.push_back( { 2 * weight, -weight } );
    coefficients.back()[0] -= 1;
    coefficients.back()[1] += 1;
    return coefficients;
}

// Cubic endpoint Hermite: reuse the final stage and append a rate at the initial state.
inline butcher_table::matrix_type hermite_dense_coefficients( const butcher_table::vector_type &b )
{
    butcher_table::matrix_type coefficients;
    for ( const auto weight : b )
        coefficients.push_back( { 0, 3 * weight, -2 * weight } );
    coefficients.back()[1] -= 1;
    coefficients.back()[2] += 1;
    coefficients.push_back( { 1, -2, 1 } );
    return coefficients;
}
}

// DIRK names use p(q)s: primary order p, embedded order q, total stages s.
inline butcher_table make_butcher_table( const std::string &name )
{
    if ( name == "EE" )
    {
        return { { { 0 } }, { 1 }, 1, {}, {}, 0, 2e-15L, { { 1 } }, 1 };
    }
    if ( name == "HE" )
    {
        return { { { 0, 0 }, { 1, 0 } }, { .5L, .5L }, 2, {}, { 1, 0 }, 1, 2e-15L, { { 1, -.5L }, { 0, .5L } }, 2 };
    }
    if ( name == "RK23" )
    {
        // Bogacki-Shampine 3(2), including the endpoint derivative and native cubic extension.
        return {
            { { 0, 0, 0, 0 }, { 1.L / 2, 0, 0, 0 }, { 0, 3.L / 4, 0, 0 }, { 2.L / 9, 1.L / 3, 4.L / 9, 0 } },
            { 2.L / 9, 1.L / 3, 4.L / 9, 0 },
            3,
            { 0, 1.L / 2, 3.L / 4, 1 },
            { 7.L / 24, 1.L / 4, 1.L / 3, 1.L / 8 },
            2,
            2e-15L,
            { { 1, -4.L / 3, 5.L / 9 }, { 0, 1, -2.L / 3 }, { 0, 4.L / 3, -8.L / 9 }, { 0, -1, 1 } },
            3
        };
    }
    if ( name == "RK33SSP" )
    {
        return {
            { { 0, 0, 0 }, { 1, 0, 0 }, { .25L, .25L, 0 } },
            { 1.L / 6, 1.L / 6, 2.L / 3 },
            3,
            {},
            { .291485418878409L, .291485418878409L, .417029162243181L },
            2
        };
    }
    if ( name == "RK43SSP" )
    {
        return {
            { { 0, 0, 0, 0 }, { .5L, 0, 0, 0 }, { .5L, .5L, 0, 0 }, { 1.L / 6, 1.L / 6, 1.L / 6, 0 } },
            { 1.L / 6, 1.L / 6, 1.L / 6, .5L },
            3,
            {},
            { 1.L / 3, 1.L / 3, 1.L / 3, 0 },
            2
        };
    }
    if ( name == "RK4" )
    {
        return {
            { { 0, 0, 0, 0 }, { .5L, 0, 0, 0 }, { 0, .5L, 0, 0 }, { 0, 0, 1, 0 } },
            { 1.L / 6, 1.L / 3, 1.L / 3, 1.L / 6 },
            4,
            { 0, .5L, .5L, 1 }
        };
    }
    if ( name == "RK45" )
    {
        return {
            { { 0, 0, 0, 0, 0, 0, 0 },
              { 1.L / 5, 0, 0, 0, 0, 0, 0 },
              { 3.L / 40, 9.L / 40, 0, 0, 0, 0, 0 },
              { 44.L / 45, -56.L / 15, 32.L / 9, 0, 0, 0, 0 },
              { 19372.L / 6561, -25360.L / 2187, 64448.L / 6561, -212.L / 729, 0, 0, 0 },
              { 9017.L / 3168, -355.L / 33, 46732.L / 5247, 49.L / 176, -5103.L / 18656, 0, 0 },
              { 35.L / 384, 0, 500.L / 1113, 125.L / 192, -2187.L / 6784, 11.L / 84, 0 } },
            { 35.L / 384, 0, 500.L / 1113, 125.L / 192, -2187.L / 6784, 11.L / 84, 0 },
            5,
            { 0, 1.L / 5, 3.L / 10, 4.L / 5, 8.L / 9, 1, 1 },
            { 5179.L / 57600, 0, 7571.L / 16695, 393.L / 640, -92097.L / 339200, 187.L / 2100, 1.L / 40 },
            4,
            // Shampine's quartic extension (1986); rational coefficients also used by SciPy RK45:
            // https://github.com/scipy/scipy/blob/v1.16.0/scipy/integrate/_ivp/rk.py
            2e-15L,
            { { 1, -8048581381.L / 2820520608, 8663915743.L / 2820520608, -12715105075.L / 11282082432 },
              { 0, 0, 0, 0 },
              { 0, 131558114200.L / 32700410799, -68118460800.L / 10900136933, 87487479700.L / 32700410799 },
              { 0, -1754552775.L / 470086768, 14199869525.L / 1410260304, -10690763975.L / 1880347072 },
              { 0, 127303824393.L / 49829197408, -318862633887.L / 49829197408, 701980252875.L / 199316789632 },
              { 0, -282668133.L / 205662961, 2019193451.L / 616988883, -1453857185.L / 822651844 },
              { 0, 40617522.L / 29380423, -110615467.L / 29380423, 69997945.L / 29380423 } },
            4
        };
    }
    if ( name == "DOP853" )
    {
        return detail::make_dop853_table();
    }
    if ( name == "RK64SSP" )
    {
        // Published decimals: Fekete et al., JCAM 412 (2022) 114325, Table 4.
        // No long-double accuracy for this truncated table.
        return {
            { { 0, 0, 0, 0, 0, 0 },
              { .3552975516919L, 0, 0, 0, 0, 0 },
              { .2704882223931L, .3317866983600L, 0, 0, 0, 0 },
              { .1223997401356L, .1501381660925L, .1972127376054L, 0, 0, 0 },
              { .0763425067155L, .0936433683640L, .1230044665810L, .2718245927242L, 0, 0 },
              { .0763425067155L, .0936433683640L, .1230044665810L, .2718245927242L, .4358156542577L, 0 } },
            { .1522491819555L, .1867521364225L, .1555370561501L, .1348455085546L, .2161974490441L, .1544186678729L },
            4,
            {},
            { .1210663237182L, .2308844004550L, .0853424972752L, .3450614904457L, .0305351538213L, .1871101342844L },
            3,
            2e-12L
        };
    }
    if ( name == "IE" )
    {
        return { { { 1 } }, { 1 }, 1, {}, {}, 0, 2e-15L, { { 1 } }, 1 };
    }
    if ( name == "IM" )
    {
        // The extra initial rate makes the quadratic extension second-order.
        return { { { .5L } }, { 1 }, 2, {}, {}, 0, 2e-15L, { { 0, 1 }, { 1, -1 } }, 2, { { 0, 0 } }, { 0 } };
    }
    if ( name == "CN" )
    {
        return {
            { { 0, 0 }, { .5L, .5L } },
            { .5L, .5L },
            2,
            {},
            {},
            0,
            2e-15L,
            detail::quadratic_dense_coefficients( { .5L, .5L } ),
            2
        };
    }
    const auto gamma2 = 1.L - std::sqrt( 2.L ) / 2;
    if ( name == "SDIRK2(1)2" )
    {
        return {
            { { gamma2, 0 }, { 1 - gamma2, gamma2 } },
            { 1 - gamma2, gamma2 },
            2,
            {},
            { .5L, .5L },
            1,
            2e-15L,
            detail::quadratic_dense_coefficients( { 1 - gamma2, gamma2 } ),
            2
        };
    }
    if ( name == "ESDIRK2(1)3" )
    {
        const auto g      = gamma2;
        const auto b2     = g * ( -2 + 7 * g - 5 * g * g + 4 * g * g * g ) / ( 2 * ( 2 * g - 1 ) );
        const auto b3     = -2 * g * g * ( 1 - g + g * g ) / ( 2 * g - 1 );
        const auto middle = ( 1 - 2 * g ) / ( 4 * g );
        return {
            { { 0, 0, 0 }, { g, g, 0 }, { 1 - middle - g, middle, g } },
            { 1 - middle - g, middle, g },
            2,
            {},
            { 1 - b2 - b3, b2, b3 },
            1,
            2e-15L,
            detail::quadratic_dense_coefficients( { 1 - middle - g, middle, g } ),
            2
        };
    }
    if ( name == "SDIRK3(1)3" )
    {
        const auto g     = .43586652150845899941601945119356L;
        const auto alpha = 1 - 4 * g + 2 * g * g;
        const auto beta  = -1 + 6 * g - 9 * g * g + 3 * g * g * g;
        const auto b2    = -3 * alpha * alpha / ( 4 * beta );
        const auto c2    = ( 2 - 9 * g + 6 * g * g ) / ( 3 * alpha );
        // Alexander's third-order primary method with the inherited first-order estimator.
        return {
            { { g, 0, 0 }, { c2 - g, g, 0 }, { 1 - b2 - g, b2, g } },
            { 1 - b2 - g, b2, g },
            3,
            {},
            { .5L - b2, b2, .5L },
            1,
            2e-15L,
            detail::hermite_dense_coefficients( { 1 - b2 - g, b2, g } ),
            3,
            { { 0, 0, 0, 0 } },
            { 0 }
        };
    }
    if ( name == "ARK324L2SA" )
    {
        // Implicit ARK3(2)4L[2]SA: Kennedy-Carpenter, four-stage ESDIRK 3(2).
        // Coefficients: LLNL/sundials src/arkode/arkode_butcher_dirk.def,
        // ARKODE_ARK324L2SA_DIRK_4_2_3 (high-precision representation).
        const auto                       g = .4358665215084589994160194511935568425293L;
        const butcher_table::vector_type b{
            .1876410243467238251612921441668043913795L, -.5952974735769549480478230275858851737782L,
            .9717899277217721234705114322255239398694L, g
        };
        auto dense = detail::hermite_dense_coefficients( b );
        // The explicit first stage already supplies the initial Hermite derivative.
        for ( std::size_t j = 0; j < dense.back().size(); ++j )
            dense.front()[j] += dense.back()[j];
        dense.pop_back();
        return {
            { { 0, 0, 0, 0 },
              { g, g, 0, 0 },
              { .2576482460664272457999960162840797092643L, -.09351476757488624521601546747763655179361L, g, 0 },
              b },
            b,
            3,
            { 0, 2 * g, .6L, 1 },
            { .2147402862233891404862383406484193714659L, -.4851622638849390928209050808398155895845L,
              .86872500252038755116621237682951240796L, .4016969751411624011684543633618838101586L },
            2,
            2e-15L,
            std::move( dense ),
            3
        };
    }
    throw std::invalid_argument( "Unknown RK scheme: " + name );
}

struct imex_butcher_table
{
    enum class dense_output_type
    {
        none,
        endpoint_hermite
    };
    struct dense_weights_type
    {
        long double initial_state, final_state, initial_rate, final_rate;
    };
    butcher_table explicit_table, implicit_table;
    unsigned int  order;

    // A combined embedding is opt-in; independent partition embeddings are not sufficient.
    imex_butcher_table(
        butcher_table e, butcher_table i, unsigned int p, unsigned int embedded_order = 0,
        dense_output_type dense = dense_output_type::none
    )
        : explicit_table( std::move( e ) ), implicit_table( std::move( i ) ), order( p ),
          embedded_order_( embedded_order ), dense_type_( dense )
    {
        if ( explicit_table.type() != butcher_table::scheme_type::explicit_rk ||
             implicit_table.type() == butcher_table::scheme_type::irk ||
             explicit_table.size() != implicit_table.size() || p == 0 || p > explicit_table.order() ||
             p > implicit_table.order() )
        {
            throw std::invalid_argument( "Incompatible IMEX tables" );
        }
        if ( embedded_order && ( embedded_order >= p || !explicit_table.is_embedded() ||
                                 !implicit_table.is_embedded() || embedded_order != explicit_table.embedded_order() ||
                                 embedded_order != implicit_table.embedded_order() ) )
            throw std::invalid_argument( "Incompatible combined IMEX embedding" );
        for ( std::size_t j = 0; j < explicit_table.size(); ++j )
        {
            if ( std::abs( explicit_table.c( j ) - implicit_table.c( j ) ) >
                 std::max( explicit_table.coefficient_tolerance(), implicit_table.coefficient_tolerance() ) )
            {
                throw std::invalid_argument( "IMEX stage times differ" );
            }
        }
        if ( dense != dense_output_type::none )
        {
            if ( dense != dense_output_type::endpoint_hermite || p < 3 || explicit_table.size() < 2 ||
                 explicit_table.c( 0 ) != 0 || implicit_table.c( 0 ) != 0 )
                throw std::invalid_argument( "Incompatible combined IMEX dense extension" );
            for ( std::size_t j = 0; j < explicit_table.size(); ++j )
                if ( explicit_table.a( 0, j ) != 0 || implicit_table.a( 0, j ) != 0 )
                    throw std::invalid_argument( "IMEX Hermite extension requires an initial-state stage" );
        }
    }

    bool is_embedded() const
    {
        return embedded_order_ != 0;
    }
    unsigned int embedded_order() const
    {
        return embedded_order_;
    }
    unsigned int error_order() const
    {
        return is_embedded() ? embedded_order_ + 1 : 0;
    }
    dense_output_type dense_type() const
    {
        return dense_type_;
    }
    bool has_dense_output() const
    {
        return dense_type_ != dense_output_type::none;
    }
    unsigned int dense_order() const
    {
        return has_dense_output() ? 3 : 0;
    }
    dense_weights_type dense_weights( long double theta ) const
    {
        if ( !has_dense_output() )
            throw std::logic_error( "No combined IMEX continuous extension" );
        if ( !std::isfinite( theta ) || theta < 0 || theta > 1 )
            throw std::invalid_argument( "Dense-output theta must be in [0,1]" );
        const auto complement = 1 - theta;
        return {
            ( 1 + 2 * theta ) * complement * complement, theta * theta * ( 3 - 2 * theta ),
            theta * complement * complement, theta * theta * ( theta - 1 )
        };
    }

private:
    unsigned int      embedded_order_;
    dense_output_type dense_type_;
};

// ARS names use the published (implicit stages, explicit stages, order) triplet,
// not the dimensions of the padded tableaux stored here.
inline imex_butcher_table make_imex_butcher_table( const std::string &name )
{
    if ( name == "IMEX_ARK324L2SA" )
    {
        // Kennedy-Carpenter ARK3(2)4L[2]SA; LLNL/sundials
        // src/arkode/arkode_butcher_erk.def, ARKODE_ARK324L2SA_ERK_4_2_3.
        auto                       implicit = make_butcher_table( "ARK324L2SA" );
        butcher_table::vector_type b, embedded, c;
        for ( std::size_t j = 0; j < implicit.size(); ++j )
        {
            b.push_back( implicit.b( j ) );
            embedded.push_back( implicit.embedded_b( j ) );
            c.push_back( implicit.c( j ) );
        }
        butcher_table explicit_table(
            { { 0, 0, 0, 0 },
              { .8717330430169179988320389023871136850586L, 0, 0, 0 },
              { .52758901197630041156180797140291790433L, .07241098802369958843819202859708209566999L, 0, 0 },
              { .3990960076760701320627260736092142797856L, -.437557654613519443722846363831022571942L,
                1.038461646937449311660120290221808292156L, 0 } },
            std::move( b ), 3, std::move( c ), std::move( embedded ), 2
        );
        return {
            std::move( explicit_table ), std::move( implicit ), 3, 2,
            imex_butcher_table::dense_output_type::endpoint_hermite
        };
    }
    if ( name == "IMEX_EULER" )
    {
        return { { { { 0, 0 }, { 1, 0 } }, { 1, 0 }, 1 }, { { { 0, 0 }, { 0, 1 } }, { 0, 1 }, 1 }, 1 };
    }
    if ( name == "IMEX_HEUN_TR2" )
    {
        return { { { { 0, 0 }, { 1, 0 } }, { .5L, .5L }, 2 }, { { { 0, 0 }, { .5L, .5L } }, { .5L, .5L }, 2 }, 2 };
    }
    if ( name == "IMEX_ARS233" )
    {
        const auto g = ( 3 + std::sqrt( 3.L ) ) / 6;
        return {
            { { { 0, 0, 0 }, { g, 0, 0 }, { g - 1, 2 * ( 1 - g ), 0 } }, { 0, .5L, .5L }, 3 },
            { { { 0, 0, 0 }, { 0, g, 0 }, { 0, 1 - 2 * g, g } }, { 0, .5L, .5L }, 3 },
            3
        };
    }
    if ( name == "IMEX_ARS222" )
    {
        const auto g = 1.L - std::sqrt( 2.L ) / 2, k = 1 - 1 / ( 2 * g );
        return {
            { { { 0, 0, 0 }, { g, 0, 0 }, { k, 1 - k, 0 } }, { k, 1 - k, 0 }, 2 },
            { { { 0, 0, 0 }, { 0, g, 0 }, { 0, 1 - g, g } }, { 0, 1 - g, g }, 2 },
            2
        };
    }
    throw std::invalid_argument( "Unknown IMEX scheme: " + name );
}
}
}
}
#endif
