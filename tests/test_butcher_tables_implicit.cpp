#include "table_order_checks.h"

using namespace nmfd::time_steppers::runge_kutta;
using table_tests::require;

int main()
try
{
    const struct
    {
        const char  *name;
        std::size_t  stages;
        unsigned int order, embedded_order;
    } methods[] = { { "IE", 1, 1, 0 },         { "IM", 1, 2, 0 },          { "CN", 2, 2, 0 },
                    { "SDIRK2(1)2", 2, 2, 1 }, { "ESDIRK2(1)3", 3, 2, 1 }, { "SDIRK3(1)3", 3, 3, 1 },
                    { "ARK324L2SA", 4, 3, 2 } };

    for ( const auto &method : methods )
    {
        const auto *name  = method.name;
        const auto  table = make_butcher_table( name );
        require(
            table.size() == method.stages && table.order() == method.order &&
                table.embedded_order() == method.embedded_order,
            std::string( name ) + ": name/metadata mismatch"
        );
        require( table.type() != butcher_table::scheme_type::explicit_rk, name );
        table_tests::check_order( table, name );
        require( table.dense_order() == method.order, "Missing implicit continuous extension" );
        table_tests::check_order( table, name, false, true );
        if ( table.is_embedded() )
        {
            table_tests::check_order( table, name, true );
        }
    }
    require( make_butcher_table( "CN" ).type() == butcher_table::scheme_type::dirk, "CN classification" );
    require( make_butcher_table( "SDIRK3(1)3" ).error_order() == 2, "SDIRK3(1)3 error scales as h^2" );
    require( make_butcher_table( "ARK324L2SA" ).error_order() == 3, "ARK324L2SA error scales as h^3" );
    require(
        make_butcher_table( "ARK324L2SA" ).dense_outout_stage_count() == 4,
        "ARK324L2SA must reuse its initial rate for dense output"
    );

    const struct
    {
        const char  *name;
        std::size_t  stored_stages;
        unsigned int order, embedded_order;
    } pairs[] = {
        { "IMEX_EULER", 2, 1, 0 },
        { "IMEX_HEUN_TR2", 2, 2, 0 },
        { "IMEX_ARS233", 3, 3, 0 },
        { "IMEX_ARS222", 3, 2, 0 },
        { "IMEX_ARK324L2SA", 4, 3, 2 }
    };

    for ( const auto &method : pairs )
    {
        const auto *name = method.name;
        const auto  pair = make_imex_butcher_table( name );
        const auto &e    = pair.explicit_table;
        const auto &i    = pair.implicit_table;
        require(
            pair.order == method.order && e.size() == method.stored_stages && i.size() == method.stored_stages &&
                pair.embedded_order() == method.embedded_order &&
                pair.error_order() == ( method.embedded_order ? method.embedded_order + 1 : 0 ),
            std::string( name ) + ": name/metadata mismatch"
        );
        table_tests::check_order( e, std::string( name ) + " explicit" );
        table_tests::check_order( i, std::string( name ) + " implicit" );
        // Both primary and embedded weights must satisfy the colored-tree conditions.
        for ( const bool embedded : { false, true } )
        {
            if ( embedded && !pair.is_embedded() )
                continue;
            if ( embedded )
            {
                table_tests::check_order( e, std::string( name ) + " explicit embedding", true );
                table_tests::check_order( i, std::string( name ) + " implicit embedding", true );
            }
            const auto order  = embedded ? pair.embedded_order() : pair.order;
            const auto weight = [embedded]( const butcher_table &table, std::size_t j ) {
                return embedded ? table.embedded_b( j ) : table.b( j );
            };
            for ( const auto *root : { &e, &i } )
            {
                for ( const auto *child : { &e, &i } )
                {
                    long double bc = 0, bcc = 0;
                    for ( std::size_t j = 0; j < e.size(); ++j )
                    {
                        bc += weight( *root, j ) * child->c( j );
                        bcc += weight( *root, j ) * child->c( j ) * child->c( j );
                    }
                    if ( order >= 2 )
                    {
                        require( std::abs( bc - .5L ) < 1e-14L, "IMEX coupling order two" );
                    }
                    if ( order >= 3 )
                    {
                        require( std::abs( bcc - 1.L / 3 ) < 1e-14L, "IMEX bush order three" );
                        for ( const auto *grandchild : { &e, &i } )
                        {
                            long double bac = 0;
                            for ( std::size_t j = 0; j < e.size(); ++j )
                            {
                                for ( std::size_t k = 0; k < e.size(); ++k )
                                {
                                    bac += weight( *root, j ) * child->a( j, k ) * grandchild->c( k );
                                }
                            }
                            require( std::abs( bac - 1.L / 6 ) < 1e-14L, "IMEX chain order three" );
                        }
                    }
                }
            }
        }
    }
    const auto               embedded_pair = make_imex_butcher_table( "IMEX_ARK324L2SA" );
    const imex_butcher_table unpaired( embedded_pair.explicit_table, embedded_pair.implicit_table, 3 );
    require(
        !unpaired.is_embedded() && unpaired.error_order() == 0,
        "Independent partition embeddings silently enabled an IMEX estimator"
    );
    for ( const unsigned int order : { 1u, 3u } )
    {
        bool refused = false;
        try
        {
            imex_butcher_table bad( embedded_pair.explicit_table, embedded_pair.implicit_table, 3, order );
        }
        catch ( const std::invalid_argument & )
        {
            refused = true;
        }
        require( refused, "Invalid combined embedding order was accepted" );
    }
    bool missing_embedding = false;
    try
    {
        const auto         fixed = make_imex_butcher_table( "IMEX_HEUN_TR2" );
        imex_butcher_table bad( fixed.explicit_table, fixed.implicit_table, 2, 1 );
    }
    catch ( const std::invalid_argument & )
    {
        missing_embedding = true;
    }
    require( missing_embedding, "IMEX accepted missing partition embeddings" );
    bool rejected = false;
    try
    {
        imex_butcher_table bad( make_butcher_table( "HE" ), make_butcher_table( "SDIRK2(1)2" ), 2 );
    }
    catch ( const std::invalid_argument & )
    {
        rejected = true;
    }
    require( rejected, "Mismatched IMEX stage times" );
    std::cout << "Implicit/IMEX tables: PASS (no implicit stepper in this test)\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
