#include "implicit_rosenbrock_mass_problem.h"

using namespace rosenbrock_tests;

// M*u'=M*q(u) has exactly the original q trajectory for every nonsingular M.
// Coupled decay starts from (1,0); independent logistic components start from
// (1/4,2/5). Noncommuting approximate Jacobians test the W-only guarantees.
struct initial_values
{
    vector_type         out;
    bool                logistic;
    __DEVICE_TAG__ void operator()( std::ptrdiff_t i ) const
    {
        out( i ) = logistic ? ( i % 2 ? .4 : .25 ) : ( i % 2 ? 0 : 1 );
    }
};
double exact_value( bool logistic, unsigned int i, double t )
{
    if ( logistic )
        return 1 / ( 1 + ( i ? 1.5 : 3 ) * std::exp( -t ) );
    return ( std::exp( -t ) + ( i ? -1 : 1 ) * std::exp( -3 * t ) ) / 2;
}

template <class Model, bool Dense>
std::pair<double, double> trajectory( const char *method, mass_matrix mass, Model model, bool logistic, double h )
{
    auto             ops       = std::make_shared<operations_type>( 2 );
    auto             p         = std::make_shared<mass_problem<Model>>( ops, model, mass );
    auto             reference = std::make_shared<problem<Model>>( ops, model );
    vector_wrap_type in( *ops ), native( *ops ), ordinary( *ops ), sample( *ops ), difference( *ops );
    for ( auto *v : { &in, &native, &ordinary, &sample, &difference } )
        v->start_use();
    operations_type::for_each_type{}( initial_values{ *in, logistic }, 2 );
    ops->assign( *in, *native );
    ops->assign( *in, *ordinary );
    auto make_step = [&]( auto problem, bool mass_form ) {
        using P = typename decltype( problem )::element_type;
        using L = direct_solver<P>;
        using S = rosenbrock::time_step<operations_type, P, adaptation_type, L, Dense>;
        typename S::params sp;
        sp.method    = method;
        sp.mass_form = mass_form;
        return std::make_shared<S>(
            ops, problem, std::make_shared<adaptation_type>( adaptation_type::params{ h } ), std::make_shared<L>( ops ),
            sp
        );
    };
    auto               step        = make_step( p, true );
    auto               baseline    = make_step( reference, false );
    double             dense_error = 0;
    const unsigned int count       = std::lround( 1 / h );
    for ( unsigned int n = 0; n < count; ++n )
    {
        const double t = n * h;
        step->set_time( t );
        baseline->set_time( t );
        step->set_target_time( 1 );
        baseline->set_target_time( 1 );
        ops->assign( *native, *in );
        step->apply( *in, *native );
        ops->assign( *ordinary, *in );
        baseline->apply( *in, *ordinary );
        check(
            step->get_status() == single_step_status::converged &&
                baseline->get_status() == single_step_status::converged,
            "Mass-form fixed step failed"
        );
        ops->assign_lin_comb( 1., *native, -1., *ordinary, *difference );
        check( ops->norm( *difference ) < 2e-11, "Mass-form stage disagrees with mass-inverted reference" );
        if constexpr ( Dense )
        {
            auto view = step->get_continuous_integration();
            auto ref  = baseline->get_continuous_integration();
            for ( double theta : { 0., .25, .5, .75, 1. } )
            {
                // Check interior accuracy every step; the full sample grid at
                // both ends avoids repeating identical endpoint/view checks.
                if ( theta != .5 && n != 0 && n + 1 != count )
                    continue;
                const double time = view.evaluate( theta, *sample );
                ref.evaluate( theta, *difference );
                ops->add_lin_comb( -1., *sample, *difference );
                check( ops->norm( *difference ) < 3e-11, "Mass-form dense sample disagrees with rate-form reference" );
                for ( unsigned int i = 0; i < 2; ++i )
                    dense_error = std::max(
                        dense_error,
                        std::abs( ops->get_value_at_point( i, *sample ) - exact_value( logistic, i, time ) )
                    );
            }
        }
        const double endpoint = step->get_dt() == 1 - t ? 1 : t + step->get_dt();
        step->finalize( adaptation_status::accepted, endpoint, *native );
        baseline->finalize( adaptation_status::accepted, endpoint, *ordinary );
    }
    double error = 0;
    for ( unsigned int i = 0; i < 2; ++i )
        error = std::max( error, std::abs( ops->get_value_at_point( i, *native ) - exact_value( logistic, i, 1 ) ) );
    const unsigned int extra = Dense ? ( step->table().stage_count() == 4   ? 2
                                         : step->table().stage_count() == 3 ? 1
                                                                            : 0 )
                                     : 0;
    check(
        p->linearizations == count && p->residuals == count * ( step->table().stage_count() + extra ),
        "Mass-form callback counts or frozen linearization changed"
    );
    return { error, dense_error };
}

template <class Model>
void orders( const char *method, mass_matrix mass, Model model, bool logistic )
{
    const double p = rosenbrock::method_table( method ).order();
    const auto   a = trajectory<Model, true>( method, mass, model, logistic, .04 );
    const auto   b = trajectory<Model, true>( method, mass, model, logistic, .02 );
    const auto   c = trajectory<Model, true>( method, mass, model, logistic, .01 );
    check(
        a.first / b.first > std::pow( 2, p - .35 ) && b.first / c.first > std::pow( 2, p - .35 ),
        "Mass-form primary order was lost"
    );
    check(
        a.second / b.second > std::pow( 2, p - .4 ) && b.second / c.second > std::pow( 2, p - .4 ),
        "Mass-form continuous order was lost"
    );
    const auto plain = trajectory<Model, false>( method, mass, model, logistic, .04 );
    check( std::abs( plain.first - a.first ) < 1e-13, "Dense output changed the mass-form primary endpoint" );
}

// Small/large signed steps and mass scales must use the same balanced equation
// as the rate-form reference, rather than silently treating M as identity.
void scaled_steps( const char *method )
{
    using P              = mass_problem<coupled_decay>;
    using Q              = problem<coupled_decay>;
    using S              = rosenbrock::time_step<operations_type, P, adaptation_type, direct_solver<P>, true>;
    using R              = rosenbrock::time_step<operations_type, Q, adaptation_type, direct_solver<Q>, true>;
    auto             ops = std::make_shared<operations_type>( 2 );
    vector_wrap_type in( *ops ), out( *ops ), reference( *ops ), sample( *ops ), other( *ops );
    for ( auto *v : { &in, &out, &reference, &sample, &other } )
        v->start_use();
    operations_type::for_each_type{}( initial_values{ *in, false }, 2 );
    for ( double scale : { 1e-30, 1., 1e30 } )
        for ( double reference_time : { 1e-6, 1., 1e6 } )
            for ( double h : { -1e-3, -1e-8, 1e-12, .01, 1., 1000. } )
            {
                typename S::params sp;
                sp.method         = method;
                sp.mass_form      = true;
                sp.reference_time = reference_time;
                typename R::params rp;
                rp.method         = method;
                rp.reference_time = reference_time;
                auto p            = std::make_shared<P>(
                    ops, coupled_decay{}, mass_matrix{ 2 * scale, .7 * scale, -.3 * scale, 1.5 * scale }
                );
                auto q = std::make_shared<Q>( ops );
                S    step(
                    ops, p, std::make_shared<adaptation_type>( adaptation_type::params{ std::abs( h ) } ),
                    std::make_shared<direct_solver<P>>( ops ), sp
                );
                R baseline(
                    ops, q, std::make_shared<adaptation_type>( adaptation_type::params{ std::abs( h ) } ),
                    std::make_shared<direct_solver<Q>>( ops ), rp
                );
                step.set_target_time( h );
                baseline.set_target_time( h );
                step.apply( *in, *out );
                baseline.apply( *in, *reference );
                check(
                    step.get_status() == single_step_status::converged &&
                        baseline.get_status() == single_step_status::converged,
                    "Scaled/signed mass-form step failed"
                );
                ops->add_lin_comb( -1., *out, *reference );
                check( ops->norm( *reference ) < 5e-11, "Balanced mass-form endpoint changed under scaling" );
                auto view = step.get_continuous_integration();
                auto ref  = baseline.get_continuous_integration();
                for ( double theta : { 0., .25, .5, .75, 1. } )
                {
                    view.evaluate( theta, *sample );
                    ref.evaluate( theta, *other );
                    ops->add_lin_comb( -1., *sample, *other );
                    check( ops->norm( *other ) < 1e-10, "Balanced mass-form dense output changed under scaling" );
                }
            }
}

int main()
try
{
    backend_type::init_device();
    static_assert(
        !detail::has_rate_evaluation<mass_problem<coupled_decay>, vector_type>::value,
        "Mass regression must not accidentally provide a rate callback"
    );
    for ( const char *method : { "ROS2", "ROS3", "ROS34PW2" } )
    {
        scaled_steps( method );
        for ( mass_matrix mass :
              { mass_matrix{ 1, 0, 0, 1 }, mass_matrix{ 2, 0, 0, .5 }, mass_matrix{ 2, .4, .4, 1 }, mass_matrix{} } )
        {
            orders( method, mass, coupled_decay{}, false );
            orders( method, mass, logistic{}, true );
        }
    }
    orders( "ROS34PW2", mass_matrix{}, noncommuting_decay{}, false );
    approximate_model<logistic> approximate;
    approximate.factor = .5;
    orders( "ROS34PW2", mass_matrix{}, approximate, true );
    std::cout << "PASS: Rosenbrock constant mass (" << checks << " checks)\n";
}
catch ( const std::exception &e )
{
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
