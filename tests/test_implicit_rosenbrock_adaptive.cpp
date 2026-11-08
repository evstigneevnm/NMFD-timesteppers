#include "implicit_rosenbrock_test.h"
#include <nmfd/time_steppers/integration/time_step_adaptation_filter.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_matlab.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_scipy.h>

using namespace rosenbrock_tests;

// Separate linear decay and logistic trajectories have exact finite-time
// endpoints. Tightening local tolerances must improve physical accuracy without
// committing rejected trials or retaining filter history between trajectories.
template <class C>
struct observed_controller : C
{
    using C::C;
    unsigned int assessments = 0, rejections = 0, commits = 0, numerical_rejections = 0;
    void         reset()
    {
        C::reset();
        assessments = rejections = commits = numerical_rejections = 0;
    }
    adaptation_status assess(
        double t, double h, const vector_type &in, const vector_type &out, double &next, unsigned int order = 0,
        const vector_type *error = nullptr
    )
    {
        check( error && order == 2, "ROS2 controller lost the state error or its h^2 exponent" );
        ++assessments;
        const auto decision = C::assess( t, h, in, out, next, order, error );
        rejections += decision == adaptation_status::rejected;
        return decision;
    }
    void update( adaptation_status outcome, double t, double h, const vector_type &state )
    {
        commits += outcome == adaptation_status::accepted || outcome == adaptation_status::accepted_modified;
        C::update( outcome, t, h, state );
    }
    adaptation_status reject_step( double h )
    {
        ++numerical_rejections;
        return C::reject_step( h );
    }
};

template <class C, class Model>
void trajectory( Model model, double initial, double exact, const char *filter = nullptr )
{
    using P                 = problem<Model>;
    using L                 = direct_solver<P>;
    using A                 = observed_controller<C>;
    using S                 = rosenbrock::time_step<operations_type, P, A, L>;
    auto             ops    = std::make_shared<operations_type>( 1 );
    auto             p      = std::make_shared<P>( ops, model );
    auto             linear = std::make_shared<L>( ops );
    vector_wrap_type in( *ops ), out( *ops );
    in.start_use();
    out.start_use();
    ops->assign_scalar( initial, *in );
    double      errors[2]{};
    std::size_t steps[2]{};
    for ( int level = 0; level < 2; ++level )
    {
        typename C::params settings;
        settings.relative_tolerance = level ? 2e-5 : 2e-3;
        settings.absolute_tolerance = settings.relative_tolerance * .01;
        settings.initial_step = settings.maximum_step = .5;
        if constexpr ( std::is_same_v<C, integration::time_step_adaptation_filter<operations_type>> )
            settings.controller = filter;
        auto                                             control = std::make_shared<A>( ops, settings );
        auto                                             step    = std::make_shared<S>( ops, p, control, linear );
        integration::time_integrator<operations_type, S> run( ops, step );
        run.set_time_interval( .37, .87 );
        double       first            = 0;
        unsigned int first_rejections = 0;
        for ( int restart = 0; restart < 2; ++restart )
        {
            run.apply( *in, *out );
            const double result = ops->get_value_at_point( 0, *out );
            check(
                run.get_status() == integration_status::completed && run.get_final_time() == .87,
                "Adaptive ROS2 did not reach its exact target"
            );
            check(
                control->commits == run.get_steps() && control->assessments == control->commits + control->rejections,
                "Adaptive ROS2 committed a rejected candidate"
            );
            check( ops->get_value_at_point( 0, *in ) == initial, "Adaptive ROS2 changed its input" );
            if ( !restart )
            {
                first            = result;
                first_rejections = control->rejections;
                steps[level]     = run.get_steps();
            }
            else
                check(
                    std::abs( first - result ) < 1e-13 && first_rejections == control->rejections &&
                        steps[level] == run.get_steps(),
                    "Adaptive ROS2 restart retained controller history"
                );
            errors[level] = std::abs( result - exact );
        }
        check( control->rejections > 0, "ROS2 did not exercise embedded-error rejection" );
    }
    check(
        errors[1] < .15 * errors[0] && errors[1] < 5e-5 && steps[1] > steps[0],
        "ROS2 tolerance refinement did not improve physical accuracy"
    );
}

// The embedded estimate is u_high-u_low, where u_low=u+k1/gamma.
// Verify that actual vector (not just its norm), including the h^2 scaling.
struct estimator_controller : adaptation_type
{
    using adaptation_type::adaptation_type;
    std::shared_ptr<operations_type> ops;
    double                           error = 0;
    bool                             requires_error_estimate() const
    {
        return true;
    }
    adaptation_status assess(
        double, double, const vector_type &, const vector_type &, double &, unsigned int exponent = 0,
        const vector_type *estimate = nullptr
    )
    {
        check( estimate && exponent == 2, "ROS2 estimator metadata is invalid" );
        error = ops->get_value_at_point( 0, *estimate );
        return adaptation_status::accepted;
    }
};

void test_estimator()
{
    auto ops = std::make_shared<operations_type>( 1 );
    using P  = problem<decay>;
    using L  = direct_solver<P>;
    using S  = rosenbrock::time_step<operations_type, P, estimator_controller, L>;
    vector_wrap_type in( *ops ), out( *ops );
    in.start_use();
    out.start_use();
    ops->assign_scalar( 1., *in );
    double errors[2]{};
    for ( int i = 0; i < 2; ++i )
    {
        const double h       = i ? .0005 : .001;
        auto         p       = std::make_shared<P>( ops );
        auto         control = std::make_shared<estimator_controller>( adaptation_type::params{ h } );
        control->ops         = ops;
        S step( ops, p, control, std::make_shared<L>( ops ) );
        step.apply( *in, *out );
        check( step.get_status() == single_step_status::converged, "ROS2 estimator test failed" );
        const double embedded = 1 - h / ( 1 + step.table().gamma * h );
        check(
            std::abs( control->error - ( ops->get_value_at_point( 0, *out ) - embedded ) ) < 5e-16,
            "ROS2 error is not the primary/embedded state difference"
        );
        errors[i] = std::abs( control->error );
        step.finalize( adaptation_status::accepted, h, *out );
    }
    check( errors[0] / errors[1] > 3.97 && errors[0] / errors[1] < 4.03, "ROS2 error has the wrong power of h" );
}

void test_adaptive_rollback()
{
    using Base      = integration::time_step_adaptation_filter<operations_type>;
    using C         = observed_controller<Base>;
    using P         = problem<decay>;
    using L         = direct_solver<P>;
    using S         = rosenbrock::time_step<operations_type, P, C, L>;
    auto ops        = std::make_shared<operations_type>( 1 );
    auto p          = std::make_shared<P>( ops );
    auto linear     = std::make_shared<L>( ops );
    linear->fail_on = 2;
    Base::params cp;
    cp.relative_tolerance = 1e-4;
    cp.absolute_tolerance = 1e-6;
    cp.initial_step = cp.maximum_step = .1;
    auto             control          = std::make_shared<C>( ops, cp );
    S                step( ops, p, control, linear );
    vector_wrap_type in( *ops ), out( *ops );
    in.start_use();
    out.start_use();
    ops->assign_scalar( 1., *in );
    step.apply( *in, *out );
    check(
        step.get_status() == single_step_status::converged && control->numerical_rejections == 1 &&
            p->linearizations == control->assessments + control->numerical_rejections && control->commits == 0,
        "Adaptive ROS2 assessed or committed a failed linear stage"
    );
    check(
        std::abs( ops->get_value_at_point( 0, *out ) - std::exp( -step.get_dt() ) ) < 1e-5,
        "Adaptive ROS2 recovery published an inaccurate candidate"
    );
    step.finalize( adaptation_status::rejected, 0, *in );
    const auto bindings = linear->bindings;
    step.apply( *in, *out );
    check(
        step.get_status() == single_step_status::converged && linear->bindings > bindings,
        "External ROS2 rejection reused stale stage data"
    );
    const double shortened = step.get_dt() / 2;
    ops->assign_scalar( std::exp( -shortened ), *out );
    step.finalize( adaptation_status::accepted_modified, shortened, *out );
    check( control->get_dt() <= shortened, "Modified ROS2 endpoint retained an excessive filter proposal" );
    step.set_time( shortened );
    step.apply( *out, *out );
    check(
        step.get_status() == single_step_status::converged &&
            std::abs( ops->get_value_at_point( 0, p->previous_state() ) - std::exp( -shortened ) ) < 1e-15,
        "ROS2 did not prepare from the externally modified state"
    );
    step.finalize( adaptation_status::accepted, shortened + step.get_dt(), *out );

    using FixedMinimum = integration::time_step_adaptation_scipy<operations_type>;
    using MinimumStep  = rosenbrock::time_step<operations_type, P, FixedMinimum, L>;
    FixedMinimum::params mp;
    mp.relative_tolerance = 1e-12;
    mp.absolute_tolerance = 1e-14;
    mp.initial_step = mp.minimum_step = mp.maximum_step = .1;
    MinimumStep stopped( ops, p, std::make_shared<FixedMinimum>( ops, mp ), linear );
    ops->assign_scalar( 42., *out );
    stopped.apply( *in, *out );
    check(
        stopped.get_status() == single_step_status::failed_minimum_dt && ops->get_value_at_point( 0, *out ) == 42,
        "ROS2 error rejection ignored the minimum step or changed output"
    );
}

template <class C>
void systems( const char *filter = nullptr )
{
    trajectory<C>( decay{}, 1, std::exp( -.5 ), filter );
    trajectory<C>( logistic{}, .25, 1 / ( 1 + 3 * std::exp( -.5 ) ), filter );
}

int main()
try
{
    backend_type::init_device();
    test_estimator();
    test_adaptive_rollback();
    systems<integration::time_step_adaptation_matlab<operations_type>>();
    systems<integration::time_step_adaptation_scipy<operations_type>>();
    for ( const char *filter : { "I", "H211", "PC", "PID", "H312", "PPID", "H321" } )
        systems<integration::time_step_adaptation_filter<operations_type>>( filter );
    std::cout << "Adaptive ROS2 embedding, all controllers, refinement and restart: " << checks << " checks PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
