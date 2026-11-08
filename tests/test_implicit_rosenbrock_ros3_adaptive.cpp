#include "implicit_rosenbrock_time_problem.h"
#include <nmfd/time_steppers/integration/time_step_adaptation_filter.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_matlab.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_scipy.h>

using namespace rosenbrock_tests;

// The scalar error is u_high-u_low, of local order three. Applying the low
// endpoint after each step independently verifies the embedded global order two.
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
        check( estimate && exponent == 3, "ROS3 estimator did not supply its h^3 state defect" );
        error = ops->get_value_at_point( 0, *estimate );
        return adaptation_status::accepted;
    }
};

double embedded( double h, bool low )
{
    using P      = problem<decay>;
    using L      = direct_solver<P>;
    using S      = rosenbrock::time_step<operations_type, P, estimator_controller, L>;
    auto ops     = std::make_shared<operations_type>( 1 );
    auto control = std::make_shared<estimator_controller>( adaptation_type::params{ h } );
    control->ops = ops;
    S                step( ops, std::make_shared<P>( ops ), control, std::make_shared<L>( ops ), S::params{ "ROS3" } );
    vector_wrap_type state( *ops ), out( *ops );
    state.start_use();
    out.start_use();
    ops->assign_scalar( 1., *state );
    const int count = std::lround( 1 / h );
    for ( int i = 0; i < count; ++i )
    {
        const double t = i * h, next = ( i + 1 ) * h;
        step.set_time( t );
        step.set_target_time( next );
        step.apply( *state, *out );
        check( step.get_status() == single_step_status::converged, "ROS3 embedded trajectory failed" );
        if ( low )
            ops->add_mul_scalar( -control->error, 1., *out );
        step.finalize( low ? adaptation_status::accepted_modified : adaptation_status::accepted, next, *out );
        ops->assign( *out, *state );
    }
    return std::abs( ops->get_value_at_point( 0, *state ) - std::exp( -1. ) );
}

void test_estimator()
{
    for ( bool low : { false, true } )
    {
        const double ratio = embedded( .02, low ) / embedded( .01, low );
        check(
            low ? ratio > 3.7 && ratio < 4.3 : ratio > 7.4 && ratio < 8.6,
            "ROS3 primary/embedded global orders are incorrect"
        );
    }
    auto ops = std::make_shared<operations_type>( 1 );
    using P  = problem<decay>;
    using L  = direct_solver<P>;
    using S  = rosenbrock::time_step<operations_type, P, estimator_controller, L>;
    vector_wrap_type in( *ops ), out( *ops );
    in.start_use();
    out.start_use();
    ops->assign_scalar( 1., *in );
    double                         errors[2]{};
    const rosenbrock::method_table table( "ROS3" );
    for ( int level = 0; level < 2; ++level )
    {
        const double h = level ? .005 : .01, inverse = 1 / ( 1 / ( h * table.gamma ) + 1 );
        // An independent scalar evaluation of the three increment equations.
        const double k1 = -inverse, f2 = -( 1 + k1 );
        const double k2      = inverse * ( f2 + table.c( 1, 0 ) * k1 / h );
        const double k3      = inverse * ( f2 + ( table.c( 2, 0 ) * k1 + table.c( 2, 1 ) * k2 ) / h );
        const double defect  = table.e( 0 ) * k1 + table.e( 1 ) * k2 + table.e( 2 ) * k3;
        auto         control = std::make_shared<estimator_controller>( adaptation_type::params{ h } );
        control->ops         = ops;
        S step( ops, std::make_shared<P>( ops ), control, std::make_shared<L>( ops ), S::params{ "ROS3" } );
        step.apply( *in, *out );
        check(
            step.get_status() == single_step_status::converged && std::abs( control->error - defect ) < 3e-16,
            "ROS3 error vector disagrees with its increment equations"
        );
        errors[level] = std::abs( control->error );
        step.finalize( adaptation_status::accepted, h, *out );
    }
    check( errors[0] / errors[1] > 7.8 && errors[0] / errors[1] < 8.2, "ROS3 estimate has the wrong local power of h" );
}

template <class Base>
struct observed_controller : Base
{
    using Base::Base;
    unsigned int assessments = 0, rejections = 0, commits = 0;
    void         reset()
    {
        Base::reset();
        assessments = rejections = commits = 0;
    }
    adaptation_status assess(
        double t, double h, const vector_type &in, const vector_type &out, double &next, unsigned int order = 0,
        const vector_type *error = nullptr
    )
    {
        check( error && order == 3, "Adaptive ROS3 lost its h^3 estimator" );
        ++assessments;
        const auto outcome = Base::assess( t, h, in, out, next, order, error );
        rejections += outcome == adaptation_status::rejected;
        return outcome;
    }
    void update( adaptation_status outcome, double t, double h, const vector_type &state )
    {
        commits += outcome == adaptation_status::accepted || outcome == adaptation_status::accepted_modified;
        Base::update( outcome, t, h, state );
    }
};

// u'=-2u and logistic growth test accuracy control and repeatable controller
// reset. The non-autonomous manufactured exponential additionally tests f_t.
template <class Base, class P>
void trajectory(
    std::shared_ptr<operations_type> ops, std::shared_ptr<P> p, double initial, double exact, bool autonomous,
    const char *filter = nullptr
)
{
    using C                 = observed_controller<Base>;
    using L                 = direct_solver<P>;
    using S                 = rosenbrock::time_step<operations_type, P, C, L>;
    auto             linear = std::make_shared<L>( ops );
    vector_wrap_type in( *ops ), out( *ops );
    in.start_use();
    out.start_use();
    ops->assign_scalar( initial, *in );
    double      errors[2]{};
    std::size_t steps[2]{};
    for ( int level = 0; level < 2; ++level )
    {
        typename Base::params cp;
        cp.relative_tolerance = level ? 2e-5 : 2e-3;
        cp.absolute_tolerance = cp.relative_tolerance * .01;
        cp.initial_step = cp.maximum_step = .5;
        if constexpr ( std::is_same_v<Base, integration::time_step_adaptation_filter<operations_type>> )
            cp.controller = filter;
        auto               control = std::make_shared<C>( ops, cp );
        typename S::params sp;
        sp.method                                             = "ROS3";
        sp.autonomous                                         = autonomous;
        auto                                             step = std::make_shared<S>( ops, p, control, linear, sp );
        integration::time_integrator<operations_type, S> run( ops, step );
        run.set_time_interval( .37, .87 );
        double       first      = 0;
        unsigned int rejections = 0;
        for ( int restart = 0; restart < 2; ++restart )
        {
            run.apply( *in, *out );
            const double result = ops->get_value_at_point( 0, *out );
            check(
                run.get_status() == integration_status::completed && run.get_final_time() == .87,
                "Adaptive ROS3 failed to reach its target"
            );
            check(
                control->commits == run.get_steps() && control->assessments == control->commits + control->rejections &&
                    ops->get_value_at_point( 0, *in ) == initial,
                "Adaptive ROS3 committed a rejected trial or changed input"
            );
            if ( !restart )
            {
                first        = result;
                rejections   = control->rejections;
                steps[level] = run.get_steps();
            }
            else
                check(
                    std::abs( first - result ) < 1e-13 && rejections == control->rejections &&
                        steps[level] == run.get_steps(),
                    "Adaptive ROS3 retained history across trajectories"
                );
            errors[level] = std::abs( result - exact );
        }
        if ( level )
            check( control->rejections > 0, "Tight ROS3 trajectory did not exercise estimator rejection" );
    }
    check(
        errors[1] < .15 * errors[0] && errors[1] < 5e-5 && steps[1] > steps[0],
        "ROS3 tolerance refinement did not improve physical accuracy"
    );
}

template <class Base>
void systems( const char *filter = nullptr )
{
    auto ops = std::make_shared<operations_type>( 1 );
    trajectory<Base>( ops, std::make_shared<problem<decay>>( ops, decay{ 2 } ), 1, std::exp( -1. ), true, filter );
    trajectory<Base>(
        ops, std::make_shared<problem<logistic>>( ops ), .25, 1 / ( 1 + 3 * std::exp( -.5 ) ), true, filter
    );
}

int main()
try
{
    backend_type::init_device();
    test_estimator();
    systems<integration::time_step_adaptation_matlab<operations_type>>();
    systems<integration::time_step_adaptation_scipy<operations_type>>();
    for ( const char *filter : { "I", "H211", "PC", "PID", "H312", "PPID", "H321" } )
        systems<integration::time_step_adaptation_filter<operations_type>>( filter );
    auto ops = std::make_shared<operations_type>( 1 );
    trajectory<integration::time_step_adaptation_scipy<operations_type>>(
        ops, std::make_shared<time_problem<>>( ops ), std::exp( .74 ), std::exp( 1.74 ), false
    );
    std::cout << "ROS3 embedded orders, all adaptive controllers, time dependence and restart: " << checks
              << " checks PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
