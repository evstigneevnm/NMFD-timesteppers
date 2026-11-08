#include "implicit_rosenbrock_w_problem.h"
#include <nmfd/time_steppers/integration/time_step_adaptation_filter.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_matlab.h>
#include <nmfd/time_steppers/integration/time_step_adaptation_scipy.h>

using namespace rosenbrock_tests;

// For u'=-u with the intentionally wrong B=-0.3, apply both endpoints along
// separate trajectories: high is order three, low is order two, their difference
// has local order three. The low endpoint is explicitly committed as modified.
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
        double, double, const vector_type &, const vector_type &, double &, unsigned int order = 0,
        const vector_type *estimate = nullptr
    )
    {
        check( order == 3 && estimate, "W estimator lost its h^3 state defect" );
        error = ops->get_value_at_point( 0, *estimate );
        return adaptation_status::accepted;
    }
};

double embedded( double h, bool low, bool single = false )
{
    using P         = w_problem<decay>;
    using L         = direct_solver<P>;
    using S         = rosenbrock::time_step<operations_type, P, estimator_controller, L>;
    auto ops        = std::make_shared<operations_type>( 1 );
    auto p          = std::make_shared<P>( ops );
    p->model.factor = .3;
    auto control    = std::make_shared<estimator_controller>( adaptation_type::params{ h } );
    control->ops    = ops;
    S                step( ops, p, control, std::make_shared<L>( ops ), S::params{ "ROS34PW2" } );
    vector_wrap_type state( *ops ), out( *ops );
    state.start_use();
    out.start_use();
    ops->assign_scalar( 1., *state );
    const int count = single ? 1 : std::lround( 1 / h );
    for ( int i = 0; i < count; ++i )
    {
        step.set_time( i * h );
        step.set_target_time( ( i + 1 ) * h );
        step.apply( *state, *out );
        check( step.get_status() == single_step_status::converged, "W embedded trajectory failed" );
        if ( low )
            ops->add_mul_scalar( -control->error, 1., *out );
        step.finalize( low ? adaptation_status::accepted_modified : adaptation_status::accepted, ( i + 1 ) * h, *out );
        ops->assign( *out, *state );
    }
    return single ? std::abs( control->error ) : std::abs( ops->get_value_at_point( 0, *state ) - std::exp( -1. ) );
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
        check( error && order == 3, "Adaptive W method lost its estimator" );
        ++assessments;
        const auto result = Base::assess( t, h, in, out, next, order, error );
        rejections += result == adaptation_status::rejected;
        return result;
    }
    void update( adaptation_status result, double t, double h, const vector_type &state )
    {
        commits += result == adaptation_status::accepted || result == adaptation_status::accepted_modified;
        Base::update( result, t, h, state );
    }
};

// Logistic growth and manufactured exp(2t) test physical accuracy refinement,
// not merely convergence of shifted solves. All B approximations have O(1)
// errors independent of the timestep; repeated trajectories test controller reset.
template <class Base, class P>
void trajectory(
    std::shared_ptr<operations_type> ops, std::shared_ptr<P> p, double initial, double exact, bool autonomous,
    const char *filter = nullptr
)
{
    using C = observed_controller<Base>;
    using L = direct_solver<P>;
    using S = rosenbrock::time_step<operations_type, P, C, L>;
    vector_wrap_type in( *ops ), out( *ops );
    in.start_use();
    out.start_use();
    ops->assign_scalar( initial, *in );
    double      errors[2]{};
    std::size_t steps[2]{};
    for ( int level = 0; level < 2; ++level )
    {
        typename Base::params cp;
        cp.relative_tolerance = level ? 2e-6 : 2e-3;
        cp.absolute_tolerance = cp.relative_tolerance * .01;
        cp.initial_step = cp.maximum_step = .5;
        if constexpr ( std::is_same_v<Base, integration::time_step_adaptation_filter<operations_type>> )
            cp.controller = filter;
        auto               control = std::make_shared<C>( ops, cp );
        auto               linear  = std::make_shared<L>( ops );
        typename S::params sp;
        sp.method                                             = "ROS34PW2";
        sp.autonomous                                         = autonomous;
        auto                                             step = std::make_shared<S>( ops, p, control, linear, sp );
        integration::time_integrator<operations_type, S> run( ops, step );
        run.set_time_interval( .37, .87 );
        double       first    = 0;
        unsigned int rejected = 0;
        for ( int repeat = 0; repeat < 2; ++repeat )
        {
            const auto rates = p->rates, preparations = p->linearizations, calls = linear->calls;
            run.apply( *in, *out );
            check(
                run.get_status() == integration_status::completed && run.get_final_time() == .87 &&
                    ops->get_value_at_point( 0, *in ) == initial,
                "Adaptive W trajectory failed or changed its initial state"
            );
            const auto attempts = control->assessments;
            check(
                control->commits == run.get_steps() && attempts == control->commits + control->rejections &&
                    p->rates - rates == 4 * attempts && p->linearizations - preparations == attempts &&
                    linear->calls - calls == 4 * attempts,
                "W rejection committed a trial or reused stale stage data"
            );
            const double result = ops->get_value_at_point( 0, *out );
            if ( !repeat )
            {
                first        = result;
                rejected     = control->rejections;
                steps[level] = run.get_steps();
            }
            else
                check(
                    std::abs( first - result ) < 1e-13 && rejected == control->rejections &&
                        steps[level] == run.get_steps(),
                    "W controller retained trajectory history"
                );
            errors[level] = std::abs( result - exact );
        }
        if ( level )
            check( control->rejections > 0, "W test did not exercise error rejection" );
    }
    check(
        errors[1] < .1 * errors[0] && errors[1] < 5e-5 && steps[1] > steps[0],
        "W tolerance refinement did not improve physical accuracy"
    );
}

template <class Base>
void systems( const char *filter = nullptr )
{
    auto ops        = std::make_shared<operations_type>( 1 );
    auto p          = std::make_shared<w_problem<logistic>>( ops );
    p->model.factor = .3;
    p->model.offset = -.4;
    trajectory<Base>( ops, p, .25, 1 / ( 1 + 3 * std::exp( -.5 ) ), true, filter );
    trajectory<Base>(
        ops, std::make_shared<w_time_problem<>>( ops ), std::exp( .74 ), std::exp( 1.74 ), false, filter
    );
}

int main()
try
{
    backend_type::init_device();
    for ( bool low : { false, true } )
    {
        const double ratio = embedded( .02, low ) / embedded( .01, low );
        check(
            low ? ratio > 3.6 && ratio < 4.4 : ratio > 7.1 && ratio < 8.9,
            "W primary/embedded endpoints have incorrect global orders"
        );
    }
    const double ratio = embedded( .01, false, true ) / embedded( .005, false, true );
    check( ratio > 7.6 && ratio < 8.4, "W estimate does not scale as h^3" );
    systems<integration::time_step_adaptation_matlab<operations_type>>();
    systems<integration::time_step_adaptation_scipy<operations_type>>();
    for ( const char *filter : { "I", "H211", "PC", "PID", "H312", "PPID", "H321" } )
        systems<integration::time_step_adaptation_filter<operations_type>>( filter );
    std::cout << "ROS34PW2 embedded orders, all controllers, approximate Jacobians and repeatability: " << checks
              << " checks PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
