#ifndef NMFD_TIME_STEPPERS_TESTS_IMPLICIT_BDF_DENSE_H
#define NMFD_TIME_STEPPERS_TESTS_IMPLICIT_BDF_DENSE_H

#include "implicit_bdf_test.h"
#include <nmfd/time_steppers/integration/time_step_adaptation_scipy.h>

namespace nmfd::time_steppers::tests::bdf_test
{
template <class Problem, class Controller = adaptation_type>
using dense_step =
    bdf::implicit_time_step<operations_type, Problem, Controller, typename newton_types<Problem>::solver_type, true>;

template <class Problem>
std::shared_ptr<dense_step<Problem>>
make_dense( std::shared_ptr<operations_type> ops, std::shared_ptr<Problem> problem, double h, const char *method )
{
    return std::make_shared<dense_step<Problem>>(
        ops, problem, std::make_shared<adaptation_type>( adaptation_type::params{ h } ),
        newton_types<Problem>::make_solver( ops ), typename dense_step<Problem>::params{ method }
    );
}

template <class Exception, class Action>
void throws( Action action, const char *message )
{
    bool caught = false;
    try
    {
        action();
    }
    catch ( const Exception & )
    {
        caught = true;
    }
    require( caught, message );
}

// Prescribe a reproducible grid while retaining the estimator and order policy.
// Numerical accuracy tests also use the ordinary SciPy controller separately.
struct dense_controller : adaptation_type
{
    explicit dense_controller( double h ) : adaptation_type( { h } ), next( h )
    {
    }
    double       next;
    unsigned int assessments = 0, commits = 0;
    bool         reject = false;
    double       get_dt() const
    {
        return next;
    }
    bool requires_error_estimate() const
    {
        return true;
    }
    adaptation_status assess(
        double, double h, const vector_type &, const vector_type &, double &proposal, unsigned int exponent = 0,
        const vector_type *error = nullptr
    )
    {
        require( error && exponent >= 2 && exponent <= 6, "Dense BDF lost its error estimator" );
        ++assessments;
        if ( reject )
        {
            reject = false;
            next   = std::abs( h ) / 2;
            return adaptation_status::rejected;
        }
        proposal = next;
        return adaptation_status::accepted;
    }
    void update( adaptation_status status, double, double, const vector_type & )
    {
        commits += status == adaptation_status::accepted || status == adaptation_status::accepted_modified;
    }
};
}
#endif
