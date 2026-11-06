#ifndef NMFD_TIME_STEPPERS_TESTS_IMPLICIT_IMEX_ADAPTIVE_H
#define NMFD_TIME_STEPPERS_TESTS_IMPLICIT_IMEX_ADAPTIVE_H

#include "implicit_imex_test.h"
#include "implicit_adaptive_test.h"

namespace nmfd::time_steppers::tests
{
struct imex_error_capture : adaptation_type
{
    std::shared_ptr<operations_type> operations;
    bool                             enabled  = true;
    double                           error    = 0;
    unsigned int                     exponent = 0, assessments = 0;
    imex_error_capture( std::shared_ptr<operations_type> ops, double step )
        : adaptation_type( params{ step } ), operations( std::move( ops ) )
    {
    }
    bool requires_error_estimate() const
    {
        return enabled;
    }
    adaptation_status assess(
        double, double, const vector_type &, const vector_type &, double &next, unsigned int order = 0,
        const vector_type *estimate = nullptr
    )
    {
        require( ( estimate != nullptr ) == enabled, "Unexpected combined IMEX error storage" );
        error    = estimate ? operations->get_value_at_point( 0, *estimate ) : 0;
        exponent = order;
        ++assessments;
        next = get_dt();
        return adaptation_status::accepted;
    }
};
}
#endif
