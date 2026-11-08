#ifndef NMFD_TIME_STEPPERS_ROSENBROCK_COEFFICIENTS_H
#define NMFD_TIME_STEPPERS_ROSENBROCK_COEFFICIENTS_H

#include <cmath>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace nmfd::time_steppers::rosenbrock
{
// ROS2 in the increment formulation documented by KPP:
// https://kpp.readthedocs.io/en/stable/num_methods/rosenbrock-methods.html#ros-2
// A*k1=f(t,u)+h*gamma1*f_t, A*k2=f(t+h,u+a21*k1)+(c21/h)*k1+h*gamma2*f_t.
// A=I/(gamma*h)-J(t,u); f_t is the partial derivative at the base time/state.
struct coefficients
{
    static constexpr unsigned int stages = 2, order = 2, embedded_order = 1;
    const double                  gamma = 1 + 1 / std::sqrt( 2. );
    const double                  a21 = 1 / gamma, c21 = -2 / gamma;
    const double                  m1 = 3 / ( 2 * gamma ), m2 = 1 / ( 2 * gamma );
    const double                  e1 = 1 / ( 2 * gamma ), e2 = 1 / ( 2 * gamma );
    const double                  gamma1 = gamma, gamma2 = -gamma;
    static constexpr unsigned int error_order = embedded_order + 1;

    explicit coefficients( const std::string &method = "ROS2" )
    {
        if ( method != "ROS2" )
            throw std::invalid_argument( "Unknown Rosenbrock method: " + method );
    }
};

// Named increment-form tables. Keep the original ROS2 descriptor above intact;
// stage counts and orders here describe the selected method, not static ROS2 data.
class method_table
{
    struct data
    {
        unsigned int          stages, order, embedded_order;
        double                gamma;
        std::array<double, 6> a, c;
        std::array<double, 4> m, e, time, time_derivative;
        std::array<bool, 4>   reuse;
        unsigned int          dense_order;
        bool                  approximate_jacobian;
    };
    const data data_;

    static data select( const std::string &method )
    {
        if ( method == "ROS2" )
        {
            const coefficients p;
            return {
                2,
                2,
                1,
                p.gamma,
                { p.a21, 0, 0 },
                { p.c21, 0, 0 },
                { p.m1, p.m2, 0 },
                { p.e1, p.e2, 0 },
                { 0, 1, 0 },
                { p.gamma1, p.gamma2, 0 },
                { false, false, false },
                2,
                false
            };
        }
        if ( method == "ROS3" )
        {
            // Full-precision Ros3 constants, not the rounded documentation table:
            // https://github.com/KineticPreProcessor/KPP/blob/main/int/rosenbrock.f90
            constexpr double g = 0.43586652150845899941601945119356;
            return {
                3,
                3,
                2,
                g,
                { 1, 1, 0 },
                { -0.10156171083877702091975600115545e1, 0.40759956452537699824805835358067e1,
                  0.92076794298330791242156818474003e1 },
                { 1, 0.61697947043828245592553615689730e1, -0.42772256543218573326238373806514 },
                { .5, -0.29079558716805469821718236208017e1, 0.22354069897811569627360909276199 },
                { 0, g, g },
                { g, 0.24291996454816804366592249683314, 0.21851380027664058511513169485832e1 },
                { false, false, true },
                3,
                false
            };
        }
        if ( method == "ROS34PW2" )
        {
            // Rang/Angermann RA34PW2, as tabulated in PETSc's rosw.c:
            // https://petsc.org/main/src/ts/impls/rosw/rosw.c.html
            // Convert once offline: a=A*Gamma^-1, c=-lower(Gamma^-1),
            // m=b^T*Gamma^-1, e=(b-bhat)^T*Gamma^-1. Stage times and
            // f_t weights are the row sums of the original A and Gamma.
            return {
                4,
                3,
                2,
                .435866521508459,
                { 2, 1.419217317455764650002648964852719, -.2592322116729697137830264607535575,
                  4.184760482319160731154865710549846, -.2851920173554959137014835526287717,
                  2.294280360279041716748138882049989 },
                { -4.588560720558083486133501479721273, -4.184760482319160731154865710549846,
                  .2851920173554959137014835526287717, -6.368179200128357763538327614748235,
                  -6.795620944466836184391369586498675, 2.870098604331056089209571124574071 },
                { 4.184760482319160145288398185867343, -.2851920173554958737746011228593438,
                  2.294280360279041395548888442984149, 1 },
                { .2777499476479681103770681569531932, -1.403239895175999024215416205098263,
                  1.772630127667550745234421928975938, .5 },
                { 0, .87173304301691801, .73157995778885238, 1 },
                { .435866521508459, -.43586652150845901, -.413333376233886494, 0 },
                { false, false, false, false },
                0,
                true
            };
        }
        throw std::invalid_argument( "Unknown Rosenbrock method: " + method );
    }

    void require_stage( std::size_t i ) const
    {
        if ( i >= data_.stages )
            throw std::out_of_range( "Rosenbrock stage index" );
    }
    double lower( const std::array<double, 6> &values, std::size_t i, std::size_t j ) const
    {
        require_stage( i );
        require_stage( j );
        return j < i ? values[i * ( i - 1 ) / 2 + j] : 0;
    }

public:
    // Preserve the existing step.table().gamma diagnostic for ROS2 callers.
    const double gamma;

    explicit method_table( const std::string &method = "ROS2" ) : data_( select( method ) ), gamma( data_.gamma )
    {
    }
    unsigned int stage_count() const
    {
        return data_.stages;
    }
    unsigned int order() const
    {
        return data_.order;
    }
    unsigned int embedded_order() const
    {
        return data_.embedded_order;
    }
    unsigned int error_order() const
    {
        return data_.embedded_order + 1;
    }
    unsigned int dense_output_order() const
    {
        return data_.dense_order;
    }
    bool supports_approximate_jacobian() const
    {
        return data_.approximate_jacobian;
    }
    double a( std::size_t i, std::size_t j ) const
    {
        return lower( data_.a, i, j );
    }
    double c( std::size_t i, std::size_t j ) const
    {
        return lower( data_.c, i, j );
    }
    double m( std::size_t i ) const
    {
        require_stage( i );
        return data_.m[i];
    }
    double e( std::size_t i ) const
    {
        require_stage( i );
        return data_.e[i];
    }
    double stage_time( std::size_t i ) const
    {
        require_stage( i );
        return data_.time[i];
    }
    double time_derivative( std::size_t i ) const
    {
        require_stage( i );
        return data_.time_derivative[i];
    }
    bool reuse_previous_rate( std::size_t i ) const
    {
        require_stage( i );
        return data_.reuse[i];
    }
};
}
#endif
