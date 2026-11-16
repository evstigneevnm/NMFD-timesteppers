#include <nmfd/time_steppers/detail/apply_operator.h>
#include <iostream>
#include <stdexcept>

using nmfd::time_steppers::detail::apply_operator;
using nmfd::time_steppers::detail::operator_apply_traits;

struct void_operator
{
    int  calls = 0;
    void apply( int &value )
    {
        ++calls;
        ++value;
    }
};
struct bool_operator
{
    mutable int calls   = 0;
    bool        success = true;
    bool        apply( int &value ) const
    {
        ++calls;
        ++value;
        return success;
    }
};
struct overloaded_operator
{
    int  calls = 0;
    void apply( int &value )
    {
        ++calls;
        ++value;
    }
    int apply( const int & )
    {
        throw std::runtime_error( "Wrong overload selected" );
    }
};
struct missing_apply
{
};
struct wrong_argument
{
    void apply( int * )
    {
    }
};
struct integer_result
{
    int apply( int & )
    {
        return 1;
    }
};
struct proxy_result
{
    struct proxy
    {
        operator bool() const
        {
            return true;
        }
    };
    proxy apply( int & )
    {
        return {};
    }
};
struct reference_result
{
    bool &apply( int & );
};
struct throwing_operator
{
    void apply( int & )
    {
        throw std::runtime_error( "Expected operator exception" );
    }
};

static_assert( operator_apply_traits<void_operator, int>::value );
static_assert( operator_apply_traits<const bool_operator, int>::value );
static_assert( operator_apply_traits<overloaded_operator, int>::value );
static_assert( !operator_apply_traits<missing_apply, int>::value );
static_assert( !operator_apply_traits<wrong_argument, int>::value );
static_assert( !operator_apply_traits<integer_result, int>::value );
static_assert( !operator_apply_traits<proxy_result, int>::value );
static_assert( !operator_apply_traits<reference_result, int>::value );

int main()
try
{
    int                 value = 0;
    void_operator       mapping;
    bool_operator       flagged;
    overloaded_operator overloaded;
    if ( !apply_operator( mapping, value ) || mapping.calls != 1 || value != 1 )
        throw std::runtime_error( "Void operator not invoked exactly once" );
    if ( !apply_operator( flagged, value ) || flagged.calls != 1 || value != 2 )
        throw std::runtime_error( "Boolean operator not invoked exactly once" );
    flagged.success = false;
    if ( apply_operator( flagged, value ) || flagged.calls != 2 || value != 3 )
        throw std::runtime_error( "Boolean failure was discarded" );
    if ( !apply_operator( overloaded, value ) || overloaded.calls != 1 || value != 4 )
        throw std::runtime_error( "In-place overload dispatch failed" );
    throwing_operator throws;
    bool              propagated = false;
    try
    {
        apply_operator( throws, value );
    }
    catch ( const std::runtime_error & )
    {
        propagated = true;
    }
    if ( !propagated )
        throw std::runtime_error( "Operator exception swallowed" );
    std::cout << "apply_operator type detection, exactly-once dispatch and failure propagation: PASS\n";
}
catch ( const std::exception &e )
{
    std::cerr << e.what() << '\n';
    return 1;
}
