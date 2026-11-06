#include "ode_benchmark_imex.h"
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <string>
#include <nlohmann/json.hpp>

using namespace nmfd::time_steppers;
using namespace nmfd::time_steppers::tests;
using json = nlohmann::json;

#ifdef TEST_VECTOR_BACKEND_CUDA
constexpr const char *backend_name = "cuda";
#elif defined( TEST_VECTOR_BACKEND_OMP )
constexpr const char *backend_name = "cpu_omp";
#else
constexpr const char *backend_name = "cpu";
#endif

template <class Equation>
struct trajectory_writer
{
    std::shared_ptr<operations_type> operations;
    std::ofstream                    output;
    double                           time = 0, maximum_constraint = 0;
    std::size_t                      rows = 0;

    trajectory_writer( std::shared_ptr<operations_type> ops, const std::string &path, const json &metadata )
        : operations( std::move( ops ) )
    {
        output.exceptions( std::ios::failbit | std::ios::badbit );
        output.imbue( std::locale::classic() );
        output.open( path );
        output << std::setprecision( std::numeric_limits<double>::max_digits10 );
        output << "# " << metadata.dump() << "\n# time state... constraint\n";
    }
    void record( double t, const vector_type &state )
    {
        scfd::static_vec::vec<double, Equation::size> values;
        operations->get( state, values.d, Equation::size );
        double constraint = 0;
        if constexpr ( std::is_same_v<Equation, coupled_dae> || std::is_same_v<Equation, moving_constraint_dae> ||
                       std::is_same_v<Equation, stokes_dae> )
            constraint = constraint_monitor<Equation>::constraint( values, t );
        require( std::isfinite( constraint ) && std::abs( constraint ) < 2e-9, "Accepted DAE constraint violation" );
        maximum_constraint = std::max( maximum_constraint, std::abs( constraint ) );
        output << t;
        for ( const auto value : values.d )
        {
            require( std::isfinite( value ), "Nonfinite accepted state" );
            output << ' ' << value;
        }
        output << ' ' << constraint << '\n';
        ++rows;
    }
    void set_time_interval( const double &, double &end )
    {
        time = end;
    }
    bool apply( const vector_type &, vector_type &state )
    {
        record( time, state );
        return false;
    }
    integration_status get_status() const
    {
        return integration_status::running;
    }
    void complete( json report )
    {
        report["rows"]               = rows;
        report["maximum_constraint"] = maximum_constraint;
        output << "# " << report.dump() << '\n';
        output.close();
        std::cout << report.dump( 2 ) << '\n';
    }
};

struct benchmark_constant_controller : adaptation_type
{
    using adaptation_type::adaptation_type;
    unsigned int error_rejections = 0, numerical_rejections = 0, commits = 0;
    void         reset()
    {
        adaptation_type::reset();
        numerical_rejections = commits = 0;
    }
    adaptation_status reject_step( double h )
    {
        ++numerical_rejections;
        return adaptation_type::reject_step( h );
    }
    void update( adaptation_status outcome, double t, double h, const vector_type &state )
    {
        commits += outcome == adaptation_status::accepted || outcome == adaptation_status::accepted_modified;
        adaptation_type::update( outcome, t, h, state );
    }
};

template <class Equation, class Controller, bool IMEX = false>
int integrate(
    const Equation &equation, const json &config, const std::string &path, const std::string &method,
    const std::string &controller
)
{
    using problem_type = std::conditional_t<IMEX, benchmark_implicit_partition<Equation>, stage_problem<Equation>>;
    using types        = newton_types<problem_type>;
    using control_type = std::conditional_t<
        std::is_same_v<Controller, adaptation_type>, benchmark_constant_controller, observed_controller<Controller>>;
    using flow_problem_type = std::conditional_t<IMEX, benchmark_imex_problem<Equation>, problem_type>;
    using step_type         = std::conditional_t<
                IMEX,
                runge_kutta::imex_time_step<operations_type, flow_problem_type, control_type, typename types::solver_type>,
                runge_kutta::implicit_time_step<operations_type, problem_type, control_type, typename types::solver_type>>;
    const auto &settings   = config.at( "adaptation" );
    auto        ops        = std::make_shared<operations_type>( Equation::size );
    auto        adaptation = [&]() {
        typename Controller::params p;
        if constexpr ( std::is_same_v<Controller, adaptation_type> )
        {
            p.step_size = settings.at( "initial_step" ).get<double>();
            require( p.step_size <= settings.at( "maximum_step" ).get<double>(), "Constant step exceeds maximum" );
            return std::make_shared<control_type>( p );
        }
        else
        {
            p.relative_tolerance = settings.at( "relative_tolerance" ).get<double>();
            p.absolute_tolerance = settings.at( "absolute_tolerance" ).get<double>();
            p.initial_step       = settings.at( "initial_step" ).get<double>();
            p.minimum_step       = settings.at( "minimum_step" ).get<double>();
            p.maximum_step       = settings.at( "maximum_step" ).get<double>();
            if constexpr ( std::is_same_v<Controller, integration::time_step_adaptation_filter<operations_type>> )
                p.controller = controller;
            auto result            = std::make_shared<control_type>( ops, p );
            result->expected_order = IMEX ? runge_kutta::make_imex_butcher_table( method ).error_order()
                                                 : runge_kutta::make_butcher_table( method ).error_order();
            return result;
        }
    }();
    auto problem      = std::make_shared<problem_type>( ops, equation );
    auto flow_problem = [&]() {
        if constexpr ( IMEX )
        {
            using explicit_type = benchmark_explicit_partition<Equation>;
            auto bundle         = std::make_shared<typename flow_problem_type::imex_operator_type>(
                std::make_shared<explicit_type>( explicit_type{ equation } ), problem
            );
            return std::make_shared<flow_problem_type>( flow_problem_type{ std::move( bundle ) } );
        }
        else
            return problem;
    }();
    auto solver = types::make_solver( ops );
    auto step =
        std::make_shared<step_type>( ops, flow_problem, adaptation, solver, typename step_type::params{ method } );
    const double start = config.at( "start_time" ).get<double>(), end = config.at( "end_time" ).get<double>();
    require(
        std::isfinite( start ) && std::isfinite( end ) && end > start && std::isfinite( end - start ),
        "Invalid time interval"
    );
    const auto initial_values = config.at( "initial_state" ).get<std::vector<double>>();
    require( initial_values.size() == Equation::size, "Initial-state dimension mismatch" );
    vector_wrap_type initial( *ops ), result( *ops );
    initial.start_use();
    result.start_use();
    ops->set( initial_values.data(), *initial, Equation::size );
    auto metadata          = config;
    metadata["format"]     = "nmfd.dirk_system.v1";
    metadata["dimension"]  = Equation::size;
    metadata["method"]     = method;
    metadata["controller"] = controller;
    metadata["backend"]    = backend_name;
    metadata["scheme"]     = IMEX ? "imex" : "dirk";
    if constexpr ( IMEX )
        metadata["splitting"] = config.at( "splitting" );
    auto writer = std::make_shared<trajectory_writer<Equation>>( ops, path, metadata );
    writer->record( start, *initial );
    integration::time_integrator<operations_type, step_type, trajectory_writer<Equation>> integrator(
        ops, step, { config.at( "maximum_steps" ).get<std::size_t>() }, writer
    );
    integrator.set_time_interval( start, end );
    integrator.apply( *initial, *result );
    const bool completed =
        integrator.get_status() == integration_status::completed && integrator.get_final_time() == end;
    require(
        writer->rows == integrator.get_steps() + 1 && adaptation->commits == integrator.get_steps(),
        "Accepted-step accounting mismatch"
    );
    writer->complete(
        { { "status", completed ? "completed" : "failed" },
          { "final_time", integrator.get_final_time() },
          { "accepted_steps", integrator.get_steps() },
          { "error_rejections", adaptation->error_rejections },
          { "numerical_rejections", adaptation->numerical_rejections },
          { "linearizations", problem->linearizations },
          { "step_status", static_cast<int>( step->get_status() ) },
          { "newton_status", solver->convergence_strategy()->get_result_status() } }
    );
    return completed ? 0 : 1;
}

template <class Equation, class Controller>
int select_method(
    const Equation &equation, const json &config, const std::string &path, const std::string &method,
    const std::string &controller
)
{
    if ( method.rfind( "IMEX_", 0 ) == 0 )
    {
        if constexpr ( benchmark_supports_imex<Equation> )
        {
            require(
                controller == "constant" || runge_kutta::make_imex_butcher_table( method ).is_embedded(),
                "This IMEX table requires the constant controller"
            );
            return integrate<Equation, Controller, true>( equation, config, path, method, controller );
        }
        else
            throw std::invalid_argument( "IMEX benchmarks do not support singular-mass DAEs" );
    }
    return integrate<Equation, Controller>( equation, config, path, method, controller );
}

template <class Equation>
int select_controller(
    const Equation &equation, const json &config, const std::string &path, const std::string &method,
    const std::string &controller
)
{
    if ( controller == "constant" )
        return select_method<Equation, adaptation_type>( equation, config, path, method, controller );
    if ( controller == "matlab" )
        return select_method<Equation, integration::time_step_adaptation_matlab<operations_type>>(
            equation, config, path, method, controller
        );
    if ( controller == "scipy" )
        return select_method<Equation, integration::time_step_adaptation_scipy<operations_type>>(
            equation, config, path, method, controller
        );
    return select_method<Equation, integration::time_step_adaptation_filter<operations_type>>(
        equation, config, path, method, controller
    );
}

int main( int argc, char **argv )
try
{
    require( argc == 5, "Usage: ode_benchmarks.bin config.json trajectory.dat METHOD CONTROLLER" );
    std::ifstream input( argv[1] );
    input.exceptions( std::ios::failbit | std::ios::badbit );
    const auto  config     = json::parse( input );
    const auto  system     = config.at( "system" ).get<std::string>();
    const auto &parameters = config.at( "parameters" );
    for ( const auto &value : parameters )
        require( std::isfinite( value.get<double>() ), "Nonfinite system parameter" );
    backend_type::init_device();
    if ( system == "lorenz" )
    {
        lorenz_equation equation;
        equation.sigma   = parameters.at( "sigma" ).get<double>();
        equation.rho     = parameters.at( "rho" ).get<double>();
        equation.beta    = parameters.at( "beta" ).get<double>();
        equation.epsilon = parameters.at( "epsilon" ).get<double>();
        equation.delta   = parameters.at( "delta" ).get<double>();
        return select_controller( equation, config, argv[2], argv[3], argv[4] );
    }
    if ( system == "rossler" )
    {
        rossler_equation equation;
        equation.a = parameters.at( "a" ).get<double>();
        equation.b = parameters.at( "b" ).get<double>();
        equation.c = parameters.at( "c" ).get<double>();
        return select_controller( equation, config, argv[2], argv[3], argv[4] );
    }
    if ( system == "vdp" || system == "vdp_stiff" )
    {
        vdp_equation equation;
        equation.mu = parameters.at( "mu" ).get<double>();
        return select_controller( equation, config, argv[2], argv[3], argv[4] );
    }
    if ( system == "coupled_mass" )
        return select_controller( coupled_mass{}, config, argv[2], argv[3], argv[4] );
    if ( system == "state_mass" )
        return select_controller( variable_mass{}, config, argv[2], argv[3], argv[4] );
    if ( system == "time_mass" )
        return select_controller( time_dependent_mass{}, config, argv[2], argv[3], argv[4] );
    if ( system == "coupled_dae" )
        return select_controller( coupled_dae{}, config, argv[2], argv[3], argv[4] );
    if ( system == "moving_dae" )
        return select_controller( moving_constraint_dae{}, config, argv[2], argv[3], argv[4] );
    if ( system == "stokes_dae" )
        return select_controller( stokes_dae{}, config, argv[2], argv[3], argv[4] );
    if ( system == "time_coefficient" )
        return select_controller(
            scalar_time_equation<coefficient_growth<double>>{}, config, argv[2], argv[3], argv[4]
        );
    if ( system == "stiff_tracking" )
        return select_controller(
            scalar_time_equation<stiff_tracking<double>>{
                stiff_tracking<double>{ parameters.at( "kappa" ).get<double>() }
            },
            config, argv[2], argv[3], argv[4]
        );
    throw std::invalid_argument( "Unknown system: " + system );
}
catch ( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
