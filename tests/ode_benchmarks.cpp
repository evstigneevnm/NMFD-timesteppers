#include "ode_benchmark_imex.h"
#include "ode_benchmark_rosenbrock.h"
#include "implicit_bdf_adaptive_test.h"
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

// Sample through the external-manager boundary, before the integrator finalizes
// the pending step. Keep accepted-step accounting in the original trajectory.
template <class Equation, class Step>
struct dense_trajectory_writer : trajectory_writer<Equation>
{
    using base = trajectory_writer<Equation>;
    trajectory_writer<Equation> samples;
    std::shared_ptr<Step>       step;
    vector_wrap_type            sample;
    double                      start, spacing, target, begin = 0, last_sample;
    std::size_t                 index = 1;

    dense_trajectory_writer(
        std::shared_ptr<operations_type> ops, const std::string &path, const json &metadata,
        std::shared_ptr<Step> method
    )
        : base( ops, path, metadata ), samples(
                                           ops, path.substr( 0, path.size() - 4 ) + ".dense.dat",
                                           [&] {
                                               auto m        = metadata;
                                               m["sampling"] = "uniform_dense";
                                               return m;
                                           }()
                                       ),
          step( std::move( method ) ), sample( *ops ), start( metadata.at( "start_time" ).template get<double>() ),
          spacing( metadata.at( "sample_step" ).template get<double>() ),
          target( metadata.at( "end_time" ).template get<double>() ), last_sample( start )
    {
        sample.start_use();
    }
    void record( double time, const vector_type &state )
    {
        base::record( time, state );
        samples.record( time, state );
    }
    void set_time_interval( const double &from, double &to )
    {
        begin = from;
        base::set_time_interval( from, to );
    }
    bool apply( const vector_type &in, vector_type &out )
    {
        const auto view = step->get_continuous_integration();
        for ( double next = start + double( index ) * spacing; next <= this->time;
              next        = start + double( ++index ) * spacing )
        {
            require( std::isfinite( next ) && next > last_sample, "Uniform sample time did not advance" );
            const auto theta = std::clamp( ( next - begin ) / step->get_dt(), 0., 1. );
            last_sample      = view.evaluate( theta, *sample );
            samples.record( last_sample, *sample );
        }
        if ( this->time == target && last_sample < target )
        {
            samples.record( target, out );
            last_sample = target;
        }
        return base::apply( in, out );
    }
    void complete( json report )
    {
        samples.complete( report );
        base::complete( std::move( report ) );
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

template <
    class Equation, class Controller, bool IMEX = false, bool Dense = false, bool BDF = false, bool Rosenbrock = false>
int integrate(
    const Equation &equation, const json &config, const std::string &path, const std::string &method,
    const std::string &controller
)
{
    using problem_type = std::conditional_t<
        Rosenbrock, benchmark_rosenbrock_problem<Equation>,
        std::conditional_t<IMEX, benchmark_implicit_partition<Equation>, stage_problem<Equation>>>;
    using types        = newton_types<problem_type>;
    using solver_type  = std::conditional_t<Rosenbrock, typename types::linear_type, typename types::solver_type>;
    using control_type = std::conditional_t<
        std::is_same_v<Controller, adaptation_type>, benchmark_constant_controller,
        std::conditional_t<BDF, bdf_test::observed_bdf_controller<Controller>, observed_controller<Controller>>>;
    using flow_problem_type = std::conditional_t<IMEX, benchmark_imex_problem<Equation>, problem_type>;
    using rk_step_type      = std::conditional_t<
             IMEX,
             runge_kutta::imex_time_step<
                 operations_type, flow_problem_type, control_type, typename types::solver_type, Dense>,
             runge_kutta::implicit_time_step<
                 operations_type, problem_type, control_type, typename types::solver_type, Dense>>;
    using step_type = std::conditional_t<
        Rosenbrock, rosenbrock::time_step<operations_type, problem_type, control_type, solver_type, Dense>,
        std::conditional_t<
            BDF, bdf::implicit_time_step<operations_type, problem_type, control_type, solver_type, Dense>,
            rk_step_type>>;
    using writer_type =
        std::conditional_t<Dense, dense_trajectory_writer<Equation, step_type>, trajectory_writer<Equation>>;
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
            auto result = std::make_shared<control_type>( ops, p );
            if constexpr ( !BDF )
                result->expected_order = Rosenbrock ? rosenbrock::method_table( method ).error_order()
                                                : IMEX ? runge_kutta::make_imex_butcher_table( method ).error_order()
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
    auto solver = [&] {
        if constexpr ( Rosenbrock )
        {
            typename solver_type::params lp;
            lp.basis_size            = 4;
            lp.batch_size            = 1;
            lp.monitor.abs_tol       = 0;
            lp.monitor.rel_tol       = 1e-12;
            lp.monitor.max_iters_num = 40;
            return std::make_shared<solver_type>( ops, nullptr, lp );
        }
        else
            return types::make_solver( ops );
    }();
    typename step_type::params step_settings{ method };
    if constexpr ( Rosenbrock )
        step_settings.autonomous = benchmark_rosenbrock_autonomous<Equation>;
    if constexpr ( BDF )
    {
        step_settings.adaptive      = !std::is_same_v<Controller, adaptation_type>;
        step_settings.maximum_order = config.value( "bdf_maximum_order", 5u );
    }
    auto         step  = std::make_shared<step_type>( ops, flow_problem, adaptation, solver, step_settings );
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
    auto metadata                  = config;
    metadata["format"]             = "nmfd.dirk_system.v1";
    metadata["dimension"]          = Equation::size;
    metadata["method"]             = method;
    metadata["controller"]         = controller;
    metadata["backend"]            = backend_name;
    metadata["scheme"]             = Rosenbrock ? "rosenbrock" : BDF ? "bdf" : IMEX ? "imex" : "dirk";
    metadata["dense_output_order"] = step->dense_output_order();
    if constexpr ( BDF )
    {
        metadata["adaptive"] = step_settings.adaptive;
        if constexpr ( Dense )
            metadata["dense_output_order_policy"] = "used_bdf_order";
        if ( method == "BDF" )
            metadata["maximum_order"] = step_settings.maximum_order;
    }
    if constexpr ( IMEX )
        metadata["splitting"] = config.at( "splitting" );
    auto writer = [&]() {
        if constexpr ( Dense )
            return std::make_shared<writer_type>( ops, path, metadata, step );
        else
            return std::make_shared<writer_type>( ops, path, metadata );
    }();
    writer->record( start, *initial );
    integration::time_integrator<operations_type, step_type, writer_type> integrator(
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
    json report = {
        { "status", completed ? "completed" : "failed" },
        { "final_time", integrator.get_final_time() },
        { "accepted_steps", integrator.get_steps() },
        { "error_rejections", adaptation->error_rejections },
        { "numerical_rejections", adaptation->numerical_rejections },
        { "linearizations", problem->linearizations },
        { "step_status", static_cast<int>( step->get_status() ) }
    };
    if constexpr ( !Rosenbrock )
        report["newton_status"] = solver->convergence_strategy()->get_result_status();
    if constexpr ( BDF && !std::is_same_v<Controller, adaptation_type> )
    {
        report["orders_used"] = json::array();
        for ( unsigned int q = 1; q <= 5; ++q )
            if ( adaptation->committed_orders & ( 1u << q ) )
                report["orders_used"].push_back( q );
    }
    writer->complete( report );
    return completed ? 0 : 1;
}

template <class Equation, class Controller>
int select_method(
    const Equation &equation, const json &config, const std::string &path, const std::string &method,
    const std::string &controller
)
{
    if ( method == "ROS2" || method == "ROS3" || method == "ROS34PW2" )
    {
        if constexpr ( benchmark_supports_rosenbrock<Equation> )
        {
            if ( config.value( "sample_step", 0. ) > 0 )
                return integrate<Equation, Controller, false, true, false, true>(
                    equation, config, path, method, controller
                );
            return integrate<Equation, Controller, false, false, false, true>(
                equation, config, path, method, controller
            );
        }
        else
            throw std::invalid_argument( "Rosenbrock benchmarks support identity-mass ODEs only" );
    }
    if ( method == "BDF1" || method == "BDF2" || method == "BDF" )
    {
        if ( config.value( "sample_step", 0. ) > 0 )
        {
            if constexpr ( benchmark_supports_imex<Equation> )
                return integrate<Equation, Controller, false, true, true>( equation, config, path, method, controller );
            else
                throw std::invalid_argument( "Uniform DAE sampling requires problem-owned reconstruction" );
        }
        return integrate<Equation, Controller, false, false, true>( equation, config, path, method, controller );
    }
    if ( method.rfind( "IMEX_", 0 ) == 0 )
    {
        if constexpr ( benchmark_supports_imex<Equation> )
        {
            require(
                controller == "constant" || runge_kutta::make_imex_butcher_table( method ).is_embedded(),
                "This IMEX table requires the constant controller"
            );
            if ( config.value( "sample_step", 0. ) > 0 )
                return integrate<Equation, Controller, true, true>( equation, config, path, method, controller );
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
