#!/usr/bin/env python3
"""Build and run DIRK/IMEX/BDF/Rosenbrock benchmarks with reference comparisons and plots."""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
CONTROLLERS = ("matlab", "scipy", "I", "H211", "PC", "PID", "H312", "PPID", "H321")
DEFAULT_METHODS = ("SDIRK2(1)2", "SDIRK3(1)3", "IMEX_ARK324L2SA")
FIXED_IMEX_METHODS = ("IMEX_EULER", "IMEX_HEUN_TR2", "IMEX_ARS222", "IMEX_ARS233")
BDF_METHODS = ("BDF1", "BDF2", "BDF")
ROSENBROCK_METHODS = ("ROS2", "ROS3")
ROSENBROCK_SYSTEMS = ("lorenz", "rossler", "vdp", "vdp_stiff", "time_coefficient", "stiff_tracking")
METHODS = DEFAULT_METHODS + FIXED_IMEX_METHODS + BDF_METHODS + ROSENBROCK_METHODS
SYSTEMS = ("lorenz", "rossler", "vdp", "vdp_stiff", "coupled_mass", "state_mass",
           "time_mass", "coupled_dae", "moving_dae", "stokes_dae", "time_coefficient", "stiff_tracking")
IMEX_SPLITS = {
    "lorenz": "Explicit nonlinear coupling/forcing; implicit linear terms.",
    "rossler": "Explicit z*x; implicit remaining linear/affine terms.",
    "vdp": "Explicit (v,-x); implicit (0,mu*(1-x^2)*v).",
    "vdp_stiff": "Explicit (v,-x); implicit (0,mu*(1-x^2)*v).",
    "coupled_mass": "Explicit off-diagonal physical RHS; implicit diagonal RHS; shared coupled mass.",
    "state_mass": "Physical RHS split 1:3, explicit:implicit; shared state-dependent mass.",
    "time_mass": "Physical RHS split 1:3, explicit:implicit; shared time-dependent mass.",
    "time_coefficient": "Time-dependent RHS split 1:3, explicit:implicit.",
    "stiff_tracking": "Explicit kappa*sin(t)+cos(t); implicit -kappa*u.",
}
BUILD_ENVIRONMENT = ("CXX", "CXXFLAGS", "CPPFLAGS", "CPPSTD", "OPENMP_FLAGS", "NVCC", "NVCCFLAGS",
                     "CUDA_ROOT", "CUDA_ARCH", "SCFD_ROOT", "NMFD_OPERATIONS_ROOT", "NMFD_NEWTON_ROOT",
                     "NMFD_LINSOLVERS_ROOT", "JSON_INCLUDE")


def system_definition(name):
    decay = lambda t, xp: [xp.exp(-t)]
    coupled = lambda t, xp: [xp.exp(-t), xp.exp(-2 * t)]
    exact, rhs, parameters = None, None, {}
    start, end, window = 0., 5., None
    if name == "lorenz":
        labels, initial, end, window = ("x", "y", "z"), [1, 1, 1], 50., 2.
        parameters = dict(sigma=10, rho=28, beta=8 / 3, epsilon=0, delta=0)
        def rhs(t, u):
            x, y, z = u
            return [parameters["sigma"] * (y - x) - parameters["epsilon"] * x**3,
                    parameters["rho"] * x - y - x * z + parameters["delta"],
                    x * y - parameters["beta"] * z - parameters["delta"]]
    elif name == "rossler":
        labels, initial, end, window = ("x", "y", "z"), [2, 0, 0], 50., 2.
        parameters = dict(a=.2, b=.2, c=5.7)
        def rhs(t, u):
            x, y, z = u
            return [-y - z, x + parameters["a"] * y, parameters["b"] + z * (x - parameters["c"])]
    elif name in ("vdp", "vdp_stiff"):
        labels, initial = ("x", "v"), [2, 0]
        mu = 1 if name == "vdp" else 1000
        parameters, end = dict(mu=mu), 20. if mu == 1 else 10.
        def rhs(t, u):
            x, v = u
            return [v, parameters["mu"] * (1 - x * x) * v - x]
    elif name in ("coupled_mass", "coupled_dae"):
        labels, exact = ("x", "z"), coupled
    elif name == "state_mass":
        labels, exact = ("u",), decay
    elif name == "time_mass":
        labels, exact = ("u",), lambda t, xp: [1 / (1 + t)]
        start = .25
    elif name == "moving_dae":
        labels, start = ("x", "z"), .25
        exact = lambda t, xp: [xp.exp(-t), xp.exp(-t) * xp.sin(t)]
    elif name == "stokes_dae":
        labels = ("u0", "u1", "p")
        exact = lambda t, xp: [xp.exp(-t), -xp.exp(-t), xp.sin(t)]
    elif name == "time_coefficient":
        labels, start, end = ("u",), .25, 2.
        exact = lambda t, xp: [xp.exp(t ** 2 / 2)]
    elif name == "stiff_tracking":
        labels, start = ("u",), .25
        parameters, exact = dict(kappa=10000), lambda t, xp: [xp.sin(t)]
    else:
        raise ValueError(f"Unknown system: {name}")
    return dict(labels=labels, exact=exact, rhs=rhs, parameters=parameters,
                start=start, end=end, window=window,
                initial=None if exact else initial,
                reference_method="Radau" if name.startswith("vdp") else "DOP853")


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive finite value")
    return number


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-B", dest="build_only", action="store_true",
                      help="Force rebuilding and assembling the benchmarks, without running them")
    mode.add_argument("--plot-only", type=Path, help="Replot an existing run without building or integrating")
    parser.add_argument("--build-dir", type=Path, default=Path("build/ode_benchmarks"),
                        help="Build and result directory, relative to the caller's working directory")
    parser.add_argument("--systems", "--system", choices=SYSTEMS, nargs="+", default=list(SYSTEMS))
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--initial", type=float, nargs="+")
    parser.add_argument("--rtol", type=positive, default=1e-5)
    parser.add_argument("--atol", type=positive, default=1e-7)
    parser.add_argument("--step", type=positive, default=.001)
    parser.add_argument("--min-step", type=positive, default=1e-8)
    parser.add_argument("--max-step", type=positive, default=.05)
    parser.add_argument("--sample-step", type=positive, default=0,
                        help="Uniform dense-output spacing for BDF, IMEX_ARK324L2SA or ROS2/ROS3 on supported ODEs")
    parser.add_argument("--max-steps", type=int, default=2_000_000)
    parser.add_argument("--max-relative-error", type=positive, default=.01)
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--bdf-max-order", type=int, choices=range(1, 6), default=5,
                        help="Maximum order for adaptive --methods BDF")
    parser.add_argument("--controllers", choices=(*CONTROLLERS, "constant"), nargs="+", default=list(CONTROLLERS))
    parser.add_argument("--backend", choices=("cpu", "cpu_omp", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=positive, default=300)
    args = parser.parse_args(argv)
    args.build_dir = args.build_dir.resolve()
    args.systems = list(dict.fromkeys(args.systems))
    if args.min_step > args.max_step or args.max_steps < 1 or args.threads < 1:
        parser.error("Invalid step bounds, step budget, or thread count")
    if args.initial is not None and len(args.systems) != 1:
        parser.error("--initial requires exactly one --system")
    if "constant" in args.controllers and not args.min_step <= args.step <= args.max_step:
        parser.error("Constant --step must lie within --min-step and --max-step")
    return args


def configuration(args, system):
    definition = system_definition(system)
    start = definition["start"] if args.start is None else args.start
    end = definition["end"] if args.end is None else args.end
    if not all(math.isfinite(value) for value in (start, end, end - start)) or end <= start:
        raise ValueError(f"{system}: use a finite increasing interval")
    if system == "time_mass" and start <= -1:
        raise ValueError("Time-dependent mass requires start > -1")
    if definition["exact"]:
        if args.initial is not None:
            raise ValueError("Manufactured systems use their exact consistent initial state at --start")
        initial = definition["exact"](start, math)
    else:
        initial = definition["initial"] if args.initial is None else args.initial
    if len(initial) != len(definition["labels"]) or not all(math.isfinite(value) for value in initial):
        raise ValueError("Invalid initial-state dimension or value")
    methods = [method for method in dict.fromkeys(args.methods)
               if (not method.startswith("IMEX_") or system in IMEX_SPLITS)
               and (method not in ROSENBROCK_METHODS or system in ROSENBROCK_SYSTEMS)]
    if not methods:
        raise ValueError(f"{system}: no compatible methods; IMEX excludes singular-mass DAEs; "
                         "Rosenbrock requires identity-mass ODEs")
    if any(method in FIXED_IMEX_METHODS for method in methods) and "constant" not in args.controllers:
        raise ValueError("Fixed IMEX tables require --controllers constant (no combined embedded estimator)")
    if "BDF" in methods and not any(controller != "constant" for controller in args.controllers):
        raise ValueError("Automatic BDF requires an error-requiring controller")
    if args.sample_step and (any(method not in (*BDF_METHODS, "IMEX_ARK324L2SA", *ROSENBROCK_METHODS) for method in methods)
                             or system not in IMEX_SPLITS):
        raise ValueError("Uniform dense sampling requires BDF, IMEX_ARK324L2SA or Rosenbrock on a supported ODE/nonsingular-mass system")
    if args.sample_step and (start + args.sample_step == start or end - args.sample_step == end):
        raise ValueError("Dense sample spacing cannot advance the time coordinate")
    cases = [dict(method=method, controller=controller,
                  filename=method.replace("(", "_").replace(")", "_") + "_" + controller + ".dat")
             for method in methods for controller in dict.fromkeys(args.controllers)
             if (method not in FIXED_IMEX_METHODS or controller == "constant")
             and (method != "BDF" or controller != "constant")]
    return dict(system=system, backend=args.backend, start_time=start, end_time=end,
                  initial_state=initial, parameters=definition["parameters"], labels=definition["labels"],
                  maximum_steps=args.max_steps, max_relative_error=args.max_relative_error,
                  newton_absolute_tolerance=2e-10, newton_relative_tolerance=1e-12,
                  adaptation=dict(relative_tolerance=args.rtol, absolute_tolerance=args.atol,
                                  initial_step=args.step, minimum_step=args.min_step, maximum_step=args.max_step),
                  splitting=IMEX_SPLITS.get(system),
                  skipped_methods=[method for method in dict.fromkeys(args.methods) if method not in methods],
                  sample_step=args.sample_step,
                  bdf_maximum_order=args.bdf_max_order,
                  cases=cases)


def write_if_changed(path, text):
    if not path.exists() or path.read_text() != text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def build_and_assemble(args, configurations):
    binary_dir = args.build_dir / "bin"
    options = binary_dir / f"{args.backend}_build_options.json"
    write_if_changed(options, json.dumps(dict(module=str(MODULE), environment={
        name: os.environ.get(name) for name in BUILD_ENVIRONMENT
    }), indent=2) + "\n")
    executable = binary_dir / f"ode_benchmarks_{args.backend}.bin"
    command = ["make", "--no-print-directory", "-C", str(MODULE)]
    if args.build_only:
        command.append("-B")
    command.extend([f"ode_benchmarks_{args.backend}.bin", f"BUILD_DIR={binary_dir}", f"BENCHMARK_CONFIG={options}"])
    build_log = args.build_dir / f"build_{args.backend}.log"
    with build_log.open("w") as log:
        try:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Benchmark build failed; see {build_log}") from error
    for config in configurations:
        system_dir = args.build_dir / config["system"]
        write_if_changed(system_dir / "config.json", json.dumps(config, indent=2) + "\n")
        launch = ["bash", str(HERE / "run_all.sh"), "--build-dir", str(args.build_dir),
                  "--system", config["system"], "--backend", args.backend,
                  "--start", str(config["start_time"]), "--end", str(config["end_time"]),
                  "--max-steps", str(args.max_steps), "--max-relative-error", str(args.max_relative_error),
                  "--threads", str(args.threads), "--timeout", str(args.timeout)]
        for option, key in (("--rtol", "relative_tolerance"), ("--atol", "absolute_tolerance"),
                            ("--step", "initial_step"), ("--min-step", "minimum_step"), ("--max-step", "maximum_step")):
            launch.extend([option, str(config["adaptation"][key])])
        if not system_definition(config["system"])["exact"]:
            launch.extend(["--initial", *map(str, config["initial_state"])])
        launch.extend(["--methods", *dict.fromkeys(case["method"] for case in config["cases"]),
                       "--controllers", *dict.fromkeys(case["controller"] for case in config["cases"])])
        if args.sample_step:
            launch.extend(["--sample-step", str(args.sample_step)])
        launcher = system_dir / "run.sh"
        write_if_changed(launcher, "#!/usr/bin/env bash\nset -euo pipefail\nexec " + shlex.join(launch) + ' "$@"\n')
        if not os.access(launcher, os.X_OK):
            launcher.chmod(0o755)
    return executable


def run(args, config, executable, batch):
    directory = args.build_dir / config["system"] / "runs" / batch
    # A run is immutable; assembly configurations may change without overwriting results.
    directory.mkdir(parents=True, exist_ok=False)
    config_path = directory / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC="FALSE", OPENBLAS_NUM_THREADS="1")
    for case in config["cases"]:
        path = directory / case["filename"]
        before = time.perf_counter()
        with path.with_suffix(".log").open("w") as log:
            try:
                process = subprocess.run([str(executable), str(config_path), str(path), case["method"], case["controller"]],
                                         env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                log.write("\nLocal verification timeout reached\n")
                returncode = 124
        case.update(returncode=returncode, elapsed_seconds=time.perf_counter() - before)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return directory


def read_trajectory(path, config, case, sampled=False):
    import numpy as np
    with path.open() as stream:
        first, last = stream.readline(), ""
        for line in stream:
            if line.strip():
                last = line
    if not first.startswith("# ") or not last.startswith("# "):
        raise ValueError("Missing metadata or completion footer")
    metadata, report = json.loads(first[2:]), json.loads(last[2:])
    data = np.loadtxt(path, comments="#", ndmin=2)
    if (metadata.get("format") != "nmfd.dirk_system.v1" or metadata.get("dimension") != len(config["labels"])
            or data.shape[1] != len(config["labels"]) + 2):
        raise ValueError("Unexpected trajectory format")
    if not np.isfinite(data).all() or not (np.diff(data[:, 0]) > 0).all():
        raise ValueError("Nonfinite state or nonincreasing time")
    for key in ("system", "start_time", "end_time", "initial_state", "parameters", "adaptation"):
        if metadata[key] != config[key]:
            raise ValueError(f"Inconsistent {key} metadata")
    for key in ("method", "controller"):
        if metadata[key] != case[key]:
            raise ValueError(f"Unexpected {key}")
    scheme = ("rosenbrock" if case["method"] in ROSENBROCK_METHODS else "bdf" if case["method"] in BDF_METHODS
              else "imex" if case["method"].startswith("IMEX_") else "dirk")
    if metadata.get("scheme", "dirk") != scheme:
        raise ValueError("Unexpected integration scheme")
    if case["method"] == "BDF":
        cap = config.get("bdf_maximum_order", 5)
        orders = report.get("orders_used", [])
        if (metadata.get("maximum_order") != cap or metadata.get("adaptive") is not True
                or not orders or any(type(order) is not int or not 1 <= order <= cap for order in orders)
                or orders != sorted(set(orders)) or orders[0] != 1):
            raise ValueError("Invalid automatic BDF order report")
    if scheme == "imex" and (not config.get("splitting") or metadata.get("splitting") != config["splitting"]):
        raise ValueError("Inconsistent IMEX splitting metadata")
    dense_order = 2 if case["method"] == "ROS2" else 1 if scheme == "bdf" else 3
    if sampled and (metadata.get("sampling") != "uniform_dense"
                    or metadata.get("dense_output_order") != dense_order
                    or (scheme == "bdf" and metadata.get("dense_output_order_policy") != "used_bdf_order")
                    or metadata.get("sample_step") != config.get("sample_step") or not config.get("sample_step")):
        raise ValueError("Invalid dense sampling metadata")
    if "backend" in config and metadata["backend"] != config["backend"]:
        raise ValueError("Unexpected backend")
    if (report["rows"] != len(data) or (not sampled and report["accepted_steps"] != len(data) - 1)
            or data[0, 0] != config["start_time"] or data[-1, 0] != report["final_time"]
            or not np.array_equal(data[0, 1:-1], config["initial_state"])):
        raise ValueError("Invalid row/step accounting, times, or initial state")
    if report["status"] != "completed" or data[-1, 0] != config["end_time"] or case.get("returncode", 0):
        raise ValueError("Integration did not complete; inspect its log")
    if np.max(np.abs(data[:, -1])) >= 2e-9:
        raise ValueError("Accepted constraint residual exceeds 2e-9")
    if sampled:
        grid = config["start_time"] + np.arange(len(data) - 1) * config["sample_step"]
        roundoff = 16 * np.finfo(float).eps * max(1., abs(config["start_time"]), abs(config["end_time"]))
        if (not np.allclose(data[:-1, 0], grid, rtol=0, atol=roundoff)
                or np.max(np.diff(data[:, 0])) > config["sample_step"] * (1 + 1e-10)):
            raise ValueError("Invalid uniform dense sampling grid")
    elif np.max(np.diff(data[:, 0])) > config["adaptation"]["maximum_step"] * (1 + 1e-10):
        raise ValueError("Accepted step exceeds its configured maximum")
    return data, report


def plot(directory):
    os.environ.setdefault("MPLCONFIGDIR", str(directory / ".matplotlib"))
    import numpy as np
    from scipy.integrate import solve_ivp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = json.loads((directory / "config.json").read_text())
    definition = system_definition(config["system"])
    if set(config["parameters"]) != set(definition["parameters"]):
        raise ValueError("Unexpected system parameters in saved configuration")
    definition["parameters"].update(config["parameters"])
    start, end = config["start_time"], config["end_time"]
    reference_name = "Analytical solution"
    if definition["exact"] is not None:
        reference = lambda t: np.asarray(definition["exact"](np.asarray(t), np)).reshape(len(config["labels"]), -1)
    else:
        method = definition["reference_method"]
        solution = solve_ivp(definition["rhs"], (start, end), config["initial_state"],
                             method=method, rtol=1e-11, atol=1e-13, dense_output=True)
        if not solution.success or solution.t[-1] != end:
            raise RuntimeError("Independent reference failed: " + solution.message)
        reference, reference_name = solution.sol, f"SciPy {method}, rtol=1e-11, atol=1e-13"
    times = np.linspace(start, end, 10001)
    reference_states = reference(times).T
    np.savetxt(directory / "reference.dat", np.column_stack((times, reference_states)),
               header=reference_name + "; time " + " ".join(config["labels"]))
    components, axes = plt.subplots(len(config["labels"]), 1, figsize=(12, 3 * len(config["labels"])), sharex=True, squeeze=False)
    step_figure, step_axis = plt.subplots(figsize=(12, 5))
    error_figure, error_axis = plt.subplots(figsize=(12, 5))
    constraint_figure, constraint_axis = plt.subplots(figsize=(12, 5))
    phase_figure = phase_axis = phase_3d = None
    if len(config["labels"]) == 3:
        phase_figure = plt.figure(figsize=(12, 6))
        phase_axis = phase_figure.add_subplot(121)
        phase_3d = phase_figure.add_subplot(122, projection="3d")
    elif len(config["labels"]) == 2:
        phase_figure, phase_axis = plt.subplots(figsize=(8, 6))
    window_end = min(end, start + definition["window"]) if definition["window"] else end
    summary = dict(system=config["system"], reference=reference_name, comparison_end=window_end,
                   max_relative_error=config["max_relative_error"], cases={}, failures={})
    for case in config["cases"]:
        path = directory / case["filename"]
        try:
            data, report = read_trajectory(path, config, case)
            accepted_data = data
            if config.get("sample_step"):
                data, dense_report = read_trajectory(path.with_name(path.stem + ".dense.dat"), config, case, sampled=True)
                if dense_report["accepted_steps"] != report["accepted_steps"]:
                    raise ValueError("Dense samples disagree with accepted-step accounting")
        except (OSError, ValueError, KeyError) as error:
            summary["failures"][path.stem] = str(error)
            continue
        expected = reference(data[:, 0]).T
        differences = np.abs(data[:, 1:-1] - expected)
        relative = np.max(differences / (1 + np.abs(expected)), axis=1)
        comparison = data[:, 0] <= window_end
        endpoint_expected = reference(accepted_data[:, 0]).T
        endpoint_relative = np.max(np.abs(accepted_data[:, 1:-1] - endpoint_expected) / (1 + np.abs(endpoint_expected)), axis=1)
        maximum_error = max(float(relative[comparison].max()),
                            float(endpoint_relative[accepted_data[:, 0] <= window_end].max()))
        passed = maximum_error <= config["max_relative_error"]
        summary["cases"][path.stem] = dict(report, passed=passed, elapsed_seconds=case.get("elapsed_seconds"),
            max_comparison_relative_error=maximum_error,
            dense_rows=len(data) if config.get("sample_step") else 0,
            max_component_absolute_error=differences.max(axis=0).tolist(),
            endpoint_absolute_error=differences[-1].tolist(),
            minimum_step=float(np.diff(accepted_data[:, 0]).min()), maximum_step=float(np.diff(accepted_data[:, 0]).max()))
        if not passed:
            summary["failures"][path.stem] = "Independent reference error exceeded the configured comparison bound"
        label = case["method"] + " / " + case["controller"]
        selection = np.linspace(0, len(data) - 1, min(len(data), 10000), dtype=int)
        sample = data[selection]
        for component, axis in enumerate(axes[:, 0]):
            axis.plot(sample[:, 0], sample[:, component + 1], linewidth=.7, label=label)
        step_axis.semilogy(accepted_data[1:, 0], np.diff(accepted_data[:, 0]), linewidth=.7, label=label)
        error_axis.semilogy(sample[:, 0], np.maximum(relative[selection], 1e-16), linewidth=.7, label=label)
        constraint_axis.semilogy(sample[:, 0], np.maximum(np.abs(sample[:, -1]), 1e-18), linewidth=.7, label=label)
        if phase_axis is not None:
            phase_axis.plot(sample[:, 1], sample[:, 2], linewidth=.6, label=label)
        if phase_3d is not None:
            phase_3d.plot(sample[:, 1], sample[:, 2], sample[:, 3], linewidth=.6)
    for component, axis in enumerate(axes[:, 0]):
        axis.plot(times, reference_states[:, component], "k--", linewidth=.8, label="Independent reference")
        axis.set_ylabel(config["labels"][component])
        axis.grid(alpha=.25)
    axes[-1, 0].set_xlabel("Time")
    axes[0, 0].legend(fontsize=6, ncol=3)
    components.suptitle(config["system"])
    for figure, axis, ylabel in ((step_figure, step_axis, "Accepted step"),
                                 (error_figure, error_axis, "max |error|/(1+|reference|)"),
                                 (constraint_figure, constraint_axis, "Constraint residual")):
        axis.set(xlabel="Time", ylabel=ylabel)
        axis.grid(alpha=.25)
        axis.legend(fontsize=6, ncol=3)
    error_axis.axvline(window_end, color="black", linestyle=":", linewidth=.8)
    if phase_axis is not None:
        phase_axis.plot(reference_states[:, 0], reference_states[:, 1], "k--", linewidth=.7, label="Independent reference")
        phase_axis.set(xlabel=config["labels"][0], ylabel=config["labels"][1])
        phase_axis.legend(fontsize=6, ncol=2)
    if phase_3d is not None:
        phase_3d.plot(reference_states[:, 0], reference_states[:, 1], reference_states[:, 2], "k--", linewidth=.7)
        phase_3d.set(xlabel=config["labels"][0], ylabel=config["labels"][1], zlabel=config["labels"][2])
    figures = {"components": components, "steps": step_figure, "reference_error": error_figure, "constraints": constraint_figure}
    if phase_figure is not None:
        figures["phase"] = phase_figure
    for name, figure in figures.items():
        figure.tight_layout()
        figure.savefig(directory / (name + ".png"), dpi=150, bbox_inches="tight", pad_inches=.15)
        plt.close(figure)
    success = len(summary["cases"]) == len(config["cases"]) and not summary["failures"]
    summary["passed"] = success
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{config['system']}: {'PASS' if success else 'FAIL'} ({len(summary['cases'])}/{len(config['cases'])} trajectories)")
    print(f"Results: {directory}")
    return 0 if success else 1


def check_runtime_dependencies():
    try:
        import numpy
        import scipy
        import matplotlib
    except ImportError as error:
        raise RuntimeError("Running/plotting requires NumPy, SciPy and Matplotlib; set PYTHON to an appropriate interpreter") from error


def main(argv=None):
    args = arguments(argv)
    if not args.build_only:
        cache_root = args.plot_only.resolve() if args.plot_only else args.build_dir
        os.environ.setdefault("MPLCONFIGDIR", str(cache_root / ".matplotlib"))
    if args.plot_only:
        return plot(args.plot_only.resolve())
    configurations = [configuration(args, system) for system in args.systems]
    for config in configurations:
        if config["skipped_methods"]:
            print(f"{config['system']}: DIRK only (singular mass); skipped {', '.join(config['skipped_methods'])}")
    if not args.build_only:
        check_runtime_dependencies()
    executable = build_and_assemble(args, configurations)
    if args.build_only:
        print(f"Built and assembled {len(configurations)} systems in {args.build_dir}; no benchmarks executed")
        return 0
    batch = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report = dict(passed=True, systems={})
    for config in configurations:
        directory = run(args, config, executable, batch)
        passed = plot(directory) == 0
        report["systems"][config["system"]] = dict(passed=passed, directory=str(directory))
        report["passed"] &= passed
    reports = args.build_dir / "reports"
    reports.mkdir(exist_ok=True)
    (reports / (batch + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
