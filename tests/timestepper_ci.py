"""Registered timestepper CI cases. The normal run_all.sh invocation remains the ODE benchmark."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
BACKENDS = {"SERIAL": "cpu", "OMP": "cpu_omp", "CUDA": "cuda", "HIP": "hip"}
DIRK_GROUPS = {
    "dirk-core": ("dirk", "adaptive", "mass_dae", "lifecycle", "nonautonomous", "lorenz", "rossler", "vdp"),
    "dirk-extensions": ("esdirk", "esdirk_adaptive", "esdirk_mass_dae", "dense_output", "dense_lifecycle",
                        "dense_mass_dae", "globalization", "nonnormal_steady_state", "lorenz_steady_state",
                        "rossler_steady_state", "vdp_steady_state"),
}
IMPLICIT_GROUPS = {
    "imex": ("imex", "imex_adaptive", "imex_mass", "imex_lifecycle", "imex_backend", "imex_benchmarks",
             "imex_dense_output", "imex_dense_lifecycle"),
    "bdf": ("bdf", "bdf_lifecycle", "bdf_mass_dae", "bdf_backend", "bdf_adaptive", "bdf_adaptive_lifecycle",
            "bdf_adaptive_mass_dae", "bdf_orders", "bdf_variable_order", "bdf_stability", "bdf_dense",
            "bdf_dense_lifecycle", "bdf_dense_mass_dae"),
    "rosenbrock": (
        "rosenbrock", "rosenbrock_lifecycle", "rosenbrock_backend", "rosenbrock_adaptive", "rosenbrock_nonautonomous",
        "rosenbrock_ros3", "rosenbrock_ros3_adaptive", "rosenbrock_ros3_lifecycle", "rosenbrock_dense",
        "rosenbrock_dense_lifecycle", "rosenbrock_w", "rosenbrock_w_adaptive", "rosenbrock_w_lifecycle",
        "rosenbrock_w_dense", "rosenbrock_w_dense_lifecycle", "rosenbrock_mass", "rosenbrock_mass_adaptive",
        "rosenbrock_mass_lifecycle", "rosenbrock_mass_nonautonomous", "rosenbrock_mass_time_lifecycle",
        "rosenbrock_dae", "rosenbrock_dae_lifecycle", "rosenbrock_variable_mass", "rosenbrock_variable_mass_adaptive",
        "rosenbrock_variable_mass_lifecycle", "rosenbrock_variable_mass_benchmarks", "rosenbrock_index2",
        "rosenbrock_index2_moving", "rosenbrock_index2_dense", "rosenbrock_index2_lifecycle",
    ),
}
PROJECTION_FAMILIES = ("explicit", "implicit", "implicit_imex", "implicit_bdf", "implicit_rosenbrock")
GROUPS = ("explicit-rk", "dop853-reference", "lorenz-reference", "rk4-reference", *DIRK_GROUPS, *IMPLICIT_GROUPS,
          "projection")


@dataclass(frozen=True)
class Case:
    name: str
    target: str
    args: tuple = ()
    threads: int = 8
    timeout: int = 300
    compare: tuple = ()
    expected_exit: int = 0
    expected_message: str = ""


def projection_case(platform, name):
    return Case(name, f"test_{name}_{BACKENDS[platform]}.bin", threads=2,
                timeout=300 if platform in ("CUDA", "HIP") else 180)


def cases(platform, group):
    backend = BACKENDS[platform]
    cuda = platform == "CUDA"
    if group == "projection":
        for name in ("explicit_apply_operator", "implicit_operator_projection"):
            yield projection_case(platform, name)
        for family in PROJECTION_FAMILIES:
            for suffix in ("projection", "projection_lifecycle", "projection_backend"):
                yield projection_case(platform, f"{family}_{suffix}")
        return
    if group in IMPLICIT_GROUPS:
        for name in IMPLICIT_GROUPS[group]:
            yield Case(name, f"test_implicit_{name}_{backend}.bin", threads=2,
                       timeout=300 if platform in ("CUDA", "HIP") else 180)
        return
    if platform == "HIP":
        raise ValueError(f"HIP is registered only for: {', '.join(IMPLICIT_GROUPS)}, projection")
    if group in DIRK_GROUPS:
        if group == "dirk-core" and platform == "SERIAL":
            yield Case("table", "test_rk_tables_implicit.bin", threads=2)
        for name in DIRK_GROUPS[group]:
            longer = name in ("adaptive", "nonautonomous") or name.startswith(("esdirk", "dense_"))
            cpu_timeout = 180 if longer else 120
            yield Case(name, f"test_implicit_{name}_{backend}.bin", threads=2, timeout=300 if cuda else cpu_timeout)
        return
    def target(name):
        return f"test_explicit_{name}_{backend}.bin"

    simulation = target("lorenz_simulation")
    if group == "explicit-rk":
        if platform == "SERIAL":
            yield Case("table", "test_rk_tables_explicit.bin")
        for name in ("rk", "ownership", "fsal", "scipy_adaptation", "filter_adaptation", "dop853",
                     "dop853_dense_output", "dop853_adaptive", "lorenz", "rossler", "vdp", "nonautonomous",
                     "dense_output", "lorenz_dense_output", "rossler_dense_output", "vdp_dense_output"):
            threads = {"ownership": 4, "nonautonomous": 2}.get(name, 8)
            budget = {"ownership": (60 if cuda else 30), "filter_adaptation": (300 if cuda else 90),
                      "nonautonomous": (300 if cuda else 180)}.get(name, 300)
            yield Case(name, target(name), threads=threads, timeout=budget)
        if cuda:
            yield Case("dop853_simulation", simulation, ("1", "lorenz_DOP853_scipy.dat", "DOP853", "scipy"))
        return
    if cuda:
        raise ValueError("The reference comparisons are registered only for SERIAL and OMP")
    if group == "dop853-reference":
        samples = f"samples_{backend}.txt"
        yield Case("dop853_scipy", target("dop853_dense_output"), (samples,),
                   compare=("compare_explicit_dop853_dense_output.py", samples))
    elif group == "lorenz-reference":
        for method, controller in (("RK45", "matlab"), ("RK45", "scipy"), ("RK23", "matlab"),
                                   ("RK23", "scipy"), ("DOP853", "scipy")):
            trajectory = f"lorenz_{backend}_{method}_{controller}.dat"
            yield Case(f"{method}_{controller}", simulation, ("1", trajectory, method, controller),
                       compare=("compare_explicit_lorenz_simulation.py", trajectory))
    elif group == "rk4-reference":
        trajectory = f"lorenz_{backend}_RK4.dat"
        yield Case("rk4_numpy", simulation, ("0.1003", trajectory, "RK4", "constant", "0.0005"),
                   compare=("compare_explicit_lorenz_simulation.py", trajectory))
        yield Case("rk4_rejects_scipy", simulation, ("0.1", trajectory+".invalid", "RK4", "scipy"),
                   expected_exit=1, expected_message="This simulation uses constant stepping only for RK4")


def run_profile(argv, groups, select_cases, make_dir, probe_target=None):
    """Run a directory's registered cases with the same build, probe and reporting policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=groups, default=groups[0])
    parser.add_argument("--list", action="store_true", help="List selected cases without building or probing")
    parser.add_argument("-B", action="store_true", dest="build_only", help="Force build only")
    parser.add_argument("--build-jobs", type=int, choices=(1, 2), default=1,
                        help="Compile up to two CPU targets concurrently; numerical runs remain sequential")
    parser.add_argument("--build-dir", type=Path, help="Build and report root (default depends on the method family)")
    parser.add_argument("--make-dir", type=Path, default=make_dir, help="Makefile providing the binary targets")
    parser.add_argument("--make-option", action="append", default=[], help="Pass one variable assignment to Make")
    parser.add_argument("--device-probe-target", default=probe_target, help="Integration Make target for required CUDA/HIP preflight")
    args = parser.parse_args(argv)
    platform = os.environ.get("PLATFORM", "SERIAL")
    if platform not in BACKENDS:
        parser.error(f"Unsupported PLATFORM={platform!r}; choose from {', '.join(BACKENDS)}")
    if args.build_jobs > 1 and platform not in ("SERIAL", "OMP"):
        parser.error("Concurrent builds are supported on CPU only")
    try:
        selected = list(select_cases(platform, args.group))
    except ValueError as error:
        parser.error(str(error))
    if args.list:
        for case in selected:
            print(case.name, case.target, *case.args)
        return 0

    if platform == "HIP" and args.make_dir.resolve() == HERE.parent:
        parser.error("The standalone Makefile has no HIP targets; use the parent make test-timesteppers or --make-dir")
    family = ("dirk" if args.group in DIRK_GROUPS else
              args.group if args.group in IMPLICIT_GROUPS or args.group == "projection" else "explicit_rk")
    default_dir = f"{family}_ci"
    root = args.build_dir or Path("build")/default_dir
    directory = root.resolve()/platform/args.group
    binaries = directory/"bin"
    binaries.mkdir(parents=True, exist_ok=True)
    make = ["make", "--no-print-directory", "-C", str(args.make_dir.resolve()), *args.make_option]
    if args.build_only:
        make.append("-B")
    if platform == "CUDA":
        if args.group == "explicit-rk":
            make.append("REAL_TYPE=double")
        if os.environ.get("NVCC"):
            make.append("NVCC=" + os.environ["NVCC"])
    def make_target(target, bindir):
        return [*make, f"BUILD_DIR={bindir}", target]
    report = dict(platform=platform, group=args.group, build_only=args.build_only, status="running", passed=False,
                  selected=[case.name for case in selected], results={})
    def save():
        (directory/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    save()
    gpu = platform in ("CUDA", "HIP")
    if gpu and args.device_probe_target and not args.build_only:
        with (directory/"probe.log").open("w") as log:
            try:
                subprocess.run(make_target(args.device_probe_target, binaries), stdout=log, stderr=subprocess.STDOUT,
                               timeout=120, check=True)
            except (OSError, subprocess.SubprocessError) as error:
                report["status"] = "failed"
                report["error"] = f"{platform} probe failed: {error}"
                save()
                print(f"{report['error']}; see {directory}/probe.log", file=sys.stderr)
                return 1

    def invoke(command, log, env, timeout, expected=0):
        print("+", *command, file=log, flush=True)
        rc = subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                            env=env, timeout=timeout).returncode
        if rc != expected:
            raise RuntimeError(f"expected exit {expected}, got {rc}")

    def compile_ahead(case, bindir, env):
        # Parent Make recipes also write results.make. Isolate simultaneous targets.
        bindir.mkdir(exist_ok=True)
        start = time.monotonic()
        with (directory/(case.name+".build.log")).open("w") as log:
            try:
                invoke(make_target(case.target, bindir), log, env, 300)
            except (OSError, RuntimeError, subprocess.SubprocessError):
                if (bindir/"results.make").exists():
                    print((bindir/"results.make").read_text(errors="replace"), file=log)
                raise
        return time.monotonic()-start

    def execute(case, bindir, env, built, future):
        start = time.monotonic()
        path = directory/(case.name+".log")
        compilation = 0
        with path.open("w") as log:
            def run(command, timeout, expected=0):
                invoke(command, log, env, timeout, expected)
            try:
                if case.target not in built:
                    if future is None:
                        run(make_target(case.target, bindir), 300)
                    else:
                        compilation = future.result()
                        start = time.monotonic()
                    built.add(case.target)
                if not args.build_only:
                    run([str(bindir/case.target), *case.args], case.timeout, case.expected_exit)
                    if case.expected_message and case.expected_message not in path.read_text(errors="replace"):
                        raise RuntimeError(f"missing expected diagnostic: {case.expected_message}")
                    if case.compare:
                        run([sys.executable, str(HERE/case.compare[0]), *case.compare[1:]], 180)
                status = "built" if args.build_only else "passed"
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                print(error, file=log)
                if future is None and case.target not in built and (bindir/"results.make").exists():
                    print((bindir/"results.make").read_text(errors="replace"), file=log)
                status = "failed"
        elapsed = round(compilation+time.monotonic()-start, 2)
        report["results"][case.name] = dict(status=status, seconds=elapsed)
        save()
        print(f"{platform}/{args.group}/{case.name}: {status} ({elapsed}s, {path})", flush=True)

    def environment(case):
        return os.environ if gpu else dict(os.environ, OMP_NUM_THREADS=str(case.threads))

    built, builds = set(), {}
    with (ThreadPoolExecutor(args.build_jobs) if args.build_jobs > 1 else nullcontext()) as pool:
        for case in selected:
            if case.target not in builds:
                bindir = binaries/case.name if pool else binaries
                future = pool.submit(compile_ahead, case, bindir, environment(case)) if pool else None
                builds[case.target] = bindir, future
        for case in selected:
            bindir, future = builds[case.target]
            execute(case, bindir, environment(case), built, future)
    success = all(row["status"] != "failed" for row in report["results"].values())
    report["status"] = ("built" if args.build_only else "passed") if success else "failed"
    report["passed"] = report["status"] == "passed"
    save()
    return 0 if success else 1


def main(argv=None):
    return run_profile(argv, GROUPS, cases, HERE.parent)


if __name__ == "__main__":
    sys.exit(main())
