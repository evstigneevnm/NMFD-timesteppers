"""Benchmark assembly/build-only contracts, plus optional scientific diagnostics."""

import contextlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import ode_benchmarks as runner


class BenchmarkAssemblyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.commands = []
        environment = mock.patch.dict(os.environ, {})
        environment.start()
        self.addCleanup(environment.stop)

    def args(self, *options):
        return runner.arguments(["--build-dir", str(self.root), *options])

    def fake_make(self, command, **kwargs):
        self.commands.append(command)
        build_dir = Path(next(value.split("=", 1)[1] for value in command if value.startswith("BUILD_DIR=")))
        target = next(value for value in command if value.endswith(".bin"))
        executable = build_dir / target
        executable.write_text("fake executable\n")
        return subprocess.CompletedProcess(command, 0)

    def test_defaults_preserve_the_twelve_system_matrix(self):
        args = self.args()
        configs = {name: runner.configuration(args, name) for name in args.systems}
        self.assertEqual(len(configs), 12)
        for name, config in configs.items():
            self.assertEqual(len(config["cases"]), 27 if name in runner.IMEX_SPLITS else 18)
        self.assertEqual(sum(len(config["cases"]) for config in configs.values()), 297)
        self.assertEqual(configs["lorenz"]["end_time"], 50)
        self.assertEqual(configs["rossler"]["end_time"], 50)
        self.assertEqual(configs["vdp"]["end_time"], 20)
        self.assertEqual(configs["vdp_stiff"]["end_time"], 10)
        self.assertEqual(configs["vdp_stiff"]["parameters"]["mu"], 1000)
        self.assertAlmostEqual(configs["moving_dae"]["initial_state"][1], math.exp(-.25) * math.sin(.25))
        self.assertEqual(configs["lorenz"]["adaptation"], dict(
            relative_tolerance=1e-5, absolute_tolerance=1e-7,
            initial_step=.001, minimum_step=1e-8, maximum_step=.05))

    def test_imex_methods_and_controllers_are_selected_without_silent_fallback(self):
        args = self.args("--systems", "lorenz", "state_mass", "--methods", "IMEX_ARK324L2SA")
        for system in args.systems:
            config = runner.configuration(args, system)
            self.assertEqual(len(config["cases"]), 9)
            self.assertTrue(config["splitting"])
            self.assertEqual(config["skipped_methods"], [])
        for method in runner.FIXED_IMEX_METHODS:
            args = self.args("--system", "coupled_mass", "--methods", method)
            with self.assertRaisesRegex(ValueError, "controllers constant"):
                runner.configuration(args, "coupled_mass")
            args = self.args("--system", "coupled_mass", "--methods", method, "IMEX_ARK324L2SA",
                             "--controllers", "constant", "scipy")
            cases = runner.configuration(args, "coupled_mass")["cases"]
            self.assertEqual([(case["method"], case["controller"]) for case in cases],
                             [(method, "constant"), ("IMEX_ARK324L2SA", "constant"), ("IMEX_ARK324L2SA", "scipy")])

    def test_singular_mass_imex_is_explicitly_excluded(self):
        args = self.args("--system", "stokes_dae")
        config = runner.configuration(args, "stokes_dae")
        self.assertEqual(config["skipped_methods"], ["IMEX_ARK324L2SA"])
        self.assertTrue(all(not case["method"].startswith("IMEX_") for case in config["cases"]))
        args = self.args("--system", "stokes_dae", "--methods", "IMEX_ARK324L2SA")
        with self.assertRaisesRegex(ValueError, "singular-mass"):
            runner.configuration(args, "stokes_dae")

    def test_bdf_supports_fixed_and_adaptive_controllers_on_all_systems(self):
        args = self.args("--methods", "BDF1", "BDF2", "SDIRK2(1)2", "--controllers", "constant", "scipy")
        for system in runner.SYSTEMS:
            config = runner.configuration(args, system)
            self.assertEqual([(case["method"], case["controller"]) for case in config["cases"]],
                             [("BDF1", "constant"), ("BDF1", "scipy"),
                              ("BDF2", "constant"), ("BDF2", "scipy"),
                              ("SDIRK2(1)2", "constant"), ("SDIRK2(1)2", "scipy")])
            self.assertEqual(config["skipped_methods"], [])
        args = self.args("--methods", "BDF2")
        self.assertEqual(len(runner.configuration(args, "lorenz")["cases"]), len(args.controllers))
        args = self.args("--methods", "BDF2", "--controllers", "constant", "--sample-step", ".01")
        self.assertEqual(runner.configuration(args, "lorenz")["sample_step"], .01)
        with self.assertRaisesRegex(ValueError, "dense sampling"):
            runner.configuration(args, "coupled_dae")

    def test_bdf_uniform_sampling_preserves_configuration(self):
        args = self.args("--methods", "BDF1", "BDF2", "BDF", "--controllers", "constant", "scipy",
                         "--sample-step", ".017")
        for system in runner.IMEX_SPLITS:
            config = runner.configuration(args, system)
            self.assertEqual(config["sample_step"], .017)
            self.assertEqual(len(config["cases"]), 5)

    def test_rosenbrock_selects_only_identity_mass_systems_and_supports_dense_sampling(self):
        args = self.args("--methods", "ROS2", "ROS3", "--controllers", "constant", "scipy", "--sample-step", ".017")
        for system in runner.ROSENBROCK_SYSTEMS:
            config = runner.configuration(args, system)
            self.assertEqual(config["sample_step"], .017)
            self.assertEqual(len(config["cases"]), 4)
            self.assertEqual(config["skipped_methods"], [])
        for system in set(runner.SYSTEMS) - set(runner.ROSENBROCK_SYSTEMS):
            with self.assertRaisesRegex(ValueError, "identity-mass"):
                runner.configuration(args, system)
        mixed = self.args("--methods", "BDF2", "ROS3")
        config = runner.configuration(mixed, "state_mass")
        self.assertEqual(config["skipped_methods"], ["ROS3"])
        self.assertTrue(all(case["method"] == "BDF2" for case in config["cases"]))

    def test_constant_step_respects_configured_bounds(self):
        for step in ("1e-10", "1"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args("--controllers", "constant", "--step", step)

    def test_automatic_bdf_order_cap_and_controller_selection(self):
        args = self.args("--methods", "BDF", "--controllers", "constant", "scipy", "--bdf-max-order", "3")
        for system in runner.SYSTEMS:
            config = runner.configuration(args, system)
            self.assertEqual(config["bdf_maximum_order"], 3)
            self.assertEqual([(case["method"], case["controller"]) for case in config["cases"]], [("BDF", "scipy")])
        with self.assertRaisesRegex(ValueError, "error-requiring"):
            runner.configuration(self.args("--methods", "BDF", "--controllers", "constant"), "lorenz")
        for cap in ("0", "6"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args("--methods", "BDF", "--bdf-max-order", cap)

    def test_build_only_forces_make_without_scientific_packages_or_execution(self):
        with mock.patch.object(runner.subprocess, "run", side_effect=self.fake_make), \
             mock.patch.object(runner, "run") as run, mock.patch.object(runner, "plot") as plot, \
             mock.patch.object(runner, "check_runtime_dependencies") as dependencies, \
             mock.patch.dict("sys.modules", {"numpy": None, "scipy": None, "matplotlib": None}), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(["-B", "--build-dir", str(self.root)]), 0)
        self.assertEqual(len(self.commands), 1)
        self.assertIn("-B", self.commands[0])
        run.assert_not_called()
        plot.assert_not_called()
        dependencies.assert_not_called()
        for system in runner.SYSTEMS:
            self.assertTrue((self.root / system / "config.json").is_file())
            self.assertTrue(os.access(self.root / system / "run.sh", os.X_OK))
            self.assertFalse((self.root / system / "runs").exists())
        self.assertFalse(list(self.root.rglob("*.dat")))
        self.assertFalse(list(self.root.rglob("*.png")))

    def test_unchanged_assembly_is_not_rewritten_and_launchers_keep_parameters(self):
        args = self.args("--system", "lorenz", "--end", "3", "--controllers", "H211", "PID")
        configs = [runner.configuration(args, "lorenz")]
        with mock.patch.object(runner.subprocess, "run", side_effect=self.fake_make):
            runner.build_and_assemble(args, configs)
            paths = [self.root / "lorenz/config.json", self.root / "lorenz/run.sh",
                     self.root / "bin/cpu_build_options.json"]
            before = [(path.stat().st_mtime_ns, path.read_bytes()) for path in paths]
            runner.build_and_assemble(args, configs)
        self.assertEqual(before, [(path.stat().st_mtime_ns, path.read_bytes()) for path in paths])
        self.assertNotIn("-B", self.commands[-1])
        launcher = paths[1].read_text()
        self.assertIn("--end 3.0", launcher)
        self.assertIn("--controllers H211 PID", launcher)
        self.assertIn(str(runner.HERE / "run_all.sh"), launcher)

    def test_changed_build_settings_update_the_make_prerequisite(self):
        args = self.args("--system", "lorenz")
        configs = [runner.configuration(args, "lorenz")]
        with mock.patch.object(runner.subprocess, "run", side_effect=self.fake_make):
            runner.build_and_assemble(args, configs)
            options = self.root / "bin/cpu_build_options.json"
            before = options.read_bytes()
            with mock.patch.dict(os.environ, {"CXXFLAGS": "-O0 -g -DBENCHMARK_SETTINGS_TEST"}):
                runner.build_and_assemble(args, configs)
        self.assertNotEqual(before, options.read_bytes())
        self.assertEqual(json.loads(options.read_text())["environment"]["CXXFLAGS"], "-O0 -g -DBENCHMARK_SETTINGS_TEST")
        self.assertIn(f"BENCHMARK_CONFIG={options}", self.commands[-1])

    def test_dae_launcher_preserves_its_filtered_method_selection(self):
        args = self.args("--system", "stokes_dae")
        with mock.patch.object(runner.subprocess, "run", side_effect=self.fake_make):
            runner.build_and_assemble(args, [runner.configuration(args, "stokes_dae")])
        launcher = (self.root / "stokes_dae/run.sh").read_text()
        self.assertNotIn("IMEX_", launcher)
        self.assertIn("SDIRK2", launcher)

    def test_uniform_dense_sampling_is_opt_in_and_preserved_in_launchers(self):
        args = self.args("--system", "lorenz", "--methods", "IMEX_ARK324L2SA", "--sample-step", ".017")
        config = runner.configuration(args, "lorenz")
        self.assertEqual(config["sample_step"], .017)
        with mock.patch.object(runner.subprocess, "run", side_effect=self.fake_make):
            runner.build_and_assemble(args, [config])
        self.assertIn("--sample-step 0.017", (self.root / "lorenz/run.sh").read_text())
        for options in (("--system", "lorenz", "--sample-step", ".01"),
                        ("--system", "stokes_dae", "--sample-step", ".01"),
                        ("--system", "lorenz", "--methods", "IMEX_ARK324L2SA", "--sample-step", ".001",
                         "--start", "1e20", "--end", "1.000000000001e20")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                args = self.args(*options)
                runner.configuration(args, args.systems[0])

    def test_failed_build_does_not_assemble_or_execute(self):
        args = self.args("-B")
        with mock.patch.object(runner.subprocess, "run", side_effect=subprocess.CalledProcessError(2, "make")):
            with self.assertRaisesRegex(RuntimeError, "build_cpu.log"):
                runner.build_and_assemble(args, [runner.configuration(args, "lorenz")])
        self.assertFalse((self.root / "lorenz").exists())

    def test_normal_run_builds_once_and_runs_each_selected_system(self):
        with mock.patch.object(runner.subprocess, "run", side_effect=self.fake_make), \
             mock.patch.object(runner, "check_runtime_dependencies"), \
             mock.patch.object(runner, "run", return_value=self.root) as run, \
             mock.patch.object(runner, "plot", side_effect=[0, 1]) as plot:
            self.assertEqual(runner.main(["--build-dir", str(self.root), "--systems", "lorenz", "rossler"]), 1)
        self.assertEqual(len(self.commands), 1)
        self.assertNotIn("-B", self.commands[0])
        self.assertEqual(run.call_count, 2)
        self.assertEqual(plot.call_count, 2)
        reports = list((self.root / "reports").glob("*.json"))
        self.assertEqual(len(reports), 1)
        self.assertFalse(json.loads(reports[0].read_text())["passed"])

    def test_timeout_is_recorded_and_a_previous_run_cannot_be_overwritten(self):
        args = self.args("--system", "lorenz", "--controllers", "PID", "--methods", "SDIRK2(1)2")
        config = runner.configuration(args, "lorenz")
        with mock.patch.object(runner.subprocess, "run", side_effect=subprocess.TimeoutExpired("solver", 1)):
            directory = runner.run(args, config, Path("unused.bin"), "fixed-batch")
        saved = json.loads((directory / "config.json").read_text())
        self.assertEqual(saved["cases"][0]["returncode"], 124)
        original = (directory / "config.json").read_bytes()
        with self.assertRaises(FileExistsError):
            runner.run(args, config, Path("unused.bin"), "fixed-batch")
        self.assertEqual((directory / "config.json").read_bytes(), original)

    def test_invalid_intervals_and_inconsistent_manufactured_initial_states(self):
        for options in (("--system", "time_mass", "--start", "-1"),
                        ("--system", "lorenz", "--end", "0"),
                        ("--system", "stokes_dae", "--initial", "1", "1", "1")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                args = self.args(*options)
                runner.configuration(args, args.systems[0])

    def test_make_reuses_binaries_and_tracks_local_and_dependency_headers(self):
        source = self.root / "tests/ode_benchmarks.cpp"
        header = self.root / "tests/equation.h"
        options = self.root / "options.json"
        dependencies = {
            "SCFD_ROOT": "scfd/include/scfd/backend/serial_cpu.h",
            "NMFD_OPERATIONS_ROOT": "operations/include/nmfd/detail/vector_wrap.h",
            "NMFD_NEWTON_ROOT": "newton/include/nmfd/solvers/nonlinear_solver.h",
            "NMFD_LINSOLVERS_ROOT": "linsolvers/include/nmfd/solvers/gmres.h",
        }
        for path in (source, header, options, *(self.root / value for value in dependencies.values())):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n")
        (self.root / "Makefile").write_text((runner.MODULE / "Makefile").read_text())
        external = self.root / dependencies["SCFD_ROOT"]
        counter = self.root / "compiler_calls"
        compiler = self.root / "compiler.py"
        compiler.write_text(
            "import pathlib, sys\n"
            "args = sys.argv[1:]\n"
            "assert '-MMD' in args and '-MP' in args\n"
            f"counter = pathlib.Path({str(counter)!r})\n"
            "count = int(counter.read_text()) + 1 if counter.exists() else 1\n"
            "counter.write_text(str(count))\n"
            "target = args[args.index('-o') + 1]\n"
            "pathlib.Path(target).write_text(str(count))\n"
            f"dependencies = {[str(source), str(header), str(external)]!r}\n"
            "pathlib.Path(args[args.index('-MF') + 1]).write_text(target + ': ' + ' '.join(dependencies) + '\\n')\n"
        )
        command = ["make", "--no-print-directory", "ode_benchmarks_cpu.bin",
                   f"BUILD_DIR={self.root / 'bin'}", f"BENCHMARK_CONFIG={options}",
                   "CXX=" + shlex.join([sys.executable, str(compiler)])]
        command.extend(f"{name}={self.root / value.split('/')[0]}" for name, value in dependencies.items())

        def build(force=False):
            subprocess.run([*command, *(["-B"] if force else [])], cwd=self.root,
                           check=True, capture_output=True, text=True, timeout=30)
            return int(counter.read_text())

        self.assertEqual(build(), 1)
        self.assertEqual(build(), 1)
        for expected, path in enumerate((source, header, external, options), start=2):
            path.write_text(path.read_text() + "changed\n")
            self.assertEqual(build(), expected)
            self.assertEqual(build(), expected)
        self.assertEqual(build(force=True), 6)


SCIENTIFIC = all(importlib.util.find_spec(name) is not None for name in ("numpy", "scipy", "matplotlib"))


@unittest.skipUnless(SCIENTIFIC, "Scientific packages are needed only for trajectory/plot diagnostics")
class BenchmarkDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = mock.patch.dict(os.environ, {"MPLCONFIGDIR": str(self.root / ".matplotlib")})
        environment.start()
        self.addCleanup(environment.stop)
        self.path = self.root / "trajectory.dat"
        self.case = dict(method="SDIRK2(1)2", controller="PID", filename=self.path.name)
        self.config = dict(system="coupled_mass", start_time=0, end_time=.2, initial_state=[1, 1],
                           parameters={}, labels=["x", "z"], adaptation=dict(maximum_step=.1),
                           max_relative_error=.01, cases=[self.case])
        self.metadata = dict(self.config, format="nmfd.dirk_system.v1", dimension=2, **self.case)
        self.report = dict(status="completed", final_time=.2, accepted_steps=2, rows=3)
        times = np.array([0, .1, .2])
        self.data = np.column_stack((times, np.exp(-times), np.exp(-2 * times), times * 0))
        self.save()

    def save(self):
        with self.path.open("w") as stream:
            stream.write("# " + json.dumps(self.metadata) + "\n")
            self.np.savetxt(stream, self.data, fmt="%.17g")
            stream.write("# " + json.dumps(self.report) + "\n")

    def test_valid_trajectory(self):
        data, report = runner.read_trajectory(self.path, self.config, self.case)
        self.np.testing.assert_array_equal(data, self.data)
        self.assertEqual(report["accepted_steps"], 2)

    def test_imex_trajectory_requires_correct_scheme_and_partition_metadata(self):
        self.case["method"] = "IMEX_ARK324L2SA"
        self.config["splitting"] = runner.IMEX_SPLITS["coupled_mass"]
        self.metadata.update(method=self.case["method"], scheme="imex", splitting=self.config["splitting"])
        self.save()
        runner.read_trajectory(self.path, self.config, self.case)
        self.metadata["splitting"] = "Wrong partition"
        self.save()
        with self.assertRaisesRegex(ValueError, "splitting"):
            runner.read_trajectory(self.path, self.config, self.case)
        self.metadata["scheme"] = "dirk"
        self.save()
        with self.assertRaisesRegex(ValueError, "scheme"):
            runner.read_trajectory(self.path, self.config, self.case)

    def test_bdf_trajectory_has_its_own_scheme(self):
        self.case.update(method="BDF2", controller="constant")
        self.metadata.update(method="BDF2", controller="constant", scheme="bdf", dense_output_order=0)
        self.save()
        runner.read_trajectory(self.path, self.config, self.case)
        self.metadata["scheme"] = "dirk"
        self.save()
        with self.assertRaisesRegex(ValueError, "scheme"):
            runner.read_trajectory(self.path, self.config, self.case)

    def test_rosenbrock_dense_metadata_uses_the_selected_continuous_order(self):
        for method, order in (("ROS2", 2), ("ROS3", 3)):
            self.case["method"] = method
            self.config["sample_step"] = .1
            self.metadata.update(method=method, scheme="rosenbrock", sampling="uniform_dense", sample_step=.1,
                                 dense_output_order=order)
            self.save()
            runner.read_trajectory(self.path, self.config, self.case, sampled=True)
            self.metadata["dense_output_order"] = 1
            self.save()
            with self.assertRaisesRegex(ValueError, "dense sampling"):
                runner.read_trajectory(self.path, self.config, self.case, sampled=True)

    def make_dense_samples(self, bdf=False):
        self.case["method"] = "BDF2" if bdf else "IMEX_ARK324L2SA"
        self.config.update(sample_step=.05, splitting=runner.IMEX_SPLITS["coupled_mass"])
        self.metadata.update(method=self.case["method"], scheme="bdf" if bdf else "imex", splitting=self.config["splitting"],
                             sample_step=.05, dense_output_order=1 if bdf else 3)
        if bdf:
            self.metadata["dense_output_order_policy"] = "used_bdf_order"
        self.save()
        path = self.path.with_name(self.path.stem + ".dense.dat")
        metadata = dict(self.metadata, sampling="uniform_dense")
        times = self.np.array([0, .05, .1, .15, .2])
        data = self.np.column_stack((times, self.np.exp(-times), self.np.exp(-2 * times), times * 0))
        with path.open("w") as stream:
            stream.write("# " + json.dumps(metadata) + "\n")
            self.np.savetxt(stream, data, fmt="%.17g")
            stream.write("# " + json.dumps(dict(self.report, rows=len(data))) + "\n")
        return path, data

    def test_automatic_bdf_validates_committed_order_report(self):
        self.case["method"] = "BDF"
        self.config["bdf_maximum_order"] = 3
        self.metadata.update(method="BDF", scheme="bdf", maximum_order=3, adaptive=True)
        self.report["orders_used"] = [1, 2, 3]
        self.save()
        runner.read_trajectory(self.path, self.config, self.case)
        for invalid in ([1, 4], [], [2, 3], [1, 2, 2], [True, 2]):
            self.report["orders_used"] = invalid
            self.save()
            with self.assertRaisesRegex(ValueError, "order report"):
                runner.read_trajectory(self.path, self.config, self.case)

    def test_uniform_dense_samples_keep_accepted_step_accounting(self):
        path, expected = self.make_dense_samples()
        data, report = runner.read_trajectory(path, self.config, self.case, sampled=True)
        self.np.testing.assert_array_equal(data, expected)
        self.assertEqual(report["accepted_steps"], 2)
        with self.assertRaises(ValueError):
            runner.read_trajectory(path, self.config, self.case)
        self.config["sample_step"] = .04
        with self.assertRaises(ValueError):
            runner.read_trajectory(path, self.config, self.case, sampled=True)

    def test_bdf_dense_samples_use_actual_order_policy(self):
        path, expected = self.make_dense_samples(bdf=True)
        data, report = runner.read_trajectory(path, self.config, self.case, sampled=True)
        self.np.testing.assert_array_equal(data, expected)
        self.assertEqual(report["accepted_steps"], 2)
        text = path.read_text().replace('"used_bdf_order"', '"fixed_fifth_order"')
        path.write_text(text)
        with self.assertRaisesRegex(ValueError, "dense sampling metadata"):
            runner.read_trajectory(path, self.config, self.case, sampled=True)

    def test_plot_validates_dense_interior_and_preserves_step_statistics(self):
        path, data = self.make_dense_samples()
        (self.root / "config.json").write_text(json.dumps(self.config))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.plot(self.root), 0)
        summary = json.loads((self.root / "summary.json").read_text())
        self.assertEqual(summary["cases"][self.path.stem]["dense_rows"], 5)
        self.assertEqual(summary["cases"][self.path.stem]["minimum_step"], .1)
        lines = path.read_text().splitlines()
        data[1, 1] += .5
        with path.open("w") as stream:
            stream.write(lines[0] + "\n")
            self.np.savetxt(stream, data, fmt="%.17g")
            stream.write(lines[-1] + "\n")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.plot(self.root), 1)

    def test_failed_run_is_rejected(self):
        self.report["status"] = "failed"
        self.save()
        with self.assertRaises(ValueError):
            runner.read_trajectory(self.path, self.config, self.case)

    def test_nonfinite_states_bad_times_and_constraints_are_rejected(self):
        original = self.data.copy()
        for row, column, value in ((1, 1, self.np.nan), (1, 0, 0), (1, 3, 1e-7)):
            with self.subTest(row=row, column=column):
                self.data = original.copy()
                self.data[row, column] = value
                self.save()
                with self.assertRaises(ValueError):
                    runner.read_trajectory(self.path, self.config, self.case)

    def test_wrong_metadata_or_missing_footer_is_rejected(self):
        self.metadata["controller"] = "H211"
        self.save()
        with self.assertRaises(ValueError):
            runner.read_trajectory(self.path, self.config, self.case)
        self.path.write_text("# " + json.dumps(self.metadata) + "\n0 1 1 0\n")
        with self.assertRaises(ValueError):
            runner.read_trajectory(self.path, self.config, self.case)

    def test_reference_failure_produces_a_failed_report(self):
        self.data[1, 1] += .5
        self.save()
        (self.root / "config.json").write_text(json.dumps(self.config))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.plot(self.root), 1)
        summary = json.loads((self.root / "summary.json").read_text())
        self.assertFalse(summary["passed"])
        self.assertTrue(summary["failures"])
        self.assertTrue((self.root / "components.png").is_file())


if __name__ == "__main__":
    unittest.main()
