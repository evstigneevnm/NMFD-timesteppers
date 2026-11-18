"""RK coverage inventory and suite lifecycle; no compiler or GPU required."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import rk_ci as ci


class RungeKuttaSuiteTest(unittest.TestCase):
    def test_dirk_matrix_and_execution_budgets(self):
        for platform in ("SERIAL", "OMP", "CUDA"):
            backend = ci.BACKENDS[platform]
            selected = [c for group in ("dirk-core", "dirk-extensions") for c in ci.cases(platform, group)]
            ordinary = ("dirk mass_dae lifecycle lorenz rossler vdp globalization nonnormal_steady_state "
                        "lorenz_steady_state rossler_steady_state vdp_steady_state").split()
            extended = ("adaptive nonautonomous esdirk esdirk_adaptive esdirk_mass_dae dense_output "
                        "dense_lifecycle dense_mass_dae").split()
            expected = [(f"test_implicit_{name}_{backend}.bin", 2, 300 if platform == "CUDA" else budget)
                        for names, budget in ((ordinary, 120), (extended, 180)) for name in names]
            if platform == "SERIAL":
                expected.append(("test_rk_tables_implicit.bin", 2, 300))
            self.assertCountEqual([(c.target, c.threads, c.timeout) for c in selected], expected)

    def test_imex_inventory_and_platform_limits(self):
        names = ("imex", "imex_adaptive", "imex_mass", "imex_lifecycle", "imex_backend", "imex_benchmarks",
                 "imex_dense_output", "imex_dense_lifecycle")
        for platform, suffix in (("SERIAL", "cpu"), ("OMP", "cpu_omp"), ("CUDA", "cuda"), ("HIP", "hip")):
            timeout = 180 if platform in ("SERIAL", "OMP") else 300
            selected = [(c.target, c.timeout, c.threads) for c in ci.cases(platform, "imex")]
            self.assertCountEqual(selected, [(f"test_implicit_{n}_{suffix}.bin", timeout, 2) for n in names])
        for group in set(ci.GROUPS) - {"imex"}:
            with self.assertRaises(ValueError):
                list(ci.cases("HIP", group))

    def test_hip_probe_failure_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, PLATFORM="HIP"), \
                patch.object(subprocess, "run", side_effect=subprocess.CalledProcessError(1, "probe")) as run, \
                contextlib.redirect_stderr(io.StringIO()):
            args = ["--group", "imex", "--make-dir", tmp, "--build-dir", tmp, "--device-probe-target", "probe"]
            self.assertEqual(ci.main(args), 1)
            run.assert_called_once()
            self.assertEqual(run.call_args.args[0][-1], "probe")
            report = json.loads((Path(tmp)/"HIP/imex/summary.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIn("HIP probe failed", report["error"])
            self.assertFalse(report["results"])

    def test_execution_inventory(self):
        common = set(("rk ownership fsal scipy_adaptation filter_adaptation dop853 dop853_dense_output "
                      "dop853_adaptive lorenz rossler vdp nonautonomous dense_output lorenz_dense_output "
                      "rossler_dense_output vdp_dense_output").split())
        for platform, backend in (("SERIAL", "cpu"), ("OMP", "cpu_omp"), ("CUDA", "cuda")):
            cases = list(ci.cases(platform, "explicit-rk"))
            targets = {f"test_explicit_{test}_{backend}.bin" for test in common}
            if platform == "SERIAL":
                targets.add("test_rk_tables_explicit.bin")
            if platform == "CUDA":
                targets.add("test_explicit_lorenz_simulation_cuda.bin")
                self.assertEqual(cases[-1].args, ("1", "lorenz_DOP853_scipy.dat", "DOP853", "scipy"))
            self.assertCountEqual([c.target for c in cases], targets)
            self.assertEqual(len({c.name for c in cases}), len(cases))
            by_name = {c.name: c for c in cases}
            self.assertEqual(by_name["ownership"].timeout, 60 if platform == "CUDA" else 30)
            self.assertEqual(by_name["nonautonomous"].timeout, 300 if platform == "CUDA" else 180)
            self.assertEqual(by_name["filter_adaptation"].timeout, 300 if platform == "CUDA" else 90)
            self.assertEqual(by_name["ownership"].threads, 4)
            self.assertEqual(by_name["nonautonomous"].threads, 2)

    def test_references_and_negative_case(self):
        for platform in ("SERIAL", "OMP"):
            lorenz = list(ci.cases(platform, "lorenz-reference"))
            self.assertCountEqual([c.args[2:] for c in lorenz], [
                ("RK45", "matlab"), ("RK45", "scipy"), ("RK23", "matlab"), ("RK23", "scipy"), ("DOP853", "scipy")])
            for c in lorenz:
                self.assertEqual(c.compare, ("compare_explicit_lorenz_simulation.py", c.args[1]))
            dense, = ci.cases(platform, "dop853-reference")
            self.assertEqual(dense.compare, ("compare_explicit_dop853_dense_output.py", *dense.args))
            valid, invalid = ci.cases(platform, "rk4-reference")
            self.assertEqual((valid.args[0], valid.args[2:]), ("0.1003", ("RK4", "constant", "0.0005")))
            self.assertEqual(invalid.expected_exit, 1)
            self.assertEqual(invalid.args[2:], ("RK4", "scipy"))
        with self.assertRaises(ValueError):
            list(ci.cases("CUDA", "rk4-reference"))

    def test_list_build_only_and_failed_probe(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, PLATFORM="CUDA"), \
                patch.object(subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            args = ["--build-dir", tmp, "--device-probe-target", "probe", "--make-option", "G++ = clang++"]
            self.assertEqual(ci.main([*args, "--list"]), 0)
            run.assert_not_called()
            run.return_value.returncode = 0
            self.assertEqual(ci.main([*args, "-B"]), 0)
            summary = Path(tmp)/"CUDA/explicit-rk/summary.json"
            report = json.loads(summary.read_text())
            self.assertEqual(report["status"], "built")
            self.assertFalse(report["passed"])
            for call in run.call_args_list:
                self.assertEqual(call.args[0][0], "make")
                self.assertIn("G++ = clang++", call.args[0])
                self.assertIn("-B", call.args[0])
                self.assertNotIn("probe", call.args[0])
            run.reset_mock()
            run.side_effect = subprocess.CalledProcessError(1, "probe")
            self.assertEqual(ci.main(args), 1)
            run.assert_called_once()
            report = json.loads(summary.read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["passed"])
            self.assertFalse(report["results"])

    def test_failure_does_not_hide_later_cases_or_use_stale_binaries(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, PLATFORM="OMP"), \
                contextlib.redirect_stdout(io.StringIO()):
            def invoke(command, **kwargs):
                if command[0] == "make" and command[-1] == "test_explicit_fsal_cpu_omp.bin":
                    return subprocess.CompletedProcess(command, 2)
                self.assertFalse(str(command[0]).endswith("/test_explicit_fsal_cpu_omp.bin"))
                return subprocess.CompletedProcess(command, 0)
            with patch.object(subprocess, "run", side_effect=invoke):
                self.assertEqual(ci.main(["--build-dir", tmp]), 1)
            report = json.loads((Path(tmp)/"OMP/explicit-rk/summary.json").read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(report["results"]["fsal"]["status"], "failed")
            self.assertEqual(report["results"]["vdp_dense_output"]["status"], "passed")
            self.assertEqual(set(report["results"]), set(report["selected"]))


if __name__ == "__main__":
    unittest.main()
