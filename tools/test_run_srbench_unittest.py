"""Focused stdlib tests for the replayable SRBench wrapper."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_srbench

VALID = {"report_kind": "operon_gp", "schema_version": 1, "symbolic_model": "x", "tree_node_count": 1, "seed": 1,
         "metrics": {"r2_train": 0.1, "r2_test": 0.2, "mae_train": 1.0, "mae_test": 2.0, "nmse_train": 1.0, "nmse_test": 2.0,
                     "mse_train": 1.0, "mse_test": 2.0, "best_fitness": 1.0, "average_fitness": 1.0,
                     "average_tree_node_count": 1.0, "elapsed_seconds": 0.01, "evaluator_calls": 1,
                     "result_evaluations": 1, "jacobian_evaluations": 1, "optimizer_seconds": 0.01}}


def _fake_binary(root, report=VALID):
    fake = root / "fake.py"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json,sys\n"
        "if '--version' in sys.argv: print('fake-1'); sys.exit(0)\n"
        "path=sys.argv[sys.argv.index('--report-json')+1]\n"
        f"open(path, 'w').write({json.dumps(report)!r})\n"
    )
    fake.chmod(0o755)
    (root / "flake.lock").write_text("{}\n")
    return fake

def _run_wrapper(root, output, fake, data, *extra):
    cmd = [sys.executable, str(run_srbench.__file__), "--output", str(output), str(fake), "--",
           "--dataset", str(data), "--seed", "1", "--evaluations", "1", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=root)


class WrapperTests(unittest.TestCase):
    def test_malformed_and_nonfinite_reports(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            run_srbench.validate_report({})
        bad = dict(VALID, metrics=dict(VALID["metrics"], mse_train=float("nan")))
        with self.assertRaisesRegex(ValueError, "finite"):
            run_srbench.validate_report(bad)

    def test_strict_report_types_duplicates_and_extras(self):
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            run_srbench.validate_report(dict(VALID, tree_node_count=True))
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.json"; p.write_text('{"a": 1, "a": 2}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                run_srbench._json_strict(p)

    def test_report_flag_is_wrapper_owned(self):
        with self.assertRaisesRegex(ValueError, "wrapper-owned"):
            run_srbench.parse_args(["--report-json", "out.json"])

    def test_duplicate_and_implicit_flags(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            run_srbench.parse_args(["--seed=1", "--seed=2"])
        with self.assertRaisesRegex(ValueError, "requires a value"):
            run_srbench.parse_args(["--dataset"])
        self.assertEqual(run_srbench.parse_args(["--shuffle"])["shuffle"], True)
        self.assertEqual(run_srbench.parse_args(["--jit"])["jit"], "all")

    def test_canonical_paths_and_atomic_failure_retry(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); data = root / "data.csv"; data.write_text("x,y\n1,2\n")
            args, _ = run_srbench.canonical_args(["--dataset", str(data), "--shuffle", "--seed", "4"], root)
            self.assertEqual(args, [f"--dataset={data}", "--shuffle", "--seed=4"])
            out = root / "artifact"; fake = root / "fake.py"
            fake.write_text("#!/usr/bin/env python3\nimport sys\nif '--version' in sys.argv: print('fake-1'); sys.exit(0)\np=sys.argv[sys.argv.index('--report-json')+1]\nopen(p,'w').write('not json')\n")
            fake.chmod(0o755)
            cmd = [sys.executable, str(run_srbench.__file__), "--output", str(out), str(fake), "--", "--dataset", str(data), "--seed", "1", "--evaluations", "1"]
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0); self.assertFalse(out.exists())
            fake.write_text("#!/usr/bin/env python3\nimport sys,json\nif '--version' in sys.argv: print('fake-1'); sys.exit(0)\np=sys.argv[sys.argv.index('--report-json')+1]\nopen(p,'w').write(" + repr(json.dumps(VALID)) + ")\n")
            fake.chmod(0o755)
            self.assertEqual(subprocess.run(cmd, capture_output=True).returncode, 0); self.assertTrue((out / "manifest.json").is_file())

    def test_timeout_leaves_no_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); data = root / "data.csv"; data.write_text("x,y\n1,2\n")
            fake = root / "slow.py"; fake.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(10)\n"); fake.chmod(0o755)
            out = root / "artifact"; cmd = [sys.executable, str(run_srbench.__file__), "--timeout", "0.1", "--output", str(out), str(fake), "--", "--dataset", str(data), "--seed", "1", "--evaluations", "1"]
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0); self.assertFalse(out.exists())

    def test_attached_staging_and_nested_parent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); data = root / "data.csv"; data.write_text("x,y\n1,2\n")
            output = root / "nested" / "deep" / "artifact"; fake = _fake_binary(root)
            result = _run_wrapper(root, output, fake, data)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output / "results.json").is_file())
            self.assertEqual(run_srbench._json_strict(output / "results.json")["manifest_sha256"], run_srbench.sha256(output / "manifest.json"))

    def test_sink_rejection_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); data = root / "data.csv"; data.write_text("x,y\n1,2\n"); fake = _fake_binary(root)
            output = root / "artifact"; output.write_text("sentinel")
            result = _run_wrapper(root, output, fake, data)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_text(), "sentinel")

    def test_strict_binding_rejects_tampered_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); data = root / "data.csv"; data.write_text("x,y\n1,2\n"); fake = _fake_binary(root); output = root / "artifact"
            result = _run_wrapper(root, output, fake, data)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = output / "manifest.json"; manifest.write_text(manifest.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "bound"):
                run_srbench._manifest_and_results(output)

    def test_manifest_path_replay(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); data = root / "data.csv"; data.write_text("x,y\n1,2\n"); fake = _fake_binary(root)
            first = root / "first"; second = root / "second"
            self.assertEqual(_run_wrapper(root, first, fake, data).returncode, 0)
            self.assertEqual(_run_wrapper(root, second, fake, data).returncode, 0)
            cmd = [sys.executable, str(run_srbench.__file__), "--replay-manifest", str(first / "manifest.json"), "--against-artifact", str(second)]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_strict_artifact_schema_comparator(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); data = root / "data.csv"; data.write_text("x,y\n1,2\n"); fake = _fake_binary(root); output = root / "artifact"
            result = _run_wrapper(root, output, fake, data)
            self.assertEqual(result.returncode, 0, result.stderr)
            results_path = output / "results.json"; payload = run_srbench._json_strict(results_path); payload["unexpected"] = True
            results_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema"):
                run_srbench._manifest_and_results(output)


if __name__ == "__main__": unittest.main()
