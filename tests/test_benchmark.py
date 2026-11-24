import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from nccl_benchmark.catalog import WORKS
from nccl_benchmark.registry import BENCHMARK_METHODS, get_method
from nccl_benchmark.results import summarize
from nccl_benchmark.survey_runner import _summary
from nccl_benchmark.upstream import RECIPES


class BenchmarkTests(unittest.TestCase):
    def test_registry_matches_execution_paths(self):
        catalog = {work.id: work for work in WORKS}
        native = {method.id for method in BENCHMARK_METHODS}
        self.assertEqual(len(catalog), len(WORKS))
        self.assertEqual(len(native), len(BENCHMARK_METHODS))
        self.assertFalse(native.intersection(RECIPES))
        for key in native:
            self.assertEqual(catalog[key].status(native), "native")
            self.assertEqual(get_method(key).source, catalog[key].paper)
        for key in RECIPES:
            self.assertEqual(catalog[key].status(native), "upstream")
            self.assertTrue(catalog[key].code.startswith("https://"))
        self.assertEqual(get_method("C2-AFCL").id, "c2afcl")
        self.assertEqual(get_method("rcfcl").id, "rc-fcl")

    def test_task_end_forgetting_uses_only_earlier_stage_ends(self):
        history, task_history = [], []
        # A high mid-stage score must not be confused with a task-end score.
        evaluations = [(1, [0.99]), (1, [0.8]), (2, [0.7, 0.6]), (3, [0.5, 0.8, 1.0])]
        for index, (stage, scores) in enumerate(evaluations):
            history.append(dict(task=stage, test_accuracy=sum(scores) / len(scores), round_in_task=index))
            task_history.extend(dict(task=stage, test_task=j + 1, test_accuracy=value)
                                for j, value in enumerate(scores))
        report = _summary(history, task_history, 3)
        self.assertEqual(report["task_end_accuracy_matrix"][0], [0.8, None, None])
        self.assertAlmostEqual(report["final_average_forgetting"], 0.05)
        self.assertAlmostEqual(report["final_task_macro_accuracy"], 2.3 / 3)
        self.assertIsNone(_summary(history[:2], task_history[:2], 1)["final_average_forgetting"])

    def test_export_preserves_native_units_and_upstream_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native, external = root / "rc", root / "feroma"
            native.mkdir()
            external.mkdir()
            record = dict(method="rc-fcl", protocol="asynchronous-concept-drift", profile="smoke",
                          config=dict(dataset="synthetic", seed=2))
            (native / "run.json").write_text(json.dumps(record))
            (native / "metrics.csv").write_text("round,AC,FS\n1,90,8\n2,75,15\n")
            (external / "run.json").write_text(json.dumps(dict(method="feroma", backend="upstream", status="failed")))
            stream = io.StringIO()
            summarize(root, stream)
            rows = {row["metric"]: row for row in csv.DictReader(io.StringIO(stream.getvalue()))}
            self.assertEqual((rows["AC"]["value"], rows["AC"]["unit"]), ("75", "percent"))
            self.assertEqual(rows["FS"]["unit"], "percentage_points")
            self.assertEqual(rows["launch_status"]["value"], "failed")
