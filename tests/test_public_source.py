"""Source-release checks using invented fixtures, with no research data or GPU.

Run with: python3 -m unittest discover -s tests -p 'test_public_source.py'
"""

from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EVALUATOR = load_script("evaluate_predictions")
PREPARE = load_script("prepare_data")
ENDPOINT = load_script("evaluate_endpoints")


def fields(**values: str) -> dict[str, str]:
    return {field: values.get(field, "") for field in EVALUATOR.FIELDS}


def write_csv(path: Path, columns, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class SyntheticFixtureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        # These names and descriptors are invented solely for these tests.
        self.rows = [
            {"TRANSACTION": "SYNTHETIC ALPHA FORGE", **fields(MERCHANT_NAME="Alpha Forge", CITY="Eastford")},
            {"TRANSACTION": "SYNTHETIC BETA ORCHARD", **fields(MERCHANT_NAME="Beta Orchard", CITY="Westford")},
        ]
        self.reference_path = self.root / "references.csv"
        write_csv(self.reference_path, ("TRANSACTION", *EVALUATOR.FIELDS), self.rows)
        self.references = EVALUATOR.load_references(self.reference_path)

    def predictions(self, entries, *, compressed=False) -> Path:
        path = self.root / ("predictions.jsonl.gz" if compressed else "predictions.jsonl")
        opener = gzip.open if compressed else open
        with opener(path, "wt", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry) + "\n")
        return path

    def test_canonical_and_strict_metrics_are_distinct(self) -> None:
        path = self.predictions([
            {"sample_id": self.references[0].sample_id, "predict": json.dumps(fields(MERCHANT_NAME="alpha-*forge", CITY="EASTFORD"))},
            {"sample_id": self.references[1].sample_id, "predict": json.dumps(fields(MERCHANT_NAME="Beta Orchard", CITY="Elsewhere"))},
        ])
        metrics, details = EVALUATOR.evaluate(path, self.reference_path)
        self.assertEqual(metrics["n"], 2)
        for name in ("precision", "recall", "f1"):
            self.assertEqual(metrics[f"canonical_{name}"], 0.75)
            self.assertEqual(metrics[f"strict_{name}"], 0.25)
        self.assertEqual(metrics["canonical_record_exact_match"], 0.5)
        self.assertEqual(metrics["strict_record_exact_match"], 0.0)
        self.assertEqual(metrics["parse_rate"], 1.0)
        self.assertEqual(metrics["schema_valid_rate"], 1.0)
        by_field = {row["field"]: row for row in details}
        self.assertEqual(by_field["MERCHANT_NAME"]["tp"], 2)
        self.assertEqual(by_field["CITY"]["tp"], 1)

    def test_reordered_compressed_prompts_map_by_transaction(self) -> None:
        path = self.predictions([
            {"prompt": f"Extract attributes. TRANSACTION: {row['TRANSACTION']}<|eot_id|>ignored suffix", "predict": json.dumps({key: row[key] for key in EVALUATOR.FIELDS})}
            for row in reversed(self.rows)
        ], compressed=True)
        metrics, _ = EVALUATOR.evaluate(path, self.reference_path)
        self.assertEqual(metrics["strict_f1"], 1.0)
        self.assertEqual(metrics["strict_record_exact_match"], 1.0)

    def test_subset_requires_only_included_predictions(self) -> None:
        subset = self.root / "subset.csv"
        write_csv(subset, ("sample_id", "include"), [
            {"sample_id": ref.sample_id, "include": "yes" if index == 0 else "false"}
            for index, ref in enumerate(self.references)
        ])
        path = self.predictions([{"sample_id": self.references[0].sample_id, "predict": json.dumps(self.references[0].values)}])
        metrics, _ = EVALUATOR.evaluate(path, self.reference_path, subset)
        self.assertEqual(metrics["n"], 1)
        self.assertEqual(metrics["strict_record_exact_match"], 1.0)
        with self.assertRaisesRegex(ValueError, "missing 1 included samples"):
            EVALUATOR.evaluate(path, self.reference_path)

    def test_explicit_mapping_rejects_duplicate_mixed_and_unknown_ids(self) -> None:
        first = {"sample_id": self.references[0].sample_id, "predict": "{}"}
        for entries, message in (
            ([first, first], "duplicate sample ID"),
            ([first, {"predict": "{}"}], "mixes explicit sample IDs"),
            ([{"sample_id": "not-a-reference", "predict": "{}"}], "unknown or invalid sample ID"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                EVALUATOR.load_predictions(self.predictions(entries), self.references)

    def test_positional_fallback_rejects_partial_file(self) -> None:
        with self.assertRaisesRegex(ValueError, "positional fallback is unsafe"):
            EVALUATOR.load_predictions(self.predictions([{"predict": "{}"}]), self.references)

    def test_reference_and_subset_validation(self) -> None:
        write_csv(self.reference_path, ("TRANSACTION", *EVALUATOR.FIELDS), [self.rows[0], {**self.rows[0], "TRANSACTION": " synthetic alpha forge "}])
        with self.assertRaisesRegex(ValueError, "duplicate normalized transactions"):
            EVALUATOR.load_references(self.reference_path)
        subset = self.root / "subset.csv"
        write_csv(subset, ("sample_id", "include"), [{"sample_id": "unknown", "include": "true"}])
        with self.assertRaisesRegex(ValueError, "unknown sample IDs"):
            EVALUATOR.load_subset(subset, self.references)

    def test_endpoint_scoring_joins_and_deduplicates(self) -> None:
        export = self.root / "endpoint_predictions_synthetic_20260101_000000.csv"
        rows = [{"transaction": row["TRANSACTION"], "prediction": json.dumps({key: row[key] for key in EVALUATOR.FIELDS})} for row in reversed(self.rows)]
        write_csv(export, ("transaction", "prediction"), [*rows, rows[0]])
        metrics = ENDPOINT.score_export(export, self.references, {ref.sample_id for ref in self.references})
        self.assertEqual(metrics["variant"], "synthetic")
        self.assertEqual(metrics["n"], 2)
        self.assertEqual(metrics["canonical_f1"], 1.0)
        write_csv(export, ("transaction", "prediction"), rows[:1])
        with self.assertRaisesRegex(ValueError, "missing 1 included samples"):
            ENDPOINT.score_export(export, self.references, {ref.sample_id for ref in self.references})

    def test_evaluation_cli_emits_metrics_and_field_counts(self) -> None:
        path = self.predictions([{"sample_id": ref.sample_id, "predict": json.dumps(ref.values)} for ref in self.references])
        output = self.root / "summary.json"
        details = self.root / "details.csv"
        result = subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_predictions.py"), str(path), "--test-csv", str(self.reference_path), "--output-json", str(output), "--details-csv", str(details)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(output.read_text())["strict_f1"], 1.0)
        with details.open(newline="", encoding="utf-8") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), len(EVALUATOR.FIELDS))

    def prepare_root(self) -> None:
        for split in PREPARE.EXPECTED_COUNTS:
            write_csv(self.root / "data/raw" / f"{split}.csv", ("TRANSACTION", *EVALUATOR.FIELDS), self.rows[:1])
        (self.root / "prompts").mkdir()
        for name in PREPARE.PROMPTS.values():
            (self.root / "prompts" / name).write_text("Synthetic test instruction.\n", encoding="utf-8")

    def run_prepare(self, *options: str):
        return subprocess.run([sys.executable, str(ROOT / "scripts/prepare_data.py"), "--root", str(self.root), *options], capture_output=True, text=True)

    def test_custom_count_cli_builds_all_splits_and_prompt_variants(self) -> None:
        self.prepare_root()
        result = self.run_prepare("--allow-custom-counts")
        self.assertEqual(result.returncode, 0, result.stderr)
        outputs = list((self.root / "data/llamafactory").glob("*.json"))
        self.assertEqual(len(outputs), 6)
        for output in outputs:
            samples = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["input"], self.rows[0]["TRANSACTION"])
            self.assertEqual(samples[0]["instruction"], "Synthetic test instruction.")
            self.assertEqual(json.loads(samples[0]["output"]), self.references[0].values)

    def test_default_cli_keeps_paper_split_counts(self) -> None:
        self.prepare_root()
        result = self.run_prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("has 1 rows; expected 6125", result.stderr)
        self.assertEqual(list((self.root / "data/llamafactory").glob("*.json")), [])

    def test_custom_counts_still_reject_invalid_splits(self) -> None:
        for columns, rows, message in (
            (("TRANSACTION",), [{"TRANSACTION": "SYNTHETIC"}], "missing columns"),
            (("TRANSACTION", *EVALUATOR.FIELDS), [], "contains no records"),
            (("TRANSACTION", *EVALUATOR.FIELDS), [{"TRANSACTION": "  ", **fields()}], "blank TRANSACTION"),
        ):
            with self.subTest(message=message):
                write_csv(self.reference_path, columns, rows)
                with self.assertRaisesRegex(ValueError, message):
                    PREPARE.load_rows(self.reference_path, expected_count=None)


class PublicSourceSmokeTest(unittest.TestCase):
    def test_permissive_recovery_is_distinct_from_strict_parsing(self) -> None:
        malformed = '{"MERCHANT_NAME": "Synthetic Shop"'
        self.assertEqual(EVALUATOR.parse_object(malformed), ({"MERCHANT_NAME": "Synthetic Shop"}, True))
        self.assertEqual(EVALUATOR.parse_strict_object(malformed), ({}, False))
        wrapped = '<think>synthetic reasoning</think>```json\n{"PAYMENT_FACILITATOR":"Example"}\n```'
        self.assertEqual(EVALUATOR.parse_strict_object(wrapped), ({"INTERMEDIARY": "Example"}, True))

    def test_training_manifest_references_shipped_configs(self) -> None:
        with (ROOT / "configs/experiments.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 23)
        self.assertEqual(len({row["experiment_id"] for row in rows}), len(rows))
        for row in rows:
            path = (ROOT / row["config_path"]).resolve()
            self.assertTrue(path.is_relative_to(ROOT))
            self.assertTrue(path.is_file(), path)

    def test_cli_help_requires_no_research_artifacts(self) -> None:
        for script in ("prepare_data.py", "evaluate_predictions.py", "evaluate_endpoints.py"):
            with self.subTest(script=script):
                result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--help"], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_zero_shot_runner_shell_syntax(self) -> None:
        result = subprocess.run(["bash", "-n", str(ROOT / "scripts/run_zero_shot.sh")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
