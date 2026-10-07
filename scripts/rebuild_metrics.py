#!/usr/bin/env python3
"""Rebuild checkpoint and selected-checkpoint CSVs from released predictions."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

from evaluate_predictions import evaluate


METRIC_KEYS = (
    "canonical_precision",
    "canonical_recall",
    "canonical_f1",
    "strict_precision",
    "strict_recall",
    "strict_f1",
    "canonical_record_exact_match",
    "strict_record_exact_match",
    "parse_rate",
    "schema_valid_rate",
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=root / "results" / "predictions")
    parser.add_argument("--test-csv", type=Path, default=root / "data" / "raw" / "test.csv")
    parser.add_argument(
        "--subset-manifest", type=Path, default=root / "data" / "evaluation_subset.csv"
    )
    args = parser.parse_args()

    with (root / "configs" / "experiments.csv").open(newline="", encoding="utf-8") as handle:
        experiments = {row["experiment_id"]: row for row in csv.DictReader(handle)}

    rows: list[dict[str, Any]] = []
    for experiment_id, experiment in experiments.items():
        paths = sorted(
            (args.predictions_dir / experiment_id).glob("checkpoint-*.jsonl.gz"),
            key=lambda path: int(re.search(r"checkpoint-(\d+)", path.name).group(1)),
        )
        if not paths:
            raise FileNotFoundError(f"no released predictions for {experiment_id}")
        for prediction_path in paths:
            step = int(re.search(r"checkpoint-(\d+)", prediction_path.name).group(1))
            metrics, _ = evaluate(prediction_path, args.test_csv, args.subset_manifest)
            row: dict[str, Any] = {
                "experiment_id": experiment_id,
                "model_label": experiment["model_label"],
                "step": step,
                "epoch": round(min(step / 383, 6), 4),
                "selected": str(step == int(experiment["best_step"])).lower(),
                "n": metrics["n"],
            }
            row.update({key: f"{metrics[key]:.10f}" for key in METRIC_KEYS})
            row["source_artifact"] = str(prediction_path.relative_to(root))
            rows.append(row)

    checkpoint_path = root / "results" / "checkpoint_metrics.csv"
    selected_path = root / "results" / "best_checkpoint_metrics.csv"
    write_csv(checkpoint_path, rows)
    write_csv(selected_path, [row for row in rows if row["selected"] == "true"])
    print(f"wrote {len(rows)} rows to {checkpoint_path.relative_to(root)}")
    print(
        f"wrote {sum(row['selected'] == 'true' for row in rows)} rows "
        f"to {selected_path.relative_to(root)}"
    )


if __name__ == "__main__":
    main()
