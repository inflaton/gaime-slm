#!/usr/bin/env python3
"""Stage and import the matched batch-4 throughput benchmark behind the paper.

Every speed the manuscript reports (the ``s/s`` column of the main results
table, both speed columns of the Think/Nothink table, and the horizontal axis
of the Pareto figure) is the median samples per second of repeated full
test-set evaluations of that configuration's selected checkpoint, run under one
protocol and one software stack on a single DGX Spark by
``scripts/benchmark/benchmark_dgx_spark_selected.sh`` and
``scripts/benchmark/benchmark_dgx_spark_all.sh``.

Two modes:

``--stage-from DIR``
    Copy the aggregate provenance of a ``benchmark_outputs/`` tree into
    ``data/benchmarks/``. Per-record predictions are never copied; absolute
    paths are rewritten so the staged copies carry no local home directory.

(no arguments)
    Rebuild ``results/benchmark_throughput.csv`` from the staged copies.

The checkpoint-sweep ``samples_per_sec`` columns in
``results/published_checkpoint_metrics.csv`` and
``results/nothink_checkpoint_throughput.csv`` are a *different*, superseded
measurement: they mix batch sizes across models and predate the current
software stack, so they are not comparable across rows and no reported speed
comes from them. They stay in the release as the historical record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from decimal import Decimal
from pathlib import Path


FIELDS = (
    "experiment_id",
    "model_label",
    "step",
    "benchmark_run",
    "repetitions",
    "batch_size",
    "samples_per_sec_median",
    "samples_per_sec_min",
    "samples_per_sec_max",
    "seconds_per_sample_median",
    "predictions_identical_across_repetitions",
    "prediction_sha256",
    "benchmark_canonical_f1",
    "benchmark_strict_record_exact_match",
    "source_artifact",
    "source_sha256",
)

# Benchmark run directory -> the harness that produced it. The sweep of all
# remaining configurations reuses the two headline results from the selected
# run, so between them the two runs cover every manifest entry exactly once.
RUNS = (
    "dgx-spark-selected/20260828T015239Z",
    "dgx-spark-all/all-v1",
)
HARNESS = {
    "dgx-spark-selected/20260828T015239Z": "scripts/benchmark/benchmark_dgx_spark_selected.sh",
    "dgx-spark-all/all-v1": "scripts/benchmark/benchmark_dgx_spark_all.sh",
}

# The two headline configurations are keyed by short model ids in the selected
# run; every other benchmark configuration id is already a manifest id.
SELECTED_TO_EXPERIMENT = {
    "llama-8b-ft": "llama-3.1-8b-ft-r8",
    "qwen-4b-jo": "qwen-3.5-4b-jo-think-r8",
}

# Copied verbatim; none of these carry absolute paths or per-record output.
VERBATIM_FILES = ("summary.json", "dependency-versions.json")
# Copied with local home directories rewritten (see redact_home).
REDACTED_FILES = ("runs.csv", "provenance.txt")
# device-initial.txt is deliberately not staged: it is an nvidia-smi capture
# that lists unrelated desktop processes running on the benchmark host.

HOME_PATTERN = re.compile(r"/(?:home|Users)/[^/\s,\"']+")
HOME_PLACEHOLDER = "<home>"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def redact_home(text: str) -> str:
    return HOME_PATTERN.sub(HOME_PLACEHOLDER, text)


def read_provenance(path: Path) -> dict[str, str]:
    provenance: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key == "config":
            continue
        provenance[key] = value
    return provenance


def stage(root: Path, source_root: Path, staged_root: Path) -> None:
    manifest: list[dict[str, str]] = []
    for run in RUNS:
        source_dir = source_root / run
        if not source_dir.is_dir():
            raise FileNotFoundError(f"missing benchmark run directory {source_dir}")
        target_dir = staged_root / run
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in VERBATIM_FILES + REDACTED_FILES:
            source = source_dir / name
            if not source.is_file():
                raise FileNotFoundError(f"missing {source}")
            target = target_dir / name
            if name in VERBATIM_FILES:
                shutil.copyfile(source, target)
                transform = "verbatim"
            else:
                text = source.read_text(encoding="utf-8")
                redacted = redact_home(text)
                target.write_text(redacted, encoding="utf-8")
                transform = "verbatim" if redacted == text else "home-paths-redacted"
            manifest.append(
                {
                    "benchmark_run": run,
                    "file": target.relative_to(root).as_posix(),
                    "transform": transform,
                    "staged_sha256": sha256(target),
                    "source_sha256": sha256(source),
                }
            )
    manifest_path = staged_root / "MANIFEST.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("benchmark_run", "file", "transform", "staged_sha256", "source_sha256"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(manifest)
    print(f"staged {len(manifest)} files under {staged_root}")


def load_run(staged_root: Path, run: str) -> tuple[dict[str, dict], dict[str, str], Path]:
    run_dir = staged_root / run
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"), parse_float=Decimal)
    provenance = read_provenance(run_dir / "provenance.txt")
    if summary.get("incomplete"):
        raise ValueError(f"{summary_path} reports incomplete configurations: {summary['incomplete']}")
    entries = summary.get("configurations") or summary.get("models")
    if not entries:
        raise ValueError(f"{summary_path} contains no benchmark entries")
    return entries, provenance, summary_path


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build(root: Path, staged_root: Path, output: Path, paper_output: Path | None = None) -> None:
    with (root / "configs" / "experiments.csv").open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))

    collected: dict[str, dict[str, object]] = {}
    for run in RUNS:
        entries, provenance, summary_path = load_run(staged_root, run)
        batch_size = int(provenance["batch_size"])
        if batch_size != 4:
            raise ValueError(f"{run} ran at batch size {batch_size}, expected 4")
        source_artifact = summary_path.relative_to(root).as_posix()
        source_sha256 = sha256(summary_path)
        for config_id, entry in entries.items():
            experiment_id = SELECTED_TO_EXPERIMENT.get(config_id, config_id)
            if experiment_id in collected:
                raise ValueError(f"{experiment_id} is benchmarked in more than one run")
            if not entry["predictions_identical_across_repetitions"]:
                raise ValueError(f"{config_id} predictions differ across repetitions")
            hashes = set(entry["prediction_sha256"])
            if len(hashes) != 1:
                raise ValueError(f"{config_id} has {len(hashes)} distinct prediction hashes")
            median = entry["median_samples_per_second"]
            low = entry["min_samples_per_second"]
            high = entry["max_samples_per_second"]
            if not (0 < low <= median <= high) or not math.isfinite(float(median)):
                raise ValueError(f"{config_id} has an implausible throughput summary")
            repetitions = int(entry["repetitions"])
            if repetitions != len(entry["prediction_sha256"]):
                raise ValueError(f"{config_id} repetition count disagrees with its hash list")
            f1_values = set(entry["common_canonical_f1"])
            rem_values = set(entry["common_strict_record_exact_match"])
            if len(f1_values) != 1 or len(rem_values) != 1:
                raise ValueError(f"{config_id} scored differently across identical predictions")
            collected[experiment_id] = {
                "benchmark_run": run,
                "repetitions": repetitions,
                "batch_size": batch_size,
                "samples_per_sec_median": str(median),
                "samples_per_sec_min": str(low),
                "samples_per_sec_max": str(high),
                "seconds_per_sample_median": repr(1.0 / float(median)),
                "predictions_identical_across_repetitions": "true",
                "prediction_sha256": hashes.pop(),
                "benchmark_canonical_f1": str(f1_values.pop()),
                "benchmark_strict_record_exact_match": str(rem_values.pop()),
                "source_artifact": source_artifact,
                "source_sha256": source_sha256,
            }

    missing = [row["experiment_id"] for row in manifest if row["experiment_id"] not in collected]
    if missing:
        raise ValueError(f"no matched benchmark for {missing}")
    unknown = set(collected) - {row["experiment_id"] for row in manifest}
    if unknown:
        raise ValueError(f"benchmark configurations absent from the manifest: {sorted(unknown)}")

    rows = [
        {
            "experiment_id": row["experiment_id"],
            "model_label": row["model_label"],
            "step": row["best_step"],
            **collected[row["experiment_id"]],
        }
        for row in manifest
    ]

    write_csv(output, rows)
    if paper_output:
        write_csv(paper_output, rows)
        if output.read_bytes() != paper_output.read_bytes():
            raise RuntimeError("paper output is not byte-identical to the release artifact")
    print(f"wrote {len(rows)} rows to {output}")
    print(f"sha256={sha256(output)}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-from",
        type=Path,
        help="benchmark_outputs/ directory to refresh data/benchmarks/ from",
    )
    parser.add_argument(
        "--staged-root",
        type=Path,
        default=root / "data" / "benchmarks",
        help="staged benchmark provenance (default: data/benchmarks)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results" / "benchmark_throughput.csv",
    )
    parser.add_argument(
        "--paper-output",
        type=Path,
        help="optional byte-identical copy for the paper repository",
    )
    args = parser.parse_args()

    if args.stage_from:
        stage(root, args.stage_from.resolve(), args.staged_root)
    build(root, args.staged_root, args.output, args.paper_output)


if __name__ == "__main__":
    main()
