#!/usr/bin/env python3
"""Stage the six Qwen Nothink checkpoint-throughput sweeps used by the paper."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from decimal import Decimal
from pathlib import Path


FIELDS = (
    "experiment_id",
    "model_label",
    "step",
    "samples_per_sec",
    "source_artifact",
    "source_sha256",
)
EXPECTED_STEPS = (200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2298)
SOURCE_DIRS = {
    "qwen-3.5-0.8b-ft-nothink-r8": "Qwen3.5-0.8B/lora/sft-ft-nothink-6ep",
    "qwen-3.5-0.8b-jo-nothink-r8": "Qwen3.5-0.8B/lora/sft-json-only-nothink-6ep",
    "qwen-3.5-2b-ft-nothink-r8": "Qwen3.5-2B/lora/sft-ft-nothink-6ep",
    "qwen-3.5-2b-jo-nothink-r8": "Qwen3.5-2B/lora/sft-json-only-nothink-6ep",
    "qwen-3.5-4b-ft-nothink-r8": "Qwen3.5-4B/lora/sft-ft-nothink-6ep",
    "qwen-3.5-4b-jo-nothink-r8": "Qwen3.5-4B/lora/sft-json-only-nothink-6ep",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=root,
        help="repository containing the staged data/models runtime JSONs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results" / "nothink_checkpoint_throughput.csv",
    )
    parser.add_argument(
        "--paper-output",
        type=Path,
        help="optional byte-identical copy for the paper repository",
    )
    args = parser.parse_args()

    with (root / "configs" / "experiments.csv").open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    labels = {
        row["experiment_id"]: row["model_label"]
        for row in manifest
        if row["experiment_id"] in SOURCE_DIRS
    }
    ordered_ids = [row["experiment_id"] for row in manifest if row["experiment_id"] in SOURCE_DIRS]
    if set(labels) != set(SOURCE_DIRS) or len(ordered_ids) != len(SOURCE_DIRS):
        raise ValueError("configs/experiments.csv does not contain the six expected Nothink runs")

    rows: list[dict[str, object]] = []
    for experiment_id in ordered_ids:
        run_dir = args.source_root / "data" / "models" / SOURCE_DIRS[experiment_id]
        paths = sorted(
            run_dir.glob("predict-checkpoint-*/predict_results.json"),
            key=lambda path: int(re.search(r"predict-checkpoint-(\d+)", str(path)).group(1)),
        )
        steps = tuple(int(path.parent.name.rsplit("-", 1)[1]) for path in paths)
        if steps != EXPECTED_STEPS:
            raise ValueError(f"{experiment_id} has checkpoint steps {steps}, expected {EXPECTED_STEPS}")
        for step, path in zip(steps, paths):
            payload = json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)
            speed = payload.get("predict_samples_per_second")
            if not isinstance(speed, Decimal) or not math.isfinite(float(speed)) or speed <= 0:
                raise ValueError(f"{path} has invalid predict_samples_per_second={speed!r}")
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "model_label": labels[experiment_id],
                    "step": step,
                    "samples_per_sec": str(speed),
                    "source_artifact": path.relative_to(args.source_root).as_posix(),
                    "source_sha256": sha256(path),
                }
            )

    if len(rows) != 72:
        raise ValueError(f"expected 72 Nothink throughput rows, found {len(rows)}")
    write_csv(args.output, rows)
    if args.paper_output:
        write_csv(args.paper_output, rows)
        if args.output.read_bytes() != args.paper_output.read_bytes():
            raise RuntimeError("paper output is not byte-identical to the release artifact")
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
