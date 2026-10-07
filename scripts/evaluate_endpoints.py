#!/usr/bin/env python3
"""Score the managed-endpoint exports with the paper's fixed metrics.

Each file in ``data/endpoint_predictions/`` is one per-request Databricks export
for a deployed variant. This script joins each export to the labelled test split
by normalized-transaction hash, scores it with the same canonicalization and
micro-averaged F1 used for the local checkpoint sweeps, and writes one aggregate
row per variant to ``results/endpoint_metrics.csv``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_predictions", ROOT / "scripts" / "evaluate_predictions.py"
)
_EVAL = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _EVAL
_SPEC.loader.exec_module(_EVAL)

FIELDS = _EVAL.FIELDS
FILENAME_RE = re.compile(r"^endpoint_predictions_(?P<variant>.+?)_\d{8}_\d{6}")


def variant_id(path: Path) -> str:
    match = FILENAME_RE.match(path.stem)
    if not match:
        raise ValueError(f"unexpected endpoint export name: {path.name}")
    return match.group("variant")


def score_export(path: Path, references, included: set[str]) -> dict[str, object]:
    by_id = {reference.sample_id: reference for reference in references}
    rows, parsed_count, seen = [], 0, set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_id = _EVAL.make_sample_id(row["transaction"])
            if sample_id not in included or sample_id in seen:
                continue
            seen.add(sample_id)
            prediction, parsed = _EVAL.parse_object(row["prediction"])
            parsed_count += parsed
            rows.append((by_id[sample_id].values, prediction, parsed))

    missing = included.difference(seen)
    if missing:
        raise ValueError(f"{path.name} is missing {len(missing)} included samples")

    overall, _ = _EVAL.prf(rows, _EVAL.canonicalize)
    return {
        "variant": variant_id(path),
        "source_file": path.name,
        "n": len(rows),
        "canonical_precision": overall["precision"],
        "canonical_recall": overall["recall"],
        "canonical_f1": overall["f1"],
        "parse_rate": parsed_count / len(rows) if rows else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, default=ROOT / "data" / "endpoint_predictions")
    parser.add_argument("--references", type=Path, default=ROOT / "data" / "raw" / "test.csv")
    parser.add_argument("--subset-manifest", type=Path, default=ROOT / "data" / "evaluation_subset.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "endpoint_metrics.csv")
    args = parser.parse_args()

    references = _EVAL.load_references(args.references)
    included = _EVAL.load_subset(args.subset_manifest, references)

    exports = sorted(args.exports.glob("*.csv"))
    if not exports:
        raise SystemExit(f"no endpoint exports found under {args.exports}")

    results = [score_export(path, references, included) for path in exports]
    results.sort(key=lambda row: row["variant"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    for row in results:
        print(f"{row['variant']:20s} n={row['n']:4d} F1={row['canonical_f1']:.4f} parse={row['parse_rate']:.3f}")
    print(f"\nwrote {len(results)} rows to {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
