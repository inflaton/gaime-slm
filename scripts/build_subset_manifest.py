#!/usr/bin/env python3
"""Rebuild the hashed common-test manifest from the internal legacy baseline CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from evaluate_predictions import load_references, make_sample_id


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_csv", type=Path, help="Internal merged CSV with a transaction column.")
    parser.add_argument("--test-csv", type=Path, default=root / "data" / "raw" / "test.csv")
    parser.add_argument("--output", type=Path, default=root / "data" / "evaluation_subset.csv")
    args = parser.parse_args()

    with args.legacy_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "transaction" not in (reader.fieldnames or ()):
            raise ValueError(f"{args.legacy_csv} has no transaction column")
        legacy_ids = {
            make_sample_id(row["transaction"])
            for row in reader
            if (row.get("transaction") or "").strip()
        }

    references = load_references(args.test_csv)
    rows = []
    for reference in references:
        included = reference.sample_id in legacy_ids
        rows.append(
            {
                "sample_id": reference.sample_id,
                "include": str(included).lower(),
                "reason": "common_legacy_llama_subset" if included else "absent_from_legacy_llama_baseline",
            }
        )

    included_count = sum(row["include"] == "true" for row in rows)
    if (len(rows), included_count) != (1222, 1213):
        raise ValueError(
            f"unexpected manifest counts: {len(rows)} test records, {included_count} included"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "include", "reason"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {included_count:,}/{len(rows):,} included sample IDs to {args.output}")


if __name__ == "__main__":
    main()
