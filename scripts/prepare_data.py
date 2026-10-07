#!/usr/bin/env python3
"""Build six LLaMA-Factory datasets from locally supplied labelled CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "INTERMEDIARY",
    "MERCHANT_NAME",
    "DYNAMIC_DESCRIPTOR",
    "STREET",
    "ZIP",
    "CITY",
    "STATE",
    "COUNTRY",
    "PHONE",
    "WEBSITE",
)

EXPECTED_COUNTS = {"train": 6125, "validation": 668, "test": 1222}
OUTPUT_STEMS = {"train": "train", "validation": "val", "test": "test"}
PROMPTS = {"free_thinking": "free_thinking.txt", "json_only": "json_only.txt"}


def load_rows(path: Path, expected_count: int | None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"TRANSACTION", *FIELDS}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]

    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"{path} has {len(rows)} rows; expected {expected_count}")
    if not rows:
        raise ValueError(f"{path} contains no records")
    if any(not (row.get("TRANSACTION") or "").strip() for row in rows):
        raise ValueError(f"{path} contains a blank TRANSACTION")
    return rows


def to_alpaca(rows: list[dict[str, str]], instruction: str) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    for row in rows:
        answer = {field: row.get(field, "") or "" for field in FIELDS}
        samples.append(
            {
                "instruction": instruction,
                "input": row["TRANSACTION"],
                "output": json.dumps(answer, ensure_ascii=False),
            }
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (defaults to the parent of scripts/).",
    )
    parser.add_argument(
        "--allow-custom-counts",
        action="store_true",
        help="Accept nonempty splits of any size for your own data; otherwise enforce the paper's split counts.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    raw_dir = root / "data" / "raw"
    prompt_dir = root / "prompts"
    output_dir = root / "data" / "llamafactory"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_rows = {
        split: load_rows(raw_dir / f"{split}.csv", None if args.allow_custom_counts else count)
        for split, count in EXPECTED_COUNTS.items()
    }

    for variant, prompt_name in PROMPTS.items():
        instruction = (prompt_dir / prompt_name).read_text(encoding="utf-8").rstrip()
        suffix = "" if variant == "free_thinking" else "_json_only"
        for split, rows in split_rows.items():
            filename = f"{OUTPUT_STEMS[split]}_merchant_matching{suffix}.json"
            destination = output_dir / filename
            with destination.open("w", encoding="utf-8") as handle:
                json.dump(to_alpaca(rows, instruction), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            print(f"wrote {len(rows):,} records to {destination.relative_to(root)}")


if __name__ == "__main__":
    main()
