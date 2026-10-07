#!/usr/bin/env python3
"""Evaluate LLaMA-Factory JSONL predictions with the paper's fixed metrics."""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


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
CHAT_BOUNDARIES = (
    "<|im_end|>",
    "<end_of_turn>",
    "<|END_OF_TURN_TOKEN|>",
    "<|eot_id|>",
    "<|end|>",
)


@dataclass(frozen=True)
class Reference:
    sample_id: str
    transaction: str
    values: dict[str, str]


def make_sample_id(transaction: str) -> str:
    normalized = transaction.strip().upper().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def canonicalize(value: Any) -> str:
    if value is None or value == "":
        return ""
    return re.sub(r"[ *,\-]", "", str(value).upper())


def strict_value(value: Any) -> str:
    return "" if value is None else str(value)


def parse_object(text: Any) -> tuple[dict[str, Any], bool]:
    if isinstance(text, dict):
        return text, True
    if not isinstance(text, str):
        return {}, False

    cleaned = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", cleaned, flags=re.IGNORECASE)
    for parser in (json.loads, ast.literal_eval):
        try:
            value = parser(cleaned)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            if "PAYMENT_FACILITATOR" in value and "INTERMEDIARY" not in value:
                value["INTERMEDIARY"] = value.pop("PAYMENT_FACILITATOR")
            return value, True
    # Preserve the permissive field-pair fallback used for the paper's published
    # canonicalized metric. It recovers simple key/value pairs from malformed JSON.
    pairs = re.findall(r'"([^"]+)"\s*:\s*"([^"]*)"', cleaned)
    if pairs:
        value = dict(pairs)
        if "PAYMENT_FACILITATOR" in value and "INTERMEDIARY" not in value:
            value["INTERMEDIARY"] = value.pop("PAYMENT_FACILITATOR")
        return value, True
    return {}, False


def parse_strict_object(text: Any) -> tuple[dict[str, Any], bool]:
    """Parse the JSON object used by strict metrics after removing wrappers."""
    if isinstance(text, dict):
        return text, True
    if not isinstance(text, str):
        return {}, False
    cleaned = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", cleaned, flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    candidate = cleaned[start : end + 1] if start >= 0 and end >= start else cleaned
    for parser in (json.loads, ast.literal_eval):
        try:
            value = parser(candidate)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            if "PAYMENT_FACILITATOR" in value and "INTERMEDIARY" not in value:
                value["INTERMEDIARY"] = value.pop("PAYMENT_FACILITATOR")
            return value, True
    return {}, False


def extract_transaction(prompt: Any) -> str | None:
    if not isinstance(prompt, str) or "TRANSACTION:" not in prompt:
        return None
    transaction = prompt.rsplit("TRANSACTION:", 1)[1].lstrip(" \n\r\t")
    positions = [transaction.find(marker) for marker in CHAT_BOUNDARIES]
    positions = [position for position in positions if position >= 0]
    if positions:
        transaction = transaction[: min(positions)]
    transaction = transaction.strip()
    return transaction or None


def load_references(path: Path) -> list[Reference]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"TRANSACTION", *FIELDS}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        references = []
        for row in reader:
            transaction = row["TRANSACTION"]
            references.append(
                Reference(
                    sample_id=make_sample_id(transaction),
                    transaction=transaction,
                    values={field: row.get(field, "") or "" for field in FIELDS},
                )
            )
    if len({reference.sample_id for reference in references}) != len(references):
        raise ValueError(f"{path} contains duplicate normalized transactions")
    return references


def load_subset(path: Path | None, references: Iterable[Reference]) -> set[str]:
    all_ids = {reference.sample_id for reference in references}
    if path is None:
        return all_ids
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"sample_id", "include"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} must contain sample_id and include columns")
        included = {
            row["sample_id"]
            for row in reader
            if row["include"].strip().lower() in {"1", "true", "yes", "y"}
        }
    unknown = included.difference(all_ids)
    if unknown:
        raise ValueError(f"{path} contains {len(unknown)} unknown sample IDs")
    return included


def load_predictions(path: Path, references: list[Reference]) -> dict[str, dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, mode="rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or "predict" not in value:
                raise ValueError(f"{path}:{line_number} is not a prediction object")
            entries.append(value)

    reference_ids = {reference.sample_id for reference in references}

    # Private reference artifacts can omit raw prompts and carry only the
    # stable SHA-256 sample identifier used by data/evaluation_subset.csv.
    # Require an all-or-none, one-to-one mapping so partial or mixed files never
    # fall through to the positional compatibility path below.
    has_explicit_ids = ["sample_id" in entry for entry in entries]
    if any(has_explicit_ids):
        if not all(has_explicit_ids):
            raise ValueError(f"{path} mixes explicit sample IDs with prompt-mapped rows")
        mapped_by_id: dict[str, dict[str, Any]] = {}
        for entry in entries:
            sample_id = entry["sample_id"]
            if not isinstance(sample_id, str) or sample_id not in reference_ids:
                raise ValueError(f"{path} contains an unknown or invalid sample ID")
            if sample_id in mapped_by_id:
                raise ValueError(f"{path} contains duplicate sample ID {sample_id}")
            mapped_by_id[sample_id] = entry
        return mapped_by_id

    mapped: dict[str, dict[str, Any]] = {}
    mapping_failed = False
    for entry in entries:
        transaction = extract_transaction(entry.get("prompt"))
        if transaction is None:
            mapping_failed = True
            break
        sample_id = make_sample_id(transaction)
        if sample_id not in reference_ids or sample_id in mapped:
            mapping_failed = True
            break
        mapped[sample_id] = entry

    if not mapping_failed and len(mapped) == len(entries):
        return mapped
    if len(entries) != len(references):
        raise ValueError(
            "Could not map prompts to test rows, and positional fallback is unsafe: "
            f"{len(entries)} predictions versus {len(references)} references"
        )
    return {reference.sample_id: entry for reference, entry in zip(references, entries)}


def prf(
    rows: Iterable[tuple[dict[str, str], dict[str, Any], bool]],
    transform: Callable[[Any], str],
) -> tuple[dict[str, float | int], list[dict[str, float | int | str]]]:
    totals = {field: {"tp": 0, "truth": 0, "predicted": 0} for field in FIELDS}
    for truth, prediction, _ in rows:
        for field in FIELDS:
            truth_value = truth.get(field, "")
            predicted_value = prediction.get(field, "")
            if truth_value:
                totals[field]["truth"] += 1
            if predicted_value:
                totals[field]["predicted"] += 1
            if truth_value and predicted_value and transform(truth_value) == transform(predicted_value):
                totals[field]["tp"] += 1

    details: list[dict[str, float | int | str]] = []
    for field, counts in totals.items():
        precision = counts["tp"] / counts["predicted"] if counts["predicted"] else 0.0
        recall = counts["tp"] / counts["truth"] if counts["truth"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        details.append({"field": field, **counts, "precision": precision, "recall": recall, "f1": f1})

    overall = {
        key: sum(int(counts[key]) for counts in totals.values())
        for key in ("tp", "truth", "predicted")
    }
    precision = overall["tp"] / overall["predicted"] if overall["predicted"] else 0.0
    recall = overall["tp"] / overall["truth"] if overall["truth"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    overall.update({"precision": precision, "recall": recall, "f1": f1})
    return overall, details


def evaluate(
    prediction_path: Path,
    test_csv: Path,
    subset_manifest: Path | None = None,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    references = load_references(test_csv)
    included = load_subset(subset_manifest, references)
    predictions = load_predictions(prediction_path, references)
    missing = included.difference(predictions)
    if missing:
        raise ValueError(f"prediction file is missing {len(missing)} included samples")

    canonical_rows: list[tuple[dict[str, str], dict[str, Any], bool]] = []
    strict_rows: list[tuple[dict[str, str], dict[str, Any], bool]] = []
    schema_valid = 0
    for reference in references:
        if reference.sample_id not in included:
            continue
        prediction_text = predictions[reference.sample_id]["predict"]
        canonical_prediction, canonical_parsed = parse_object(prediction_text)
        strict_prediction, strict_parsed = parse_strict_object(prediction_text)
        if strict_parsed and set(strict_prediction) == set(FIELDS) and all(
            isinstance(strict_prediction[field], str) for field in FIELDS
        ):
            schema_valid += 1
        canonical_rows.append((reference.values, canonical_prediction, canonical_parsed))
        strict_rows.append((reference.values, strict_prediction, strict_parsed))

    canonical, canonical_details = prf(canonical_rows, canonicalize)
    strict, _ = prf(strict_rows, strict_value)
    strict_records = sum(
        parsed
        and all(strict_value(truth[field]) == strict_value(prediction.get(field, "")) for field in FIELDS)
        for truth, prediction, parsed in strict_rows
    )
    canonical_records = sum(
        parsed
        and all(canonicalize(truth[field]) == canonicalize(prediction.get(field, "")) for field in FIELDS)
        for truth, prediction, parsed in canonical_rows
    )
    parsed_records = sum(parsed for _, _, parsed in strict_rows)
    n = len(strict_rows)
    metrics: dict[str, float | int | str] = {
        "prediction_file": str(prediction_path),
        "n": n,
        "canonical_precision": canonical["precision"],
        "canonical_recall": canonical["recall"],
        "canonical_f1": canonical["f1"],
        "strict_precision": strict["precision"],
        "strict_recall": strict["recall"],
        "strict_f1": strict["f1"],
        "canonical_record_exact_match": canonical_records / n if n else 0.0,
        "strict_record_exact_match": strict_records / n if n else 0.0,
        "parse_rate": parsed_records / n if n else 0.0,
        "schema_valid_rate": schema_valid / n if n else 0.0,
    }
    return metrics, canonical_details


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--test-csv", type=Path, default=root / "data" / "raw" / "test.csv")
    parser.add_argument("--subset-manifest", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--details-csv", type=Path)
    args = parser.parse_args()

    metrics, details = evaluate(args.predictions, args.test_csv, args.subset_manifest)
    serialized = json.dumps(metrics, indent=2)
    print(serialized)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n", encoding="utf-8")
    if args.details_csv:
        write_csv(args.details_csv, details)


if __name__ == "__main__":
    main()
