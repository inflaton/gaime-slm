#!/usr/bin/env python3
"""Train or evaluate one experiment from configs/experiments.csv."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml


TRAIN_ONLY_KEYS = {
    "data_seed",
    "dataset",
    "do_train",
    "eval_steps",
    "eval_strategy",
    "gradient_accumulation_steps",
    "learning_rate",
    "logging_steps",
    "lora_alpha",
    "lora_rank",
    "lora_target",
    "lr_scheduler_type",
    "num_train_epochs",
    "per_device_train_batch_size",
    "plot_loss",
    "save_only_model",
    "save_steps",
    "seed",
    "val_size",
    "warmup_ratio",
}


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["experiment_id"]: row for row in rows}


def run_config(config_path: Path, cli: str, root: Path, dry_run: bool) -> None:
    command = [cli, "train", str(config_path)]
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=root, check=True)


def prediction_config(
    training: dict[str, Any], row: dict[str, str], checkpoint: Path, output_dir: Path
) -> dict[str, Any]:
    config = {key: value for key, value in training.items() if key not in TRAIN_ONLY_KEYS}
    config.update(
        {
            "stage": "sft",
            "do_predict": True,
            "finetuning_type": "lora",
            "adapter_name_or_path": str(checkpoint),
            "dataset_dir": "data/llamafactory",
            "eval_dataset": (
                "merchant_matching_test" if row["prompt"] == "FT" else "merchant_matching_json_only_test"
            ),
            "overwrite_cache": True,
            "output_dir": str(output_dir),
            "overwrite_output_dir": True,
            "report_to": "none",
            "predict_with_generate": True,
            "do_sample": False,
            "temperature": 0.0,
            "ddp_timeout": 180000000,
        }
    )
    return config


def checkpoint_paths(output_dir: Path, best_step: str, all_checkpoints: bool) -> list[Path]:
    if all_checkpoints:
        checkpoints = sorted(
            (path for path in output_dir.glob("checkpoint-*") if path.is_dir()),
            key=lambda path: int(path.name.rsplit("-", 1)[1]),
        )
    else:
        checkpoints = [output_dir / f"checkpoint-{best_step}"]
    missing = [path for path in checkpoints if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing checkpoint directory: {missing[0]}")
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint directories under {output_dir}")
    return checkpoints


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "predict"))
    parser.add_argument("experiment_id")
    parser.add_argument("--all-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cli", default=shutil.which("llamafactory-cli") or "llamafactory-cli")
    args = parser.parse_args()

    manifest = read_manifest(root / "configs" / "experiments.csv")
    if args.experiment_id not in manifest:
        choices = ", ".join(sorted(manifest))
        raise SystemExit(f"unknown experiment {args.experiment_id!r}; choose one of: {choices}")
    row = manifest[args.experiment_id]
    training_path = root / row["config_path"]

    if args.action == "train":
        run_config(training_path, args.cli, root, args.dry_run)
        return

    training = yaml.safe_load(training_path.read_text(encoding="utf-8"))
    experiment_output = root / "outputs" / args.experiment_id
    for checkpoint in checkpoint_paths(experiment_output, row["best_step"], args.all_checkpoints):
        prediction_output = experiment_output / f"predict-{checkpoint.name}"
        config_path = prediction_output / "predict_config.yaml"
        config = prediction_config(training, row, checkpoint, prediction_output)
        print(f"prepare {config_path.relative_to(root)}")
        if not args.dry_run:
            prediction_output.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        run_config(config_path, args.cli, root, args.dry_run)


if __name__ == "__main__":
    main()
