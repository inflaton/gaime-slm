#!/usr/bin/env python3
"""Generate deterministic two-prompt base-model evaluation configs."""

from __future__ import annotations

from pathlib import Path

import yaml


MODELS = (
    ("gemma-3-270m", "google/gemma-3-270m-it", "ac82b4e820549b854eebf28ce6dedaf9fdfa17b3", "gemma", 4),
    ("gemma-3-1b", "google/gemma-3-1b-it", "dcc83ea841ab6100d6b47a070329e1ba4cf78752", "gemma", 4),
    ("gemma-3-4b", "google/gemma-3-4b-it", "093f9f388b31de276ce2de164bdc2081324b9767", "gemma3", 4),
    ("qwen-3.5-0.8b", "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17", "qwen3_5", 4),
    ("qwen-3.5-2b", "Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc", "qwen3_5", 4),
    ("qwen-3.5-4b", "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "qwen3_5", 4),
    ("aya-3.35b", "CohereLabs/tiny-aya-global", "b2f40192d990d66a6f28a4b90d72817ab9c613da", "cohere", 4),
    ("llama-3.1-8b", "meta-llama/Meta-Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659", "llama3", 2),
)
PROMPTS = (
    ("free-thinking", "merchant_matching_test"),
    ("json-only", "merchant_matching_json_only_test"),
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "configs" / "zero_shot"
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_id, model_name, model_revision, template, batch_size in MODELS:
        for prompt_id, dataset in PROMPTS:
            experiment_id = f"{model_id}-{prompt_id}"
            config = {
                "model_name_or_path": model_name,
                "model_revision": model_revision,
                "trust_remote_code": True,
                "stage": "sft",
                "do_predict": True,
                "finetuning_type": "lora",
                "eval_dataset": dataset,
                "dataset_dir": "data/llamafactory",
                "template": template,
                "cutoff_len": 2048,
                "overwrite_cache": True,
                "preprocessing_num_workers": 16,
                "dataloader_num_workers": 4,
                "output_dir": f"outputs/zero-shot/{experiment_id}",
                "overwrite_output_dir": True,
                "report_to": "none",
                "per_device_eval_batch_size": batch_size,
                "predict_with_generate": True,
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 0.7,
                "top_k": 50,
                "num_beams": 1,
                "max_new_tokens": 1024,
                "seed": 42,
                "bf16": True,
                "ddp_timeout": 180000000,
            }
            destination = output_dir / f"{experiment_id}.yaml"
            destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            print(destination.relative_to(root))


if __name__ == "__main__":
    main()
