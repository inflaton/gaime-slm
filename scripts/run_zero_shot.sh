#!/usr/bin/env bash
# Run adapter-free merchant-matching evaluations on one NVIDIA GPU.
#
# Runs on discrete-VRAM cards and on unified-memory devices alike: an NVIDIA
# GB10 (DGX Spark) reports "Not Supported" for per-GPU memory because the GPU
# shares system LPDDR5x, so the memory probe below is required only when
# LLAMA_8B_MODE=auto has to pick a precision from it.
#
# The canonical configs remain unchanged. Runtime copies use batch size 1 by
# default for 16 GB cards. With no experiment IDs, the eight Free-Thinking
# configurations run; select JSON-Only later with PROMPT_SET=jo. Every run,
# including the 8B Llama, defaults to bfloat16, which is the precision the
# camera-ready reports; LLAMA_8B_MODE=4bit remains available for GPUs that
# cannot hold the 8B model in bfloat16 and tags its artifacts -bnb4-nf4.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="$ROOT_DIR/configs/zero_shot"
RUNTIME_CONFIG_DIR="$ROOT_DIR/outputs/zero-shot/_runtime_configs"
RESULT_DIR="$ROOT_DIR/results/zero_shot"
PREDICTION_RELEASE_DIR="$ROOT_DIR/results/predictions/zero_shot"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PREPROCESSING_WORKERS="${PREPROCESSING_WORKERS:-8}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-2}"
LLAMA_8B_MODE="${LLAMA_8B_MODE:-bf16}" # bf16, 4bit, auto, or skip
PROMPT_SET="${PROMPT_SET:-ft}" # ft, jo, or all when no experiment IDs are given
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
EXPECTED_ROWS=1222

usage() {
  cat <<'EOF'
Usage:
  scripts/run_zero_shot.sh [experiment-id ...]

With no experiment IDs, the eight Free-Thinking runs execute sequentially.
Set PROMPT_SET=jo for the eight JSON-Only runs, or PROMPT_SET=all for all 16.
Explicit experiment IDs are always honored regardless of PROMPT_SET.
Completed 1,222-row outputs are skipped unless FORCE=1.

Useful environment variables:
  PYTHON_BIN=python3                 Python with LLaMA-Factory dependencies
  CUDA_DEVICE=0                     Physical GPU index for nvidia-smi/CUDA
  BATCH_SIZE=1                      Safe default for a 16 GB card
  LLAMA_8B_MODE=bf16                bf16|4bit|auto|skip ('auto' needs a device
                                    that reports per-GPU memory)
  PROMPT_SET=ft                     ft|jo|all (when no IDs are supplied)
  FORCE=1                           Rerun completed outputs
  DRY_RUN=1                         Validate and print the run plan only

Examples:
  ./scripts/run_zero_shot.sh                       # eight FT runs
  PROMPT_SET=jo ./scripts/run_zero_shot.sh         # eight JO runs
  PROMPT_SET=all ./scripts/run_zero_shot.sh        # all 16 runs
  ./scripts/run_zero_shot.sh qwen-3.5-4b-json-only
  LLAMA_8B_MODE=skip ./scripts/run_zero_shot.sh
  LLAMA_8B_MODE=bf16 ./scripts/run_zero_shot.sh llama-3.1-8b-free-thinking
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE must be a positive integer." >&2
  exit 2
fi
if [[ ! "$PREPROCESSING_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "PREPROCESSING_WORKERS must be a positive integer." >&2
  exit 2
fi
if [[ ! "$DATALOADER_WORKERS" =~ ^[0-9]+$ ]]; then
  echo "DATALOADER_WORKERS must be a non-negative integer." >&2
  exit 2
fi
case "$LLAMA_8B_MODE" in
  auto|bf16|4bit|skip) ;;
  *) echo "LLAMA_8B_MODE must be auto, bf16, 4bit, or skip." >&2; exit 2 ;;
esac
case "$PROMPT_SET" in
  ft|jo|all) ;;
  *) echo "PROMPT_SET must be ft, jo, or all." >&2; exit 2 ;;
esac

cd "$ROOT_DIR"
mkdir -p "$RUNTIME_CONFIG_DIR" "$RESULT_DIR/metrics" "$RESULT_DIR/per_field" \
  "$RESULT_DIR/runs" "$PREDICTION_RELEASE_DIR"

"$PYTHON_BIN" - <<'PY'
import importlib.util
missing = [
    name
    for name in ("torch", "yaml", "packaging", "transformers", "llamafactory")
    if importlib.util.find_spec(name) is None
]
if missing:
    raise SystemExit("Missing Python packages: " + ", ".join(missing))
PY

required_data=(
  data/raw/test.csv
  data/evaluation_subset.csv
  data/llamafactory/dataset_info.json
  data/llamafactory/test_merchant_matching.json
  data/llamafactory/test_merchant_matching_json_only.json
)
missing_data=0
for path in "${required_data[@]}"; do
  if [[ ! -f "$path" ]]; then
    missing_data=1
  fi
done
if [[ "$missing_data" == "1" ]]; then
  echo "Preparing LLaMA-Factory datasets..."
  "$PYTHON_BIN" scripts/prepare_data.py
fi

CONFIG_FILES=()
SELECTION_LABEL="$PROMPT_SET"
if [[ "$#" -gt 0 ]]; then
  SELECTION_LABEL=explicit
  for experiment_id in "$@"; do
    config_path="$CONFIG_DIR/$experiment_id.yaml"
    if [[ ! -f "$config_path" ]]; then
      echo "Unknown zero-shot experiment: $experiment_id" >&2
      exit 2
    fi
    CONFIG_FILES+=("$config_path")
  done
else
  case "$PROMPT_SET" in
    ft) config_pattern='*-free-thinking.yaml' ;;
    jo) config_pattern='*-json-only.yaml' ;;
    all) config_pattern='*.yaml' ;;
  esac
  while IFS= read -r config_path; do
    CONFIG_FILES+=("$config_path")
  done < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name "$config_pattern" | sort)
fi

if [[ "${#CONFIG_FILES[@]}" -eq 0 ]]; then
  echo "No zero-shot configurations found under $CONFIG_DIR." >&2
  exit 2
fi

SELECTS_LLAMA=0
for config_path in "${CONFIG_FILES[@]}"; do
  if [[ "$(basename "$config_path")" == llama-3.1-8b-* ]]; then
    SELECTS_LLAMA=1
    break
  fi
done

GPU_NAME="dry-run"
GPU_MEMORY_MIB=0
GPU_FREE_MEMORY_MIB=0
if [[ "$DRY_RUN" != "1" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required for this runner." >&2
    exit 2
  fi
  GPU_NAME="$(nvidia-smi --id="$CUDA_DEVICE" --query-gpu=name --format=csv,noheader | head -n 1 | xargs)"
  GPU_MEMORY_MIB="$(nvidia-smi --id="$CUDA_DEVICE" --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | xargs)"
  GPU_FREE_MEMORY_MIB="$(nvidia-smi --id="$CUDA_DEVICE" --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | xargs)"
  if [[ ! "$GPU_MEMORY_MIB" =~ ^[0-9]+$ || ! "$GPU_FREE_MEMORY_MIB" =~ ^[0-9]+$ ]]; then
    # Unified-memory devices (for example NVIDIA GB10 on a DGX Spark) report
    # "Not Supported" here because the GPU shares system memory instead of
    # owning discrete VRAM. The figure is only used to choose the 8B precision
    # automatically, so it is fatal only for LLAMA_8B_MODE=auto.
    if [[ "$LLAMA_8B_MODE" == "auto" ]]; then
      echo "Could not determine GPU memory for CUDA_DEVICE=$CUDA_DEVICE; set LLAMA_8B_MODE=bf16 or 4bit explicitly." >&2
      exit 2
    fi
    echo "nvidia-smi reports no per-GPU memory for ${GPU_NAME:-this device} (unified memory); recording 0 and honoring LLAMA_8B_MODE=$LLAMA_8B_MODE."
    GPU_MEMORY_MIB=0
    GPU_FREE_MEMORY_MIB=0
  fi
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  "$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA in this environment.")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("The selected GPU/runtime does not support bfloat16 inference.")
print(f"CUDA ready: {torch.cuda.get_device_name(0)}; torch={torch.__version__}; CUDA={torch.version.cuda}")
PY
fi

RESOLVED_LLAMA_MODE="$LLAMA_8B_MODE"
if [[ "$RESOLVED_LLAMA_MODE" == "auto" ]]; then
  if [[ "$GPU_FREE_MEMORY_MIB" -ge 22000 ]]; then
    RESOLVED_LLAMA_MODE=bf16
  else
    RESOLVED_LLAMA_MODE=4bit
  fi
fi
if [[ "$DRY_RUN" == "1" && "$LLAMA_8B_MODE" == "auto" ]]; then
  RESOLVED_LLAMA_MODE=4bit
fi
if [[ "$SELECTS_LLAMA" == "1" && "$RESOLVED_LLAMA_MODE" == "4bit" && "$DRY_RUN" != "1" ]]; then
  "$PYTHON_BIN" - <<'PY'
import importlib.metadata
from packaging.version import Version
try:
    version = importlib.metadata.version("bitsandbytes")
except importlib.metadata.PackageNotFoundError:
    raise SystemExit("LLAMA_8B_MODE=4bit requires bitsandbytes>=0.39.0 (pip install 'bitsandbytes>=0.39').")
if Version(version) < Version("0.39.0"):
    raise SystemExit(f"bitsandbytes {version} is too old; install bitsandbytes>=0.39.0.")
print(f"bitsandbytes={version}; the 8B exception will be recorded as NF4 4-bit.")
PY
fi

LLAMAFACTORY_MODULE_PATH="$("$PYTHON_BIN" - <<'PY'
from pathlib import Path
import llamafactory
print(Path(llamafactory.__file__).resolve())
PY
)"
LLAMAFACTORY_SOURCE_DIR="$("$PYTHON_BIN" - "$LLAMAFACTORY_MODULE_PATH" <<'PY'
from pathlib import Path
import sys
module_path = Path(sys.argv[1])
for parent in (module_path.parent, *module_path.parents):
    if (parent / ".git").exists():
        print(parent)
        break
PY
)"
LLAMAFACTORY_COMMIT=unknown
LLAMAFACTORY_DIRTY=unknown
LLAMAFACTORY_DIFF_SHA256=unknown
LLAMAFACTORY_MODULE_RECORD="llamafactory/__init__.py"
if [[ -n "$LLAMAFACTORY_SOURCE_DIR" ]] && git -C "$LLAMAFACTORY_SOURCE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  LLAMAFACTORY_COMMIT="$(git -C "$LLAMAFACTORY_SOURCE_DIR" rev-parse HEAD)"
  LLAMAFACTORY_MODULE_RECORD="$("$PYTHON_BIN" - "$LLAMAFACTORY_SOURCE_DIR" "$LLAMAFACTORY_MODULE_PATH" <<'PY'
from pathlib import Path
import sys
source, module = map(lambda value: Path(value).resolve(), sys.argv[1:3])
try:
    print(module.relative_to(source).as_posix())
except ValueError:
    print("llamafactory/__init__.py")
PY
)"
  LLAMAFACTORY_DIRTY=false
  if ! git -C "$LLAMAFACTORY_SOURCE_DIR" diff --quiet --ignore-submodules -- || \
     ! git -C "$LLAMAFACTORY_SOURCE_DIR" diff --cached --quiet --ignore-submodules --; then
    LLAMAFACTORY_DIRTY=true
    LLAMAFACTORY_DIFF_SHA256="$(git -C "$LLAMAFACTORY_SOURCE_DIR" diff --binary HEAD -- | shasum -a 256 | awk '{print $1}')"
  fi
fi

echo "GPU: $GPU_NAME (${GPU_MEMORY_MIB} MiB total; ${GPU_FREE_MEMORY_MIB} MiB free)"
echo "Runs: ${#CONFIG_FILES[@]}; selection: $SELECTION_LABEL; batch size: $BATCH_SIZE; Llama 8B mode: $RESOLVED_LLAMA_MODE"
echo "LLaMA-Factory: $LLAMAFACTORY_MODULE_PATH (commit $LLAMAFACTORY_COMMIT; dirty=$LLAMAFACTORY_DIRTY)"

for canonical_config in "${CONFIG_FILES[@]}"; do
  experiment_id="$(basename "$canonical_config" .yaml)"
  if [[ "$experiment_id" == llama-3.1-8b-* && "$RESOLVED_LLAMA_MODE" == "skip" ]]; then
    echo "[$experiment_id] skipped by LLAMA_8B_MODE=skip"
    continue
  fi

  runtime_mode=bf16
  if [[ "$experiment_id" == llama-3.1-8b-* ]]; then
    runtime_mode="$RESOLVED_LLAMA_MODE"
  fi
  artifact_id="$experiment_id"
  if [[ "$runtime_mode" == "4bit" ]]; then
    artifact_id="${experiment_id}-bnb4-nf4"
  fi
  runtime_config="$RUNTIME_CONFIG_DIR/$artifact_id.yaml"

  "$PYTHON_BIN" - "$canonical_config" "$runtime_config" "$BATCH_SIZE" \
    "$PREPROCESSING_WORKERS" "$DATALOADER_WORKERS" "$runtime_mode" "$artifact_id" <<'PY'
from pathlib import Path
import sys
import yaml

source, destination = map(Path, sys.argv[1:3])
batch_size, prep_workers, loader_workers = map(int, sys.argv[3:6])
mode = sys.argv[6]
artifact_id = sys.argv[7]
config = yaml.safe_load(source.read_text(encoding="utf-8"))
if config.get("adapter_name_or_path"):
    raise SystemExit(f"Refusing adapter-bearing config: {source}")
config.pop("max_samples", None)
config["per_device_eval_batch_size"] = batch_size
config["preprocessing_num_workers"] = prep_workers
config["dataloader_num_workers"] = loader_workers
config["output_dir"] = f"outputs/zero-shot/{artifact_id}"
if mode == "4bit":
    config.update(
        quantization_method="bnb",
        quantization_bit=4,
        quantization_type="nf4",
        double_quantization=True,
        low_cpu_mem_usage=True,
    )
else:
    for key in ("quantization_method", "quantization_bit", "quantization_type", "double_quantization", "low_cpu_mem_usage"):
        config.pop(key, None)
destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

  output_dir="$ROOT_DIR/outputs/zero-shot/$artifact_id"
  raw_predictions="$output_dir/generated_predictions.jsonl"
  run_log="$output_dir/run.log"
  run_metadata="$RESULT_DIR/runs/$artifact_id.json"
  mkdir -p "$output_dir"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[$artifact_id] would run $runtime_mode using $runtime_config"
    continue
  fi

  started_at=""
  resume_ok=0
  if [[ "$FORCE" != "1" && -f "$raw_predictions" && -f "$run_metadata" ]]; then
    if "$PYTHON_BIN" - "$run_metadata" "$runtime_config" "$raw_predictions" \
      "$artifact_id" "$runtime_mode" "$EXPECTED_ROWS" <<'PY'
from pathlib import Path
import hashlib
import json
import sys

metadata_path, config_path, raw_path = map(Path, sys.argv[1:4])
artifact_id, precision_mode, expected_rows = sys.argv[4], sys.argv[5], int(sys.argv[6])
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
checks = (
    metadata.get("status") == "complete",
    metadata.get("artifact_id") == artifact_id,
    metadata.get("precision_mode") == precision_mode,
    metadata.get("adapter_name_or_path") is None,
    metadata.get("records") == expected_rows,
    metadata.get("runtime_config_sha256") == digest(config_path),
    metadata.get("raw_prediction_sha256") == digest(raw_path),
    sum(1 for line in raw_path.open(encoding="utf-8") if line.strip()) == expected_rows,
)
raise SystemExit(0 if all(checks) else 1)
PY
    then
      resume_ok=1
    fi
  fi

  if [[ "$resume_ok" == "1" ]]; then
    echo "[$artifact_id] verified completed generation; rebuilding derived results"
  else
    started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[$artifact_id] generating ($runtime_mode)..."
    set +e
    "$PYTHON_BIN" -m llamafactory.cli train "$runtime_config" 2>&1 | tee "$run_log"
    run_status="${PIPESTATUS[0]}"
    set -e
    if [[ "$run_status" -ne 0 ]]; then
      echo "[$artifact_id] failed; see $run_log" >&2
      exit "$run_status"
    fi
  fi

  if [[ ! -f "$raw_predictions" ]]; then
    echo "[$artifact_id] missing $raw_predictions after inference." >&2
    exit 1
  fi
  prediction_rows="$(wc -l < "$raw_predictions" | xargs)"
  if [[ "$prediction_rows" != "$EXPECTED_ROWS" ]]; then
    echo "[$artifact_id] expected $EXPECTED_ROWS predictions, found $prediction_rows." >&2
    exit 1
  fi

  compact_predictions="$PREDICTION_RELEASE_DIR/$artifact_id.jsonl.gz"
  compact_predictions_relative="results/predictions/zero_shot/$artifact_id.jsonl.gz"
  "$PYTHON_BIN" - "$ROOT_DIR" "$raw_predictions" "$compact_predictions" <<'PY'
from pathlib import Path
import gzip
import io
import json
import sys

root, source, destination = map(Path, sys.argv[1:4])
sys.path.insert(0, str(root / "scripts"))
from evaluate_predictions import extract_transaction, load_references, make_sample_id

references = load_references(root / "data/raw/test.csv")
reference_ids = {reference.sample_id for reference in references}
mapped = {}
with source.open(encoding="utf-8") as incoming:
    for line_number, line in enumerate(incoming, start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or "predict" not in record:
            raise SystemExit(f"{source}:{line_number}: missing prediction object")
        transaction = extract_transaction(record.get("prompt"))
        if transaction is None:
            raise SystemExit(f"{source}:{line_number}: cannot extract transaction from prompt")
        sample_id = make_sample_id(transaction)
        if sample_id not in reference_ids:
            raise SystemExit(f"{source}:{line_number}: prompt does not map to the test set")
        if sample_id in mapped:
            raise SystemExit(f"{source}:{line_number}: duplicate test prompt")
        mapped[sample_id] = record
if set(mapped) != reference_ids:
    raise SystemExit(f"Strict prompt mapping covered {len(mapped)}/{len(reference_ids)} test records")

destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open("wb") as raw_out:
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw_out, mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as outgoing:
            for reference in references:
                record = mapped[reference.sample_id]
                outgoing.write(json.dumps({"predict": record["predict"]}, ensure_ascii=False, separators=(",", ":")) + "\n")
PY

  common_metrics="$RESULT_DIR/metrics/$artifact_id-common.json"
  full_metrics="$RESULT_DIR/metrics/$artifact_id-full.json"
  "$PYTHON_BIN" scripts/evaluate_predictions.py "$compact_predictions_relative" \
    --subset-manifest data/evaluation_subset.csv \
    --output-json "$common_metrics" \
    --details-csv "$RESULT_DIR/per_field/$artifact_id-common.csv" >/dev/null
  "$PYTHON_BIN" scripts/evaluate_predictions.py "$compact_predictions_relative" \
    --output-json "$full_metrics" \
    --details-csv "$RESULT_DIR/per_field/$artifact_id-full.csv" >/dev/null

  if [[ "$resume_ok" == "1" ]]; then
    "$PYTHON_BIN" - "$run_metadata" "$compact_predictions" "$common_metrics" "$full_metrics" <<'PY'
from pathlib import Path
import hashlib
import json
import sys

metadata_path, compact_path, common_path, full_path = map(Path, sys.argv[1:])
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
compact_sha256 = hashlib.sha256(compact_path.read_bytes()).hexdigest()
common = json.loads(common_path.read_text(encoding="utf-8"))
full = json.loads(full_path.read_text(encoding="utf-8"))
checks = (
    metadata.get("compact_prediction_sha256") == compact_sha256,
    metadata.get("common_metrics") == common,
    metadata.get("full_metrics") == full,
)
if not all(checks):
    raise SystemExit(
        "Rebuilt derived artifacts differ from the retained run record; "
        "preserving generation provenance and refusing to rewrite metadata."
    )
PY
    echo "[$artifact_id] derived results verified; retained original generation metadata"
    continue
  fi

  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$PYTHON_BIN" - "$experiment_id" "$artifact_id" "$canonical_config" "$runtime_config" "$raw_predictions" \
    "$compact_predictions" "$common_metrics" "$full_metrics" "$runtime_mode" \
    "$GPU_NAME" "$GPU_MEMORY_MIB" "$GPU_FREE_MEMORY_MIB" "$LLAMAFACTORY_MODULE_RECORD" \
    "$LLAMAFACTORY_COMMIT" "$LLAMAFACTORY_DIRTY" "$LLAMAFACTORY_DIFF_SHA256" "$started_at" "$finished_at" \
    "$run_log" "$run_metadata" <<'PY'
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import sys
import yaml

(
    experiment_id, artifact_id, canonical_config_path, config_path, raw_path,
    compact_path, common_path, full_path, precision_mode, gpu_name,
    gpu_memory_mib, gpu_free_memory_mib, lf_module_path, lf_commit, lf_dirty,
    lf_diff_sha256, started_at, finished_at, log_path, destination,
) = sys.argv[1:]

def digest(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
metadata = {
    "experiment_id": experiment_id,
    "artifact_id": artifact_id,
    "status": "complete",
    "model_name_or_path": config["model_name_or_path"],
    "model_revision": config["model_revision"],
    "eval_dataset": config["eval_dataset"],
    "template": config["template"],
    "adapter_name_or_path": None,
    "precision_mode": precision_mode,
    "quantization": "bitsandbytes-nf4-4bit" if precision_mode == "4bit" else "none",
    "batch_size": config["per_device_eval_batch_size"],
    "records": sum(1 for _ in Path(raw_path).open(encoding="utf-8")),
    "canonical_config": str(Path(canonical_config_path).relative_to(Path.cwd())),
    "canonical_config_sha256": digest(canonical_config_path),
    "runtime_config": str(Path(config_path).relative_to(Path.cwd())),
    "runtime_config_sha256": digest(config_path),
    "raw_prediction_sha256": digest(raw_path),
    "compact_prediction": str(Path(compact_path).relative_to(Path.cwd())),
    "compact_prediction_sha256": digest(compact_path),
    "gpu_name": gpu_name,
    "gpu_memory_mib": int(gpu_memory_mib),
    "gpu_free_memory_mib_at_start": int(gpu_free_memory_mib),
    "llamafactory_module_path": lf_module_path,
    "llamafactory_commit": lf_commit,
    "llamafactory_dirty": None if lf_dirty == "unknown" else lf_dirty == "true",
    "llamafactory_diff_sha256": None if lf_diff_sha256 == "unknown" else lf_diff_sha256,
    "python": platform.python_version(),
    "packages": {},
    "started_at": started_at,
    "finished_at": finished_at,
    "common_metrics": json.loads(Path(common_path).read_text(encoding="utf-8")),
    "full_metrics": json.loads(Path(full_path).read_text(encoding="utf-8")),
}
for name in ("torch", "transformers", "peft", "accelerate", "llamafactory", "bitsandbytes"):
    try:
        metadata["packages"][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        metadata["packages"][name] = None
if Path(log_path).is_file():
    metadata["run_log_sha256"] = digest(log_path)
Path(destination).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

  echo "[$artifact_id] complete"
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run complete; no inference was started."
  exit 0
fi

"$PYTHON_BIN" - "$RESULT_DIR" <<'PY'
from pathlib import Path
import csv
import json
import sys

root = Path(sys.argv[1])
records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((root / "runs").glob("*.json"))]
rows = []
for record in records:
    common = record["common_metrics"]
    full = record["full_metrics"]
    rows.append({
        "artifact_id": record["artifact_id"],
        "experiment_id": record["experiment_id"],
        "model_name_or_path": record["model_name_or_path"],
        "model_revision": record["model_revision"],
        "prompt": "FT" if record["eval_dataset"] == "merchant_matching_test" else "JO",
        "template": record["template"],
        "precision_mode": record["precision_mode"],
        "quantization": record["quantization"],
        "common_n": common["n"],
        "common_canonical_precision": common["canonical_precision"],
        "common_canonical_recall": common["canonical_recall"],
        "common_canonical_f1": common["canonical_f1"],
        "common_strict_record_exact_match": common["strict_record_exact_match"],
        "common_parse_rate": common["parse_rate"],
        "common_schema_valid_rate": common["schema_valid_rate"],
        "full_n": full["n"],
        "full_canonical_f1": full["canonical_f1"],
        "full_strict_record_exact_match": full["strict_record_exact_match"],
        "raw_prediction_sha256": record["raw_prediction_sha256"],
        "compact_prediction_sha256": record["compact_prediction_sha256"],
    })

destination = root / "zero_shot_metrics.csv"
if rows:
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
for row in rows:
    print(
        f"{row['artifact_id']:<47} "
        f"F1={float(row['common_canonical_f1']):.4f} "
        f"R-EM={float(row['common_strict_record_exact_match']):.4f} "
        f"mode={row['precision_mode']}"
    )
print(f"Wrote {destination} ({len(rows)} completed runs).")
PY
