# Zero-shot aggregates

The study includes 16 no-adapter comparisons: eight Free-Thinking and eight JSON-Only runs. This directory retains their common/full-test metrics, per-field totals, and selected model, software, and hardware metadata. Private input/output paths and artifact hashes have been removed. Individual predictions and logs are withheld.

All reported runs used bfloat16. The 8B LLaMA evaluations required a larger GPU than the 16 GB device used for the other models. Source configurations retain the paper's protocol; adapt them to your own datasets and hardware to run a new experiment. The original common-subset identifiers are not supplied.

`scripts/run_zero_shot.sh` expects the original protocol's test count and common-subset manifest. For custom data, prepare your own CSVs and invoke an appropriate configuration directly through LLaMA-Factory. Four-bit or skipped-model fallbacks are not precision-matched reproductions of the reported results.
