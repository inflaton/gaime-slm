# Aggregate results

This directory contains aggregate model and checkpoint results. Per-record predictions, merchant descriptors, sample identifiers, raw reference labels, private input/output file hashes, and internal source paths are excluded.

- `best_checkpoint_metrics.csv` contains the selected checkpoints; `checkpoint_metrics.csv` contains the larger retained sweep.
- `published_checkpoint_metrics.csv`, `published_think_vs_nothink.csv`, and `published_per_field_metrics.csv` retain the experiment aggregates used to prepare the paper. “Published” refers to their role in the study, not to a release authorization for their underlying records.
- `benchmark_throughput.csv` contains the matched batch-4 throughput figures. The legacy sweep throughput columns in the checkpoint/Think-Nothink tables and `nothink_checkpoint_throughput.csv` mix batch sizes and are retained as historical measurements, not comparable performance estimates.
- `endpoint_metrics.csv` contains model-level scores and counts for the managed-serving experiments; per-request endpoint exports are withheld.
- `baseline_reference_metrics.json` contains aggregate scores and counts for the separately trained production reference, with private source and match-vector provenance removed. Its historical rank-32 attribution is author-attested because the adapter/configuration metadata was not retained; the underlying prediction evidence remains internal.
- `zero_shot/` contains the 16 no-adapter aggregate comparisons, per-field totals, and selected run metadata. Predictions and logs are withheld.
- `compute_accounting.csv` and `training_environment.json` describe recorded compute use and recoverable software facts. Missing historical versions remain explicit.

Aggregate numeric values are retained from the internal artifacts. Columns and JSON fields containing private file paths, record/output hashes, and other withheld provenance have been removed. Public upstream model revisions may remain to identify the base models.

The checkpoint selection uses test performance and is an oracle characterization. The original datasets, adapters, and record-level outputs are unavailable in this package, so its original end-to-end metric reconstruction tests cannot run here. The included source tests instead check scoring and preparation behavior with invented fixtures.
