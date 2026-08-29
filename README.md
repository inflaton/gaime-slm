# GAIME-SLM

Artifact repository for the IEEE ICDM 2026 paper **"How Small Can You Go? LoRA
Fine-Tuning 270M–8B Models for Merchant Information Extraction in Financial
Transactions"** — Donghao Huang, Tomáš Drietomský, Benjamin Barrett, Zhaoxia Wang.
26th IEEE International Conference on Data Mining, Shenyang, China, 12–15 November 2026.

## Status

**The artifact is not published yet.** The dataset is derived from Mastercard
transaction data, so the snapshot cannot be released until an internal
data-release review completes and the dataset terms are approved. This
repository is the permanent home the paper points to; the approved snapshot
will be published here when that review clears.

You are in the right place — nothing has moved.

## What the release will contain

- The labeled dataset and the stable identifiers for the 1,213-record common
  evaluation subset, subject to the separately approved dataset terms.
- Both prompt templates (Free-Thinking and JSON-Only) and portable
  LLaMA-Factory configurations for all 23 retained fine-tuning runs.
- Deterministic evaluation code, and the aggregate results behind every table
  in the paper: checkpoint metrics, zero-shot baselines, per-field scores, the
  managed-endpoint comparison, and the matched batch-4 throughput benchmark
  together with the harness that produced it.
- Compact prediction archives and a release-integrity test suite that
  recomputes the published numbers from them.

Model weights and optimizer states are intentionally excluded. Base-model
licenses and gated-access requirements continue to apply.

## Licensing

Repository code is licensed under the Apache License 2.0. That license does not
grant dataset redistribution rights; dataset release terms are handled
separately and will be stated with the data.

## Citation

Until the artifact is published, cite the paper. A `CITATION.cff` with the
proceedings DOI ships with the release.

## Contact

Corresponding author: Zhaoxia Wang — zxwang@smu.edu.sg

_Last updated: 29 August 2026._
