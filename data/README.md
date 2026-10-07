# Data interface and aggregate measurements

The original merchant dataset is confidential and is not distributed. This directory contains the six LLaMA-Factory dataset registrations and selected aggregate benchmark/runtime measurements only. There are no raw descriptors, labels, common-subset identifiers, prepared training examples, or per-request exports.

To use your own data, create local `raw/train.csv`, `raw/validation.csv`, and `raw/test.csv` following the schema in the repository README, then run `python3 scripts/prepare_data.py --allow-custom-counts`. The generated datasets remain local.

`models/` contains aggregate runtime JSONs for the historical Nothink checkpoint sweeps, not weights or prediction text. `benchmarks/` contains selected run-level measurements and software metadata with private paths and artifact hashes removed. These sanitized copies support inspection of the aggregate measurements; they do not preserve the internal provenance chain required by the original benchmark importer.
