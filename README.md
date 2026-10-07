# GAIME-SLM: Merchant Information Extraction with Small Language Models

Research code and aggregate results accompanying **“How Small Can You Go? LoRA Fine-Tuning 270M–8B Models for Merchant Information Extraction in Financial Transactions,”** by Donghao Huang, Tomáš Drietomský, Benjamin Barrett, and Zhaoxia Wang, accepted at IEEE ICDM 2026. [Read the paper](https://arxiv.org/abs/2606.08051).

**Approved for public release.** This package contains prompts, training and evaluation code, configurations, and aggregate model results. The confidential merchant datasets, per-record identifiers, individual predictions, endpoint request exports, trained weights, and logs are excluded. Merchant-related examples in the public prompts use synthetic replacements. The aggregate tables retain the recorded metrics; private input/output paths and file hashes have been removed. No private Git history is included.

## Contents

| Directory | Contents |
|---|---|
| `prompts/` | Free-Thinking and JSON-Only templates, with synthetic merchant examples and the same three example omissions disclosed in Appendix A |
| `configs/` | Manifest and configurations for 23 fine-tuning runs and 16 zero-shot evaluations |
| `scripts/` | Data preparation, training orchestration, evaluation, and aggregate import helpers |
| `patches/` | The retained LLaMA-Factory source changes used by the experiments |
| `results/` | Model/checkpoint metrics, per-field totals, throughput, compute accounting, and selected environment metadata |
| `data/` | Dataset registration schema and selected aggregate runtime/benchmark measurements; no merchant records |
| `tests/` | Source checks using invented temporary fixtures |

The best sub-8B configuration is Qwen 3.5 4B JSON-Only with LoRA rank 8. Its recorded common-subset canonicalized field F1 is 0.9660 and strict record exact match is 0.9167. Checkpoints were selected on test performance, so the reported best values are oracle characterizations. Canonicalization uppercases values and removes spaces, `*`, commas, and hyphens; strict metrics preserve those differences.

## Check the source without research data

Use Python 3.11 or newer. The synthetic source tests use the standard library and Bash; they require no model downloads, GPU, or merchant data:

```bash
python3 -m unittest discover -s tests -v
```

## Use your own data

Create local `data/raw/train.csv`, `validation.csv`, and `test.csv` with a `TRANSACTION` column and these ten string-valued columns:

```text
INTERMEDIARY,MERCHANT_NAME,DYNAMIC_DESCRIPTOR,STREET,ZIP,CITY,STATE,COUNTRY,PHONE,WEBSITE
```

Use empty strings for absent entities. All three splits must be nonempty. Prepare both prompt variants with:

```bash
python3 scripts/prepare_data.py --allow-custom-counts
```

Without `--allow-custom-counts`, preparation enforces the original paper's split sizes. Generated datasets, model outputs, and local inputs are ignored by Git. Using your own records or the public prompts with synthetic examples is a new experiment and does not reproduce the paper's exact inputs.

The historical fine-tuning framework was LLaMA-Factory `0.9.5.dev0`, based on public commit `07798465134da539993c21b263db541e07d47522` with the retained patch. For the closest available reconstruction:

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
git checkout 07798465134da539993c21b263db541e07d47522
git apply ../gaime-slm/patches/llamafactory-private-head.patch
pip install -e .
cd ../gaime-slm
pip install -r requirements.txt
python scripts/run_experiment.py train gemma-3-270m-jo-r8 --dry-run
```

Remove `--dry-run` to train after preparing your data and installing model-specific dependencies. GPU requirements and upstream model licenses apply. Historical fine-tuning base-model revisions and several package versions were not retained; `requirements-training-known.txt` is a partial record, not a complete lock file. `requirements-zero-shot.txt` records the later zero-shot environment.

Score your own prediction JSONL against your own labelled reference CSV:

```bash
python3 scripts/evaluate_predictions.py /path/to/predictions.jsonl \
  --test-csv /path/to/references.csv
```

Each JSONL object needs a `predict` value containing the predicted entity JSON. A prompt with a `TRANSACTION:` marker or an explicit `sample_id` maps predictions to references; a complete, ordered file may use positional mapping. Use `--subset-manifest` only with a manifest built for your own data. Schema validation and canonical/strict field and record metrics are reported separately.

## Aggregate results and reproducibility limits

The package retains model-level and checkpoint-level metrics, including per-field totals; it contains no per-merchant results. See [results documentation](results/README.md). Sanitized benchmark metadata summarizes repeated runs, not individual requests. The public prompts omit the three examples elided from [Appendix A](https://arxiv.org/html/2606.08051v4) and replace the remaining merchant-related examples with synthetic values; see [prompt notes](prompts/README.md).

The paper's Reproducibility section anticipated a public snapshot that would include the dataset splits, the 1,213-record common evaluation index, and the per-request endpoint exports for the deployed variants. Following confidentiality and intellectual-property review, those items are withheld. The confidential merchant datasets, their derived splits and common-subset identifiers, per-record predictions, and per-request endpoint exports are not published here. This package is limited to methodology, code, configurations, prompts, and aggregate results. Where the paper's planned release and this package differ, this package governs what is actually distributed.

Readers can inspect the methodology, run the tests, and adapt the pipeline to their own data. Exact independent reconstruction of the reported study metrics is not possible from this package because the original datasets, common-subset identifiers, predictions, and trained adapters are withheld.

`run_zero_shot.sh`, `build_subset_manifest.py`, `rebuild_metrics.py`, and the import helpers retain the original experiment protocol and require locally supplied inputs. They are included as source references. In particular, the benchmark importer expects original provenance fields that have been removed from this package, and cannot rebuild its CSV from the sanitized metadata. The historical benchmark launch scripts and production-reference importer embed private artifact hashes and are excluded. Generated outputs are not automatically suitable for redistribution.

## License and citation

The repository's existing Apache-2.0 software license is included; it does not grant rights to the withheld dataset or override upstream model licenses. Cite the paper using `CITATION.cff`. Corresponding author: Zhaoxia Wang — zxwang@smu.edu.sg.
