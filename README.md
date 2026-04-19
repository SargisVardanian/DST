# DSGD on RIPPER/FOIL Rules

This repository contains a frozen-rule evidential classifier for tabular data. The pipeline induces rules with FOIL or RIPPER, freezes the rule base, and learns Dempster-Shafer masses on top of the same fixed rules.

User-facing entry points now live at the repository root. The implementation lives under `src/`, which should be treated as the internal runtime package rather than the preferred manual entry path.

Canonical workflow:
- `train_test_runner.py` for benchmark/train-test runs
- `build_report.py` for aggregation/report generation
- `analyze_hard_cases.py` for hard-case diagnostics
- `app.py` for the local Streamlit UI

Everything else under `src/` should be treated as implementation detail, internal support code, or exploratory tooling unless explicitly called out below.

## What Is Here
- `train_test_runner.py`: canonical training and evaluation entry point
- `src/rule_generator.py`: FOIL/RIPPER rule induction and pool shaping
- `src/DSModelMultiQ.py`: rule activation, evidential fusion, probability output
- `src/DSClassifierMultiQ.py`: training loop for learned rule masses
- `build_report.py`: canonical benchmark aggregation and report generation
- `analyze_hard_cases.py`: hard-case evaluation
- `sample_rule_inspector.py`: web-first sample inspection helpers, combined-rule payloads, and validation checks
- `app.py`: Streamlit entry point for training, generation summaries, manual inspection, and bundle export

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn scipy torch matplotlib wittgenstein
```

Optional packages:
- `streamlit` for the local app
- `plotly` only if you extend optional plotting paths in `src/utils.py`
- `dill` only if you want it as a pickle backend fallback

For the local app:

```bash
pip install streamlit
```

## Data
To reproduce the benchmark, place the dataset CSV files in the project root. The current benchmark code expects files such as:

`adult.csv`, `bank-full.csv`, `BrainTumor.csv`, `breast-cancer-wisconsin.csv`, `df_wine.csv`, `dry-bean.csv`, `gas_drift.csv`, `german.csv`, `magic-gamma.csv`.

You can also adapt the loader in `src/Datasets_loader.py` for your own tabular datasets.

## Run
Train and evaluate on one dataset by direct path:

```bash
python train_test_runner.py \
  --dataset-path ./adult.csv \
  --inducers RIPPER,FOIL \
  --save-root ./src/results/raw_runs/adult_run
```

You can also use an absolute path:

```bash
python train_test_runner.py \
  --dataset-path /full/path/to/your/data.csv \
  --inducers RIPPER,FOIL \
  --save-root /full/path/to/output_dir
```

This saves everything under the chosen output directory:
- `benchmarks/`: metrics CSV files
- `rules/`: readable `.dsb` rule files with masses
- `models/`: `.pkl` payloads with learned weights, rule stats, and metrics

Run the current benchmark set with the standard protocol:

```bash
python train_test_runner.py \
  --datasets adult,bank-full,BrainTumor,breast-cancer-wisconsin,df_wine,dry-bean,gas_drift,german,magic-gamma \
  --inducers RIPPER,FOIL \
  --save-root src/results/raw_runs
```

Run the frozen paper protocol explicitly:

```bash
python train_test_runner.py \
  --datasets adult,bank-full,BrainTumor,breast-cancer-wisconsin,df_wine,dry-bean,gas_drift,german,magic-gamma \
  --inducers RIPPER,FOIL \
  --save-root src/results/raw_runs \
  --paper-mode
```

Build aggregated reports:

```bash
python build_report.py --out-root src/results/raw_runs --results-dir src/results --include-hard-cases
python analyze_hard_cases.py --out-root src/results/raw_runs --out-dir src/results/hard_cases
```

Inspect one sample:

```bash
python sample_rule_inspector.py \
  --dataset adult \
  --algo RIPPER \
  --row-index 123 \
  --split test \
  --combine-rule dempster \
  --mode export-combined \
  --out-root src/results/raw_runs
```

Run the local app:

```bash
streamlit run app.py
```

The app lets a user:
- upload a CSV
- choose output location and training parameters
- train FOIL/RIPPER models
- download the saved `rules/`, `models/`, and `benchmarks/` artifacts as a ZIP
- inspect how rules were generated, how many rules fired, and how the final combined rule was formed
- enter feature values manually with categorical drop-downs and numeric inputs
- load a sample that already activates rules so the combined explanation is visible immediately
- keep trained artifacts across refreshes because runs are stored under `src/results/raw_runs/app_sessions/`

The `Inspect` workflow is intentionally web-first:
- `How Rules Were Generated` explains the rule source, stage mix, pool shaping, depth mix, and top rules by support/precision
- `Activation And Combination` shows fired rules, the final combined condition, fused mass, and validation checks
- dataset-row inspection is available as a secondary convenience path, but the canonical surface is the manual/sample payload inspection view

## Current Benchmark Snapshot
Metrics below come from `src/results/ALL_DATASETS_metrics.csv`.

| Method | Acc | Macro-F1 | NLL | ECE |
|---|---:|---:|---:|---:|
| RF | 0.9249 | 0.8896 | 0.1985 | 0.0242 |
| RIPPER:dsgd_dempster | 0.9120 | 0.8721 | 0.2722 | 0.0733 |
| RIPPER:weighted_vote | 0.9080 | 0.8703 | 1.0994 | 0.0590 |
| FOIL:dsgd_dempster | 0.8956 | 0.8442 | 0.3012 | 0.0814 |
| FOIL:weighted_vote | 0.8935 | 0.8388 | 1.3962 | 0.0599 |
| FOIL:native_ordered_rule | 0.8886 | 0.8289 | 3.0789 | 0.1114 |
| FOIL:first_hit_laplace | 0.8886 | 0.8289 | 0.4112 | 0.0757 |
| RIPPER:native_ordered_rule | 0.8531 | 0.7755 | 4.0582 | 0.1469 |
| RIPPER:first_hit_laplace | 0.8531 | 0.7755 | 0.5364 | 0.1091 |

The current snapshot aggregates five training seeds on one fixed train/test split per dataset. In that fixed-split multi-seed view, the learned Dempster rows sit near the top of the rule-based averages shown here, while `RF` remains the strongest overall external reference. That should still be read as a descriptive benchmark result, not as evidence of split-robust dominance or of a broader mechanism.

Protocol note:
- Standard runs use the seeds and `--test-size` you provide.
- The fixed `seed=42`, `test_size=0.2` paper protocol is only activated when you pass `--paper-mode`.
- Each benchmark run writes a manifest JSON recording the effective protocol and any overrides.

Research-supporting artifacts:
- computational cost summary: `src/results/COMPUTATIONAL_COST.md`
- pool-shaping ablation status: `src/results/POOL_SHAPING_ABLATION.md`
- the current pool-shaping report is still research debt unless matching on/off rows exist for the same dataset, seed, and split

## Outputs
- benchmark artifacts from `train_test_runner.py`: `src/results/raw_runs/`
- app-private session storage: `src/results/raw_runs/app_sessions/`
- sample-level inspection exports: `src/results/raw_runs/inspection/`
- final benchmark tables and plots: `src/results/`

For custom runs with `--save-root`, the main outputs are:
- `<save-root>/benchmarks/*.csv`
- `<save-root>/rules/*.dsb`
- `<save-root>/models/*.pkl`

## Notes
- In the current descriptive snapshot, learned evidential fusion often improves over the raw rule aggregators built from the same frozen ruleset, but that should not be read as proof of a broader mechanism or as a standalone interpretability result without separate evidence.
- Rule generation uses seeded randomness where the inducer path requires it, so runs are controlled and reproducible rather than ``not random'' in an absolute sense.
- Exploratory scripts such as `src/evaluate_outliers.py` are kept only for legacy analysis and are not part of the canonical benchmark/report path.
- Older experimental code is not part of the maintained repository anymore; use the repository-root entry points and treat `src/` as internal implementation.
