# DSGD on RIPPER/FOIL Rules

This repository contains a frozen-rule evidential classifier for tabular data. The pipeline induces rules with FOIL or RIPPER, freezes the rule base, and learns Dempster-Shafer masses on top of the same fixed rules.

All maintained runtime code lives in `Common_code/`. Treat `Common_code/` as the supported code path when downloading or reusing the repository.

## What Is Here
- `Common_code/test_Ripper_DST.py`: training and evaluation entry point
- `Common_code/rule_generator.py`: FOIL/RIPPER rule induction and pool shaping
- `Common_code/DSModelMultiQ.py`: rule activation, evidential fusion, probability output
- `Common_code/DSClassifierMultiQ.py`: training loop for learned rule masses
- `Common_code/report_builder.py`: benchmark aggregation
- `Common_code/analyze_hard_cases.py`: hard-case evaluation
- `Common_code/sample_rule_inspector.py`: web-first sample inspection helpers, combined-rule payloads, and validation checks
- `Common_code/app.py`: Streamlit app for training, generation summaries, manual inspection, and bundle export

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn scipy torch matplotlib wittgenstein
```

Optional packages:
- `streamlit` for the local app
- `plotly` only if you extend optional plotting paths in `Common_code/utils.py`
- `dill` only if you want it as a pickle backend fallback

For the local app:

```bash
pip install streamlit
```

## Data
To reproduce the benchmark, place the dataset CSV files in the project root. The current benchmark code expects files such as:

`adult.csv`, `bank-full.csv`, `BrainTumor.csv`, `breast-cancer-wisconsin.csv`, `df_wine.csv`, `dry-bean.csv`, `gas_drift.csv`, `german.csv`, `magic-gamma.csv`.

You can also adapt the loader in `Common_code/Datasets_loader.py` for your own tabular datasets.

## Run
Train and evaluate on one dataset by direct path:

```bash
python Common_code/test_Ripper_DST.py \
  --dataset-path ./adult.csv \
  --inducers RIPPER,FOIL \
  --save-root ./artifacts/adult_run
```

You can also use an absolute path:

```bash
python Common_code/test_Ripper_DST.py \
  --dataset-path /full/path/to/your/data.csv \
  --inducers RIPPER,FOIL \
  --save-root /full/path/to/output_dir
```

This saves everything under the chosen output directory:
- `benchmarks/`: metrics CSV files
- `rules/`: readable `.dsb` rule files with masses
- `models/`: `.pkl` payloads with learned weights, rule stats, and metrics

Run the current benchmark set:

```bash
python Common_code/test_Ripper_DST.py \
  --datasets adult,bank-full,BrainTumor,breast-cancer-wisconsin,df_wine,dry-bean,gas_drift,german,magic-gamma \
  --inducers RIPPER,FOIL \
  --save-root artifacts/full_benchmark
```

Build aggregated reports:

```bash
python Common_code/report_builder.py --out-root artifacts --include-hard-cases
python Common_code/analyze_hard_cases.py --out-root artifacts
```

Inspect one sample:

```bash
python Common_code/sample_rule_inspector.py \
  --dataset adult \
  --algo RIPPER \
  --row-index 123 \
  --split test \
  --combine-rule dempster \
  --mode export-combined \
  --out-root artifacts
```

Run the local app:

```bash
streamlit run Common_code/app.py
```

The app lets a user:
- upload a CSV
- choose output location and training parameters
- train FOIL/RIPPER models
- download the saved `rules/`, `models/`, and `benchmarks/` artifacts as a ZIP
- inspect how rules were generated, how many rules fired, and how the final combined rule was formed
- enter feature values manually with categorical drop-downs and numeric inputs
- load a sample that already activates rules so the combined explanation is visible immediately
- keep trained artifacts across refreshes because runs are stored under `artifacts/app_sessions/`

The `Inspect` workflow is intentionally web-first:
- `How Rules Were Generated` explains the rule source, stage mix, pool shaping, depth mix, and top rules by support/precision
- `Activation And Combination` shows fired rules, the final combined condition, fused mass, and validation checks
- the old row-index workflow exists only as a hidden debug fallback

## Current Benchmark Snapshot
Metrics below come from `Common_code/results/ALL_DATASETS_metrics.csv`.

| Method | Acc | Macro-F1 | NLL | ECE |
|---|---:|---:|---:|---:|
| RF | 0.9034 | 0.8679 | 0.2406 | 0.0244 |
| FOIL:dsgd_dempster | 0.8848 | 0.8468 | 0.4535 | 0.0377 |
| RIPPER:dsgd_dempster | 0.8625 | 0.8219 | 0.3927 | 0.0752 |
| FOIL:weighted_vote | 0.8579 | 0.7927 | 1.0100 | 0.0740 |
| FOIL:first_hit_laplace | 0.8231 | 0.7502 | 0.5179 | 0.1018 |
| RIPPER:weighted_vote | 0.7699 | 0.6959 | 2.7821 | 0.1559 |
| RIPPER:first_hit_laplace | 0.7389 | 0.6292 | 0.6964 | 0.1696 |

In the current snapshot, `FOIL:dsgd_dempster` is the strongest rule-based configuration on average. `RF` remains the strongest overall external reference.

## Outputs
- training artifacts: `artifacts/`
- final benchmark tables and plots: `Common_code/results/`
- sample-level combined-rule exports: `artifacts/inspection/`

For custom runs with `--save-root`, the main outputs are:
- `<save-root>/benchmarks/*.csv`
- `<save-root>/rules/*.dsb`
- `<save-root>/models/*.pkl`

## Notes
- Yager fusion is kept as a diagnostic branch, not as the main training path.
- The strongest gains appear when the rule pool is diverse enough to provide competing evidence but not so noisy that fusion becomes unstable.
- Rule generation is intentionally not random: RIPPER/FOIL proposals are scored, then pool shaping can re-balance novelty, depth, class coverage, and overlap so the final rule base stays diverse without becoming unstable.
- Older experimental code is not part of the maintained repository anymore; use `Common_code/` as the only supported code path.
