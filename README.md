# DSGD on RIPPER/FOIL-Induced Rules

This repository keeps the runtime code and the paper-facing benchmark outputs for the current frozen-rule DSGD pipeline. The core idea is simple: induce readable rules with FOIL or RIPPER, freeze the resulting rule base, and compare raw rule aggregation against learned evidential aggregation on exactly the same rules.

## Runtime core
- `Common_code/test_Ripper_DST.py`
- `Common_code/report_builder.py`
- `Common_code/analyze_hard_cases.py`
- `Common_code/sample_rule_inspector.py`

## Standard flow
1. Run `test_Ripper_DST.py` to generate benchmark artifacts under `artifacts/`.
2. Run `analyze_hard_cases.py` to build the hard-case subset and summary.
3. Run `report_builder.py` to produce final paper-facing tables, CSV files, and plots in `Common_code/results/`.
4. Use `sample_rule_inspector.py` when a sample-level combined-rule export is needed.

## Key outputs
- `Common_code/results/ALL_DATASETS_metrics.csv`
- `Common_code/results/method_suite/REPORT.md`
- `Common_code/results/hard_cases/HARD_CASE_ANALYSIS.md`
- `short_report.tex`

## Quick commands
```bash
python Common_code/test_Ripper_DST.py --dataset adult --inducers RIPPER,FOIL --out-root artifacts
python Common_code/report_builder.py --out-root artifacts --include-hard-cases
python Common_code/analyze_hard_cases.py --out-root artifacts
python Common_code/sample_rule_inspector.py --dataset adult --algo RIPPER --row-index 123 --split test --combine-rule dempster --mode export-combined --out-root artifacts
```

## Repository policy
Local datasets, manuscript drafts, LaTeX build artifacts, and private project notes are intentionally ignored in Git. The repository is meant to track the runnable pipeline and its benchmark-facing outputs, not every local working file in the project directory.

## Notes
- Yager is a diagnostic ablation, not the main training path.
- `STATIC` remains an internal historical baseline.
- The current short report is in `short_report.tex`.
