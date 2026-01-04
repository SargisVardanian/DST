# Outlier Evaluation Report

## Pipeline Status
The project state has been consolidated into a single, robust pipeline. Obsolete files have been archived, and redundant logic has been removed from the core model.

### Key Components
1.  **`outlier_pipeline.py`**: Unified KMeans-based outlier generator. Automatically discovers datasets, creates subsets, and writes `meta.json`.
2.  **`evaluate_outliers.py`**: Refined evaluation script. Now uses robust feature alignment to handle small outlier subsets correctly.
3.  **`DSModelMultiQ.py`**: Cleaned of experimental explanation logic, focusing on core prediction and multi-rule combination.

## Benchmark Results Summary
Evaluation performed on 8 datasets with 10% contamination.

| Dataset | Algo | Best Accuracy | Best F1 | Avg Omega (Dempster) |
| :--- | :--- | :--- | :--- | :--- |
| BrainTumor | FOIL | 0.9973 | 0.9976 | 0.1386 |
| BrainTumor | RIPPER | 0.9867 | 0.9880 | 0.1312 |
| gas_drift | FOIL | 0.9978 | 0.9978 | 0.0059 |
| gas_drift | RIPPER | 0.9878 | 0.9878 | 0.1491 |
| german | FOIL | 0.9600 | 0.9355 | 0.6037 |
| german | RIPPER | 0.8700 | 0.8000 | 0.6743 |
| adult | RIPPER | 0.7775 | 0.5248 | 0.9794 |
| bank-full | RIPPER | 0.7540 | 0.2261 | 0.8816 |

## Cleanup Summary
- **Files Archived to `_archive/`**:
  - `outlier_dataset_builder.py`
  - `outlier_detector.py`
  - `outlier_notes.md`
  - `study_scores.py`
  - `run_experiments.py`
  - `rule_explainer.py`
  - `run_outlier_benchmarks.py`
- **Model Improvements**:
  - Simplified `DSModelMultiQ.py` by removing experimental `get_combined_rule` and associated explanation logic.
  - Kept essential baseline methods: `predict_by_rule_vote` and `sample_uncertainty`.
- **Plotting**: Aggregated comparison plots (Voting vs DST Methods) generated in `results/outlier_plots/`.
