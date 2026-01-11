#!/usr/bin/env python3
"""
Evaluate saved RIPPER/FOIL models on outlier subsets.

Pipeline:
1) Discover datasets (or use --datasets list).
2) Load dataset metadata for label mapping + categorical encoders.
3) Load <dataset>_outliers.csv and encode features to match model expectations.
4) Run Voting + DST variants and collect Acc/F1/Precision/Recall/Omega.
5) Plot consolidated graphs and print a summary table.

Notes:
- This is an *outlier-only* evaluation; results are not comparable to full-dataset metrics.
- Matplotlib uses the non-GUI backend to avoid GUI crashes.
"""
import argparse
import os
import sys
from pathlib import Path

# Ensure Matplotlib cache is writable in sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# Ensure Common_code is in path
THIS = Path(__file__).resolve()
COMMON = THIS.parent
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from Datasets_loader import load_dataset
from DSClassifierMultiQ import DSClassifierMultiQ
from utils import nll_log_loss, expected_calibration_error


def _resolve_dataset_csv(name_or_path: str) -> Path:
    """Resolve dataset name to an on-disk CSV path."""
    p = Path(name_or_path)
    if p.suffix.lower() == ".csv" and p.is_file():
        return p
    for c in [COMMON.parent / f"{name_or_path}.csv", COMMON / f"{name_or_path}.csv", Path.cwd() / f"{name_or_path}.csv"]:
        if c.is_file():
            return c
    return COMMON.parent / f"{name_or_path}.csv"


def _discover_datasets(out_dir: Path) -> list[str]:
    """List datasets that already have outlier CSVs in out_dir."""
    datasets: list[str] = []
    if not out_dir.exists():
        return []
    for p in out_dir.iterdir():
        if p.is_dir() and (p / f"{p.name}_outliers.csv").is_file():
            datasets.append(p.name)
    return sorted(datasets)


def _invert_value_decoders(value_decoders: dict) -> dict:
    """Build encoder mapping (string value -> code) from decoder mapping."""
    encoders = {}
    for col, dec in (value_decoders or {}).items():
        encoders[col] = {str(v).strip(): int(k) for k, v in dec.items()}
    return encoders


def _encode_outliers(df_out: pd.DataFrame, label_col: str, classes: list, value_decoders: dict):
    """Encode outlier dataframe with the same mappings as training."""
    df = df_out.copy()
    if label_col not in df.columns:
        for alt in ["label", "labels", "income", "class", "target"]:
            if alt in df.columns:
                label_col = alt
                break

    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in outliers CSV")

    # Map labels using the same class ordering as training.
    class_map = {str(c).strip(): i for i, c in enumerate(classes)}
    y_raw = df[label_col].astype(str).str.strip()
    y = y_raw.map(class_map)
    valid_mask = y.notna()
    if not valid_mask.all():
        df = df.loc[valid_mask].copy()
        y = y.loc[valid_mask]

    y_out = y.astype(int).to_numpy()
    X_df = df.drop(columns=[label_col], errors="ignore").copy()

    # Apply categorical decoders and keep unknowns as -1.
    encoders = _invert_value_decoders(value_decoders)
    for col, enc in encoders.items():
        if col in X_df.columns:
            mapped = X_df[col].astype(str).str.strip().map(enc)
            X_df[col] = mapped.fillna(-1)

    # Coerce remaining columns to numeric.
    for col in X_df.columns:
        if col not in encoders:
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

    return X_df, y_out, label_col


def run_evaluation():
    parser = argparse.ArgumentParser(description="Evaluate DST models on outliers.")
    parser.add_argument("--datasets", default="auto", help="Comma-separated datasets or 'auto'")
    parser.add_argument("--algos", default="RIPPER,FOIL", help="Comma-separated algos")
    parser.add_argument("--out-dir", default=str(COMMON / "results" / "outlier_plots"), help="Output dir for plots")
    parser.add_argument("--contamination", type=float, default=0.05, help="Outlier contamination rate")
    parser.add_argument("--ece-bins", type=int, default=15, help="Bins for Expected Calibration Error (ECE)")
    args = parser.parse_args()

    # Output directory.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset list.
    if args.datasets.strip().lower() in {"auto", "all", ""}:
        datasets = _discover_datasets(out_dir)
    else:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]

    # Master results dictionary: dataset -> subset (inlier/outlier) -> algo -> method -> metrics.
    all_results: dict[str, dict] = {}

    for ds_name in datasets:
        print(f"\nProcessing {ds_name}...")

        # Load dataset metadata.
        try:
            csv_path = _resolve_dataset_csv(ds_name)
            if not csv_path.exists():
                print(f"Dataset {ds_name} not found.")
                continue
            X, y, feature_names, value_decoders, stats = load_dataset(csv_path, return_stats=True)
            k = int(len(np.unique(np.asarray(y, dtype=int))))
            label_col = stats.get("label_column", "label")
            classes = stats.get("classes", [])
        except Exception as e:
            print(f"Error loading {ds_name}: {e}")
            continue

        ds_results: dict[str, dict] = {}
        
        # We evaluate on both outliers and inliers
        for subset_type in ["outliers", "inliers"]:
            csv_file = out_dir / ds_name / f"{ds_name}_{subset_type}.csv"
            if not csv_file.exists():
                print(f"  [Error] {subset_type} file not found: {csv_file}")
                continue

            try:
                df_sub = pd.read_csv(csv_file)
                X_sub_df, y_sub, label_col = _encode_outliers(df_sub, label_col, classes, value_decoders)
                if y_sub.size == 0: continue
                print(f"  Loaded {len(y_sub)} {subset_type} samples.")
            except Exception as e:
                print(f"  Error loading {subset_type}: {e}")
                continue

            subset_results: dict[str, dict] = {}

            for algo in algos:
                model_path = COMMON / "pkl_rules" / f"{algo.lower()}_{ds_name}_dst.pkl"
                if not model_path.exists():
                    continue

                print(f"  Evaluating {algo} on {subset_type}...")
                try:
                    clf = DSClassifierMultiQ(k=k, rule_algo=algo, device="cpu")
                    clf.load_model(str(model_path))
                    clf.model.eval()

                    if getattr(clf.model, "feature_names", None):
                        X_sub_aligned = X_sub_df[clf.model.feature_names].to_numpy(dtype=np.float32)
                    else:
                        X_sub_aligned = X_sub_df.to_numpy(dtype=np.float32)
                except Exception as e:
                    print(f"    Error: {e}")
                    continue

                methods = {"Vote": "vote", "Dempster": "dempster", "Yager": "yager"}
                
                classes_present = np.unique(y_sub)
                if len(classes_present) == 2 and set(classes_present) <= {0, 1}:
                    avg_mode = "binary"
                else:
                    avg_mode = "macro"

                metrics: dict[str, dict] = {}
                for m_name, rule_key in methods.items():
                    proba = clf.model.predict_with_dst(X_sub_aligned, combination_rule=rule_key)
                    y_pred = proba.argmax(axis=1)
                    acc = accuracy_score(y_sub, y_pred)
                    f1 = f1_score(y_sub, y_pred, average=avg_mode, zero_division=0)
                    rec = recall_score(y_sub, y_pred, average=avg_mode, zero_division=0)
                    prec = precision_score(y_sub, y_pred, average=avg_mode, zero_division=0)
                    nll = nll_log_loss(y_sub, proba)
                    ece = expected_calibration_error(y_sub, proba, n_bins=int(args.ece_bins))
                    
                    unc_stats = clf.model.uncertainty_stats(X_sub_aligned, combination_rule=rule_key)
                    omega = np.nan if m_name == "Vote" else float(np.nanmean(unc_stats["unc_rule"]))

                    metrics[m_name] = {
                        "accuracy": acc,
                        "f1": f1,
                        "recall": rec,
                        "precision": prec,
                        "avg_omega": omega,
                        "nll": nll,
                        "logloss": nll,
                        "ece": ece,
                    }
                subset_results[algo] = metrics
            
            ds_results[subset_type] = subset_results

        if ds_results:
            all_results[ds_name] = ds_results

    # Consolidated plots and summary.
    if all_results:
        print("\nGenerating consolidated plots (5 metrics)...")
        plot_consolidated_metrics(all_results, out_dir)
        print_summary_table(all_results, out_dir)

    print(f"\nEvaluation finished. Results saved to {out_dir}")


def plot_consolidated_metrics(results, out_dir: Path) -> None:
    """
    Generate plots for each metric, with Outlier vs Inlier comparison.
    """
    method_colors = {
        "Dempster": "#a6c8ff",
        "Yager": "#b8e0b8",
        "Vote": "#f6b0b0",
    }
    datasets = sorted(list(results.keys()))
    if not datasets:
        return

    algos = set()
    for ds in datasets:
        for sub in results[ds]:
            algos.update(results[ds][sub].keys())
    algos = sorted(list(algos))

    metrics_map = {
        "Accuracy": "accuracy",
        "F1-Score": "f1",
        "Precision": "precision",
        "Recall": "recall",
        "Uncertainty (Omega, rule-avg)": "avg_omega",
        "NLL": "nll",
        "ECE": "ece",
    }

    plt.style.use("ggplot")

    for title, metric_key in metrics_map.items():
        n_rows = len(algos)
        fig, axes = plt.subplots(n_rows, 2, figsize=(20, 5 * n_rows), sharey=True)
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for ai, algo in enumerate(algos):
            for si, subset in enumerate(["outliers", "inliers"]):
                ax = axes[ai, si]
                
                # Find active methods for this subset
                available_methods = set()
                for ds in datasets:
                    if subset in results[ds] and algo in results[ds][subset]:
                        available_methods.update(results[ds][subset][algo].keys())
                
                active_m = sorted([m for m in available_methods if not (metric_key == "avg_omega" and m == "Vote")])
                if not active_m:
                    continue

                x = np.arange(len(datasets))
                width = 0.7 / len(active_m)
                offsets = np.arange(len(active_m)) * width
                offsets -= offsets.mean()

                for i, m in enumerate(active_m):
                    vals = []
                    for ds in datasets:
                        v = results[ds].get(subset, {}).get(algo, {}).get(m, {}).get(metric_key, 0.0)
                        vals.append(v if not np.isnan(v) else 0.0)
                    
                    ax.bar(x + offsets[i], vals, width=width, label=m, color=method_colors.get(m, "#cfcfcf"))

                ax.set_title(f"{algo} - {subset.capitalize()} - {title}", fontweight="bold")
                ax.set_xticks(x)
                ax.set_xticklabels(datasets, rotation=25, ha="right")
                if si == 1:
                    ax.legend(loc="upper right", fontsize=8)

        plt.suptitle(f"Comparison of {title}: Outliers vs Inliers", fontsize=16, y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        safe_name = title.split()[0].replace("-", "")
        plt.savefig(out_dir / f"ALL_DATASETS_{safe_name}.png", dpi=150)
        plt.close(fig)


def print_summary_table(results, out_dir=None):
    rows = []
    for ds in results:
        for subset in results[ds]:
            for algo in results[ds][subset]:
                for method, metrics in results[ds][subset][algo].items():
                    rows.append({
                        "Dataset": ds,
                        "Subset": subset,
                        "Algo": algo,
                        "Method": method,
                        "Acc": metrics.get("accuracy", 0),
                        "F1": metrics.get("f1", 0),
                        "Omega": metrics.get("avg_omega", 0),
                        "NLL": metrics.get("nll", 0),
                        "ECE": metrics.get("ece", 0),
                    })

    df = pd.DataFrame(rows)
    if df.empty: return
    df = df.sort_values(["Dataset", "Subset", "Algo", "Method"])
    
    print("\n" + "=" * 100)
    print("                      CONSOLIDATED EVALUATION SUMMARY (OUTLIERS & INLIERS)")
    print("=" * 100)
    print(df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x) if isinstance(x, (float, int)) else x))
    print("=" * 100 + "\n")

    if out_dir:
        df.to_csv(out_dir / "full_evaluation_summary.csv", index=False)


if __name__ == "__main__":
    run_evaluation()
