#!/usr/bin/env python3
"""
Evaluate saved rule models (STATIC/RIPPER/FOIL) on:
- outliers/inliers subsets produced by `outlier_pipeline.py`, and/or
- the standard full-dataset train/test split (benchmark-style).

Pipeline:
1) Discover datasets (or use --datasets list).
2) Load dataset metadata for label mapping + categorical encoders.
3) Load <dataset>_outliers.csv and encode features to match model expectations.
4) Run Voting + DST variants and collect Acc/F1/Precision/Recall/Omega.
5) Plot consolidated graphs and print a summary table.
6) Plot rule-base stats (rule count, avg literals/rule, combined-rule literal count).

Notes:
- When --scenario=full, metrics are computed on the standard test split (no outlier/inlier subdivision).
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
from core import split_train_test
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


def _discover_root_csv_datasets() -> list[str]:
    """Discover datasets from CSVs located in the repo root (COMMON.parent)."""
    root = COMMON.parent
    out: list[str] = []
    for p in sorted(root.glob("*.csv")):
        if p.is_file():
            out.append(p.stem)
    return out


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
    # Drop label + any auxiliary columns that are not real features.
    drop_cols = {label_col, "sample_idx"}
    drop_cols.update([c for c in df.columns if str(c).startswith("__")])
    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").copy()

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


def _model_rule_stats(clf: DSClassifierMultiQ) -> dict[str, float]:
    rules = getattr(clf.model, "rules", None) or []
    n_rules = int(len(rules))
    lens = [len(r.get("specs") or ()) for r in rules]
    avg_len = float(np.mean(lens)) if lens else 0.0
    return {"n_rules": float(n_rules), "avg_rule_len": float(avg_len)}


def _combined_rule_length_stats(
    clf: DSClassifierMultiQ,
    X: np.ndarray,
    *,
    combination_rule: str = "dempster",
    n_samples: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        return {"mean": float("nan"), "median": float("nan"), "empty_rate": float("nan")}

    n = min(int(n_samples), int(X.shape[0]))
    if n <= 0:
        return {"mean": float("nan"), "median": float("nan"), "empty_rate": float("nan")}

    rng = np.random.default_rng(int(seed))
    idx = rng.choice(int(X.shape[0]), size=int(n), replace=False)
    expl = clf.model.predict_masses(X[idx], combination_rule=str(combination_rule), explain=True)
    if isinstance(expl, dict):
        expl = [expl]

    lens = []
    empty = 0
    for d in expl:
        cl = (d.get("combined_literals") or []) if isinstance(d, dict) else []
        ln = int(len(cl))
        lens.append(ln)
        if ln == 0:
            empty += 1

    if not lens:
        return {"mean": float("nan"), "median": float("nan"), "empty_rate": float("nan")}

    return {
        "mean": float(np.mean(lens)),
        "median": float(np.median(lens)),
        "empty_rate": float(empty) / float(len(lens)),
    }


def run_evaluation():
    parser = argparse.ArgumentParser(description="Evaluate DST models on outliers.")
    parser.add_argument("--datasets", default="auto", help="Comma-separated datasets or 'auto'")
    parser.add_argument("--algos", default="RIPPER,FOIL", help="Comma-separated algos")
    parser.add_argument("--out-dir", default=str(COMMON / "results" / "outlier_plots"), help="Output dir for plots")
    parser.add_argument(
        "--full-out-dir",
        default=str(COMMON / "results"),
        help="Output dir for full-split consolidated plots/metrics (ALL_DATASETS_*.png + combined CSVs).",
    )
    parser.add_argument("--contamination", type=float, default=0.05, help="Outlier contamination rate")
    parser.add_argument("--ece-bins", type=int, default=15, help="Bins for Expected Calibration Error (ECE)")
    parser.add_argument("--combined-n", type=int, default=200, help="How many samples per subset to estimate combined-rule length")
    parser.add_argument("--combined-seed", type=int, default=0, help="RNG seed for combined-rule length sampling")
    parser.add_argument("--combined-rule", default="dempster", choices=["dempster", "yager"], help="Fusion rule used to build combined-rule explanation")
    parser.add_argument(
        "--scenario",
        default="outliers_inliers",
        choices=["outliers_inliers", "full", "both"],
        help="What to evaluate: outliers/inliers from outlier_pipeline, full test split, or both.",
    )
    args = parser.parse_args()

    # Output directory.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_out_dir = Path(args.full_out_dir)
    full_out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset list.
    scenario = str(args.scenario).strip().lower()
    if args.datasets.strip().lower() in {"auto", "all", ""}:
        if scenario == "outliers_inliers":
            datasets = _discover_datasets(out_dir)
        elif scenario == "full":
            datasets = _discover_root_csv_datasets()
        else:  # both
            datasets = sorted(set(_discover_datasets(out_dir)) | set(_discover_root_csv_datasets()))
    else:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]

    # Master results dictionary: dataset -> subset (inlier/outlier) -> algo -> method -> metrics.
    all_results: dict[str, dict] = {}
    rule_base_stats: dict[str, dict[str, dict[str, float]]] = {}
    combined_stats: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    model_cache: dict[tuple[str, str, int], DSClassifierMultiQ] = {}

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
        rule_base_stats.setdefault(ds_name, {})
        combined_stats.setdefault(ds_name, {})
        
        def eval_subset(*, subset_type: str, X_sub_df: pd.DataFrame, y_sub: np.ndarray, seed_offset: int) -> dict[str, dict]:
            subset_results: dict[str, dict] = {}

            for algo in algos:
                model_path = COMMON / "pkl_rules" / f"{algo.lower()}_{ds_name}_dst.pkl"
                if not model_path.exists():
                    continue

                print(f"  Evaluating {algo} on {subset_type}...")
                try:
                    cache_key = (ds_name, str(algo).upper(), int(k))
                    clf = model_cache.get(cache_key)
                    if clf is None:
                        clf = DSClassifierMultiQ(k=k, rule_algo=algo, device="cpu")
                        clf.load_model(str(model_path))
                        clf.model.eval()
                        model_cache[cache_key] = clf

                    if getattr(clf.model, "feature_names", None):
                        X_sub_aligned = X_sub_df[clf.model.feature_names].to_numpy(dtype=np.float32)
                    else:
                        X_sub_aligned = X_sub_df.to_numpy(dtype=np.float32)
                except Exception as e:
                    print(f"    Error: {e}")
                    continue

                # Rule-base stats (once per dataset/algo).
                if algo not in rule_base_stats[ds_name]:
                    rule_base_stats[ds_name][algo] = _model_rule_stats(clf)

                # Combined-rule length stats (subset-specific; sampled).
                combined_stats[ds_name].setdefault(algo, {})
                if subset_type not in combined_stats[ds_name][algo]:
                    combined_stats[ds_name][algo][subset_type] = _combined_rule_length_stats(
                        clf,
                        X_sub_aligned,
                        combination_rule=str(args.combined_rule),
                        n_samples=int(args.combined_n),
                        seed=int(args.combined_seed) + int(seed_offset),
                    )

                methods = {"Vote": "vote", "Dempster": "dempster", "Yager": "yager"}
                labels_all = list(range(int(k)))

                metrics: dict[str, dict] = {}
                for m_name, rule_key in methods.items():
                    if str(algo).upper() == "STATIC" and m_name == "Vote":
                        continue
                    proba = clf.model.predict_with_dst(X_sub_aligned, combination_rule=rule_key)
                    y_pred = proba.argmax(axis=1)
                    acc = accuracy_score(y_sub, y_pred)
                    f1 = f1_score(y_sub, y_pred, average="macro", labels=labels_all, zero_division=0)
                    rec = recall_score(y_sub, y_pred, average="macro", labels=labels_all, zero_division=0)
                    prec = precision_score(y_sub, y_pred, average="macro", labels=labels_all, zero_division=0)
                    nll = nll_log_loss(y_sub, proba)
                    ece = expected_calibration_error(y_sub, proba, n_bins=int(args.ece_bins))

                    unc_stats = clf.model.uncertainty_stats(X_sub_aligned, combination_rule=rule_key)
                    omega = np.nan if m_name == "Vote" else float(np.nanmean(unc_stats["unc_comb"]))

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

            return subset_results

        if scenario in {"outliers_inliers", "both"}:
            for subset_type, seed_off in [("outliers", 0), ("inliers", 1)]:
                csv_file = out_dir / ds_name / f"{ds_name}_{subset_type}.csv"
                if not csv_file.exists():
                    print(f"  [Error] {subset_type} file not found: {csv_file}")
                    continue

                try:
                    df_sub = pd.read_csv(csv_file)
                    X_sub_df, y_sub, label_col = _encode_outliers(df_sub, label_col, classes, value_decoders)
                    if y_sub.size == 0:
                        continue
                    print(f"  Loaded {len(y_sub)} {subset_type} samples.")
                except Exception as e:
                    print(f"  Error loading {subset_type}: {e}")
                    continue

                ds_results[subset_type] = eval_subset(subset_type=subset_type, X_sub_df=X_sub_df, y_sub=y_sub, seed_offset=seed_off)

        if scenario in {"full", "both"}:
            try:
                X_all, y_all, feature_names, value_decoders_full, stats_full = load_dataset(csv_path, return_stats=True)
                X_all = np.asarray(X_all, dtype=np.float32)
                y_all = np.asarray(y_all, dtype=int)
                # Evaluate on the standard test split (matches benchmark), but without outlier/inlier subdivision.
                X_tr, X_te, y_tr, y_te, idx_tr, idx_te = split_train_test(
                    X_all, y_all, test_size=0.16, seed=42, stratify=True
                )
                X_te_df = pd.DataFrame(X_te, columns=list(feature_names))
                ds_results["full"] = eval_subset(subset_type="full", X_sub_df=X_te_df, y_sub=y_te, seed_offset=2)
                print(f"  Full split: n_test={len(y_te)}")
            except Exception as e:
                print(f"  Error computing full split: {e}")

        if ds_results:
            all_results[ds_name] = ds_results

    # Consolidated plots and summary.
    if all_results:
        print("\nGenerating consolidated plots (all metrics)...")
        if scenario in {"full", "both"}:
            plot_consolidated_metrics(all_results, full_out_dir, subsets=["full"], filename_suffix="")
        if scenario in {"outliers_inliers", "both"}:
            plot_consolidated_metrics(all_results, out_dir, subsets=["outliers", "inliers"], filename_suffix="_OUTLIERS_INLIERS")

        df_summary = print_summary_table(all_results)
        if df_summary is not None and not df_summary.empty:
            # Unified tables (paper-facing).
            df_summary.to_csv(full_out_dir / "ALL_DATASETS_evaluation_summary.csv", index=False)
            df_summary[df_summary["Subset"].eq("full")].to_csv(full_out_dir / "ALL_DATASETS_metrics.csv", index=False)
            df_summary[df_summary["Subset"].isin(["outliers", "inliers"])].to_csv(
                full_out_dir / "ALL_DATASETS_metrics_OUTLIERS_INLIERS.csv", index=False
            )

            # Backward compatible location.
            df_summary.to_csv(out_dir / "full_evaluation_summary.csv", index=False)

        plot_rule_base_stats(rule_base_stats, combined_stats, out_dir)

    print(f"\nEvaluation finished. Results saved to {out_dir} (outliers/inliers + rule-base plots) and {full_out_dir} (full-split plots/tables)")


def plot_consolidated_metrics(results, out_dir: Path, *, subsets: list[str], filename_suffix: str = "") -> None:
    """
    Generate plots for each metric, for the requested subset(s).
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
        "Uncertainty (Omega, fused)": "avg_omega",
        "NLL": "nll",
        "ECE": "ece",
    }

    plt.style.use("ggplot")

    for title, metric_key in metrics_map.items():
        n_rows = len(algos)
        n_cols = len(subsets)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20 * n_cols, 5 * n_rows), sharey=True)
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        if n_cols == 1:
            axes = np.expand_dims(axes, axis=1)

        for ai, algo in enumerate(algos):
            for si, subset in enumerate(subsets):
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
                if si == n_cols - 1:
                    ax.legend(loc="upper right", fontsize=8)

        if subsets == ["outliers", "inliers"]:
            sup = f"Comparison of {title}: Outliers vs Inliers"
        elif subsets == ["full"]:
            sup = f"{title} on full test split (no outlier/inlier subdivision)"
        else:
            sup = f"{title}"
        plt.suptitle(sup, fontsize=16, y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        safe_name = title.split()[0].replace("-", "")
        plt.savefig(out_dir / f"ALL_DATASETS_{safe_name}{filename_suffix}.png", dpi=150)
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
    if df.empty:
        return None
    df = df.sort_values(["Dataset", "Subset", "Algo", "Method"])
    
    print("\n" + "=" * 100)
    print("                           CONSOLIDATED EVALUATION SUMMARY")
    print("=" * 100)
    print(df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x) if isinstance(x, (float, int)) else x))
    print("=" * 100 + "\n")

    return df


def plot_rule_base_stats(rule_base_stats: dict, combined_stats: dict, out_dir: Path) -> None:
    datasets = sorted(list(rule_base_stats.keys()))
    if not datasets:
        return

    algos = set()
    for ds in datasets:
        algos.update(rule_base_stats.get(ds, {}).keys())
    algos = sorted(list(algos))
    if not algos:
        return

    rows = []
    for ds in datasets:
        for algo in algos:
            rs = rule_base_stats.get(ds, {}).get(algo)
            if not rs:
                continue
            out_s = combined_stats.get(ds, {}).get(algo, {}).get("outliers", {})
            in_s = combined_stats.get(ds, {}).get(algo, {}).get("inliers", {})
            rows.append(
                {
                    "Dataset": ds,
                    "Algo": algo,
                    "n_rules": float(rs.get("n_rules", float("nan"))),
                    "avg_rule_len": float(rs.get("avg_rule_len", float("nan"))),
                    "comb_len_out_mean": float(out_s.get("mean", float("nan"))),
                    "comb_len_in_mean": float(in_s.get("mean", float("nan"))),
                    "comb_empty_out": float(out_s.get("empty_rate", float("nan"))),
                    "comb_empty_in": float(in_s.get("empty_rate", float("nan"))),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return
    df.to_csv(out_dir / "rule_base_stats.csv", index=False)

    plt.style.use("ggplot")
    palette = {"STATIC": "#a6c8ff", "RIPPER": "#b8e0b8", "FOIL": "#f6b0b0"}

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharex=True)

    ax = axes[0]
    for algo in algos:
        dfa = df[df["Algo"] == algo]
        ax.plot(dfa["Dataset"], dfa["n_rules"], marker="o", linewidth=2, label=algo, color=palette.get(algo, None))
    ax.set_title("Rule count by dataset", fontweight="bold")
    ax.set_ylabel("# rules")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    for algo in algos:
        dfa = df[df["Algo"] == algo]
        ax.plot(dfa["Dataset"], dfa["avg_rule_len"], marker="o", linewidth=2, label=algo, color=palette.get(algo, None))
    ax.set_title("Avg literals per rule by dataset", fontweight="bold")
    ax.set_ylabel("avg # literals/rule")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "rule_counts_pastel.png", dpi=200)
    plt.close(fig)

    # Combined-rule length plot (if present).
    if df[["comb_len_out_mean", "comb_len_in_mean"]].notna().any().any():
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
        for ai, subset_key in enumerate([("outliers", "comb_len_out_mean"), ("inliers", "comb_len_in_mean")]):
            subset_name, col = subset_key
            ax = axes[ai]
            for algo in algos:
                dfa = df[df["Algo"] == algo]
                ax.plot(dfa["Dataset"], dfa[col], marker="o", linewidth=2, label=algo, color=palette.get(algo, None))
            ax.set_title(f"Combined-rule literal count ({subset_name})", fontweight="bold")
            ax.set_ylabel("avg # literals in combined rule")
            ax.tick_params(axis="x", rotation=25)
            ax.legend(loc="upper right", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / "combined_rule_literals.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    run_evaluation()
