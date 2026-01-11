#!/usr/bin/env python3
"""
Honest outlier generation pipeline (feature-only, unsupervised).

Methodology (per-dataset):
1) Load dataset via `load_dataset` (same encoding as training).
2) Split into train/test (same split parameters as the benchmark; stratified when possible).
3) Fit unsupervised anomaly detectors on X_train only (no labels/predictions used for scoring).
4) Score X_test and take the top `contamination` fraction as outliers.
5) Save only the test subset split into:
   - results/outlier_plots/<dataset>/<dataset>_outliers.csv
   - results/outlier_plots/<dataset>/<dataset>_inliers.csv
   - results/outlier_plots/<dataset>/meta.json

This avoids selection bias toward rule conflict (and therefore avoids "tuning" the subset
to highlight differences between Dempster/Yager).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure Common_code is in path
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from Datasets_loader import load_dataset, _read_csv_any, _drop_id_like, _pick_label_column
from core import split_train_test


def load_outlier_samples(ds_name: str, root_dir: Path = THIS_DIR):
    """Helper to load and encode outlier samples for evaluate_outliers.py."""
    ds_dir = root_dir / "results" / "outlier_plots" / ds_name
    meta_path = ds_dir / "meta.json"
    out_path = ds_dir / f"{ds_name}_outliers.csv"

    if not meta_path.exists() or not out_path.exists():
        raise FileNotFoundError(f"Missing outliers or meta for {ds_name}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    label_col = meta.get("label_column")
    df = pd.read_csv(out_path)
    y_out = df[label_col].values
    X_out = df.drop(columns=[label_col])
    return X_out, y_out, X_out.columns.tolist()


def _discover_datasets(root: Path) -> list[Path]:
    """Find CSV datasets in repo root or Common_code/outlier_datasets."""
    datasets = list(root.glob("*.csv"))
    ds_dir = THIS_DIR / "outlier_datasets"
    if ds_dir.exists():
        datasets.extend(list(ds_dir.glob("*.csv")))
    return sorted([p for p in datasets if p.is_file()])


def _load_clean_dataframe(
    csv_path: Path,
    label_col: str | None,
    *,
    drop_id_like: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Load raw CSV and mirror loader cleanup (drop NaNs, optionally drop ID-like columns)."""
    df = _read_csv_any(csv_path)
    df = df.dropna(axis=0, how="any").reset_index(drop=True)
    if drop_id_like:
        df = _drop_id_like(df)
    if label_col is None or label_col not in df.columns:
        label_col = _pick_label_column(df)
    return df, label_col


def _normalize01(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize array to [0, 1] with safe fallback for constant inputs."""
    x = np.asarray(x, dtype=np.float64)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < eps:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo + eps)


def _standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(X_train, axis=0)
    sd = np.nanstd(X_train, axis=0)
    sd[sd == 0] = 1.0
    return (X_train - mu) / sd, (X_test - mu) / sd


def _rank01(scores: np.ndarray) -> np.ndarray:
    """Rank-normalize to [0, 1] (higher => more anomalous), stable under scaling."""
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, num=scores.size, endpoint=True)
    return ranks


def _anomaly_scores_fit_train_score_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    detectors: list[str],
    weights: list[float] | None,
    knn_k: int,
    iforest_estimators: int,
    lof_k: int,
    seed: int,
) -> np.ndarray:
    """Return ensemble anomaly scores for X_test (higher => more anomalous)."""
    X_train = np.nan_to_num(np.asarray(X_train, dtype=np.float64), nan=0.0)
    X_test = np.nan_to_num(np.asarray(X_test, dtype=np.float64), nan=0.0)

    dets = [d.lower().strip() for d in detectors if d.strip()]
    if not dets:
        raise ValueError("detectors list must be non-empty")

    if weights is None:
        w = np.ones(len(dets), dtype=np.float64)
    else:
        if len(weights) != len(dets):
            raise ValueError("--weights must have the same length as --detectors")
        w = np.asarray(weights, dtype=np.float64)
        if np.any(~np.isfinite(w)) or np.any(w < 0):
            raise ValueError("--weights must be finite and non-negative")
        if float(w.sum()) <= 0:
            raise ValueError("--weights sum must be > 0")

    parts: list[np.ndarray] = []
    for det in dets:
        if det == "knn":
            from sklearn.neighbors import NearestNeighbors

            n_train = int(X_train.shape[0])
            k_eff = int(max(1, min(int(knn_k), n_train)))
            nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
            nn.fit(X_train)
            dists, _ = nn.kneighbors(X_test)
            # mean distance to train neighbors (higher => more anomalous)
            parts.append(dists.mean(axis=1))
            continue

        if det == "iforest":
            from sklearn.ensemble import IsolationForest

            iso = IsolationForest(
                n_estimators=int(iforest_estimators),
                random_state=int(seed),
            )
            iso.fit(X_train)
            # score_samples: higher = more normal; flip so higher = more anomalous
            parts.append(-iso.score_samples(X_test))
            continue

        if det == "lof":
            from sklearn.neighbors import LocalOutlierFactor

            n_train = int(X_train.shape[0])
            k_eff = int(max(2, min(int(lof_k), max(2, n_train - 1))))
            lof = LocalOutlierFactor(
                n_neighbors=k_eff,
                novelty=True,
            )
            lof.fit(X_train)
            # score_samples: higher = more normal; flip so higher = more anomalous
            parts.append(-lof.score_samples(X_test))
            continue

        raise ValueError(f"Unknown detector: {det}. Use knn, iforest, lof.")

    ranked = np.vstack([_rank01(_normalize01(p)) for p in parts])
    combined = (w[:, None] * ranked).sum(axis=0) / float(w.sum())
    return combined.astype(np.float64)


def _select_topk(scores: np.ndarray, contamination: float) -> np.ndarray:
    """Select the top-k scores (boolean mask), where k = round(contamination * n)."""
    scores = np.asarray(scores, dtype=np.float64)
    n = int(scores.size)
    k = int(max(1, min(n, round(float(contamination) * n))))
    idx = np.argsort(scores)[-k:]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def generate_outliers(args: argparse.Namespace) -> None:
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    project_root = THIS_DIR.parent
    datasets = _discover_datasets(project_root)
    if args.datasets and args.datasets.strip().lower() not in {"auto", "all"}:
        wanted = {d.strip() for d in args.datasets.split(",") if d.strip()}
        datasets = [p for p in datasets if p.stem in wanted]

    if not datasets:
        print("No datasets found!")
        return

    print(f"Found {len(datasets)} datasets.")

    for csv_path in datasets:
        ds_name = csv_path.stem
        if ds_name.endswith("_outliers") or ds_name.endswith("_inliers"):
            continue

        print(f"\nProcessing {ds_name}...")

        try:
            X, y, _, _, stats = load_dataset(csv_path, return_stats=True)
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=int)
            n_samples = int(X.shape[0])

            if n_samples < 20:
                print(f"  Skipping {ds_name} (too small: {n_samples} samples)")
                continue

            label_col = stats.get("label_column", None)
            df_raw, label_col = _load_clean_dataframe(csv_path, label_col, drop_id_like=False)
            if len(df_raw) != n_samples:
                print(f"  [Skip] Row mismatch after cleaning: df={len(df_raw)} vs X={n_samples}")
                continue

            X_tr, X_te, y_tr, y_te, idx_tr, idx_te = split_train_test(
                X,
                y,
                test_size=float(args.test_size),
                seed=int(args.seed),
                stratify=(not args.no_stratify_split),
            )
            n_train, n_test = int(len(X_tr)), int(len(X_te))
            if n_train < 5 or n_test < 5:
                print(f"  [Skip] Too few samples after split: train={n_train}, test={n_test}")
                continue

            X_tr_s, X_te_s = (X_tr, X_te)
            if not args.no_standardize:
                X_tr_s, X_te_s = _standardize_train_test(X_tr_s, X_te_s)

            if args.detectors:
                detectors = [d.strip() for d in str(args.detectors).split(",") if d.strip()]
            else:
                detectors = [str(args.detector)]
            weights = None
            if args.weights:
                weights = [float(w.strip()) for w in str(args.weights).split(",") if w.strip()]

            scores_te = _anomaly_scores_fit_train_score_test(
                X_tr_s,
                X_te_s,
                detectors=detectors,
                weights=weights,
                knn_k=int(args.knn_k),
                iforest_estimators=int(args.iforest_estimators),
                lof_k=int(args.lof_k),
                seed=int(args.seed),
            )
            out_mask_te = _select_topk(scores_te, args.contamination)

            df_test = df_raw.iloc[idx_te].reset_index(drop=True)
            df_outliers = df_test.iloc[out_mask_te]
            df_inliers = df_test.iloc[~out_mask_te]

            ds_out_dir = out_root / ds_name
            ds_out_dir.mkdir(parents=True, exist_ok=True)
            out_file = ds_out_dir / f"{ds_name}_outliers.csv"
            in_file = ds_out_dir / f"{ds_name}_inliers.csv"
            meta_file = ds_out_dir / "meta.json"

            df_outliers.to_csv(out_file, index=False)
            df_inliers.to_csv(in_file, index=False)

            meta = {
                "dataset": ds_name,
                "subset": "test_only",
                "label_column": str(label_col),
                "n_samples_total": int(n_samples),
                "n_train": int(n_train),
                "n_test": int(n_test),
                "n_outliers": int(len(df_outliers)),
                "n_inliers": int(len(df_inliers)),
                "contamination_requested": float(args.contamination),
                "contamination_actual_test": float(len(df_outliers)) / float(max(1, n_test)),
                "split": {
                    "test_size": float(args.test_size),
                    "seed": int(args.seed),
                    "stratify_requested": bool(not args.no_stratify_split),
                },
                "detectors": detectors,
                "weights": weights,
                "detector_params": {
                    "knn_k": int(args.knn_k) if "knn" in [d.lower() for d in detectors] else None,
                    "lof_k": int(args.lof_k) if "lof" in [d.lower() for d in detectors] else None,
                    "iforest_estimators": int(args.iforest_estimators) if "iforest" in [d.lower() for d in detectors] else None,
                },
                "standardize_features": bool(not args.no_standardize),
                "score_summary_on_test": {
                    "min": float(np.min(scores_te)),
                    "p25": float(np.quantile(scores_te, 0.25)),
                    "median": float(np.median(scores_te)),
                    "p75": float(np.quantile(scores_te, 0.75)),
                    "max": float(np.max(scores_te)),
                },
            }
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            print(f"  Test split: n_test={n_test}, outliers={len(df_outliers)} ({len(df_outliers)/n_test:.1%})")
            print(f"  Saved to {out_file}")

        except Exception as e:
            print(f"  Error processing {ds_name}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate feature-only outliers on the test split (unsupervised).")
    parser.add_argument("--out-dir", default=str(THIS_DIR / "results" / "outlier_plots"), help="Output directory")
    parser.add_argument("--datasets", default="auto", help="Comma-separated dataset names or 'auto'/'all'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (split + detector)")
    parser.add_argument("--test-size", type=float, default=0.16, help="Test split fraction (matches benchmark)")
    parser.add_argument("--no-stratify-split", action="store_true", help="Disable stratified train/test split")
    parser.add_argument("--contamination", type=float, default=0.1, help="Outlier fraction within the test split")

    parser.add_argument("--detector", default="knn", choices=["knn", "iforest", "lof"], help="Unsupervised detector (single)")
    parser.add_argument("--detectors", default="", help="Comma-separated ensemble, e.g. 'knn,lof,iforest'")
    parser.add_argument("--weights", default="", help="Comma-separated weights for --detectors (same length)")
    parser.add_argument("--knn-k", type=int, default=20, help="k for kNN distance (train→test)")
    parser.add_argument("--lof-k", type=int, default=20, help="k for LOF neighbors (train→test)")
    parser.add_argument("--iforest-estimators", type=int, default=200, help="IsolationForest trees")
    parser.add_argument("--no-standardize", action="store_true", help="Disable z-scoring using train stats")

    args = parser.parse_args()
    generate_outliers(args)


if __name__ == "__main__":
    main()
