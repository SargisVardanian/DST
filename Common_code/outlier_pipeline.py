#!/usr/bin/env python3
"""
Outlier generation pipeline (two modes).

Methodology (per-dataset):
1) Load dataset via `load_dataset` (same encoding as training).
2) Split into train/test (same split parameters as the benchmark; stratified when possible).
3) Choose the outlier scoring mode:
   - mode=feature (default): fit unsupervised anomaly detectors on X_train only (no labels/predictions used for scoring),
     score X_test and take the top `contamination` fraction as outliers.
   - mode=uncertainty: (diagnostic) load an already trained DST model (from pkl_rules) and score X_test by fused uncertainty Ω
     (combined Omega under a chosen fusion rule). Take the top `contamination` fraction as outliers.
5) Save only the test subset split into:
   - results/outlier_plots/<dataset>/<dataset>_outliers.csv
   - results/outlier_plots/<dataset>/<dataset>_inliers.csv
   - results/outlier_plots/<dataset>/meta.json

Notes:
* mode=feature is "honest" (feature-only) and avoids selection bias toward rule conflict.
* mode=uncertainty intentionally surfaces "uncertain / not smooth" cases to audit the
  behavior of already trained rules + their DST fusion (not recommended for a clean benchmark).
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
from DSClassifierMultiQ import DSClassifierMultiQ


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
    kmeans_k: int,
    gmm_components: int,
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

        # --- Clustering-based ---
        if det == "kmeans":
            from sklearn.cluster import KMeans

            n_train = int(X_train.shape[0])
            k_req = int(max(2, int(kmeans_k)))
            k_eff = int(max(2, min(k_req, n_train)))
            km = KMeans(n_clusters=k_eff, random_state=int(seed), n_init="auto")
            km.fit(X_train)
            centers = np.asarray(km.cluster_centers_, dtype=np.float64)
            # distance to nearest centroid (higher => more anomalous)
            d2 = ((X_test[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            parts.append(np.sqrt(np.min(d2, axis=1)))
            continue

        if det == "kmeans_margin":
            from sklearn.cluster import KMeans

            n_train = int(X_train.shape[0])
            k_req = int(max(2, int(kmeans_k)))
            k_eff = int(max(2, min(k_req, n_train)))
            km = KMeans(n_clusters=k_eff, random_state=int(seed), n_init="auto")
            km.fit(X_train)
            centers = np.asarray(km.cluster_centers_, dtype=np.float64)
            # ambiguity score: smaller gap between 1st and 2nd closest centroid => larger score
            d2 = ((X_test[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            d2s = np.sort(d2, axis=1)
            d1 = np.sqrt(d2s[:, 0])
            d2n = np.sqrt(d2s[:, 1])
            margin = np.maximum(0.0, d2n - d1)
            parts.append(1.0 / (margin + 1e-9))
            continue

        if det == "gmm":
            from sklearn.mixture import GaussianMixture

            n_train = int(X_train.shape[0])
            c_req = int(max(1, int(gmm_components)))
            c_eff = int(max(1, min(c_req, max(1, n_train))))
            gm = GaussianMixture(n_components=c_eff, covariance_type="full", random_state=int(seed))
            gm.fit(X_train)
            # score_samples: higher = more normal; flip so higher = more anomalous
            parts.append(-gm.score_samples(X_test))
            continue

        if det == "gmm_entropy":
            from sklearn.mixture import GaussianMixture

            n_train = int(X_train.shape[0])
            c_req = int(max(1, int(gmm_components)))
            c_eff = int(max(1, min(c_req, max(1, n_train))))
            gm = GaussianMixture(n_components=c_eff, covariance_type="full", random_state=int(seed))
            gm.fit(X_train)
            proba = np.asarray(gm.predict_proba(X_test), dtype=np.float64)
            # entropy over responsibilities (higher => more ambiguous)
            eps = 1e-12
            ent = -(proba * np.log(proba + eps)).sum(axis=1)
            parts.append(ent)
            continue

        raise ValueError(f"Unknown detector: {det}. Use knn, iforest, lof, kmeans, kmeans_margin, gmm, gmm_entropy.")

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


def _dst_uncertainty_scores_on_test(
    *,
    ds_name: str,
    X_te: np.ndarray,
    feature_names: list[str],
    k: int,
    algo: str,
    fusion_rule: str,
    model_dir: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    """Score X_test by DST fused uncertainty (combined Omega)."""
    algo_u = str(algo).upper()
    model_path = model_dir / f"{algo_u.lower()}_{ds_name}_dst.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"trained model not found: {model_path}")

    clf = DSClassifierMultiQ(k=k, rule_algo=algo_u, device="cpu")
    clf.load_model(str(model_path))
    clf.model.eval()

    # Align columns to the model's feature_names if present.
    if getattr(clf.model, "feature_names", None):
        wanted = list(clf.model.feature_names)
        f2i = {n: i for i, n in enumerate(feature_names)}
        idxs = [f2i[n] for n in wanted]
        X_aligned = np.asarray(X_te[:, idxs], dtype=np.float32)
    else:
        wanted = list(feature_names)
        X_aligned = np.asarray(X_te, dtype=np.float32)

    omega = clf.model.uncertainty_stats(X_aligned, combination_rule=str(fusion_rule))["unc_comb"]

    # Optional diagnostics to store in the CSV for later inspection.
    diag: dict[str, np.ndarray] = {"__omega": np.asarray(omega, dtype=np.float32)}
    try:
        unc_y = clf.model.uncertainty_stats(X_aligned, combination_rule="yager")["unc_comb"]
        diag["__omega_yager"] = np.asarray(unc_y, dtype=np.float32)
    except Exception:
        pass
    try:
        proba = clf.model.predict_with_dst(X_aligned, combination_rule=str(fusion_rule))
        diag["__pred"] = np.asarray(proba.argmax(axis=1), dtype=int)
        diag["__pmax"] = np.asarray(proba.max(axis=1), dtype=np.float32)
    except Exception:
        pass
    try:
        import torch

        with torch.no_grad():
            X_t = clf.model._prepare_numeric_tensor(X_aligned)  # noqa: SLF001 (script-level diagnostic)
            act = clf.model._activation_matrix(X_t)  # noqa: SLF001
            diag["__n_rules_fired"] = act.sum(dim=1).detach().cpu().numpy().astype(int, copy=False)
    except Exception:
        pass

    detector_name = [f"dst_omega_{algo_u}_{str(fusion_rule).lower()}"]
    return np.asarray(omega, dtype=np.float32), diag, detector_name


def _dst_selection_scores_on_test(
    *,
    ds_name: str,
    X_te: np.ndarray,
    feature_names: list[str],
    k: int,
    algos: list[str],
    fusion_rule: str,
    score: str,
    reduce_mode: str,
    model_dir: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    """Score X_test using one or more pre-trained DST models.

    Returns
    -------
    scores : np.ndarray
        Higher => more "outlier / uncertain" (top-k is selected).
    diag : dict[str, np.ndarray]
        Extra per-sample columns to store into the saved CSV (all prefixed with '__').
    detectors : list[str]
        Human-readable descriptor for meta.json.
    """
    score = str(score).strip().lower()
    reduce_mode = str(reduce_mode).strip().lower()
    fusion_rule = str(fusion_rule).strip().lower()

    if reduce_mode not in {"max", "mean", "min"}:
        raise ValueError("--select-reduce must be one of: max, mean, min")

    X_te = np.asarray(X_te, dtype=np.float32)
    per_algo_scores: list[np.ndarray] = []
    diag: dict[str, np.ndarray] = {}

    for algo in [str(a).upper() for a in algos]:
        model_path = model_dir / f"{algo.lower()}_{ds_name}_dst.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"trained model not found: {model_path}")

        clf = DSClassifierMultiQ(k=k, rule_algo=algo, device="cpu")
        clf.load_model(str(model_path))
        clf.model.eval()

        if getattr(clf.model, "feature_names", None):
            wanted = list(clf.model.feature_names)
            f2i = {n: i for i, n in enumerate(feature_names)}
            idxs = [f2i[n] for n in wanted]
            X_aligned = np.asarray(X_te[:, idxs], dtype=np.float32)
        else:
            X_aligned = np.asarray(X_te, dtype=np.float32)

        omega_f = clf.model.uncertainty_stats(X_aligned, combination_rule=fusion_rule)["unc_comb"]
        diag[f"__omega_{algo}_{fusion_rule}"] = np.asarray(omega_f, dtype=np.float32)

        # Optional companions for diagnostics/scoring.
        omega_y = None
        if score == "omega_gap":
            omega_y = clf.model.uncertainty_stats(X_aligned, combination_rule="yager")["unc_comb"]
            diag[f"__omega_{algo}_yager"] = np.asarray(omega_y, dtype=np.float32)

        pmax = None
        p2 = None
        if score in {"pmax", "final_conf"}:
            proba = clf.model.predict_with_dst(X_aligned, combination_rule=fusion_rule)
            proba = np.asarray(proba, dtype=np.float64)
            if proba.ndim != 2 or proba.shape[0] != X_aligned.shape[0]:
                raise ValueError("predict_with_dst returned invalid shape")
            p_sorted = np.sort(proba, axis=1)
            pmax = p_sorted[:, -1]
            p2 = p_sorted[:, -2] if proba.shape[1] >= 2 else np.zeros_like(pmax)
            diag[f"__pmax_{algo}_{fusion_rule}"] = np.asarray(pmax, dtype=np.float32)

        fired = None
        if score == "fired":
            try:
                import torch
                with torch.no_grad():
                    X_t = clf.model._prepare_numeric_tensor(X_aligned)  # noqa: SLF001 (diagnostic)
                    act = clf.model._activation_matrix(X_t)  # noqa: SLF001
                    fired = act.sum(dim=1).detach().cpu().numpy().astype(np.float64, copy=False)
                diag[f"__n_rules_fired_{algo}"] = np.asarray(fired, dtype=np.int32)
            except Exception:
                fired = None

        if score == "omega":
            s = np.asarray(omega_f, dtype=np.float64)
        elif score == "pmax":
            # lower pmax => more uncertain => higher score
            if pmax is None:
                raise RuntimeError("pmax not computed")
            s = 1.0 - np.asarray(pmax, dtype=np.float64)
        elif score == "final_conf":
            # final_conf = (1 - omega) * (pmax - p2); lower => less confident => higher score
            if pmax is None or p2 is None:
                raise RuntimeError("pmax/p2 not computed")
            sep = np.maximum(np.asarray(pmax, dtype=np.float64) - np.asarray(p2, dtype=np.float64), 0.0)
            final_conf = (1.0 - np.asarray(omega_f, dtype=np.float64)) * sep
            diag[f"__final_conf_{algo}_{fusion_rule}"] = np.asarray(final_conf, dtype=np.float32)
            s = 1.0 - final_conf
        elif score == "omega_gap":
            if omega_y is None:
                raise RuntimeError("omega_yager not computed")
            s = np.abs(np.asarray(omega_f, dtype=np.float64) - np.asarray(omega_y, dtype=np.float64))
            diag[f"__omega_gap_{algo}"] = np.asarray(s, dtype=np.float32)
        elif score == "fired":
            if fired is None:
                # If we can't compute fired count, degrade gracefully to omega (still meaningful).
                s = np.asarray(omega_f, dtype=np.float64)
                diag[f"__warn_{algo}_fired_unavailable"] = np.ones_like(s, dtype=np.int8)
            else:
                s = np.asarray(fired, dtype=np.float64)
        else:
            raise ValueError("--select-score must be one of: omega, pmax, final_conf, omega_gap, fired")

        per_algo_scores.append(np.asarray(s, dtype=np.float64))

    stacked = np.vstack(per_algo_scores) if per_algo_scores else np.zeros((0, X_te.shape[0]), dtype=np.float64)
    if stacked.size == 0:
        raise ValueError("no scores computed")

    if reduce_mode == "max":
        combined = np.max(stacked, axis=0)
    elif reduce_mode == "min":
        combined = np.min(stacked, axis=0)
    else:
        combined = np.mean(stacked, axis=0)

    detectors = [f"dst_{score}_{reduce_mode}_{fusion_rule}_" + "+".join([str(a).upper() for a in algos])]
    diag["__score"] = combined.astype(np.float32, copy=False)
    return combined.astype(np.float64, copy=False), diag, detectors


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
            X, y, feature_names, _, stats = load_dataset(csv_path, return_stats=True)
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=int)
            n_samples = int(X.shape[0])
            k = int(len(np.unique(y)))

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

            mode = str(getattr(args, "mode", "feature")).strip().lower()
            diag_cols: dict[str, np.ndarray] = {}

            if mode == "feature":
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
                    kmeans_k=int(args.kmeans_k),
                    gmm_components=int(args.gmm_components),
                    seed=int(args.seed),
                )
            elif mode == "uncertainty":
                if getattr(args, "select_algos", ""):
                    sel_algos = [a.strip() for a in str(args.select_algos).split(",") if a.strip()]
                else:
                    sel_algos = [str(args.select_algo)]

                scores_te, diag_cols, detectors = _dst_selection_scores_on_test(
                    ds_name=ds_name,
                    X_te=X_te,
                    feature_names=list(feature_names),
                    k=k,
                    algos=sel_algos,
                    fusion_rule=str(args.select_fusion),
                    score=str(getattr(args, "select_score", "omega")),
                    reduce_mode=str(getattr(args, "select_reduce", "max")),
                    model_dir=Path(args.model_dir).expanduser().resolve(),
                )
                weights = None
            else:
                raise ValueError("--mode must be 'feature' or 'uncertainty'")

            out_mask_te = _select_topk(scores_te, args.contamination)

            df_test = df_raw.iloc[idx_te].copy()
            df_test.insert(0, "sample_idx", np.asarray(idx_te, dtype=int))
            for col, vals in (diag_cols or {}).items():
                if col in df_test.columns:
                    continue
                try:
                    df_test[col] = vals
                except Exception:
                    pass
            df_test = df_test.reset_index(drop=True)
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
                "mode": mode,
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
                "uncertainty_selector": {
                    "algos": [str(a).upper() for a in ([a.strip() for a in str(getattr(args, 'select_algos', '')).split(',') if a.strip()] or [str(args.select_algo)])],
                    "fusion": str(args.select_fusion).lower(),
                    "score": str(getattr(args, "select_score", "omega")).lower(),
                    "reduce": str(getattr(args, "select_reduce", "max")).lower(),
                    "model_dir": str(Path(args.model_dir).expanduser().resolve()),
                } if mode == "uncertainty" else None,
                "detector_params": {
                    "knn_k": int(args.knn_k) if "knn" in [d.lower() for d in detectors] else None,
                    "lof_k": int(args.lof_k) if "lof" in [d.lower() for d in detectors] else None,
                    "iforest_estimators": int(args.iforest_estimators) if "iforest" in [d.lower() for d in detectors] else None,
                    "kmeans_k": int(args.kmeans_k) if any(d.lower() in {"kmeans", "kmeans_margin"} for d in detectors) else None,
                    "gmm_components": int(args.gmm_components) if any(d.lower() in {"gmm", "gmm_entropy"} for d in detectors) else None,
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
            try:
                def _dist(a: np.ndarray) -> dict[str, int]:
                    u, c = np.unique(np.asarray(a, dtype=int), return_counts=True)
                    return {str(int(ui)): int(ci) for ui, ci in zip(u.tolist(), c.tolist())}

                meta["label_distribution"] = {
                    "test": _dist(y_te),
                    "outliers": _dist(np.asarray(y_te)[out_mask_te]),
                    "inliers": _dist(np.asarray(y_te)[~out_mask_te]),
                }
            except Exception:
                pass
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            print(f"  Test split: n_test={n_test}, outliers={len(df_outliers)} ({len(df_outliers)/n_test:.1%})")
            print(f"  Saved to {out_file}")

        except Exception as e:
            print(f"  Error processing {ds_name}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate outliers on the test split (feature-only or uncertainty-based).")
    parser.add_argument("--out-dir", default=str(THIS_DIR / "results" / "outlier_plots"), help="Output directory")
    parser.add_argument("--datasets", default="auto", help="Comma-separated dataset names or 'auto'/'all'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (split + detector)")
    parser.add_argument("--test-size", type=float, default=0.16, help="Test split fraction (matches benchmark)")
    parser.add_argument("--no-stratify-split", action="store_true", help="Disable stratified train/test split")
    parser.add_argument("--contamination", type=float, default=0.1, help="Outlier fraction within the test split")

    parser.add_argument(
        "--mode",
        default="feature",
        choices=["feature", "uncertainty"],
        help="Outlier scoring mode: feature (unsupervised) or uncertainty (DST Ω on pre-trained model).",
    )
    parser.add_argument("--select-algo", default="FOIL", help="Algo to use for mode=uncertainty (FOIL/RIPPER/STATIC).")
    parser.add_argument("--select-algos", default="", help="Optional comma-separated algos for mode=uncertainty (overrides --select-algo).")
    parser.add_argument("--select-fusion", default="dempster", choices=["dempster", "yager"], help="Fusion rule to score Ω in mode=uncertainty.")
    parser.add_argument("--select-score", default="omega", choices=["omega", "pmax", "final_conf", "omega_gap", "fired"], help="Score used in mode=uncertainty.")
    parser.add_argument("--select-reduce", default="max", choices=["max", "mean", "min"], help="How to combine scores across --select-algos.")
    parser.add_argument("--model-dir", default=str(THIS_DIR / "pkl_rules"), help="Directory with trained *_dst.pkl models.")

    parser.add_argument(
        "--detector",
        default="kmeans",
        choices=["knn", "iforest", "lof", "kmeans", "kmeans_margin", "gmm", "gmm_entropy"],
        help="Unsupervised detector (single). Use kmeans_margin/gmm_entropy to select ambiguous (uncertain) points.",
    )
    parser.add_argument("--detectors", default="", help="Comma-separated ensemble, e.g. 'knn,lof,iforest'")
    parser.add_argument("--weights", default="", help="Comma-separated weights for --detectors (same length)")
    parser.add_argument("--knn-k", type=int, default=20, help="k for kNN distance (train→test)")
    parser.add_argument("--lof-k", type=int, default=20, help="k for LOF neighbors (train→test)")
    parser.add_argument("--iforest-estimators", type=int, default=200, help="IsolationForest trees")
    parser.add_argument("--kmeans-k", type=int, default=12, help="k for KMeans clusters (train→test)")
    parser.add_argument("--gmm-components", type=int, default=6, help="Components for GaussianMixture (train→test)")
    parser.add_argument("--no-standardize", action="store_true", help="Disable z-scoring using train stats")

    args = parser.parse_args()
    generate_outliers(args)


if __name__ == "__main__":
    main()
