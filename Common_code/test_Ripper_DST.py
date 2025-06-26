# coding: utf-8
"""
Benchmark script for rule-based DST classifier on a CSV dataset (Adult/Bank/etc).

Pipeline:
  1) Load dataset (keeps string decoders for categorical rules)
  2) Split into train/test
  3) For each rule generator (STATIC/RIPPER/FOIL):
       - Generate RAW rules on train (or load from cache)
       - Evaluate RAW rules by simple voting
       - Clone RAW rules into a DST model, train with SGD (or load from cache)
       - Evaluate DST predictions (threshold chosen on a held-out calibration split)
  4) Train baseline ML models for comparison (cached)
  5) Save metrics table and a bar plot
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, OrderedDict as ODType
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Baselines
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Local imports
from Datasets_loader    import load_dataset
from DSClassifierMultiQ import DSClassifierMultiQ
from config             import SEED

# Optional helpers (fallback for .dsb dump if your model exposes it)
try:
    from core import rules_to_dsb  # optional; used as a fallback
except Exception:
    rules_to_dsb = None  # will rely on model.save_rules_dsb if available

# ----------------------------- Paths & Config -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "german.csv"   # change if you want bank-full.csv
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TAG = DATASET_PATH.stem

np.random.seed(SEED)

# ----------------------------- Utilities -----------------------------
def metrics_from_labels(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard classification metrics (binary or macro for multi)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    binary = (len(np.unique(y_true)) == 2)
    avg = "binary" if binary else "macro"
    return {
        "Accuracy":  float(accuracy_score(y_true, y_pred)),
        "F1":        float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "Recall":    float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
    }

def best_threshold_metrics(y_true: np.ndarray, pos_scores: np.ndarray, grid: int = 201) -> Tuple[float, Dict[str, float]]:
    """
    Pick a probability threshold for class-1 by maximizing F1 over a grid.
    Works for binary tasks; for multi-class, use argmax outside this helper.
    """
    y_true = np.asarray(y_true).astype(int)
    thr_candidates = np.linspace(0.0, 1.0, grid)
    best_thr, best_f1, best_stats = 0.5, -1.0, {}
    for t in thr_candidates:
        y_hat = (pos_scores >= t).astype(int)
        stats = metrics_from_labels(y_true, y_hat)
        if stats["F1"] > best_f1:
            best_f1, best_thr, best_stats = stats["F1"], float(t), stats
    return best_thr, best_stats

def extract_pos_scores(model_or_probs, X: np.ndarray | None = None) -> np.ndarray | None:
    """
    Return class-1 probabilities:
      - if `model_or_probs` is an array of logits/probs, returns probs[:,1] (if available)
      - if it's a model wrapper, tries model.predict(..., one_hot=True) or predict_proba
    """
    # 1) If array-like provided (logits or probs)
    if isinstance(model_or_probs, np.ndarray):
        arr = np.asarray(model_or_probs, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            # assume already probabilities
            if np.all((arr >= 0.0) & (arr <= 1.0)):
                return arr[:, 1]
            # else treat as logits → softmax
            e = np.exp(arr - arr.max(axis=1, keepdims=True))
            p = e / (e.sum(axis=1, keepdims=True) + 1e-12)
            return p[:, 1]
        return None

    # 2) If model wrapper was provided
    model = model_or_probs
    if X is None:
        return None
    # try our DS wrapper
    try:
        probs = model.predict(X, one_hot=True)
        if isinstance(probs, np.ndarray) and probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1]
    except Exception:
        pass
    # sklearn-way
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if isinstance(probs, np.ndarray) and probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1]
    return None

def set_test_usability(model_wrapper, X_test: np.ndarray) -> None:
    """Attach a simple usability metric per rule: % of test samples it fires on."""
    N = len(X_test)
    if not hasattr(model_wrapper, "model") or not hasattr(model_wrapper.model, "rules"):
        return
    for r in getattr(model_wrapper.model, "rules", []):
        try:
            fired = sum(r(x) for x in X_test)
        except Exception:
            fired = 0
        r.usability = (fired / max(1, N)) * 100.0

def dump_rules_dsb(model_wrapper, filename: str) -> None:
    """Save rules in human-readable .dsb format."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    # Prefer model's native saver if available
    saver = getattr(getattr(model_wrapper, "model", None), "save_rules_dsb", None)
    if callable(saver):
        saver(filename)
        return
    # Fallback to core.rules_to_dsb if available
    if rules_to_dsb is not None and hasattr(model_wrapper.model, "rules"):
        try:
            rules_to_dsb(model_wrapper.model.rules, getattr(model_wrapper.model, "_params", {}), filename)
            return
        except Exception:
            pass
    # Last resort: write a minimal dump
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# DSB dump fallback\n")
        f.write(str(getattr(model_wrapper.model, "rules", [])))

def ensure_has_trainable_params(clf: DSClassifierMultiQ) -> None:
    """
    Safety check to avoid 'optimizer got an empty parameter list'.
    The classifier must have rules/parameters before training.
    """
    params = [p for p in clf.model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise RuntimeError("DST model has no trainable parameters. "
                           "Make sure to clone/generate RAW rules into clf_dst.model before calling fit().")

# ----------------------------- Load & Split -----------------------------
X, y, feat_names, value_decoders = load_dataset(DATASET_PATH)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
k = int(np.unique(y).size)

print(f"Detected label column: 'labels'  →  classes: {list(np.unique(y))}")
print(f"X shape = {X.shape},  y distribution = {list(map(int, np.bincount(y)))}\n")

# ----------------------------- Run each algorithm -----------------------------
algos: List[Tuple[str, dict]] = [("STATIC", {}), ("RIPPER", {}), ("FOIL", {})]

metrics_tbl: ODType[str, Dict[str, float]] = OrderedDict()
rule_counts_total: ODType[str, int] = OrderedDict()
rule_counts_per_class: ODType[str, np.ndarray] = OrderedDict()

for algo, rip_kwargs in algos:
    algo_lc = algo.lower()
    print("────────────────────────────────────────────────────────────")
    print(f"[RUN] algo = {algo:<6}  (raw → dst)")
    print("────────────────────────────────────────────────────────────")

    prefix = algo[0].lower()  # s/r/f

    # ----------------- RAW rules -----------------
    clf_raw = DSClassifierMultiQ(k=k, device="cpu", algo=algo, value_decoders=value_decoders, max_iter=1)

    # Always use these exact filenames (restored):
    raw_pkl = f"pkl_rules/{algo_lc}_{TAG}_raw.pkl"
    raw_dsb = f"dsb_rules/{algo_lc}_{TAG}_raw.dsb"

    if Path(raw_pkl).is_file():
        # Load cached RAW rules
        clf_raw.model.load_rules_bin(raw_pkl)
    else:
        # Generate from train
        clf_raw.generate_raw(X_tr, y_tr if algo != "STATIC" else None, feat_names, algo=algo, **rip_kwargs)
        Path(raw_pkl).parent.mkdir(parents=True, exist_ok=True)
        Path(raw_dsb).parent.mkdir(parents=True, exist_ok=True)
        clf_raw.model.save_rules_bin(raw_pkl)
        set_test_usability(clf_raw, X_te)
        dump_rules_dsb(clf_raw, raw_dsb)

    if algo != "STATIC":
        y_pred_raw = clf_raw.predict_rules_only(X_te)
        raw_stats = metrics_from_labels(y_te, y_pred_raw)
        metrics_tbl[f"{prefix}_raw"] = raw_stats
        rule_counts_total[f"{prefix}_raw"] = len(getattr(clf_raw.model, "rules", []))
        # simple per-class rule histogram (best-effort)
        counts = np.zeros(k, dtype=int)
        for r in getattr(clf_raw.model, "rules", []):
            try:
                lbl = int(r.caption.split()[1].rstrip(":"))
                counts[lbl] += 1
            except Exception:
                counts[0] += 1
        rule_counts_per_class[f"{prefix}_raw"] = counts
        print(f"[RAW] metrics: {raw_stats}")
    else:
        print("[RAW] metrics skipped for STATIC (rules without targets).")

    # ----------------- DST training -----------------
    clf_dst = DSClassifierMultiQ(
        k=k, device="cpu", algo=algo,
        lr=1e-2, batch_size=2048, max_iter=50, optim="adam",
        value_decoders=value_decoders,
    )

    # Clone RAW rules into DST model so it has trainable parameters
    # (this is critical to avoid 'optimizer got an empty parameter list')
    if hasattr(clf_dst.model, "clone_from"):
        clf_dst.model.clone_from(clf_raw.model)
    else:
        # If no clone method exists, at least copy rules & params if exposed
        clf_dst.model.rules = list(getattr(clf_raw.model, "rules", []))
        if hasattr(clf_raw.model, "_params"):
            clf_dst.model._params = dict(clf_raw.model._params)

    ensure_has_trainable_params(clf_dst)

    # Restored exact filenames for DST stage:
    dst_pkl = f"pkl_rules/{algo_lc}_{TAG}_dst.pkl"
    dst_dsb = f"dsb_rules/{algo_lc}_{TAG}_dst.dsb"

    if Path(dst_pkl).is_file():
        # Load trained (DST) parameters if cached
        clf_dst.model.load_rules_bin(dst_pkl)
    else:
        # Train on train split; validation is internal if your fit() was updated,
        # otherwise it just trains for max_iter epochs.
        clf_dst.fit(X_tr, y_tr)
        Path(dst_pkl).parent.mkdir(parents=True, exist_ok=True)
        Path(dst_dsb).parent.mkdir(parents=True, exist_ok=True)
        clf_dst.model.save_rules_bin(dst_pkl)
        dump_rules_dsb(clf_dst, dst_dsb)

    # --- Evaluate DST on test: F1-optimized threshold via calib split from train ---
    X_cal, X_hold, y_cal, y_hold = train_test_split(
        X_tr, y_tr, test_size=0.20, random_state=SEED, stratify=y_tr
    )
    val_scores = extract_pos_scores(clf_dst, X_cal)
    if val_scores is None:
        # fallback: just predict labels
        y_pred_dst = clf_dst.predict(X_te)
        thr = 0.5
    else:
        thr, _ = best_threshold_metrics(y_cal, val_scores, grid=201)
        te_scores = extract_pos_scores(clf_dst, X_te)
        if te_scores is None:
            y_pred_dst = clf_dst.predict(X_te)
        else:
            y_pred_dst = (te_scores >= thr).astype(int)

    dst_stats = {
        'Accuracy':  float(accuracy_score(y_te, y_pred_dst)),
        'F1':        float(f1_score(y_te, y_pred_dst, zero_division=0)),
        'Precision': float(precision_score(y_te, y_pred_dst, zero_division=0)),
        'Recall':    float(recall_score(y_te, y_pred_dst, zero_division=0)),
    }
    metrics_tbl[f"{prefix}_dst"] = dst_stats
    rule_counts_total[f"{prefix}_dst"] = len(getattr(clf_dst.model, "rules", []))
    rule_counts_per_class[f"{prefix}_dst"] = rule_counts_per_class.get(f"{prefix}_dst", np.zeros(k, dtype=int))
    print(f"[DST] Thr={thr:0.3f} metrics: {dst_stats}")

# ----------------------------- Baselines (cached) -----------------------------
print("────────────────────────────────────────────────────────────")
print("[RUN] baselines: DecisionTree / RandomForest / GradientBoosting")
print("────────────────────────────────────────────────────────────")

PKL_RULES_DIR = PROJECT_ROOT / "pkl_rules"
PKL_RULES_DIR.mkdir(parents=True, exist_ok=True)

def _predict_with_calibrated_threshold(model, X_tr, y_tr, X_te) -> Tuple[np.ndarray, float]:
    """Calibrate F1-threshold on a validation split from train; predict on test."""
    Xc, Xv, yc, yv = train_test_split(X_tr, y_tr, test_size=0.20, random_state=SEED, stratify=y_tr)
    # scores on val
    if hasattr(model, "predict_proba"):
        pv = model.predict_proba(Xv)
        vs = pv[:, 1] if pv.ndim == 2 and pv.shape[1] >= 2 else None
    else:
        vs = None
    if vs is None:
        return model.predict(X_te).astype(int), 0.5
    thr, _ = best_threshold_metrics(yv, vs, grid=201)
    # test scores
    pt = model.predict_proba(X_te)
    ts = pt[:, 1] if pt.ndim == 2 and pt.shape[1] >= 2 else None
    if ts is None:
        return model.predict(X_te).astype(int), thr
    return (ts >= thr).astype(int), thr

def _baseline_train_or_load(name, estimator_ctor, X_tr, y_tr, X_te, y_te, pkl_dir: Path, tag: str | None = None) -> Dict[str, float]:
    pkl_dir.mkdir(parents=True, exist_ok=True)
    tag = tag or ""
    prefix = f"{tag}_" if tag else ""
    pkl_path = pkl_dir / f"{prefix}{name.lower()}_model.pkl"

    model = None
    status = "LOADED"
    if pkl_path.exists():
        try:
            import joblib
            model = joblib.load(pkl_path)
            n_in = getattr(model, "n_features_in_", None)
            if (n_in is not None and n_in != X_tr.shape[1]):
                model = None  # cache mismatch
        except Exception:
            model = None

    if model is None:
        status = "TRAINED"
        model = estimator_ctor()
        model.fit(X_tr, y_tr)
        try:
            import joblib
            joblib.dump(model, pkl_path)
        except Exception:
            pass

    try:
        y_pred, thr = _predict_with_calibrated_threshold(model, X_tr, y_tr, X_te)
    except Exception:
        y_pred, thr = model.predict(X_te).astype(int), 0.5

    m = {
        'Accuracy':  float(accuracy_score(y_te, y_pred)),
        'F1':        float(f1_score(y_te, y_pred, zero_division=0)),
        'Precision': float(precision_score(y_te, y_pred, zero_division=0)),
        'Recall':    float(recall_score(y_te, y_pred, zero_division=0)),
    }
    print(f"[Baseline] {name:<16} {status:<7} Thr={thr:0.3f}  "
          f"Acc={m['Accuracy']:.4f} F1={m['F1']:.4f} P={m['Precision']:.4f} R={m['Recall']:.4f}")
    return m

metrics_tbl["dt"] = _baseline_train_or_load(
    "DecisionTree",
    lambda: DecisionTreeClassifier(random_state=SEED),
    X_tr, y_tr, X_te, y_te,
    PKL_RULES_DIR, tag=TAG,
)
metrics_tbl["rf"] = _baseline_train_or_load(
    "RandomForest",
    lambda: RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1),
    X_tr, y_tr, X_te, y_te,
    PKL_RULES_DIR, tag=TAG,
)
metrics_tbl["gb"] = _baseline_train_or_load(
    "GradientBoosting",
    lambda: GradientBoostingClassifier(random_state=SEED),
    X_tr, y_tr, X_te, y_te,
    PKL_RULES_DIR, tag=TAG,
)

# ----------------------------- Save metrics & plot -----------------------------
dfm = pd.DataFrame(metrics_tbl).T
dfm.index.name = "System"
dfm.reset_index(inplace=True)
dfm.insert(0, "Dataset", TAG)

out_csv = f"results/benchmark_dataset_{TAG}_metrics.csv"
Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
dfm.to_csv(out_csv, index=False, float_format="%.4f")

df_long = dfm.melt(id_vars=["Dataset","System"], var_name="Metric", value_name="Score")
plt.figure(figsize=(10, 6))
sns.set_context("paper"); sns.set_style("whitegrid")
ax = sns.barplot(data=df_long, x="System", y="Score", hue="Metric", palette="Set2")
ax.set_ylim(0, 1)
for p in ax.patches:
    h = p.get_height()
    ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width()/2, h), ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plot_path = f"results/benchmark_dataset_{TAG}_metrics.png"
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"\n✓  Metrics  → {out_csv}")
print(f"✓  Plot     → {plot_path}")
