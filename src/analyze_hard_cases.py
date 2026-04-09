#!/usr/bin/env python3
"""Internal hard/conflict-case analysis on the benchmark test split.

Canonical public entry points now live in:
- build_report.py
- app.py
- train_test_runner.py

This script is the paper-facing replacement for the earlier outlier-focused idea.
It keeps the split identical to the benchmark and compares:
  - raw rule predictors on frozen RIPPER/FOIL rules
  - DSGD-Auto on the same frozen rules

The goal is to quantify when DSGD-Auto helps:
  - many rules fire,
  - active rules disagree,
  - the true class is supported by few rules,
  - one correct rule competes against many wrong ones,
  - raw weighted voting fails but DSGD-Auto recovers the label.

In addition to summary tables, this script also exports a dedicated
``hard-case test base``: a reusable subset of the benchmark test split that
contains only samples satisfying at least one primary difficulty criterion.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    from .Datasets_loader import load_dataset
    from .DSClassifierMultiQ import DSClassifierMultiQ
    from .core import split_train_test
except ImportError:  # pragma: no cover - direct script/import fallback
    from Datasets_loader import load_dataset
    from DSClassifierMultiQ import DSClassifierMultiQ
    from core import split_train_test


THIS = Path(__file__).resolve()
COMMON = THIS.parent
ROOT = COMMON.parent
DEFAULT_DATASETS = (
    "adult,bank-full,BrainTumor,breast-cancer-wisconsin,df_wine,dry-bean,gas_drift,german,magic-gamma"
)


def _parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _resolve_dataset_csv(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.suffix.lower() == ".csv" and p.is_file():
        return p.resolve()
    for cand in [ROOT / f"{name_or_path}.csv", COMMON / f"{name_or_path}.csv", Path.cwd() / f"{name_or_path}.csv"]:
        if cand.is_file():
            return cand.resolve()
    return (ROOT / f"{name_or_path}.csv").resolve()


def _align_X_to_model(X: np.ndarray, feature_names: list[str], model_feature_names: list[str] | None) -> np.ndarray:
    if not model_feature_names:
        return np.asarray(X, dtype=np.float32)
    feat_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    idxs = [feat_to_idx[name] for name in model_feature_names]
    return np.asarray(X[:, idxs], dtype=np.float32)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, labels: list[int]) -> dict[str, float]:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
    }


def _top2_margin(proba: np.ndarray) -> np.ndarray:
    p = np.asarray(proba, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    sorted_p = np.sort(p, axis=1)
    top1 = sorted_p[:, -1]
    top2 = sorted_p[:, -2] if p.shape[1] >= 2 else np.zeros_like(top1)
    return np.asarray(top1 - top2, dtype=np.float64)


def _align_proba_to_classes(proba: np.ndarray, classes: np.ndarray, n_classes: int) -> np.ndarray:
    p = np.asarray(proba, dtype=float)
    cls = np.asarray(classes, dtype=int).reshape(-1)
    out = np.full((p.shape[0], int(n_classes)), 1e-12, dtype=float)
    valid = (cls >= 0) & (cls < int(n_classes))
    if valid.any():
        out[:, cls[valid]] = p[:, valid]
    out = out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out


def _rule_vote_strengths(model, *, alpha: float = 1.0) -> np.ndarray:
    k = int(model.num_classes)
    out = np.zeros(len(model.rules), dtype=np.float64)
    for rid, rule in enumerate(model.rules):
        stats = rule.get("stats") or {}
        sup = float(stats.get("support", 0.0))
        neg = float(stats.get("neg_covered", 0.0))
        q = (sup + alpha) / (sup + neg + alpha * k) if (sup + neg) >= 0.0 else 1.0 / max(1, k)
        out[rid] = float(q * np.log1p(max(0.0, sup)))
    return out


def _build_label_indicator(rule_labels: np.ndarray, k: int) -> np.ndarray:
    mat = np.zeros((len(rule_labels), int(k)), dtype=np.float64)
    valid = (rule_labels >= 0) & (rule_labels < int(k))
    if valid.any():
        mat[np.where(valid)[0], rule_labels[valid]] = 1.0
    return mat


def _max_wrong(values: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    if arr.ndim != 2:
        return np.zeros(len(y_true), dtype=np.float64)
    arr[np.arange(arr.shape[0]), np.asarray(y_true, dtype=int)] = -np.inf
    best = np.max(arr, axis=1)
    best[~np.isfinite(best)] = 0.0
    return best


def _mean_if_any(mask: np.ndarray, values: np.ndarray) -> float:
    m = np.asarray(mask, dtype=bool)
    v = np.asarray(values, dtype=np.float64)
    if m.sum() == 0:
        return float("nan")
    return float(v[m].mean())


def _summarize_subset(
    df: pd.DataFrame,
    *,
    dataset: str,
    algo: str,
    bucket: str,
    mask: np.ndarray,
    labels: list[int],
) -> dict[str, Any]:
    take = np.asarray(mask, dtype=bool)
    subset = df.loc[take]
    n = int(len(subset))
    if n == 0:
        return {}

    y_true = subset["true_label"].to_numpy(dtype=int)
    weighted = subset["pred_weighted_vote"].to_numpy(dtype=int)
    first_hit = subset["pred_first_hit_laplace"].to_numpy(dtype=int)
    dsgd = subset["pred_dsgd_dempster"].to_numpy(dtype=int)
    rf = subset["pred_rf"].to_numpy(dtype=int)

    weighted_m = _compute_metrics(y_true, weighted, labels=labels)
    first_m = _compute_metrics(y_true, first_hit, labels=labels)
    dsgd_m = _compute_metrics(y_true, dsgd, labels=labels)
    rf_m = _compute_metrics(y_true, rf, labels=labels)

    weighted_wrong = weighted != y_true
    first_wrong = first_hit != y_true
    dsgd_correct = dsgd == y_true
    dsgd_wrong = ~dsgd_correct

    return {
        "dataset": dataset,
        "algo": algo,
        "bucket": bucket,
        "n_samples": n,
        "share_of_test": float(n / max(1, len(df))),
        "mean_rules_fired": float(subset["n_rules_fired"].mean()),
        "mean_true_rule_count": float(subset["true_rule_count"].mean()),
        "mean_max_wrong_rule_count": float(subset["max_wrong_rule_count"].mean()),
        "weighted_vote_acc": weighted_m["acc"],
        "weighted_vote_macro_f1": weighted_m["macro_f1"],
        "weighted_vote_precision": weighted_m["precision"],
        "weighted_vote_recall": weighted_m["recall"],
        "first_hit_acc": first_m["acc"],
        "first_hit_macro_f1": first_m["macro_f1"],
        "dsgd_dempster_acc": dsgd_m["acc"],
        "dsgd_dempster_macro_f1": dsgd_m["macro_f1"],
        "dsgd_dempster_precision": dsgd_m["precision"],
        "dsgd_dempster_recall": dsgd_m["recall"],
        "rf_acc": rf_m["acc"],
        "rf_macro_f1": rf_m["macro_f1"],
        "delta_acc_vs_weighted": float(dsgd_m["acc"] - weighted_m["acc"]),
        "delta_macro_f1_vs_weighted": float(dsgd_m["macro_f1"] - weighted_m["macro_f1"]),
        "delta_precision_vs_weighted": float(dsgd_m["precision"] - weighted_m["precision"]),
        "delta_recall_vs_weighted": float(dsgd_m["recall"] - weighted_m["recall"]),
        "delta_rf_acc_vs_weighted": float(rf_m["acc"] - weighted_m["acc"]),
        "delta_rf_macro_f1_vs_weighted": float(rf_m["macro_f1"] - weighted_m["macro_f1"]),
        "delta_dsgd_acc_vs_rf": float(dsgd_m["acc"] - rf_m["acc"]),
        "delta_dsgd_macro_f1_vs_rf": float(dsgd_m["macro_f1"] - rf_m["macro_f1"]),
        "weighted_vote_fix_count": int(np.sum(weighted_wrong & dsgd_correct)),
        "weighted_vote_break_count": int(np.sum((~weighted_wrong) & dsgd_wrong)),
        "first_hit_fix_count": int(np.sum(first_wrong & dsgd_correct)),
        "first_hit_break_count": int(np.sum((~first_wrong) & dsgd_wrong)),
        "mean_raw_margin_weighted": float(subset["weighted_vote_margin"].mean()),
        "mean_dsgd_margin": float(subset["dsgd_margin"].mean()),
        "mean_rf_margin": float(subset["rf_margin"].mean()),
        "mean_unc_comb": float(subset["unc_comb_dempster"].mean()),
        "mean_true_raw_vote_strength": float(subset["true_raw_vote_strength"].mean()),
        "mean_max_wrong_raw_vote_strength": float(subset["max_wrong_raw_vote_strength"].mean()),
        "mean_true_dsgd_support": float(subset["true_dsgd_support_sum"].mean()),
        "mean_max_wrong_dsgd_support": float(subset["max_wrong_dsgd_support_sum"].mean()),
    }


def _select_example_rows(df: pd.DataFrame, *, limit: int) -> pd.DataFrame:
    mask = (
        df["weighted_vote_wrong"].astype(bool)
        & df["dsgd_dempster_correct"].astype(bool)
        & df["disagreement"].astype(bool)
    )
    cols = ["max_wrong_rule_count", "n_rules_fired", "weighted_vote_margin", "dsgd_margin"]
    return (
        df.loc[mask]
        .sort_values(cols, ascending=[False, False, True, False])
        .head(int(limit))
        .copy()
    )


def _describe_example_case(
    clf: DSClassifierMultiQ,
    x_row: np.ndarray,
    *,
    true_label: int,
    raw_strengths: np.ndarray,
) -> dict[str, str]:
    if clf.model is None:
        raise RuntimeError("Classifier model is not available for hard-case description.")
    export = clf.model.prepare_rules_for_export(x_row)
    active = export.get("activated_rules", [])
    combined = clf.model.get_combined_rule(x_row, return_details=True, combination_rule="dempster")

    def _pick_caption(rule_class_match: bool, score_key: str) -> str:
        best_caption = ""
        best_score = -np.inf
        for item in active:
            rid = int(item.get("id", -1))
            lbl = item.get("class")
            if lbl is None:
                continue
            if bool(int(lbl) == int(true_label)) != bool(rule_class_match):
                continue
            if score_key == "raw":
                score = float(raw_strengths[rid]) if 0 <= rid < len(raw_strengths) else 0.0
            else:
                mass = np.asarray(item.get("mass") or [], dtype=np.float64)
                score = float(1.0 - mass[-1]) if mass.size else 0.0
            if score > best_score:
                best_score = score
                best_caption = str(item.get("condition") or "")
        return best_caption

    return {
        "combined_condition": str(combined.get("combined_condition", "")),
        "top_true_rule_raw": _pick_caption(True, "raw"),
        "top_wrong_rule_raw": _pick_caption(False, "raw"),
        "top_true_rule_certainty": _pick_caption(True, "certainty"),
        "top_wrong_rule_certainty": _pick_caption(False, "certainty"),
    }


PRIMARY_HARD_FLAGS = [
    "disagreement",
    "high_depth",
    "low_weighted_margin",
    "minority_true_rule_support",
    "one_true_many_wrong",
]


def _build_hard_case_test_base(df: pd.DataFrame, *, min_hard_flags: int = 1) -> pd.DataFrame:
    out = df.copy()
    for col in PRIMARY_HARD_FLAGS:
        out[col] = out[col].astype(bool)

    out["hard_case_flag_count"] = out[PRIMARY_HARD_FLAGS].sum(axis=1).astype(int)
    out["hard_case_union"] = (out["hard_case_flag_count"] >= int(min_hard_flags)) | out["one_true_many_wrong"].astype(bool)

    def _reasons(row: pd.Series) -> str:
        reasons = [name for name in PRIMARY_HARD_FLAGS if bool(row[name])]
        return "|".join(reasons)

    out["hard_case_reasons"] = out.apply(_reasons, axis=1)
    return out.loc[out["hard_case_union"]].copy()


def _summarize_hard_case_test_base(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()

    for (dataset, algo), group in df.groupby(["dataset", "algo"], dropna=False):
        rows.append(
            {
                "dataset": dataset,
                "algo": algo,
                "n_hard_cases": int(len(group)),
                "mean_hard_flag_count": float(group["hard_case_flag_count"].mean()),
                "share_disagreement": float(group["disagreement"].mean()),
                "share_high_depth": float(group["high_depth"].mean()),
                "share_low_weighted_margin": float(group["low_weighted_margin"].mean()),
                "share_minority_true_rule_support": float(group["minority_true_rule_support"].mean()),
                "share_one_true_many_wrong": float(group["one_true_many_wrong"].mean()),
                "weighted_vote_acc": float(group["weighted_vote_correct"].mean()),
                "first_hit_acc": float(group["first_hit_correct"].mean()),
                "dsgd_dempster_acc": float(group["dsgd_dempster_correct"].mean()),
                "rf_acc": float(group["rf_correct"].mean()),
                "weighted_vote_fix_rate": float(group["weighted_vote_fixed_by_dsgd"].mean()),
                "weighted_vote_break_rate": float(group["weighted_vote_broken_by_dsgd"].mean()),
                "mean_rules_fired": float(group["n_rules_fired"].mean()),
                "mean_true_rule_count": float(group["true_rule_count"].mean()),
                "mean_max_wrong_rule_count": float(group["max_wrong_rule_count"].mean()),
                "mean_weighted_vote_margin": float(group["weighted_vote_margin"].mean()),
                "mean_dsgd_margin": float(group["dsgd_margin"].mean()),
                "mean_rf_margin": float(group["rf_margin"].mean()),
                "mean_unc_comb_dempster": float(group["unc_comb_dempster"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "algo"]).reset_index(drop=True)


def _hard_base_method_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()
    method_cols = {
        "weighted_vote": "pred_weighted_vote",
        "first_hit_laplace": "pred_first_hit_laplace",
        "dsgd_dempster": "pred_dsgd_dempster",
        "rf": "pred_rf",
    }
    for (dataset, algo), group in df.groupby(["dataset", "algo"], dropna=False):
        y_true = group["true_label"].to_numpy(dtype=int)
        labels = sorted(np.unique(y_true).tolist())
        for method_name, pred_col in method_cols.items():
            metrics = _compute_metrics(y_true, group[pred_col].to_numpy(dtype=int), labels=labels)
            rows.append({"dataset": dataset, "algo": algo, "method": method_name, **metrics})
    return pd.DataFrame(rows).sort_values(["dataset", "algo", "method"]).reset_index(drop=True)


def _plot_hard_base_size(summary_df: pd.DataFrame, *, out_dir: Path) -> None:
    if summary_df.empty:
        return
    datasets = summary_df["dataset"].drop_duplicates().tolist()
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(datasets), dtype=float)
    algos = sorted(summary_df["algo"].dropna().unique().tolist())
    width = 0.35
    colors = ["#4C78A8", "#F58518"]
    for i, algo in enumerate(algos):
        sub = summary_df[summary_df["algo"] == algo].set_index("dataset").reindex(datasets)
        vals = sub["n_hard_cases"].to_numpy(dtype=float)
        ax.bar(x + (i - (len(algos) - 1) / 2.0) * width, vals, width=width, label=algo, color=colors[i % len(colors)], alpha=0.95)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("N hard cases")
    ax.set_title("Hard-case base size by dataset", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "hard_case_size_by_dataset.png", dpi=180)
    plt.close(fig)


def _plot_method_metric_bars(method_df: pd.DataFrame, *, metric: str, out_dir: Path) -> None:
    if method_df.empty:
        return
    datasets = method_df["dataset"].drop_duplicates().tolist()
    methods = ["first_hit_laplace", "weighted_vote", "dsgd_dempster", "rf"]
    colors = {
        "first_hit_laplace": "#cfe1f2",
        "weighted_vote": "#f3e7b3",
        "dsgd_dempster": "#d73027",
        "rf": "#d0d0d0",
    }
    for algo in sorted(method_df["algo"].dropna().unique().tolist()):
        sub = method_df[(method_df["algo"] == algo) & (method_df["method"].isin(methods))].copy()
        pvt = sub.pivot(index="dataset", columns="method", values=metric).reindex(index=datasets, columns=methods)
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(datasets), dtype=float)
        width = 0.15
        for i, method in enumerate(methods):
            vals = pvt[method].to_numpy(dtype=float)
            ax.bar(
                x + (i - (len(methods) - 1) / 2.0) * width,
                vals,
                width=width,
                label=method,
                color=colors[method],
                alpha=0.98 if method == "dsgd_dempster" else 0.88,
                edgecolor="#555555",
                linewidth=0.6 if method == "dsgd_dempster" else 0.3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=20, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"Hard-case {metric} by dataset ({algo})", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout()
        fig.savefig(out_dir / f"hard_case_{metric}__{algo}.png", dpi=180)
        plt.close(fig)


def _analyze_dataset_algo(
    *,
    dataset: str,
    algo: str,
    model_path: Path,
    out_dir: Path,
    examples_per_pair: int,
    seed: int,
    test_size: float,
    depth_quantile: float,
    low_margin_thresh: float,
    minority_gap: int,
    wrong_many_min: int,
    min_depth_floor: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    csv_path = _resolve_dataset_csv(dataset)
    X, y, feature_names, _ = load_dataset(csv_path)
    X = np.asarray(X)
    y = np.asarray(y, dtype=int)
    k = int(np.unique(y).size)
    labels = list(range(k))

    X_tr, X_te, y_tr, y_te, _, idx_te = split_train_test(X, y, test_size=float(test_size), seed=int(seed), stratify=True)

    clf = DSClassifierMultiQ(k=k, rule_algo=str(algo).upper(), device="cpu")
    clf.load_model(str(model_path))
    if clf.model is None:
        raise RuntimeError(f"Loaded classifier from {model_path} does not contain a model.")
    clf.model.eval()

    X_te_aligned = _align_X_to_model(X_te, list(feature_names), list(clf.model.feature_names or []))
    with torch.no_grad():
        X_te_t = clf.model._prepare_numeric_tensor(X_te_aligned)  # noqa: SLF001
        act = clf.model._activation_matrix(X_te_t).detach().cpu().numpy().astype(bool, copy=False)  # noqa: SLF001
        masses = clf.model.get_rule_masses().detach().cpu().numpy().astype(np.float64, copy=False)

    first_proba = np.asarray(clf.raw_predict_proba(X_te_aligned, method="first_hit_laplace"), dtype=np.float64)
    weighted_proba = np.asarray(clf.raw_predict_proba(X_te_aligned, method="weighted_vote"), dtype=np.float64)
    dsgd_proba = np.asarray(clf.predict_proba(X_te_aligned, combination_rule="dempster"), dtype=np.float64)
    yager_proba = np.asarray(clf.predict_proba(X_te_aligned, combination_rule="yager"), dtype=np.float64)

    first_pred = np.asarray(clf.raw_predict(X_te_aligned, method="first_hit_laplace"), dtype=int)
    weighted_pred = np.asarray(clf.raw_predict(X_te_aligned, method="weighted_vote"), dtype=int)
    dsgd_pred = np.asarray(clf.predict(X_te_aligned, combination_rule="dempster"), dtype=int)
    yager_pred = np.asarray(clf.predict(X_te_aligned, combination_rule="yager"), dtype=int)

    rf = RandomForestClassifier(n_estimators=400, random_state=int(seed), n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_proba_raw = np.asarray(rf.predict_proba(X_te), dtype=np.float64)
    rf_proba = _align_proba_to_classes(rf_proba_raw, getattr(rf, "classes_", np.arange(k)), k)
    rf_pred = np.asarray(rf.predict(X_te), dtype=int)

    unc_stats = clf.model.uncertainty_stats(X_te_aligned, combination_rule="dempster")
    unc_comb = np.asarray(unc_stats.get("unc_comb", np.ones(len(y_te))), dtype=np.float64)
    unc_stats_yager = clf.model.uncertainty_stats(X_te_aligned, combination_rule="yager")
    unc_comb_yager = np.asarray(unc_stats_yager.get("unc_comb", np.ones(len(y_te))), dtype=np.float64)

    rule_labels = np.array(
        [int(r.get("label")) if r.get("label") is not None else -1 for r in clf.model.rules],
        dtype=int,
    )
    label_indicator = _build_label_indicator(rule_labels, k)
    act_f = act.astype(np.float64, copy=False)
    class_rule_counts = act_f @ label_indicator

    n_rules_fired = act.sum(axis=1).astype(int, copy=False)
    labeled_rule_count = class_rule_counts.sum(axis=1)
    distinct_rule_labels = (class_rule_counts > 0).sum(axis=1).astype(int, copy=False)
    true_rule_count = class_rule_counts[np.arange(len(y_te)), y_te]
    max_wrong_rule_count = _max_wrong(class_rule_counts, y_te)
    wrong_rule_count_total = labeled_rule_count - true_rule_count

    raw_strengths = _rule_vote_strengths(clf.model)
    raw_strength_matrix = label_indicator * raw_strengths[:, None]
    raw_vote_support = act_f @ raw_strength_matrix
    true_raw_vote_strength = raw_vote_support[np.arange(len(y_te)), y_te]
    max_wrong_raw_vote_strength = _max_wrong(raw_vote_support, y_te)

    dsgd_mass_support = act_f @ np.asarray(masses[:, :k], dtype=np.float64)
    true_dsgd_support = dsgd_mass_support[np.arange(len(y_te)), y_te]
    max_wrong_dsgd_support = _max_wrong(dsgd_mass_support, y_te)

    weighted_margin = _top2_margin(weighted_proba)
    dsgd_margin = _top2_margin(dsgd_proba)
    rf_margin = _top2_margin(rf_proba)
    depth_threshold = int(max(int(min_depth_floor), np.ceil(np.quantile(n_rules_fired, float(depth_quantile))))) if len(n_rules_fired) else int(min_depth_floor)

    df = pd.DataFrame(
        {
            "dataset": dataset,
            "algo": algo,
            "depth_threshold": int(depth_threshold),
            "sample_idx": np.asarray(idx_te, dtype=int),
            "true_label": np.asarray(y_te, dtype=int),
            "pred_first_hit_laplace": first_pred,
            "pred_weighted_vote": weighted_pred,
            "pred_dsgd_dempster": dsgd_pred,
            "pred_rf": rf_pred,
            "first_hit_correct": first_pred == y_te,
            "weighted_vote_correct": weighted_pred == y_te,
            "dsgd_dempster_correct": dsgd_pred == y_te,
            "rf_correct": rf_pred == y_te,
            "first_hit_wrong": first_pred != y_te,
            "weighted_vote_wrong": weighted_pred != y_te,
            "n_rules_fired": n_rules_fired,
            "n_distinct_rule_labels": distinct_rule_labels,
            "labeled_rule_count": labeled_rule_count.astype(int),
            "true_rule_count": true_rule_count.astype(int),
            "max_wrong_rule_count": max_wrong_rule_count.astype(int),
            "wrong_rule_count_total": wrong_rule_count_total.astype(int),
            "true_raw_vote_strength": true_raw_vote_strength,
            "max_wrong_raw_vote_strength": max_wrong_raw_vote_strength,
            "true_dsgd_support_sum": true_dsgd_support,
            "max_wrong_dsgd_support_sum": max_wrong_dsgd_support,
            "weighted_vote_margin": weighted_margin,
            "dsgd_margin": dsgd_margin,
            "rf_margin": rf_margin,
            "unc_comb_dempster": unc_comb,
        }
    )

    df["disagreement"] = df["n_distinct_rule_labels"] >= 2
    df["high_depth"] = df["n_rules_fired"] >= int(depth_threshold)
    df["low_weighted_margin"] = df["weighted_vote_margin"] <= float(low_margin_thresh)
    df["minority_true_rule_support"] = (df["true_rule_count"] >= 1) & ((df["max_wrong_rule_count"] - df["true_rule_count"]) >= int(minority_gap))
    df["one_true_many_wrong"] = (df["true_rule_count"] == 1) & (df["max_wrong_rule_count"] >= int(wrong_many_min))
    df["weighted_vote_fixed_by_dsgd"] = df["weighted_vote_wrong"] & df["dsgd_dempster_correct"]
    df["first_hit_fixed_by_dsgd"] = df["first_hit_wrong"] & df["dsgd_dempster_correct"]
    df["weighted_vote_broken_by_dsgd"] = df["weighted_vote_correct"] & (~df["dsgd_dempster_correct"])

    bucket_masks = {
        "all_test": np.ones(len(df), dtype=bool),
        "disagreement": df["disagreement"].to_numpy(dtype=bool),
        "high_depth": df["high_depth"].to_numpy(dtype=bool),
        "low_weighted_margin": df["low_weighted_margin"].to_numpy(dtype=bool),
        "minority_true_rule_support": df["minority_true_rule_support"].to_numpy(dtype=bool),
        "one_true_many_wrong": df["one_true_many_wrong"].to_numpy(dtype=bool),
        "weighted_vote_wrong": df["weighted_vote_wrong"].to_numpy(dtype=bool),
        "weighted_vote_wrong_and_disagreement": df["weighted_vote_wrong"].to_numpy(dtype=bool)
        & df["disagreement"].to_numpy(dtype=bool),
        "weighted_vote_fixed_by_dsgd": df["weighted_vote_fixed_by_dsgd"].to_numpy(dtype=bool),
        "first_hit_fixed_by_dsgd": df["first_hit_fixed_by_dsgd"].to_numpy(dtype=bool),
    }

    summary_rows: list[dict[str, Any]] = []
    for bucket, mask in bucket_masks.items():
        row = _summarize_subset(df, dataset=dataset, algo=algo, bucket=bucket, mask=mask, labels=labels)
        if row:
            row["depth_threshold"] = int(depth_threshold)
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    example_rows: list[dict[str, Any]] = []
    examples_df = _select_example_rows(df, limit=int(examples_per_pair))
    for _, ex in examples_df.iterrows():
        pos = int(np.where(np.asarray(idx_te, dtype=int) == int(ex["sample_idx"]))[0][0])
        details = _describe_example_case(
            clf,
            X_te_aligned[pos],
            true_label=int(ex["true_label"]),
            raw_strengths=raw_strengths,
        )
        payload = ex.to_dict()
        payload.update(details)
        example_rows.append(payload)

    return df, summary_df, example_rows


def _write_markdown_report(
    summary_mean: pd.DataFrame,
    examples_df: pd.DataFrame,
    hard_base_summary: pd.DataFrame,
    hard_method_summary: pd.DataFrame,
    *,
    depth_quantile: float,
    low_margin_thresh: float,
    minority_gap: int,
    wrong_many_min: int,
    min_depth_floor: int,
    min_hard_flags: int,
    out_path: Path,
) -> None:
    lines = [
        "# Hard-Case Analysis",
        "",
        "This report compares raw rule prediction against DSGD-Auto on the standard benchmark test split.",
        "It is a descriptive analysis of conflict-heavy cases computed from the same frozen RIPPER/FOIL ruleset used for each dataset/inducer evaluation.",
        "",
    ]
    if not hard_base_summary.empty:
        lines.extend(
            [
                "## Hard-Case Test Base",
                "",
                "Reusable subset exported from the standard benchmark test split.",
                f"A sample is included if it satisfies at least {int(min_hard_flags)} primary hard-case criterion/criteria.",
                "All criteria below are computed from fired rules of the frozen ruleset for the same dataset/algo split.",
                "`disagreement`, `high_depth`, `low_weighted_margin`, `minority_true_rule_support`, or `one_true_many_wrong`.",
                "",
                "| Dataset | Algo | N hard cases | Weighted acc | DSGD acc | RF acc | Fix rate |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in hard_base_summary.iterrows():
            lines.append(
                f"| {row['dataset']} | {row['algo']} | {int(row['n_hard_cases'])} | "
                f"{row['weighted_vote_acc']:.4f} | {row['dsgd_dempster_acc']:.4f} | "
                f"{row['rf_acc']:.4f} | {row['weighted_vote_fix_rate']:.4f} |"
            )
        lines.append("")
        lines.extend(
            [
                "## Bucket Definitions",
                "",
                "- `disagreement`: at least two distinct labels are supported by fired rules from that frozen ruleset.",
                f"- `high_depth`: `n_rules_fired >= ceil(Q{int(round(depth_quantile * 100))}(n_rules_fired))` inside the same dataset/algo test split, with floor at {int(min_depth_floor)}.",
                f"- `low_weighted_margin`: weighted-vote top-1 minus top-2 probability from that frozen ruleset is `<= {float(low_margin_thresh):.2f}`.",
                f"- `minority_true_rule_support`: the true class is supported by at least one fired rule, but the strongest competing wrong class has at least {int(minority_gap)} more fired rule(s).",
                f"- `one_true_many_wrong`: exactly one fired rule supports the true class and at least {int(wrong_many_min)} fired rules support a wrong class.",
                "",
            ]
        )
    if not summary_mean.empty:
        focus = summary_mean[summary_mean["bucket"].isin(
            [
                "disagreement",
                "high_depth",
                "minority_true_rule_support",
                "one_true_many_wrong",
                "weighted_vote_wrong_and_disagreement",
            ]
        )].copy()
        if not focus.empty:
            lines.extend(
                [
                    "## Mean Delta Vs Weighted Vote",
                    "",
                    "| Algo | Bucket | Mean dAcc | Mean dMacro-F1 | Mean fixes | Mean breaks |",
                    "|---|---|---:|---:|---:|---:|",
                ]
            )
            for _, row in focus.sort_values(["algo", "bucket"]).iterrows():
                lines.append(
                    f"| {row['algo']} | {row['bucket']} | "
                    f"{row['delta_acc_vs_weighted_mean']:.4f} | "
                    f"{row['delta_macro_f1_vs_weighted_mean']:.4f} | "
                    f"{row['weighted_vote_fix_count_mean']:.2f} | "
                    f"{row['weighted_vote_break_count_mean']:.2f} |"
            )
            lines.append("")
    if not hard_method_summary.empty:
        lines.extend(
            [
                "## Hard-Base Method Summary",
                "",
                "| Dataset | Algo | Method | Acc | Macro-F1 | Precision | Recall |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for _, row in hard_method_summary.iterrows():
            lines.append(
                f"| {row['dataset']} | {row['algo']} | {row['method']} | "
                f"{row['acc']:.4f} | {row['macro_f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} |"
            )
        lines.append("")
    if not examples_df.empty:
        lines.extend(
            [
                "## Example Fixes",
                "",
                "Top cases where weighted vote is wrong but DSGD-Auto is correct:",
                "",
            ]
        )
        for _, row in examples_df.head(12).iterrows():
            lines.extend(
                [
                    f"- `{row['dataset']}` / `{row['algo']}` / sample `{int(row['sample_idx'])}`:"
                    f" true=`{int(row['true_label'])}` weighted_vote=`{int(row['pred_weighted_vote'])}`"
                    f" dsgd=`{int(row['pred_dsgd_dempster'])}`"
                    f" fired_rules=`{int(row['n_rules_fired'])}`"
                    f" true_rule_count=`{int(row['true_rule_count'])}`"
                    f" max_wrong_rule_count=`{int(row['max_wrong_rule_count'])}`",
                    f"  combined_condition: `{row.get('combined_condition', '')}`",
                ]
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Analyze hard/conflict cases on the standard benchmark split.")
    ap.add_argument("--datasets", default=DEFAULT_DATASETS, help="Comma-separated dataset names.")
    ap.add_argument("--algos", default="RIPPER,FOIL", help="Comma-separated rule inducers.")
    ap.add_argument("--seed", type=int, default=42, help="Benchmark split seed.")
    ap.add_argument("--test-size", type=float, default=0.2, help="Benchmark test split fraction.")
    ap.add_argument("--out-root", default=str(COMMON / "results" / "raw_runs"), help="Benchmark/model artifact root used by the pipeline.")
    ap.add_argument("--model-dir", default="", help="Directory with trained *_dst.pkl payloads (default: <out-root>/models).")
    ap.add_argument("--out-dir", default="", help="Output directory (default: results/hard_cases).")
    ap.add_argument("--examples-per-pair", type=int, default=12, help="How many fix examples to keep per dataset/algo.")
    ap.add_argument("--depth-quantile", type=float, default=0.75, help="Quantile used for the high-depth threshold.")
    ap.add_argument("--low-margin-thresh", type=float, default=0.15, help="Upper bound for low weighted-vote margin.")
    ap.add_argument("--minority-gap", type=int, default=1, help="Required wrong-vs-true rule-count gap for minority_true_rule_support.")
    ap.add_argument("--wrong-many-min", type=int, default=2, help="Minimum wrong-class rule count for one_true_many_wrong.")
    ap.add_argument("--min-depth-floor", type=int, default=3, help="Minimum absolute threshold for high_depth.")
    ap.add_argument("--min-hard-flags", type=int, default=1, help="Minimum number of primary hard flags for hard-case test base inclusion.")
    args = ap.parse_args(argv)

    out_root = Path(args.out_root).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve() if str(args.model_dir).strip() else (out_root / "models")
    default_out = COMMON / "results" / "hard_cases"
    out_dir = Path(args.out_dir).expanduser().resolve() if str(args.out_dir).strip() else default_out
    _reset_dir(out_dir)

    datasets = _parse_csv_list(args.datasets)
    algos = [x.upper() for x in _parse_csv_list(args.algos)]

    all_samples: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    all_examples: list[dict[str, Any]] = []

    for dataset in datasets:
        for algo in algos:
            model_path = model_dir / f"{algo.lower()}_{dataset}_dst.pkl"
            if not model_path.exists():
                print(f"[skip] missing model: {model_path}")
                continue
            print(f"[run] {dataset} / {algo}")
            samples_df, summary_df, example_rows = _analyze_dataset_algo(
                dataset=dataset,
                algo=algo,
                model_path=model_path,
                out_dir=out_dir,
                examples_per_pair=int(args.examples_per_pair),
                seed=int(args.seed),
                test_size=float(args.test_size),
                depth_quantile=float(args.depth_quantile),
                low_margin_thresh=float(args.low_margin_thresh),
                minority_gap=int(args.minority_gap),
                wrong_many_min=int(args.wrong_many_min),
                min_depth_floor=int(args.min_depth_floor),
            )
            all_samples.append(samples_df)
            all_summaries.append(summary_df)
            all_examples.extend(example_rows)

    samples_df = pd.concat(all_samples, ignore_index=True) if all_samples else pd.DataFrame()
    summary_df = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    examples_df = pd.DataFrame(all_examples)

    if not samples_df.empty:
        samples_df.to_csv(out_dir / "hard_case_sample_level.csv", index=False)
    if not summary_df.empty:
        summary_df.to_csv(out_dir / "hard_case_bucket_detail.csv", index=False)
        summary_mean = (
            summary_df.groupby(["algo", "bucket"], as_index=False)
            .agg(
                n_samples_mean=("n_samples", "mean"),
                share_of_test_mean=("share_of_test", "mean"),
                delta_acc_vs_weighted_mean=("delta_acc_vs_weighted", "mean"),
                delta_macro_f1_vs_weighted_mean=("delta_macro_f1_vs_weighted", "mean"),
                delta_rf_acc_vs_weighted_mean=("delta_rf_acc_vs_weighted", "mean"),
                delta_rf_macro_f1_vs_weighted_mean=("delta_rf_macro_f1_vs_weighted", "mean"),
                delta_dsgd_acc_vs_rf_mean=("delta_dsgd_acc_vs_rf", "mean"),
                delta_dsgd_macro_f1_vs_rf_mean=("delta_dsgd_macro_f1_vs_rf", "mean"),
                weighted_vote_fix_count_mean=("weighted_vote_fix_count", "mean"),
                weighted_vote_break_count_mean=("weighted_vote_break_count", "mean"),
                mean_rules_fired_mean=("mean_rules_fired", "mean"),
            )
        )
        summary_mean.to_csv(out_dir / "hard_case_bucket_profile.csv", index=False)
    else:
        summary_mean = pd.DataFrame()

    hard_base_df = _build_hard_case_test_base(samples_df, min_hard_flags=int(args.min_hard_flags)) if not samples_df.empty else pd.DataFrame()
    if not hard_base_df.empty:
        hard_base_df.to_csv(out_dir / "hard_case_test_base.csv", index=False)
        hard_base_summary = _summarize_hard_case_test_base(hard_base_df)
        hard_base_summary.to_csv(out_dir / "hard_case_overview.csv", index=False)
        hard_method_summary = _hard_base_method_summary(hard_base_df)
        hard_method_summary.to_csv(out_dir / "hard_case_method_summary.csv", index=False)
        _plot_hard_base_size(hard_base_summary, out_dir=out_dir)
        _plot_method_metric_bars(hard_method_summary, metric="acc", out_dir=out_dir)
        _plot_method_metric_bars(hard_method_summary, metric="macro_f1", out_dir=out_dir)

        test_bases_dir = out_dir / "test_bases"
        test_bases_dir.mkdir(parents=True, exist_ok=True)
        for (dataset, algo), group in hard_base_df.groupby(["dataset", "algo"], dropna=False):
            name = f"{dataset}__{algo}__hard_test_base.csv"
            group.sort_values(["sample_idx"]).to_csv(test_bases_dir / name, index=False)
    else:
        hard_base_summary = pd.DataFrame()
        hard_method_summary = pd.DataFrame()

    if not examples_df.empty:
        examples_df.to_csv(out_dir / "hard_case_examples.csv", index=False)

    _write_markdown_report(
        summary_mean,
        examples_df,
        hard_base_summary,
        hard_method_summary,
        depth_quantile=float(args.depth_quantile),
        low_margin_thresh=float(args.low_margin_thresh),
        minority_gap=int(args.minority_gap),
        wrong_many_min=int(args.wrong_many_min),
        min_depth_floor=int(args.min_depth_floor),
        min_hard_flags=int(args.min_hard_flags),
        out_path=out_dir / "HARD_CASE_ANALYSIS.md",
    )
    print(f"[done] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
