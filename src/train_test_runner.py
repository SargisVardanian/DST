"""Canonical benchmark/report engine for the DST project.

Public entry points should call this module rather than the legacy process files.
It owns:
- train/test benchmark orchestration,
- hard-case analysis invocation,
- result aggregation / plots,
- brief report generation,
- full pipeline execution.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

try:
    from .benchmark_protocol import protocol_from_cli
    from .DSClassifierMultiQ import DSClassifierMultiQ, TrainConfig
    from .Datasets_loader import load_dataset  # returns X, y, feature_names, value_decoders, meta/stats
    from .core import split_train_test
except ImportError:  # pragma: no cover - direct script/import fallback
    from benchmark_protocol import protocol_from_cli
    from DSClassifierMultiQ import DSClassifierMultiQ, TrainConfig
    from Datasets_loader import load_dataset  # returns X, y, feature_names, value_decoders, meta/stats
    from core import split_train_test


SYSTEMS_ALLOWLIST = [
    "native_ordered_rule",
    "first_hit_laplace",
    "weighted_vote",
    "rf",
    "dsgd_dempster",
]

THIS_FILE = Path(__file__).resolve()
COMMON = THIS_FILE.parent
ROOT = COMMON.parent
DEFAULT_OUT_ROOT = COMMON / "results" / "raw_runs"
DEFAULT_RULES_DIR = DEFAULT_OUT_ROOT / "rules"
DEFAULT_PKL_DIR = DEFAULT_OUT_ROOT / "models"
DEFAULT_PAPER_DATASETS = (
    "adult,bank-full,BrainTumor,breast-cancer-wisconsin,df_wine,dry-bean,gas_drift,german,magic-gamma"
)


def parse_list(s: str) -> List[str]:
    if s is None:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    conf = np.asarray(conf).astype(float)
    correct = np.asarray(correct).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        m = int(mask.sum())
        if m == 0:
            continue
        acc = correct[mask].mean()
        cbar = conf[mask].mean()
        out += (m / n) * abs(acc - cbar)
    return float(out)


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, *, y_pred: np.ndarray | None = None) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).astype(float)
    proba = np.clip(proba, 1e-12, None)
    proba = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)
    if y_pred is None:
        y_pred = proba.argmax(axis=1)
    else:
        y_pred = np.asarray(y_pred).astype(int)

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # NLL for multiclass
    nll = log_loss(y_true, proba, labels=list(range(proba.shape[1])))

    conf = proba.max(axis=1)
    corr = (y_pred == y_true).astype(float)
    ece_val = ece(conf, corr, n_bins=15)

    return {
        "acc": float(acc),
        "macro_f1": float(mf1),
        "precision": float(prec),
        "recall": float(rec),
        "nll": float(nll),
        "ece": float(ece_val),
    }


def align_proba_to_classes(proba: np.ndarray, classes: np.ndarray, n_classes: int) -> np.ndarray:
    """Map classifier probabilities onto fixed class index space [0..n_classes-1]."""
    p = np.asarray(proba, dtype=float)
    cls = np.asarray(classes).astype(int).reshape(-1)
    out = np.full((p.shape[0], int(n_classes)), 1e-12, dtype=float)
    valid = (cls >= 0) & (cls < int(n_classes))
    if valid.any():
        out[:, cls[valid]] = p[:, valid]
    out = out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out


def safe_nanmean(x) -> float:
    arr = np.asarray(x, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _to_numpy_tensor(tensor) -> np.ndarray | None:
    if tensor is None:
        return None
    return tensor.detach().cpu().numpy()


def rule_progress(prefix: str):
    last = {"n": 0}

    def _cb(n: int):
        last["n"] = n
        print(f"\r[{prefix}] rules: {n}", end="", flush=True)

    return _cb, last


def _dataset_profile(X: np.ndarray, y: np.ndarray, value_decoders: Dict[str, Dict[int, str]], feature_names: List[str]) -> Dict[str, float | int | bool]:
    n_samples, n_features = X.shape
    n_classes = int(np.unique(y).size)
    counts = np.bincount(np.asarray(y, dtype=int))
    majority_ratio = float(counts.max() / max(1, counts.sum()))
    num_like = 0
    disc_like = 0
    bin_like = 0
    for j, name in enumerate(feature_names):
        uniq = np.unique(X[:, j][np.isfinite(X[:, j])])
        if uniq.size <= 2:
            bin_like += 1
            disc_like += 1
        elif name in value_decoders or (uniq.size <= 20 and np.allclose(uniq, np.round(uniq))):
            disc_like += 1
        else:
            num_like += 1
    return {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_classes": int(n_classes),
        "majority_ratio": majority_ratio,
        "num_like": int(num_like),
        "disc_like": int(disc_like),
        "bin_like": int(bin_like),
        "binary_task": bool(n_classes == 2),
        "low_d_continuous": bool(n_features <= 8 and num_like >= disc_like),
        "categorical_heavy": bool(disc_like >= num_like + 4),
        "high_dim_continuous": bool(num_like >= 32),
        "imbalanced": bool(majority_ratio >= 0.80),
        "small_dataset": bool(n_samples < 1500),
        "threshold_tuning_candidate": bool(n_classes == 2 and n_samples >= 5000),
    }


def _derive_train_profile(args, profile: Dict[str, float | int | bool], seed: int) -> Tuple[TrainConfig, float]:
    val_split = float(args.val_split)
    early_stop = int(args.early_stop)
    class_weight_power = float(args.class_weight_power)
    batch_size = int(args.batch_size)
    rule_uncertainty = 0.65
    class_weight_beta = 0.999

    if profile["small_dataset"]:
        val_split = max(val_split, 0.25)
        early_stop = max(early_stop, 25)
        batch_size = min(batch_size, 256)
        rule_uncertainty = 0.68
    if profile["low_d_continuous"]:
        # Low-dimensional continuous problems tend to overfit the validation checkpoint
        # when Ω starts too high; a slightly larger val split and lower Ω stabilize selection.
        val_split = max(val_split, 0.25)
        rule_uncertainty = min(rule_uncertainty, 0.55)
    if profile["imbalanced"]:
        class_weight_power = max(class_weight_power, 0.40)
        rule_uncertainty = max(rule_uncertainty, 0.68)
    if profile["high_dim_continuous"]:
        rule_uncertainty = min(rule_uncertainty, 0.58)
    if profile["categorical_heavy"]:
        rule_uncertainty = min(rule_uncertainty, 0.55)
    if profile["binary_task"] and profile["categorical_heavy"] and profile["small_dataset"] and not profile["imbalanced"]:
        # On small categorical binary tasks, softer class reweighting generalized better than
        # the default 0.35 in local sweeps (notably german).
        class_weight_power = min(class_weight_power, 0.20)

    if profile["n_samples"] < 5000:
        class_weight_beta = 0.99
    elif profile["n_samples"] < 20000:
        class_weight_beta = 0.995

    cfg = TrainConfig(
        max_epochs=args.max_epochs,
        batch_size=batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_split=val_split,
        seed=seed,
        early_stop_patience=early_stop,
        class_weight_power=class_weight_power,
        class_weight_beta=class_weight_beta,
        verbose=True,
        enable_binary_threshold_tuning=False,
        optimizer_name="adamw",
        record_history=True,
    )
    return cfg, float(rule_uncertainty)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _require_model(clf: DSClassifierMultiQ):
    if clf.model is None:
        raise RuntimeError("Classifier model is not available. Fit or load a model before exporting artifacts.")
    return clf.model


def rulebase_stats(rules: List[dict]) -> Dict[str, float]:
    n = len(rules)
    if n == 0:
        return {"n_rules": 0, "avg_literals": 0.0, "labeled_ratio": 0.0}
    lens = np.asarray([len(r.get("specs", ()) or ()) for r in rules], dtype=float)
    labeled = sum(1 for r in rules if r.get("label", None) is not None)
    return {
        "n_rules": int(n),
        "avg_literals": float(lens.mean()) if lens.size else 0.0,
        "labeled_ratio": float(labeled / max(1, n)),
    }


def export_readable_rules(clf: DSClassifierMultiQ, rules_path: Path) -> Path:
    model = _require_model(clf)
    model.save_rules_dsb(str(rules_path), decimals=4)
    return rules_path


def summarize_rules_for_pkl(rules: List[dict]) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for r in rules:
        st = r.get("stats") or {}
        out.append(
            {
                "label": r.get("label", None),
                "support": float(st.get("support", np.nan)),
                "precision": float(st.get("precision", np.nan)),
                "neg_covered": float(st.get("neg_covered", np.nan)),
                "literals": int(st.get("literals", len(r.get("specs", ()) or ()))),
            }
        )
    return out


def save_weights_pkl(
    *,
    pkl_path: Path,
    dataset: str,
    inducer: str,
    seed: int,
    split_seed: int,
    clf: DSClassifierMultiQ,
    metrics_dem: Dict[str, float],
    metrics_yag: Dict[str, float] | None,
    metrics_weighted_vote: Dict[str, float] | None,
    metrics_first_hit_laplace: Dict[str, float] | None,
    metrics_rf: Dict[str, float] | None,
) -> None:
    model = _require_model(clf)
    learned = _to_numpy_tensor(model.get_rule_masses())
    init_m = None
    if getattr(model, "initial_rule_masses", None) is not None:
        init_m = _to_numpy_tensor(model.initial_rule_masses)
    prior = None
    if getattr(model, "class_prior", None) is not None:
        prior = _to_numpy_tensor(model.class_prior)

    if pkl_path.exists():
        with pkl_path.open("rb") as f:
            payload = pickle.load(f)
    else:
        payload = {}

    # Keep binary-rule schema compatible with load_rules_bin()
    payload.update(
        {
            "k": int(model.num_classes),
            "algo": str(model.algo),
            "feature_names": list(model.feature_names or []),
            "value_decoders": model.value_names,
            "rules": model.rules,
            "combination_rule": str(model.combination_rule),
            "class_prior": prior,
            "rule_mass_params": learned,
            "initial_rule_masses": init_m,
        }
    )

    payload.update(
        {
        "dataset": dataset,
        "inducer": inducer,
        "seed": int(seed),
        "split_seed": int(split_seed),
        "num_classes": int(model.num_classes),
        "n_rules": int(len(model.rules)),
        "device": str(model.device),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rules_summary": summarize_rules_for_pkl(model.rules),
        "masses_learned": learned,
        "masses_initial": init_m,
        "class_prior": prior,
        "train_config": vars(clf.train_cfg).copy(),
        "fit_meta": dict(getattr(clf, "_last_fit_meta", {})),
        "metrics_dsgd_dempster": metrics_dem,
        "metrics_dsgd_yager": metrics_yag,
        "metrics_weighted_vote": metrics_weighted_vote,
        "metrics_first_hit_laplace": metrics_first_hit_laplace,
        "metrics_rf": metrics_rf,
        }
    )
    with pkl_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _resolve_csv(name: str) -> Path:
    p = Path(str(name))
    if p.suffix.lower() == ".csv" and p.is_file():
        return p
    for base in (ROOT, COMMON, Path.cwd()):
        cand = base / f"{name}.csv"
        if cand.is_file():
            return cand
    return Path(name)


def run_single_dataset_inducer(
    *,
    ds_name: str,
    inducer_u: str,
    seed: int,
    split_seed: int,
    X_tr: np.ndarray,
    X_te: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    k: int,
    feature_names: List[str],
    value_decoders: Dict[str, Dict[int, str]],
    ds_profile: Dict[str, float | int | bool],
    args,
    rules_dir: Path,
    pkl_dir: Path,
    raw_dir: Path,
    rf_metrics: Dict[str, float] | None,
    rf_fit_wall_sec: float | None,
    emit_rf_row: bool,
) -> None:
    if ds_name == "gas_drift" and inducer_u == "STATIC" and not args.allow_static_gas_drift:
        print(f"[{ds_name} | {inducer_u}] skipped by default (static rule generation is too slow here)")
        return

    run_tag = f"{ds_name}__{inducer_u}__split{split_seed}__seed{seed}"
    rules_path = rules_dir / f"{run_tag}.dsb"
    pkl_name = f"{inducer_u.lower()}_{ds_name}_split{split_seed}_seed{seed}_dst.pkl"
    pkl_path = pkl_dir / pkl_name
    cache_candidates: list[str] = [str(pkl_path)]
    split_seed_pkl = pkl_dir / f"{inducer_u.lower()}_{ds_name}_split{split_seed}_seed{split_seed}_dst.pkl"
    if split_seed_pkl != pkl_path:
        cache_candidates.append(str(split_seed_pkl))
    legacy_unsuffixed = pkl_dir / f"{inducer_u.lower()}_{ds_name}_dst.pkl"
    cache_candidates.append(str(legacy_unsuffixed))
    if str(getattr(args, "cache_models_dir", "")).strip():
        cache_dir = Path(str(args.cache_models_dir)).expanduser().resolve()
        cache_candidates.extend(
            [
                str(cache_dir / pkl_name),
                str(cache_dir / split_seed_pkl.name),
                str(cache_dir / legacy_unsuffixed.name),
            ]
        )
    deduped_cache_candidates: list[str] = []
    seen_cache = set()
    for candidate in cache_candidates:
        if candidate not in seen_cache:
            deduped_cache_candidates.append(candidate)
            seen_cache.add(candidate)

    cfg, rule_uncertainty = _derive_train_profile(args, ds_profile, seed)
    clf = DSClassifierMultiQ(
        k,
        device=args.device,
        train_cfg=cfg,
        rule_uncertainty=rule_uncertainty,
        combination_rule="dempster",
    )
    rule_gen_params: Dict[str, float | int | bool] = {
        "verbose": True,
        "enable_pool_shaping": bool(args.enable_pool_shaping),
    }
    cb, last = rule_progress(f"{ds_name}/{inducer_u}")
    print(f"\n[{ds_name} | {inducer_u} | seed={seed}] fit pipeline")
    fit_started = time.perf_counter()
    fit_meta = clf.fit(
        X_tr,
        y_tr,
        algo=inducer_u,
        feature_names=feature_names,
        value_decoders=value_decoders,
        rule_gen_params=rule_gen_params or None,
        on_rule=cb,
        rules_path=deduped_cache_candidates,
        use_cached_rules=bool(args.use_cached_rules),
        save_rules_path=str(pkl_path),
        verify_raw_on=None if args.no_raw else X_te,
    )
    fit_wall_sec = float(time.perf_counter() - fit_started)
    if fit_meta.get("rule_source") == "cache":
        print(f"[{ds_name} | {inducer_u}] rules source -> {fit_meta.get('loaded_from')}")
    else:
        print("")
        print(f"[{ds_name} | {inducer_u}] rules generated -> {fit_meta.get('saved_to')}")

    readable_rules = export_readable_rules(clf, rules_path)
    model = _require_model(clf)
    rs = rulebase_stats(model.rules)
    print(
        f"[{ds_name} | {inducer_u}] device={model.device} "
        f"rules={rs['n_rules']} avg_literals={rs['avg_literals']:.2f} "
        f"labeled_ratio={rs['labeled_ratio']:.2f}"
    )
    print(f"[{ds_name} | {inducer_u}] readable rules -> {readable_rules}")

    rows = []
    raw_cache = {}
    raw_metrics: Dict[str, Dict[str, float]] = {}

    if not args.no_raw:
        for method in ["native_ordered_rule", "first_hit_laplace", "weighted_vote"]:
            p = clf.raw_predict_proba(X_te, method=method)
            raw_cache[method] = np.asarray(p, dtype=float)
            m = compute_metrics(y_te, p)
            raw_metrics[method] = m
            rows.append(
                {
                    "dataset": ds_name,
                    "inducer": inducer_u,
                    "seed": seed,
                    "split_seed": split_seed,
                    "system": method,
                    "pool_shaping": bool(args.enable_pool_shaping),
                    "train_wall_sec": fit_wall_sec,
                    "rule_source": fit_meta.get("rule_source"),
                    "n_rules": rs["n_rules"],
                    "avg_literals": rs["avg_literals"],
                    **m,
                }
            )

    if rf_metrics is not None and emit_rf_row:
        rows.append(
            {
                "dataset": ds_name,
                "inducer": "RF",
                "seed": seed,
                "split_seed": split_seed,
                "system": "rf",
                "pool_shaping": bool(args.enable_pool_shaping),
                "train_wall_sec": float(rf_fit_wall_sec) if rf_fit_wall_sec is not None else np.nan,
                "rule_source": "rf",
                "n_rules": np.nan,
                "avg_literals": np.nan,
                **rf_metrics,
            }
        )

    dem_rule = str(model.combination_rule)
    p_dem = clf.predict_proba(X_te, combination_rule=dem_rule)
    y_dem = clf.predict(X_te, combination_rule=dem_rule)
    m_dem = compute_metrics(y_te, p_dem, y_pred=y_dem)
    u_dem = clf.model.uncertainty_stats(X_te, combination_rule=dem_rule)
    unc_mean_dem = np.asarray(u_dem.get("unc_mean", u_dem.get("unc_rule", np.array([np.nan]))), dtype=float)
    unc_comb_dem = np.asarray(u_dem.get("unc_comb", np.array([np.nan])), dtype=float)
    fusion_dem = np.asarray(u_dem.get("fusion_depth", np.array([np.nan])), dtype=float)
    rows.append(
        {
            "dataset": ds_name,
            "inducer": inducer_u,
            "seed": seed,
            "split_seed": split_seed,
            "system": "dsgd_dempster",
            "pool_shaping": bool(args.enable_pool_shaping),
            "train_wall_sec": fit_wall_sec,
            "rule_source": fit_meta.get("rule_source"),
            "n_rules": rs["n_rules"],
            "avg_literals": rs["avg_literals"],
            **m_dem,
            "unc_mean": safe_nanmean(unc_mean_dem),
            "unc_comb": safe_nanmean(unc_comb_dem),
            "fusion_depth": safe_nanmean(fusion_dem),
        }
    )

    rows = [r for r in rows if r.get("system") in SYSTEMS_ALLOWLIST]

    import pandas as pd

    df = pd.DataFrame(rows)
    df = df.sort_values(["dataset", "inducer", "system"])
    out_csv = raw_dir / f"bench__{ds_name}__{inducer_u}__split{split_seed}__seed{seed}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[{ds_name} | {inducer_u}] benchmark -> {out_csv}")

    save_weights_pkl(
        pkl_path=pkl_path,
        dataset=ds_name,
        inducer=inducer_u,
        seed=seed,
        split_seed=split_seed,
        clf=clf,
        metrics_dem=m_dem,
        metrics_yag=None,
        metrics_weighted_vote=raw_metrics.get("weighted_vote"),
        metrics_first_hit_laplace=raw_metrics.get("first_hit_laplace"),
        metrics_rf=rf_metrics,
    )
    print(f"[{ds_name} | {inducer_u}] weights -> {pkl_path}")


def benchmark_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", type=str, default="", help="Single dataset name or direct CSV path (overrides --datasets when provided).")
    ap.add_argument("--dataset-path", type=str, default="", help="Explicit absolute or relative path to a CSV file.")
    ap.add_argument("--datasets", type=str, default=DEFAULT_PAPER_DATASETS)
    ap.add_argument("--inducers", type=str, default="RIPPER,FOIL")
    ap.add_argument("--include-static", action="store_true")

    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--device", type=str, default="auto")

    ap.add_argument(
        "--paper-mode",
        dest="paper_mode",
        action="store_true",
        default=False,
        help="Use the frozen paper protocol (seed=42, test_size=0.2). Disabled by default.",
    )
    ap.add_argument(
        "--no-paper-mode",
        dest="paper_mode",
        action="store_false",
        help="Run the standard protocol with the provided seeds and test-size.",
    )

    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=2e-4)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--early-stop", type=int, default=20)
    ap.add_argument("--class-weight-power", type=float, default=0.35, help="Class-balance power for DSGD weighted NLL (0 disables balancing).")

    ap.add_argument("--use-cached-rules", action="store_true")
    ap.add_argument("--allow-static-gas-drift", action="store_true", help="Force STATIC on gas_drift even though it is slow.")

    ap.add_argument("--no-raw", action="store_true", help="Skip raw rule baselines.")
    ap.add_argument("--no-rf", action="store_true", help="Skip RandomForest baseline.")
    ap.add_argument("--rf-n-jobs", type=int, default=-1, help="RandomForest n_jobs. Use 1 for parallel multi-process runs.")

    ap.add_argument("--out-root", type=str, default=str(DEFAULT_OUT_ROOT), help="Root directory for saved run outputs.")
    ap.add_argument("--save-root", type=str, default="", help="Alias for --out-root. Saves rules, models, and benchmarks under one chosen directory.")
    ap.add_argument("--rules-dir", type=str, default="", help="Optional rules output directory (default: <out-root>/rules)")
    ap.add_argument("--models-dir", type=str, default="", help="Optional model output directory (default: <out-root>/models)")
    ap.add_argument("--enable-pool-shaping", action="store_true", help="Enable adaptive depth/class/novelty-aware post-growth pool shaping.")
    ap.add_argument("--fixed-split-seed", type=int, default=None, help="Keep the train/test split fixed to one seed while varying the training seed.")
    ap.add_argument("--cache-models-dir", type=str, default="", help="Optional external model-cache directory to reuse frozen rules from a previous run.")
    args = ap.parse_args(argv)

    if str(args.dataset_path).strip():
        datasets = [str(args.dataset_path).strip()]
    elif str(args.dataset).strip():
        datasets = [str(args.dataset).strip()]
    else:
        datasets = parse_list(args.datasets)
    inducers = parse_list(args.inducers)
    if args.include_static and "STATIC" not in [x.upper() for x in inducers]:
        inducers.append("STATIC")
    allowed_inducers = {"RIPPER", "FOIL", "STATIC"}
    bad = [x for x in inducers if x.upper().strip() not in allowed_inducers]
    if bad:
        raise ValueError(f"Unsupported inducers: {bad}. Allowed: RIPPER, FOIL, STATIC.")
    protocol = protocol_from_cli(raw_seeds=args.seeds, raw_test_size=float(args.test_size), paper_mode=bool(args.paper_mode))
    seeds = list(protocol.seeds)
    args.test_size = float(protocol.test_size)

    effective_out_root = str(args.save_root).strip() or str(args.out_root).strip()
    out_root = Path(effective_out_root).expanduser().resolve()
    benchmark_dir = out_root / "benchmarks"
    rules_dir = Path(args.rules_dir).expanduser().resolve() if str(args.rules_dir).strip() else (out_root / "rules")
    pkl_dir = Path(args.models_dir).expanduser().resolve() if str(args.models_dir).strip() else (out_root / "models")
    raw_dir = benchmark_dir
    ensure_dir(rules_dir)
    ensure_dir(pkl_dir)
    ensure_dir(raw_dir)

    print(f"[RUN] datasets={datasets}  inducers={inducers}  seeds={seeds}")
    print(f"[RUN] out_root={out_root}")
    print(f"[RUN] protocol_mode={protocol.mode_label}  paper_mode={args.paper_mode}  test_size={args.test_size}")
    for message in protocol.override_messages:
        print(message)
    if args.fixed_split_seed is not None:
        print(f"[RUN] fixed_split_seed={int(args.fixed_split_seed)}")
    print(f"[RUN] systems={SYSTEMS_ALLOWLIST}")
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": int(os.getpid()),
        "datasets": datasets,
        "inducers": [x.upper().strip() for x in inducers],
        "seeds": seeds,
        "requested_seeds": list(protocol.requested_seeds),
        "test_size": float(args.test_size),
        "requested_test_size": float(protocol.requested_test_size),
        "fixed_split_seed": int(args.fixed_split_seed) if args.fixed_split_seed is not None else None,
        "paper_mode": bool(args.paper_mode),
        "protocol_mode": protocol.mode_label,
        "override_messages": list(protocol.override_messages),
        "paths": {
            "out_root": str(out_root),
            "benchmarks": str(benchmark_dir),
            "rules": str(rules_dir),
            "models": str(pkl_dir),
            "cache_models_dir": str(Path(args.cache_models_dir).expanduser().resolve()) if str(args.cache_models_dir).strip() else None,
        },
        "generator_path": "canonical",
    }
    manifest_name = f"run_manifest__pid{os.getpid()}__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    (benchmark_dir / manifest_name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for ds in datasets:
        ds_path = _resolve_csv(ds)
        X, y, feature_names, value_decoders = load_dataset(ds_path)
        ds_name = ds_path.stem

        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        k = int(len(np.unique(y)))
        ds_profile = _dataset_profile(X, y, value_decoders, feature_names)
        print(
            f"[PROFILE] {ds_name}: n={ds_profile['n_samples']} d={ds_profile['n_features']} "
            f"num={ds_profile['num_like']} disc={ds_profile['disc_like']} "
            f"maj={float(ds_profile['majority_ratio']):.3f}"
        )

        for seed in seeds:
            split_seed = int(args.fixed_split_seed) if args.fixed_split_seed is not None else int(seed)
            X_tr, X_te, y_tr, y_te, _, _ = split_train_test(X, y, test_size=float(args.test_size), seed=split_seed)

            # RF baseline (same split / preproc)
            rf_proba = None
            rf_metrics = None
            rf_fit_wall_sec = None
            if not args.no_rf:
                rf_started = time.perf_counter()
                rf = RandomForestClassifier(
                    n_estimators=400,
                    random_state=seed,
                    n_jobs=int(args.rf_n_jobs),
                )
                rf.fit(X_tr, y_tr)
                rf_fit_wall_sec = float(time.perf_counter() - rf_started)
                rf_proba_raw = rf.predict_proba(X_te)
                rf_proba = align_proba_to_classes(rf_proba_raw, getattr(rf, "classes_", np.arange(k)), k)
                rf_metrics = compute_metrics(y_te, rf_proba)

            for inducer_idx, inducer in enumerate(inducers):
                inducer_u = inducer.upper().strip()
                run_single_dataset_inducer(
                    ds_name=ds_name,
                    inducer_u=inducer_u,
                    seed=seed,
                    split_seed=split_seed,
                    X_tr=X_tr,
                    X_te=X_te,
                    y_tr=y_tr,
                    y_te=y_te,
                    k=k,
                    feature_names=feature_names,
                    value_decoders=value_decoders,
                    ds_profile=ds_profile,
                    args=args,
                    rules_dir=rules_dir,
                    pkl_dir=pkl_dir,
                    raw_dir=raw_dir,
                    rf_metrics=rf_metrics,
                    rf_fit_wall_sec=rf_fit_wall_sec,
                    emit_rf_row=bool(rf_proba is not None and inducer_idx == 0),
                )

    print("\n[DONE]")
    return 0

if __name__ == "__main__":
    raise SystemExit(benchmark_main())
