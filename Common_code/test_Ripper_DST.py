"""Benchmark script for RIPPER/FOIL/STATIC rule models with DST training (compact)."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --- repo paths / imports -------------------------------------------------
THIS_FILE = Path(__file__).resolve()
COMMON = THIS_FILE.parent
PROJECT_ROOT = COMMON.parent

import sys

if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from Datasets_loader import load_dataset
from DSClassifierMultiQ import DSClassifierMultiQ
from core import masses_to_pignistic

# --- presets (short) ------------------------------------------------------
EXPERIMENT_CONFIGS: dict[str, dict[str, Any]] = {
    "stable": {"label": "Stable", "tag": "stable", "params": {"lr": 1e-3, "weight_decay": 2e-4, "batch_size": 128, "val_split": 0.2}},
    "conservative": {"label": "Conservative", "tag": "conserv", "params": {"lr": 5e-4, "weight_decay": 3e-4, "batch_size": 128, "val_split": 0.25}},
    "baseline": {"label": "Baseline", "tag": "baseline", "params": {"lr": 2e-3, "weight_decay": 1e-4, "batch_size": 128, "val_split": 0.2}},
}


# --- tiny helpers ---------------------------------------------------------
def tspan(f, *a, **kw) -> tuple[float, Any]:
    t0 = time.perf_counter(); r = f(*a, **kw); t1 = time.perf_counter(); return (t1 - t0), r


def metrics(y_true, y_pred, k: int | None = None) -> dict[str, float]:
    """Core metrics: Accuracy, F1, Precision, Recall.

    Для согласованности с обучением:
      • при K=2 → усреднение по положительному классу ("binary");
      • при K>2 → взвешенное усреднение ("weighted").
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    y_true_arr = np.asarray(y_true)
    if k is None:
        k = int(len(np.unique(y_true_arr)))
    avg = "binary" if k == 2 else "weighted"

    return {
        "Accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "F1": float(f1_score(y_true_arr, y_pred, average=avg, zero_division=0)),
        "Precision": float(precision_score(y_true_arr, y_pred, average=avg, zero_division=0)),
        "Recall": float(recall_score(y_true_arr, y_pred, average=avg, zero_division=0)),
    }


def _resolve_dataset_csv(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.suffix.lower() == ".csv" and p.is_file():
        return p
    for c in [PROJECT_ROOT / f"{name_or_path}.csv", COMMON / f"{name_or_path}.csv", Path.cwd() / f"{name_or_path}.csv"]:
        if c.is_file():
            return c
    return PROJECT_ROOT / f"{name_or_path}.csv"


# --- main ----------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="adult")
    ap.add_argument("--breaks", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--experiment", type=str, default="stable", choices=["stable", "conservative", "baseline", "all"], help="Hyper-parameter preset or 'all'")
    ap.add_argument("--debug-rules", action="store_true")
    ap.add_argument("--verbose-rules", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--diversity-threshold", type=float, default=None, help="Enable diversity filtering with given Jaccard threshold (e.g. 0.9)")
    ap.add_argument("--cat-threshold", type=int, default=None, help="Threshold for treating numerical features as categorical")
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"[warn] Ignoring unknown arguments: {' '.join(unknown)}")

    csv_path = _resolve_dataset_csv(args.dataset)

    # Load dataset
    X, y, feat_names, value_decoders, stats = load_dataset(csv_path, return_stats=True)
    try:
        shape = stats.get("shape", (len(X), len(feat_names)))
        print(f"Detected label column: '{stats.get('label_column')}'  →  classes: {stats.get('classes')}")
        print(f"X shape = ( {shape[0]}, {shape[1]} ),  y distribution = {stats.get('distribution')}")
    except Exception:
        pass
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=feat_names)
    y = np.asarray(y).astype(int)
    classes = sorted(list(map(int, np.unique(y))))

    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(X_df.values, y, test_size=0.16, random_state=42, stratify=y)

    # Artifacts
    out_dir_pkl = COMMON / "pkl_rules"; out_dir_dsb = COMMON / "dsb_rules"; out_dir_res = COMMON / "results"
    for d in (out_dir_pkl, out_dir_dsb, out_dir_res):
        d.mkdir(parents=True, exist_ok=True)

    rows_for_plot: list[dict[str, float]] = []
    system_order: list[str] = []

    def _append_row(exp_tag: str, system_tag: str, md: dict[str, float]) -> None:
        name = f"{exp_tag}:{system_tag}"; rows_for_plot.append({"Experiment": exp_tag, "System": name, **md});
        (name not in system_order) and system_order.append(name)
    
    def _predict_labels(mdl, X):
        """Unified prediction to labels for any number of classes (argmax on proba if available)."""
        if hasattr(mdl, "predict_proba"):
            proba = mdl.predict_proba(X)
            return proba.argmax(axis=1).astype(int)
        return mdl.predict(X)

    exp_choice = args.experiment.lower()
    experiment_sequence = ["stable", "conservative", "baseline"] if exp_choice == "all" else [exp_choice]

    for exp_key in experiment_sequence:
        cfg = EXPERIMENT_CONFIGS.get(exp_key, EXPERIMENT_CONFIGS["stable"])
        cfg_params = dict(cfg.get("params", {}))
        if args.batch_size is not None:
            cfg_params["batch_size"] = int(args.batch_size)
        exp_label, exp_tag = cfg.get("label", exp_key), cfg.get("tag", exp_key)
        print("\n" + "=" * 60); print(f"[EXPERIMENT] {exp_label} ({exp_key})"); print("=" * 60)
        lr = cfg_params.pop("lr", 5e-3); weight_decay = cfg_params.pop("weight_decay", 2e-4)
        bits = [f"lr={lr}", f"weight_decay={weight_decay}"]
        if "batch_size" in cfg_params: bits.append(f"batch_size={cfg_params['batch_size']}")
        if "val_split" in cfg_params: bits.append(f"val_split={cfg_params['val_split']}")
        if bits: print(f"Hyper-params: {', '.join(bits)}")

        for algo in ("STATIC", "RIPPER", "FOIL"):
            print("\n" + "─" * 60); print(f"[RUN] exp={exp_tag:<8} algo={algo:<6}  (raw → dst)"); print("─" * 60)
            clf = DSClassifierMultiQ(k=len(classes), device=args.device, rule_algo=algo, max_iter=args.epochs, lr=lr, weight_decay=weight_decay, debug=args.debug_rules, **cfg_params)
            clf.model.feature_names = list(X_df.columns); clf.model.value_names = dict(value_decoders or {})

            stem = Path(csv_path).stem; multi = len(experiment_sequence) > 1 or exp_tag != "stable"; prefix = f"{exp_tag}_" if multi else ""
            pkl_path = out_dir_pkl / f"{prefix}{algo.lower()}_{stem}_dst.pkl"
            raw_path = out_dir_dsb / f"{prefix}{algo.lower()}_{stem}_raw.dsb"
            dst_path = out_dir_dsb / f"{prefix}{algo.lower()}_{stem}_dst.dsb"

            # Rule generation parameters
            gen_params: dict[str, Any] = {}
            if args.diversity_threshold is not None:
                # Пользователь явно включает фильтр разнообразия.
                gen_params["enable_diversity_filter"] = True
                gen_params["diversity_threshold"] = float(args.diversity_threshold)

            if args.cat_threshold is not None:
                gen_params["cat_threshold"] = int(args.cat_threshold)

            t, _ = tspan(
                clf.generate_rules,
                X_tr,
                y_tr,
                feature_names=list(X_df.columns),
                rule_algo=algo,
                verbose=args.verbose_rules,
                **({"breaks": int(args.breaks)} if algo == "STATIC" else {}),
                **gen_params,
            )
            print(f"[time] generate RAW rules: {t:.3f}s")
            try:
                clf.model.save_rules_dsb(str(raw_path)); print(f"✓  Saved RAW DSB → {raw_path}")
            except Exception as raw_err:
                print(f"[warn] failed to save RAW DSB ({raw_err})")

            if args.debug_rules and hasattr(clf.model, "summarize_rules"):
                try: clf.model.summarize_rules(X_tr, y_tr, top_n=10)
                except Exception as dbg_err: print(f"[debug] summarize_rules error: {dbg_err}")

            maj = int(np.argmax(np.bincount(y_tr))) if y_tr.size else 0
            y_raw_vote = clf.predict(X_te, default_label=maj, vote_mode="majority")
            m_raw_vote = metrics(y_te, y_raw_vote, k=len(classes))
            print(f"[RAW] metrics: {m_raw_vote}")
            if algo != "STATIC":
                _append_row(exp_tag, ("r_raw" if algo == "RIPPER" else "f_raw"), m_raw_vote)

            tfit, _ = tspan(clf.fit, X_tr, y_tr)
            if clf.best_val_metrics_ is not None:
                b = clf.best_val_metrics_
                print(
                    f"[VAL] best epoch {int(b['epoch']):03d} → train_loss={b['train_loss']:.5f} val_loss={b['val_loss']:.5f} "
                    f"Acc={b['Accuracy']:.4f} F1={b['F1']:.4f} P={b['Precision']:.4f} R={b['Recall']:.4f} unc={b.get('val_uncertainty', 0.0):.4f}"
                )
            print(f"[time] train DST: {tfit:.3f}s")

            # DST prediction
            def _masses_to_labels(m: np.ndarray, k: int) -> np.ndarray:
                if m.size == 0 or k == 0: return np.zeros(m.shape[0], dtype=int)
                # m is now probabilities (N, K)
                return m.argmax(axis=1).astype(int)

            tpred, masses = tspan(clf.model.predict_masses, X_te)
            masses = np.asarray(masses, dtype=np.float32)
            probs = masses_to_pignistic(masses)
            y_hat = _masses_to_labels(probs, clf.k)
            m_dst = metrics(y_te, y_hat, k=clf.k)
            if masses.size:
                # Test-set uncertainty: mean Omega of active rules (pre-combination)
                unc_samples = clf.model.sample_uncertainty(X_te)
                m_dst["Uncertainty"] = float(np.asarray(unc_samples, dtype=np.float32).mean())
            
            # Uncertainty analysis skipped as we don't have explicit Omega
            print(f"[DST] metrics: {m_dst}  [time predict={tpred:.3f}s]")
            dst_tag = "s_dst" if algo == "STATIC" else ("r_dst" if algo == "RIPPER" else "f_dst")
            _append_row(exp_tag, dst_tag, m_dst)

            clf.save_model(str(pkl_path)); clf.model.save_rules_dsb(str(dst_path))
            print(f"✓  Saved PKL → {pkl_path}"); print(f"✓  Saved DST DSB → {dst_path}")

    # Baselines (short)
    try:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        print("\nBaselines:")
        for sysname, M in [("dt", DecisionTreeClassifier(random_state=42)), ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)), ("gb", GradientBoostingClassifier(random_state=42))]:
            mdl = M.fit(X_tr, y_tr)
            yb = _predict_labels(mdl, X_te)
            mb = metrics(y_te, yb); print(f"[Baseline] {sysname} Acc={mb['Accuracy']:.4f} F1={mb['F1']:.4f} P={mb['Precision']:.4f} R={mb['Recall']:.4f}"); _append_row("bench", f"baseline_{sysname}", mb)
    except Exception:
        pass

    # Save metrics + plot in the requested Seaborn style
    import matplotlib.pyplot as plt, seaborn as sns
    df = pd.DataFrame(rows_for_plot)
    metrics_csv = out_dir_res / f"benchmark_dataset_{Path(csv_path).stem}_metrics.csv"; df.to_csv(metrics_csv, index=False)
    try:
        with open(metrics_csv, "rb+") as fh: os.fsync(fh.fileno())
    except Exception:
        pass
    print(f"\n✓  Metrics  → {metrics_csv}")

    df_long = df.melt(id_vars=["Experiment", "System"], var_name="Metric", value_name="Score")
    order = [name for name in system_order if name in df["System"].values]
    plt.figure(figsize=(12, 7)); sns.set_context("paper"); sns.set_style("whitegrid")
    ax = sns.barplot(data=df_long, x="System", y="Score", hue="Metric", palette="Set2", order=order)
    ax.set_ylim(0.0, 1.0); ax.set_xlabel("Experiment • System"); ax.set_ylabel("Score")
    for p in ax.patches:
        h = p.get_height(); ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2, h), ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    ax.legend(title="Metric", loc="upper right", frameon=True); plt.tight_layout()
    png_path = out_dir_res / f"benchmark_dataset_{Path(csv_path).stem}_metrics.png"
    try:
        if png_path.exists(): png_path.unlink()
    except Exception:
        pass
    plt.savefig(png_path, dpi=300); plt.close(); print(f"✓  Plot     → {png_path}")


if __name__ == "__main__":
    main()
