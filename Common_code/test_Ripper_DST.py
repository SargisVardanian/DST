# coding: utf-8
from __future__ import annotations

import os, time, argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np  # <— импорт только на уровне модуля!

# --- paths / imports ---
THIS_FILE = Path(__file__).resolve()
COMMON = THIS_FILE.parent                  # .../Common_code
PROJECT_ROOT = COMMON.parent               # корень репозитория

import sys
sys.path.insert(0, str(COMMON))

from Datasets_loader import load_dataset
from DSClassifierMultiQ import DSClassifierMultiQ

EXPERIMENT_CONFIGS = {
    "stable": {
        "label": "Stable",
        "tag": "stable",
        "params": {
            "lr": 1e-3,
            "weight_decay": 2e-4,
            "batch_size": 128,
            "val_split": 0.2,
        },
    },
    "conservative": {
        "label": "Conservative",
        "tag": "conserv",
        "params": {
            "lr": 5e-4,
            "weight_decay": 3e-4,
            "batch_size": 128,
            "val_split": 0.25,
        },
    },
    "baseline": {
        "label": "Baseline",
        "tag": "baseline",
        "params": {
            "lr": 2e-3,
            "weight_decay": 1e-4,
            "batch_size": 128,
            "val_split": 0.2,
        },
    },
}

# ---- быстрые утилиты времени ----
def tspan(f, *a, **kw) -> Tuple[float, object]:
    t0 = time.perf_counter()
    r = f(*a, **kw)
    t1 = time.perf_counter()
    return (t1 - t0), r

def metrics(y_true, y_pred) -> Dict[str,float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1":       float(f1_score(y_true, y_pred, zero_division=0)),
        "Precision":float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall":   float(recall_score(y_true, y_pred, zero_division=0)),
    }

def _resolve_dataset_csv(name_or_path: str) -> Path:
    """
    Ищет CSV по нескольким корням:
      1) то, что передали (если абсолютный/относительный путь существует)
      2) PROJECT_ROOT/<name>.csv
      3) COMMON/<name>.csv
      4) CWD/<name>.csv
    """
    p = Path(name_or_path)
    if p.suffix.lower() != ".csv":
        candidates = [
            PROJECT_ROOT / f"{name_or_path}.csv",
            COMMON       / f"{name_or_path}.csv",
            Path.cwd()   / f"{name_or_path}.csv",
        ]
    else:
        candidates = [p, PROJECT_ROOT / p.name, COMMON / p.name, Path.cwd() / p.name]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="adult")
    ap.add_argument("--breaks",  type=int, default=2)
    ap.add_argument("--epochs",  type=int, default=50)
    ap.add_argument("--device",  type=str, default="cpu")
    ap.add_argument("--experiment", type=str, default="stable",
                    choices=["stable", "conservative", "baseline", "all"],
                    help="Which training hyper-parameter preset to use (or 'all' to run every preset).")
    ap.add_argument("--debug-rules", action="store_true",
                    help="Print detailed rule/activation diagnostics during the run.")
    ap.add_argument("--verbose-rules", action="store_true",
                    help="Log each rule accepted during generation.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override training batch size for all experiments.")
    args = ap.parse_args()

    csv_path = _resolve_dataset_csv(args.dataset)

    # -------- load data --------
    X, y, feat_names, value_decoders, stats = load_dataset(csv_path, return_stats=True)
    print(f"Detected label column: '{stats['label_column']}'  →  classes: {stats['classes']}")
    shape = stats.get("shape", (len(X), len(feat_names)))
    print(f"X shape = ( {shape[0]}, {shape[1]} ),  y distribution = {stats['distribution']}")
    import pandas as pd
    X_df = pd.DataFrame(X, columns=feat_names) if not isinstance(X, pd.DataFrame) else X
    y = np.asarray(y).astype(int)
    classes = sorted(list(map(int, np.unique(y))))

    print()

    # split
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X_df.values, y, test_size=0.16, random_state=42, stratify=y)

    # --- артефакты ---
    out_dir_pkl = COMMON / "pkl_rules"
    out_dir_dsb = COMMON / "dsb_rules"
    out_dir_res = COMMON / "results"          # единственное место для результатов
    out_dir_pkl.mkdir(parents=True, exist_ok=True)
    out_dir_dsb.mkdir(parents=True, exist_ok=True)
    out_dir_res.mkdir(parents=True, exist_ok=True)

    rows_for_plot: List[Dict[str, float]] = []
    system_order: List[str] = []

    def _append_row(exp_tag: str, system_tag: str, metrics_dict: Dict[str, float]) -> None:
        name = f"{exp_tag}:{system_tag}"
        row = {"Experiment": exp_tag, "System": name, **metrics_dict}
        rows_for_plot.append(row)
        if name not in system_order:
            system_order.append(name)

    exp_choice = args.experiment.lower()
    if exp_choice == "all":
        experiment_sequence = ["stable", "conservative", "baseline"]
    else:
        experiment_sequence = [exp_choice]

    for exp_key in experiment_sequence:
        cfg = EXPERIMENT_CONFIGS.get(exp_key, EXPERIMENT_CONFIGS["stable"])
        cfg_params = dict(cfg.get("params", {}))
        if args.batch_size is not None:
            cfg_params["batch_size"] = int(args.batch_size)
        exp_label = cfg.get("label", exp_key)
        exp_tag = cfg.get("tag", exp_key)
        print("\n" + "="*60)
        print(f"[EXPERIMENT] {exp_label} ({exp_key})")
        print("="*60)
        lr = cfg_params.pop("lr", 5e-3)
        weight_decay = cfg_params.pop("weight_decay", 2e-4)
        summary_bits = [f"lr={lr}", f"weight_decay={weight_decay}"]
        if "batch_size" in cfg_params:
            summary_bits.append(f"batch_size={cfg_params['batch_size']}")
        if "val_split" in cfg_params:
            summary_bits.append(f"val_split={cfg_params['val_split']}")
        if summary_bits:
            print(f"Hyper-params: {', '.join(summary_bits)}")

        for algo in ["STATIC", "RIPPER", "FOIL"]:
            print("\n" + "─"*60)
            print(f"[RUN] exp={exp_tag:<8} algo={algo:<6}  (raw → dst)")
            print("─"*60)

            algo_params = dict(cfg_params)
            clf = DSClassifierMultiQ(
                k=len(classes),
                device=args.device,
                algo=algo,
                value_decoders=value_decoders,
                feature_names=list(X_df.columns),
                max_iter=args.epochs,
                lr=lr,
                weight_decay=weight_decay,
                debug=args.debug_rules,
                **algo_params,
            )

            stem = Path(csv_path).stem
            multi_runs = len(experiment_sequence) > 1 or exp_tag != "stable"
            prefix = f"{exp_tag}_" if multi_runs else ""
            pkl_path = out_dir_pkl / f"{prefix}{algo.lower()}_{stem}_dst.pkl"
            raw_path = out_dir_dsb / f"{prefix}{algo.lower()}_{stem}_raw.dsb"
            dst_path = out_dir_dsb / f"{prefix}{algo.lower()}_{stem}_dst.dsb"

            t, _ = tspan(
                clf.generate_raw,
                X_tr, (y_tr if algo != "STATIC" else None),
                feature_names=list(X_df.columns),
                algo=algo,
                verbose_rules=args.verbose_rules,
                **({"breaks": int(args.breaks)} if algo == "STATIC" else {})
            )
            print(f"[time] generate RAW rules: {t:.3f}s")

            try:
                clf.model.save_rules_dsb(str(raw_path))
                print(f"✓  Saved RAW DSB → {raw_path}")
            except Exception as raw_err:
                print(f"[warn] failed to save RAW DSB ({raw_err})")

            if args.debug_rules and hasattr(clf.model, "summarize_rules"):
                try:
                    clf.model.summarize_rules(X_tr, y_tr, top_n=10)
                except Exception as dbg_err:
                    print(f"[debug] summarize_rules error: {dbg_err}")

            if algo != "STATIC":
                y_raw = clf.model.predict_by_rule_labels(X_te)
                m_raw = metrics(y_te, y_raw)
                print(f"[RAW] metrics: {m_raw}")
                raw_tag = "r_raw" if algo == "RIPPER" else "f_raw"
                _append_row(exp_tag, raw_tag, m_raw)
            else:
                print("[RAW] metrics skipped for STATIC.")

            tfit, _ = tspan(clf.fit, X_tr, y_tr)
            if clf.best_val_metrics_ is not None:
                best = clf.best_val_metrics_
                print(
                    f"[VAL] best epoch {int(best['epoch']):03d} → "
                    f"train_loss={best['train_loss']:.5f} val_loss={best['val_loss']:.5f} "
                    f"Acc={best['Accuracy']:.4f} F1={best['F1']:.4f} "
                    f"P={best['Precision']:.4f} R={best['Recall']:.4f} "
                    f"unc={best.get('val_uncertainty', 0.0):.4f}"
                )
            print(f"[time] train DST: {tfit:.3f}s")

            tpred, y_hat = tspan(clf.predict, X_te)
            m_dst = metrics(y_te, y_hat)
            print(f"[DST] metrics: {m_dst}  [time predict={tpred:.3f}s]")
            if getattr(clf, "last_uncertain_mask_", None) is not None:
                mask = np.asarray(clf.last_uncertain_mask_, dtype=bool)
                uncertain_count = int(mask.sum())
                print(f"[DST] uncertain predictions: {uncertain_count} / {mask.size} ({uncertain_count / max(1, mask.size):.2%})")

            dst_tag = "s_dst" if algo == "STATIC" else ("r_dst" if algo == "RIPPER" else "f_dst")
            _append_row(exp_tag, dst_tag, m_dst)

            clf.save_model(str(pkl_path))
            clf.model.save_rules_dsb(str(dst_path))
            print(f"✓  Saved PKL → {pkl_path}")
            print(f"✓  Saved DST DSB → {dst_path}")

    # Бейзлайны
    try:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

        print("\nBaselines:")
        for sysname, M in [
            ("dt", DecisionTreeClassifier(random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
            ("gb", GradientBoostingClassifier(random_state=42)),
        ]:
            mdl = M.fit(X_tr, y_tr)
            if hasattr(mdl, "predict_proba"):
                yb = (mdl.predict_proba(X_te)[:, 1] >= 0.5).astype(int)
            else:
                yb = mdl.predict(X_te)
            mb = metrics(y_te, yb)
            print(f"[Baseline] {sysname} Acc={mb['Accuracy']:.4f} F1={mb['F1']:.4f} "
                  f"P={mb['Precision']:.4f} R={mb['Recall']:.4f}")
            _append_row("bench", f"baseline_{sysname}", mb)
    except Exception:
        pass

    # -------- save metrics + plot (SEABORN) --------
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.DataFrame(rows_for_plot)
    metrics_csv = out_dir_res / f"benchmark_dataset_{Path(csv_path).stem}_metrics.csv"
    df.to_csv(metrics_csv, index=False)
    try:
        with open(metrics_csv, "rb+") as fh:
            os.fsync(fh.fileno())
    except Exception:
        pass
    print(f'\n✓  Metrics  → {metrics_csv}')

    df_long = df.melt(id_vars=["Experiment", "System"], var_name="Metric", value_name="Score")
    order = [name for name in system_order if name in df["System"].values]

    plt.figure(figsize=(12, 7))
    sns.set_context("paper"); sns.set_style("whitegrid")
    ax = sns.barplot(data=df_long, x="System", y="Score", hue="Metric",
                     palette="Set2", order=order)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Experiment • System")
    ax.set_ylabel("Score")
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    ax.legend(title="Metric", loc="upper right", frameon=True)
    plt.tight_layout()

    png_path = out_dir_res / f"benchmark_dataset_{Path(csv_path).stem}_metrics.png"
    try:
        if png_path.exists():
            png_path.unlink()
    except Exception:
        pass
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"✓  Plot     → {png_path}")

if __name__ == "__main__":
    main()
