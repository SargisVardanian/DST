"""Benchmark script for RIPPER/FOIL/STATIC rule models with DST training.

Legacy name: `test_Ripper_DST.py`.

This script is the main experiment entrypoint:
  - single dataset benchmark (STATIC/RIPPER/FOIL + baselines + plot)
  - optional multi-dataset parallel runner (writes per-dataset logs)

Artifacts are written into:
  - `Common_code/dsb_rules/`  (raw + dst rulebases)
  - `Common_code/pkl_rules/`  (trained DST models)
  - `Common_code/results/`    (metrics CSV/PNG + per-sample CSV)

Use `--out-dir` if you want to redirect those folders elsewhere.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

THIS_FILE = Path(__file__).resolve()
COMMON = THIS_FILE.parent
ROOT = COMMON.parent
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from Datasets_loader import load_dataset
from DSClassifierMultiQ import DSClassifierMultiQ
from core import split_train_test
from utils import nll_log_loss, expected_calibration_error

# --- Hyper-parameters ---
LR, WD, BS, VS = 5e-3, 2e-4, 128, 0.2
TEST_SIZE, SEED = 0.16, 42


def tspan(f, *a, **kw):
    """Measure execution time of a function."""
    t0 = time.perf_counter()
    r = f(*a, **kw)
    dt = time.perf_counter() - t0
    return dt, r


def metrics(y_true, y_pred, k: int, *, proba=None, ece_bins: int = 15) -> dict[str, float]:
    """Compute classification metrics (+ optional calibration metrics from probabilities)."""
    labels_all = list(range(int(k)))
    avg = "macro"
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, average=avg, labels=labels_all, zero_division=0),
        "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Precision": precision_score(y_true, y_pred, average=avg, labels=labels_all, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, labels=labels_all, zero_division=0),
    }
    if proba is not None:
        nll = nll_log_loss(y_true, proba)
        out["NLL"] = nll
        out["LogLoss"] = nll  # alias for paper terminology
        out["ECE"] = expected_calibration_error(y_true, proba, n_bins=int(ece_bins))
    return out


def _resolve_csv(name: str) -> Path:
    # Accept explicit file paths.
    try:
        p_in = Path(name)
        if p_in.suffix.lower() == ".csv" and p_in.is_file():
            return p_in
    except Exception:
        pass

    # Accept bare dataset names (without .csv).
    for base in (COMMON.parent, COMMON, Path.cwd()):
        p = base / f"{name}.csv"
        if p.is_file():
            return p

    # Fallback: treat it as-is (useful for relative paths).
    return Path(name)


def _thread_env(seed: int, device: str) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", str(int(seed)))
    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env.setdefault(k, "1")
    dev = str(device).strip().lower()
    if dev == "metal":
        dev = "mps"
    if dev == "cpu":
        env.setdefault("CUDA_VISIBLE_DEVICES", "")
    if dev == "mps":
        # Allows CPU fallback for ops that are not implemented on MPS.
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return env


def run_single(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default=None)
    ap.add_argument("--dataset", dest="dataset_opt", default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument(
        "--device",
        default="auto",
        help="Device: auto|cpu|cuda|cuda:0|mps (Apple Metal). 'metal' is accepted as alias for 'mps'.",
    )
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--debug-rules", action="store_true")
    ap.add_argument("--ece-bins", type=int, default=15, help="Bins for Expected Calibration Error (ECE)")
    ap.add_argument("--seed", type=int, default=SEED, help="Seed for split + initialization")
    ap.add_argument("--run-tag", default="", help="Optional suffix for output filenames (prevents overwrites)")
    ap.add_argument(
        "--out-dir",
        default="",
        help="Optional output root directory. If set, writes pkl_rules/, dsb_rules/, results/ under it.",
    )
    ap.add_argument(
        "--uncertainty-rule-weight",
        type=float,
        default=0.0,
        help="Optional regularization weight to penalize average rule Ω during training (default: 0.0, vanilla DSGD/DSGD++).",
    )
    ap.add_argument("--sort-rules", action="store_true", help="After training, reorder rules by certainty score (affects saved PKL order)")
    ap.add_argument("--certainty-score-mode", default="certainty_label_mass", choices=["certainty", "certainty_label_mass"], help="Certainty score for --sort-rules")
    args = ap.parse_args(argv)

    args.dataset = args.dataset_opt or args.dataset
    if not args.dataset:
        ap.error("dataset required")

    csv_path = _resolve_csv(args.dataset)
    X, y, feat_names, decoders, stats = load_dataset(csv_path, return_stats=True)
    print(f"Dataset: {csv_path.stem}, shape={X.shape}, classes={stats.get('classes')}")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    k = len(np.unique(y))

    # Make the whole run reproducible (rule gen + mass init + baselines).
    try:
        random.seed(int(args.seed))
    except Exception:
        pass
    np.random.seed(int(args.seed))
    try:
        import torch
        torch.manual_seed(int(args.seed))
    except Exception:
        pass

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = split_train_test(
        X,
        y,
        test_size=TEST_SIZE,
        seed=int(args.seed),
        stratify=True,
    )
    stem = csv_path.stem
    out_root = Path(args.out_dir).expanduser().resolve() if str(args.out_dir).strip() else COMMON
    out_pkl = out_root / "pkl_rules"
    out_dsb = out_root / "dsb_rules"
    out_res = out_root / "results"
    for d in [out_pkl, out_dsb, out_res]:
        d.mkdir(parents=True, exist_ok=True)

    rows, order = [], []

    def add(tag, m):
        rows.append({"System": tag, **m})
        if tag not in order:
            order.append(tag)

    bs = args.batch_size or BS
    print(f"\nHyper-params: lr={LR}, wd={WD}, bs={bs}, val_split={VS}")
    if float(args.uncertainty_rule_weight) > 0.0:
        print(f"Uncertainty regularization: weight={float(args.uncertainty_rule_weight)}")

    run_tag = f"_{args.run_tag}" if str(args.run_tag).strip() else ""

    for algo in ("STATIC", "RIPPER", "FOIL"):
        print(f"\n{'─'*60}\n[{algo}]\n{'─'*60}")
        clf = DSClassifierMultiQ(
            k=k,
            device=args.device,
            rule_algo=algo,
            max_iter=args.epochs,
            lr=LR,
            weight_decay=WD,
            batch_size=bs,
            val_split=VS,
            debug=args.debug_rules,
            seed=int(args.seed),
            combination_rule="dempster",
            uncertainty_rule_weight=float(args.uncertainty_rule_weight),
            sort_rules_by_certainty=bool(args.sort_rules),
            certainty_score_mode=str(args.certainty_score_mode),
        )
        clf.model.feature_names = feat_names
        clf.model.value_names = decoders or {}

        # Generate rules
        t, _ = tspan(clf.generate_rules, X_tr, y_tr, feature_names=feat_names, rule_algo=algo)
        print(f"[RAW] rules generated in {t:.2f}s")
        clf.model.save_rules_dsb(str(out_dsb / f"{algo.lower()}_{stem}{run_tag}_raw.dsb"))

        # RAW vote baseline (for RIPPER/FOIL)
        if algo != "STATIC":
            clf.model.eval()
            try:
                p_raw = clf.model.predict_with_dst(X_te, combination_rule="vote")
                y_raw = p_raw.argmax(axis=1)
                m_raw = metrics(y_te, y_raw, k, proba=p_raw, ece_bins=args.ece_bins)
                print(f"[RAW] Acc={m_raw['Accuracy']:.4f} F1={m_raw['F1']:.4f}")
                add(f"{algo[0].lower()}_raw", m_raw)
            finally:
                clf.model.train()

        # Train DST
        t, _ = tspan(clf.fit, X_tr, y_tr)
        print(f"[DST] trained in {t:.2f}s")

        # Predict with Dempster
        p_dst = clf.model.predict_with_dst(X_te, combination_rule="dempster")
        y_dst = p_dst.argmax(axis=1)
        m_dst = metrics(y_te, y_dst, k, proba=p_dst, ece_bins=args.ece_bins)

        # Uncertainty: report BOTH
        unc = clf.model.uncertainty_stats(X_te, combination_rule="dempster")
        unc_rule = float(np.nanmean(unc["unc_rule"]))
        unc_comb = float(np.nanmean(unc["unc_comb"]))

        m_dst["unc_comb"] = unc_comb
        m_dst["unc_mean"] = unc_rule

        # Backward-compatible names (older plots/scripts may expect these).
        m_dst["Uncertainty"] = unc_comb
        m_dst["Uncertainty_combined"] = unc_comb
        m_dst["Uncertainty_rule_avg"] = unc_rule
        print(f"[DST] Acc={m_dst['Accuracy']:.4f} F1={m_dst['F1']:.4f} Unc={unc_comb:.4f} Unc_rule={unc_rule:.4f}")

        tag = {"STATIC": "s_dst", "RIPPER": "r_dst", "FOIL": "f_dst"}[algo]
        add(tag, m_dst)

        # Per-class report
        print(classification_report(y_te, y_dst, digits=3, zero_division=0))

        # Save per-sample: idx, y_true, y_pred, omega_rule
        samples_dir = out_res / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        samples_df = pd.DataFrame({
            "sample_idx": idx_te,
            "y_true": y_te,
            "y_pred": y_dst,
            "uncertainty_rule": unc["unc_rule"],
        })
        samples_csv = samples_dir / f"{stem}_{algo.lower()}{run_tag}_samples.csv"
        samples_df.to_csv(samples_csv, index=False)

        clf.save_model(str(out_pkl / f"{algo.lower()}_{stem}{run_tag}_dst.pkl"))
        clf.model.save_rules_dsb(str(out_dsb / f"{algo.lower()}_{stem}{run_tag}_dst.dsb"))

    # Baselines
    print("\nBaselines:")
    for name, mdl in [("dt", DecisionTreeClassifier(random_state=int(args.seed))),
                      ("rf", RandomForestClassifier(n_estimators=200, random_state=int(args.seed), n_jobs=-1)),
                      ("gb", GradientBoostingClassifier(random_state=int(args.seed)))]:
        mdl.fit(X_tr, y_tr)
        yb = mdl.predict(X_te)
        pb = mdl.predict_proba(X_te) if hasattr(mdl, "predict_proba") else None
        mb = metrics(y_te, yb, k, proba=pb, ece_bins=args.ece_bins)
        print(f"[{name}] Acc={mb['Accuracy']:.4f} F1={mb['F1']:.4f}")
        add(f"baseline_{name}", mb)

    # Save metrics CSV + plot
    df = pd.DataFrame(rows)
    csv_out = out_res / f"benchmark_{stem}{run_tag}_metrics.csv"
    df.to_csv(csv_out, index=False)
    print(f"\n✓ Metrics → {csv_out}")

    plot_cols = ["Accuracy", "F1", "F1_macro", "Precision", "Recall", "unc_mean", "unc_comb"]
    plot_cols = [c for c in plot_cols if c in df.columns]
    df_long = df.melt(id_vars=["System"], value_vars=plot_cols, var_name="Metric", value_name="Score")

    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df_long, x="System", y="Score", hue="Metric", palette="Set2", order=order)
    ax.set_ylim(0, 1)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.2f}",
                (p.get_x() + p.get_width() / 2, h),
                ha="center",
                va="bottom",
                fontsize=7,
                xytext=(0, 1),
                textcoords="offset points",
            )
    plt.tight_layout()
    png_out = out_res / f"benchmark_{stem}{run_tag}_metrics.png"
    plt.savefig(png_out, dpi=200)
    plt.close()
    print(f"✓ Plot → {png_out}")
    return 0


def run_batch(argv=None) -> int:
    ap = argparse.ArgumentParser("Parallel batch runner")
    ap.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset paths or names (CSV).",
    )
    ap.add_argument("--jobs", type=int, default=3, help="Parallel jobs.")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--debug-rules", action="store_true")
    ap.add_argument("--ece-bins", type=int, default=15)
    ap.add_argument("--out-dir", default="", help="Artifact root (creates pkl_rules/, dsb_rules/, results/ under it).")
    ap.add_argument(
        "--uncertainty-rule-weight",
        type=float,
        default=0.0,
        help="Forwarded to the single-run benchmark.",
    )
    ap.add_argument("--sort-rules", action="store_true")
    ap.add_argument("--certainty-score-mode", default="certainty_label_mass", choices=["certainty", "certainty_label_mass"])
    ap.add_argument(
        "--batch-tag",
        default="",
        help="Log folder tag; defaults to timestamp like 20260201_175200",
    )
    ap.add_argument(
        "--log-root",
        default=str(COMMON / "results" / "batch_runs"),
        help="Where to create the batch log directory.",
    )
    args = ap.parse_args(argv)

    datasets = [_resolve_csv(d) for d in args.datasets]
    for ds in datasets:
        if not ds.exists():
            raise SystemExit(f"dataset not found: {ds}")

    tag = str(args.batch_tag).strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = Path(args.log_root).expanduser().resolve() / tag
    log_root.mkdir(parents=True, exist_ok=True)

    print(f"Batch tag: {tag}")
    print(f"Log root: {log_root}")
    print(f"Jobs: {int(args.jobs)}  Epochs: {int(args.epochs)}  Seed: {int(args.seed)}  Device: {args.device}")
    print("")

    def _one(ds: Path) -> tuple[str, int]:
        stem = ds.stem
        log_dir = log_root / stem
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "run.log"

        cmd = [
            sys.executable,
            "-u",
            str(THIS_FILE),
            "--dataset",
            str(ds),
            "--epochs",
            str(int(args.epochs)),
            "--seed",
            str(int(args.seed)),
            "--device",
            str(args.device),
            "--ece-bins",
            str(int(args.ece_bins)),
            "--uncertainty-rule-weight",
            str(float(args.uncertainty_rule_weight)),
            "--certainty-score-mode",
            str(args.certainty_score_mode),
            "--out-dir",
            str(args.out_dir),
        ]
        if args.batch_size is not None:
            cmd += ["--batch-size", str(int(args.batch_size))]
        if args.debug_rules:
            cmd += ["--debug-rules"]
        if args.sort_rules:
            cmd += ["--sort-rules"]

        with log_path.open("w", encoding="utf-8") as f:
            f.write("CMD: " + " ".join(cmd) + "\n\n")
            f.flush()
            p = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=_thread_env(seed=int(args.seed), device=str(args.device)),
                cwd=str(ROOT),
            )
        return stem, int(p.returncode)

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as ex:
        futs = {ex.submit(_one, ds): ds for ds in datasets}
        for fut in as_completed(futs):
            ds = futs[fut]
            stem, code = fut.result()
            status = "OK" if code == 0 else f"FAIL({code})"
            print(f"{stem}: {status}  (log: {log_root / stem / 'run.log'})")
            if code != 0:
                failures.append(stem)

    if failures:
        print("\nFailed datasets:", ", ".join(failures))
        return 2
    return 0


def main(argv=None) -> int:
    # Lightweight dispatch:
    # - if user passes --datasets -> batch mode
    # - else -> single dataset benchmark
    if argv is None:
        argv = sys.argv[1:]
    if any(a == "--datasets" for a in argv):
        return run_batch(argv)
    return run_single(argv)


if __name__ == "__main__":
    raise SystemExit(main())
