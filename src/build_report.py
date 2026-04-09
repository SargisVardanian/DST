#!/usr/bin/env python3
"""Canonical full benchmark/report pipeline for the DST project.

Public role:
- run benchmark/train-test through train_test_runner.py
- aggregate benchmark outputs into CSVs and plots
- run hard-case analysis
- generate the brief markdown report
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .benchmark_protocol import protocol_from_passthrough
    from .report_brief import build_summary, read_rows, render_report, write_brief_report
    from .train_test_runner import benchmark_main
    from . import analyze_hard_cases
except ImportError:  # pragma: no cover - direct script/import fallback
    from benchmark_protocol import protocol_from_passthrough
    from report_brief import build_summary, read_rows, render_report, write_brief_report
    from train_test_runner import benchmark_main
    import analyze_hard_cases


THIS = Path(__file__).resolve()
COMMON = THIS.parent
ROOT = COMMON.parent

DEFAULT_DATASETS = [
    "adult",
    "bank-full",
    "BrainTumor",
    "breast-cancer-wisconsin",
    "df_wine",
    "dry-bean",
    "gas_drift",
    "german",
    "magic-gamma",
]
DEFAULT_OUT_ROOT = COMMON / "results" / "raw_runs"
DEFAULT_RESULTS_DIR = COMMON / "results"
DEFAULT_HARD_CASE_DIR = DEFAULT_RESULTS_DIR / "hard_cases"
DEFAULT_BRIEF_OUTPUT = DEFAULT_RESULTS_DIR / "brief_report.md"

BASE_METHOD_ORDER = [
    "RF",
    "RIPPER:native_ordered_rule",
    "RIPPER:first_hit_laplace",
    "RIPPER:weighted_vote",
    "RIPPER:dsgd_dempster",
    "FOIL:native_ordered_rule",
    "FOIL:first_hit_laplace",
    "FOIL:weighted_vote",
    "FOIL:dsgd_dempster",
]
PASTEL_METHOD_COLORS = {
    "RF": "#c9c9c9",
    "RIPPER:native_ordered_rule": "#d7e3f7",
    "RIPPER:first_hit_laplace": "#cfe1f2",
    "RIPPER:weighted_vote": "#cfe8d5",
    "FOIL:native_ordered_rule": "#f6decf",
    "FOIL:first_hit_laplace": "#f8d9c4",
    "FOIL:weighted_vote": "#f3e7b3",
}
BRIGHT_METHOD_COLORS = {
    "RIPPER:dsgd_dempster": "#d73027",
    "FOIL:dsgd_dempster": "#ff7f0e",
}
UNCERTAINTY_METHOD_ORDER = [
    "RIPPER:dsgd_dempster",
    "FOIL:dsgd_dempster",
]
METRIC_CONFIG = {
    "acc": {"title": "Accuracy", "lower_is_better": False},
    "macro_f1": {"title": "Macro-F1", "lower_is_better": False},
    "precision": {"title": "Precision", "lower_is_better": False},
    "recall": {"title": "Recall", "lower_is_better": False},
    "nll": {"title": "NLL", "lower_is_better": True},
    "ece": {"title": "ECE", "lower_is_better": True},
    "unc_mean": {"title": "Average Rule Uncertainty", "lower_is_better": True},
    "unc_comb": {"title": "Average Combined Uncertainty", "lower_is_better": True},
}
LOWER_IS_BETTER = {"nll", "ece", "unc_mean", "unc_comb"}


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.resolve().relative_to(other.resolve())
        return True
    except Exception:
        return False


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_manifest(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except Exception:
        shutil.copy2(src, dst)
        return
    paths = data.get("paths")
    if isinstance(paths, dict) and "reports" in paths:
        paths = dict(paths)
        paths.pop("reports", None)
        data["paths"] = paths
    dst.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _normalize_method(row: pd.Series) -> str:
    system = str(row.get("system", "")).strip()
    inducer = str(row.get("inducer", "")).strip().upper()
    if system == "rf" or inducer == "RF":
        return "RF"
    return f"{inducer}:{system}"


def _all_method_order() -> list[str]:
    return list(BASE_METHOD_ORDER)


def _present_method_order(df: pd.DataFrame, *, uncertainty_only: bool = False) -> list[str]:
    present = set(df["method"].astype(str).tolist()) if not df.empty and "method" in df.columns else set()
    base = UNCERTAINTY_METHOD_ORDER if uncertainty_only else _all_method_order()
    ordered = [method for method in base if method in present]
    tail = sorted(present - set(ordered))
    return ordered + tail


def _collect(bench_dir: Path, datasets: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(bench_dir.glob("bench__*.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        ds = str(df["dataset"].iloc[0]).strip()
        if datasets and ds not in datasets:
            continue
        cur = df.copy()
        cur["dataset"] = ds
        cur["method"] = cur.apply(_normalize_method, axis=1)
        cur = cur[cur["method"].astype(str).str.contains(":", regex=False) | (cur["method"] == "RF")].copy()
        if cur.empty:
            continue
        rows.append(cur)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    dedupe_subset = [col for col in ("dataset", "method", "seed") if col in out.columns] or ["dataset", "method"]
    method_order = _present_method_order(out)
    out = out.sort_values(dedupe_subset).drop_duplicates(subset=dedupe_subset, keep="first")
    out["method"] = pd.Categorical(out["method"], categories=method_order, ordered=True)
    sort_cols = [col for col in ("dataset", "method", "seed") if col in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


def _aggregate_runs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    numeric_cols = [metric for metric in (*METRIC_CONFIG.keys(), "fusion_depth") if metric in df.columns]
    group_cols = ["dataset", "method"]
    grouped = df.groupby(group_cols, observed=False)
    if numeric_cols:
        frames = []
        for metric in numeric_cols:
            metric_frame = grouped[metric].agg(["mean", "std", "min", "max"]).reset_index()
            metric_frame = metric_frame.rename(
                columns={
                    "mean": metric,
                    "std": f"std_{metric}",
                    "min": f"min_{metric}",
                    "max": f"max_{metric}",
                }
            )
            frames.append(metric_frame)
        agg = frames[0]
        for frame in frames[1:]:
            agg = agg.merge(frame, on=group_cols, how="outer")
    else:
        agg = grouped.size().reset_index(name="n_runs")
    agg = agg.merge(grouped.size().reset_index(name="n_runs"), on=group_cols, how="left")
    if "seed" in df.columns:
        agg = agg.merge(grouped["seed"].nunique().reset_index(name="n_seeds"), on=group_cols, how="left")
    method_order = _present_method_order(df)
    agg["method"] = pd.Categorical(agg["method"], categories=method_order, ordered=True)
    return agg.sort_values(["dataset", "method"]).reset_index(drop=True)


def _metric_method_order(df: pd.DataFrame, metric: str) -> list[str]:
    base_order = _present_method_order(df, uncertainty_only=metric in {"unc_mean", "unc_comb"})
    cur = df[df["method"].isin(base_order)].copy()
    if cur.empty or metric not in cur.columns:
        return list(base_order)
    means = cur.groupby("method", observed=False)[metric].mean()
    lower_is_better = bool(METRIC_CONFIG[metric]["lower_is_better"])
    return sorted(
        [m for m in base_order if m in means.index],
        key=lambda method: (float(means.get(method, np.nan)), base_order.index(method)),
        reverse=not lower_is_better,
    )


def _pivot(df: pd.DataFrame, metric: str, datasets: list[str]) -> pd.DataFrame:
    method_order = _metric_method_order(df, metric)
    cur = df[df["method"].isin(method_order)].copy()
    pvt = cur.pivot(index="dataset", columns="method", values=metric)
    return pvt.reindex(index=datasets, columns=method_order)


def _method_color(method: str) -> str:
    if method in BRIGHT_METHOD_COLORS:
        return BRIGHT_METHOD_COLORS[method]
    return PASTEL_METHOD_COLORS.get(method, "#d9d9d9")


def _mean_rank_table(df: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    rows = []
    for metric, cfg in METRIC_CONFIG.items():
        pvt = _pivot(df, metric, datasets)
        ranks = pvt.rank(axis=1, method="average", ascending=bool(cfg["lower_is_better"]))
        for method in _metric_method_order(df, metric):
            if method not in ranks.columns:
                continue
            vals = ranks[method].dropna()
            if vals.empty:
                continue
            rows.append({"metric": metric, "method": method, "mean_rank": float(vals.mean()), "std_rank": float(vals.std(ddof=0))})
    out = pd.DataFrame(rows)
    return out.sort_values(["metric", "mean_rank", "method"]).reset_index(drop=True) if not out.empty else out


def _best_rule_based_vs_rf(df: pd.DataFrame, datasets: list[str], metric: str) -> pd.DataFrame:
    rows = []
    lower_is_better = bool(METRIC_CONFIG[metric]["lower_is_better"])
    for dataset in datasets:
        cur = df[df["dataset"] == dataset].copy()
        if cur.empty:
            continue
        rf_row = cur[cur["method"] == "RF"]
        if rf_row.empty or metric not in rf_row.columns:
            continue
        rf_val = float(rf_row.iloc[0][metric])
        rule_cur = cur[cur["method"] != "RF"].copy()
        rule_cur = rule_cur[np.isfinite(rule_cur[metric].astype(float))]
        if rule_cur.empty:
            continue
        best = rule_cur.sort_values(metric, ascending=lower_is_better).iloc[0]
        best_val = float(best[metric])
        delta = (rf_val - best_val) if lower_is_better else (best_val - rf_val)
        rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "rf_method": "RF",
                "rf_value": rf_val,
                "best_rule_method": str(best["method"]),
                "best_rule_value": best_val,
                "delta_vs_rf": float(delta),
                "beats_rf": bool(delta > 0),
            }
        )
    return pd.DataFrame(rows)


def _plot_grouped_bars(pvt: pd.DataFrame, *, title: str, out_path: Path, lower_is_better: bool) -> None:
    if pvt.empty:
        return
    fig, ax = plt.subplots(figsize=(max(16.0, 1.25 * len(pvt.index)), 7.0))
    x = np.arange(len(pvt.index), dtype=float)
    n_methods = len(pvt.columns)
    width = min(0.9 / max(1, n_methods), 0.14)
    for i, method in enumerate(pvt.columns):
        vals = pvt[method].to_numpy(dtype=float)
        pos = x + (i - (n_methods - 1) / 2.0) * width
        ax.bar(
            pos,
            vals,
            width=width,
            label=method,
            color=_method_color(str(method)),
            alpha=0.98 if "dsgd_dempster" in str(method) else 0.88,
            edgecolor="#555555",
            linewidth=0.5 if "dsgd_dempster" in str(method) else 0.3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(pvt.index.tolist(), rotation=20, ha="right")
    ax.set_title(f"All {len(pvt.index)} datasets: {title}" + (" (lower is better)" if lower_is_better else ""), fontweight="bold")
    ax.set_ylabel(title)
    ax.grid(axis="y", alpha=0.35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_heatmap(pvt: pd.DataFrame, *, title: str, out_path: Path, lower_is_better: bool) -> None:
    if pvt.empty:
        return
    vals = pvt.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(12.0, 0.95 * len(pvt.columns)), max(6.0, 0.70 * len(pvt.index))))
    im = ax.imshow(vals, cmap="viridis_r" if lower_is_better else "viridis", aspect="auto")
    ax.set_xticks(np.arange(len(pvt.columns)))
    ax.set_xticklabels(pvt.columns.tolist(), rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pvt.index)))
    ax.set_yticklabels(pvt.index.tolist())
    ax.set_title(f"All {len(pvt.index)} datasets: {title} heatmap", fontweight="bold")
    for tick, method in zip(ax.get_xticklabels(), pvt.columns.tolist()):
        if "dsgd_dempster" in str(method):
            tick.set_color(_method_color(str(method)))
            tick.set_fontweight("bold")
        else:
            tick.set_color("#6a6a6a")
    finite = vals[np.isfinite(vals)]
    if finite.size:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        thresh = vmin + 0.55 * (vmax - vmin if vmax > vmin else 1.0)
    else:
        thresh = 0.0
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            txt = "NA" if not np.isfinite(v) else f"{v:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black" if (not np.isfinite(v) or v < thresh) else "white")
    fig.colorbar(im, ax=ax, shrink=0.88)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _load_hard_case_tables(hard_case_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_summary = hard_case_dir / "hard_case_overview.csv"
    method_summary = hard_case_dir / "hard_case_method_summary.csv"
    if not base_summary.exists() or not method_summary.exists():
        return pd.DataFrame(), pd.DataFrame()
    return pd.read_csv(base_summary), pd.read_csv(method_summary)


def _write_dataset_plots(long_csv: Path, out_dir: Path) -> None:
    if not long_csv.exists():
        return
    df = pd.read_csv(long_csv)
    if df.empty:
        return
    method_order = _present_method_order(df)
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)
    for dataset in sorted(df["dataset"].dropna().unique()):
        cur = df[df["dataset"] == dataset].copy().sort_values("method")
        vals = cur[["method", "macro_f1"]].dropna()
        if vals.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(vals["method"], vals["macro_f1"], color=[_method_color(str(method)) for method in vals["method"]], edgecolor="#555555", linewidth=0.4)
        ax.set_title(f"{dataset}: Macro-F1 by method", fontweight="bold")
        ax.set_ylabel("Macro-F1")
        ax.set_ylim(0.0, min(1.0, max(1.0, float(vals["macro_f1"].max()) + 0.05)))
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{dataset}__macro_f1_methods.png", dpi=180)
        plt.close(fig)


def _write_method_suite_report(
    df: pd.DataFrame,
    rank_df: pd.DataFrame,
    rf_delta_tables: dict[str, pd.DataFrame],
    hard_case_base: pd.DataFrame,
    hard_case_methods: pd.DataFrame,
    *,
    out_path: Path,
) -> None:
    methods = _present_method_order(df)
    lines = ["# Unified Method Report", "", "Methods compared on the same benchmark split across all datasets:", *[f"- {method}" for method in methods], ""]
    for metric, cfg in METRIC_CONFIG.items():
        sub = rank_df[rank_df["metric"] == metric].copy()
        if sub.empty:
            continue
        lines.extend([f"## Mean rank: {cfg['title']}", "", "| Method | Mean rank | Std |", "|---|---:|---:|"])
        for _, row in sub.sort_values("mean_rank").iterrows():
            lines.append(f"| {row['method']} | {row['mean_rank']:.3f} | {row['std_rank']:.3f} |")
        lines.append("")
    for metric, cfg in METRIC_CONFIG.items():
        delta_df = rf_delta_tables.get(metric)
        if delta_df is None or delta_df.empty:
            continue
        lines.extend([f"## Best rule-based vs RF: {cfg['title']}", "", "| Dataset | RF | Best rule-based | Delta vs RF | Beats RF |", "|---|---:|---|---:|---|"])
        for _, row in delta_df.iterrows():
            lines.append(f"| {row['dataset']} | {row['rf_value']:.4f} | {row['best_rule_method']} ({row['best_rule_value']:.4f}) | {row['delta_vs_rf']:.4f} | {'yes' if bool(row['beats_rf']) else 'no'} |")
        lines.append("")
    if not hard_case_base.empty:
        lines.extend(["## Hard-case base summary", "", "| Dataset | Algo | N hard | Weighted acc | DSGD acc | RF acc |", "|---|---|---:|---:|---:|---:|"])
        for _, row in hard_case_base[["dataset", "algo", "n_hard_cases", "weighted_vote_acc", "dsgd_dempster_acc", "rf_acc"]].iterrows():
            lines.append(f"| {row['dataset']} | {row['algo']} | {int(row['n_hard_cases'])} | {row['weighted_vote_acc']:.4f} | {row['dsgd_dempster_acc']:.4f} | {row['rf_acc']:.4f} |")
        lines.append("")
    if not hard_case_methods.empty:
        lines.extend(["## Hard-case mean by method", "", "| Algo | Method | Acc | Macro-F1 | Precision | Recall |", "|---|---|---:|---:|---:|---:|"])
        grp = hard_case_methods.groupby(["algo", "method"], as_index=False)[["acc", "macro_f1", "precision", "recall"]].mean().sort_values(["algo", "method"])
        for _, row in grp.iterrows():
            lines.append(f"| {row['algo']} | {row['method']} | {row['acc']:.4f} | {row['macro_f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} |")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def report_outputs_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build final result tables and plots into the canonical results directory.")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Benchmark/model artifact root.")
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--bench-dir", default="", help="Optional explicit benchmark dir (defaults to <out-root>/benchmarks).")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Final results directory.")
    ap.add_argument("--hard-case-dir", default=str(DEFAULT_HARD_CASE_DIR), help="Directory for hard-case outputs.")
    ap.add_argument("--include-hard-cases", action="store_true", help="Append hard-case summaries when files exist.")
    args = ap.parse_args(argv)

    out_root = Path(args.out_root).expanduser().resolve()
    bench_dir = Path(args.bench_dir).expanduser().resolve() if str(args.bench_dir).strip() else (out_root / "benchmarks")
    results_dir = Path(args.results_dir).expanduser().resolve()
    method_suite_dir = results_dir / "method_suite"
    plots_metric_dir = results_dir / "plots_by_metric"
    plots_dataset_dir = results_dir / "plots_by_dataset"
    raw_runs_dir = results_dir / "raw_runs"
    hard_case_dir = Path(args.hard_case_dir).expanduser().resolve()
    datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]

    raw_df = _collect(bench_dir, datasets=datasets)
    if raw_df.empty:
        print(f"[error] no benchmark files in {bench_dir}")
        return 2
    df = _aggregate_runs(raw_df)
    bench_files = sorted(bench_dir.glob("bench__*.csv"))
    manifest_files = sorted(bench_dir.glob("run_manifest*.json"))

    results_dir.mkdir(parents=True, exist_ok=True)
    for old in list(results_dir.glob("ALL_DATASETS_*")):
        if old.is_file():
            old.unlink()
    for path in [method_suite_dir, plots_metric_dir, plots_dataset_dir]:
        _reset_dir(path)
    bench_inside_raw_runs = _is_relative_to(bench_dir, raw_runs_dir)
    if not bench_inside_raw_runs:
        _reset_dir(raw_runs_dir)
    hard_case_dir.mkdir(parents=True, exist_ok=True)
    if not bench_inside_raw_runs:
        for src in bench_files:
            _copy_file(src, raw_runs_dir / src.name)
        for src in manifest_files:
            _copy_manifest(src, raw_runs_dir / src.name)

    raw_df.to_csv(method_suite_dir / "method_suite_long_raw.csv", index=False)
    df.to_csv(method_suite_dir / "method_suite_long.csv", index=False)
    df.to_csv(results_dir / "ALL_DATASETS_metrics.csv", index=False)
    rank_df = _mean_rank_table(df, datasets=datasets)
    rank_df.to_csv(method_suite_dir / "method_suite_mean_rank.csv", index=False)
    rank_df.to_csv(results_dir / "ALL_DATASETS_evaluation_summary.csv", index=False)
    rf_delta_tables: dict[str, pd.DataFrame] = {}
    for metric, cfg in METRIC_CONFIG.items():
        pvt = _pivot(df, metric, datasets=datasets)
        pvt.to_csv(method_suite_dir / f"method_suite_{metric}_pivot.csv")
        rf_delta = pd.DataFrame() if metric in {"unc_mean", "unc_comb"} else _best_rule_based_vs_rf(df, datasets=datasets, metric=metric)
        rf_delta_tables[metric] = rf_delta
        if not rf_delta.empty:
            rf_delta.to_csv(method_suite_dir / f"best_rule_based_vs_rf__{metric}.csv", index=False)
        metric_bar = results_dir / f"ALL_DATASETS_{metric}_methods_bar.png"
        metric_heat = results_dir / f"ALL_DATASETS_{metric}_methods_heatmap.png"
        _plot_grouped_bars(pvt, title=cfg["title"], out_path=metric_bar, lower_is_better=bool(cfg["lower_is_better"]))
        _plot_heatmap(pvt, title=cfg["title"], out_path=metric_heat, lower_is_better=bool(cfg["lower_is_better"]))
        _copy_file(metric_bar, plots_metric_dir / metric_bar.name)
        _copy_file(metric_heat, plots_metric_dir / metric_heat.name)

    hard_case_base = pd.DataFrame()
    hard_case_methods = pd.DataFrame()
    if args.include_hard_cases:
        hard_case_base, hard_case_methods = _load_hard_case_tables(hard_case_dir)
    _write_method_suite_report(df, rank_df, rf_delta_tables, hard_case_base, hard_case_methods, out_path=method_suite_dir / "REPORT.md")
    _write_dataset_plots(method_suite_dir / "method_suite_long.csv", plots_dataset_dir)
    print(f"[done] results -> {results_dir}")
    return 0


def _extract_option(args: list[str], *names: str) -> str:
    for idx, token in enumerate(args):
        if token in names and idx + 1 < len(args):
            return str(args[idx + 1]).strip()
        for name in names:
            prefix = f"{name}="
            if token.startswith(prefix):
                return str(token[len(prefix):]).strip()
    return ""


def _extract_csv_list(args: list[str], *names: str) -> list[str]:
    raw = _extract_option(args, *names)
    return [x.strip() for x in raw.split(",") if x.strip()]


def _resolve_pipeline_datasets(args: list[str]) -> list[str]:
    dataset_path = _extract_option(args, "--dataset-path")
    if dataset_path:
        return [Path(dataset_path).stem]
    dataset = _extract_option(args, "--dataset")
    if dataset:
        value = Path(dataset)
        return [value.stem if value.suffix.lower() == ".csv" else value.name]
    datasets = _extract_csv_list(args, "--datasets")
    out = []
    for item in datasets:
        value = Path(item)
        out.append(value.stem if value.suffix.lower() == ".csv" else value.name)
    return out or list(DEFAULT_DATASETS)


def _resolve_pipeline_inducers(args: list[str]) -> list[str]:
    inducers = [x.upper() for x in _extract_csv_list(args, "--inducers")]
    if "--include-static" in args and "STATIC" not in inducers:
        inducers.append("STATIC")
    return inducers or ["RIPPER", "FOIL"]


def _resolve_pipeline_seed(args: list[str]) -> int:
    return int(protocol_from_passthrough(args).seeds[0])


def _resolve_pipeline_test_size(args: list[str]) -> str:
    return str(protocol_from_passthrough(args).test_size)


def _resolve_out_root(args: list[str]) -> Path:
    raw = _extract_option(args, "--save-root", "--out-root")
    return Path(raw).expanduser().resolve() if raw else DEFAULT_OUT_ROOT.resolve()


def pipeline_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full benchmark/report pipeline.")
    parser.add_argument("--results-dir", default="", help="Where to write final tables/plots.")
    parser.add_argument("--hard-case-dir", default="", help="Where to write hard-case outputs.")
    parser.add_argument("--skip-hard-cases", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--skip-brief-report", action="store_true")
    args, passthrough = parser.parse_known_args(argv)

    out_root = _resolve_out_root(passthrough)
    results_dir = Path(args.results_dir).expanduser().resolve() if str(args.results_dir).strip() else DEFAULT_RESULTS_DIR.resolve()
    hard_case_dir = Path(args.hard_case_dir).expanduser().resolve() if str(args.hard_case_dir).strip() else DEFAULT_HARD_CASE_DIR.resolve()
    datasets = _resolve_pipeline_datasets(passthrough)
    inducers = _resolve_pipeline_inducers(passthrough)
    seed = _resolve_pipeline_seed(passthrough)
    test_size = _resolve_pipeline_test_size(passthrough)

    rc = benchmark_main(passthrough)
    if rc != 0:
        return rc
    if not args.skip_hard_cases:
        analyze_hard_cases.main(
            [
                "--datasets",
                ",".join(datasets),
                "--algos",
                ",".join(inducers),
                "--seed",
                str(seed),
                "--test-size",
                str(test_size),
                "--out-root",
                str(out_root),
                "--out-dir",
                str(hard_case_dir),
            ]
        )
    if not args.skip_report:
        rc = report_outputs_main(
            [
                "--out-root",
                str(out_root),
                "--results-dir",
                str(results_dir),
                "--hard-case-dir",
                str(hard_case_dir),
                "--datasets",
                ",".join(datasets),
                *(["--include-hard-cases"] if not args.skip_hard_cases else []),
            ]
        )
        if rc != 0:
            return rc
    if not args.skip_brief_report:
        rc = write_brief_report(metrics_path=results_dir / "ALL_DATASETS_metrics.csv", hard_cases_path=hard_case_dir / "HARD_CASE_ANALYSIS.md", out_path=results_dir / "brief_report.md")
        if rc != 0:
            return rc
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or [])
    if argv and argv[0] == "report":
        return report_outputs_main(argv[1:])
    if argv and argv[0] == "brief":
        results_dir = DEFAULT_RESULTS_DIR.resolve()
        return write_brief_report(
            metrics_path=results_dir / "ALL_DATASETS_metrics.csv",
            hard_cases_path=DEFAULT_HARD_CASE_DIR.resolve() / "HARD_CASE_ANALYSIS.md",
            out_path=results_dir / "brief_report.md",
        )
    return pipeline_main(argv)


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
