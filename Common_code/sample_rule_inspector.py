from __future__ import annotations

"""Web-first inspection helpers for a saved DSGD rule model.

The Streamlit app uses the helpers in this module to:
  - summarize how rules were generated and diversified,
  - inspect which rules fired for a sample,
  - verify that the combined explanation is consistent.

The CLI below is kept as a lightweight fallback for direct terminal debugging.
Modes:
  - inspect (default): print concise sample/rule/prediction summary
  - export-combined: save combined-rule payload to JSON + Markdown
  - conformance: run DS sanity checks + raw-invariance check
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np

THIS = Path(__file__).resolve()
COMMON = THIS.parent
ROOT = COMMON.parent

if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from Datasets_loader import load_dataset
from core import split_train_test, ds_combine_pair
from DSClassifierMultiQ import DSClassifierMultiQ, TrainConfig


def resolve_dataset(path_or_name: str) -> Path:
    path = Path(path_or_name)
    if path.suffix.lower() == ".csv" and path.is_file():
        return path.resolve()
    for base in (ROOT, COMMON, Path.cwd()):
        candidate = base / f"{path_or_name}.csv"
        if candidate.is_file():
            return candidate.resolve()
    return path.resolve()


def default_model_path(algo: str, dataset_name: str, out_root: Path) -> Path:
    return out_root / "models" / f"{algo.lower()}_{dataset_name}_dst.pkl"


def resolve_model_path(path: Path, *, algo: str, dataset_name: str, out_root: Path) -> Path:
    if path.suffix.lower() == ".pkl":
        return path.resolve()
    if path.suffix.lower() == ".dsb":
        mapped = out_root / "models" / f"{algo.lower()}_{dataset_name}_dst.pkl"
        if mapped.exists():
            return mapped.resolve()
    return path.resolve()


def load_view_split(
    csv_path: Path,
    *,
    split: str,
    test_size: float,
    seed: int,
) -> dict[str, Any]:
    X, y, feature_names, value_decoders, stats = load_dataset(csv_path, return_stats=True)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    idx_all = np.arange(len(X))

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = split_train_test(
        X,
        y,
        test_size=float(test_size),
        seed=int(seed),
        stratify=True,
    )
    if split == "train":
        X_view, y_view, idx_view = X_tr, y_tr, idx_tr
    elif split == "test":
        X_view, y_view, idx_view = X_te, y_te, idx_te
    else:
        X_view, y_view, idx_view = X, y, idx_all

    original_classes = stats.get("classes", [])
    idx_to_orig = {idx: label for idx, label in enumerate(original_classes)} if original_classes else {}
    return {
        "X": X,
        "y": y,
        "feature_names": list(feature_names),
        "value_decoders": value_decoders,
        "X_tr": X_tr,
        "X_te": X_te,
        "y_tr": y_tr,
        "y_te": y_te,
        "X_view": X_view,
        "y_view": y_view,
        "idx_view": np.asarray(idx_view, dtype=int),
        "idx_to_orig": idx_to_orig,
    }


def resolve_sample(
    *,
    idx: int,
    row_index: int | None,
    X_view: np.ndarray,
    y_view: np.ndarray,
    idx_view: np.ndarray,
) -> dict[str, Any]:
    if row_index is not None:
        hits = np.where(idx_view == int(row_index))[0]
        if len(hits) == 0:
            raise SystemExit(f"row-index {row_index} not found in current split")
        sample_pos = int(hits[0])
    else:
        sample_pos = int(idx) % len(X_view)
    return {
        "sample_pos": sample_pos,
        "sample": X_view[sample_pos],
        "true_label_idx": int(y_view[sample_pos]),
        "row_index": int(idx_view[sample_pos]),
    }


def _decode_value(name: str, value: float, decoders: dict[str, dict[int, str]]) -> str:
    mapping = decoders.get(name)
    if not mapping:
        return f"{value:.4f}" if isinstance(value, float) else str(value)
    if value in mapping:
        return str(mapping[value])
    try:
        iv = int(value)
        if iv in mapping:
            return str(mapping[iv])
    except Exception:
        pass
    return str(value)


def _shorten(text: str, limit: int = 220) -> str:
    s = str(text or "")
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def _top_items(items: list[dict], *, key: str = "weight", limit: int = 8) -> list[dict]:
    rows = sorted(list(items or []), key=lambda x: abs(float(x.get(key, 0.0))), reverse=True)
    if int(limit) <= 0:
        return rows
    return rows[: max(1, int(limit))]


def _sample_feature_rows(sample: np.ndarray, feature_names: list[str], decoders: dict[str, dict[int, str]], selected: list[str]) -> list[dict[str, Any]]:
    index = {name: i for i, name in enumerate(feature_names)}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in selected:
        if name in seen or name not in index:
            continue
        seen.add(name)
        raw = float(sample[index[name]])
        rows.append(
            {
                "feature": name,
                "raw": raw,
                "display": _decode_value(name, raw, decoders),
            }
        )
    return rows


def _inspection_base_dir(out_root: Path) -> Path:
    return out_root if out_root.name == "inspection" else out_root / "inspection"


def _summary_line(*, rules_pool: list[dict[str, Any]], proba: list[float], masses: list[float] | None) -> str:
    rule_rows = list(rules_pool)
    rule_part = " + ".join(f"R{int(row['rule_id'])}" for row in rule_rows) or "no_rules"
    p_part = "[" + ", ".join(f"{float(x):.4f}" for x in proba) + "]"
    if masses is None:
        return f"rules_pool: {rule_part} => p={p_part}"
    m_part = "[" + ", ".join(f'{float(x):.4f}' for x in masses) + "]"
    return f"rules_pool: {rule_part} => p={p_part}, m={m_part}"


def predict_sample(
    *,
    clf: DSClassifierMultiQ,
    sample: np.ndarray,
    combine_rule: str,
    merged_rule: bool = False,
    merged_rule_beta: float = 1.0,
) -> dict[str, Any]:
    k = int(clf.k)
    batch = sample.reshape(1, -1)
    comb = str(combine_rule).strip().lower()
    explain_rule = comb if comb in {"dempster", "yager", "vote"} else "dempster"
    if comb == "vote":
        proba = np.asarray(clf.raw_predict_proba(batch, method="weighted_vote"), dtype=float)
        pred = np.asarray(clf.raw_predict(batch, method="weighted_vote"), dtype=int)
        unc_rule = float("nan")
        unc_comb = float("nan")
        masses = None
    else:
        proba = np.asarray(clf.predict_proba(batch, combination_rule=comb), dtype=float)
        pred = np.asarray(clf.predict(batch, combination_rule=comb, use_tuned_threshold=(comb == "dempster")), dtype=int)
        unc = clf.model.uncertainty_stats(batch, combination_rule=comb)
        unc_rule = float(np.nanmean(np.asarray(unc.get("unc_rule", np.array([np.nan])), dtype=float)))
        unc_comb = float(np.nanmean(np.asarray(unc.get("unc_comb", np.array([np.nan])), dtype=float)))
        masses = np.asarray(clf.model.predict_masses(batch, combination_rule=comb)[0], dtype=float).tolist()

    export = clf.model.prepare_rules_for_export(sample)
    activated = list(export.get("activated_rules", []))
    combined = clf.model.get_combined_rule(
        sample,
        return_details=True,
        combination_rule=explain_rule,
        decision_class=int(pred[0]) if len(pred) else None,
        include_merged_rule=bool(merged_rule),
        merged_rule_beta=float(merged_rule_beta),
    )

    return {
        "proba": proba[0].tolist() if proba.size else [0.0] * k,
        "pred_label_idx": int(pred[0]) if len(pred) else -1,
        "unc_rule": None if not np.isfinite(unc_rule) else float(unc_rule),
        "unc_comb": None if not np.isfinite(unc_comb) else float(unc_comb),
        "masses": masses,
        "activated_rules": activated,
        "combined": combined if isinstance(combined, dict) else {},
    }


def build_combined_rule_payload(
    *,
    csv_path: Path,
    algo: str,
    model_path: Path,
    split: str,
    combine_rule: str,
    sample_info: dict[str, Any],
    pred_info: dict[str, Any],
    idx_to_orig: dict[int, Any],
    feature_names: list[str],
    value_decoders: dict[str, dict[int, str]],
) -> dict[str, Any]:
    true_idx = int(sample_info["true_label_idx"])
    pred_idx = int(pred_info["pred_label_idx"])
    combined = pred_info["combined"]
    rules_pool = list(combined.get("rules_pool", []))
    rule_ids = {int(row.get("rule_id", -1)) for row in rules_pool}
    activated_by_rule = {int(row.get("id", -1)): row for row in list(pred_info.get("activated_rules", []))}
    selected_features = sorted(
        {
            str(spec[0])
            for rid in rule_ids
            for spec in list((activated_by_rule.get(rid) or {}).get("specs") or [])
            if isinstance(spec, (list, tuple)) and len(spec) == 3
        }
    )
    sample_rows = _sample_feature_rows(np.asarray(sample_info["sample"], dtype=float), feature_names, value_decoders, selected_features)
    summary_line = _summary_line(
        rules_pool=rules_pool,
        proba=list(pred_info["proba"]),
        masses=pred_info["masses"],
    )
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": csv_path.stem,
        "algo": str(algo).upper(),
        "model_path": str(model_path),
        "split": split,
        "combine_rule": combine_rule,
        "sample": {
            "row_index": int(sample_info["row_index"]),
            "true_label_idx": true_idx,
            "true_label_original": idx_to_orig.get(true_idx, true_idx),
            "features_used": sample_rows,
        },
        "prediction": {
            "selected_class_idx": pred_idx,
            "selected_class_original": idx_to_orig.get(pred_idx, pred_idx),
            "proba": pred_info["proba"],
            "masses": pred_info["masses"],
            "unc_rule": pred_info["unc_rule"],
            "unc_comb": pred_info["unc_comb"],
        },
        "activation": {
            "n_rules_fired": int(len(pred_info["activated_rules"])),
        },
        "combined_rule": {
            "n_rules_fired": int(combined.get("n_rules_fired", len(pred_info["activated_rules"]))),
            "rules_pool_count": int(len(rules_pool)),
            "rules_pool": rules_pool,
            "combined_summary_literals": list(combined.get("combined_summary_literals", [])),
            "combined_summary": str(combined.get("combined_summary", "none")),
            "summary_line": summary_line,
        },
    }
    return payload


def build_manual_prediction_payload(
    *,
    clf: DSClassifierMultiQ,
    sample_values: list[Any],
    feature_names: list[str],
    value_decoders: dict[str, dict[int, str]],
    combine_rule: str,
    merged_rule: bool = False,
    merged_rule_beta: float = 1.0,
    dataset_name: str = "manual",
    algo: str = "RIPPER",
    model_path: str | Path = "",
) -> dict[str, Any]:
    sample = np.asarray(sample_values, dtype=np.float32)
    pred_info = predict_sample(
        clf=clf,
        sample=sample,
        combine_rule=combine_rule,
        merged_rule=merged_rule,
        merged_rule_beta=merged_rule_beta,
    )
    idx_to_orig = {i: str(i) for i in range(int(clf.k))}
    sample_rows = _sample_feature_rows(sample, feature_names, value_decoders, list(feature_names))
    combined = pred_info.get("combined") or {}
    rules_pool = list(combined.get("rules_pool", []))
    summary_line = _summary_line(rules_pool=rules_pool, proba=list(pred_info.get("proba", [])), masses=pred_info.get("masses"))
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_name),
        "algo": str(algo).upper(),
        "model_path": str(model_path),
        "split": "manual",
        "combine_rule": str(combine_rule),
        "sample": {
            "row_index": -1,
            "true_label_idx": -1,
            "true_label_original": "n/a",
            "features_used": sample_rows,
        },
        "prediction": {
            "selected_class_idx": int(pred_info["pred_label_idx"]),
            "selected_class_original": idx_to_orig.get(int(pred_info["pred_label_idx"]), int(pred_info["pred_label_idx"])),
            "proba": pred_info["proba"],
            "masses": pred_info["masses"],
            "unc_rule": pred_info["unc_rule"],
            "unc_comb": pred_info["unc_comb"],
            "activated_rules": list(pred_info["activated_rules"]),
        },
        "activation": {
            "n_rules_fired": int(len(pred_info["activated_rules"])),
        },
        "combined_rule": {
            "n_rules_fired": int(combined.get("n_rules_fired", len(pred_info["activated_rules"]))),
            "rules_pool_count": int(len(rules_pool)),
            "rules_pool": rules_pool,
            "combined_summary_literals": list(combined.get("combined_summary_literals", [])),
            "combined_summary": str(combined.get("combined_summary", "none")),
            "summary_line": summary_line,
        },
    }
    return payload


def summarize_rule_generation(clf: DSClassifierMultiQ, *, top_k: int = 12) -> dict[str, Any]:
    model = clf.model
    if model is None:
        return {}
    rules = list(model.rules or [])
    stage_counts: dict[str, int] = {}
    depth_counts: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    has_pool_shaping = False
    rows: list[dict[str, Any]] = []

    for rid, rule in enumerate(rules):
        stats = dict(rule.get("stats") or {})
        stage = str(stats.get("stage", "unknown"))
        depth_bin = str(stats.get("depth_bin", "unknown"))
        label = rule.get("label", None)
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        depth_counts[depth_bin] = depth_counts.get(depth_bin, 0) + 1
        class_key = "None" if label is None else str(int(label))
        class_counts[class_key] = class_counts.get(class_key, 0) + 1
        has_pool_shaping = has_pool_shaping or ("pool_score" in stats)
        rows.append(
            {
                "rule_id": rid,
                "class": label,
                "stage": stage,
                "depth_bin": depth_bin,
                "support": stats.get("support"),
                "precision": round(float(stats.get("precision", 0.0)), 6) if "precision" in stats else None,
                "recall": round(float(stats.get("recall", 0.0)), 6) if "recall" in stats else None,
                "f1": round(float(stats.get("f1", 0.0)), 6) if "f1" in stats else None,
                "literals": stats.get("literals"),
                "pool_score": round(float(stats.get("pool_score", 0.0)), 6) if "pool_score" in stats else None,
                "proposal_quality": round(float(stats.get("proposal_quality", 0.0)), 6) if "proposal_quality" in stats else None,
                "caption": rule.get("caption", ""),
            }
        )

    support_sorted = sorted(rows, key=lambda row: float(row.get("support") or 0.0), reverse=True)[: max(1, int(top_k))]
    precision_sorted = sorted(rows, key=lambda row: float(row.get("precision") or 0.0), reverse=True)[: max(1, int(top_k))]
    return {
        "algo": str(model.algo).upper(),
        "combination_rule": str(model.combination_rule),
        "n_rules": int(len(rules)),
        "n_classes": int(model.k),
        "n_features": int(len(model.feature_names or [])),
        "class_prior": None if model.class_prior is None else np.asarray(model.class_prior.detach().cpu().numpy(), dtype=float).round(6).tolist(),
        "stage_counts": stage_counts,
        "depth_counts": depth_counts,
        "class_counts": class_counts,
        "pool_shaping_used": bool(has_pool_shaping),
        "top_rules_by_support": support_sorted,
        "top_rules_by_precision": precision_sorted,
    }


def validate_web_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    prediction = dict(payload.get("prediction", {}))
    combined = dict(payload.get("combined_rule", {}))
    activation = dict(payload.get("activation", {}))
    proba = np.asarray(prediction.get("proba", []), dtype=float)
    checks = {
        "probabilities_sum_to_one": bool(proba.size == 0 or abs(float(proba.sum()) - 1.0) <= 1e-4),
        "fired_rule_count_matches": int(combined.get("n_rules_fired", activation.get("n_rules_fired", 0))) == int(activation.get("n_rules_fired", 0)),
        "combined_rule_present_when_rules_fired": bool(int(activation.get("n_rules_fired", 0)) == 0 or str(combined.get("combined_condition", "")).strip() or str(combined.get("combined_summary", "")).strip() != "none"),
        "summary_present": bool(str(combined.get("summary_line", "")).strip()),
    }
    checks["all_passed"] = bool(all(checks.values()))
    warnings = []
    if int(activation.get("n_rules_fired", 0)) == 0:
        warnings.append("No rules fired for this sample. This usually means the input is outside the learned rule support.")
    if not checks["combined_rule_present_when_rules_fired"]:
        warnings.append("A rule fired, but the combined explanation is empty. This should be inspected.")
    return {
        "checks": checks,
        "warnings": warnings,
    }


def build_web_inspection_payload(
    *,
    clf: DSClassifierMultiQ,
    sample_values: list[Any],
    feature_names: list[str],
    value_decoders: dict[str, dict[int, str]],
    combine_rule: str,
    merged_rule: bool = False,
    merged_rule_beta: float = 1.0,
    dataset_name: str = "manual",
    algo: str = "RIPPER",
    model_path: str | Path = "",
    top_k: int = 12,
) -> dict[str, Any]:
    prediction = build_manual_prediction_payload(
        clf=clf,
        sample_values=sample_values,
        feature_names=feature_names,
        value_decoders=value_decoders,
        combine_rule=combine_rule,
        merged_rule=merged_rule,
        merged_rule_beta=merged_rule_beta,
        dataset_name=dataset_name,
        algo=algo,
        model_path=model_path,
    )
    generation = summarize_rule_generation(clf, top_k=top_k)
    validation = validate_web_prediction(prediction)
    return {
        "generation": generation,
        "prediction": prediction,
        "validation": validation,
    }

def save_combined_rule_artifacts(payload: dict[str, Any], *, out_root: Path, top_k: int = 0) -> tuple[Path, Path]:
    dataset = str(payload["dataset"])
    algo = str(payload["algo"])
    row_index = int(payload["sample"]["row_index"])
    split = str(payload["split"])
    comb = str(payload["combine_rule"])
    out_dir = _inspection_base_dir(out_root) / dataset / algo
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sample_{row_index}_{split}_{comb}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    pred = payload["prediction"]
    rule = payload["combined_rule"]
    pool_rows = list(rule.get("rules_pool", []))
    if int(top_k) > 0:
        pool_rows = pool_rows[: max(1, int(top_k))]
    lines = [
        f"# Combined Rule Export: {dataset} / {algo}",
        "",
        f"- row_index: `{row_index}`",
        f"- split: `{split}`",
        f"- combine_rule: `{comb}`",
        f"- true_label: `{payload['sample']['true_label_original']}` (idx={payload['sample']['true_label_idx']})",
        f"- selected_class: `{pred['selected_class_original']}` (idx={pred['selected_class_idx']})",
        f"- n_rules_fired: `{payload['activation']['n_rules_fired']}`",
        f"- unc_rule: `{pred['unc_rule']}`",
        f"- unc_comb: `{pred['unc_comb']}`",
        "",
        "## Sample values used in explanation",
        "",
    ]
    for item in payload["sample"].get("features_used", []):
        lines.append(f"- `{item['feature']} = {item['display']}`")
    lines.extend(["", "## Rules pool", ""])
    if pool_rows:
        for item in pool_rows:
            mass_vector = np.round(np.asarray(item.get("mass_vector", []), dtype=float), 6).tolist()
            lines.append(
                f"- `R{int(item['rule_id'])} :: m={mass_vector} :: {_shorten(str(item.get('rule_text') or ''), 180)}`"
            )
    else:
        lines.append("- `none`")
    merged_rows = _top_items(list(rule.get("combined_summary_literals", [])), key="confidence", limit=top_k)
    lines.extend(["", "## Combined summary literals", ""])
    if merged_rows:
        for item in merged_rows:
            lines.append(f"- `{float(item.get('confidence', 0.0)):.3f} :: {item.get('expression', '')}`")
            lines.append("  contributors:")
            for contrib in item.get("contributors", []):
                lines.append(f"  - `{contrib}`")
        lines.extend(["", f"`{rule.get('combined_summary', 'none')}`", ""])
    else:
        lines.extend(["- `none`", ""])
    lines.extend(["", "## Rule summary", "", f"`{rule['summary_line']}`"])
    lines.extend(["", "## Probabilities", "", f"`{np.round(np.asarray(pred['proba'], dtype=float), 6).tolist()}`"])
    if pred.get("masses") is not None:
        lines.extend(["", "## Fused mass", "", f"`{np.round(np.asarray(pred['masses'], dtype=float), 6).tolist()}`"])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def _toy_ds_combine_checks() -> dict[str, Any]:
    eps = 1e-6
    m1 = np.array([[0.62, 0.18, 0.20]], dtype=np.float32)
    m2 = np.array([[0.25, 0.55, 0.20]], dtype=np.float32)
    vac = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    out_12 = np.asarray(ds_combine_pair(m1, m2), dtype=np.float64).reshape(-1)
    out_21 = np.asarray(ds_combine_pair(m2, m1), dtype=np.float64).reshape(-1)
    out_v = np.asarray(ds_combine_pair(m1, vac), dtype=np.float64).reshape(-1)
    checks = {
        "sum_to_one": bool(abs(float(out_12.sum()) - 1.0) <= eps),
        "non_negative": bool(np.all(out_12 >= -eps)),
        "commutative": bool(float(np.max(np.abs(out_12 - out_21))) <= eps),
        "vacuous_identity": bool(float(np.max(np.abs(out_v - m1.reshape(-1)))) <= eps),
    }
    checks["all_passed"] = bool(all(checks.values()))
    checks["example_out"] = out_12.tolist()
    return checks


def _run_raw_invariance_check(
    *,
    k: int,
    algo: str,
    model_path: Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    audit_cfg = TrainConfig(
        max_epochs=25,
        early_stop_patience=6,
        batch_size=512,
        lr=1e-3,
        verbose=False,
    )
    clf = DSClassifierMultiQ(k=k, rule_algo=algo, device="cpu", train_cfg=audit_cfg)
    clf.load_model(str(model_path))
    p_before = np.asarray(clf.raw_predict_proba(X_eval, method="weighted_vote"), dtype=np.float64)
    clf.fit(X_train, y_train)
    p_after = np.asarray(clf.raw_predict_proba(X_eval, method="weighted_vote"), dtype=np.float64)
    delta = float(np.max(np.abs(p_before - p_after))) if p_before.size else 0.0
    return delta, dict(getattr(clf, "_last_fit_meta", {}))


def run_conformance(
    *,
    out_root: Path,
    csv_path: Path,
    algo: str,
    model_path: Path,
    split_bundle: dict[str, Any],
) -> Path:
    checks = _toy_ds_combine_checks()
    X_tr = np.asarray(split_bundle["X_tr"], dtype=np.float32)
    y_tr = np.asarray(split_bundle["y_tr"], dtype=int)
    X_te = np.asarray(split_bundle["X_te"], dtype=np.float32)
    y = np.asarray(split_bundle["y"], dtype=int)
    k = int(np.unique(y).size)
    raw_delta, fit_meta = _run_raw_invariance_check(
        k=k,
        algo=algo,
        model_path=model_path,
        X_train=X_tr,
        y_train=y_tr,
        X_eval=X_te,
    )
    out_dir = _inspection_base_dir(out_root) / csv_path.stem / str(algo).upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "conformance.md"
    lines = [
        f"# Conformance: {csv_path.stem} / {algo}",
        "",
        f"- timestamp_utc: `{datetime.now(timezone.utc).isoformat()}`",
        f"- model_path: `{model_path}`",
        f"- ds_sum_to_one: `{checks['sum_to_one']}`",
        f"- ds_non_negative: `{checks['non_negative']}`",
        f"- ds_commutative: `{checks['commutative']}`",
        f"- ds_vacuous_identity: `{checks['vacuous_identity']}`",
        f"- ds_all_passed: `{checks['all_passed']}`",
        f"- raw_invariance_max_abs_delta: `{raw_delta:.3e}`",
        f"- raw_invariance_pass: `{raw_delta <= 1e-10}`",
        "",
        "## Fit meta",
        "",
        f"`{json.dumps(fit_meta, indent=2, ensure_ascii=False)}`",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser("Inspect/export a single sample with combined-rule payload")
    parser.add_argument("--dataset", required=True, help="Dataset name or path")
    parser.add_argument("--algo", default="RIPPER", choices=["STATIC", "RIPPER", "FOIL"])
    parser.add_argument("--idx", type=int, default=0, help="Index inside selected split")
    parser.add_argument("--row-index", type=int, default=None, help="Original dataset row index inside selected split")
    parser.add_argument("--split", choices=["full", "train", "test"], default="test")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--combine-rule", default="dempster", choices=["dempster", "yager", "vote"])
    parser.add_argument("--mode", default="inspect", choices=["inspect", "export-combined", "conformance"])
    parser.add_argument("--out-root", default="artifacts")
    parser.add_argument("--model", default="", help="Optional explicit .pkl model path")
    parser.add_argument("--top-k", type=int, default=0, help="How many literals/rule contributions to print/export; 0 means all")
    parser.add_argument("--merged-rule", action="store_true", help="Include optional post-hoc merged-rule explanation block")
    parser.add_argument("--merged-rule-beta", type=float, default=1.0, help="Softmax sharpness for merged-rule certainty importance")
    args = parser.parse_args()

    out_root = Path(args.out_root).expanduser().resolve()
    csv_path = resolve_dataset(args.dataset)
    if not csv_path.exists():
        raise SystemExit(f"dataset not found: {csv_path}")

    split_bundle = load_view_split(
        csv_path,
        split=str(args.split),
        test_size=float(args.test_size),
        seed=int(args.seed),
    )
    sample_info = resolve_sample(
        idx=int(args.idx),
        row_index=args.row_index,
        X_view=split_bundle["X_view"],
        y_view=split_bundle["y_view"],
        idx_view=split_bundle["idx_view"],
    )

    if str(args.model).strip():
        model_path = resolve_model_path(
            Path(args.model).expanduser(),
            algo=str(args.algo),
            dataset_name=csv_path.stem,
            out_root=out_root,
        )
    else:
        model_path = default_model_path(str(args.algo), csv_path.stem, out_root)
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")

    if args.mode == "conformance":
        out_path = run_conformance(
            out_root=out_root,
            csv_path=csv_path,
            algo=str(args.algo),
            model_path=model_path,
            split_bundle=split_bundle,
        )
        print(f"[done] conformance report -> {out_path}")
        return

    clf = DSClassifierMultiQ(k=int(np.unique(split_bundle["y"]).size), rule_algo=str(args.algo), device="cpu")
    clf.load_model(str(model_path))
    pred_info = predict_sample(
        clf=clf,
        sample=np.asarray(sample_info["sample"], dtype=np.float32),
        combine_rule=str(args.combine_rule),
        merged_rule=bool(args.merged_rule),
        merged_rule_beta=float(args.merged_rule_beta),
    )
    payload = build_combined_rule_payload(
        csv_path=csv_path,
        algo=str(args.algo),
        model_path=model_path,
        split=str(args.split),
        combine_rule=str(args.combine_rule),
        sample_info=sample_info,
        pred_info=pred_info,
        idx_to_orig=split_bundle["idx_to_orig"],
        feature_names=list(split_bundle["feature_names"]),
        value_decoders=dict(split_bundle["value_decoders"]),
    )

    if args.mode == "export-combined":
        json_path, md_path = save_combined_rule_artifacts(payload, out_root=out_root, top_k=int(args.top_k))
        print(f"[done] combined payload -> {json_path}")
        print(f"[done] combined summary -> {md_path}")
        return

    # inspect mode
    true_orig = payload["sample"]["true_label_original"]
    pred_orig = payload["prediction"]["selected_class_original"]
    print(f"dataset={payload['dataset']} algo={payload['algo']} split={payload['split']} row={payload['sample']['row_index']}")
    print(f"true={true_orig} selected_class={pred_orig} combine_rule={payload['combine_rule']}")
    print(f"n_rules_fired={payload['activation']['n_rules_fired']} unc_comb={payload['prediction']['unc_comb']}")
    limit = int(args.top_k)
    sample_rows = payload["sample"].get("features_used", [])
    rule_rows = payload["combined_rule"].get("rules_pool", [])
    if limit > 0:
        sample_rows = sample_rows[:limit]
        rule_rows = rule_rows[:limit]
    print("sample_values:")
    for item in sample_rows:
        print(f"  - {item['feature']} = {item['display']}")
    print("rules_pool:")
    for item in rule_rows:
        mass_vector = np.round(np.asarray(item.get("mass_vector", []), dtype=float), 6).tolist()
        print(f"  - R{int(item['rule_id'])} :: m={mass_vector} :: {_shorten(str(item.get('rule_text') or ''), 160)}")
    merged_rows = _top_items(list(payload["combined_rule"].get("combined_summary_literals", [])), key="confidence", limit=limit)
    if merged_rows:
        print("combined_summary_literals:")
        for item in merged_rows:
            print(f"  - {float(item.get('confidence', 0.0)):.3f} :: {item.get('expression', '')}")
            for contrib in item.get("contributors", []):
                print(f"      * {contrib}")
        print(f"  summary: {payload['combined_rule'].get('combined_summary', 'none')}")
    print(f"rule_summary: {payload['combined_rule']['summary_line']}")


if __name__ == "__main__":
    main()
