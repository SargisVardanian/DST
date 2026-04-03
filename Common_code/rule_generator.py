from __future__ import annotations

import itertools
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

try:
    import wittgenstein as lw
    from wittgenstein.discretize import BinTransformer
except Exception:  # pragma: no cover - import failure handled at runtime
    lw = None
    BinTransformer = None

Literal = Tuple[str, str, Any]
Condition = Tuple[Literal, ...]

_ALLOWED_OPS = {"==", "<", ">"}
_FLOAT_FMT = "{:.6g}"
_EPS = 1e-9


def _normalise_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isfinite(value) and abs(value - round(value)) <= _EPS:
            return float(round(value))
        return float(value)
    if isinstance(value, int):
        return int(value)
    return value


def _values_equal(left: Any, right: Any) -> bool:
    left = _normalise_value(left)
    right = _normalise_value(right)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= _EPS
    return left == right


def _canonicalize_condition(literals: Sequence[Literal]) -> Condition | None:
    if not literals:
        return ()

    bounds: Dict[str, Dict[str, Any]] = {}
    for feature, op, raw_value in literals:
        if op not in _ALLOWED_OPS:
            raise ValueError(f"Unsupported op {op!r}")
        value = _normalise_value(raw_value)
        state = bounds.setdefault(feature, {"eq": None, "gt": None, "lt": None})
        if op == "==":
            if state["eq"] is not None and not _values_equal(state["eq"], value):
                return None
            state["eq"] = value
        elif op == ">":
            state["gt"] = float(value) if state["gt"] is None else max(state["gt"], float(value))
        else:
            state["lt"] = float(value) if state["lt"] is None else min(state["lt"], float(value))

    merged: List[Literal] = []
    for feature in sorted(bounds):
        eq, gt, lt = bounds[feature]["eq"], bounds[feature]["gt"], bounds[feature]["lt"]
        if eq is not None:
            eq_value = float(eq) if isinstance(eq, (int, float)) else eq
            if gt is not None and eq_value <= gt + _EPS:
                return None
            if lt is not None and eq_value >= lt - _EPS:
                return None
            merged.append((feature, "==", eq))
            continue
        if gt is not None and lt is not None and gt >= lt - _EPS:
            return None
        if gt is not None:
            merged.append((feature, ">", gt))
        if lt is not None:
            merged.append((feature, "<", lt))

    op_order = {"==": 0, ">": 1, "<": 2}
    return tuple(sorted(merged, key=lambda lit: (lit[0], op_order[lit[1]], repr(_normalise_value(lit[2])))))


def _condition_signature(cond: Condition) -> str:
    parts = []
    for name, op, value in cond:
        value = _normalise_value(value)
        parts.append(f"{name}{op}{_FLOAT_FMT.format(value) if isinstance(value, float) else value}")
    return "&&".join(parts)


def _resolve_value_name(name: str, value: Any, mapping: Dict[str, Dict[Any, Any]]) -> Any:
    decoder = mapping.get(name, {})
    if value in decoder:
        return decoder[value]
    text_value = str(value)
    if text_value in decoder:
        return decoder[text_value]
    if isinstance(value, (int, float)) and float(value).is_integer():
        int_value = int(value)
        return decoder.get(int_value, decoder.get(str(int_value), value))
    return value


def _format_literal_value(value: Any) -> str:
    value = _normalise_value(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return _FLOAT_FMT.format(float(value))
    if isinstance(value, (str, np.str_)):
        return json.dumps(str(value), ensure_ascii=False)
    return json.dumps(str(value), ensure_ascii=False) if not isinstance(value, bool) else str(value)


def _condition_caption(cond: Sequence[Literal], value_names: Dict[str, Dict[Any, Any]], label: int | None = None) -> str:
    parts = []
    for name, op, value in cond:
        display = _resolve_value_name(name, value, value_names)
        parts.append(f"{name} {op} {_format_literal_value(display)}")
    caption = " & ".join(parts) if parts else "<empty>"
    return f"{caption} -> class {label}" if label is not None else caption


def _mask_from_literal(col: np.ndarray, op: str, value: Any) -> np.ndarray:
    if op == "==":
        if np.issubdtype(col.dtype, np.number):
            return np.isfinite(col) & np.isclose(col, float(value), atol=_EPS)
        return col == value
    if op == "<":
        return np.isfinite(col) & (col < float(value))
    return np.isfinite(col) & (col > float(value))


def _condition_mask(X: np.ndarray, feat_to_idx: Dict[str, int], cond: Sequence[Literal]) -> np.ndarray:
    mask = np.ones(X.shape[0], dtype=bool)
    for name, op, value in cond:
        idx = feat_to_idx.get(name)
        if idx is None:
            return np.zeros(X.shape[0], dtype=bool)
        mask &= _mask_from_literal(X[:, idx], op, value)
        if not mask.any():
            break
    return mask


def _decoded_dataframe(
    X: np.ndarray,
    names: Sequence[str],
    decoders: Dict[str, Dict[Any, Any]],
    is_cat: Sequence[bool],
) -> pd.DataFrame:
    cols: Dict[str, Any] = {}
    for idx, name in enumerate(names):
        col = np.asarray(X[:, idx])
        if is_cat[idx] and name in decoders:
            values = []
            for value in col:
                if isinstance(value, np.generic):
                    value = value.item()
                display = _resolve_value_name(name, value, decoders)
                values.append(str(display))
            cols[name] = values
        else:
            cols[name] = col.astype(float)
    return pd.DataFrame(cols, columns=list(names))


def _reverse_value_decoders(decoders: Dict[str, Dict[Any, Any]]) -> Dict[str, Dict[Any, Any]]:
    reverse: Dict[str, Dict[Any, Any]] = {}
    for name, mapping in (decoders or {}).items():
        feature_reverse: Dict[Any, Any] = {}
        for encoded, display in mapping.items():
            encoded_norm = _normalise_value(encoded)
            feature_reverse[display] = encoded_norm
            feature_reverse[str(display)] = encoded_norm
            feature_reverse[encoded_norm] = encoded_norm
            feature_reverse[str(encoded_norm)] = encoded_norm
        reverse[name] = feature_reverse
    return reverse


def _upper_bound_to_open(value: float) -> float:
    return float(np.nextafter(float(value), float("inf")))


def _backend_cond_to_literals(
    feature: str,
    value: Any,
    *,
    numeric_binned_features: set[str],
    reverse_value_maps: Dict[str, Dict[Any, Any]],
) -> Condition:
    if feature in numeric_binned_features and isinstance(value, str):
        text = value.strip()
        if text.startswith("<"):
            upper = float(text[1:].strip())
            return ((feature, "<", _upper_bound_to_open(upper)),)
        if text.startswith(">"):
            lower = float(text[1:].strip())
            return ((feature, ">", lower),)
        if " - " in text:
            lower_s, upper_s = text.split(" - ", 1)
            lower = float(lower_s.strip())
            upper = float(upper_s.strip())
            return (
                (feature, ">", lower),
                (feature, "<", _upper_bound_to_open(upper)),
            )
    encoded = reverse_value_maps.get(feature, {}).get(value)
    if encoded is None:
        encoded = reverse_value_maps.get(feature, {}).get(str(value), _normalise_value(value))
    return ((feature, "==", encoded),)


def _feature_score(col: np.ndarray) -> float:
    if col.size == 0:
        return 0.0
    if np.issubdtype(col.dtype, np.number):
        finite = col[np.isfinite(col)]
        return float(np.var(finite)) if finite.size else 0.0
    _, counts = np.unique(col, return_counts=True)
    return 0.0 if counts.size == 0 else float(1.0 - (counts / counts.sum()).max())


def _split_feature_kinds(
    X: np.ndarray, names: Sequence[str], decoders: Dict[str, Dict[Any, Any]], *, cat_threshold: int = 32
) -> tuple[list[int], list[int]]:
    numeric, categorical = [], []
    for idx, name in enumerate(names):
        col = X[:, idx]
        if (decoders and name in decoders) or not np.issubdtype(col.dtype, np.number):
            categorical.append(idx)
            continue
        finite = col[np.isfinite(col)]
        unique = np.unique(finite)
        is_discrete = unique.size <= cat_threshold and np.all(np.isclose(unique, np.round(unique)))
        (categorical if is_discrete else numeric).append(idx)
    return numeric, categorical


def _categorical_flags(X: np.ndarray, names: Sequence[str], decoders: Dict[str, Dict[Any, Any]], *, cat_threshold: int = 20) -> list[bool]:
    _, categorical = _split_feature_kinds(X, names, decoders, cat_threshold=cat_threshold)
    flags = [False] * X.shape[1]
    for idx in categorical:
        flags[idx] = True
    return flags


def _bump_failures(failures: int, limit: int) -> tuple[int, bool]:
    failures += 1
    return failures, failures >= limit


def _foil_gain(pos_mask: np.ndarray, neg_mask: np.ndarray, curr_mask: np.ndarray, literal_mask: np.ndarray) -> float:
    p0 = int((curr_mask & pos_mask).sum())
    n0 = int((curr_mask & neg_mask).sum())
    new_mask = curr_mask & literal_mask
    p1 = int((new_mask & pos_mask).sum())
    if p1 < 1:
        return -math.inf
    n1 = int((new_mask & neg_mask).sum())

    def _lp(p: int, n: int) -> float:
        return math.log((p + 1e-9) / (p + n + 1e-9))

    return p1 * (_lp(p1, n1) - _lp(p0, n0))


def _candidate_literals(
    X: np.ndarray,
    is_cat: Sequence[bool],
    config: "GeneratorConfig",
    feat_idx: int,
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
) -> list[tuple[str, Any]]:
    col = X[:, feat_idx]
    if is_cat[feat_idx]:
        if np.issubdtype(col.dtype, np.number):
            values = np.unique(col[np.isfinite(col)])
        else:
            values = np.unique(col)
        return [("==", _normalise_value(v)) for v in values]

    finite = col[np.isfinite(col)]
    if finite.size == 0:
        return []
    pos_vals = col[pos_mask & np.isfinite(col)]
    neg_vals = col[neg_mask & np.isfinite(col)]
    chunks = [np.quantile(finite, config.quantiles)]
    if pos_vals.size:
        chunks.append(np.quantile(pos_vals, config.quantiles))
    if neg_vals.size:
        chunks.append(np.quantile(neg_vals, config.quantiles))
    if config.use_class_midpoints and pos_vals.size and neg_vals.size:
        pos_q = np.quantile(pos_vals, [0.25, 0.50, 0.75])
        neg_q = np.quantile(neg_vals, [0.25, 0.50, 0.75])
        midpoints = np.asarray(
            [
                0.5 * (float(pos_q[1]) + float(neg_q[1])),
                0.5 * (float(pos_q[0]) + float(neg_q[2])),
                0.5 * (float(pos_q[2]) + float(neg_q[0])),
                0.5 * (float(np.mean(pos_vals)) + float(np.mean(neg_vals))),
            ],
            dtype=float,
        )
        chunks.append(midpoints[np.isfinite(midpoints)])
    thresholds = np.unique(np.concatenate(chunks))
    thresholds = thresholds[np.isfinite(thresholds)]
    return [(op, float(t)) for t in thresholds for op in (">", "<")]


def _foil_boundary_thresholds(col: np.ndarray, pos_mask: np.ndarray, neg_mask: np.ndarray) -> np.ndarray:
    valid = np.isfinite(col) & (pos_mask | neg_mask)
    values = col[valid]
    if values.size < 2:
        return np.empty(0, dtype=float)

    labels = pos_mask[valid].astype(np.int8)
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    labels = labels[order]

    thresholds: list[float] = []
    last_added: float | None = None
    for i in range(values.size - 1):
        left = float(values[i])
        right = float(values[i + 1])
        if abs(left - right) <= _EPS:
            continue
        if labels[i] == labels[i + 1]:
            continue
        thr = 0.5 * (left + right)
        if last_added is None or abs(thr - last_added) > 1e-9:
            thresholds.append(thr)
            last_added = thr
    return np.asarray(thresholds, dtype=float)


def _foil_candidate_literals(
    X: np.ndarray,
    is_cat: Sequence[bool],
    config: "GeneratorConfig",
    feat_idx: int,
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
    *,
    max_numeric_thresholds: int | None = None,
) -> list[tuple[str, Any]]:
    col = X[:, feat_idx]
    if is_cat[feat_idx]:
        if np.issubdtype(col.dtype, np.number):
            values = np.unique(col[np.isfinite(col)])
        else:
            values = np.unique(col)
        return [("==", _normalise_value(v)) for v in values]

    thresholds = _foil_boundary_thresholds(col, pos_mask, neg_mask)
    if thresholds.size == 0:
        return []
    if max_numeric_thresholds is not None and int(max_numeric_thresholds) > 0 and thresholds.size > int(max_numeric_thresholds):
        idxs = np.linspace(0, thresholds.size - 1, int(max_numeric_thresholds))
        idxs = np.unique(np.clip(np.round(idxs).astype(int), 0, thresholds.size - 1))
        thresholds = thresholds[idxs]
    return [(op, float(t)) for t in thresholds for op in (">", "<")]


def _build_foil_literal_cache(
    X: np.ndarray,
    feat_names: Sequence[str],
    is_cat: Sequence[bool],
    config: "GeneratorConfig",
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
) -> list[tuple[Literal, np.ndarray]]:
    cache: list[tuple[Literal, np.ndarray]] = []
    numeric_threshold_cap = max(8, min(24, int(round(96.0 / math.sqrt(max(1.0, float(X.shape[1])))))))
    for feat_idx, feat_name in enumerate(feat_names):
        for op, thr in _foil_candidate_literals(
            X,
            is_cat,
            config,
            feat_idx,
            pos_mask,
            neg_mask,
            max_numeric_thresholds=None if is_cat[feat_idx] else numeric_threshold_cap,
        ):
            literal = (feat_name, op, thr)
            trial = _canonicalize_condition((literal,))
            if not trial:
                continue
            literal_mask = _mask_from_literal(X[:, feat_idx], op, thr)
            if not literal_mask.any():
                continue
            # Literals that never cover the target positives only add search cost and noise.
            if int((literal_mask & pos_mask).sum()) < max(1, int(config.min_pos_support)):
                continue
            cache.append((literal, literal_mask))
    return cache


def _foil_gain_from_counts(p0: int, n0: int, p1: np.ndarray, n1: np.ndarray) -> np.ndarray:
    p1 = np.asarray(p1, dtype=float)
    n1 = np.asarray(n1, dtype=float)
    out = np.full_like(p1, -math.inf, dtype=float)
    valid = p1 >= 1.0
    if not valid.any():
        return out
    base = math.log((p0 + 1e-9) / (p0 + n0 + 1e-9))
    out[valid] = p1[valid] * (
        np.log((p1[valid] + 1e-9) / (p1[valid] + n1[valid] + 1e-9)) - base
    )
    return out


def _grow_rule(
    X: np.ndarray,
    feat_names: Sequence[str],
    is_cat: Sequence[bool],
    config: "GeneratorConfig",
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
    *,
    max_literals: int,
    min_literals: int,
    min_pos_support: int,
    min_gain: float,
    candidate_fn=_candidate_literals,
) -> tuple[Condition, np.ndarray] | None:
    cond_seq: list[Literal] = []
    curr_mask = np.ones_like(pos_mask, dtype=bool)
    min_literals = max(1, int(min_literals))
    max_literals = max(1, int(max_literals))

    while len(cond_seq) < max_literals and (len(cond_seq) < min_literals or (curr_mask & neg_mask).any()):
        best_gain = -math.inf
        best_literal = None
        best_new_mask = None
        current_sig = _condition_signature(_canonicalize_condition(cond_seq) or ())

        for feat_idx, feat_name in enumerate(feat_names):
            for op, thr in candidate_fn(X, is_cat, config, feat_idx, pos_mask, neg_mask):
                trial = _canonicalize_condition(tuple(cond_seq) + ((feat_name, op, thr),))
                if not trial or _condition_signature(trial) == current_sig:
                    continue
                literal_mask = _mask_from_literal(X[:, feat_idx], op, thr)
                new_mask = curr_mask & literal_mask
                if int((new_mask & pos_mask).sum()) < min_pos_support:
                    continue
                gain = _foil_gain(pos_mask, neg_mask, curr_mask, literal_mask)
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_literal = (feat_name, op, thr)
                    best_new_mask = new_mask

        if best_literal is None or best_gain < min_gain:
            break
        cond_seq.append(best_literal)
        curr_mask = best_new_mask
    return (tuple(cond_seq), curr_mask) if cond_seq else None


def _grow_rule_cached(
    literal_cache: Sequence[tuple[Literal, np.ndarray]],
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
    *,
    max_literals: int,
    min_literals: int,
    min_pos_support: int,
    min_gain: float,
    batch_size: int = 256,
) -> tuple[Condition, np.ndarray] | None:
    cond_seq: list[Literal] = []
    curr_mask = np.ones_like(pos_mask, dtype=bool)
    min_literals = max(1, int(min_literals))
    max_literals = max(1, int(max_literals))

    while len(cond_seq) < max_literals and (len(cond_seq) < min_literals or (curr_mask & neg_mask).any()):
        base_pos = curr_mask & pos_mask
        base_neg = curr_mask & neg_mask
        p0 = int(base_pos.sum())
        n0 = int(base_neg.sum())
        current_sig = _condition_signature(_canonicalize_condition(cond_seq) or ())

        best_gain = -math.inf
        best_literal: Literal | None = None
        best_new_mask: np.ndarray | None = None

        for start in range(0, len(literal_cache), batch_size):
            chunk = literal_cache[start:start + batch_size]
            masks = np.stack([mask for _, mask in chunk], axis=0)
            p1 = np.count_nonzero(masks & base_pos, axis=1)
            valid = p1 >= int(min_pos_support)
            if not valid.any():
                continue
            n1 = np.count_nonzero(masks & base_neg, axis=1)
            gains = _foil_gain_from_counts(p0, n0, p1, n1)

            for local_idx, ((literal, literal_mask), gain, p1_i) in enumerate(zip(chunk, gains, p1)):
                if p1_i < int(min_pos_support):
                    continue
                trial = _canonicalize_condition(tuple(cond_seq) + (literal,))
                if not trial or _condition_signature(trial) == current_sig:
                    continue
                if gain > best_gain + 1e-12:
                    best_gain = float(gain)
                    best_literal = literal
                    best_new_mask = curr_mask & literal_mask

        if best_literal is None or best_new_mask is None or best_gain < float(min_gain):
            break
        cond_seq.append(best_literal)
        curr_mask = best_new_mask
    return (tuple(cond_seq), curr_mask) if cond_seq else None


def _foil_rule_stats(
    X: np.ndarray,
    y: np.ndarray,
    feat_to_idx: Dict[str, int],
    label: int,
    cond: Condition,
    *,
    min_pos_support: int,
    min_precision: float,
    stage: str = "foil",
) -> tuple[np.ndarray | None, Dict[str, Any] | None]:
    rule_mask = _condition_mask(X, feat_to_idx, cond)
    pos = y == label
    neg = ~pos
    support = int((rule_mask & pos).sum())
    neg_covered = int((rule_mask & neg).sum())
    if support < max(1, int(min_pos_support)):
        return None, None
    precision = support / (support + neg_covered + 1e-12)
    if precision < float(min_precision):
        return None, None
    recall = support / max(1, int(pos.sum()))
    stats = {
        "kind": "learned",
        "stage": stage,
        "label": int(label),
        "support": support,
        "neg_covered": neg_covered,
        "precision": precision,
        "recall": recall,
        "f1": (2.0 * precision * recall) / (precision + recall + 1e-12),
        "coverage": float(rule_mask.mean()),
        "literals": len(cond),
        "min_precision": float(min_precision),
    }
    return rule_mask, stats


def _fit_foil_class_worker(
    X: np.ndarray,
    y: np.ndarray,
    feat_names: Sequence[str],
    is_cat: Sequence[bool],
    config: "GeneratorConfig",
    cls: int,
    *,
    verbose: bool = False,
) -> list[tuple[int, Condition, Dict[str, Any]]]:
    feat_to_idx = {name: i for i, name in enumerate(feat_names)}
    pos_all = y == cls
    neg_all = y != cls
    pos_remaining = pos_all.copy()
    literal_cache = _build_foil_literal_cache(X, feat_names, is_cat, config, pos_all, neg_all)
    effective = _effective_foil_config(
        config,
        pos_count=int(pos_all.sum()),
        neg_count=int(neg_all.sum()),
        literal_pool_size=len(literal_cache),
        n_features=int(X.shape[1]),
    )
    if verbose:
        print(
            f"[rulegen/FOIL] class={cls} start positives={int(pos_all.sum())} "
            f"literal_pool={len(literal_cache)} budget="
            f"(max_rules={int(effective['max_rules_per_class'])}, max_literals={int(effective['max_literals'])}, "
            f"min_precision={float(effective['min_precision']):.3f}, min_gain={float(effective['min_gain']):.2e})",
            flush=True,
        )
    ordered: list[tuple[int, Condition, Dict[str, Any]]] = []
    seen_signatures: set[str] = set()

    while pos_remaining.any() and len(ordered) < int(effective["max_rules_per_class"]):
        grown = _grow_rule_cached(
            literal_cache,
            pos_remaining,
            neg_all,
            max_literals=max(1, int(effective["max_literals"])),
            min_literals=max(1, int(config.min_literals)),
            min_pos_support=max(1, int(config.min_pos_support)),
            min_gain=float(effective["min_gain"]),
        )
        if grown is None:
            break
        cond = _canonicalize_condition(grown[0])
        if not cond:
            break
        signature = _condition_signature(cond)
        if signature in seen_signatures:
            break
        rule_mask, stats = _foil_rule_stats(
            X,
            y,
            feat_to_idx,
            cls,
            cond,
            min_pos_support=int(config.min_pos_support),
            min_precision=float(effective["min_precision"]),
        )
        if rule_mask is None or stats is None:
            break
        covered_pos = rule_mask & pos_remaining
        if not covered_pos.any():
            break
        stats["runtime_max_rules_per_class"] = int(effective["max_rules_per_class"])
        stats["runtime_max_literals"] = int(effective["max_literals"])
        stats["runtime_min_precision"] = float(effective["min_precision"])
        stats["runtime_min_gain"] = float(effective["min_gain"])
        stats["literal_pool_size"] = int(len(literal_cache))
        ordered.append((int(cls), cond, stats))
        seen_signatures.add(signature)
        pos_remaining &= ~covered_pos
        if verbose and (len(ordered) == 1 or len(ordered) % 10 == 0 or not pos_remaining.any()):
            print(
                f"[rulegen/FOIL] class={cls} accepted={len(ordered)} "
                f"remaining_pos={int(pos_remaining.sum())} support={int(stats['support'])} "
                f"precision={float(stats['precision']):.3f} literals={int(stats['literals'])}",
                flush=True,
            )
    return ordered


def _effective_foil_config(
    config: "GeneratorConfig",
    *,
    pos_count: int,
    neg_count: int,
    literal_pool_size: int,
    n_features: int,
) -> dict[str, float | int]:
    """Derive a flexible per-class FOIL search budget from the observed class/pool size.

    The base config stays conservative and readable. Expansion happens only when the
    candidate literal pool is genuinely large or the positive class is relatively rare,
    which is exactly where DSGD can benefit from a broader but still controlled rule set.
    """
    pos_count = max(0, int(pos_count))
    neg_count = max(0, int(neg_count))
    total = max(1, pos_count + neg_count)
    prevalence = float(pos_count / total)
    rarity = 1.0 - prevalence
    pool_pressure = min(1.0, float(literal_pool_size) / max(24.0, 3.0 * float(max(1, n_features))))

    base_rules = int(config.max_rules_per_class)
    extra_rules = int(round(base_rules * (0.45 * pool_pressure + 0.35 * max(0.0, rarity - 0.5))))
    max_rules = min(int(config.max_total_rules), max(base_rules, base_rules + max(0, extra_rules)))

    base_precision = float(config.min_precision)
    precision_relief = 0.12 * pool_pressure + 0.08 * max(0.0, rarity - 0.5)
    min_precision = float(np.clip(base_precision - precision_relief, 0.28, base_precision))

    base_literals = int(config.max_literals)
    extra_literals = 1 if pool_pressure >= 0.55 or (rarity >= 0.65 and literal_pool_size >= 40) else 0
    max_literals = min(base_literals + extra_literals, max(base_literals, 7))

    min_gain = float(config.min_gain)
    if literal_pool_size >= 80:
        min_gain *= 0.5
    if literal_pool_size >= 160:
        min_gain *= 0.5

    return {
        "max_rules_per_class": int(max_rules),
        "min_precision": float(min_precision),
        "max_literals": int(max_literals),
        "min_gain": float(min_gain),
    }


@dataclass
class GeneratorConfig:
    max_rules_per_class: int = 80
    max_total_rules: int = 400
    min_gain: float = 1e-6
    min_pos_support: int = 1
    neg_ratio: float = 0.50
    quantiles: Tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
    use_class_midpoints: bool = False
    enable_diversity_filter: bool = False
    diversity_threshold: float = 0.80
    random_state: int = 0
    breaks: int = 7
    top_k_cats: int = 12
    pair_top: int = 8
    triple_top: int = 24
    max_rules: int | None = 2000
    min_precision: float = 0.50
    max_literals: int = 6
    min_literals: int = 1
    ripper_k: int = 2
    ripper_dl_allowance: int = 64
    ripper_prune_size: float = 0.33
    ripper_n_discretize_bins: int = 10
    ripper_max_rule_conds: int | None = None
    foil_parallel_workers: int = 0
    enable_pool_shaping: bool = False
    pool_family_max_steps: int = 4
    pool_family_precision_relax: float = 0.08
    pool_family_quality_relax: float = 0.04
    pool_family_support_growth: float = 1.75
    pool_quality_beta: float = 0.75
    pool_depth_weight: float = 0.18
    pool_class_weight: float = 0.12
    pool_novelty_weight: float = 0.16
    pool_overlap_penalty: float = 0.10
    pool_max_growth_factor: float = 1.20

    @classmethod
    def default_for_algo(cls, algo: str) -> "GeneratorConfig":
        algo = str(algo or "ripper").lower().strip()
        if algo == "foil":
            return cls(
                min_precision=0.45,
                max_literals=5,
                foil_parallel_workers=0,
                pool_family_max_steps=4,
                pool_family_precision_relax=0.08,
                pool_family_quality_relax=0.04,
                pool_family_support_growth=1.75,
                pool_max_growth_factor=1.20,
            )
        return cls(
            max_rules_per_class=120,
            max_total_rules=600,
            min_precision=0.0,
            max_literals=6,
            ripper_k=3,
            ripper_n_discretize_bins=12,
            pool_family_max_steps=5,
            pool_family_precision_relax=0.15,
            pool_family_quality_relax=0.15,
            pool_family_support_growth=4.00,
            pool_novelty_weight=0.18,
            pool_overlap_penalty=0.08,
            pool_max_growth_factor=1.60,
        )

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any], *, algo: str = "ripper") -> "GeneratorConfig":
        d = cls.default_for_algo(algo)
        return cls(
            max_rules_per_class=int(kwargs.get("max_rules_per_class", d.max_rules_per_class)),
            max_total_rules=int(kwargs.get("max_total_rules", d.max_total_rules)),
            min_gain=float(kwargs.get("min_gain", d.min_gain)),
            min_pos_support=int(kwargs.get("min_pos_support", d.min_pos_support)),
            neg_ratio=float(kwargs.get("neg_ratio", d.neg_ratio)),
            quantiles=tuple(kwargs.get("quantiles") or d.quantiles),
            use_class_midpoints=bool(kwargs.get("use_class_midpoints", d.use_class_midpoints)),
            enable_diversity_filter=bool(kwargs.get("enable_diversity_filter", d.enable_diversity_filter)),
            diversity_threshold=float(kwargs.get("diversity_threshold", d.diversity_threshold)),
            random_state=int(kwargs.get("random_state", d.random_state)),
            breaks=int(kwargs.get("breaks", d.breaks)),
            top_k_cats=int(kwargs.get("top_k_cats", d.top_k_cats)),
            pair_top=int(kwargs.get("pair_top", d.pair_top)),
            triple_top=int(kwargs.get("triple_top", d.triple_top)),
            max_rules=kwargs.get("max_rules", d.max_rules),
            min_precision=float(kwargs.get("min_precision", kwargs.get("classic_min_precision", d.min_precision))),
            max_literals=int(kwargs.get("max_literals", kwargs.get("classic_max_literals", d.max_literals))),
            min_literals=int(kwargs.get("min_literals", kwargs.get("classic_min_literals", d.min_literals))),
            ripper_k=int(kwargs.get("ripper_k", d.ripper_k)),
            ripper_dl_allowance=int(kwargs.get("ripper_dl_allowance", d.ripper_dl_allowance)),
            ripper_prune_size=float(kwargs.get("ripper_prune_size", d.ripper_prune_size)),
            ripper_n_discretize_bins=int(kwargs.get("ripper_n_discretize_bins", d.ripper_n_discretize_bins)),
            ripper_max_rule_conds=kwargs.get("ripper_max_rule_conds", d.ripper_max_rule_conds),
            foil_parallel_workers=int(kwargs.get("foil_parallel_workers", d.foil_parallel_workers)),
            enable_pool_shaping=bool(kwargs.get("enable_pool_shaping", d.enable_pool_shaping)),
            pool_family_max_steps=int(kwargs.get("pool_family_max_steps", d.pool_family_max_steps)),
            pool_family_precision_relax=float(kwargs.get("pool_family_precision_relax", d.pool_family_precision_relax)),
            pool_family_quality_relax=float(kwargs.get("pool_family_quality_relax", d.pool_family_quality_relax)),
            pool_family_support_growth=float(kwargs.get("pool_family_support_growth", d.pool_family_support_growth)),
            pool_quality_beta=float(kwargs.get("pool_quality_beta", d.pool_quality_beta)),
            pool_depth_weight=float(kwargs.get("pool_depth_weight", d.pool_depth_weight)),
            pool_class_weight=float(kwargs.get("pool_class_weight", d.pool_class_weight)),
            pool_novelty_weight=float(kwargs.get("pool_novelty_weight", d.pool_novelty_weight)),
            pool_overlap_penalty=float(kwargs.get("pool_overlap_penalty", d.pool_overlap_penalty)),
            pool_max_growth_factor=float(kwargs.get("pool_max_growth_factor", d.pool_max_growth_factor)),
        )


class BaseInducer:
    def __init__(
        self,
        *,
        config: GeneratorConfig,
        value_decoders=None,
        verbose: bool = False,
        on_emit_rule=None,
        fixed_default_label: int | None = None,
    ):
        self.config = config
        self.value_decoders = value_decoders or {}
        self.verbose = bool(verbose)
        self.on_emit_rule = on_emit_rule
        self.fixed_default_label = None if fixed_default_label is None else int(fixed_default_label)

    def _emit(self, count: int) -> None:
        if callable(self.on_emit_rule):
            try:
                self.on_emit_rule(int(count))
            except Exception:
                pass


def _mask_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    inter = float(np.count_nonzero(left & right))
    union = float(np.count_nonzero(left | right))
    if union <= 0.0:
        return 0.0
    return inter / union


def _rule_depth_bin(n_literals: int) -> str:
    n_literals = int(max(0, n_literals))
    if n_literals <= 1:
        return "short"
    if n_literals <= 3:
        return "medium"
    return "long"


def _f_beta(precision: float, recall: float, beta: float) -> float:
    precision = float(max(0.0, precision))
    recall = float(max(0.0, recall))
    beta = float(max(1e-6, beta))
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    if denom <= 0.0:
        return 0.0
    return float((1.0 + beta_sq) * precision * recall / denom)


def _proposal_quality(stats: Dict[str, Any], *, beta: float) -> float:
    return _f_beta(float(stats.get("precision", 0.0)), float(stats.get("recall", 0.0)), beta=beta)


def _build_rule_family(
    X: np.ndarray,
    y: np.ndarray,
    feat_to_idx: Dict[str, int],
    *,
    label: int,
    cond: Condition,
    stats: Dict[str, Any],
    config: "GeneratorConfig",
) -> list[dict[str, Any]]:
    family: list[dict[str, Any]] = []
    relaxed_precision = max(0.0, float(config.min_precision) - float(config.pool_family_precision_relax))
    quality_relax = max(0.0, float(config.pool_family_quality_relax))
    support_growth = max(1.0, float(config.pool_family_support_growth))
    current = _canonicalize_condition(cond)
    if not current:
        return family

    full_mask = _condition_mask(X, feat_to_idx, current)
    full_stats = dict(stats)
    full_stats["depth_bin"] = _rule_depth_bin(len(current))
    full_quality = _proposal_quality(full_stats, beta=float(config.pool_quality_beta))
    family.append({"label": int(label), "cond": current, "mask": full_mask, "stats": full_stats})

    seen = {_condition_signature(current)}
    steps = 0
    current_stats = full_stats
    while len(current) > 1 and steps < int(config.pool_family_max_steps):
        best_entry = None
        best_score = float("-inf")
        current_quality = _proposal_quality(current_stats, beta=float(config.pool_quality_beta))
        current_support = max(1, int(current_stats.get("support", int(full_mask.sum()))))
        for lit_idx in range(len(current)):
            trial = _canonicalize_condition(current[:lit_idx] + current[lit_idx + 1 :])
            if not trial:
                continue
            signature = _condition_signature(trial)
            if signature in seen:
                continue
            trial_mask, trial_stats = _foil_rule_stats(
                X,
                y,
                feat_to_idx,
                label,
                trial,
                min_pos_support=int(config.min_pos_support),
                min_precision=relaxed_precision,
                stage=f"{stats.get('stage', 'learned')}_ancestor",
            )
            if trial_mask is None or trial_stats is None:
                continue
            trial_stats["depth_bin"] = _rule_depth_bin(len(trial))
            score = _proposal_quality(trial_stats, beta=float(config.pool_quality_beta))
            if score + 1e-12 < max(full_quality - quality_relax, current_quality - quality_relax):
                continue
            trial_support = int(trial_stats.get("support", 0))
            if float(trial_support) > float(current_support) * support_growth + 1e-12:
                continue
            if score > best_score + 1e-12:
                best_score = score
                best_entry = {"label": int(label), "cond": trial, "mask": trial_mask, "stats": trial_stats}
        if best_entry is None:
            break
        family.append(best_entry)
        current = best_entry["cond"]
        current_stats = best_entry["stats"]
        seen.add(_condition_signature(current))
        steps += 1
    return family


def _target_depth_shares(candidates: Sequence[dict[str, Any]]) -> Dict[str, float]:
    counts = {"short": 1.0, "medium": 1.0, "long": 1.0}
    for cand in candidates:
        counts[_rule_depth_bin(len(cand["cond"]))] += 1.0
    weights = {key: math.sqrt(value) for key, value in counts.items()}
    total = sum(weights.values())
    return {key: float(value / total) for key, value in weights.items()}


def _target_class_shares(y: np.ndarray) -> Dict[int, float]:
    classes, counts = np.unique(np.asarray(y, dtype=int), return_counts=True)
    if counts.size == 0:
        return {}
    prevalence = counts.astype(float) / float(counts.sum())
    uniform = np.full_like(prevalence, 1.0 / float(len(prevalence)))
    blended = 0.5 * prevalence + 0.5 * uniform
    return {int(cls): float(weight) for cls, weight in zip(classes, blended)}


def _select_shaped_rule_pool(
    proposals: Sequence[tuple[int, Condition, Dict[str, Any]]],
    *,
    X: np.ndarray,
    y: np.ndarray,
    feat_to_idx: Dict[str, int],
    config: "GeneratorConfig",
    verbose: bool = False,
) -> list[tuple[int, Condition, Dict[str, Any]]]:
    if not proposals:
        return []

    families: list[dict[str, Any]] = []
    dedup: Dict[str, dict[str, Any]] = {}
    for label, cond, stats in proposals:
        for cand in _build_rule_family(
            X,
            y,
            feat_to_idx,
            label=int(label),
            cond=cond,
            stats=stats,
            config=config,
        ):
            signature = f"{int(label)}::{_condition_signature(cand['cond'])}"
            prev = dedup.get(signature)
            cand_quality = _proposal_quality(cand["stats"], beta=float(config.pool_quality_beta))
            cand["stats"]["proposal_quality"] = cand_quality
            if prev is None or cand_quality > float(prev["stats"].get("proposal_quality", -math.inf)) + 1e-12:
                dedup[signature] = cand
    families = list(dedup.values())
    target_depth = _target_depth_shares(families)
    target_class = _target_class_shares(y)
    pool_cap = min(
        int(config.max_total_rules),
        max(1, int(math.ceil(len(proposals) * float(config.pool_max_growth_factor)))),
    )

    selected: list[dict[str, Any]] = []
    remaining = list(families)
    class_counts: Dict[int, int] = {}
    depth_counts: Dict[str, int] = {"short": 0, "medium": 0, "long": 0}

    if verbose:
        print(
            f"[rulegen/pool] proposals={len(proposals)} family_candidates={len(remaining)} "
            f"pool_cap={pool_cap} target_depth={target_depth}",
            flush=True,
        )

    while remaining and len(selected) < pool_cap:
        total_sel = max(1, len(selected))
        best_idx = -1
        best_score = float("-inf")
        for idx, cand in enumerate(remaining):
            label = int(cand["label"])
            cond = cand["cond"]
            stats = cand["stats"]
            depth_bin = _rule_depth_bin(len(cond))
            quality = float(stats.get("proposal_quality", 0.0))
            current_depth_share = float(depth_counts[depth_bin] / total_sel) if selected else 0.0
            current_class_share = float(class_counts.get(label, 0) / total_sel) if selected else 0.0
            depth_bonus = max(0.0, float(target_depth.get(depth_bin, 0.0)) - current_depth_share)
            class_bonus = max(0.0, float(target_class.get(label, 0.0)) - current_class_share)
            overlap = max((_mask_jaccard(cand["mask"], item["mask"]) for item in selected), default=0.0)
            novelty = 1.0 - overlap
            score = (
                quality
                + float(config.pool_depth_weight) * depth_bonus
                + float(config.pool_class_weight) * class_bonus
                + float(config.pool_novelty_weight) * novelty
                - float(config.pool_overlap_penalty) * overlap
            )
            if score > best_score + 1e-12:
                best_score = score
                best_idx = idx

        if best_idx < 0:
            break
        chosen = remaining.pop(best_idx)
        label = int(chosen["label"])
        depth_bin = _rule_depth_bin(len(chosen["cond"]))
        chosen["stats"]["pool_score"] = float(best_score)
        chosen["stats"]["depth_bin"] = depth_bin
        chosen["stats"]["pool_target_depth_share"] = float(target_depth.get(depth_bin, 0.0))
        chosen["stats"]["pool_target_class_share"] = float(target_class.get(label, 0.0))
        selected.append(chosen)
        class_counts[label] = class_counts.get(label, 0) + 1
        depth_counts[depth_bin] += 1

    if verbose:
        print(
            f"[rulegen/pool] selected={len(selected)} depth_counts={depth_counts} "
            f"class_counts={class_counts}",
            flush=True,
        )
    return [(int(item["label"]), item["cond"], item["stats"]) for item in selected]


class StaticInducer(BaseInducer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feat_to_idx: Dict[str, int] = {}

    def _append_rule(self, rules: List[Dict[str, Any]], seen: set[str], *, cond: Condition | None, stage: str, mask: np.ndarray) -> bool:
        if not cond or mask.sum() == 0:
            return False
        signature = _condition_signature(cond)
        if signature in seen:
            return False
        rules.append({
            "specs": cond,
            "label": None,
            "caption": _condition_caption(cond, self.value_decoders, None),
            "stats": {"kind": "static", "stage": stage, "support": int(mask.sum()), "coverage": float(mask.mean()), "literals": len(cond)},
        })
        seen.add(signature)
        self._emit(len(rules))
        return True

    @staticmethod
    def _select_entries(entries: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
        return sorted(entries, key=lambda x: (float(x.get("score", 0.0)), float(x.get("coverage", 0.0))), reverse=True)[:limit]

    def _literal_pool(self, X: np.ndarray, names: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        pool: Dict[str, List[Dict[str, Any]]] = {}
        numeric, categorical = _split_feature_kinds(X, names, self.value_decoders, cat_threshold=max(20, self.config.top_k_cats * 2))
        qs = np.linspace(0.0, 1.0, max(3, self.config.breaks + 2))[1:-1]

        for i in numeric:
            finite = X[:, i][np.isfinite(X[:, i])]
            if finite.size < 4:
                continue
            thresholds = np.unique(np.quantile(finite, np.unique(qs)))
            thresholds = thresholds[np.isfinite(thresholds)]
            if thresholds.size == 0:
                continue
            cleaned: List[float] = []
            for t in sorted(thresholds):
                if not cleaned or abs(t - cleaned[-1]) > 1e-7:
                    cleaned.append(float(t))
            score = float(_feature_score(X[:, i]))
            entries: List[Dict[str, Any]] = []
            for t in cleaned:
                for op in ("<", ">"):
                    mask = _mask_from_literal(X[:, i], op, t)
                    if mask.sum() == 0:
                        continue
                    entries.append({"literal": (names[i], op, t), "mask": mask, "coverage": float(mask.mean()), "score": score})
            pool[names[i]] = entries

        for i in categorical:
            values, counts = np.unique(X[:, i], return_counts=True)
            chosen = values[np.argsort(-counts)[: min(self.config.top_k_cats, len(values))]]
            score = float(_feature_score(X[:, i]))
            entries = []
            for value in chosen:
                mask = _mask_from_literal(X[:, i], "==", value)
                if mask.sum() == 0:
                    continue
                entries.append({"literal": (names[i], "==", _normalise_value(value)), "mask": mask, "coverage": float(mask.mean()), "score": score})
            pool[names[i]] = entries
        return pool

    def _filter_redundant_rules(self, rules: List[Dict[str, Any]], X: np.ndarray) -> List[Dict[str, Any]]:
        if len(rules) < 2:
            return rules
        masks = []
        for rule in rules:
            mask = np.ones(X.shape[0], dtype=bool)
            for name, op, value in rule["specs"]:
                idx = self._feat_to_idx.get(name)
                if idx is None:
                    mask[:] = False
                    break
                mask &= _mask_from_literal(X[:, idx], op, value)
            masks.append(mask)
        kept = [0]
        for i in range(1, len(rules)):
            too_close = False
            for j in kept:
                inter = float((masks[i] & masks[j]).sum())
                union = float((masks[i] | masks[j]).sum())
                if inter / (union + 1e-9) > float(self.config.diversity_threshold):
                    too_close = True
                    break
            if not too_close:
                kept.append(i)
        return [rules[i] for i in kept]

    def _limit(self, X: np.ndarray, n_single: int) -> tuple[int, int]:
        if self.config.max_rules is None:
            n_samples, n_features = X.shape
            cap_by_features = 40 * int(n_features)
            cap_by_samples = max(100, int(0.2 * int(n_samples)))
            max_multi = min(self.config.max_total_rules, cap_by_features, cap_by_samples)
            max_multi = max(60, int(max_multi))
            total_cap = max_multi + n_single
        else:
            total_cap = int(self.config.max_rules)
            max_multi = max(0, total_cap - n_single)
        return total_cap, max_multi

    def generate(self, X, y=None, *, feature_names) -> List[Dict[str, Any]]:
        X_np = np.asarray(X)
        if X_np.ndim != 2:
            raise ValueError("X must be 2D")
        names = list(feature_names)
        self._feat_to_idx = {name: i for i, name in enumerate(names)}
        pool = self._literal_pool(X_np, names)
        rules: List[Dict[str, Any]] = []
        seen: set[str] = set()

        for entries in pool.values():
            for entry in self._select_entries(entries, limit=len(entries)):
                self._append_rule(rules, seen, cond=_canonicalize_condition((entry["literal"],)), stage="single", mask=entry["mask"])

        total_cap, max_multi = self._limit(X_np, len(rules))
        scores = [_feature_score(X_np[:, i]) for i in range(X_np.shape[1])]
        built_multi = 0

        pairs = sorted(itertools.combinations(range(X_np.shape[1]), 2), key=lambda p: scores[p[0]] + scores[p[1]], reverse=True)
        for i, j in pairs[: max(1, self.config.pair_top)]:
            left_entries = self._select_entries(pool.get(names[i], []))
            right_entries = self._select_entries(pool.get(names[j], []))
            for left, right in itertools.product(left_entries, right_entries):
                cond = _canonicalize_condition((left["literal"], right["literal"]))
                mask = left["mask"] & right["mask"]
                if self._append_rule(rules, seen, cond=cond if cond and len(cond) >= 2 else None, stage="pair", mask=mask):
                    built_multi += 1
                    if built_multi >= max_multi:
                        break
            if built_multi >= max_multi:
                break

        if built_multi < max_multi and self.config.triple_top:
            top_features = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
            trios = list(itertools.combinations(top_features[: max(3, min(len(top_features), self.config.triple_top))], 3))
            for trio in trios[: self.config.triple_top]:
                choice_sets = [self._select_entries(pool.get(names[i], [])) for i in trio]
                if not all(choice_sets):
                    continue
                for combo in itertools.product(*choice_sets):
                    cond = _canonicalize_condition(tuple(entry["literal"] for entry in combo))
                    if not cond or len(cond) < 3:
                        continue
                    mask = combo[0]["mask"]
                    for entry in combo[1:]:
                        mask &= entry["mask"]
                    if self._append_rule(rules, seen, cond=cond, stage="triple", mask=mask):
                        built_multi += 1
                        if built_multi >= max_multi:
                            break
                if built_multi >= max_multi:
                    break

        out = rules[:total_cap]
        if self.config.enable_diversity_filter:
            out = self._filter_redundant_rules(out, X_np)
        return out[:total_cap]


class SupervisedBaseInducer(BaseInducer):
    algo_name = "base"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ruleset: Dict[int, List[Condition]] = {}
        self.ordered_rules: List[Tuple[int, Condition, Dict[str, Any]]] = []
        self.default_label: int | None = None
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.feat: List[str] = []
        self.feat_to_idx: Dict[str, int] = {}
        self.is_cat: List[bool] = []
        self.df_decoded: pd.DataFrame | None = None
        self.reverse_value_maps: Dict[str, Dict[Any, Any]] = {}
        self.seen_signatures: set[str] = set()

    def _prepare(self, X, y, *, feature_names=None):
        self.X = np.asarray(X)
        self.y = np.asarray(y, dtype=int)
        if self.X.ndim != 2:
            raise ValueError("X must be 2D")
        self.feat = list(feature_names) if feature_names else [f"X{i}" for i in range(self.X.shape[1])]
        self.feat_to_idx = {name: i for i, name in enumerate(self.feat)}
        self.is_cat = _categorical_flags(self.X, self.feat, self.value_decoders, cat_threshold=20)
        self.df_decoded = _decoded_dataframe(self.X, self.feat, self.value_decoders, self.is_cat)
        self.reverse_value_maps = _reverse_value_decoders(self.value_decoders)
        inferred_default = int(np.argmax(np.bincount(self.y)))
        self.default_label = inferred_default if self.fixed_default_label is None else int(self.fixed_default_label)
        self.ruleset = {int(c): [] for c in np.unique(self.y)}
        self.ordered_rules = []
        self.seen_signatures = set()

    def _record_rule(self, label: int, cond: Condition, *, stage: str) -> np.ndarray | None:
        cond = _canonicalize_condition(cond)
        if not cond:
            return None
        signature = f"{int(label)}::{_condition_signature(cond)}"
        if signature in self.seen_signatures:
            return None

        rule_mask = _condition_mask(self.X, self.feat_to_idx, cond)
        pos = self.y == label
        neg = ~pos
        support = int((rule_mask & pos).sum())
        neg_covered = int((rule_mask & neg).sum())
        if support < max(1, int(self.config.min_pos_support)):
            return None
        precision = support / (support + neg_covered + 1e-12)
        if precision < float(self.config.min_precision):
            return None
        recall = support / max(1, int(pos.sum()))
        stats = {
            "kind": "learned",
            "stage": stage,
            "label": int(label),
            "support": support,
            "neg_covered": neg_covered,
            "precision": precision,
            "recall": recall,
            "f1": (2.0 * precision * recall) / (precision + recall + 1e-12),
            "coverage": float(rule_mask.mean()),
            "literals": len(cond),
            "min_precision": float(self.config.min_precision),
        }
        self.seen_signatures.add(signature)
        self.ruleset[int(label)].append(cond)
        self.ordered_rules.append((int(label), cond, stats))
        self._emit(len(self.ordered_rules))
        return rule_mask

    def _finalize(self) -> List[Dict[str, Any]]:
        return [
            {
                "specs": tuple(cond),
                "label": int(label),
                "caption": _condition_caption(cond, self.value_decoders, label),
                "stats": stats,
            }
            for label, cond, stats in self.ordered_rules
        ]

    def generate(self, X, y, *, feature_names):
        self.fit(X, y, feature_names=feature_names)
        return self._finalize()


class CanonicalFoilInducer(SupervisedBaseInducer):
    algo_name = "foil"

    def _foil_worker_count(self, n_classes: int) -> int:
        configured = int(self.config.foil_parallel_workers)
        if configured > 0:
            return min(configured, n_classes)
        cpu = os.cpu_count() or 1
        if int(self.X.shape[1]) >= 64 and n_classes >= 4:
            return min(n_classes, max(1, min(8, cpu - 1)))
        return min(n_classes, max(1, min(4, cpu // 2 or 1)))

    def fit(self, X, y, *, feature_names=None):
        self._prepare(X, y, feature_names=feature_names)
        classes = [int(c) for c in sorted(np.unique(self.y), key=lambda c: int((self.y == c).sum()))]
        per_class_results: dict[int, list[tuple[int, Condition, Dict[str, Any]]]] = {cls: [] for cls in classes}
        worker_count = self._foil_worker_count(len(classes))
        if self.verbose:
            class_sizes = {cls: int((self.y == cls).sum()) for cls in classes}
            print(
                f"[rulegen/FOIL] class_order={classes} class_sizes={class_sizes} "
                f"foil_parallel_workers={worker_count}",
                flush=True,
            )

        if worker_count > 1 and len(classes) > 1:
            with ProcessPoolExecutor(max_workers=worker_count) as ex:
                futures = {
                    ex.submit(
                        _fit_foil_class_worker,
                        self.X,
                        self.y,
                        self.feat,
                        self.is_cat,
                        self.config,
                        cls,
                        verbose=self.verbose,
                    ): cls
                    for cls in classes
                }
                for fut, cls in ((f, c) for f, c in futures.items()):
                    per_class_results[cls] = fut.result()
                    if self.verbose:
                        print(
                            f"[rulegen/FOIL] class={cls} finished accepted_rules={len(per_class_results[cls])}",
                            flush=True,
                        )
        else:
            for cls in classes:
                if self.verbose:
                    print(
                        f"[rulegen/FOIL] class={cls} start positives={int((self.y == cls).sum())}",
                        flush=True,
                    )
                per_class_results[cls] = _fit_foil_class_worker(
                    self.X,
                    self.y,
                    self.feat,
                    self.is_cat,
                    self.config,
                    cls,
                    verbose=self.verbose,
                )
                if self.verbose:
                    print(
                        f"[rulegen/FOIL] class={cls} finished accepted_rules={len(per_class_results[cls])}",
                        flush=True,
                    )

        proposals: list[tuple[int, Condition, Dict[str, Any]]] = []
        for cls in classes:
            proposals.extend(per_class_results[cls])

        ordered_entries = (
            _select_shaped_rule_pool(
                proposals,
                X=self.X,
                y=self.y,
                feat_to_idx=self.feat_to_idx,
                config=self.config,
                verbose=self.verbose,
            )
            if self.config.enable_pool_shaping
            else proposals
        )

        for label, cond, stats in ordered_entries:
            if len(self.ordered_rules) >= self.config.max_total_rules:
                break
            signature = f"{int(label)}::{_condition_signature(cond)}"
            if signature in self.seen_signatures:
                continue
            self.seen_signatures.add(signature)
            self.ruleset[int(label)].append(cond)
            self.ordered_rules.append((int(label), cond, stats))
            self._emit(len(self.ordered_rules))
        if self.verbose:
            print(f"[rulegen/FOIL] total_rules={len(self.ordered_rules)}", flush=True)
        return self


class CanonicalRipperInducer(SupervisedBaseInducer):
    algo_name = "ripper"

    def _is_high_dim_numeric_multiclass(self) -> bool:
        n_classes = int(np.unique(self.y).size)
        if n_classes < 4:
            return False
        n_features = int(self.X.shape[1])
        numeric_ratio = float(sum(not flag for flag in self.is_cat)) / float(max(1, len(self.is_cat)))
        return n_features >= 64 and numeric_ratio >= 0.80

    def _base_backend_hyperparams(self) -> dict[str, Any]:
        base = {
            "k": int(self.config.ripper_k),
            "dl_allowance": int(self.config.ripper_dl_allowance),
            "prune_size": float(self.config.ripper_prune_size),
            "n_discretize_bins": int(self.config.ripper_n_discretize_bins),
            "max_rules": int(self.config.max_rules_per_class),
            "max_rule_conds": self.config.ripper_max_rule_conds,
        }
        if self._is_high_dim_numeric_multiclass():
            base["k"] = max(int(base["k"]), 3)
            base["dl_allowance"] = max(int(base["dl_allowance"]), 72)
            base["prune_size"] = min(max(float(base["prune_size"]), 0.35), 0.40)
            base["n_discretize_bins"] = max(int(base["n_discretize_bins"]), 10)
            base["max_rules"] = min(int(base["max_rules"]), max(72, 16 * int(np.unique(self.y).size)))
            base["max_rule_conds"] = 4 if base["max_rule_conds"] is None else min(int(base["max_rule_conds"]), 4)
        return base

    def _class_rule_target(self, positives: int) -> int:
        cap = max(1, int(self.config.max_rules_per_class))
        exploratory_cap = max(12, cap // 3)
        target = int(math.ceil(math.sqrt(max(1, int(positives)))))
        return min(cap, max(12, min(exploratory_cap, target)))

    def _backend_param_plan(self, primary: dict[str, Any]) -> list[dict[str, Any]]:
        all_candidates = [dict(primary)] + self._backend_param_candidates()
        ordered: list[dict[str, Any]] = []
        seen = set()
        for cand in all_candidates:
            key = tuple(sorted(cand.items()))
            if key in seen:
                continue
            seen.add(key)
            ordered.append(cand)
        max_models = 4 if self._is_high_dim_numeric_multiclass() else 3
        return ordered[:max_models]

    def _backend_param_candidates(self) -> list[dict[str, Any]]:
        base = self._base_backend_hyperparams()
        if self._is_high_dim_numeric_multiclass():
            candidates = [
                dict(base),
                {
                    **base,
                    "k": max(int(base["k"]), 4),
                    "dl_allowance": max(int(base["dl_allowance"]), 96),
                    "prune_size": min(float(base["prune_size"]), 0.32),
                    "n_discretize_bins": max(int(base["n_discretize_bins"]), 14),
                    "max_rules": min(int(self.config.max_rules_per_class), max(int(base["max_rules"]), 18 * int(np.unique(self.y).size))),
                    "max_rule_conds": 4 if base["max_rule_conds"] is None else min(max(int(base["max_rule_conds"]), 4), 4),
                },
                {
                    **base,
                    "k": max(int(base["k"]), 5),
                    "dl_allowance": max(int(base["dl_allowance"]), 128),
                    "prune_size": min(float(base["prune_size"]), 0.28),
                    "n_discretize_bins": max(int(base["n_discretize_bins"]), 16),
                    "max_rules": min(int(self.config.max_rules_per_class), max(int(base["max_rules"]), 20 * int(np.unique(self.y).size))),
                    "max_rule_conds": 5 if base["max_rule_conds"] is None else min(max(int(base["max_rule_conds"]), 5), 5),
                },
                {
                    **base,
                    "k": max(int(base["k"]), 3),
                    "dl_allowance": max(int(base["dl_allowance"]), 80),
                    "prune_size": min(max(float(base["prune_size"]), 0.38), 0.42),
                    "n_discretize_bins": max(int(base["n_discretize_bins"]), 12),
                    "max_rules": min(int(self.config.max_rules_per_class), max(int(base["max_rules"]), 16 * int(np.unique(self.y).size))),
                    "max_rule_conds": 4 if base["max_rule_conds"] is None else min(max(int(base["max_rule_conds"]), 4), 4),
                },
            ]
        else:
            candidates = [
                dict(base),
                {
                    **base,
                    "k": max(int(base["k"]), 3),
                    "dl_allowance": max(int(base["dl_allowance"]), 96),
                    "prune_size": min(float(base["prune_size"]), 0.25),
                    "n_discretize_bins": max(int(base["n_discretize_bins"]), 12),
                    "max_rules": min(int(self.config.max_rules_per_class), max(int(base["max_rules"]), 100)),
                    "max_rule_conds": 5 if base["max_rule_conds"] is None else min(max(int(base["max_rule_conds"]), 5), 5),
                },
                {
                    **base,
                    "k": max(int(base["k"]), 4),
                    "dl_allowance": max(int(base["dl_allowance"]), 128),
                    "prune_size": min(float(base["prune_size"]), 0.20),
                    "n_discretize_bins": max(int(base["n_discretize_bins"]), 16),
                    "max_rules": min(int(self.config.max_rules_per_class), max(int(base["max_rules"]), 120)),
                    "max_rule_conds": 6 if base["max_rule_conds"] is None else min(max(int(base["max_rule_conds"]), 6), 6),
                },
            ]
        unique: list[dict[str, Any]] = []
        seen = set()
        for cand in candidates:
            key = tuple(sorted(cand.items()))
            if key in seen:
                continue
            seen.add(key)
            unique.append(cand)
        return unique

    def _fit_backend_model(self, df_train: pd.DataFrame, params: dict[str, Any]):
        model = lw.RIPPER(
            k=int(params["k"]),
            dl_allowance=int(params["dl_allowance"]),
            prune_size=float(params["prune_size"]),
            n_discretize_bins=int(params["n_discretize_bins"]),
            max_rules=int(params["max_rules"]),
            max_rule_conds=params["max_rule_conds"],
            random_state=int(self.config.random_state),
            verbosity=0,
        )
        model.fit(df_train, class_feat="__target__", pos_class=1, cn_optimize=True)
        return model

    def _score_backend_params(self, df_train: pd.DataFrame, df_val: pd.DataFrame, params: dict[str, Any]) -> tuple[float, float, float]:
        model = self._fit_backend_model(df_train, params)
        y_val = df_val["__target__"].to_numpy(dtype=int)
        preds = np.asarray(model.predict(df_val.drop(columns="__target__")), dtype=bool).astype(int)
        pos_f1 = float(f1_score(y_val, preds, average="binary", zero_division=0))
        macro_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
        n_rules = len(getattr(getattr(model, "ruleset_", None), "rules", []) or [])
        complexity_penalty = float(n_rules)
        return pos_f1, macro_f1, -complexity_penalty

    def _select_backend_hyperparams(self, df_subset: pd.DataFrame) -> dict[str, Any]:
        y_binary = df_subset["__target__"].to_numpy(dtype=int)
        pos = int(y_binary.sum())
        neg = int((1 - y_binary).sum())
        if np.unique(self.y).size != 2:
            return self._base_backend_hyperparams()
        if len(df_subset) < 2000 or pos < 40 or neg < 40:
            return self._base_backend_hyperparams()

        idx = np.arange(len(df_subset), dtype=int)
        try:
            tr_idx, val_idx = train_test_split(
                idx,
                test_size=0.2,
                random_state=int(self.config.random_state),
                stratify=y_binary,
            )
        except Exception:
            return self._base_backend_hyperparams()
        if len(val_idx) == 0:
            return self._base_backend_hyperparams()

        df_train = df_subset.iloc[tr_idx].reset_index(drop=True)
        df_val = df_subset.iloc[val_idx].reset_index(drop=True)
        best_params = self._base_backend_hyperparams()
        best_score = (-1.0, -1.0, float("-inf"))
        for params in self._backend_param_candidates():
            try:
                score = self._score_backend_params(df_train, df_val, params)
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_params = params
        return best_params

    def _rule_from_wittgenstein(self, backend_rule, *, numeric_binned_features: set[str]) -> Condition | None:
        literals: List[Literal] = []
        for cond in getattr(backend_rule, "conds", []) or []:
            literals.extend(
                _backend_cond_to_literals(
                    cond.feature,
                    cond.val,
                    numeric_binned_features=numeric_binned_features,
                    reverse_value_maps=self.reverse_value_maps,
                )
            )
        return _canonicalize_condition(literals)

    def fit(self, X, y, *, feature_names=None):
        self._prepare(X, y, feature_names=feature_names)
        if lw is None:
            raise ImportError("wittgenstein is required for canonical RIPPER induction.")

        remaining_mask = np.ones(len(self.y), dtype=bool)
        classes = [int(c) for c in sorted(np.unique(self.y), key=lambda c: int((self.y == c).sum()))]
        proposals: list[tuple[int, Condition, Dict[str, Any]]] = []
        if self.verbose:
            class_sizes = {cls: int((self.y == cls).sum()) for cls in classes}
            print(
                f"[rulegen/RIPPER] class_order={classes} class_sizes={class_sizes}",
                flush=True,
            )
            if self._is_high_dim_numeric_multiclass():
                print(
                    "[rulegen/RIPPER] profile=high_dim_numeric_multiclass "
                    f"n_features={int(self.X.shape[1])} n_classes={len(classes)}",
                    flush=True,
                )

        for cls in classes:
            subset_idx = np.where(remaining_mask)[0]
            if subset_idx.size == 0:
                break
            y_subset = self.y[subset_idx]
            pos_subset = (y_subset == cls)
            if not pos_subset.any():
                continue
            if (~pos_subset).sum() == 0:
                continue
            if self.verbose:
                print(
                    f"[rulegen/RIPPER] class={cls} start subset={int(subset_idx.size)} "
                    f"positives={int(pos_subset.sum())}",
                    flush=True,
                )

            df_subset = self.df_decoded.iloc[subset_idx].reset_index(drop=True).copy()
            df_subset["__target__"] = pos_subset.astype(int)
            backend_params = self._select_backend_hyperparams(df_subset)
            covered_cls = np.zeros(len(self.y), dtype=bool)
            before = len(proposals)
            class_seen: set[str] = set()
            target_rules = self._class_rule_target(int(pos_subset.sum()))
            param_plan = self._backend_param_plan(backend_params)
            if self.verbose:
                print(
                    f"[rulegen/RIPPER] class={cls} target_rules={target_rules} primary_params={backend_params}",
                    flush=True,
                )
            for round_idx, params in enumerate(param_plan, start=1):
                model = self._fit_backend_model(df_subset, params)
                bin_transformer = getattr(model, "bin_transformer_", None)
                numeric_binned_features = set(getattr(bin_transformer, "bins_", {}).keys())
                backend_rules = getattr(getattr(model, "ruleset_", None), "rules", []) or []
                added_this_round = 0
                if self.verbose:
                    print(
                        f"[rulegen/RIPPER] class={cls} round={round_idx}/{len(param_plan)} "
                        f"backend_rules={len(backend_rules)} params={params}",
                        flush=True,
                    )
                for backend_rule in backend_rules:
                    cond = self._rule_from_wittgenstein(backend_rule, numeric_binned_features=numeric_binned_features)
                    if not cond:
                        continue
                    cond_sig = _condition_signature(cond)
                    if cond_sig in class_seen:
                        continue
                    rule_mask, stats = _foil_rule_stats(
                        self.X,
                        self.y,
                        self.feat_to_idx,
                        cls,
                        cond,
                        min_pos_support=int(self.config.min_pos_support),
                        min_precision=float(self.config.min_precision),
                        stage=f"ripper_round{round_idx}",
                    )
                    if rule_mask is None or stats is None:
                        continue
                    proposals.append((int(cls), cond, stats))
                    class_seen.add(cond_sig)
                    covered_cls |= rule_mask & (self.y == cls) & remaining_mask
                    added_this_round += 1
                if self.verbose:
                    print(
                        f"[rulegen/RIPPER] class={cls} round={round_idx} added_rules={added_this_round} "
                        f"class_total={len(class_seen)}",
                        flush=True,
                    )
                if len(class_seen) >= target_rules:
                    break
            if covered_cls.any():
                remaining_mask &= ~covered_cls
            if self.verbose:
                accepted = len(proposals) - before
                print(
                    f"[rulegen/RIPPER] class={cls} accepted_rules={accepted} "
                    f"covered_pos={int(covered_cls.sum())} remaining={int(remaining_mask.sum())}",
                    flush=True,
                )

        shaping_config = self.config
        if self.config.enable_pool_shaping and self._is_high_dim_numeric_multiclass():
            shaping_config = replace(
                self.config,
                max_total_rules=min(int(self.config.max_total_rules), 220),
                pool_family_max_steps=1,
                pool_novelty_weight=max(float(self.config.pool_novelty_weight), 0.20),
                pool_overlap_penalty=min(float(self.config.pool_overlap_penalty), 0.12),
                pool_class_weight=max(float(self.config.pool_class_weight), 0.16),
                pool_depth_weight=max(float(self.config.pool_depth_weight), 0.12),
                pool_max_growth_factor=min(float(self.config.pool_max_growth_factor), 0.90),
            )
        ordered_entries = (
            _select_shaped_rule_pool(
                proposals,
                X=self.X,
                y=self.y,
                feat_to_idx=self.feat_to_idx,
                config=shaping_config,
                verbose=self.verbose,
            )
            if self.config.enable_pool_shaping
            else proposals
        )
        for label, cond, stats in ordered_entries:
            if len(self.ordered_rules) >= self.config.max_total_rules:
                break
            signature = f"{int(label)}::{_condition_signature(cond)}"
            if signature in self.seen_signatures:
                continue
            self.seen_signatures.add(signature)
            self.ruleset[int(label)].append(cond)
            self.ordered_rules.append((int(label), cond, stats))
            self._emit(len(self.ordered_rules))
        if self.verbose:
            print(f"[rulegen/RIPPER] total_rules={len(self.ordered_rules)}", flush=True)
        return self


class RuleGenerator:
    def __init__(self, *, algo="ripper", verbose=False, **kwargs):
        algo = str(algo or "ripper").lower()
        if algo not in {"ripper", "foil", "static"}:
            raise ValueError("algo must be ripper/foil/static")
        self.algo = algo
        self.verbose = bool(verbose)
        self.config = GeneratorConfig.from_kwargs(kwargs, algo=algo)
        self.on_emit_rule = kwargs.get("on_emit_rule")
        self.value_decoders = kwargs.get("value_decoders") or {}
        self._inducer = None
        self._ordered_rules: List[Tuple[int, Condition, Dict[str, Any]]] = []
        self._default_label: int | None = None

    def _make_inducer(self, *, value_decoders=None):
        shared = {
            "config": self.config,
            "value_decoders": value_decoders or self.value_decoders,
            "verbose": self.verbose,
            "on_emit_rule": self.on_emit_rule,
        }
        if self.algo == "static":
            return StaticInducer(**shared)
        if self.algo == "ripper":
            return CanonicalRipperInducer(**shared)
        return CanonicalFoilInducer(**shared)

    def fit(self, X, y, *, feature_names=None):
        if self.algo == "static":
            raise ValueError("Static inducer does not support fit(); call generate().")
        self._inducer = self._make_inducer()
        self._inducer.fit(X, y, feature_names=feature_names)
        self._ordered_rules = list(self._inducer.ordered_rules)
        self._default_label = self._inducer.default_label
        return self

    @property
    def ordered_rules(self):
        return list(self._ordered_rules)

    @property
    def default_label(self):
        return self._default_label

    def generate(self, X, y, **kwargs):
        feature_names = kwargs.get("feature_names")
        value_decoders = kwargs.get("value_decoders") or self.value_decoders
        feature_names = list(feature_names) if feature_names else [f"X{i}" for i in range(np.asarray(X).shape[1])]
        inducer = self._make_inducer(value_decoders=value_decoders)
        self._inducer = inducer
        if self.algo == "static":
            return inducer.generate(X, y, feature_names=feature_names)
        if y is None:
            raise ValueError("Supervised inducer requires y.")
        result = inducer.generate(X, y, feature_names=feature_names)
        self._ordered_rules = list(inducer.ordered_rules)
        self._default_label = inducer.default_label
        return result


__all__ = [
    "RuleGenerator",
    "GeneratorConfig",
    "StaticInducer",
    "CanonicalRipperInducer",
    "CanonicalFoilInducer",
    "Literal",
    "Condition",
    "_canonicalize_condition",
    "_condition_signature",
    "_condition_caption",
    "_format_literal_value",
    "_normalise_value",
]
