# -*- coding: utf-8 -*-
"""
Core DST (Dempster–Shafer Theory) utilities used by DSModelMultiQ.

Exposed API (imported by the model/code elsewhere):
  - params_to_mass(W): convert rule parameters to normalized masses over K classes + Omega
  - ds_combine_pair(mA, mB): Dempster's rule combination for two mass assignments
  - ds_combine_many(m_list): left-fold combination over a sequence of masses
  - masses_to_pignistic(m): convert masses to pignistic (betP) probabilities

All functions accept and return either numpy arrays or torch tensors where sensible.
If torch is unavailable, numpy code paths are used.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional
import copy
import math
import numpy as np

try:
    from .rule_utils import condition_caption, format_literal_value, normalise_rule_value
except ImportError:  # pragma: no cover - direct script/import fallback
    from rule_utils import condition_caption, format_literal_value, normalise_rule_value

try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor
    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    F = None      # type: ignore
    Tensor = np.ndarray  # type: ignore
    _HAS_TORCH = False


ArrayLike = np.ndarray | Tensor


# -----------------------------------------------------------------------------
# Reproducible train/test split helper (shared across scripts)
# -----------------------------------------------------------------------------
def split_train_test(
    X,
    y,
    *,
    test_size: float = 0.16,
    seed: int = 42,
    stratify: bool = True,
):
    """Return (X_train, X_test, y_train, y_test, train_idx, test_idx).

    This is a thin wrapper that matches the project-wide split convention and
    falls back to an unstratified split if stratification is impossible.
    """
    from sklearn.model_selection import train_test_split

    X = np.asarray(X)
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError(f"X and y must have same length, got {len(X)} and {len(y)}")

    idx_all = np.arange(len(y))
    strat = y if (stratify and y is not None) else None
    try:
        X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
            X,
            y,
            idx_all,
            test_size=float(test_size),
            random_state=int(seed),
            stratify=strat,
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
            X,
            y,
            idx_all,
            test_size=float(test_size),
            random_state=int(seed),
        )
    return X_tr, X_te, y_tr, y_te, idx_tr, idx_te


# ----------------------------- Helpers -----------------------------
def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert torch tensor to numpy (detach+cpu), or return numpy as is."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _to_same_backend(x: np.ndarray, like: ArrayLike) -> ArrayLike:
    """Return x converted to the backend (torch/numpy) of 'like'."""
    if _HAS_TORCH and isinstance(like, torch.Tensor):
        return torch.from_numpy(x).to(like.device).to(torch.float32)
    return x.astype(np.float32, copy=False)

def _softmax_last(a: ArrayLike) -> ArrayLike:
    """Stable softmax over the last dimension for numpy/torch."""
    if _HAS_TORCH and isinstance(a, torch.Tensor):
        if F is None:
            raise RuntimeError("torch.nn.functional must be available for tensor softmax")
        return F.softmax(a, dim=-1)
    a = _to_numpy(a)
    m = np.max(a, axis=-1, keepdims=True)
    e = np.exp(a - m)
    s = np.sum(e, axis=-1, keepdims=True)
    return e / (s + 1e-12)


# ----------------------------- Params → Masses -----------------------------
def params_to_mass(W: ArrayLike) -> ArrayLike:
    """Normalize parameters so they form a valid mass vector on the simplex."""
    if _HAS_TORCH and isinstance(W, torch.Tensor):
        eps = 1e-8
        mass = torch.clamp(W, min=eps)
        mass_sum = mass.sum(dim=-1, keepdim=True)
        mass = mass / mass_sum.clamp_min(eps)
        fallback = torch.zeros_like(mass)
        fallback[..., -1:] = 1.0
        return torch.where(mass_sum > eps, mass, fallback)

    mass = _to_numpy(W).astype(np.float32, copy=False)
    eps = 1e-8
    mass = np.clip(mass, eps, None)
    mass_sum = np.sum(mass, axis=-1, keepdims=True)
    mass = mass / np.clip(mass_sum, eps, None)
    if np.any(mass_sum <= eps):
        fallback = np.zeros_like(mass)
        fallback[..., -1] = 1.0
        mask = mass_sum <= eps
        mass[mask] = fallback[mask]
    return _to_same_backend(mass, W)



def logits_to_mass(W: ArrayLike) -> ArrayLike:
    """Convert unconstrained logits to a valid mass vector on the simplex (softmax)."""
    if _HAS_TORCH and isinstance(W, torch.Tensor):
        # stable softmax
        x = W - W.max(dim=-1, keepdim=True).values
        return F.softmax(x, dim=-1)

    x = _to_numpy(W).astype(np.float32, copy=False)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    s = np.sum(e, axis=-1, keepdims=True)
    out = e / np.clip(s, 1e-12, None)
    return _to_same_backend(out, W)


def mass_to_logits(m: ArrayLike) -> ArrayLike:
    """Convert a mass vector (on simplex) to logits (log-space). Useful for reparameterization."""
    if _HAS_TORCH and isinstance(m, torch.Tensor):
        eps = 1e-12
        return torch.log(m.clamp_min(eps))
    x = _to_numpy(m).astype(np.float32, copy=False)
    eps = 1e-12
    return _to_same_backend(np.log(np.clip(x, eps, None)), m)



# ----------------------------- Pignistic -----------------------------
def masses_to_pignistic(m: ArrayLike) -> ArrayLike:
    """
    Pignistic transform betP (distributes the mass on Omega equally among K singletons).

    Input:
      m : shape (..., K+1), last column is Omega (ignorance).
    Output:
      p : shape (..., K), probabilities summing to 1 across classes.
    """
    if _HAS_TORCH and isinstance(m, torch.Tensor):
        K = m.shape[-1] - 1
        single = m[..., :K]
        omega = m[..., K:K+1]
        p = single + omega / max(1, K)
        # safe renorm
        s = p.sum(dim=-1, keepdim=True)
        return p / (s + 1e-12)
    mnp = _to_numpy(m)
    K = mnp.shape[-1] - 1
    single = mnp[..., :K]
    omega = mnp[..., [K]]
    p = single + omega / max(1, K)
    s = p.sum(axis=-1, keepdims=True)
    p = p / (s + 1e-12)
    return _to_same_backend(p.astype(np.float32, copy=False), m)


def ds_combine_pair(mA: ArrayLike, mB: ArrayLike) -> ArrayLike:
    """
    Combine two mass assignments using Dempster's rule
    under the singletons+Omega assumption (no composite subsets).

    Shapes:
      mA, mB: (..., K+1)  →  mC: (..., K+1)

    Formula:
      For i in singletons: mC[i] ∝ mA[i]*mB[i] + mA[i]*mB[Ω] + mA[Ω]*mB[i]
      mC[Ω] ∝ mA[Ω]*mB[Ω]
      κ (conflict) = Σ_{i≠j} mA[i]*mB[j]
      Normalize by (1 - κ).

    Important:
      Under normalized Dempster combination, κ is *not* added to Ω.
      Conflict is removed by the normalization factor 1-κ, which can make the
      fused mass look sharper than the input masses.
    """
    if _HAS_TORCH and isinstance(mA, torch.Tensor):
        eps = 1e-9
        K = mA.shape[-1] - 1
        a_s, a_o = mA[..., :K], mA[..., K:K+1]
        b_s, b_o = mB[..., :K], mB[..., K:K+1]
        same = a_s * b_s
        cross = a_s * b_o + a_o * b_s
        num_s = same + cross
        num_o = a_o * b_o
        tot = torch.sum(a_s, dim=-1, keepdim=True) * torch.sum(b_s, dim=-1, keepdim=True)
        diag = torch.sum(a_s * b_s, dim=-1, keepdim=True)
        kappa_raw = torch.clamp(tot - diag, min=0.0)
        kappa = torch.clamp(kappa_raw, max=1.0 - eps)
        denom = (1.0 - kappa).clamp_min(eps)
        mC_s = num_s / denom
        mC_o = num_o / denom
        out = torch.cat([mC_s, mC_o], dim=-1)
        # safety renorm to simplex
        out = out / out.sum(dim=-1, keepdim=True).clamp_min(eps)
        return out

    # numpy path
    eps = 1e-9
    a = _to_numpy(mA).astype(np.float32, copy=False)
    b = _to_numpy(mB).astype(np.float32, copy=False)
    K = a.shape[-1] - 1
    a_s, a_o = a[..., :K], a[..., [K]]
    b_s, b_o = b[..., :K], b[..., [K]]
    same = a_s * b_s
    cross = a_s * b_o + a_o * b_s
    num_s = same + cross
    num_o = a_o * b_o
    tot = np.sum(a_s, axis=-1, keepdims=True) * np.sum(b_s, axis=-1, keepdims=True)
    diag = np.sum(a_s * b_s, axis=-1, keepdims=True)
    kappa_raw = np.clip(tot - diag, 0.0, None)
    kappa = np.clip(kappa_raw, None, 1.0 - eps)
    denom = np.clip(1.0 - kappa, eps, None)
    mC_s = num_s / denom
    mC_o = num_o / denom
    out = np.concatenate([mC_s, mC_o], axis=-1)
    out = out / np.clip(out.sum(axis=-1, keepdims=True), eps, None)
    return _to_same_backend(out, mA)


def ds_combine_many(m_list: Sequence[ArrayLike]) -> ArrayLike:
    """Left-fold pairwise combination over a sequence of masses."""
    if not m_list:
        raise ValueError("m_list must be non-empty")
    acc = m_list[0]
    for m in m_list[1:]:
        acc = ds_combine_pair(acc, m)
    return acc


def _as_2d_numpy(X) -> np.ndarray:
    arr = np.asarray(X)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr


def _format_explainer_value(value: Any) -> str:
    return format_literal_value(value)


def _format_explainer_value_for_feature(feature: str, value: Any, value_names: Optional[dict] = None) -> str:
    mapping = (value_names or {}).get(str(feature)) or {}
    if mapping:
        if value in mapping:
            return str(mapping[value])
        try:
            iv = int(value)
            if iv in mapping:
                return str(mapping[iv])
        except Exception:
            pass
        sv = str(value)
        if sv in mapping:
            return str(mapping[sv])
    return _format_explainer_value(value)


def build_merged_rule_explanation(*, selected_rules, beta: float = 1.0, eps: float = 1e-8, value_names: Optional[dict] = None):
    grouped: dict[str, dict[str, Any]] = {}
    for row in selected_rules:
        rule_id = int(row.get("rule_id", -1))
        omega = float(row.get("omega", 1.0))
        rule_score = float(row.get("_score", 0.0))
        contributor_strength = max(0.0, min(1.0, rule_score * (1.0 - omega)))
        for spec in list(row.get("specs") or []):
            if not isinstance(spec, (list, tuple)) or len(spec) != 3:
                continue
            feature, op, raw_value = str(spec[0]), str(spec[1]), normalise_rule_value(spec[2])
            state = grouped.setdefault(feature, {"gt": {}, "lt": {}, "eq": {}})
            contributor = {
                "value": omega,
                "rule_score": rule_score,
                "strength": contributor_strength,
                "text": f"R{rule_id} :: {feature} {op} {_format_explainer_value_for_feature(feature, raw_value, value_names)}",
            }
            state[{"<": "lt", ">": "gt", "==": "eq"}[op]].setdefault(raw_value, []).append(contributor)

    literals = []
    for feature, state in grouped.items():
        contradiction = False
        gt_value = lt_value = eq_value = None
        contributor_rows = []
        if state["eq"]:
            eq_keys = sorted(state["eq"].keys(), key=lambda value: repr(value))
            if len(eq_keys) > 1:
                contradiction = True
                for value in eq_keys:
                    contributor_rows.extend(state["eq"][value])
            else:
                eq_value = eq_keys[0]
                contributor_rows.extend(state["eq"][eq_value])
                for value, rows in state["gt"].items():
                    contradiction = contradiction or not (float(eq_value) > float(value))
                    if float(eq_value) > float(value):
                        contributor_rows.extend(rows)
                for value, rows in state["lt"].items():
                    contradiction = contradiction or not (float(eq_value) < float(value))
                    if float(eq_value) < float(value):
                        contributor_rows.extend(rows)
        else:
            gt_keys = sorted(state["gt"].keys(), key=float)
            lt_keys = sorted(state["lt"].keys(), key=float)
            if gt_keys:
                gt_value = max(gt_keys, key=float)
                for value, rows in state["gt"].items():
                    if float(value) <= float(gt_value):
                        contributor_rows.extend(rows)
            if lt_keys:
                lt_value = min(lt_keys, key=float)
                for value, rows in state["lt"].items():
                    if float(value) >= float(lt_value):
                        contributor_rows.extend(rows)
            contradiction = gt_value is not None and lt_value is not None and float(gt_value) >= float(lt_value)

        certainty = 0.0
        if contributor_rows:
            strengths = [max(0.0, min(1.0, float(row.get("strength", 0.0)))) for row in contributor_rows]
            if strengths:
                certainty = 1.0 - float(np.prod([1.0 - s for s in strengths], dtype=np.float64))
        expr = f"CONTRADICTION({feature})" if contradiction else str(feature)
        if not contradiction:
            if eq_value is not None:
                expr = f"{feature} == {_format_explainer_value_for_feature(feature, eq_value, value_names)}"
            elif gt_value is not None and lt_value is not None:
                expr = (
                    f"{_format_explainer_value_for_feature(feature, gt_value, value_names)}"
                    f" < {feature} < "
                    f"{_format_explainer_value_for_feature(feature, lt_value, value_names)}"
                )
            elif gt_value is not None:
                expr = f"{feature} > {_format_explainer_value_for_feature(feature, gt_value, value_names)}"
            elif lt_value is not None:
                expr = f"{feature} < {_format_explainer_value_for_feature(feature, lt_value, value_names)}"
        literals.append(
            {
                "expression": expr,
                "contributors": [str(row["text"]) for row in contributor_rows],
                "_certainty": certainty,
                "confidence": 0.0,
            }
        )

    if literals:
        raw_scores = np.asarray([max(float(row["_certainty"]), eps) ** float(beta) for row in literals], dtype=np.float64)
        weights = raw_scores / max(float(raw_scores.sum()), eps)
        for row, weight in zip(literals, weights.tolist()):
            row["confidence"] = round(float(weight), 6)
            row.pop("_certainty", None)
        literals.sort(key=lambda row: float(row.get("confidence", 0.0)), reverse=True)

    summary = " + ".join(f"{float(row['confidence']):.3f}*[{row['expression']}]" for row in literals) or "none"
    return {"literals": literals, "summary": summary}


def build_rule_contributions(model, activated, *, pred: int, mode: str):
    contributions = []
    k_eff = max(2, int(model.num_classes))
    for item in activated:
        rid = int(item["id"])
        mass = np.asarray(item.get("mass") or [], dtype=float)
        label = model.rules[rid].get("label")
        omega = float(mass[-1]) if mass.size else 1.0
        if mode == "vote":
            stats = model.rules[rid].get("stats") or {}
            support = float(stats.get("support", 0.0))
            neg_covered = float(stats.get("neg_covered", 0.0))
            quality = (support + 1.0) / (support + neg_covered + k_eff)
            magnitude = float(quality * math.log1p(max(0.0, support)))
            weight = magnitude if (label is not None and int(label) == pred) else -magnitude
        else:
            pred_support = float(mass[pred]) if mass.size and 0 <= pred < (mass.size - 1) else 0.0
            oppose_support = max((float(value) for idx, value in enumerate(mass[:-1]) if idx != pred), default=0.0)
            weight = pred_support if pred_support >= oppose_support else -oppose_support
        contributions.append({"rule_id": rid, "weight": weight, "rule_class": label, "omega": omega})

    denom = sum(abs(float(row.get("weight", 0.0))) for row in contributions) or 1.0
    for row in contributions:
        row["weight"] = float(row.get("weight", 0.0)) / denom
    contributions.sort(key=lambda row: abs(float(row.get("weight", 0.0))), reverse=True)
    return contributions


def prepare_rules_for_export(model, x):
    if not model.rules or model.rule_mass_params is None:
        return {"activated_rules": []}
    X = _as_2d_numpy(x)
    if X.shape[0] != 1:
        X = X[:1]
    X_t = model._prepare_numeric_tensor(X)
    with torch.no_grad():
        act = model._to_numpy(model._activation_matrix(X_t)[0], bool_cast=True)
        masses = model._to_numpy(model.get_rule_masses())

    activated_rules = []
    for rid in np.flatnonzero(act):
        rid_i = int(rid)
        rule = model.rules[rid_i]
        activated_rules.append(
            {
                "id": rid_i,
                "condition": rule.get("caption") or condition_caption(rule.get("specs", ()) or (), model.value_names),
                "class": rule.get("label"),
                "specs": copy.deepcopy(rule.get("specs") or ()),
                "mass": np.asarray(masses[rid_i], dtype=float).tolist(),
                "stats": copy.deepcopy(rule.get("stats") or {}),
            }
        )
    return {"activated_rules": activated_rules}


def get_combined_rule(
    model,
    x,
    *,
    return_details: bool = True,
    combination_rule: Optional[str] = None,
    decision_class: Optional[int] = None,
    include_merged_rule: bool = False,
    merged_rule_beta: float = 1.0,
):
    X = _as_2d_numpy(x)
    if X.shape[0] != 1:
        X = X[:1]
    rule_in = str(combination_rule or model.combination_rule).strip().lower()
    activated = prepare_rules_for_export(model, X[0]).get("activated_rules", [])

    if rule_in == "vote":
        proba = np.asarray(model.predict_rule_baseline_proba(X, method="weighted_vote"), dtype=float)
        pred = int(decision_class) if decision_class is not None else (int(np.argmax(proba[0])) if proba.size else 0)
        contributions = build_rule_contributions(model, activated, pred=pred, mode="vote")
        fused_mass = None
    else:
        with torch.no_grad():
            probs = model._to_numpy(model.forward(X, combination_rule=rule_in))
            fused = np.asarray(model.predict_masses(X, combination_rule=rule_in), dtype=float)
        pred = int(decision_class) if decision_class is not None else (int(np.argmax(probs[0])) if probs.size else 0)
        contributions = build_rule_contributions(model, activated, pred=pred, mode="mass")
        fused_mass = np.asarray(fused[0], dtype=float).tolist() if fused.ndim == 2 and fused.shape[0] == 1 else None

    mass_by_rule = {int(item["id"]): np.asarray(item.get("mass") or [], dtype=float).tolist() for item in activated}
    total = sum(abs(float(item.get("weight", 0.0))) for item in contributions) or 1.0
    detailed_rules = []
    combined_specs = []
    for item in contributions:
        rid = int(item["rule_id"])
        if float(item.get("weight", 0.0)) > 0.0:
            combined_specs.extend(model.rules[rid].get("specs") or ())
        detailed_rules.append(
            {
                "rule_id": rid,
                "rule_class": item.get("rule_class"),
                "caption": model.rules[rid].get("caption", ""),
                "omega": float(item.get("omega", 1.0)),
                "specs": list(model.rules[rid].get("specs") or []),
                "mass_vector": list(mass_by_rule.get(rid, [])),
                "_score": abs(float(item.get("weight", 0.0))) / total,
            }
        )
    detailed_rules.sort(key=lambda row: float(row["_score"]), reverse=True)
    if not combined_specs and detailed_rules:
        anchor = detailed_rules[0]
        rid = int(anchor["rule_id"])
        combined_specs.extend(model.rules[rid].get("specs") or ())

    out = {
        "decision_class": int(pred),
        "n_rules_fired": int(len(activated)),
        "combined_condition": " AND ".join(
            f"{name} {op} {_format_explainer_value_for_feature(str(name), normalise_rule_value(value), getattr(model, 'value_names', None))}"
            for name, op, value in combined_specs
        ),
        "rules_pool": [
            {
                "rule_id": int(row["rule_id"]),
                "rule_class": None if row.get("rule_class") is None else int(row["rule_class"]),
                "rule_text": str(row.get("caption", "")),
                "mass_vector": [round(float(value), 6) for value in list(row.get("mass_vector") or [])],
            }
            for row in detailed_rules
        ],
        "combined_summary_literals": [],
        "combined_summary": "none",
    }
    if fused_mass is not None:
        out["fused_mass"] = list(fused_mass)
    if detailed_rules:
        merged_rows = []
        for row in detailed_rules:
            merged_rows.append(dict(row))
        merged = build_merged_rule_explanation(
            selected_rules=merged_rows,
            beta=float(merged_rule_beta),
            value_names=getattr(model, "value_names", None),
        )
        out["combined_summary_literals"] = list(merged.get("literals", []))
        out["combined_summary"] = str(merged.get("summary", "none"))
    return out if return_details else out.get("combined_condition", "")
