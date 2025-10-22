# -*- coding: utf-8 -*-
"""
Core DST (Dempster–Shafer Theory) utilities used by DSModelMultiQ.

Exposed API (imported by the model/code elsewhere):
  - params_to_mass(W): convert rule parameters to normalized masses over K classes + Omega
  - ds_combine_pair(mA, mB): Dempster's rule combination for two mass assignments
  - ds_combine_many(m_list): left-fold combination over a sequence of masses
  - masses_to_pignistic(m): convert masses to pignistic (betP) probabilities
  - dempster_rule_kt(m1, m2, return_conflict=False): numpy version of pairwise combination
  - unique_rules(rules): deduplicate rules by caption (helper)

All functions accept and return either numpy arrays or torch tensors where sensible.
If torch is unavailable, numpy code paths are used.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Union, overload
import math
import numpy as np

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

ArrayLike = Union[np.ndarray, 'Tensor']


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
    """Alias kept for readability at call sites that deal with logits explicitly."""
    return params_to_mass(W)


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


# ----------------------------- Dempster's Rule (pair) -----------------------------
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
    """
    if _HAS_TORCH and isinstance(mA, torch.Tensor):
        K = mA.shape[-1] - 1
        a_s, a_o = mA[..., :K], mA[..., K:K+1]
        b_s, b_o = mB[..., :K], mB[..., K:K+1]
        # contributions
        same = a_s * b_s
        cross = a_s * b_o + a_o * b_s
        num_s = same + cross
        num_o = a_o * b_o
        # conflict
        # (sum over all pairs i!=j): we can compute total outer sum minus diagonal
        tot = torch.sum(a_s, dim=-1, keepdim=True) * torch.sum(b_s, dim=-1, keepdim=True)
        diag = torch.sum(a_s * b_s, dim=-1, keepdim=True)
        kappa = torch.clamp(tot - diag, min=0.0)
        denom = (1.0 - kappa).clamp_min(1e-12)
        mC_s = num_s / denom
        mC_o = num_o / denom
        return torch.cat([mC_s, mC_o], dim=-1)

    # numpy path
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
    kappa = np.clip(tot - diag, 0.0, None)
    denom = (1.0 - kappa)
    denom = np.where(denom <= 1e-12, 1e-12, denom)
    mC_s = num_s / denom
    mC_o = num_o / denom
    out = np.concatenate([mC_s, mC_o], axis=-1)
    return _to_same_backend(out, mA)


def ds_combine_many(m_list: Sequence[ArrayLike]) -> ArrayLike:
    """Left-fold pairwise combination over a sequence of masses."""
    if not m_list:
        raise ValueError("m_list must be non-empty")
    acc = m_list[0]
    for m in m_list[1:]:
        acc = ds_combine_pair(acc, m)
    return acc


# ----------------------------- Numpy-friendly variant -----------------------------
def dempster_rule_kt(m1: np.ndarray, m2: np.ndarray, return_conflict: bool = False):
    """
    Historical numpy variant of Dempster's rule kept for backward-compat.
    Inputs: 1D masses of length K+1 (singletons + Omega).
    Returns: (m12, kappa) if return_conflict else m12.
    """
    m1 = np.asarray(m1, dtype=np.float32)
    m2 = np.asarray(m2, dtype=np.float32)
    K = m1.shape[-1] - 1
    a_s, a_o = m1[:K], m1[K]
    b_s, b_o = m2[:K], m2[K]
    same = a_s * b_s
    cross = a_s * b_o + a_o * b_s
    num_s = same + cross
    num_o = a_o * b_o
    tot = np.sum(a_s) * np.sum(b_s)
    diag = np.sum(a_s * b_s)
    kappa = max(0.0, tot - diag)
    denom = max(1e-12, 1.0 - kappa)
    out = np.concatenate([num_s / denom, np.array([num_o / denom], dtype=np.float32)])
    return (out, kappa) if return_conflict else out


# ----------------------------- Misc helpers -----------------------------
def unique_rules(rules: Sequence) -> List:
    """Deduplicate rule objects by their .caption (case-insensitive)."""
    seen = {}
    for r in rules:
        key = str(getattr(r, "caption", "")).strip().lower()
        if key and key not in seen:
            seen[key] = r
    return list(seen.values())
