# -*- coding: utf-8 -*-
"""
Dempster-Shafer Theory (DST) Model with Multi-Q Support.

This module implements a DST-based classification model that uses rule-based reasoning
with mass functions to handle uncertainty. It supports multiple rule generation algorithms
(STATIC, RIPPER, FOIL) and provides DSGD++ initialization for confidence-based mass assignment.

Classes:
    DSModelMultiQ: The core PyTorch module implementing the DST logic.
"""

import copy
from pathlib import Path
from typing import Any, Dict, Optional, List, Union, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from rule_generator import RuleGenerator
from core import params_to_mass, ds_combine_pair, masses_to_pignistic as core_masses_to_pignistic
from device_utils import resolve_torch_device


def _lazy_pickle():
    """Lazy import for dill/pickle."""
    import pickle
    try:
        import dill
        return dill
    except ImportError:
        return pickle


def _as_numpy(X):
    """Ensure input is a numpy array, handling scalars."""
    arr = np.asarray(X)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr


def masses_to_pignistic(masses):
    """Convert mass functions to pignistic probabilities via normalization."""
    return core_masses_to_pignistic(masses)


class DSModelMultiQ(nn.Module):
    """
    Dempster-Shafer Theory (DST) Multi-Query Model for classification with uncertainty.

    This model treats classification rules as independent sources of evidence.
    Each rule provides a "mass" assignment over the K classes and an "uncertainty" (Omega) mass.
    These masses are combined using Dempster's Rule of Combination (or strictly defined alternatives)
    to produce a final belief state.

    Key Features:
    - **Rule-based**: Uses logical rules (STATIC, RIPPER, FOIL) as Evidence.
    - **Uncertainty**: Explicitly models "I don't know" (Omega) vs "Class X".
    - **Differentiable**: Mass parameters are learnable via PGD.
    """
    # Operator codes and logic
    OP_EQ, OP_LT, OP_GT = 0, 1, 2
    OPS = {'==': OP_EQ, '<': OP_LT, '>': OP_GT}

    def __init__(
        self,
        k: int,
        algo: str = "STATIC",
        device: str = "auto",
        feature_names=None,
        value_decoders=None,
        rule_uncertainty=0.8,
        combination_rule: str = "dempster",
    ):
        """
        Initialize the DST model.

        Args:
            k (int): Number of classes.
            algo (str): Rule generation algorithm ('STATIC', 'RIPPER', 'FOIL').
            device (str): Device to run on ('auto', 'cpu', 'cuda', 'cuda:0', 'mps'). 'metal' is accepted as an alias for 'mps'.
            feature_names (list): List of feature names for interpretability.
            value_decoders (dict): Dictionary mapping feature names to value decoders.
            rule_uncertainty (float): Initial uncertainty mass for rules (0.0 to 1.0).
            combination_rule (str): 'dempster', 'yager', or 'vote'.
        """
        super().__init__()
        self.num_classes, self.k = int(k), int(k)
        self.algo = str(algo).upper()
        self.device = resolve_torch_device(device)
        self.feature_names = list(feature_names) if feature_names else None
        self.value_names = value_decoders or {}
        self.rule_uncertainty = float(rule_uncertainty)

        self.combination_rule = self._normalize_combination_rule(combination_rule)
        if self.combination_rule not in {"dempster", "yager", "vote"}:
            raise ValueError("combination_rule must be 'dempster', 'yager', or 'vote'")
        self.combination_weight_key = "precision"  # used only for explanation weighting

        self.class_prior = None
        self._encoders = {}
        self.rules, self._rule_signatures = [], set()

        # Cache for literals
        self._literal_to_index = {}
        self._literals = []
        self._rule_literal_indices = []
        
        # Optimized Tensors for evaluation
        self._lit2rule = None
        self._rule_len = None
        self._lit_feat_idx = None
        self._lit_op_code = None
        self._lit_value = None

        self.rule_mass_params = None
        self.initial_rule_masses = None
        self._feature_to_idx = {n: i for i, n in enumerate(self.feature_names)} if self.feature_names else {}
        self.to(self.device)

    # ----------------------------- Helpers -----------------------------
    @staticmethod
    def _normalize_combination_rule(rule: Optional[str]) -> str:
        """Normalize combination rule string (dempster, yager, vote)."""
        r = str(rule or "dempster").strip().lower().replace("-", "_")
        if r in {"dempster", "yager", "vote"}:
            return r
        return "dempster"

    def _resolve_feature_index(self, name: str, *, num_features: Optional[int] = None) -> int:
        """Resolve feature name to column index, supporting X[i] and named features."""
        idx = self._feature_to_idx.get(name)
        if idx is None and name.startswith("X[") and name.endswith("]") and name[2:-1].isdigit():
            idx = int(name[2:-1])
        if idx is None or (num_features is not None and not (0 <= idx < num_features)):
            raise KeyError(f"Unknown feature '{name}'")
        return int(idx)

    def _prior_tensor(self, k: int) -> torch.Tensor:
        """Get the class prior tensor (1, K) for mass redistribution."""
        if self.class_prior is not None and self.class_prior.sum() > 0:
            p = self.class_prior.to(self.device)
            return (p / p.sum().clamp_min(1e-12)).reshape(1, -1)
        return torch.full((1, k), 1.0 / max(1, k), device=self.device)


    def _final_conf_from_mass(self, mass: np.ndarray) -> Dict[str, Any]:
        """
        Extract detailed confidence metrics (betp, omega, certainty) from a mass vector.
        
        Args:
            mass: Mass vector of shape (K+1,)
            
        Returns:
            Dict containing 'betp', 'omega', 'certainty', 'top1', 'sep', 'final_conf'.
        """
        m = np.asarray(mass, dtype=float).reshape(-1)
        k = len(m) - 1
        if k < 1: return {"betp": np.ones(1), "omega": 1.0, "final_conf": 0.0, "top1": 0}

        omega = float(np.clip(m[-1], 0.0, 1.0))
        certainty = 1.0 - omega
        
        prior = (self.class_prior.cpu().numpy() if self.class_prior is not None else np.full(k, 1/k))
        prior = prior / max(prior.sum(), 1e-12)
        
        # Pignistic transform: m(A) + m(Omega) * |A|/|Omega| -> here |A|=1
        betp = m[:k] + omega * prior
        betp /= max(betp.sum(), 1e-12)
        
        top1 = int(np.argmax(betp)) if betp.size else 0
        sep = (betp[top1] - np.sort(betp)[-2]) if k >= 2 else (betp[top1] if k == 1 else 0.0)

        return {
            "betp": betp, "omega": omega, "certainty": certainty,
            "top1": top1, "sep": float(sep), "final_conf": float(certainty * max(sep, 0.0))
        }

    # ----------------------------- Encoders -----------------------------
    def _fit_encoders(self, X_raw):
        """Fit feature encoders for categorical columns."""
        self.feature_names = self.feature_names or [f"X[{i}]" for i in range(X_raw.shape[1])]
        self._feature_to_idx = {n: i for i, n in enumerate(self.feature_names)}
        self._encoders.clear()

        for j, name in enumerate(self.feature_names):
            col = X_raw[:, j]
            dec = self.value_names.get(name) or {}
            is_numeric = np.issubdtype(col.dtype, np.number) and col.dtype.kind != "O"

            full_dec = {}
            if dec:
                for k, v in dec.items():
                    full_dec[str(k)] = v
                    if isinstance(k, (int, float)): full_dec[k] = v
                    try: full_dec[float(k)] = v
                    except: pass
                self.value_names[name] = full_dec
            
            if not dec and is_numeric: continue

            enc = {}
            if dec:
                for uv in np.unique(col):
                    for candid in [uv, str(uv)]:
                        if candid in full_dec:
                             try: enc[uv] = float(candid)
                             except: enc[uv] = -1.0
                             break
                    if uv not in enc and not is_numeric: enc[uv] = float(len(enc))
                self._encoders[name] = enc
            else:
                u = np.unique(col)
                self._encoders[name] = {v: float(i) for i, v in enumerate(u)}

    def _encode_X(self, X_raw):
        """Encode raw input X into float32 suitable for PyTorch."""
        X_num = np.empty(X_raw.shape, dtype=np.float32)
        for j, name in enumerate(self.feature_names or []):
            col = X_raw[:, j]
            enc = self._encoders.get(name)
            if enc:
                X_num[:, j] = [enc.get(v, -1.0) for v in col]
            else:
                try: X_num[:, j] = col.astype(np.float32)
                except: X_num[:, j] = np.nan
        return X_num

    def _encode_literal_value(self, name, op, value):
        """Encode a single literal value for comparison logic."""
        if op == "==" and name in self._encoders:
            return float(self._encoders[name].get(value, -1.0))
        try: return float(value)
        except: return float("nan")

    # ----------------------------- Rule generation -----------------------------
    def generate_rules(self, X, y=None, feature_names=None, *, algo=None, verbose=False, **params):
        """
        Generate rules from data using the specified algorithm (STATIC, RIPPER, FOIL).

        Args:
            X: Input features.
            y: Target labels.
            feature_names: Optional feature names.
            algo: Algorithm name.
            verbose: Print verbose output.
            **params: Additional parameters for RuleGenerator.
        """
        X_raw = _as_numpy(X)
        y_np = np.asarray(y, dtype=int) if y is not None else None
        if feature_names: self.feature_names = list(feature_names)
        self.feature_names = self.feature_names or [f"X[{i}]" for i in range(X_raw.shape[1])]
        self.algo = str(algo or self.algo).upper()

        gen = RuleGenerator(algo=self.algo.lower(), verbose=verbose, **params)
        self.rules.clear(); self._rule_signatures.clear()

        if self.algo == "STATIC":
            for r in gen.build_static_rules(X_raw, feature_names=self.feature_names, y=y_np, value_decoders=self.value_names, verbose=verbose, **params):
                self._add_rule_struct(r)
        elif self.algo in {"RIPPER", "FOIL"}:
            if y_np is None: raise ValueError("Supervised algo needs y")
            gen.fit(X_raw, y_np, feature_names=self.feature_names)
            for label, cond, stats in gen.ordered_rules:
                self._add_rule_struct({"specs": tuple((n, o, v) for n, (o, v) in sorted(cond.items())), "label": int(label), "stats": stats})
        else:
            raise ValueError(f"Unsupported algo {self.algo}")

        # Compute empirical class prior if needed
        if y_np is not None:
            counts = np.bincount(y_np, minlength=self.num_classes)
            self.class_prior = torch.tensor(1.0 / np.maximum(counts, 1.0), device=self.device, dtype=torch.float32)
            self.class_prior /= self.class_prior.sum()

        self._fit_encoders(X_raw)
        self._rebuild_literal_cache()
        self._build_head()
        self.reset_masses()

    def _add_rule_struct(self, r):
        """Internal helper to add a rule structure, ensuring uniqueness."""
        cond = self._canonicalize_condition(r.get("specs", ()))
        if not cond: return
        sig = self._condition_signature(cond)
        if sig in self._rule_signatures: return
        self._rule_signatures.add(sig)
        
        self.rules.append({
            "specs": cond,
            "label": r.get("label"),
            "caption": r.get("caption") or self._condition_caption(cond),
            "stats": r.get("stats")
        })

    # ----------------------------- Condition Logic -----------------------------
    @staticmethod
    def _normalise_val(v):
        """Normalize numeric values (e.g. 5.0 -> 5) to avoid float mismatch."""
        try:
            f = float(v)
            return float(round(f)) if abs(f - round(f)) < 1e-9 else f
        except: return v

    def _canonicalize_condition(self, literals: tuple) -> tuple:
        """
        Canonicalize a condition by merging constraints on the same feature.
        
        Returns:
            Tuple of literals or None if the condition is contradictory.
        """
        if not literals: return ()
        scope = {}
        for name, op, v in literals:
            op_code = self.OPS.get(op)
            if op_code is None: continue
            val = self._normalise_val(v)
            if name not in scope: scope[name] = [None, None, None] # eq, lt, gt
            
            s = scope[name]
            if op_code == self.OP_EQ:
                if s[0] is not None and s[0] != val: return None # Conflict
                s[0] = val
            elif op_code == self.OP_LT:
                s[1] = min(s[1], float(val)) if s[1] is not None else float(val)
            elif op_code == self.OP_GT:
                s[2] = max(s[2], float(val)) if s[2] is not None else float(val)

        res = []
        for name, (eq, lt, gt) in sorted(scope.items()):
            if eq is not None:
                if (lt is not None and eq >= lt) or (gt is not None and eq <= gt): return None
                res.append((name, "==", eq))
            else:
                if lt is not None and gt is not None and gt >= lt: return None
                if gt is not None: res.append((name, ">", gt))
                if lt is not None: res.append((name, "<", lt))
        return tuple(res)

    def _condition_signature(self, cond):
        """Generate a unique string signature for a condition."""
        return "&&".join(f"{n}{o}{self._normalise_val(v)}" for n, o, v in cond)

    def _condition_caption(self, cond):
        """Generate a human-readable caption for a condition."""
        def _fmt(n, o, v):
            d = (self.value_names.get(n) or {}).get(v, v)
            val = f"{float(d):.4g}" if isinstance(d, float) else d
            return f"{n} {o} {val}"
        return " & ".join(_fmt(*t) for t in cond) or "<empty>"

    # ----------------------------- Literal cache + params -----------------------------
    def _rebuild_literal_cache(self):
        """
        Rebuild internal optimized caches for fast vector evaluation of rules.
        
        Creates matrices mapping rules to literals and literals to feature indices depending on OPS.
        """
        self._literal_to_index, self._rule_literal_indices = {}, []
        for r in self.rules:
            idxs = [self._literal_to_index.setdefault(l, len(self._literal_to_index)) for l in r.get("specs", ())]
            self._rule_literal_indices.append(idxs)

        self._literals = list(self._literal_to_index.keys())
        if not self._literals:
            self._lit2rule = self._rule_len = self._lit_feat_idx = self._lit_op_code = self._lit_value = None
            return

        self._lit2rule = torch.zeros((len(self._literals), len(self.rules)), dtype=torch.float32, device=self.device)
        for rid, idxs in enumerate(self._rule_literal_indices):
            if idxs: self._lit2rule[idxs, rid] = 1.0
        self._rule_len = torch.tensor([len(i) for i in self._rule_literal_indices], dtype=torch.float32, device=self.device)

        feat_idx, op_code, values = [], [], []
        for name, op, val in self._literals:
            feat_idx.append(self._resolve_feature_index(name))
            op_code.append(self.OPS.get(op, 0))
            values.append(self._encode_literal_value(name, op, val))

        self._lit_feat_idx = torch.tensor(feat_idx, device=self.device, dtype=torch.long)
        self._lit_op_code = torch.tensor(op_code, device=self.device, dtype=torch.long)
        self._lit_value = torch.tensor(values, device=self.device, dtype=torch.float32)

    def _build_head(self):
        """Initialize the learnable mass parameters."""
        self.initial_rule_masses = None
        if not self.rules:
            self.rule_mass_params = None
            return
        self.rule_mass_params = nn.Parameter(
            torch.zeros(len(self.rules), self.num_classes + 1, device=self.device, dtype=torch.float32),
            requires_grad=True
        )

    # ----------------------------- Evaluation -----------------------------
    def _eval_literals(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate all unique literals against input batch X (vectorized)."""
        if not self._literals:
            return torch.zeros((X.shape[0], 0), dtype=torch.bool, device=self.device)

        if self._lit_feat_idx is not None:
            cols = X[:, self._lit_feat_idx]
            vals = self._lit_value.view(1, -1)
            eq = torch.isclose(cols, vals, atol=1e-5)
            # Stack [eq, lt, gt] -> gather by op code (OP_EQ=0, OP_LT=1, OP_GT=2)
            # ops tensor has indices 0, 1, 2. gather(2, ops) selects from last dim of stack
            stack = torch.stack([eq, cols < vals, cols > vals], dim=2)
            ops = self._lit_op_code.view(1, -1, 1).expand(X.shape[0], -1, 1)
            return stack.gather(2, ops).squeeze(2)
        
        return torch.zeros((X.shape[0], len(self._literals)), dtype=torch.bool, device=self.device)

    def _activation_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the binary activation matrix (N, num_rules) for the batch."""
        L = self._eval_literals(X)
        if self._lit2rule is None:
            return torch.zeros((X.shape[0], len(self.rules)), dtype=torch.bool, device=self.device)
        return (L.float() @ self._lit2rule).eq(self._rule_len)

    def _combine_vote_masses(
        self,
        act_flat: torch.Tensor,
        k_plus_one: int,
        dtype=None,
    ) -> torch.Tensor:
        """Combine active rules via simple majority vote into mass assignments."""
        device = act_flat.device
        dtype = dtype or torch.float32
        labels = torch.tensor(
            [r.get("label", -1) if r.get("label") is not None else -1 for r in self.rules],
            device=device,
            dtype=torch.long,
        )
        
        scores = torch.zeros((act_flat.shape[0], k_plus_one - 1), device=device, dtype=dtype)
        act_f = act_flat.float()
        for cls in range(scores.shape[1]):
            mask = labels == cls
            if mask.any():
                scores[:, cls] = act_f[:, mask].sum(dim=1)

        denom = scores.sum(dim=1, keepdim=True)
        combined = torch.zeros((act_flat.shape[0], k_plus_one), device=device, dtype=dtype)
        has_votes = denom.squeeze(1) > 0
        combined[has_votes, : k_plus_one - 1] = scores[has_votes] / denom[has_votes]
        combined[~has_votes, -1] = 1.0
        return combined

    @staticmethod
    def _combine_yager_pair(mA: torch.Tensor, mB: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
        """Vectorized Yager-style pairwise combination (singletons+Omega)."""
        k = mA.shape[-1] - 1
        a_s, a_o = mA[..., :k], mA[..., -1:]
        b_s, b_o = mB[..., :k], mB[..., -1:]

        num_s = a_s * b_s + a_s * b_o + a_o * b_s
        num_o = a_o * b_o

        tot = a_s.sum(dim=-1, keepdim=True) * b_s.sum(dim=-1, keepdim=True)
        diag = (a_s * b_s).sum(dim=-1, keepdim=True)
        kappa = (tot - diag).clamp(min=0.0)

        out = torch.cat([num_s, num_o + kappa], dim=-1)
        out = out / out.sum(dim=-1, keepdim=True).clamp_min(eps)
        return out

    def _combine_active_masses_iterative(
        self,
        *,
        act_flat: torch.Tensor,
        masses: torch.Tensor,
        vac: torch.Tensor,
        rule: str,
    ) -> torch.Tensor:
        """Iterative (masked) fusion over the union of active rules in a batch."""
        n = int(act_flat.shape[0])
        k_plus_one = int(masses.shape[-1])
        acc = vac.expand(n, -1).clone()

        active_rids = act_flat.any(dim=0).nonzero(as_tuple=False).flatten()
        if active_rids.numel() == 0:
            return acc

        if rule == "dempster":
            for rid in active_rids:
                combined = ds_combine_pair(acc, masses[rid].view(1, k_plus_one))
                acc = torch.where(act_flat[:, rid].unsqueeze(1), combined, acc)
            return acc

        if rule == "yager":
            for rid in active_rids:
                combined = self._combine_yager_pair(acc, masses[rid].view(1, k_plus_one))
                acc = torch.where(act_flat[:, rid].unsqueeze(1), combined, acc)
            return acc

        raise ValueError(f"Unknown combination rule '{rule}'")

    def _combine_active_masses_tree(
        self,
        *,
        act_flat: torch.Tensor,
        masses: torch.Tensor,
        vac: torch.Tensor,
        rule: str,
        max_elements: int = 15_000_000,
    ) -> torch.Tensor:
        """Tree-reduction fusion (parallel over rules) for the active-rule union of a batch."""
        n = int(act_flat.shape[0])
        k_plus_one = int(masses.shape[-1])

        active_mask = act_flat.any(dim=0)
        if not bool(active_mask.any().item()):
            return vac.expand(n, -1)

        act_u = act_flat[:, active_mask]  # (N, R_act)
        masses_u = masses[active_mask]    # (R_act, K+1)
        r_act = int(act_u.shape[1])
        if r_act == 0:
            return vac.expand(n, -1)

        # If too large, fall back to the iterative implementation to avoid big allocations.
        est = int(n) * int(r_act) * int(k_plus_one)
        if est > int(max_elements) or r_act < 32:
            return self._combine_active_masses_iterative(act_flat=act_u, masses=masses_u, vac=vac, rule=rule)

        vac3 = vac.view(1, 1, k_plus_one)
        m3 = masses_u.view(1, r_act, k_plus_one).expand(n, -1, -1)
        x = torch.where(act_u.unsqueeze(-1), m3, vac3)  # (N, R_act, K+1)

        while x.shape[1] > 1:
            if x.shape[1] % 2 == 1:
                x = torch.cat([x, vac3.expand(n, 1, k_plus_one)], dim=1)
            a = x[:, 0::2, :]
            b = x[:, 1::2, :]
            if rule == "dempster":
                x = ds_combine_pair(a, b)
            elif rule == "yager":
                x = self._combine_yager_pair(a, b)
            else:
                raise ValueError(f"Unknown combination rule '{rule}'")

        return x[:, 0, :]

    def _combine_active_masses(self, masses, act, *, combination_rule=None):
        """
        Combine active rules using Dempster, Yager, or vote.
        
        Args:
            masses: Mass tensor for each rule (num_rules, K+1).
            act: Activation matrix (N, num_rules).
            combination_rule: 'dempster', 'yager', or 'vote'.
            
        Returns:
            Combined mass tensor (N, K+1).
        """
        rule = self._normalize_combination_rule(combination_rule or self.combination_rule)

        # Strict mode: during training we optimize masses under Dempster only.
        if self.training and rule != "dempster":
            raise ValueError(f"Training supports only combination_rule='dempster' (got {rule!r}).")

        batch_shape = act.shape[:-1]
        act_flat = act.reshape(-1, act.shape[-1])
        masses = masses.reshape(-1, masses.shape[-1])
        k_plus_one = masses.shape[-1]

        # Vacuous mass (pure uncertainty)
        vac = torch.zeros((1, k_plus_one), device=masses.device, dtype=masses.dtype)
        vac[0, -1] = 1.0

        if masses.numel() == 0 or act_flat.numel() == 0:
            return vac.expand(act_flat.shape[0], -1).reshape(*batch_shape, k_plus_one)

        if rule == "vote":
            combined = self._combine_vote_masses(act_flat, k_plus_one, dtype=masses.dtype)
            return combined.reshape(*batch_shape, k_plus_one)

        # Dempster/Yager fusion: associative/commutative, so we can tree-reduce across rules for speed.
        fused = self._combine_active_masses_tree(act_flat=act_flat, masses=masses, vac=vac, rule=rule)
        return fused.reshape(*batch_shape, k_plus_one)
        
        raise ValueError(f"Unknown combination rule '{rule}'")

    def _prepare_numeric_tensor(self, X):
        """Convert/Encode input data to float tensor on device (N, D)."""
        if isinstance(X, torch.Tensor): return X.to(self.device, dtype=torch.float32)
        X_raw = _as_numpy(X)
        if not self.feature_names:
            self.feature_names = [f"X[{i}]" for i in range(X_raw.shape[1])]
            self._feature_to_idx = {n: i for i, n in enumerate(self.feature_names)}
        return torch.as_tensor(self._encode_X(X_raw), dtype=torch.float32, device=self.device)

    # ----------------------------- Mass params utilities -----------------------------
    def project_rules_to_simplex(self, max_abs_logits: float = 12.0):
        """Stabilize unconstrained mass logits.

        NOTE: In v4 we learn *logits* and map them to valid masses via softmax.
        We keep the old method name for backward compatibility with training code.
        """
        if self.rule_mass_params is None:
            return
        with torch.no_grad():
            # Row-centering doesn't change softmax, but improves identifiability.
            self.rule_mass_params.data -= self.rule_mass_params.data.mean(dim=1, keepdim=True)
            # Prevent extreme logits → saturated softmax → vanishing gradients.
            if max_abs_logits is not None:
                self.rule_mass_params.data.clamp_(-float(max_abs_logits), float(max_abs_logits))

    @staticmethod
    def _masses_to_logits(masses: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Convert valid masses (rows sum to 1) to logits such that softmax(logits)=masses."""
        m = masses.clamp_min(eps)
        logits = torch.log(m)
        logits = logits - logits.mean(dim=1, keepdim=True)
        return logits

    def _rule_masses(self) -> torch.Tensor:
        """Return per-rule mass vectors (R, K+1) computed from current logits."""
        if self.rule_mass_params is None:
            raise ValueError("rule_mass_params is None; call _build_head()/reset_masses() first.")
        return torch.softmax(self.rule_mass_params, dim=1)

    def get_rule_masses(self) -> torch.Tensor:
        """Public wrapper for accessing masses (probability simplex) from learned logits."""
        return self._rule_masses()

    def forward(self, X, combination_rule=None):
        """
        Forward pass: compute class probabilities from input features.
        
        Returns:
            Probability tensor (N, K).
        """
        X = self._prepare_numeric_tensor(X)
        if not self.rules: return torch.full((X.shape[0], self.num_classes), 1.0/self.num_classes, device=self.device)
        
        masses = self._rule_masses()
        act = self._activation_matrix(X)
        combined = self._combine_active_masses(masses, act, combination_rule=combination_rule)
        return self._mass_to_prob(combined)

    def _mass_to_prob(self, masses):
        """Convert mass functions to pignistic probability distributions."""
        k = masses.shape[1] - 1
        omega = masses[:, -1:].clamp(0, 1)
        prior = self._prior_tensor(k)
        p = masses[:, :k] + omega * prior
        return p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def reset_masses(self, init_seed=None):
        """Reinitialize masses.

        Internally, we store *logits* in self.rule_mass_params and map to masses via softmax.
        We also keep a copy of the initial masses in self.initial_rule_masses (probabilities).
        """
        if self.rule_mass_params is None:
            return
        if init_seed is not None:
            torch.manual_seed(init_seed)

        base_unc = float(self.rule_uncertainty)
        num_rules = self.rule_mass_params.shape[0]
        num_classes = self.num_classes

        # Create initial masses on the simplex: random class masses + fixed omega = base_unc.
        init_masses = torch.rand(num_rules, num_classes, device=self.rule_mass_params.device, dtype=self.rule_mass_params.dtype)
        init_masses = init_masses / init_masses.sum(dim=1, keepdim=True) * (1.0 - base_unc)
        omega = torch.full((num_rules, 1), base_unc, device=self.rule_mass_params.device, dtype=self.rule_mass_params.dtype)
        init_masses = torch.cat([init_masses, omega], dim=1)  # (R, K+1)

        # Store probability-form initial masses for later use (e.g., ablations).
        self.initial_rule_masses = init_masses.detach().clone()

        # Convert to logits and load.
        logits = self._masses_to_logits(init_masses)
        self.rule_mass_params.data.copy_(logits)

        # Keep logits numerically stable.
        self.project_rules_to_simplex()

    def init_masses_dsgdpp(
        self,
        X_tensor,
        y_tensor,
        use_initial_masses: bool = False,
        *,
        # Backward-compatible knobs (kept, but the current implementation follows DSGD++'s
        # representativeness-based confidence rather than coverage-based heuristics).
        coverage_exponent: float = 0.25,  # unused (kept for API compatibility)
        laplace: float = 1.0,  # unused (kept for API compatibility)
        min_class_mass: float = 1e-4,  # unused (kept for API compatibility)
        lower_confidence_by_purity: bool = True,
        normalize_representativeness: bool = True,
        min_confidence: float = 0.05,
        max_confidence: float = 0.95,
    ):
        """Initialize masses using DSGD++ confidence (representativeness × purity).

        This follows the DSGD++ idea described in `Literature/DSGD-Enhanced`:
        - representativeness of a sample is defined via its distance to the centroid of its class;
        - rule confidence is derived from the representativeness of the samples it covers, and
          (optionally) lowered by the purity/proportion of the majority class among covered samples;
        - the mass assignment starts more confident (lower Ω) for high-confidence rules.

        The resulting masses are on the simplex and are converted to logits for training.
        """
        if self.rule_mass_params is None:
            return

        if use_initial_masses and self.initial_rule_masses is not None:
            logits = self._masses_to_logits(self.initial_rule_masses.to(self.rule_mass_params.device, self.rule_mass_params.dtype))
            self.rule_mass_params.data.copy_(logits)
            self.project_rules_to_simplex()
            return

        num_rules = int(self.rule_mass_params.shape[0])
        K = int(self.num_classes)
        if K < 1:
            return

        init_masses = torch.zeros(
            (num_rules, K + 1),
            device=self.rule_mass_params.device,
            dtype=self.rule_mass_params.dtype,
        )

        with torch.no_grad():
            X_t = self._prepare_numeric_tensor(X_tensor)  # (N, D)
            y_true = torch.as_tensor(y_tensor, device=X_t.device, dtype=torch.long).view(-1)
            if y_true.numel() != X_t.shape[0]:
                raise ValueError("X_tensor and y_tensor must have the same number of samples")

            # --- Representativeness: 1 / (1 + dist(x, centroid[y])) ---
            centroids = torch.zeros((K, X_t.shape[1]), device=X_t.device, dtype=X_t.dtype)
            for c in range(K):
                mask = y_true == c
                if mask.any():
                    centroids[c] = X_t[mask].mean(dim=0)
                else:
                    centroids[c] = X_t.mean(dim=0)

            dist = torch.norm(X_t - centroids[y_true.clamp(min=0, max=K - 1)], dim=1)
            rep = 1.0 / (1.0 + dist)
            if bool(normalize_representativeness):
                rep = rep / rep.max().clamp_min(1e-12)

            # --- Rule coverage ---
            act = self._activation_matrix(X_t)  # (N, R) bool

            min_c = float(min_confidence)
            max_c = float(max_confidence)
            min_c = max(0.0, min(min_c, 1.0))
            max_c = max(0.0, min(max_c, 1.0))
            if max_c < min_c:
                min_c, max_c = max_c, min_c

            for r in range(num_rules):
                active = act[:, r]
                if not bool(active.any().item()):
                    init_masses[r, -1] = 1.0
                    continue

                y_act = y_true[active]
                rep_act = rep[active]

                counts = torch.bincount(y_act, minlength=K).to(dtype=X_t.dtype)
                dominant = int(torch.argmax(counts).item())
                purity = float((counts[dominant] / counts.sum().clamp_min(1.0)).item())

                maj_mask = y_act == dominant
                rep_maj = rep_act[maj_mask].mean() if bool(maj_mask.any().item()) else rep_act.mean()

                confidence = rep_maj
                if bool(lower_confidence_by_purity):
                    confidence = confidence * purity
                confidence = confidence.clamp(min=min_c, max=max_c)

                # Assign confidence to dominant class, and the REST (1-confidence) to Omega.
                # Other classes get a tiny epsilon to keep logits finite.
                eps = 1e-12
                init_masses[r, :K] = eps
                init_masses[r, dominant] = confidence - (K - 1) * eps
                init_masses[r, -1] = 1.0 - confidence

        # Numerical safety: rows should already sum to 1, but keep a stable renorm anyway.
        init_masses = init_masses.clamp_min(0.0)
        init_masses = init_masses / init_masses.sum(dim=1, keepdim=True).clamp_min(1e-12)

        self.initial_rule_masses = init_masses.detach().clone()
        logits = self._masses_to_logits(init_masses)
        self.rule_mass_params.data.copy_(logits)
        self.project_rules_to_simplex()

    def renormalize_masses(self):
        """Ensure mass parameters are valid."""
        self.project_rules_to_simplex()

    # ----------------------------- Predictions & Explanation -----------------------------
    def predict_with_dst(self, X, use_initial_masses=False, combination_rule=None):
        """Predict pignistic probabilities using DST combination."""
        return masses_to_pignistic(self.predict_masses(X, use_initial_masses, combination_rule=combination_rule))

    def predict_dst_labels(self, X, use_initial_masses=False, combination_rule=None):
        """Predict class labels (argmax) using DST."""
        return self.predict_with_dst(X, use_initial_masses, combination_rule=combination_rule).argmax(axis=1)

    def predict_masses(self, X, use_initial_masses=False, *, combination_rule=None, explain=False):
        """
        Predict mass functions for input samples.
        
        Args:
            X: Input samples.
            use_initial_masses: If True, use untrained masses.
            explain: If True, returns detailed explanation dictionary.
            
        Returns:
            Mass tensor (N, K+1) or explanation dicts.
        """
        orig = self.rule_mass_params.data.clone() if use_initial_masses and self.initial_rule_masses is not None else None
        if orig is not None:
            self.rule_mass_params.data.copy_(self._masses_to_logits(self.initial_rule_masses.to(self.rule_mass_params.device, self.rule_mass_params.dtype)))

        try:
            tensor_X = self._prepare_numeric_tensor(X)
            act = self._activation_matrix(tensor_X)
            mass_rules = self._rule_masses()
            combined = self._combine_active_masses(mass_rules, act, combination_rule=combination_rule)

            if explain:
                return self._explain_predictions(tensor_X, act, mass_rules, combined, combination_rule)
            return combined.detach().cpu().numpy()
        finally:
            if orig is not None:
                self.rule_mass_params.data.copy_(orig)
    

    def _uncertainty_rule_avg(self, act: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        """Average Ω across activated rules (pre-combination)."""
        act_f = act.to(dtype=masses.dtype)
        omega = masses[:, -1].unsqueeze(0)  # (1, R)
        fired = act_f.sum(dim=1)
        denom = fired.clamp(min=1.0)
        avg = (omega * act_f).sum(dim=1) / denom
        no_fire = fired == 0
        if no_fire.any():
            avg = torch.where(no_fire, torch.ones_like(avg), avg)
        return avg

    def uncertainty_stats(self, X, *, combination_rule: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Return per-sample uncertainty: rule-average Ω and combined Ω."""
        n = _as_numpy(X).shape[0]
        if not self.rules:
            ones = np.ones(n, dtype=np.float32)
            return {"unc_rule": ones, "unc_comb": ones}

        with torch.no_grad():
            X_t = self._prepare_numeric_tensor(X)
            act = self._activation_matrix(X_t)
            masses = self._rule_masses()

            # 1. Rule-average Omega (intrinsic model confidence)
            unc_rule = self._uncertainty_rule_avg(act, masses)
            
            # 2. Combined Omega (reasoning confidence after merging)
            rule = self._normalize_combination_rule(combination_rule or self.combination_rule)
            combined = self._combine_active_masses(masses, act, combination_rule=rule)
            unc_comb = combined[..., -1]

            return {
                "unc_rule": unc_rule.detach().cpu().numpy().astype(np.float32, copy=False),
                "unc_comb": unc_comb.detach().cpu().numpy().astype(np.float32, copy=False),
            }

    def sample_uncertainty(self, X):
        """Backward-compatible uncertainty: average Ω across activated rules (pre-combination)."""
        return self.uncertainty_stats(X, combination_rule="vote")["unc_rule"]

    def predict_by_rule_vote(self, X, default_label=0):
        """Evaluate baseline: simple majority vote among activated rules."""
        X_raw = _as_numpy(X)
        if X_raw.ndim == 1:
            X_raw = X_raw.reshape(1, -1)
        if not self.rules:
            return np.full(X_raw.shape[0], int(default_label), dtype=int)
        act = self._activation_matrix(self._prepare_numeric_tensor(X_raw))
        preds = np.asarray(self.predict_dst_labels(X_raw, combination_rule="vote"), dtype=int)
        if default_label is not None:
            no_fire = (act.sum(dim=1) == 0).cpu().numpy()
            if no_fire.any():
                preds = preds.copy()
                preds[no_fire] = int(default_label)
        return preds


    def predict_by_first_rule(self, X, default_label=0):
        """Evaluate baseline: predict using first activated rule in list order."""
        if not self.rules: return np.full(_as_numpy(X).shape[0], int(default_label), dtype=int)
        act = self._activation_matrix(self._prepare_numeric_tensor(X)).cpu().numpy()
        lbls = [r.get("label") for r in self.rules]
        return np.array([int(next((lbls[i] for i, f in enumerate(m) if f and lbls[i] is not None), default_label)) for m in act], dtype=int)

    def sort_rules_by_certainty(self, *, descending: bool = True, score_mode: str = "certainty_label_mass") -> np.ndarray:
        """Reorder rules (and their masses) by a deterministic certainty score.

        This is useful for inspection/export: the most determinate rules appear first.

        Parameters
        ----------
        descending : bool
            If True, highest score first.
        score_mode : str
            - 'certainty': score = (1 - omega)
            - 'certainty_label_mass' (default): score = (1 - omega) * mass[label]
        """
        if not self.rules or self.rule_mass_params is None:
            return np.arange(len(self.rules), dtype=int)

        with torch.no_grad():
            masses = self._rule_masses().detach()  # (R, K+1)
            omega = masses[:, -1].clamp(0.0, 1.0)
            certainty = (1.0 - omega).clamp(0.0, 1.0)

            labels = torch.tensor(
                [r.get("label") if r.get("label") is not None else -1 for r in self.rules],
                device=masses.device,
                dtype=torch.long,
            )
            valid = (labels >= 0) & (labels < self.num_classes)
            label_mass = torch.zeros_like(certainty)
            if valid.any():
                rid = torch.arange(labels.numel(), device=masses.device)
                label_mass[valid] = masses[rid[valid], labels[valid]]

            mode = str(score_mode or "certainty_label_mass").strip().lower()
            if mode == "certainty":
                score = certainty
            elif mode == "certainty_label_mass":
                score = certainty * label_mass
            else:
                raise ValueError("score_mode must be 'certainty' or 'certainty_label_mass'")

            perm = torch.argsort(score, descending=bool(descending))

            # Reorder rules
            perm_list = perm.detach().cpu().numpy().tolist()
            self.rules = [self.rules[i] for i in perm_list]

            # Reorder learnable logits (rule_mass_params)
            self.rule_mass_params.data.copy_(self.rule_mass_params.data[perm])

            # Reorder stored initial masses (if present)
            if self.initial_rule_masses is not None and int(self.initial_rule_masses.shape[0]) == len(perm_list):
                self.initial_rule_masses = self.initial_rule_masses[perm].detach().clone()

            # Rebuild caches tied to rule ordering.
            self._rebuild_literal_cache()
            self._rule_signatures = {str(r.get("caption", "")).strip().lower() for r in self.rules}

            return perm.detach().cpu().numpy().astype(int, copy=False)

    def _combine_literals(self, weighted_literals):
        """Combine literal contributions into one conjunctive condition (explanation-only).

        Args:
            weighted_literals: iterable of {"literal": (name, op, value), "weight": float}.
                Only positive weights are considered.

        Returns:
            desc: human-readable string like "f1 > 3 AND f2 == 1".
            literals: list[dict] with keys {"literal", "weight"} sorted by weight desc.
        """
        if not weighted_literals:
            return "<empty>", []

        groups: dict[tuple[str, str], list[tuple[object, float]]] = {}
        for wl in weighted_literals:
            lit = wl.get("literal") if isinstance(wl, dict) else None
            if not lit or len(lit) != 3:
                continue
            name, op, val = lit
            try:
                w = float(wl.get("weight", 0.0))
            except Exception:
                w = 0.0
            if w <= 0.0:
                continue
            groups.setdefault((str(name), str(op)), []).append((val, w))

        if not groups:
            return "<empty>", []

        combined = []
        for (name, op), items in sorted(groups.items(), key=lambda kv: kv[0]):
            total_w = float(sum(w for _, w in items))

            # Choose a representative value for this (feature, op).
            if op == "==":
                candidates = {v for v, _ in items}
                best_val = max(candidates, key=lambda v: sum(w for vv, w in items if vv == v))
            else:
                fvals = []
                for v, _ in items:
                    try:
                        fvals.append(float(v))
                    except Exception:
                        continue
                if fvals:
                    best_val = max(fvals) if op == ">" else min(fvals)
                else:
                    best_val = items[0][0]

            combined.append({"literal": (name, op, best_val), "weight": total_w})

        combined.sort(key=lambda d: d["weight"], reverse=True)
        desc = " AND ".join(f"{n} {o} {v}" for (n, o, v) in [d["literal"] for d in combined]) or "<empty>"
        return desc, combined

    def _explain_predictions(self, tensor_X, act, mass_rules, combined_mass, combine_rule):
        """Generate detailed explanation for predictions including rule contributions."""
        rule = self._normalize_combination_rule(combine_rule or self.combination_rule)
        act_np = act.detach().cpu().numpy()
        combined_np = combined_mass.detach().cpu().numpy()
        masses_np = mass_rules.detach().cpu().numpy() if mass_rules is not None else None

        res = []
        for i in range(act_np.shape[0]):
            fired_ids = np.where(act_np[i])[0]
            res.append(
                self._combined_rule_summary(
                    fired_ids,
                    combined_np[i],
                    rule,
                    masses=masses_np,
                    weight_key=self.combination_weight_key,
                    return_details=True,
                )
            )
        return res[0] if len(res) == 1 else res

    def _combined_rule_summary(
        self,
        fired_ids,
        combined_mass,
        rule: str,
        *,
        masses=None,
        weight_key: str | None = None,
        return_details: bool = True,
    ) -> Dict[str, Any]:
        """Build a combined-rule explanation for a single sample.

        Notes:
            - This does NOT affect prediction; it only builds a human-friendly explanation.
            - weight_key is used only to scale contribution weights from rule stats (e.g. precision).
        """
        if rule in {"dempster", "yager"} and masses is None:
            return {"error": "Model not trained (required for DST combination explanation)"}

        mass = np.asarray(combined_mass, dtype=float).reshape(-1)
        c_info = self._final_conf_from_mass(mass)
        predicted_class = int(c_info["top1"])
        combined_omega = float(c_info["omega"])

        fired_ids = np.asarray(fired_ids, dtype=int).reshape(-1)

        base = {
            "predicted_class": predicted_class,
            "combined_omega": combined_omega,
            "final_conf": float(c_info.get("final_conf", 0.0)),
            "n_rules_fired": int(fired_ids.size),
        }

        if fired_ids.size == 0:
            if not return_details:
                return base
            return {
                **base,
                "n_agreeing": 0,
                "n_conflicting": 0,
                "combined_condition": "<no rules fired>",
                "combined_literals": [],
                "rule_contributions": [],
            }

        # Explanation weighting (stats) is optional and must never affect inference.
        wk = str(weight_key or "").strip() or None

        rule_contributions: list[dict[str, Any]] = []
        literal_contribs: list[dict[str, Any]] = []  # positive-only

        for rid in fired_ids.tolist():
            r = self.rules[int(rid)]
            rule_cls = r.get("label", None)
            rule_cls = int(rule_cls) if rule_cls is not None else -1

            # If a rule has no explicit label, treat it as agreeing for explanation-only logic.
            if rule_cls < 0:
                rule_cls = predicted_class

            agrees = (rule_cls == predicted_class)

            stats = r.get("stats") or {}
            try:
                stats_w = float(stats.get(wk, 1.0)) if wk else 1.0
            except Exception:
                stats_w = 1.0

            if rule in {"dempster", "yager"}:
                m = np.asarray(masses[int(rid)], dtype=float).reshape(-1)
                r_info = self._final_conf_from_mass(m)

                # Use mass assigned to the predicted class as the main "support" signal.
                p_mass = float(m[predicted_class]) if predicted_class < len(m) - 1 else 0.0
                if agrees:
                    weight = float(r_info["certainty"] * p_mass)
                else:
                    other_mass = float(m[rule_cls]) if 0 <= rule_cls < len(m) - 1 else 0.0
                    weight = -float(r_info["certainty"] * other_mass * 0.5)

                weight *= stats_w

                rule_contributions.append({
                    "rule_id": int(rid),
                    "caption": r.get("caption"),
                    "rule_class": int(rule_cls),
                    "agrees": bool(agrees),
                    "weight": float(weight),
                    "certainty": float(r_info.get("certainty", 0.0)),
                    "omega": float(r_info.get("omega", 1.0)),
                    "final_conf": float(r_info.get("final_conf", 0.0)),
                })
            else:
                # Simple vote explanation: +/- constant weight (optionally scaled by stats).
                weight = (1.0 if agrees else -0.5) * stats_w
                rule_contributions.append({
                    "rule_id": int(rid),
                    "caption": r.get("caption"),
                    "rule_class": int(rule_cls),
                    "agrees": bool(agrees),
                    "weight": float(weight),
                })

            if return_details and weight > 0.0:
                specs = r.get("specs", ()) or ()
                if specs:
                    w_lit = abs(float(weight)) / float(len(specs))
                    for lit in specs:
                        literal_contribs.append({"literal": lit, "weight": w_lit})

        rule_contributions.sort(key=lambda x: abs(float(x.get("weight", 0.0))), reverse=True)

        if not return_details:
            return base

        desc, combined_literals = self._combine_literals(literal_contribs)
        n_agree = sum(1 for c in rule_contributions if c.get("agrees"))
        n_conf = int(len(rule_contributions) - n_agree)

        return {
            **base,
            "n_agreeing": int(n_agree),
            "n_conflicting": int(n_conf),
            "combined_condition": desc,
            "combined_literals": combined_literals,
            "rule_contributions": rule_contributions,
        }

    def get_combined_rule(
        self,
        sample,
        return_details: bool = False,
        *,
        combination_rule: str | None = None,
        weight_key: str | None = None,
    ):
        """Explain how the model combined fired rules for a single sample.

        Args:
            sample: 1D sample features.
            return_details: if True, include rule-level contributions and combined literals.
            combination_rule: "vote" | "dempster" | "yager" (defaults to model setting).
            weight_key: optional rule stat key (e.g. "precision") used only for explanation weighting.
        """
        sample_np = _as_numpy(sample).reshape(-1)
        batch = sample_np.reshape(1, -1)

        if not self.rules:
            return {"error": "No rules available"}

        rule = self._normalize_combination_rule(combination_rule or self.combination_rule)

        # Activation for this sample
        tensor_X = self._prepare_numeric_tensor(batch)
        act_row = self._activation_matrix(tensor_X)[0].detach().cpu().numpy().astype(bool)
        fired_ids = np.where(act_row)[0]

        combined_mass = self.predict_masses(batch, combination_rule=rule)[0]

        masses_np = None
        if rule in {"dempster", "yager"}:
            if self.rule_mass_params is None:
                return {"error": "Model not trained (required for DST combination explanation)"}
            masses_np = self._rule_masses().detach().cpu().numpy()

        return self._combined_rule_summary(
            fired_ids,
            combined_mass,
            rule,
            masses=masses_np,
            weight_key=weight_key or self.combination_weight_key,
            return_details=return_details,
        )

    def save_rules_dsb(self, path):
        """Save rules in DSB text format (sorted by rule certainty for readability).

        Sorting is applied only in the exported file and does not mutate the model order.
        """
        m = self._rule_masses().detach().cpu().numpy() if self.rule_mass_params is not None else None
        with Path(path).open("w", encoding="utf-8") as f:
            order = list(range(len(self.rules)))
            if m is not None and len(m) == len(self.rules):
                omega = np.clip(m[:, -1], 0.0, 1.0)
                certainty = 1.0 - omega
                labels = np.array([r.get("label") if r.get("label") is not None else -1 for r in self.rules], dtype=int)
                label_mass = np.zeros(len(self.rules), dtype=float)
                ok = (labels >= 0) & (labels < (m.shape[1] - 1))
                label_mass[ok] = m[np.where(ok)[0], labels[ok]]
                score = certainty * label_mass
                order = np.argsort(score)[::-1].tolist()

            for i in order:
                r = self.rules[i]
                f.write(f"class {r.get('label')} :: {r.get('caption')} || masses: {m[i] if m is not None else 'n/a'}\n")

    def save_rules_bin(self, path):
        """Save rules and params to binary file."""
        d = {"k": self.num_classes, "algo": self.algo, "feature_names": self.feature_names, 
             "value_decoders": self.value_names, "rules": self.rules, 
             "combination_rule": self.combination_rule,
             "combination_weight_key": getattr(self, "combination_weight_key", "precision"),
             }
        if self.rule_mass_params is not None: d["rule_mass_params"] = self.get_rule_masses().detach().cpu().numpy().tolist()
        if self.initial_rule_masses is not None: d["initial_rule_masses"] = self.initial_rule_masses.detach().cpu().numpy().tolist()
        with Path(path).open("wb") as f: _lazy_pickle().dump(d, f)
        
    def load_rules_bin(self, path):
        """Load rules and params from binary file."""
        with Path(path).open("rb") as f: d = _lazy_pickle().load(f)
        self.num_classes = int(d.get("k", self.num_classes))
        self.algo = str(d.get("algo", self.algo))
        self.feature_names = d.get("feature_names", self.feature_names)
        self.value_names = d.get("value_decoders", self.value_names)
        self.combination_rule = str(d.get("combination_rule", self.combination_rule or "dempster")).lower()
        self.combination_weight_key = str(d.get("combination_weight_key", getattr(self, "combination_weight_key", "precision")))
        self._feature_to_idx = {n: i for i, n in enumerate(self.feature_names or [])}
        self.rules = d.get("rules", [])

        self._rebuild_literal_cache()
        self._build_head()
        if self.rule_mass_params is None:
            return
        self.reset_masses()
        if d.get("rule_mass_params") is not None:
            self.rule_mass_params.data.copy_(self._masses_to_logits(torch.tensor(d["rule_mass_params"], device=self.device, dtype=self.rule_mass_params.dtype)))
            self.project_rules_to_simplex()
        if d.get("initial_rule_masses") is not None:
            self.initial_rule_masses = torch.tensor(d["initial_rule_masses"], device=self.device)

    def prepare_rules_for_export(self, sample=None):
        """Export rules and optional sample activations for visualization."""
        out = {"algo": self.algo, "num_rules": len(self.rules), "rules": [copy.deepcopy(r) for r in self.rules]}
        if self.rule_mass_params is not None: out["masses"] = self._rule_masses().detach().cpu().numpy().tolist()
        if sample is not None:
             act = self._activation_matrix(self._prepare_numeric_tensor([sample]))[0].cpu().numpy()
             out["activated_rule_ids"] = fired = [i for i, f in enumerate(act) if f]
             out["dst_masses"] = self.predict_masses([sample])[0].tolist()
             if fired and self.rule_mass_params is not None:
                 m = self._rule_masses().detach().cpu().numpy()
                 out["activated_rules"] = [{"id": i, "condition": self.rules[i].get("caption"), "class": self.rules[i].get("label"), 
                                            "mass": m[i].tolist(), "stats": self.rules[i].get("stats")} for i in fired]
        return out
    
    def summarize_rules(self, X=None, y=None):
        """Print summary of rule set."""
        print(f"Rules: {len(self.rules)}")
