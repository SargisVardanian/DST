# -*- coding: utf-8 -*-
"""DST model core: rule base, tensor activation, mass fusion and inference."""

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from core import ds_combine_pair, get_combined_rule, prepare_rules_for_export
from rule_generator import RuleGenerator, _canonicalize_condition, _condition_caption, _condition_signature
from device_utils import resolve_torch_device
from utils import load_rules_bin, save_rules_bin, save_rules_dsb


def _as_numpy(X):
    """Ensure input is a numpy array, handling scalars."""
    arr = np.asarray(X)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr


class DSModelMultiQ(nn.Module):
    """
    Dempster-Shafer Theory (DST) Multi-Query Model for classification with uncertainty.

    This model treats classification rules as independent sources of evidence.
    Each rule provides a "mass" assignment over the K classes and an "uncertainty" (Omega) mass.
    These masses are combined using Dempster's Rule of Combination (or strictly defined alternatives)
    to produce a final belief state.

    Responsibilities:
    - keep the frozen rule base,
    - initialize/train rule masses,
    - combine masses for inference,
    - export rules and diagnostics.
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
        rule_uncertainty: float = 0.8,
        combination_rule: str = "dempster",
        **_ignored_kwargs,
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
            combination_rule (str): 'dempster' or 'yager'.
        """
        super().__init__()
        self.num_classes, self.k = int(k), int(k)
        self.algo = str(algo).upper()
        self.device = resolve_torch_device(device)
        self.feature_names = list(feature_names) if feature_names else None
        self.value_names = value_decoders or {}
        self.rule_uncertainty = float(rule_uncertainty)

        self.combination_rule = self._normalize_combination_rule(combination_rule)
        if self.combination_rule not in {"dempster", "cdsgd", "yager"}:
            raise ValueError("combination_rule must be 'dempster', 'cdsgd', or 'yager'")
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
        """Normalize combination rule string (dempster, cdsgd, yager)."""
        r = str(rule or "dempster").strip().lower().replace("-", "_")
        # Backward-compatibility for legacy "vote" setting.
        if r == "vote":
            return "dempster"
        if r == "cdsgd_dempster":
            return "cdsgd"
        if r in {"dempster", "cdsgd", "yager"}:
            return r
        return "dempster"

    def _prior_tensor(self, k: int) -> torch.Tensor:
        """Get the class prior tensor (1, K) for mass redistribution."""
        if self.class_prior is not None and self.class_prior.sum() > 0:
            p = self.class_prior.to(self.device)
            return (p / p.sum().clamp_min(1e-12)).reshape(1, -1)
        return torch.full((1, k), 1.0 / max(1, k), device=self.device)

    @staticmethod
    def _to_numpy(tensor: torch.Tensor, *, dtype=None, bool_cast: bool = False):
        arr = tensor.detach().cpu().numpy()
        if bool_cast:
            return arr.astype(bool, copy=False)
        if dtype is not None:
            return arr.astype(dtype, copy=False)
        return arr

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

    # ----------------------------- Rule generation -----------------------------
    def build_rule_base(self, X, y=None, feature_names=None, *, algo=None, verbose=False, **params):
        """Generate rules, then initialize caches and trainable mass parameters."""
        X_raw = _as_numpy(X)
        y_np = np.asarray(y, dtype=int) if y is not None else None
        if feature_names:
            self.feature_names = list(feature_names)
        self.feature_names = self.feature_names or [f"X[{i}]" for i in range(int(X_raw.shape[1]))]
        self._feature_to_idx = {name: idx for idx, name in enumerate(self.feature_names)}
        self.algo = str(algo or self.algo).upper()

        gen = RuleGenerator(algo=self.algo.lower(), verbose=verbose, **params)
        generated_rules = gen.generate(
            X_raw,
            y_np,
            feature_names=self.feature_names,
            value_decoders=self.value_names,
        )
        self.rules.clear()
        self._rule_signatures.clear()
        for rule in generated_rules:
            cond = _canonicalize_condition(rule.get("specs", ()))
            if not cond:
                continue
            sig = (rule.get("label"), _condition_signature(cond))
            if sig in self._rule_signatures:
                continue
            self._rule_signatures.add(sig)
            self.rules.append(
                {
                    "specs": cond,
                    "label": rule.get("label"),
                    "caption": rule.get("caption") or _condition_caption(cond, self.value_names),
                    "stats": rule.get("stats"),
                }
            )

        if y_np is not None:
            counts = np.bincount(y_np, minlength=self.num_classes).astype(np.float64)
            total = float(np.sum(counts))
            prior = counts / total if total > 0.0 and np.isfinite(total) else np.full(
                self.num_classes, 1.0 / max(1, self.num_classes), dtype=np.float64
            )
            self.class_prior = torch.tensor(prior, device=self.device, dtype=torch.float32)

        self._fit_encoders(X_raw)
        self._rebuild_literal_cache()
        self.initial_rule_masses = None
        self.rule_mass_params = (
            nn.Parameter(
                torch.zeros(len(self.rules), self.num_classes + 1, device=self.device, dtype=torch.float32),
                requires_grad=True,
            )
            if self.rules else None
        )
        self.reset_masses()

    def generate_rules(self, X, y=None, feature_names=None, *, algo=None, verbose=False, **params):
        """Backward-compatible wrapper around build_rule_base()."""
        self.build_rule_base(
            X,
            y,
            feature_names=feature_names,
            algo=algo,
            verbose=verbose,
            **params,
        )

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
            self._lit2rule = None
            self._rule_len = None
            self._lit_feat_idx = None
            self._lit_op_code = None
            self._lit_value = None
            return

        self._lit2rule = torch.zeros((len(self._literals), len(self.rules)), dtype=torch.float32, device=self.device)
        for rid, idxs in enumerate(self._rule_literal_indices):
            if idxs:
                self._lit2rule[idxs, rid] = 1.0
        self._rule_len = torch.tensor([len(i) for i in self._rule_literal_indices], dtype=torch.float32, device=self.device)

        feat_idx, op_code, values = [], [], []
        for name, op, val in self._literals:
            idx = self._feature_to_idx.get(name)
            if idx is None and name.startswith("X[") and name.endswith("]") and name[2:-1].isdigit():
                idx = int(name[2:-1])
            if idx is None:
                raise KeyError(f"Unknown feature '{name}'")
            op_code.append(self.OPS.get(op, 0))
            if op == "==" and name in self._encoders:
                encoded = float(self._encoders[name].get(val, -1.0))
            else:
                try:
                    encoded = float(val)
                except Exception:
                    encoded = float("nan")
            feat_idx.append(int(idx))
            values.append(encoded)

        self._lit_feat_idx = torch.tensor(feat_idx, device=self.device, dtype=torch.long)
        self._lit_op_code = torch.tensor(op_code, device=self.device, dtype=torch.long)
        self._lit_value = torch.tensor(values, device=self.device, dtype=torch.float32)

    # ----------------------------- Evaluation -----------------------------
    def _eval_literals(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate all unique literals against input batch X (vectorized)."""
        if not self._literals:
            return torch.zeros((X.shape[0], 0), dtype=torch.bool, device=self.device)

        cols = X[:, self._lit_feat_idx]
        vals = self._lit_value.view(1, -1)
        eq = torch.isclose(cols, vals, atol=1e-5)
        stack = torch.stack([eq, cols < vals, cols > vals], dim=2)
        ops = self._lit_op_code.view(1, -1, 1).expand(X.shape[0], -1, 1)
        return stack.gather(2, ops).squeeze(2)

    def _activation_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the binary activation matrix (N, num_rules) for the batch."""
        L = self._eval_literals(X)
        if self._lit2rule is None:
            return torch.zeros((X.shape[0], len(self.rules)), dtype=torch.bool, device=self.device)
        return (L.float() @ self._lit2rule).eq(self._rule_len)

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

    def _prepare_numeric_tensor(self, X):
        """Convert/Encode input data to float tensor on device (N, D)."""
        if isinstance(X, torch.Tensor): return X.to(self.device, dtype=torch.float32)
        X_raw = _as_numpy(X)
        if not self.feature_names:
            self.feature_names = [f"X[{i}]" for i in range(X_raw.shape[1])]
            self._feature_to_idx = {n: i for i, n in enumerate(self.feature_names)}
        return torch.as_tensor(self._encode_X(X_raw), dtype=torch.float32, device=self.device)

    def _activate_and_combine(self, X, *, combination_rule=None, masses: Optional[torch.Tensor] = None):
        """Main inference path: encode X, activate rules, combine masses."""
        X_t = self._prepare_numeric_tensor(X)
        act = self._activation_matrix(X_t)
        if not self.rules:
            vac = torch.zeros((X_t.shape[0], self.num_classes + 1), device=self.device, dtype=torch.float32)
            vac[:, -1] = 1.0
            return X_t, act, masses, vac

        masses = self._rule_masses() if masses is None else masses
        rule = self._normalize_combination_rule(combination_rule or self.combination_rule)
        if self.training and rule not in {"dempster", "cdsgd"}:
            raise ValueError(f"Training supports only combination_rule in {{'dempster', 'cdsgd'}} (got {rule!r}).")
        combine_pair = ds_combine_pair if rule == "dempster" else self._combine_yager_pair

        act_flat = act.reshape(-1, act.shape[-1]).to(dtype=torch.bool)
        masses_flat = masses.reshape(-1, masses.shape[-1])
        k_plus_one = masses_flat.shape[-1]
        vac = torch.zeros((1, k_plus_one), device=masses_flat.device, dtype=masses_flat.dtype)
        vac[0, -1] = 1.0
        if masses_flat.numel() == 0 or act_flat.numel() == 0:
            return X_t, act, masses, vac.expand(act_flat.shape[0], -1).reshape(*act.shape[:-1], k_plus_one)

        fused = vac.expand(int(act_flat.shape[0]), -1).clone()
        for rid in act_flat.any(dim=0).nonzero(as_tuple=False).flatten():
            rid_i = int(rid.item())
            combined = combine_pair(fused, masses_flat[rid_i].view(1, k_plus_one))
            fused = torch.where(act_flat[:, rid_i].unsqueeze(1), combined, fused)
        combined = fused.reshape(*act.shape[:-1], k_plus_one)
        return X_t, act, masses, combined

    def _cdsgd_dempster_scores(self, act: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        """CDSGD multi-class Dempster path from the literature implementation.

        The reference `Literature/DSGD-Enhanced/cdsgd/DSModelMultiQ.py` does
        not fuse multi-class masses by repeated pairwise mass combination.
        Instead, it forms per-rule commonality-like scores
        `q = m_singleton + Ω`, multiplies them across active rules, and
        normalizes the resulting class scores.
        """
        n = int(act.shape[0])
        k = int(self.num_classes)
        if masses.numel() == 0 or act.numel() == 0:
            return torch.full((n, k), 1.0 / max(1, k), device=self.device, dtype=torch.float32)

        q = masses[:, :k] + masses[:, -1:].expand(-1, k)
        q = q.clamp_min(1e-12)
        act_f = act.to(dtype=q.dtype).unsqueeze(-1)
        inactive = torch.ones((1, 1, k), device=q.device, dtype=q.dtype)
        q_masked = torch.where(act_f > 0, q.unsqueeze(0), inactive)
        scores = q_masked.prod(dim=1)
        scores = torch.where(scores.sum(dim=1, keepdim=True) > 1e-12, scores, scores + 1e-12)
        return scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def prepare_for_mass_training(self, *, init_seed=None, rule_uncertainty: Optional[float] = None) -> None:
        """Reset learnable masses before a fresh DSGD training run."""
        if rule_uncertainty is not None:
            self.rule_uncertainty = float(rule_uncertainty)
        self.reset_masses(init_seed=init_seed)

    # ----------------------------- Mass params utilities -----------------------------
    def project_rules_to_simplex(self, max_abs_logits: float = 12.0):
        """Project raw mass parameters back to the simplex.

        Kept under the old name for backward compatibility with training code.
        `max_abs_logits` is ignored in the projected-simplex regime.
        """
        if self.rule_mass_params is None:
            return
        with torch.no_grad():
            params = self.rule_mass_params.data
            params.clamp_(0.0, 1.0)
            row_sums = params.sum(dim=1, keepdim=True)
            low_mass = row_sums.squeeze(1) < 1.0
            if low_mass.any():
                params[low_mass, -1] += (1.0 - row_sums[low_mass, 0])
                row_sums = params.sum(dim=1, keepdim=True)
            params.div_(row_sums.clamp_min(1e-12))

    @staticmethod
    def _masses_to_logits(masses: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Backward-compatible adapter: sanitize masses for raw-parameter storage."""
        m = masses.clamp_min(0.0)
        m = m / m.sum(dim=1, keepdim=True).clamp_min(eps)
        return m

    def _rule_masses(self) -> torch.Tensor:
        """Return per-rule mass vectors (R, K+1)."""
        if self.rule_mass_params is None:
            raise ValueError("rule_mass_params is None; initialize rules and call reset_masses() first.")
        return self.rule_mass_params

    def get_rule_masses(self) -> torch.Tensor:
        """Public wrapper for accessing current masses on the simplex."""
        return self._rule_masses()

    def forward(self, X, combination_rule=None):
        """
        Forward pass: compute class probabilities from input features.
        
        Returns:
            Probability tensor (N, K).
        """
        rule = self._normalize_combination_rule(combination_rule or self.combination_rule)
        if rule == "cdsgd":
            X_t = self._prepare_numeric_tensor(X)
            act = self._activation_matrix(X_t)
            masses = self._rule_masses()
            return self._cdsgd_dempster_scores(act, masses)
        _, _, _, combined = self._activate_and_combine(X, combination_rule=rule)
        return self._mass_to_prob(combined)

    def _mass_to_prob(self, masses):
        """Convert mass functions to pignistic probability distributions."""
        masses = torch.nan_to_num(masses, nan=0.0, posinf=0.0, neginf=0.0)
        k = masses.shape[1] - 1
        omega = masses[:, -1:].clamp(0.0, 1.0)
        prior = self._prior_tensor(k)
        p = masses[:, :k].clamp_min(0.0) + omega * prior
        denom = p.sum(dim=1, keepdim=True)
        fallback = prior.expand(p.shape[0], -1)
        p = torch.where(denom > 1e-12, p / denom.clamp_min(1e-12), fallback)
        p = torch.nan_to_num(p, nan=1.0 / max(1, k), posinf=1.0 / max(1, k), neginf=0.0)
        p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return p

    @staticmethod
    def _rule_confidence_from_stats(stats: dict, k_eff: float) -> float:
        """Map rule stats to confidence in [0, 1]."""
        tp_raw = stats.get("support", 0.0)
        fp_raw = stats.get("neg_covered", 0.0)
        tp = float(tp_raw) if np.isfinite(tp_raw) else 0.0
        fp = float(fp_raw) if np.isfinite(fp_raw) else 0.0
        if tp + fp > 0.0:
            q = (tp + 1.0) / (tp + fp + k_eff)
        else:
            prec_raw = stats.get("precision", 1.0 / k_eff)
            q = float(prec_raw) if np.isfinite(prec_raw) else (1.0 / k_eff)
        if not np.isfinite(q):
            q = 1.0 / k_eff
        q = float(np.clip(q, 1.0 / k_eff, 1.0))
        conf = float(np.clip((q - (1.0 / k_eff)) / max(1e-12, (1.0 - (1.0 / k_eff))), 0.0, 1.0))
        if not np.isfinite(conf):
            conf = 0.0
        return conf

    def reset_masses(self, init_seed=None):
        """Reinitialize masses in probability form and copy them into raw parameters."""
        if self.rule_mass_params is None:
            return
        if init_seed is not None:
            torch.manual_seed(init_seed)

        base_unc = float(self.rule_uncertainty)
        num_rules = int(self.rule_mass_params.shape[0])
        num_classes = int(self.num_classes)

        prior = self._prior_tensor(num_classes).to(
            device=self.rule_mass_params.device,
            dtype=self.rule_mass_params.dtype,
        ).reshape(-1)
        prior = prior / prior.sum().clamp_min(1e-12)

        # Warm-start:
        # stronger rules -> lower omega and higher target-class mass.
        init_masses = torch.zeros(
            num_rules,
            num_classes + 1,
            device=self.rule_mass_params.device,
            dtype=self.rule_mass_params.dtype,
        )
        k_eff = float(max(2, num_classes))
        base_unc_t = float(np.clip(base_unc, 0.10, 0.90))
        for rid, rule in enumerate(self.rules):
            stats = rule.get("stats") or {}
            conf = self._rule_confidence_from_stats(stats, k_eff)
            omega_i = float(np.clip(base_unc_t - 0.45 * conf, 0.10, 0.92))
            if not np.isfinite(omega_i):
                omega_i = base_unc_t
            non_omega = 1.0 - omega_i

            label = rule.get("label", None)
            cls_mass = non_omega * (0.55 + 0.45 * conf)
            row = prior * max(0.0, non_omega - cls_mass)
            if label is not None:
                lbl = int(label)
                if 0 <= lbl < num_classes:
                    row = row.clone()
                    row[lbl] = row[lbl] + cls_mass
            else:
                row = prior * non_omega

            init_masses[rid, :num_classes] = row
            init_masses[rid, -1] = omega_i

        init_masses = torch.nan_to_num(init_masses, nan=0.0, posinf=0.0, neginf=0.0)
        row_sums = init_masses.sum(dim=1, keepdim=True)
        bad_rows = row_sums.squeeze(1) <= 1e-12
        if bad_rows.any():
            init_masses[bad_rows, :num_classes] = prior
            init_masses[bad_rows, -1] = 0.0
            row_sums = init_masses.sum(dim=1, keepdim=True)
        init_masses = init_masses / init_masses.sum(dim=1, keepdim=True).clamp_min(1e-12)

        # Store probability-form initial masses for later use (e.g., ablations).
        self.initial_rule_masses = init_masses.detach().clone()

        self.rule_mass_params.data.copy_(init_masses)
        self.project_rules_to_simplex()

    # ----------------------------- Predictions & Explanation -----------------------------
    def predict_masses(self, X, use_initial_masses=False, *, combination_rule=None):
        """Predict fused mass functions for input samples."""
        orig = None
        if use_initial_masses and self.initial_rule_masses is not None:
            orig = self.rule_mass_params.data.clone()
            self.rule_mass_params.data.copy_(self.initial_rule_masses.to(self.device))

        try:
            _, _, _, combined = self._activate_and_combine(X, combination_rule=combination_rule)
            return self._to_numpy(combined)
        finally:
            if orig is not None:
                self.rule_mass_params.data.copy_(orig)
    

    def uncertainty_stats(self, X, *, combination_rule: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Return per-sample uncertainty: rule-average Ω and combined Ω."""
        n = _as_numpy(X).shape[0]
        if not self.rules:
            ones = np.ones(n, dtype=np.float32)
            depth = np.zeros(n, dtype=np.float32)
            return {
                "unc_rule": ones,
                "unc_mean": ones,
                "unc_comb": ones,
                "fusion_depth": depth,
            }

        with torch.no_grad():
            rule = self._normalize_combination_rule(combination_rule or self.combination_rule)
            if rule == "cdsgd":
                X_t = self._prepare_numeric_tensor(X)
                act = self._activation_matrix(X_t)
                masses = self._rule_masses()
                probs = self._cdsgd_dempster_scores(act, masses)

                act_f = act.to(dtype=masses.dtype)
                omega = masses[:, -1].unsqueeze(0)
                fired = act_f.sum(dim=1)
                unc_rule = (omega * act_f).sum(dim=1) / fired.clamp(min=1.0)
                unc_rule = torch.where(fired == 0, torch.ones_like(unc_rule), unc_rule)
                unc_comb = 1.0 - probs.max(dim=1).values
                fusion_depth = fired.to(dtype=torch.float32)
                return {
                    "unc_rule": self._to_numpy(unc_rule, dtype=np.float32),
                    "unc_mean": self._to_numpy(unc_rule, dtype=np.float32),
                    "unc_comb": self._to_numpy(unc_comb, dtype=np.float32),
                    "fusion_depth": self._to_numpy(fusion_depth, dtype=np.float32),
                }

            _, act, masses, combined = self._activate_and_combine(X, combination_rule=combination_rule)

            # 1. Rule-average Omega (intrinsic model confidence)
            act_f = act.to(dtype=masses.dtype)
            omega = masses[:, -1].unsqueeze(0)
            fired = act_f.sum(dim=1)
            unc_rule = (omega * act_f).sum(dim=1) / fired.clamp(min=1.0)
            unc_rule = torch.where(fired == 0, torch.ones_like(unc_rule), unc_rule)
            unc_comb = combined[..., -1]
            fusion_depth = fired.to(dtype=torch.float32)

            return {
                "unc_rule": self._to_numpy(unc_rule, dtype=np.float32),
                "unc_mean": self._to_numpy(unc_rule, dtype=np.float32),
                "unc_comb": self._to_numpy(unc_comb, dtype=np.float32),
                "fusion_depth": self._to_numpy(fusion_depth, dtype=np.float32),
            }

    def predict_rule_baseline_proba(
        self,
        X,
        *,
        method: str = "first_hit_laplace",
        alpha: float = 1.0,
    ):
        """Predict class probabilities using rules only (no DSGD training).

        Supported methods (minimal, paper-facing):
        - first_hit_laplace: first fired rule, Laplace-smoothed precision -> prob vector
        - weighted_vote: sum over fired rules (Laplace precision × log1p(support))

        If no rule fires, falls back to class prior (estimated from training labels if available,
        else uniform).
        """
        method = str(method).lower().strip()
        if method not in {"first_hit_laplace", "weighted_vote"}:
            raise ValueError(f"Unknown raw baseline '{method}'. Allowed: first_hit_laplace, weighted_vote")

        X_raw = np.asarray(X)
        X_t = self._prepare_numeric_tensor(X_raw)
        act = self._to_numpy(self._activation_matrix(X_t), bool_cast=True)  # (N, R) bool
        n = int(act.shape[0])
        k = int(self.num_classes)
        if self.class_prior is not None:
            prior = np.asarray(self._to_numpy(self.class_prior), dtype=float).reshape(-1)
            prior_sum = float(prior.sum())
            if prior.size != k or not np.isfinite(prior_sum) or prior_sum <= 0.0:
                prior = np.full(k, 1.0 / max(1, k), dtype=float)
            else:
                prior = prior / prior_sum
        else:
            prior = np.full(k, 1.0 / max(1, k), dtype=float)

        # Precompute rule qualities
        q = np.zeros(len(self.rules), dtype=float)
        w = np.ones(len(self.rules), dtype=float)
        for i, r in enumerate(self.rules):
            stats = (r.get("stats") or {})
            sup = float(stats.get("support", 0.0))
            neg = float(stats.get("neg_covered", 0.0))
            q[i] = (sup + alpha) / (sup + neg + alpha * k) if (sup + neg) >= 0 else 1.0 / k
            if method == "weighted_vote":
                w[i] = math.log1p(max(0.0, sup))

        out = np.zeros((n, k), dtype=float)

        if method == "first_hit_laplace":
            for j in range(n):
                fired = np.flatnonzero(act[j])
                if fired.size == 0:
                    out[j] = prior
                    continue
                rid = int(fired[0])
                label = self.rules[rid].get("label", None)
                if label is None:
                    out[j] = prior
                    continue
                label = int(label)
                qq = float(q[rid])
                pj = (1.0 - qq) * prior
                if 0 <= label < k:
                    pj[label] += qq
                else:
                    pj = prior
                out[j] = pj
            row_sum = np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
            return out / row_sum

        # weighted_vote
        for j in range(n):
            fired = np.flatnonzero(act[j])
            if fired.size == 0:
                out[j] = prior
                continue
            scores = np.zeros(k, dtype=float)
            for rid in fired:
                label = self.rules[int(rid)].get("label", None)
                if label is None:
                    continue
                label = int(label)
                if 0 <= label < k:
                    scores[label] += q[int(rid)] * w[int(rid)]
            if scores.sum() <= 0:
                out[j] = prior
            else:
                out[j] = scores / scores.sum()
        row_sum = np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
        return out / row_sum

    def prepare_rules_for_export(self, x):
        return prepare_rules_for_export(self, x)

    def get_combined_rule(
        self,
        x,
        *,
        return_details: bool = True,
        combination_rule: Optional[str] = None,
        decision_class: Optional[int] = None,
        include_merged_rule: bool = False,
        merged_rule_beta: float = 1.0,
    ):
        return get_combined_rule(
            self,
            x,
            return_details=return_details,
            combination_rule=combination_rule,
            decision_class=decision_class,
            include_merged_rule=include_merged_rule,
            merged_rule_beta=merged_rule_beta,
        )

    def save_rules_dsb(self, path, *, decimals: int = 4, include_masses: bool = True):
        return save_rules_dsb(self, path, decimals=decimals, include_masses=include_masses)

    def save_rules_bin(self, path, *, decimals: int = 4):
        return save_rules_bin(self, path, decimals=decimals)

    def load_rules_bin(self, path):
        return load_rules_bin(self, path)
