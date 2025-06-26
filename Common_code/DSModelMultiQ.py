"""
DSModelMultiQ
--------------
Multi-class Dempster–Shafer (DST) rule model with a single `forward`:
  X → rule activations (binary) → per-rule masses → DST combination → pignistic → logits.

Key ideas
=========
• Each rule r has a learnable vector of length K+1: [w_0..w_{K-1}, theta], stored as nn.Parameter.

  These parameters are converted to a valid mass function m_r over {classes ∪ Ω} via softmax.

  (Ω is the ignorance mass, index K.)

• A rule fires on x iff all its literals are true (strict AND). Literals are kept in `_specs`.

• Vectorized activation is used when `_specs` exist for all rules; otherwise Python predicates are used.

• Combination uses Dempster’s rule, applied only where the rule fires (sparse updates per rule).

• `forward` accepts a **batch** tensor and returns class logits (log of pignistic probabilities).

• Feature value decoders keep human-readable strings in saved `.dsb` files (e.g., `education == Bachelors`).


Public surface
--------------
- set_prior(prior_probs)

- set_value_decoders(mapping)

- generate_raw(X, y=None, column_names=None, algo="STATIC"|"RIPPER"|"FOIL")

- activation_matrix(X: np.ndarray) -> np.ndarray[uint8]

- get_rule_targets() -> np.ndarray[int]  (rule-level target; -1 if absent)

- save_rules_bin(path), load_rules_bin(path)

- save_rules_dsb(path, decimals=3)

"""
import dill

import numpy as np

import torch

from torch import nn

from typing import Iterable, List, Dict, Tuple, Optional

from scipy.stats import norm



from DSRule import DSRule

from utils import is_categorical



# Core DST transforms (vectorized, torch-first; fall back to local numpy ops if missing)

try:

    from core import params_to_mass, masses_to_pignistic, ds_combine_pair

    _HAS_CORE = True

except Exception:

    _HAS_CORE = False

    def params_to_mass(W: torch.Tensor) -> torch.Tensor:

        # Softmax over the last dimension → normalized masses per rule

        return torch.softmax(W, dim=-1)

    def masses_to_pignistic(m: torch.Tensor) -> torch.Tensor:

        # Move Ω mass equally to K singletons: betP = m_k + m_Ω / K

        K = m.shape[-1] - 1

        single = m[..., :K]

        omega = m[..., K:K+1]

        return single + omega / float(max(1, K))

    def ds_combine_pair(mA: torch.Tensor, mB: torch.Tensor) -> torch.Tensor:

        # Torch-safe, batched Dempster’s rule for (K+Ω) masses

        K = mA.shape[-1] - 1

        a_k, a_o = mA[..., :K], mA[..., K:K+1]

        b_k, b_o = mB[..., :K], mB[..., K:K+1]

        # conflict (only disjoint singletons collide). This simplified variant ignores cross-singleton conflict

        # and is suitable when rules vote mostly on singletons/Ω.

        # Combine singletons: s = a_k * b_k + a_k * b_o + a_o * b_k

        s = a_k * b_k + a_k * b_o + a_o * b_k

        # Combine omega: ω = a_o * b_o

        o = a_o * b_o

        mass = torch.cat([s, o], dim=-1)

        mass = mass / (mass.sum(dim=-1, keepdim=True) + 1e-12)

        return mass



try:

    from DSRipper import DSRipper  # optional rule inducer

except Exception:

    DSRipper = None



EPS = 1e-12



# ------------------------------- helpers (specs) -------------------------------

def _make_predicate_from_spec(spec_list: Iterable[Tuple[str, str, float]],

                              feature_names: List[str]):

    """Build a Python predicate from (name, op, value) literals; ALL literals must hold (AND)."""

    name2idx = {n: i for i, n in enumerate(feature_names)}

    def _pred(x_row, _spec=tuple(spec_list)):

        for (name, op, val) in _spec:

            j = name2idx[name]

            v = x_row[j]

            if   op == ">":









                if not (v >  val): return False

            elif op == "<":
                if not (v <  val): return False

            elif op == ">=":
                if not (v >= val): return False

            elif op == "<=":
                if not (v <= val): return False

            elif op == "==":
                if not (v == val): return False

            elif op == "!=":
                if not (v != val): return False

            else:

                raise ValueError(f"Unknown operator '{op}' in spec")

        return True

    return _pred



def _caption_from_spec(spec_list: Iterable[Tuple[str, str, float]]) -> str:

    return " & ".join(f"{n} {op} {v}" for n, op, v in spec_list) if spec_list else "<rule>"



# =============================== MODEL ===============================

class DSModelMultiQ(nn.Module):

    """k-class DST model with rule activation + combination in a single `forward`."""

    def __init__(

        self,

        k: int,

        *,

        device: str = "cpu",

        use_vectorized_activation: bool = True,

    ) -> None:

        super().__init__()

        self.k: int = int(k)

        self.device = torch.device(device)

        self.use_vectorized_activation = bool(use_vectorized_activation)

        # Rules and learnable parameters [w_0..w_{K-1}, theta]

        self.preds: List[DSRule] = []
        self._params = nn.ParameterList()
        self.n: int = 0

        # Feature names for vectorized activation
        self.feature_names: Optional[List[str]] = None
        self._name2idx: Dict[str, int] = {}
        self.n_features: Optional[int] = None

        # Value decoders: feature_name or index → {code -> string}
        self.value_decoders: Dict[object, Dict[int, str]] = {}

        # Prior mass/logits
        self._prior_mass = torch.zeros(self.k + 1, dtype=torch.float32, device=self.device)
        self._prior_log = torch.log(torch.full((self.k,), 1.0 / max(1, self.k), device=self.device))

    # --------------------------- basic API ---------------------------
    @property
    def rules(self) -> List[DSRule]:
        return self.preds

    @rules.setter
    def rules(self, value: Iterable[DSRule]):
        self.preds = list(value) if value is not None else []
        self.n = len(self.preds)
        self._params = nn.ParameterList([
            nn.Parameter(torch.zeros(self.k + 1, dtype=torch.float32, device=self.device))
            for _ in range(self.n)
        ])

    def parameters(self, recurse: bool = True):  # keep torch API compatible
        return self._params

    def _set_feature_names(self, names: Iterable[str]):
        self.feature_names = list(names)
        self._name2idx = {n: i for i, n in enumerate(self.feature_names)}
        self.n_features = len(self.feature_names)

    def set_value_decoders(self, decoders: Optional[Dict[object, Dict[int, str]]]):
        """Register decoders to render categorical values as strings in saved rules.
        Keys may be feature names or indices; we mirror both for convenience."""
        self.value_decoders = {}
        if not decoders:
            return
        self.value_decoders.update(decoders)
        # Mirror by name/index if feature names are already known
        if self.feature_names:
            for k, mapping in list(self.value_decoders.items()):
                if isinstance(k, str) and k in self._name2idx:
                    self.value_decoders[self._name2idx[k]] = dict(mapping)
                if isinstance(k, int) and 0 <= k < len(self.feature_names):
                    self.value_decoders[self.feature_names[k]] = dict(mapping)

    def set_prior(self, prior: Optional[np.ndarray | torch.Tensor]):
        """Set prior class probabilities; builds both prior logits and prior mass (with small Ω)."""
        if prior is None:
            p = torch.full((self.k,), 1.0 / max(1, self.k), dtype=torch.float32, device=self.device)
        else:
            p = torch.as_tensor(prior, dtype=torch.float32, device=self.device)
            p = p / (p.sum() + EPS)
        self._prior_log = torch.log(p.clamp_min(1e-12))
        # Put small ignorance mass to avoid zero Ω; renormalize
        omega = 0.02
        prior_mass = torch.cat([p * (1.0 - omega), torch.tensor([omega], device=self.device)])
        self._prior_mass = prior_mass / prior_mass.sum()

    def add_rule(self, rule: DSRule, *, init_for_class: Optional[int] = None, init_std: float = 0.01, omega0: float = 0.8):
        """Register a rule and create its parameter vector [K+1].

        We keep initial Ω (uncertainty) exactly 0.8 but trainable:
          softmax_omega = exp(b) / (exp(b) + sum_i exp(a_i)) = 0.8
          => exp(b) = (0.8 / 0.2) * sum_i exp(a_i) = 4 * sum_i exp(a_i)
        """
        # Build class logits (a[0:k]) and omega-logit (b)
        if getattr(rule, "caption", "") == "<bias>":
            vec = torch.zeros(self.k + 1, dtype=torch.float32, device=self.device)
            vec[: self.k] = self._prior_log.detach().clone()
            s = torch.exp(vec[: self.k]).sum()
            vec[-1] = torch.log(torch.tensor(omega0 / (1.0 - omega0), device=self.device) * s)
        else:
            a = torch.empty(self.k, dtype=torch.float32, device=self.device).normal_(0.0, float(init_std))
            # Небольшой «толчок» к init_for_class только если он задан (для STATIC он None)
            if init_for_class is not None and 0 <= int(init_for_class) < self.k:
                a[int(init_for_class)] += 0.10
            s = torch.exp(a).sum()
            b = torch.log(torch.tensor(omega0 / (1.0 - omega0), device=self.device) * s)
            vec = torch.cat([a, b.unsqueeze(0)], dim=0)

        # Trainable
        self._params.append(nn.Parameter(vec))
        self.preds.append(rule)
        self.n = len(self.preds)


    # --------------------------- forward path ---------------------------
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Full pipeline: activations → per-rule masses → DST combination → pignistic → logits.
        Returns class logits (log pignistic)."""
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device)

        # Allow optional leading index column (drop it if present)
        Xnp = X.detach().cpu().numpy()
        if self.n_features is not None and Xnp.shape[1] == self.n_features + 1:
            X_plain = Xnp[:, 1:]
        else:
            X_plain = Xnp

        # Build activation matrix (B, R)
        A_np = self._fired_mask_numpy(X_plain)
        A = torch.from_numpy(A_np.astype(np.float32)).to(self.device)

        # If no rules: return prior logits for every row
        B = X.shape[0]
        if self.n == 0:
            return self._prior_log[None, :].repeat(B, 1)

        # Convert parameters → per-rule masses [R, K+1]
        W = torch.stack(list(self._params), dim=0) if self._params else torch.zeros(0, self.k + 1, device=self.device)
        M_rules = params_to_mass(W) if _HAS_CORE else torch.softmax(W, dim=-1)

        # Combine masses per sample, applying only fired rules
        m = self._prior_mass.unsqueeze(0).repeat(B, 1)  # [B, K+1]
        if self.n > 0 and A.any():
            for j in range(self.n):
                idx = torch.nonzero(A[:, j] > 0.5, as_tuple=False).squeeze(-1)
                if idx.numel() == 0:
                    continue
                m_sel = m.index_select(0, idx)
                m_rule = M_rules[j].unsqueeze(0).expand_as(m_sel)
                m_comb = ds_combine_pair(m_sel, m_rule)
                m.index_copy_(0, idx, m_comb)

        # Pignistic transform → logits
        p = masses_to_pignistic(m)
        logits = torch.log(p.clamp_min(1e-12))
        return logits

    # --------------------------- activation utils ---------------------------
    def _fired_mask_numpy(self, X_plain: np.ndarray) -> np.ndarray:
        """Vectorized rule activation over a 2D numpy array (drops index column upstream)."""
        B = X_plain.shape[0]
        fired = np.zeros((B, self.n), dtype=bool)
        can_vec = (
            self.use_vectorized_activation and
            (self.feature_names is not None) and
            all((getattr(r, "_is_bias", False) or hasattr(r, "_specs")) for r in self.preds)
        )
        if can_vec:
            cols = {name: X_plain[:, j] for name, j in self._name2idx.items()}
            for j, r in enumerate(self.preds):
                if getattr(r, "_is_bias", False) or getattr(r, "caption", "") == "<bias>":
                    fired[:, j] = True
                    continue
                spec = getattr(r, "_specs", [])
                m = np.ones(B, dtype=bool)
                for (name, op, val) in spec:
                    col = cols[name]
                    if   op == ">":  m &= (col >  val)
                    elif op == "<":  m &= (col <  val)
                    elif op == ">=": m &= (col >= val)
                    elif op == "<=": m &= (col <= val)
                    elif op == "==": m &= (col == val)
                    elif op == "!=": m &= (col != val)
                    else: raise ValueError(f"Unknown op '{op}'")
                    if not m.any():
                        break
                fired[:, j] = m
        else:
            for i, row in enumerate(X_plain):
                for j, r in enumerate(self.preds):
                    if getattr(r, "_is_bias", False) or getattr(r, "caption", "") == "<bias>":
                        fired[i, j] = True
                    else:
                        try:
                            fn = getattr(r, "pred", None) or getattr(r, "predicate", None) or r
                            fired[i, j] = bool(fn(row))
                        except Exception:
                            fired[i, j] = False
        return fired

    def activation_matrix(self, X: np.ndarray) -> np.ndarray:
        """Public numpy API: (N, R) binary activation matrix (uint8)."""
        X = np.asarray(X)
        if self.n_features is not None and X.shape[1] == self.n_features + 1:
            X = X[:, 1:]
        return self._fired_mask_numpy(X).astype(np.uint8)

    def get_rule_targets(self) -> np.ndarray:
        """Return per-rule target class if known, else -1.
        For induced rules we store `_label`. Otherwise -1."""
        t = np.full((self.n,), -1, dtype=int)
        for j, r in enumerate(self.preds):
            if hasattr(r, "_label"):
                try:
                    t[j] = int(getattr(r, "_label"))
                except Exception:
                    pass
        return t

    # --------------------------- rule builders ---------------------------
    def _display_value(self, feat, val):
        """Pretty-print categorical values using the decoder, if available."""
        keys = []
        if isinstance(feat, str):
            keys.extend([feat, self._name2idx.get(feat, None)])
        else:
            keys.extend([feat, self.feature_names[feat] if self.feature_names else None])
        for k in keys:
            if k is None: continue
            mp = self.value_decoders.get(k)
            if not mp: continue
            if val in mp: return mp[val]
            iv = int(round(val))
            if iv in mp and abs(val - iv) < 1e-9: return mp[iv]
            fv = float(val)
            if fv in mp: return mp[fv]
        return str(int(val)) if float(val).is_integer() else f"{float(val):g}"

    def _caption_from_spec_display(self, spec_list):
        parts = []
        for (name, op, val) in spec_list:
            show = self._display_value(name, val) if op in {"==", "!="} else (int(val) if float(val).is_integer() else f"{float(val):g}")
            parts.append(f"{name} {op} {show}")
        return " & ".join(parts) if parts else "<rule>"

    def _add_rule_with_specs(self, spec_list, caption=None, *, init_for_class=None):
        assert self.feature_names is not None, "feature_names must be set"
        pred = _make_predicate_from_spec(spec_list, self.feature_names)
        rule = DSRule(pred, caption or _caption_from_spec(spec_list))
        setattr(rule, "_specs", list(spec_list))

        algo_name = str(getattr(self, "algo", "")).upper()
        if init_for_class is not None and algo_name in {"RIPPER", "FOIL"}:
            setattr(rule, "_label", int(init_for_class))  # only for inductive rules

        pretty = self._caption_from_spec_display(spec_list)
        rule.caption = pretty
        setattr(rule, "_specs_display",
                [(n, op, self._display_value(n, v) if op in {"==", "!="} else v) for (n, op, v) in spec_list])

        self.add_rule(rule, init_for_class=(int(init_for_class) if algo_name in {"RIPPER", "FOIL"} else None))


    # --------- generators: STATIC / RIPPER / FOIL ---------
    def generate_statistic_single_rules(self, X: np.ndarray, breaks: int = 2,
                                        column_names: Optional[Iterable[str]] = None,
                                        generated_columns: Optional[Iterable[bool]] = None):
        X = np.asarray(X)
        m = X.shape[1]
        if column_names is None:
            column_names = [f"X[{i}]" for i in range(m)]
        self._set_feature_names(column_names)

        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        brks = norm.ppf(np.linspace(0, 1, breaks + 2))[1:-1]
        if generated_columns is None:
            generated_columns = np.repeat(True, m)
        for i in range(m):
            if not generated_columns[i]:
                continue
            fname = self.feature_names[i]
            col = X[:, i]
            if is_categorical(col):
                cats = np.unique(col[~np.isnan(col)])
                for cat in cats:
                    self._add_rule_with_specs([(fname, "==", float(cat))], f"{fname} == {self._display_value(fname, cat)}")
            else:
                v = mean[i] + std[i] * brks[0]
                self._add_rule_with_specs([(fname, "<=", float(v))], f"{fname} <= {v:.6f}")
                for j in range(1, len(brks)):
                    vl = v; v = mean[i] + std[i] * brks[j]
                    self._add_rule_with_specs([(fname, ">=", float(vl)), (fname, "<", float(v))], f"{vl:.6f} < {fname} < {v:.6f}")
                self._add_rule_with_specs([(fname, ">", float(v))], f"{fname} > {v:.6f}")


    def generate_categorical_rules(self, X: np.ndarray, column_names: Optional[Iterable[str]] = None,
                                   exclude: Optional[Iterable[str]] = None):
        X = np.asarray(X)
        m = X.shape[1]
        if column_names is None:
            column_names = [f"X[{i}]" for i in range(m)]
        self._set_feature_names(column_names)
        exclude = set(exclude or [])
        for i in range(m):
            fname = self.feature_names[i]
            if fname in exclude:
                continue
            col = X[:, i]
            if is_categorical(col):
                cats = np.unique(col[~np.isnan(col)])
                for cat in cats:
                    self._add_rule_with_specs([(fname, "==", float(cat))], f"{fname} = {cat}")

    def generate_mult_pair_rules(self, X: np.ndarray, column_names: Optional[Iterable[str]] = None,
                                 include_square: bool = False):
        X = np.asarray(X)
        mean = np.nanmean(X, axis=0)
        m = len(mean)
        if column_names is None:
            column_names = [f"X[{i}]" for i in range(m)]
        self._set_feature_names(column_names)
        offset = 0 if include_square else 1
        for i in range(m):
            for j in range(i + offset, m):
                fi, fj = self.feature_names[i], self.feature_names[j]
                mi, mj = float(mean[i]), float(mean[j])
                self._add_rule_with_specs([(fi, ">=", mi), (fj, ">=", mj)],
                                          f"Positive {fi}-{mi:.3f}, {fj}-{mj:.3f}")
                self._add_rule_with_specs([(fi, "<", mi), (fj, "<", mj)],
                                          f"Positive {fi}-{mi:.3f}, {fj}-{mj:.3f}")
                self._add_rule_with_specs([(fi, ">=", mi), (fj, "<", mj)],
                                          f"Negative {fi}-{mi:.3f}, {fj}-{mj:.3f}")
                self._add_rule_with_specs([(fi, "<", mi), (fj, ">=", mj)],
                                          f"Negative {fi}-{mi:.3f}, {fj}-{mj:.3f}")

    def generate_custom_range_single_rules(self, column_names: Iterable[str], name: str, breaks: List[float]):
        names = list(column_names)
        self._set_feature_names(names)
        v = breaks[0]
        self._add_rule_with_specs([(name, "<=", float(v))], f"{name} <= {v:.3f}")
        for j in range(1, len(breaks)):
            vl = v; v = breaks[j]
            self._add_rule_with_specs([(name, ">=", float(vl)), (name, "<", float(v))],
                                      f"{vl:.3f} < {name} < {v:.3f}")
        self._add_rule_with_specs([(name, ">", float(v))], f"{name} > {v:.3f}")

    def generate_outside_range_pair_rules(self, column_names: Iterable[str],
                                          ranges: List[Tuple[float, float, str]]):
        names = list(column_names)
        self._set_feature_names(names)
        # ranges: [(lo_i, hi_i, col_i), ...]
        for a in range(len(ranges)):
            li, hi, col_i = ranges[a]
            for b in range(a + 1, len(ranges)):
                lj, hj, col_j = ranges[b]
                if not np.isnan(li) and not np.isnan(lj):
                    self._add_rule_with_specs([(col_i, "<", float(li)), (col_j, "<", float(lj))],
                                              f"Low {col_i} and Low {col_j}")
                if not np.isnan(hi) and not np.isnan(lj):
                    self._add_rule_with_specs([(col_i, ">", float(hi)), (col_j, "<", float(lj))],
                                              f"High {col_i} and Low {col_j}")
                if not np.isnan(hi) and not np.isnan(hj):
                    self._add_rule_with_specs([(col_i, ">", float(hi)), (col_j, ">", float(hj))],
                                              f"High {col_i} and High {col_j}")
                if not np.isnan(li) and not np.isnan(hj):
                    self._add_rule_with_specs([(col_i, "<", float(li)), (col_j, ">", float(hj))],
                                              f"Low {col_i} and High {col_j}")

    def _add_induced_rule(self, cond: Dict[str, Tuple[str, float]], cls: int, column_names: List[str]):
        self._set_feature_names(column_names)
        spec = [(name, op, float(val)) for name, (op, val) in cond.items()]
        caption = f"Class {cls}: " + self._caption_from_spec_display(spec)
        self._add_rule_with_specs(spec, caption, init_for_class=int(cls))

    def _dedup_and_add(self, ordered_rules, X, column_names, y=None, min_pos_coverage: float = 0.003):
        X = np.asarray(X)
        names = list(column_names)
        self._set_feature_names(names)
        y = None if y is None else np.asarray(y).astype(int)
        seen = set()
        for cls, cond in ordered_rules:
            if not cond:
                continue
            spec = tuple((k, str(cond[k][0]), float(cond[k][1])) for k in sorted(cond.keys()))
            if spec in seen:
                continue
            pred = _make_predicate_from_spec(spec, names)
            mask = np.fromiter((bool(pred(row)) for row in X), count=len(X), dtype=bool)
            if y is not None:
                cls = int(cls)
                pos = (y == cls)
                pos_cnt = pos.sum()
                if pos_cnt == 0: continue
                pos_cov = float((mask & pos).sum()) / float(pos_cnt)
                if pos_cov < float(min_pos_coverage):
                    continue
            seen.add(spec)
            self._add_induced_rule({k: (op, val) for k, op, val in spec}, int(cls), names)

    def generate_ripper_rules(self, X: np.ndarray, y: np.ndarray, column_names=None) -> None:
        if DSRipper is None:
            raise NotImplementedError("DSRipper is not available")
        X = np.asarray(X); y = np.asarray(y).astype(int)
        if column_names is None:
            column_names = [f"X[{i}]" for i in range(X.shape[1])]
        ripper = DSRipper(algo="ripper")
        ripper.fit(X, y, feature_names=column_names)
        self._dedup_and_add(ripper._ordered_rules, X, column_names, y=y, min_pos_coverage=0.003)

    def generate_foil_rules(self, X: np.ndarray, y: np.ndarray, column_names=None) -> None:
        if DSRipper is None:
            raise NotImplementedError("DSRipper is not available")
        X = np.asarray(X); y = np.asarray(y).astype(int)
        if column_names is None:
            column_names = [f"X[{i}]" for i in range(X.shape[1])]
        foil = DSRipper(algo="foil")
        foil.fit(X, y, feature_names=column_names)
        self._dedup_and_add(foil._ordered_rules, X, column_names, y=y, min_pos_coverage=0.003)

    def generate_raw(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                     column_names: Optional[List[str]] = None, *, algo: str = "STATIC", **kwargs):
        # Remember current generator to drive labeling logic elsewhere
        self.algo = (algo or "STATIC").upper()
        algo = self.algo
        if algo == "STATIC":
            breaks = int(kwargs.get("breaks", 2))
            self.generate_statistic_single_rules(X, breaks=breaks, column_names=column_names)
        elif algo == "RIPPER":
            if y is None:
                raise ValueError("y must be provided for RIPPER rule generation.")
            self.generate_ripper_rules(X, y, column_names=column_names)
        elif algo == "FOIL":
            if y is None:
                raise ValueError("y must be provided for FOIL rule generation.")
            self.generate_foil_rules(X, y, column_names=column_names)
        else:
            raise ValueError(f"Unknown algo '{algo}'. Choose from STATIC, RIPPER, FOIL.")


    # --------------------------- save / load ---------------------------
    def clone_from(self, other: "DSModelMultiQ") -> None:
        self.feature_names = list(other.feature_names) if other.feature_names else None
        self._name2idx = {n: i for i, n in enumerate(self.feature_names)} if self.feature_names else {}
        self.n_features = len(self.feature_names) if self.feature_names else None
        self.value_decoders = dict(getattr(other, "value_decoders", {}))
        self.preds = []
        self._params = nn.ParameterList()
        for r_other, p_other in zip(other.preds, other._params):
            fn = getattr(r_other, "pred", None) or getattr(r_other, "predicate", None)
            rule = DSRule(fn if fn is not None else (lambda _: False), getattr(r_other, "caption", "<rule>"))
            for attr in ("_specs", "_is_bias", "usability", "_label", "_specs_display"):
                if hasattr(r_other, attr):
                    setattr(rule, attr, getattr(r_other, attr))
            self.preds.append(rule)
            self._params.append(nn.Parameter(p_other.detach().clone().to(self.device)))
        self.n = len(self.preds)
        self._prior_log = other._prior_log.detach().clone().to(self.device)
        self._prior_mass = other._prior_mass.detach().clone().to(self.device)

    def save_rules_bin(self, path: str) -> None:
        data = {
            "feature_names": self.feature_names,
            "prior_logits": self._prior_log.detach().cpu().numpy(),
            "prior_mass": self._prior_mass.detach().cpu().numpy(),
            "value_decoders": self.value_decoders,
            "rules": [],
            "params": [p.detach().cpu().numpy() for p in self._params],
        }
        for r in self.preds:
            entry = {"caption": getattr(r, "caption", "<rule>")}
            if hasattr(r, "_specs"):
                entry["specs"] = list(getattr(r, "_specs"))
            else:
                fn = getattr(r, "pred", None) or getattr(r, "predicate", None)
                entry["pred_bytes"] = dill.dumps(fn if fn is not None else (lambda _: False))
            if getattr(r, "_is_bias", False):
                entry["is_bias"] = True
            if hasattr(r, "_label"):
                entry["_label"] = int(getattr(r, "_label"))
            if hasattr(r, "_specs_display"):
                entry["_specs_display"] = list(getattr(r, "_specs_display"))
            data["rules"].append(entry)
        with open(path, "wb") as f:
            dill.dump(data, f)

    def load_rules_bin(self, path: str) -> None:
        with open(path, "rb") as f:
            data = dill.load(f)
        self.feature_names = data.get("feature_names")
        self._name2idx = {n: i for i, n in enumerate(self.feature_names)} if self.feature_names else {}
        self.n_features = len(self.feature_names) if self.feature_names else None
        prior = data.get("prior_logits")
        if prior is not None:
            self._prior_log = torch.from_numpy(prior).to(self.device)
        prior_m = data.get("prior_mass")
        if prior_m is not None:
            self._prior_mass = torch.from_numpy(prior_m).to(self.device)
        self.value_decoders = data.get("value_decoders", {})
        self.preds = []
        self._params = nn.ParameterList()
        for entry, p_arr in zip(data.get("rules", []), data.get("params", [])):
            caption = entry.get("caption", "<rule>")
            if "specs" in entry and self.feature_names is not None:
                pred = _make_predicate_from_spec(entry["specs"], self.feature_names)
                rule = DSRule(pred, caption)
                setattr(rule, "_specs", list(entry["specs"]))
            else:
                pred = dill.loads(entry.get("pred_bytes", dill.dumps(lambda _: False)))
                rule = DSRule(pred, caption)
            if entry.get("is_bias", False):
                setattr(rule, "_is_bias", True)
            if "_label" in entry:
                setattr(rule, "_label", int(entry["_label"]))
            if "_specs_display" in entry:
                setattr(rule, "_specs_display", list(entry["_specs_display"]))
            self.preds.append(rule)
            self._params.append(nn.Parameter(torch.from_numpy(np.asarray(p_arr)).float().to(self.device)))
        self.n = len(self.preds)

    def save_rules_dsb(self, path: str, decimals: int = 3) -> None:
        """Write a human-readable .dsb; no 'Class ...' prefix for unlabeled (STATIC) rules."""
        lines = []
        for p, r in zip(self._params, self.preds):
            vec = p.detach().cpu().numpy().astype(np.float64)
            probs = np.exp(vec - vec.max()); probs = probs / probs.sum()
            cls_m = probs[: self.k]            # per-class mass
            unc  = probs[-1]                   # Ω mass

            # Human-readable body
            if hasattr(r, "_specs_display"):
                parts = [f"{name} {op} {val}" for (name, op, val) in getattr(r, "_specs_display")]
                pretty = " & ".join(parts) if parts else getattr(r, "caption", "<rule>")
            else:
                pretty = getattr(r, "caption", "<rule>")

            # Prefix only if the rule was induced with a label (RIPPER/FOIL).
            # STATIC rules do not carry _label and must have no class prefix.
            prefix = f"Class {int(getattr(r, '_label'))}: " if hasattr(r, "_label") else ""

            cls_part = ", ".join([f"{m:.{decimals}f}" for m in cls_m])
            lines.append(f"{prefix}{pretty}  ||  mass=[{cls_part}, unc={unc:.{decimals}f}]")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

