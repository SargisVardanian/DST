"""
Dempster-Shafer Theory (DST) Model with Multi-Q Support.

This module implements a DST-based classification model that uses rule-based reasoning
with mass functions to handle uncertainty. It supports multiple rule generation algorithms
(STATIC, RIPPER, FOIL) and provides DSGD++ initialization for confidence-based mass assignment.
"""

import copy, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence
from rule_generator import Condition, Literal, RuleGenerator
from core import params_to_mass, ds_combine_pair, masses_to_pignistic as core_masses_to_pignistic

# Allowed operators for rule conditions, float formatting, and epsilon for numerical stability
_ALLOWED_OPS, _FLOAT_FMT, _EPS = {"==", "<", ">"}, "{:.6g}", 1e-9

def _lazy_pickle():
    """Lazy import pickle or dill for serialization."""
    try: import dill as pickle
    except ImportError: import pickle
    return pickle

def _as_numpy(X):
    """Convert input to numpy array, ensuring 2D shape."""
    return np.asarray(X).reshape(1, -1) if np.asarray(X).ndim == 1 else np.asarray(X)

def masses_to_pignistic(masses):
    """Convert mass functions to pignistic probabilities."""
    return core_masses_to_pignistic(masses)

class DSModelMultiQ(nn.Module):
    """
    Dempster-Shafer Theory (DST) Multi-Query Model for classification with uncertainty.
    
    This model generates and applies rules to data, combining their mass functions
    using Dempster's combination rule to produce predictions with quantified uncertainty.
    """
    
    def __init__(self, k: int, algo: str = "STATIC", device: str = "cpu", feature_names=None, value_decoders=None, rule_uncertainty=0.8):
        """
        Initialize the DST model.
        
        Args:
            k (int): Number of classes in the classification problem
            algo (str): Rule generation algorithm ('STATIC', 'RIPPER', or 'FOIL'). Default: 'STATIC'
            device (str): PyTorch device for computation ('cpu' or 'cuda'). Default: 'cpu'
            feature_names (list): Names of features in the dataset. If None, will be auto-generated
            value_decoders (dict): Mapping from feature names to value decoders for categorical features
            rule_uncertainty (float): Base uncertainty level for mass initialization (0.0-1.0). Default: 0.8
        """
        super().__init__()
        self.num_classes, self.k, self.algo, self.device = int(k), int(k), str(algo).upper(), torch.device(device)
        self.feature_names, self.value_names = list(feature_names) if feature_names else None, value_decoders or {}
        self.rule_uncertainty = float(rule_uncertainty)
        self.class_prior = None
        self._encoders, self._decoders = {}, {}
        self.rules, self._rule_signatures = [], set()
        self._literal_to_index, self._literals, self._rule_literal_indices = {}, [], []
        self._lit2rule = self._rule_len = self.rule_class_mask = self.head = self.rule_mass_params = self.initial_rule_masses = None
        self._feature_to_idx = {n: i for i, n in enumerate(self.feature_names)} if self.feature_names else {}
        self._last_algo = None
        self.to(self.device)

    def _fit_encoders(self, X_raw):
        """
        Fit encoders for categorical features in the dataset.
        
        Args:
            X_raw (ndarray): Raw input data with potentially mixed types
        """
        self.feature_names = self.feature_names or [f"X[{i}]" for i in range(X_raw.shape[1])]
        self._feature_to_idx = {n: i for i, n in enumerate(self.feature_names)}
        self._encoders.clear(); self._decoders.clear()
        for j, name in enumerate(self.feature_names):
            col = X_raw[:, j]
            dec = self.value_names.get(name, {})
            if not (bool(dec) or not (np.issubdtype(col.dtype, np.number) and col.dtype.kind != "O")): continue
            uniq, enc = np.unique(col), {}
            if dec:
                for code, raw_val in dec.items():
                    try: code_f = float(code)
                    except: continue
                    for key in (raw_val, str(raw_val), code, str(code), code_f): enc[key] = code_f
                for u in uniq: enc.setdefault(u, float(u))
                dec_full = {**dec, **{str(k): v for k, v in dec.items()}}
                self._decoders[name], self.value_names[name] = dec_full, dec_full
            else:
                enc = {val: float(i) for i, val in enumerate(uniq)}
                dec_map = {int(i): val for i, val in enumerate(uniq)}
                self._decoders[name] = dec_map
                self.value_names[name] = {**self.value_names.get(name, {}), **dec_map, **{str(k): v for k, v in dec_map.items()}}
            self._encoders[name] = enc

    def _encode_X(self, X_raw):
        """
        Encode raw data using fitted encoders.
        
        Args:
            X_raw (ndarray): Raw input data
            
        Returns:
            ndarray: Encoded numerical representation
        """
        X_num = np.empty(X_raw.shape, dtype=np.float32)
        for j, name in enumerate(self.feature_names or []):
            col, enc = X_raw[:, j], self._encoders.get(name)
            if enc: X_num[:, j] = [enc.get(v, -1) for v in col]
            elif np.issubdtype(col.dtype, np.number): X_num[:, j] = col.astype(np.float32)
            else: X_num[:, j] = np.nan
        return X_num

    def _encode_literal_value(self, name, op, value):
        """
        Encode a literal value for comparison.
        
        Args:
            name (str): Feature name
            op (str): Comparison operator
            value: Raw value to encode
            
        Returns:
            float: Encoded numerical value
        """
        if op == "==" and name in self._encoders: return float(self._encoders[name].get(value, -1))
        try: return float(value)
        except: return float("nan")

    def generate_rules(self, X, y=None, feature_names=None, *, algo=None, verbose=False, **params):
        """
        Generate classification rules using specified algorithm.
        
        Args:
            X (ndarray): Input features
            y (ndarray, optional): Target labels (required for RIPPER/FOIL)
            feature_names (list, optional): Names of features
            algo (str, optional): Algorithm to use ('STATIC', 'RIPPER', 'FOIL')
            verbose (bool): Print detailed progress information. Default: False
            **params: Additional parameters for rule generation
        """
        X_raw, y_np = _as_numpy(X), np.asarray(y, dtype=int) if y is not None else None
        if feature_names: self.feature_names = list(feature_names)
        self.feature_names = self.feature_names or [f"X[{i}]" for i in range(X_raw.shape[1])]
        self.algo = self._last_algo = str(algo or self._last_algo or self.algo or "STATIC").upper()
        gen = RuleGenerator(algo=self.algo.lower(), verbose=verbose, **params)
        self.rules.clear(); self._rule_signatures.clear()

        if self.algo == "STATIC":
            for r in gen.build_static_rules(X_raw, feature_names=self.feature_names, y=y_np, value_decoders=self.value_names, verbose=verbose, **params):
                self._add_rule_specs(r.get("specs", ()), r.get("label"), r.get("caption"), r.get("stats"))
        elif self.algo in {"RIPPER", "FOIL"}:
            if y_np is None: raise ValueError("Supervised rule generation requires target labels (y)")
            gen.fit(X_raw, y_np, feature_names=self.feature_names)
            for label, cond, stats in gen.ordered_rules:
                self._add_rule_specs(tuple((n, o, v) for n, (o, v) in sorted(cond.items())), int(label), None, stats)
        else: raise ValueError(f"Unsupported algo {self.algo}")

        if y_np is not None and self.num_classes >= 2:
            counts = np.bincount(y_np, minlength=self.num_classes).astype(np.float32)
            counts[counts <= 0.0] = 1.0
            inv = 1.0 / counts
            self.class_prior = torch.tensor(inv / float(inv.sum()), device=self.device, dtype=torch.float32) if np.isfinite(inv.sum()) and inv.sum() > 0.0 else None

        self._fit_encoders(X_raw); self._rebuild_literal_cache(); self._build_head(); self.reset_masses()

    def _add_rule_specs(self, specs, label, caption, stats=None):
        """
        Add a rule specification to the model if it's unique.
        
        Args:
            specs (tuple): Tuple of (feature_name, operator, value) literals
            label (int): Target class label for the rule
            caption (str): Human-readable description of the rule
            stats (dict, optional): Statistics about the rule (precision, recall, etc.)
        """
        cond = self._canonicalize_condition(specs)
        if not cond: return
        sig = self._condition_signature(cond)
        if sig in self._rule_signatures: return
        self._rule_signatures.add(sig)
        self.rules.append({"specs": cond, "label": int(label) if label is not None else None, "caption": caption or self._condition_caption(cond, self.value_names, label), "stats": stats})

    @staticmethod
    def _normalise_value(v):
        """
        Normalize numeric values for consistent comparison.
        
        Args:
            v: Value to normalize
            
        Returns:
            Normalized value (float or int)
        """
        if isinstance(v, (np.floating, float)):
            f = float(v)
            if math.isfinite(f) and abs(f - round(f)) < 1e-9: return float(round(f))
        return float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, (np.integer, int)) else v)

    @classmethod
    def _values_equal(cls, a, b):
        """
        Check if two values are equal with numerical tolerance.
        
        Args:
            a: First value
            b: Second value
            
        Returns:
            bool: True if values are considered equal
        """
        an, bn = cls._normalise_value(a), cls._normalise_value(b)
        return abs(an - bn) <= 1e-9 if isinstance(an, float) or isinstance(bn, float) else an == bn

    @classmethod
    def _canonicalize_condition(cls, literals):
        """
        Canonicalize a condition by merging and validating literals.
        
        This method combines multiple literals on the same feature, checks for
        contradictions, and returns a standardized sorted tuple.
        
        Args:
            literals (iterable): Sequence of (name, op, value) tuples
            
        Returns:
            tuple: Canonicalized condition or None if contradictory
        """
        if not literals: return ()
        scope = {}
        for name, op, raw_val in literals:
            if op not in _ALLOWED_OPS: raise ValueError(f"Unsupported op {op}")
            val = cls._normalise_value(raw_val)
            s = scope.setdefault(name, {"eq": None, "lt": None, "gt": None})
            if op == "==":
                if s["eq"] is not None and not cls._values_equal(s["eq"], val): return None
                s["eq"] = val
            elif op == "<": s["lt"] = min(s["lt"], float(val)) if s["lt"] is not None else float(val)
            elif op == ">": s["gt"] = max(s["gt"], float(val)) if s["gt"] is not None else float(val)
        res = []
        for name, s in sorted(scope.items()):
            eq, lt, gt = s["eq"], s["lt"], s["gt"]
            if eq is not None:
                if (lt is not None and float(eq) >= lt - _EPS) or (gt is not None and float(eq) <= gt + _EPS): return None
                res.append((name, "==", eq))
            else:
                if gt is not None and lt is not None and gt >= lt - _EPS: return None
                if gt is not None: res.append((name, ">", gt))
                if lt is not None: res.append((name, "<", lt))
        return tuple(sorted(res, key=lambda x: (x[0], {"==": 0, ">": 1, "<": 2}[x[1]], repr(cls._normalise_value(x[2])))))

    @classmethod
    def _condition_signature(cls, cond):
        """
        Generate unique string signature for a condition.
        
        Args:
            cond (tuple): Canonicalized condition
            
        Returns:
            str: Unique signature string
        """
        return "&&".join(f"{n}{o}{(_FLOAT_FMT.format(cls._normalise_value(val)) if isinstance(cls._normalise_value(val), float) else str(cls._normalise_value(val)))}" for n, o, val in cond)

    @staticmethod
    def _resolve_value_name(name, value, mapping):
        """
        Resolve feature value to its human-readable name.
        
        Args:
            name (str): Feature name
            value: Encoded value
            mapping (dict): Value decoder mappings
            
        Returns:
            Decoded value or original if not found
        """
        d = mapping.get(name) or {}
        for k in (value, str(value), int(value) if isinstance(value, (int, float, str)) and str(value).replace('.','',1).isdigit() else None):
            if k in d: return d[k]
        return value

    @classmethod
    def _condition_caption(cls, cond, value_names, label=None):
        """
        Generate human-readable caption for a condition.
        
        Args:
            cond (tuple): Canonicalized condition
            value_names (dict): Value decoder mappings
            label (int, optional): Target class label
            
        Returns:
            str: Human-readable condition description
        """
        def _fmt(n, o, v):
            d = cls._resolve_value_name(n, v, value_names)
            return f"{n} {o} {(_FLOAT_FMT.format(float(d)) if isinstance(d, float) else d)}"
        return (" & ".join(_fmt(*t) for t in cond) or "<empty>") + (f" → class {label}" if label is not None else "")

    def _rebuild_literal_cache(self):
        """
        Rebuild internal caches for efficient rule evaluation.
        
        Creates literal-to-index mappings and rule-literal matrices for fast lookup.
        """
        self._literal_to_index, self._rule_literal_indices = {}, []
        for r in self.rules:
            self._rule_literal_indices.append([self._literal_to_index.setdefault(l, len(self._literal_to_index)) for l in r.get("specs", ())])
        self._literals = list(self._literal_to_index.keys())
        if not self._literals: self._lit2rule = self._rule_len = None; return
        self._lit2rule = torch.zeros((len(self._literals), len(self.rules)), dtype=torch.bool, device=self.device)
        for rid, idxs in enumerate(self._rule_literal_indices): self._lit2rule[idxs, rid] = True
        self._rule_len = torch.tensor([len(i) for i in self._rule_literal_indices], dtype=torch.long, device=self.device)

    def _build_head(self):
        """
        Build the neural network head for mass parameters.
        
        Creates learnable parameters for mass functions, one per rule.
        """
        self.head, self.initial_rule_masses = None, None
        if not self.rules: self.rule_mass_params = None; return
        self.rule_mass_params = nn.Parameter(torch.zeros(len(self.rules), self.num_classes + 1, device=self.device, dtype=torch.float32), requires_grad=True)

    def _eval_literals(self, X):
        """
        Evaluate all literals against input data.
        
        Args:
            X (Tensor): Input features (N, D)
            
        Returns:
            Tensor: Boolean mask (N, num_literals) indicating which literals are satisfied
        """
        if not self._literals: return torch.zeros((X.shape[0], 0), dtype=torch.bool, device=self.device)
        masks = []
        for name, op, val in self._literals:
            idx = self._feature_to_idx.get(name)
            if idx is None and name.startswith("X[") and name.endswith("]") and name[2:-1].isdigit(): idx = int(name[2:-1])
            if idx is None or not (0 <= idx < X.shape[1]): raise KeyError(f"Unknown feature '{name}'")
            col, v = X[:, idx], self._encode_literal_value(name, op, val)
            if op == "==": masks.append(torch.isclose(col, torch.tensor(v, device=self.device), atol=1e-5))
            elif op == "<": masks.append(col < float(v))
            elif op == ">": masks.append(col > float(v))
        return torch.stack(masks, dim=1)

    def _activation_matrix(self, X):
        """
        Compute rule activation matrix.
        
        Args:
            X (Tensor): Input features (N, D)
            
        Returns:
            Tensor: Boolean mask (N, num_rules) indicating which rules fire for each sample
        """
        if not self.rules: return torch.zeros((X.shape[0], 0), dtype=torch.bool, device=self.device)
        L = self._eval_literals(X)
        return torch.stack([L[:, idxs].all(dim=1) if idxs else torch.ones(X.shape[0], dtype=torch.bool, device=self.device) for idxs in self._rule_literal_indices], dim=1)

    def _combine_active_masses(self, masses: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """
        Combine mass functions of activated rules using Dempster's combination rule.
        
        Args:
            masses (Tensor): Mass parameters (num_rules, num_classes+1)
            act (Tensor): Activation matrix (N, num_rules)
            
        Returns:
            Tensor: Combined masses (N, num_classes+1) after applying Dempster's rule
        """
        if masses.numel() == 0 or act.numel() == 0:
            return torch.cat([torch.zeros((act.shape[0], self.num_classes), device=self.device, dtype=masses.dtype), torch.ones((act.shape[0], 1), device=self.device, dtype=masses.dtype)], dim=1)
        vac = torch.zeros((1, 1, self.num_classes + 1), device=self.device, dtype=masses.dtype); vac[..., -1] = 1.0
        m = torch.where(act.unsqueeze(-1), masses.unsqueeze(0).expand(act.shape[0], -1, -1), vac.expand(act.shape[0], masses.shape[0], -1))
        if m.shape[1] == 1: return m[:, 0, :]
        if m.shape[1] % 2 == 1: m = torch.cat([m, vac.expand(m.shape[0], 1, -1)], dim=1)
        while m.shape[1] > 1:
            m = ds_combine_pair(m[:, 0::2, :], m[:, 1::2, :])
            if m.dim() == 2: m = m.unsqueeze(1)
            if m.shape[1] % 2 == 1 and m.shape[1] > 1: m = torch.cat([m, vac.expand(m.shape[0], 1, -1)], dim=1)
        return m[:, 0, :]

    def _prepare_numeric_tensor(self, X):
        """
        Convert input to encoded numeric tensor.
        
        Args:
            X: Input data (ndarray or Tensor)
            
        Returns:
            Tensor: Encoded features on device
        """
        if isinstance(X, torch.Tensor): return X.to(self.device, dtype=torch.float32)
        X_raw = _as_numpy(X)
        if not self.feature_names: self.feature_names = [f"X[{i}]" for i in range(X_raw.shape[1])]
        return torch.as_tensor(self._encode_X(X_raw), dtype=torch.float32, device=self.device)

    def project_rules_to_simplex(self):
        """
        Project rule mass parameters onto the probability simplex.
        
        Ensures all mass parameters are valid probability distributions (sum to 1, all non-negative).
        """
        if self.rule_mass_params is None: return
        with torch.no_grad():
            rp, eps = self.rule_mass_params.data, 1e-8
            rp.clamp_(min=eps)
            bad = rp.sum(dim=1, keepdim=True) <= eps
            if bad.any(): rp[bad] = 0.0; rp[bad, -1] = 1.0
            rp /= rp.sum(dim=1, keepdim=True).clamp_min(eps)

    def forward(self, X):
        """
        Forward pass: compute class probabilities.
        
        Args:
            X: Input features (ndarray or Tensor)
            
        Returns:
            Tensor: Class probabilities (N, num_classes)
        """
        X = self._prepare_numeric_tensor(X)
        if not self.rules or self.rule_mass_params is None: return torch.full((X.shape[0], self.num_classes), 1.0/max(1, self.num_classes), device=self.device)
        return self._mass_to_prob(self._combine_active_masses(params_to_mass(self.rule_mass_params), self._activation_matrix(X)))

    def _mass_to_prob(self, masses: torch.Tensor) -> torch.Tensor:
        """
        Convert mass functions to probability distributions using pignistic transformation.
        
        Args:
            masses (Tensor): Mass functions (N, num_classes+1)
            
        Returns:
            Tensor: Probability distributions (N, num_classes)
        """
        if masses.ndim != 2 or masses.shape[1] < 2: raise ValueError("masses tensor must have shape (N, K+1)")
        p = masses[:, :masses.shape[1]-1] + masses[:, masses.shape[1]-1:] * (self.class_prior.to(masses.device) / self.class_prior.sum() if self.class_prior is not None and self.class_prior.sum() > 0 else 1.0/masses.shape[1])
        return p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def reset_masses(self):
        """
        Reset mass parameters to random initialization.
        
        Initializes each rule with a random distribution over classes plus base uncertainty.
        """
        if self.rule_mass_params is None: return
        with torch.no_grad():
            base_unc = min(max(float(self.rule_uncertainty), 0.0), 1.0)
            m = torch.zeros_like(self.rule_mass_params)
            for i in range(len(self.rules)):
                arr = torch.rand(self.num_classes + 1, device=self.device); arr[-1] = 0.0
                arr = arr / arr.sum().clamp_min(1e-9) * (1.0 - base_unc); arr[-1] = base_unc
                m[i] = arr
            if m.shape[0] > len(self.rules): m[len(self.rules):, -1] = 1.0
            self.rule_mass_params.copy_(m); self.project_rules_to_simplex()
        self.initial_rule_masses = self.rule_mass_params.detach().clone()

    def init_masses_dsgdpp(self, X, y):
        """
        DSGD++ mass initialization based on confidence from data representativeness and rule purity.
        
        This initialization method assigns higher mass to rules that cover representative samples
        with high purity (majority class agreement), reducing initial uncertainty.
        
        Args:
            X: Training features
            y: Training labels
        """
        if self.rule_mass_params is None: return
        X_np = _as_numpy(X)
        y_np = np.asarray(y, dtype=int)
        
        # 1. Calculate class centroids and sample representativeness
        # Simple approach: one centroid per class
        centroids = {}
        for c in np.unique(y_np):
            centroids[c] = np.mean(X_np[y_np == c], axis=0)
            
        # Calculate sample representativeness: Rep(x) = 1 / (1 + dist(x, centroid_y))
        reps = np.zeros(len(X_np))
        for i, (x, label) in enumerate(zip(X_np, y_np)):
            c = centroids.get(label)
            if c is not None:
                dist = np.linalg.norm(x - c)
                reps[i] = 1.0 / (1.0 + dist)
        
        # Calculate confidence for each rule based on coverage statistics
        with torch.no_grad():
            m = torch.zeros_like(self.rule_mass_params)
            # Evaluate all rules on all data to find coverage
            X_t = self._prepare_numeric_tensor(X_np)
            act = self._activation_matrix(X_t).cpu().numpy()
            
            for i in range(len(self.rules)):
                covered_indices = np.where(act[:, i])[0]
                if len(covered_indices) == 0:
                    # No coverage -> full uncertainty
                    m[i, -1] = 1.0
                    continue
                
                covered_y = y_np[covered_indices]
                covered_reps = reps[covered_indices]
                
                # Calculate purity as proportion of majority class
                counts = np.bincount(covered_y, minlength=self.num_classes)
                maj_cls = np.argmax(counts)
                purity = counts[maj_cls] / len(covered_y)
                
                # Average representativeness of covered samples
                avg_rep = np.mean(covered_reps)
                
                # Confidence is product of purity and average representativeness
                c = purity * avg_rep
                
                # Assign mass: m(maj_cls) = c, m(Omega) = 1 - c
                m[i, maj_cls] = c
                m[i, -1] = 1.0 - c
                
            # Handle padding for extra rule slots
            if m.shape[0] > len(self.rules): m[len(self.rules):, -1] = 1.0
            
            self.rule_mass_params.copy_(m)
            self.project_rules_to_simplex()
        self.initial_rule_masses = self.rule_mass_params.detach().clone()

    def renormalize_masses(self):
        """Renormalize mass parameters to valid probability simplex."""
        self.project_rules_to_simplex()
    
    def init_masses_random(self):
        """Initialize masses with random values."""
        self.reset_masses()

    def predict_with_dst(self, X, use_initial_masses=False):
        """
        Predict using DST mass combination with pignistic transformation.
        
        Args:
            X: Input features
            use_initial_masses (bool): Use initial (untrained) masses instead of learned ones
            
        Returns:
            ndarray: Pignistic probabilities (N, num_classes)
        """
        return masses_to_pignistic(self.predict_masses(X, use_initial_masses))
    
    def predict_dst_labels(self, X, use_initial_masses=False):
        """
        Predict class labels using DST.
        
        Args:
            X: Input features
            use_initial_masses (bool): Use initial masses instead of learned ones
            
        Returns:
            ndarray: Predicted class labels
        """
        return self.predict_with_dst(X, use_initial_masses).argmax(axis=1).astype(int)

    def predict_masses(self, X, use_initial_masses=False):
        """
        Predict mass functions for input samples.
        
        Args:
            X: Input features
            use_initial_masses (bool): Use initial masses instead of learned ones
            
        Returns:
            ndarray: Combined mass functions (N, num_classes+1)
        """
        orig = self.rule_mass_params.data.clone() if use_initial_masses and self.initial_rule_masses is not None else None
        if orig is not None: self.rule_mass_params.data.copy_(self.initial_rule_masses.data)
        try: return self._combine_active_masses(params_to_mass(self.rule_mass_params), self._activation_matrix(self._prepare_numeric_tensor(X))).detach().cpu().numpy()
        finally:
            if orig is not None: self.rule_mass_params.data.copy_(orig)

    def sample_uncertainty(self, X):
        """
        Compute uncertainty (mass on Omega) for each sample.
        
        Args:
            X: Input features
            
        Returns:
            ndarray: Uncertainty values (0-1) for each sample
        """
        if not self.rules or self.rule_mass_params is None: return np.ones(_as_numpy(X).shape[0], dtype=np.float32)
        act = self._activation_matrix(self._prepare_numeric_tensor(X))
        unc = (params_to_mass(self.rule_mass_params)[:, -1].unsqueeze(0) * act).sum(dim=1) / act.sum(dim=1).clamp(min=1)
        if (act.sum(dim=1) == 0).any(): unc[act.sum(dim=1) == 0] = 1.0
        return unc.detach().cpu().numpy()

    def predict_by_rule_vote(self, X, default_label=0):
        """
        Predict using simple majority vote among activated rules.
        
        Args:
            X: Input features
            default_label (int): Default class when no rules fire
            
        Returns:
            ndarray: Predicted class labels
        """
        if not self.rules: return np.full(_as_numpy(X).shape[0], int(default_label), dtype=int)
        act = self._activation_matrix(self._prepare_numeric_tensor(X)).cpu().numpy()
        lbls = np.array([r.get("label", -1) if r.get("label") is not None else -1 for r in self.rules], dtype=int)
        return np.array([int(np.bincount(lbls[m & (lbls != -1)], minlength=2).argmax()) if (m & (lbls != -1)).any() else int(default_label) for m in act], dtype=int)

    def predict_by_weighted_rule_vote(self, X, default_label=0, weight_key="precision"):
        """
        Predict using weighted vote among activated rules.
        
        Args:
            X: Input features
            default_label (int): Default class when no rules fire
            weight_key (str): Rule statistic to use as weight (e.g., 'precision', 'recall')
            
        Returns:
            ndarray: Predicted class labels
        """
        if not self.rules: return np.full(_as_numpy(X).shape[0], int(default_label), dtype=int)
        act = self._activation_matrix(self._prepare_numeric_tensor(X)).cpu().numpy()
        lbls = np.array([r.get("label", -1) if r.get("label") is not None else -1 for r in self.rules], dtype=int)
        w = np.array([float((r.get("stats") or {}).get(weight_key, 1.0)) for r in self.rules], dtype=float); w[~np.isfinite(w) | (w <= 0.0)] = 1.0
        return np.array([int(np.bincount(lbls[m & (lbls != -1)], weights=w[m & (lbls != -1)], minlength=max(lbls.max()+1, 2)).argmax()) if (m & (lbls != -1)).any() else int(default_label) for m in act], dtype=int)

    def predict_by_first_rule(self, X, default_label=0):
        """
        Predict using the first activated rule (top-to-bottom priority).
        
        Args:
            X: Input features
            default_label (int): Default class when no rules fire
            
        Returns:
            ndarray: Predicted class labels
        """
        if not self.rules: return np.full(_as_numpy(X).shape[0], int(default_label), dtype=int)
        act = self._activation_matrix(self._prepare_numeric_tensor(X)).cpu().numpy()
        lbls = [r.get("label") for r in self.rules]
        return np.array([int(next((lbls[i] for i, f in enumerate(m) if f and lbls[i] is not None), default_label)) for m in act], dtype=int)

    def save_rules_dsb(self, path):
        """
        Save rules in human-readable .dsb format with mass distributions.
        
        Args:
            path (str): Output file path
        """
        m = params_to_mass(self.rule_mass_params).detach().cpu().numpy() if self.rule_mass_params is not None else None
        order = list(np.argsort(m[:, -1])) if m is not None and len(m) == len(self.rules) else list(range(len(self.rules)))
        with Path(path).open("w", encoding="utf-8") as f:
            for i in order:
                r = self.rules[i]
                f.write(f"class {r.get('label') if r.get('label') is not None else '?'} :: {r.get('caption', '')}" + (f" || masses: {', '.join(f'{x:.6f}' for x in m[i])}" if m is not None and i < len(m) else "") + "\n")

    def save_rules_bin(self, path):
        """
        Save model rules and parameters in binary format (pickle).
        
        Args:
            path (str): Output file path
        """
        d = {"k": self.num_classes, "algo": self.algo, "feature_names": self.feature_names, "value_decoders": self.value_names, "rules": self.rules}
        if self.rule_mass_params is not None: d["rule_mass_params"] = self.rule_mass_params.detach().cpu().numpy().tolist()
        if self.initial_rule_masses is not None: d["initial_rule_masses"] = self.initial_rule_masses.detach().cpu().numpy().tolist()
        with Path(path).open("wb") as f: _lazy_pickle().dump(d, f)

    def load_rules_bin(self, path):
        """
        Load model rules and parameters from binary format.
        
        Args:
            path (str): Input file path
        """
        with Path(path).open("rb") as f: d = _lazy_pickle().load(f)
        self.num_classes = self.k = int(d.get("k", self.num_classes))
        self.algo, self.feature_names, self.value_names = str(d.get("algo", self.algo)), d.get("feature_names", self.feature_names), d.get("value_decoders", self.value_names)
        self._feature_to_idx = {n: i for i, n in enumerate(self.feature_names or [])}
        self.rules, self._rule_signatures = [], set()
        for r in d.get("rules", []): self._add_rule_specs(r.get("specs", ()), r.get("label"), r.get("caption"), r.get("stats"))
        self._rebuild_literal_cache(); self._build_head()
        if d.get("rule_mass_params") is not None:
            with torch.no_grad(): self.rule_mass_params.copy_(torch.tensor(d["rule_mass_params"], device=self.device))
        if d.get("initial_rule_masses") is not None: self.initial_rule_masses = torch.tensor(d["initial_rule_masses"], device=self.device)
        if self.rule_mass_params is None: self.reset_masses()

    def prepare_rules_for_export(self, sample=None):
        """
        Prepare rules and predictions for export as dictionary.
        
        Args:
            sample (optional): If provided, include sample-specific activation and mass info
            
        Returns:
            dict: Comprehensive rule information including masses and activations
        """
        s = {"algo": self.algo, "num_rules": len(self.rules), "rules": [copy.deepcopy(r) for r in self.rules]}
        if self.rule_mass_params is not None: s["masses"] = params_to_mass(self.rule_mass_params).detach().cpu().numpy().tolist()
        if sample is None: return s
        act = self._activation_matrix(self._prepare_numeric_tensor([sample]))[0].cpu().numpy()
        s["activated_rule_ids"] = fired = [i for i, f in enumerate(act) if f]
        s["dst_masses"] = self.predict_masses([sample])[0].tolist()
        if fired:
            m = params_to_mass(self.rule_mass_params).detach().cpu().numpy() if self.rule_mass_params is not None else None
            s["activated_rules"] = [{"id": i, "condition": self.rules[i].get("caption", f"rule#{i}"), "class": self.rules[i].get("label"), "mass": m[i].tolist() if m is not None else None, "stats": self.rules[i].get("stats")} for i in fired]
        return s

    def summarize_rules(self, X=None, y=None, *, top_n=10, samples=3, sample_size=256):
        """
        Print comprehensive summary of rules and their performance.
        
        Args:
            X (optional): Input features for evaluation
            y (optional): Target labels for evaluation
            top_n (int): Number of examples to show. Default: 10
            samples (int): Number of random samples to evaluate. Default: 3
            sample_size (int): Size of each sample. Default: 256
        """
        if not self.rules: return print("[summary] No rules available.")
        lens = np.array([len(r.get("specs", ())) for r in self.rules])
        print(f"[summary] rules={len(self.rules)} len[min/med/max]={int(lens.min())}/{int(np.median(lens))}/{int(lens.max())}")
        om = params_to_mass(self.rule_mass_params)[:, -1].detach().cpu().numpy() if self.rule_mass_params is not None else None
        def _fmt(i): return f"#{i} L={len(self.rules[i].get('specs', ()))} Ω={f'{om[i]:.3f}' if om is not None else 'n/a'} class={self.rules[i].get('label')} :: {self.rules[i].get('caption', f'rule#{i}')}"
        print("[short] examples:"); [print("  ", _fmt(i)) for i in np.argsort(lens)[:top_n]]
        print("[long] examples:"); [print("  ", _fmt(i)) for i in np.argsort(-lens)[:top_n]]
        if X is None or y is None: return
        X_np, y_np, rng = _as_numpy(X), np.asarray(y, dtype=int), np.random.default_rng(42)
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        for k in range(max(1, int(samples))):
            idx = rng.choice(len(X_np), min(sample_size, len(X_np)), replace=False)
            Xs, ys = X_np[idx], y_np[idx]
            y_raw = self.predict_by_rule_vote(Xs, default_label=int(np.bincount(ys).argmax()))
            y_dst = self.predict_dst_labels(Xs)
            m_raw = {k: float(f(ys, y_raw, zero_division=0) if k!="Acc" else f(ys, y_raw)) for k, f in [("Acc", accuracy_score), ("F1", f1_score), ("P", precision_score), ("R", recall_score)]}
            m_dst = {k: float(f(ys, y_dst, zero_division=0) if k!="Acc" else f(ys, y_dst)) for k, f in [("Acc", accuracy_score), ("F1", f1_score), ("P", precision_score), ("R", recall_score)]}
            print(f"[sample {k+1}] vote={m_raw}  dst={m_dst}")

__all__ = ["DSModelMultiQ", "Literal", "Condition", "masses_to_pignistic"]
