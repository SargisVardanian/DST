import itertools, math, numpy as np
from collections import defaultdict
from typing import Any, Dict, Sequence, Tuple, List

try:
    from .rule_utils import condition_caption, normalise_rule_value, resolve_rule_value
except ImportError:  # pragma: no cover - direct script/import fallback
    from rule_utils import condition_caption, normalise_rule_value, resolve_rule_value

Literal, Condition = Tuple[str, str, Any], Tuple[Tuple[str, str, Any], ...]
_ALLOWED_OPS, _FLOAT_FMT, _EPS = {"==", "<", ">"}, "{:.6g}", 1e-9

def _normalise_value(v):
    return normalise_rule_value(v)

def _values_equal(a, b):
    an, bn = _normalise_value(a), _normalise_value(b)
    if isinstance(an, float) or isinstance(bn, float):
        return abs(float(an) - float(bn)) <= 1e-9
    return an == bn

def _canonicalize_condition(lits):
    if not lits:
        return ()
    scope = {}
    for n, o, rv in lits:
        if o not in _ALLOWED_OPS: raise ValueError(f"Unsupported op {o!r}")
        v, s = _normalise_value(rv), scope.setdefault(n, {"eq": None, "lt": None, "gt": None})
        if o == "==":
            if s["eq"] is not None and not _values_equal(s["eq"], v): return None
            s["eq"] = v
        elif o == "<": s["lt"] = min(s["lt"], float(v)) if s["lt"] is not None else float(v)
        elif o == ">": s["gt"] = max(s["gt"], float(v)) if s["gt"] is not None else float(v)

    res = []
    for n, s in sorted(scope.items()):
        eq, lt, gt = s["eq"], s["lt"], s["gt"]
        if eq is not None:
            if (lt is not None and float(eq) >= lt - _EPS) or (gt is not None and float(eq) <= gt + _EPS): return None
            res.append((n, "==", eq))
        else:
            if gt is not None and lt is not None and gt >= lt - _EPS: return None
            if gt is not None:
                res.append((n, ">", gt))
            if lt is not None:
                res.append((n, "<", lt))
    
    def sort_key(x):
        return (x[0], {"==": 0, ">": 1, "<": 2}[x[1]], repr(_normalise_value(x[2])))
        
    return tuple(sorted(res, key=sort_key))

def _condition_signature(c):
    parts = []
    for n, o, rv in c:
        v = _normalise_value(rv)
        v_str = _FLOAT_FMT.format(v) if isinstance(v, float) else str(v)
        parts.append(f"{n}{o}{v_str}")
    return "&&".join(parts)

def _condition_caption(c, vn, l=None):
    return condition_caption(c, vn, l)

def _mask_from_literal(c, o, v):
    if o == "==":
        if np.issubdtype(c.dtype, np.number):
            return np.isfinite(c) & np.isclose(c, float(v), atol=1e-9)
        return c == v
    
    if o == "<":
        return np.isfinite(c) & (c < float(v))
    return np.isfinite(c) & (c > float(v))

def _feature_score(c):
    if c.size == 0:
        return 0.0
    if np.issubdtype(c.dtype, np.number):
        fin_c = c[np.isfinite(c)]
        return float(np.var(fin_c)) if fin_c.size else 0.0
    
    v, cnt = np.unique(c, return_counts=True)
    if not cnt.size:
        return 0.0
    return float(1.0 - (cnt / cnt.sum()).max())

def _split_feature_kinds(X, names, decs, *, cat_threshold=32):
    num, cat = [], []
    for i, n in enumerate(names):
        is_categorical_v = (decs and n in decs) or not np.issubdtype(X[:, i].dtype, np.number)
        if is_categorical_v:
            cat.append(i)
            continue
            
        u = np.unique(X[:, i][np.isfinite(X[:, i])])
        is_discrete = u.size <= cat_threshold and np.all(np.isclose(u, np.round(u)))
        if is_discrete:
            cat.append(i)
        else:
            num.append(i)
    return num, cat


# ----------------------------- core generator -----------------------------
class RuleGenerator:
    def __init__(self, *, algo="ripper", verbose=False, **kwargs):
        algo = str(algo or "ripper").lower()
        if algo not in {"ripper", "foil", "static"}:
            raise ValueError("algo must be ripper/foil/static")
        self.algo = algo
        self.verbose = bool(verbose)
        self._MAX_RULES_PER_CLASS = int(kwargs.get("max_rules_per_class", 400))
        self._MAX_TOTAL_RULES = int(kwargs.get("max_total_rules", 1600))
        # Core growth / split thresholds (FOIL/RIPPER-style)
        self._MIN_GAIN, self._MIN_POS_SUPPORT = 1e-6, 1
        self._GROW_RATIO, self._NEG_RATIO = 0.67, 0.50
        # Numeric split quantiles (used for candidate thresholds)
        self._QUANTILES = kwargs.get("quantiles") or (0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95)
        self._NUMERIC_CANDIDATE_LIMIT = int(kwargs.get("numeric_candidate_limit", 24))
        self._MIDPOINT_LIMIT = int(kwargs.get("midpoint_limit", 12))
        self._STUMP_THRESHOLD_LIMIT = int(kwargs.get("stump_threshold_limit", 6))
        # Optional diversity filter is explicitly enabled via kwargs
        self._enable_diversity_filter = bool(kwargs.get("enable_diversity_filter", False))
        self._diversity_threshold = float(kwargs.get("diversity_threshold", 0.80))
        # Precision / length tiers (high → low precision) to control recall vs precision
        self._precision_tiers = kwargs.get("precision_tiers") or [
            {"min_precision": 0.80, "max_literals": 8, "min_literals": 2},
            {"min_precision": 0.60, "max_literals": 6, "min_literals": 1},
            {"min_precision": 0.40, "max_literals": 5, "min_literals": 1},
            {"min_precision": 0.20, "max_literals": 4, "min_literals": 1},
        ]
        self._enable_pool_shaping = bool(kwargs.get("enable_pool_shaping", False))
        self._pool_target_ratio = float(kwargs.get("pool_target_ratio", 0.7))
        self._pool_min_keep_per_class = int(kwargs.get("pool_min_keep_per_class", 8))
        self._pool_novelty_weight = float(kwargs.get("pool_novelty_weight", 0.12))
        self._pool_length_penalty = float(kwargs.get("pool_length_penalty", 0.02))
        self._rng = np.random.default_rng(int(kwargs.get("seed", 42)))
        self.ruleset = defaultdict(list)
        self._ordered_rules = []
        self._rule_metrics = {}
        
        self._X = None
        self._y = None
        self._default_label = None
        self._feat = []
        self._is_cat = []
        self._seen_signatures = set()
        self._rule_masks = []

    def _post_induction_layers(self) -> tuple[str, ...]:
        layers: list[str] = []
        if self._enable_diversity_filter:
            layers.append("diversity_filter")
        if self._enable_pool_shaping:
            layers.append("pool_shaping")
        return tuple(layers)

    def _filter_redundant_rules(self, rules, X, threshold=0.75):
        if not rules: return rules
        masks = []
        for r in rules:
            m = np.ones(X.shape[0], dtype=bool)
            for n, o, v in r["specs"]:
                idx = next((i for i, f in enumerate(self._feat) if f == n), None)
                if idx is not None:
                    try:
                        m &= _mask_from_literal(X[:, idx], o, v)
                    except Exception:
                        pass
            masks.append(m)
        kept = [0]
        for i in range(1, len(rules)):
            if all(((masks[i] & masks[j]).sum() / ((masks[i] | masks[j]).sum() + 1e-9)) <= threshold for j in kept): kept.append(i)
        if self.verbose and len(kept) < len(rules): print(f"[RuleGenerator] Diversity filter: {len(rules)} → {len(kept)} rules")
        return [rules[i] for i in kept]

    # ------------------------- static pool -------------------------
    def _stat_literal_pool(self, X, names, decs, y, *, breaks=7, top_k_cats=8):
        pool, (num, cat) = {}, _split_feature_kinds(X, names, decs, cat_threshold=max(20, top_k_cats * 2))
        qs = np.linspace(0.0, 1.0, max(3, breaks + 2))[1:-1]
        for i in num:
            fin = X[:, i][np.isfinite(X[:, i])]
            if fin.size < 4:
                continue
            thrs = np.unique(np.quantile(fin, np.unique(qs)))
            thrs = thrs[np.isfinite(thrs)]
            if not thrs.size:
                continue
            cleaned = [float(t) for j, t in enumerate(sorted(thrs)) if j == 0 or abs(t - sorted(thrs)[j-1]) > 1e-7]
            
            entries = []
            for t in cleaned:
                for op in ("<", ">"):
                    m = _mask_from_literal(X[:, i], op, t)
                    if m.sum() > 0:
                        entries.append({
                            "literal": (names[i], op, float(t)),
                            "mask": m,
                            "coverage": float(m.mean()),
                            "score": float(_feature_score(X[:, i]))
                        })
            pool[names[i]] = entries

        for i in cat:
            v, c = np.unique(X[:, i], return_counts=True)
            ch = v[np.argsort(-c)[:min(top_k_cats, len(v))]]
            entries = []
            for val in ch:
                m = _mask_from_literal(X[:, i], "==", val)
                if m.sum() > 0:
                    lit_val = resolve_rule_value(names[i], val, decs) if names[i] in decs else _normalise_value(val)
                    entries.append({
                        "literal": (names[i], "==", lit_val),
                        "mask": m,
                        "coverage": float(m.mean()),
                        "score": float(_feature_score(X[:, i]))
                    })
            pool[names[i]] = entries
        return pool

    def _static_single_rules_from_pool(self, pool, decs, seen, verbose):
        rules = []
        for n, entries in pool.items():
            sorted_entries = sorted(entries, key=lambda x: (float(x.get("score", 0)), float(x.get("coverage", 0))), reverse=True)
            for e in sorted_entries:
                cond = _canonicalize_condition((e["literal"],))
                if not cond:
                    continue
                sig = _condition_signature(cond)
                if sig in seen:
                    continue

                cap = _condition_caption(cond, decs, None)
                rules.append({
                    "specs": cond,
                    "label": None,
                    "caption": cap,
                    "stats": {"kind": "static", "stage": "single", "coverage": float(e.get("coverage", 0)), "literals": 1}
                })
                seen.add(sig)
                if verbose:
                    print(f"[RuleGenerator] emit(single): {cap}")
        return rules

    def generate_mult_pair_rules(self, X, *, feature_names, y=None, value_decoders=None, literal_pool=None, seen_signatures=None, pair_top=12, triple_top=24, max_rules=None, verbose=False):
        X_np, names, decs, y_np = np.asarray(X), list(feature_names), value_decoders or {}, np.asarray(y, dtype=int) if y is not None else None
        if X_np.ndim != 2: raise ValueError("X must be 2D")
        pool = literal_pool or self._stat_literal_pool(X_np, names, decs, y_np)
        seen, v_flag = seen_signatures or set(), bool(verbose or self.verbose)
        f_scores = [_feature_score(X_np[:, i]) for i in range(X_np.shape[1])]
        rules = []
        
        def _sel(entries, limit):
            return sorted(entries, key=lambda x: (float(x.get("score", 0)), float(x.get("coverage", 0))), reverse=True)[:limit]

        pairs = sorted(itertools.combinations(range(X_np.shape[1]), 2), key=lambda p: f_scores[p[0]] + f_scores[p[1]], reverse=True)[:max(1, pair_top)]
        for i, j in pairs:
            ci = _sel(pool.get(names[i], []), 3)
            cj = _sel(pool.get(names[j], []), 3)
            if not ci or not cj:
                continue
            for li, lj in itertools.product(ci, cj):
                cond = _canonicalize_condition((li["literal"], lj["literal"]))
                if not cond or len(cond) < 2:
                    continue
                sig = _condition_signature(cond)
                if sig in seen:
                    continue
                m = li["mask"] & lj["mask"]
                if m.sum() == 0:
                    continue
                
                cap = _condition_caption(cond, decs, None)
                rules.append({
                    "specs": cond,
                    "label": None,
                    "caption": cap,
                    "stats": {"kind": "static", "stage": "pair", "support": int(m.sum()), "coverage": float(m.mean()), "literals": len(cond)}
                })
                seen.add(sig)
                if v_flag:
                    print(f"[RuleGenerator] emit(pair): {cap}")
                if max_rules and len(rules) >= max_rules:
                    return rules

        top_f = [i for i, _ in sorted(enumerate(f_scores), key=lambda x: x[1], reverse=True)]
        trios = list(itertools.combinations(top_f[:max(3, min(len(top_f), triple_top))], 3))[:triple_top] if triple_top else []
        for trio in trios:
            c_sets = [_sel(pool.get(names[i], []), 3) for i in trio]
            if not c_sets or not all(c_sets):
                continue
            for combo in itertools.product(*c_sets):
                cond = _canonicalize_condition(tuple(e["literal"] for e in combo))
                if not cond or len(cond) < 3:
                    continue
                sig = _condition_signature(cond)
                if sig in seen:
                    continue
                
                m = combo[0]["mask"]
                for e in combo[1:]:
                    m = m & e["mask"]
                
                if m.sum() == 0:
                    continue
                
                cap = _condition_caption(cond, decs, None)
                rules.append({
                    "specs": cond,
                    "label": None,
                    "caption": cap,
                    "stats": {"kind": "static", "stage": "triple", "support": int(m.sum()), "coverage": float(m.mean()), "literals": len(cond)}
                })
                seen.add(sig)
                if v_flag:
                    print(f"[RuleGenerator] emit(triple): {cap}")
                if max_rules and len(rules) >= max_rules:
                    return rules
        return rules

    def build_static_rules(self, X, *, feature_names, y=None, value_decoders=None, verbose=False, breaks=7, top_k_cats=12, pair_top=8, triple_top=24, max_rules=2000):
        X_np, names, decs, y_np = np.asarray(X), list(feature_names), value_decoders or {}, np.asarray(y, dtype=int) if y is not None else None
        if X_np.ndim != 2: raise ValueError("X must be 2D")
        pool = self._stat_literal_pool(X_np, names, decs, y_np, breaks=breaks, top_k_cats=top_k_cats)
        seen, v_flag = set(), bool(verbose or self.verbose)
        s_rules = self._static_single_rules_from_pool(pool, decs, seen, v_flag)

        # Adaptive cap for statistical (static) rule generation to avoid rule explosion
        if max_rules is None:
            n_samples, n_features = X_np.shape
            cap_by_features = 40 * int(n_features)
            cap_by_samples = max(100, int(0.2 * int(n_samples)))
            max_rules_mp = min(self._MAX_TOTAL_RULES, cap_by_features, cap_by_samples)
            max_rules_mp = max(60, int(max_rules_mp))
            total_cap = max_rules_mp + len(s_rules)
        else:
            total_cap = int(max_rules)
            max_rules_mp = max(0, total_cap - len(s_rules))

        m_rules = self.generate_mult_pair_rules(
            X_np,
            feature_names=names,
            y=y_np,
            value_decoders=decs,
            literal_pool=pool,
            seen_signatures=seen,
            pair_top=pair_top,
            triple_top=triple_top,
            max_rules=max_rules_mp,
            verbose=v_flag,
        )
        all_r = list(m_rules) + list(s_rules)
        if self._enable_diversity_filter:
            self._feat = names; all_r = self._filter_redundant_rules(all_r, X_np, threshold=float(self._diversity_threshold))
        return all_r[:total_cap]

    # ------------------------- FOIL / RIPPER -------------------------
    @staticmethod
    def _is_categorical(col):
        if col.dtype.kind in {"U", "S", "O"}:
            return True
        col_fin = col[np.isfinite(col)] if col.dtype.kind == "f" else col
        return np.unique(col_fin).size <= 20

    def _subset_mask(self, m, r, *, minimum=1):
        idx = np.where(m)[0]
        if idx.size == 0:
            return np.zeros_like(m, dtype=bool)
        
        cnt = min(idx.size, max(minimum, int(round(idx.size * r))))
        out = np.zeros_like(m, dtype=bool)
        if cnt >= idx.size:
            out[idx] = True
            return out
        
        out[self._rng.choice(idx, size=cnt, replace=False)] = True
        return out

    def _split_positive_masks(self, pm):
        g = self._subset_mask(pm, self._GROW_RATIO)
        grow_set = g if g.any() else pm.copy()
        
        remainder = pm & ~g
        prune_set = remainder if remainder.any() else g.copy()
        return grow_set, prune_set

    def _split_negative_masks(self, nm):
        g = self._subset_mask(nm, self._NEG_RATIO)
        grow_set = g if g.any() else nm.copy()
        
        remainder = nm & ~g
        prune_set = remainder if remainder.any() else g.copy()
        return grow_set, prune_set

    def _literal_mask(self, fi, op, thr):
        c = self._X[:, fi]
        if op == "==": return np.isfinite(c) & np.isclose(c, float(thr), atol=1e-9) if np.issubdtype(c.dtype, np.number) else c == thr
        return np.isfinite(c) & (c < float(thr) if op == "<" else c > float(thr))

    @staticmethod
    def _precision(p, n, m):
        tp = int((m & p).sum())
        fp = int((m & n).sum())
        return tp / (tp + fp + 1e-12)

    @staticmethod
    def _foil_gain(p, n, cm, nm):
        p0 = int((cm & p).sum())
        n0 = int((cm & n).sum())
        new = cm & nm
        p1 = int((new & p).sum())
        if p1 < 1:
            return -math.inf
        n1 = int((new & n).sum())
        
        def lp(pos, neg):
            return math.log((pos + 1e-9) / (pos + neg + 1e-9))
            
        return p1 * (lp(p1, n1) - lp(p0, n0))

    @staticmethod
    def _unique_sorted_thresholds(values):
        arr = np.asarray(values, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.asarray([], dtype=float)
        arr = np.unique(arr)
        out = [float(arr[0])]
        for val in arr[1:]:
            if abs(float(val) - out[-1]) > 1e-9:
                out.append(float(val))
        return np.asarray(out, dtype=float)

    @staticmethod
    def _binary_gini(labels: np.ndarray) -> float:
        if labels.size == 0:
            return 0.0
        p = float(labels.mean())
        return 1.0 - (p * p) - ((1.0 - p) * (1.0 - p))

    @classmethod
    def _rank_thresholds_by_gain(
        cls,
        sorted_values: np.ndarray,
        sorted_labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if sorted_values.size < 2:
            return (
                np.asarray([], dtype=float),
                np.asarray([], dtype=float),
                np.asarray([], dtype=bool),
            )
        split_idx = np.where(np.abs(np.diff(sorted_values)) > 1e-12)[0]
        if split_idx.size == 0:
            return (
                np.asarray([], dtype=float),
                np.asarray([], dtype=float),
                np.asarray([], dtype=bool),
            )

        cum_pos = np.cumsum(sorted_labels.astype(float), dtype=float)
        total = float(sorted_labels.size)
        total_pos = float(cum_pos[-1])
        parent = cls._binary_gini(sorted_labels.astype(float))

        left_total = (split_idx + 1).astype(float)
        right_total = total - left_total
        left_pos = cum_pos[split_idx]
        right_pos = total_pos - left_pos

        left_prob = np.divide(left_pos, left_total, out=np.zeros_like(left_pos), where=left_total > 0)
        right_prob = np.divide(right_pos, right_total, out=np.zeros_like(right_pos), where=right_total > 0)
        left_gini = 1.0 - (left_prob * left_prob) - ((1.0 - left_prob) * (1.0 - left_prob))
        right_gini = 1.0 - (right_prob * right_prob) - ((1.0 - right_prob) * (1.0 - right_prob))
        child = (left_total / total) * left_gini + (right_total / total) * right_gini
        gains = parent - child
        thresholds = 0.5 * (sorted_values[split_idx] + sorted_values[split_idx + 1])
        label_change = sorted_labels[split_idx] != sorted_labels[split_idx + 1]
        return thresholds.astype(float), gains.astype(float), label_change.astype(bool)

    @staticmethod
    def _top_thresholds_by_score(thresholds: np.ndarray, gains: np.ndarray, limit: int) -> list[float]:
        if thresholds.size == 0:
            return []
        ranked = sorted(zip(thresholds.tolist(), gains.tolist()), key=lambda x: (x[1], -abs(x[0])), reverse=True)
        out: list[float] = []
        seen: set[str] = set()
        for thr, _ in ranked:
            key = f"{float(thr):.12g}"
            if key in seen:
                continue
            seen.add(key)
            out.append(float(thr))
            if len(out) >= limit:
                break
        return out

    def _numeric_threshold_candidates(self, xj: np.ndarray, pm: np.ndarray) -> list[float]:
        finite = np.isfinite(xj)
        values = xj[finite].astype(float, copy=False)
        if values.size == 0:
            return []

        candidates: list[float] = []
        candidates.extend(float(t) for t in np.quantile(values, self._QUANTILES))

        pos_values = xj[pm & np.isfinite(xj)]
        if pos_values.size:
            candidates.extend(float(t) for t in np.quantile(pos_values, self._QUANTILES))

        labels = pm[finite].astype(np.int8, copy=False)
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        sorted_labels = labels[order]
        thresholds, gains, label_change = self._rank_thresholds_by_gain(sorted_values, sorted_labels)
        if thresholds.size:
            if label_change.any():
                candidates.extend(
                    self._top_thresholds_by_score(
                        thresholds[label_change],
                        gains[label_change],
                        self._MIDPOINT_LIMIT,
                    )
                )
            candidates.extend(
                self._top_thresholds_by_score(
                    thresholds,
                    gains,
                    self._STUMP_THRESHOLD_LIMIT,
                )
            )

        ordered: list[float] = []
        seen: set[str] = set()
        for thr in candidates:
            if not np.isfinite(thr):
                continue
            key = f"{float(thr):.12g}"
            if key in seen:
                continue
            seen.add(key)
            ordered.append(float(thr))
            if len(ordered) >= self._NUMERIC_CANDIDATE_LIMIT:
                break
        return ordered

    def _candidates_for_feature(self, j, pm):
        xj = self._X[:, j]
        if self._is_cat[j]:
            vals = np.unique(xj[np.isfinite(xj)] if np.issubdtype(xj.dtype, np.number) else xj)
            return [("==", _normalise_value(v)) for v in vals]
        if (cl := xj[np.isfinite(xj)]).size == 0:
            return []
        ordered_thresholds = self._numeric_threshold_candidates(xj, pm)

        out = []
        neg_mask = ~pm
        for thr in ordered_thresholds:
            left = np.isfinite(xj) & (xj < float(thr))
            right = np.isfinite(xj) & (xj > float(thr))
            left_precision = self._precision(pm, neg_mask, left)
            right_precision = self._precision(pm, neg_mask, right)
            ops = (">", "<") if right_precision >= left_precision else ("<", ">")
            for op in ops:
                out.append((op, float(thr)))
        return out

    def _grow_rule(self, pos, neg, *, max_literals, min_literals, min_pos_support, min_gain):
        cond, curr, min_l, max_l = {}, np.ones_like(pos, dtype=bool), max(1, int(min_literals)), max(1, int(max_literals))
        while len(cond) < max_l and (len(cond) < min_l or (curr & neg).any()):
            bg, bc = -math.inf, None
            for j, n in enumerate(self._feat):
                if n in cond: continue
                for o, t in self._candidates_for_feature(j, pos):
                    lm = self._literal_mask(j, o, t); new = curr & lm
                    if int((new & pos).sum()) < min_pos_support: continue
                    g = self._foil_gain(pos, neg, curr, lm)
                    if g > bg:
                        bg = g
                        bc = (n, (o, t), lm)
            if not bc or bg < min_gain: break
            cond[bc[0]], curr = bc[1], curr & bc[2]
        return (cond, curr) if cond else None

    def _prune_score(self, pos: np.ndarray, neg: np.ndarray, mask: np.ndarray) -> float:
        p = int((mask & pos).sum())
        n = int((mask & neg).sum())
        if p + n == 0:
            return -math.inf
        total = max(1, int(pos.sum()) + int(neg.sum()))
        tn = int((~mask & neg).sum())
        accuracy = (p + tn) / total
        laplace = (p + 1.0) / (p + n + 2.0)
        return float(accuracy + 1e-3 * laplace)

    def _prune_rule(self, cond, prune_pos, prune_neg):
        if not cond: return None
        lits, best_metric, best_lits = list(cond.items()), -math.inf, []
        for k in range(1, len(lits) + 1):
            sub = dict(lits[:k]); m = np.ones(self._X.shape[0], dtype=bool)
            for n, (o, v) in sub.items():
                idx = next((i for i, f in enumerate(self._feat) if f == n), None)
                if idx is not None:
                    m &= self._literal_mask(idx, o, v)
            
            p = int((m & prune_pos).sum())
            if p == 0:
                continue
            
            metric = self._prune_score(prune_pos, prune_neg, m)
            if metric > best_metric:
                best_metric = metric
                best_lits = sub
                
        return best_lits if best_lits else None

    def _reduced_error_prune(self, cond, prune_pos, prune_neg):
        """Backward pruning wrapper to mirror RIPPER style reduced-error pruning."""
        return self._prune_rule(cond, prune_pos, prune_neg)

    def _record_rule(
        self,
        label: int,
        cond: dict[str, tuple[str, Any]],
        rule_mask: np.ndarray,
        stage: str,
        min_precision: float,
    ) -> bool:
        condition = tuple(sorted((name, spec[0], _normalise_value(spec[1])) for name, spec in cond.items()))
        signature = _condition_signature(condition)
        if signature in self._seen_signatures:
            return False

        pos = (self._y == label)
        neg = ~pos
        support = int((rule_mask & pos).sum())
        neg_cov = int((rule_mask & neg).sum())
        total_pos = int(pos.sum())
        precision = support / (support + neg_cov + 1e-12)
        # Precision filter enabled to improve raw metrics
        if precision < min_precision:
            return False

        recall = support / max(1, total_pos)
        denom = precision + recall + 1e-12
        f1 = (2.0 * precision * recall) / denom
        stats = {
            "kind": "learned",
            "stage": stage,
            "label": label,
            "core_induction": self.algo,
            "candidate_layer": "ranked_numeric_proposals",
            "prune_strategy": "none" if self.algo != "ripper" else "reduced_error",
            "post_layers": self._post_induction_layers(),
            "support": support,
            "neg_covered": neg_cov,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "coverage": float(rule_mask.mean()),
            "literals": len(cond),
            "min_precision": float(min_precision),
        }

        self._seen_signatures.add(signature)
        self.ruleset[label].append(dict(cond))
        self._ordered_rules.append((label, dict(cond), stats))
        self._rule_masks.append(rule_mask.copy())
        if self.verbose:
            caption = _condition_caption(condition, {}, label)
            print(
                f"[RuleGenerator] stage={stage} class={label} literals={len(cond)} support={support} "
                f"precision={precision:.3f} recall={recall:.3f} -> {caption}"
            )
        return True

    def _apply_pool_shaping(self) -> None:
        if not self._enable_pool_shaping or not self._ordered_rules:
            return

        selected_rules = []
        selected_masks = []
        grouped = defaultdict(list)
        for idx, (label, cond, stats) in enumerate(self._ordered_rules):
            grouped[int(label)].append((idx, dict(cond), dict(stats), self._rule_masks[idx]))

        for label in sorted(grouped):
            items = grouped[label]
            if not items:
                continue
            target = min(
                len(items),
                max(self._pool_min_keep_per_class, int(round(len(items) * self._pool_target_ratio))),
            )
            chosen: list[tuple[int, dict, dict, np.ndarray]] = []
            remaining = list(items)
            while remaining and len(chosen) < target:
                best = None
                best_score = -math.inf
                for item in remaining:
                    _, _, stats, mask = item
                    precision = float(stats.get("precision", 0.0))
                    recall = float(stats.get("recall", 0.0))
                    f1 = float(stats.get("f1", 0.0))
                    literals = int(stats.get("literals", 0))
                    if chosen:
                        overlaps = []
                        for _, _, _, chosen_mask in chosen:
                            denom = float((mask | chosen_mask).sum())
                            overlaps.append(float((mask & chosen_mask).sum()) / denom if denom else 0.0)
                        novelty = 1.0 - max(overlaps)
                    else:
                        novelty = 1.0
                    score = (
                        precision
                        + 0.25 * recall
                        + 0.15 * f1
                        + self._pool_novelty_weight * novelty
                        - self._pool_length_penalty * max(0, literals - 1)
                    )
                    if score > best_score:
                        best = item
                        best_score = score
                if best is None:
                    break
                remaining.remove(best)
                idx, cond, stats, mask = best
                shaped_stats = dict(stats)
                shaped_stats["pool_score"] = float(best_score)
                shaped_stats["pool_shaped"] = True
                chosen.append((idx, cond, shaped_stats, mask))
            for _, cond, stats, mask in chosen:
                selected_rules.append((label, cond, stats))
                selected_masks.append(mask)

        if selected_rules:
            if self.verbose:
                print(f"[RuleGenerator] Pool shaping: {len(self._ordered_rules)} -> {len(selected_rules)} rules")
            self._ordered_rules = selected_rules
            self._rule_masks = selected_masks
            self.ruleset.clear()
            self._seen_signatures.clear()
            for label, cond, _ in self._ordered_rules:
                self.ruleset[int(label)].append(dict(cond))
                signature = _condition_signature(tuple(sorted((name, spec[0], _normalise_value(spec[1])) for name, spec in cond.items())))
                self._seen_signatures.add(signature)

    def _rules_for_class(self, cls: int) -> None:
        pos_all = (self._y == cls)
        neg_all = ~pos_all
        if not pos_all.any():
            return

        covered_pos = np.zeros_like(pos_all, dtype=bool)
        pos = pos_all.copy()
        per_class_count = 0

        # Adaptive precision tiers: start strict, then relax; no hard stages.
        for tier in self._precision_tiers:
            if per_class_count >= self._MAX_RULES_PER_CLASS or not pos.any():
                break

            tier_budget = max(1, int(self._MAX_RULES_PER_CLASS / len(self._precision_tiers)))
            tier_budget = min(tier_budget, self._MAX_RULES_PER_CLASS - per_class_count)
            added = 0
            failures = 0

            while pos.any() and added < tier_budget and failures < 10:
                if len(self._ordered_rules) >= self._MAX_TOTAL_RULES:
                    break

                # Split for RIPPER; FOIL uses full set
                if self.algo == "ripper":
                    grow_pos, prune_pos = self._split_positive_masks(pos)
                    grow_neg, prune_neg = self._split_negative_masks(neg_all)
                    if not grow_pos.any():
                        grow_pos, prune_pos = pos, pos
                        grow_neg, prune_neg = neg_all, neg_all
                else:
                    grow_pos, prune_pos = pos, pos
                    grow_neg, prune_neg = neg_all, neg_all

                grown = self._grow_rule(
                    grow_pos,
                    grow_neg,
                    max_literals=int(tier["max_literals"]),
                    min_literals=int(tier["min_literals"]),
                    min_pos_support=self._MIN_POS_SUPPORT,
                    min_gain=self._MIN_GAIN,
                )
                if grown is None:
                    failures += 1
                    continue

                cond, _ = grown
                if self.algo == "ripper":
                    cond = self._reduced_error_prune(cond, prune_pos=prune_pos, prune_neg=prune_neg)
                if not cond:
                    failures += 1
                    continue

                rule_mask = np.ones_like(pos, dtype=bool)
                for name, (op, val) in cond.items():
                    idx = next((i for i, f in enumerate(self._feat) if f == name), None)
                    if idx is None:
                        rule_mask = np.zeros_like(pos, dtype=bool); break
                    rule_mask &= self._literal_mask(idx, op, val)

                if not (rule_mask & pos).any():
                    failures += 1
                    continue

                added_flag = self._record_rule(
                    cls,
                    cond,
                    rule_mask,
                    stage=f"tier_{tier['min_precision']:.2f}",
                    min_precision=float(tier["min_precision"]),
                )
                if added_flag:
                    per_class_count += 1
                    added += 1
                    covered_pos |= (rule_mask & pos_all)
                    pos &= ~rule_mask
                    failures = 0
                else:
                    failures += 1

    def fit(self, X, y, *, feature_names=None, value_decoders=None):
        self._X, self._y = np.asarray(X), np.asarray(y, dtype=int)
        if self._X.ndim != 2: raise ValueError("X must be 2D")
        self._feat = list(feature_names) if feature_names else [f"X{i}" for i in range(self._X.shape[1])]
        decs = value_decoders or {}
        self._is_cat = [bool(self._feat[i] in decs) or self._is_categorical(self._X[:, i]) for i in range(self._X.shape[1])]
        self._default_label = int(np.argmax(np.bincount(self._y)))
        self.ruleset.clear(); self._ordered_rules.clear(); self._seen_signatures.clear(); self._rule_masks.clear()
        # Tiered adaptive search (high → low precision) without rigid stages
        classes = sorted(np.unique(self._y), key=lambda c: (self._y == c).sum())
        for cls in classes:
            self._rules_for_class(cls)
        self._apply_pool_shaping()
        return self

    @property
    def ordered_rules(self):
        return list(self._ordered_rules)

    @property
    def default_label(self):
        return self._default_label

    def generate(self, X, y, **kwargs):
        """Unified interface to generate rules (STATIC/RIPPER/FOIL)."""
        feature_names = kwargs.get("feature_names")
        value_decoders = kwargs.get("value_decoders") or {}
        
        if feature_names:
            self._feat = list(feature_names) 
            
        if self.algo == "static":
            # Extract only relevant args to avoid TypeError
            valid_args = ["breaks", "top_k_cats", "pair_top", "triple_top", "max_rules"]
            sub_kw = {k: v for k, v in kwargs.items() if k in valid_args}
            return self.build_static_rules(
                X, 
                feature_names=self._feat if hasattr(self, '_feat') else [f"X{i}" for i in range(X.shape[1])],
                y=y, 
                value_decoders=value_decoders, 
                verbose=self.verbose,
                **sub_kw
            )
            
        # RIPPER / FOIL
        self.fit(X, y, feature_names=feature_names, value_decoders=value_decoders)
        
        res = []
        for label, cond_dict, stats in self.ordered_rules:
            # Convert {n: (o, v)} -> ((n, o, v), ...) sorted by name
            specs = []
            for name in sorted(cond_dict.keys()):
                op, val = cond_dict[name]
                specs.append((name, op, val))
            
            cap = _condition_caption(specs, value_decoders, label)
            
            res.append({
                "specs": tuple(specs),
                "label": int(label),
                "caption": cap,
                "stats": stats
            })
        return res

__all__ = ["RuleGenerator", "Literal", "Condition"]
