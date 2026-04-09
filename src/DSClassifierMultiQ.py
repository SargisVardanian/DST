"""
Sklearn-style wrapper around DSModelMultiQ with PGD training loop.

This module provides a scikit-learn compatible classifier interface for the DST model,
including training via Projected Gradient Descent (PGD), early stopping, and class balancing.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

try:
    from .DSModelMultiQ import DSModelMultiQ
except ImportError:  # pragma: no cover - direct script/import fallback
    from DSModelMultiQ import DSModelMultiQ


@dataclass
class TrainConfig:
    max_epochs: int = 50
    batch_size: int = 512
    lr: float = 2e-3
    weight_decay: float = 2e-4
    val_split: float = 0.2
    seed: int = 42
    early_stop_patience: int = 10
    class_weight_power: float = 0.5
    class_weight_beta: float = 0.999
    verbose: bool = True
    enable_binary_threshold_tuning: bool = False
    optimizer_name: str = "adamw"
    record_history: bool = True


def _as_numpy(x):
    """Convert input to numpy array if needed."""
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int = None, **kwargs) -> dict[str, float]:
    """Compute basic classification metrics."""
    if k is None:
        unique = np.unique(y_true)
        k = len(unique)
    
    avg = 'binary' if k == 2 else 'weighted'
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        "F1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
    }


class DSClassifierMultiQ(ClassifierMixin):
    """
    Scikit-learn compatible DST classifier with uncertainty quantification.

    This classifier combines rule-based learning with Dempster-Shafer Theory
    to provide predictions with quantified uncertainty. It supports multiple
    rule generation algorithms (STATIC, RIPPER, FOIL) and trains mass functions
    using Projected Gradient Descent.
    """

    def __init__(
        self,
        k,
        train_cfg: TrainConfig | None = None,
        device="auto",
        rule_algo="STATIC",
        max_iter=50,
        batch_size=512,
        lr=2e-3,
        val_split=0.2,
        seed=42,
        print_every=5,
        weight_decay=2e-4,
        class_weight_power=0.5,
        grad_clip=1.0,
        early_stop_patience=10,
        combination_rule="dempster",
        debug=False,
        rule_uncertainty=0.8,
        lossfn: str = "MSE",
        uncertainty_rule_weight: float = 0.1,
        rule_gen_params: dict | None = None,
        init_mode: str = "dsgdpp",
        sort_rules_by_certainty: bool = False,
        certainty_score_mode: str = "certainty_label_mass",
    ) -> None:
        """Initialize the DST classifier."""
        cfg = train_cfg or TrainConfig(
            max_epochs=max_iter,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            val_split=val_split,
            seed=seed,
            early_stop_patience=early_stop_patience,
            class_weight_power=class_weight_power,
            verbose=bool(print_every),
        )
        self.train_cfg = copy.deepcopy(cfg)
        self.k = int(k)
        self.device = str(device)
        self.rule_algo = str(rule_algo or "STATIC").upper()
        self.max_iter = int(getattr(cfg, "max_epochs", max_iter))
        self.batch_size = int(getattr(cfg, "batch_size", batch_size))
        self.lr = float(getattr(cfg, "lr", lr))
        self.val_split = float(getattr(cfg, "val_split", val_split))
        self.seed = int(getattr(cfg, "seed", seed))
        self.print_every = int(print_every if print_every is not None else 5)
        if bool(getattr(cfg, "verbose", True)) and self.print_every <= 0:
            self.print_every = 5
        if not bool(getattr(cfg, "verbose", True)):
            self.print_every = 0
        self.weight_decay = float(getattr(cfg, "weight_decay", weight_decay))
        self.class_weight_power = float(max(0.0, getattr(cfg, "class_weight_power", class_weight_power)))
        self.grad_clip = float(max(0.0, grad_clip))
        self.early_stop_patience = max(1, int(getattr(cfg, "early_stop_patience", early_stop_patience)))
        self.debug = bool(debug)
        self.lossfn = str(lossfn or "MSE").upper()
        self.uncertainty_rule_weight = float(max(0.0, uncertainty_rule_weight))
        self.rule_gen_params = rule_gen_params or {
            "enable_diversity_filter": True,
            "diversity_threshold": 0.82,
            "pair_top": 12,
            "triple_top": 32,
        }
        self.sort_rules_by_certainty = bool(sort_rules_by_certainty)
        self.certainty_score_mode = str(certainty_score_mode or "certainty_label_mass")
        self._trained = False
        self.init_mode = str(init_mode or "dsgdpp").lower()
        
        self.combination_rule = str(combination_rule or "dempster").lower()
        if self.combination_rule not in {"dempster", "yager", "vote"}:
            raise ValueError("combination_rule must be 'dempster', 'yager', or 'vote'")

        self.model = DSModelMultiQ(
            k=self.k,
            algo=self.rule_algo,
            device=self.device,
            rule_uncertainty=rule_uncertainty,
            combination_rule=self.combination_rule,
        )

        self.default_class_ = 0
        self.history_ = []
        self.best_epoch_ = None
        self.best_val_metrics_ = None
        self._last_fit_meta = {}

    # --------------------------- Rule generation ---------------------------
    def generate_rules(self, X, y=None, feature_names=None, rule_algo: str | None = None, **kwargs) -> None:
        """Explicitly generate classification rules."""
        algo = rule_algo or self.rule_algo
        self.model.algo = str(algo).upper()
        verbose_flag = bool(kwargs.pop("verbose", kwargs.pop("verbose_rules", False)))
        kwargs.pop("algo", None)

        self.model.generate_rules(
            X,
            y,
            feature_names=feature_names,
            algo=self.model.algo,
            verbose=verbose_flag,
            **kwargs,
        )
        self._trained = False

    # --------------------------- Training ---------------------------
    def fit(self, X, y, feature_names=None, value_decoders=None, rule_params: dict | None = None, **kwargs):
        """Fit the DST model: generate rules (if needed) and train weights.

        Extra kwargs are accepted for compatibility with the benchmark runner.
        """
        X = _as_numpy(X)
        y = _as_numpy(y).astype(int)
        algo = kwargs.pop("algo", None)
        if algo is not None:
            self.rule_algo = str(algo).upper()
            self.model.algo = self.rule_algo
        rule_gen_params = kwargs.pop("rule_gen_params", None)
        on_rule = kwargs.pop("on_rule", None)
        rules_path = kwargs.pop("rules_path", "")
        use_cached_rules = bool(kwargs.pop("use_cached_rules", False))
        fail_on_cache_mismatch = bool(kwargs.pop("fail_on_cache_mismatch", False))
        save_rules_path = kwargs.pop("save_rules_path", "")
        kwargs.pop("verify_raw_on", None)

        fit_meta = {
            "rule_source": "generated",
            "loaded_from": None,
            "saved_to": None,
            "n_rules": 0,
            "cache_misses": [],
            "cache_failures": [],
            "warnings": [],
        }

        candidate_cache_paths = []
        if isinstance(rules_path, (list, tuple)):
            for candidate in rules_path:
                if str(candidate).strip():
                    candidate_cache_paths.append(str(candidate))
        elif str(rules_path).strip():
            candidate_cache_paths.append(str(rules_path))

        if use_cached_rules:
            for candidate in candidate_cache_paths:
                try:
                    self.load_model(str(candidate))
                    fit_meta["rule_source"] = "cache"
                    fit_meta["loaded_from"] = str(candidate)
                    fit_meta["n_rules"] = int(len(self.model.rules or []))
                    if fit_meta["cache_failures"]:
                        print(
                            f"[warn] incompatible cached artifacts were rejected before using {candidate}: "
                            + "; ".join(str(msg) for msg in fit_meta["cache_failures"])
                        )
                    elif fit_meta["cache_misses"]:
                        print(
                            f"[info] cache miss on requested path(s); using fallback cached artifact {candidate}"
                        )
                    self._last_fit_meta = dict(fit_meta)
                    return dict(fit_meta)
                except FileNotFoundError as exc:
                    msg = f"{candidate}: {type(exc).__name__}: {exc}"
                    fit_meta["cache_misses"].append(msg)
                except Exception as exc:
                    msg = f"{candidate}: {type(exc).__name__}: {exc}"
                    fit_meta["cache_failures"].append(msg)
            if fit_meta["cache_failures"]:
                if fail_on_cache_mismatch:
                    raise RuntimeError(
                        "incompatible cached artifacts were rejected and fail_on_cache_mismatch=True: "
                        + "; ".join(str(msg) for msg in fit_meta["cache_failures"])
                    )
                print(
                    "[warn] incompatible cached artifacts were rejected; regenerating rules. "
                    + "; ".join(str(msg) for msg in fit_meta["cache_failures"])
                )
            elif fit_meta["cache_misses"]:
                print("[info] no matching cached artifact found; generating rules for this seed.")

        # Best-effort reproducibility when this class is used standalone (outside benchmark scripts).
        try:
            random.seed(int(self.seed))
        except Exception:
            pass
        try:
            np.random.seed(int(self.seed))
        except Exception:
            pass
        try:
            torch.manual_seed(int(self.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.seed))
        except Exception:
            pass

        if feature_names is not None:
            self.model.feature_names = list(feature_names)
            self.model._feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        if value_decoders is not None:
            self.model.value_names = value_decoders

        # 1. Automatic Rule Generation
        if not self.model.rules:
            if self.debug:
                print(f"[DSClassifier] Generating rules using {self.rule_algo}...")
            
            gen_params = self.rule_gen_params.copy()
            if rule_gen_params:
                gen_params.update(rule_gen_params)
            if rule_params:
                gen_params.update(rule_params)
            if on_rule is not None:
                gen_params["on_emit_rule"] = on_rule
            
            if self.rule_algo == "STATIC":
                gen_params.setdefault("enable_diversity_filter", True)
                gen_params.setdefault("diversity_threshold", 0.82)
                gen_params.setdefault("pair_top", 12)
                gen_params.setdefault("triple_top", 32)
            gen_params.setdefault("seed", self.seed)
                
            fnames = feature_names if feature_names is not None else self.model.feature_names
            self.generate_rules(
                X,
                y,
                feature_names=fnames,
                rule_algo=self.rule_algo,
                **gen_params
            )

        # 2. Training Loop preparation
        if X.shape[0] < 2:
            print("[warn] need at least two samples to fit")
            return self

        rng = np.random.default_rng(self.seed)
        indices = np.arange(len(X))
        rng.shuffle(indices)

        val_count = max(1, int(round(self.val_split * len(X)))) if self.val_split > 0 else 0
        if val_count >= len(X):
            val_count = len(X) - 1
            
        train_idx = indices[val_count:]
        val_idx = indices[:val_count]
        
        if train_idx.size == 0:
            print("[warn] empty train split")
            return self

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = (X[train_idx], y[train_idx]) if val_count == 0 else (X[val_idx], y[val_idx])

        params = [p for p in [self.model.rule_mass_params] if p is not None and p.requires_grad]
        if not params:
            self._trained = True
            fit_meta["n_rules"] = int(len(self.model.rules or []))
            if str(save_rules_path).strip():
                self.save_model(str(save_rules_path))
                fit_meta["saved_to"] = str(save_rules_path)
            self._last_fit_meta = dict(fit_meta)
            return dict(fit_meta)

        device = torch.device(getattr(self.model, "device", self.device))
        X_train_t = torch.from_numpy(X_train)
        y_train_t = torch.from_numpy(y_train.astype(np.int64, copy=False))

        # Init masses (DSGD++ if requested).
        try:
            # DSGD++ init does not require rule labels; it uses (rule coverage × class purity)
            # estimated from (X_train, y_train), so it is valid for STATIC as well.
            use_dsgdpp = (self.init_mode == "dsgdpp")
            if use_dsgdpp:
                self.model.init_masses_dsgdpp(
                    X_train_t.to(device=device, dtype=torch.float32),
                    y_train_t.to(device=device),
                )
            else:
                self.model.reset_masses()
        except Exception as e:
            msg = f"{self.init_mode} init failed ({type(e).__name__}: {e}); falling back to reset_masses()"
            fit_meta["warnings"].append(msg)
            print(f"[warn] {msg}")
            self.model.reset_masses()

        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        # Class weights
        class_counts = np.bincount(y_train, minlength=self.k).astype(np.float32)
        class_counts[class_counts == 0.0] = 1.0
        inv_freq = class_counts.sum() / class_counts
        class_weights = inv_freq ** self.class_weight_power
        class_weights_t = torch.from_numpy(class_weights).to(device=device, dtype=torch.float32)
        
        # Class prior
        exp = 0.5 + max(self.class_weight_power, 0.0)
        prior_raw = inv_freq ** exp
        prior = prior_raw / (prior_raw.sum() + 1e-12)
        try:
            self.model.class_prior = torch.from_numpy(prior).to(device=device, dtype=torch.float32)
        except Exception:
            self.model.class_prior = None

        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        
        X_val_t = torch.from_numpy(X_val).to(device=device, dtype=torch.float32)
        y_val_t = torch.from_numpy(y_val.astype(np.int64, copy=False)).to(device=device)

        best_state, best_val = None, float("inf")
        stale_epochs = 0
        
        for epoch in range(1, self.max_iter + 1):
            self.model.train()
            total_loss, batches = 0.0, 0

            for xb, yb in loader:
                xb = xb.to(device=device, dtype=torch.float32)
                yb = yb.to(device=device, dtype=torch.int64)

                optimizer.zero_grad(set_to_none=True)
                
                # Training usually uses the initialized combination rule (mostly Dempster)
                probs = torch.clamp(
                    self.model.forward(xb, combination_rule="dempster"),
                    min=1e-9,
                )

                if self.lossfn == "CE":
                    nll = torch.nn.functional.nll_loss(torch.log(probs), yb, reduction="none")
                    if self.class_weight_power > 0.0:
                        nll = nll * class_weights_t[yb]
                    loss = nll.mean()
                else: # MSE
                    y_onehot = torch.nn.functional.one_hot(yb, num_classes=self.k).to(device=device, dtype=torch.float32)
                    if self.class_weight_power > 0.0:
                        w = class_weights_t[yb].unsqueeze(1)
                        loss = (w * (probs - y_onehot).pow(2)).mean()
                    else:
                        loss = torch.nn.functional.mse_loss(probs, y_onehot)

                loss.backward()
                if self.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
                optimizer.step()
                self.model.project_rules_to_simplex()

                total_loss += float(loss.detach())
                batches += 1

            train_loss = total_loss / max(1, batches)

            # Validation
            self.model.eval()
            with torch.inference_mode():
                probs_val = torch.clamp(self.model.forward(X_val_t, combination_rule="dempster"), min=1e-9)
                
                if self.lossfn == "CE":
                    nll_val = torch.nn.functional.nll_loss(torch.log(probs_val), y_val_t, reduction="none")
                    if self.class_weight_power > 0.0:
                        nll_val = nll_val * class_weights_t[y_val_t]
                    val_loss = nll_val.mean()
                else:
                    y_val_onehot = torch.nn.functional.one_hot(y_val_t, num_classes=self.k).to(device=device, dtype=torch.float32)
                    if self.class_weight_power > 0.0:
                        w_val = class_weights_t[y_val_t].unsqueeze(1)
                        val_loss = (w_val * (probs_val - y_val_onehot).pow(2)).mean()
                    else:
                        val_loss = torch.nn.functional.mse_loss(probs_val, y_val_onehot)

                val_loss_float = float(val_loss.detach().cpu())
                preds_val = probs_val.argmax(dim=1)
                
                # Uncertainty metrics for logging
                unc_stats = self.model.uncertainty_stats(X_val_t, combination_rule="dempster")
                unc_rule_avg = float(np.nanmean(unc_stats["unc_rule"]))

            metrics_val = _metrics(y_val, preds_val.detach().cpu().numpy(), k=self.k)
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss_float,
                "val_unc_rule": unc_rule_avg,
                **metrics_val,
            }
            self.history_.append(record)

            improved = val_loss_float < best_val - 1e-6
            if improved:
                best_val = val_loss_float
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch_ = epoch
                self.best_val_metrics_ = record
                stale_epochs = 0
            else:
                stale_epochs += 1

            if epoch == 1 or epoch % max(1, self.print_every) == 0 or improved:
                print(
                    f"[epoch {epoch:03d}] train_loss={train_loss:.5f} val_loss={val_loss_float:.5f} "
                    f"Acc={metrics_val['Accuracy']:.4f} F1={metrics_val['F1']:.4f} "
                    f"P={metrics_val['Precision']:.4f} R={metrics_val['Recall']:.4f} "
                    f"unc_rule={unc_rule_avg:.4f}"
                )

            if stale_epochs >= self.early_stop_patience:
                if self.print_every:
                    print(f"[early-stop] patience {self.early_stop_patience} reached at epoch {epoch:03d}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.eval()

        binc = np.bincount(y_train)
        if binc.size:
            self.default_class_ = int(binc.argmax())

        # Optional: reorder learned rules by certainty score (can affect FP results slightly).
        if self.sort_rules_by_certainty:
            try:
                self.model.sort_rules_by_certainty(
                    descending=True,
                    score_mode=self.certainty_score_mode,
                )
            except Exception as e:
                if self.debug:
                    print(f"[warn] sort_rules_by_certainty failed: {e}")

        self._trained = True
        fit_meta["n_rules"] = int(len(self.model.rules or []))
        if str(save_rules_path).strip():
            self.save_model(str(save_rules_path))
            fit_meta["saved_to"] = str(save_rules_path)
        self._last_fit_meta = dict(fit_meta)
        return dict(fit_meta)

    @torch.no_grad()
    # --------------------------- Prediction ---------------------------
    def predict(self, X, *, default_label=None, combination_rule: str | None = None, use_initial_masses=None):
        """Predict class labels for input samples."""
        X_np = _as_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)

        rule = combination_rule or self.combination_rule
        return np.asarray(self.model.predict_dst_labels(
            X_np,
            use_initial_masses=bool(use_initial_masses),
            combination_rule=rule,
        ), dtype=int)

    @torch.no_grad()
    def predict_proba(self, X, *, combination_rule: str | None = None, use_initial_masses=None):
        """Predict class probabilities for input samples."""
        X_np = _as_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
        rule = combination_rule or self.combination_rule
        return np.asarray(
            self.model.predict_with_dst(
                X_np,
                use_initial_masses=bool(use_initial_masses),
                combination_rule=rule,
            ),
            dtype=float,
        )

    @torch.no_grad()
    def raw_predict_proba(self, X, *, method: str = "weighted_vote", default_label=None):
        """Compatibility wrapper for rule-only baselines used by the runner and inspector."""
        X_np = _as_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)

        method_l = str(method or "weighted_vote").lower()
        if method_l == "weighted_vote":
            return np.asarray(self.model.predict_with_dst(X_np, combination_rule="vote"), dtype=float)
        if method_l == "native_ordered_rule":
            pred = np.asarray(
                self.model.predict_by_first_rule(
                    X_np,
                    default_label=self.default_class_ if default_label is None else default_label,
                ),
                dtype=int,
            )
            proba = np.zeros((len(pred), self.k), dtype=float)
            proba[np.arange(len(pred)), pred] = 1.0
            return proba
        if method_l == "first_hit_laplace":
            pred = np.asarray(
                self.model.predict_by_first_rule(
                    X_np,
                    default_label=self.default_class_ if default_label is None else default_label,
                ),
                dtype=int,
            )
            proba = np.full((len(pred), self.k), 1.0 / max(1, self.k), dtype=float)
            proba *= 0.1
            proba[np.arange(len(pred)), pred] = 0.9
            proba = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)
            return proba
        raise ValueError(f"Unknown raw prediction method: {method}")

    @torch.no_grad()
    def raw_predict(self, X, *, method: str = "weighted_vote", default_label=None):
        """Compatibility wrapper for rule-only baseline labels."""
        X_np = _as_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)

        method_l = str(method or "weighted_vote").lower()
        if method_l == "weighted_vote":
            return np.asarray(
                self.model.predict_by_rule_vote(
                    X_np,
                    default_label=self.default_class_ if default_label is None else default_label,
                ),
                dtype=int,
            )
        if method_l == "native_ordered_rule":
            return np.asarray(
                self.model.predict_by_first_rule(
                    X_np,
                    default_label=self.default_class_ if default_label is None else default_label,
                ),
                dtype=int,
            )
        if method_l == "first_hit_laplace":
            return np.asarray(
                self.model.predict_by_first_rule(
                    X_np,
                    default_label=self.default_class_ if default_label is None else default_label,
                ),
                dtype=int,
            )
        raise ValueError(f"Unknown raw prediction method: {method}")

    # --------------------------- Persistence ---------------------------
    def save_model(self, path: str) -> None:
        """Save model rules and parameters to binary file."""
        self.model.save_rules_bin(path)

    def load_model(self, path: str) -> None:
        """Load model rules and parameters from binary file."""
        self.model.load_rules_bin(path)
        self.rule_algo = getattr(self.model, "algo", self.rule_algo)
        self._trained = self.model.rule_mass_params is not None
        self.model.eval()

    def prepare_rules_for_export(self, sample=None):
        """Prepare rules and predictions for export."""
        return self.model.prepare_rules_for_export(sample)
