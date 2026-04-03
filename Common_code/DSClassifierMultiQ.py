"""
DSClassifierMultiQ

Lean DST classifier wrapper:
- frozen rule generation/loading (RIPPER / FOIL / STATIC)
- DSGD mass training with a class-balanced objective and objective-based checkpointing
- raw baselines remain unchanged (fair comparison protocol)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from DSModelMultiQ import DSModelMultiQ
from device_utils import resolve_torch_device


@dataclass
class TrainConfig:
    max_epochs: int = 100
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 2e-4
    val_split: float = 0.2
    seed: int = 42
    early_stop_patience: int = 20
    min_train_epochs: int = 15
    min_train_obj_delta: float = 1e-4
    verbose: bool = True
    class_weight_power: float = 0.35
    class_weight_beta: float = 0.999
    optimizer_name: str = "adamw"
    record_history: bool = True
    # Legacy compatibility field.
    nll_weight: float = 1.0
    enable_binary_threshold_tuning: bool = False


class DSClassifierMultiQ:
    def __init__(
        self,
        k: int,
        *,
        rule_algo: Optional[str] = None,
        device: str = "auto",
        rule_uncertainty: float = 0.65,
        combination_rule: str = "dempster",
        train_cfg: Optional[TrainConfig] = None,
        **_ignored_kwargs,
    ):
        self.k = int(k)
        self.device = resolve_torch_device(device)
        self.rule_uncertainty = float(rule_uncertainty)
        self.combination_rule = str(combination_rule)
        self.train_cfg = train_cfg or TrainConfig()
        self.default_rule_algo = str(rule_algo).upper() if rule_algo is not None else None

        self.model: Optional[DSModelMultiQ] = None
        self.rules_ready: bool = False
        self.trained: bool = False
        self._last_fit_meta: Dict[str, Any] = {}
        self._decision_threshold: Optional[float] = None

    def _ensure_model(self, *, feature_names=None, value_decoders=None, algo: str = "STATIC"):
        if self.model is None:
            self.model = DSModelMultiQ(
                self.k,
                algo=algo,
                device=self.device,
                feature_names=feature_names,
                value_decoders=value_decoders,
                rule_uncertainty=self.rule_uncertainty,
                combination_rule=self.combination_rule,
            )
        else:
            self.model.algo = str(algo or self.model.algo).upper()
            if feature_names is not None:
                self.model.feature_names = list(feature_names)
                self.model._feature_to_idx = {name: idx for idx, name in enumerate(self.model.feature_names)}
            if value_decoders is not None:
                self.model.value_names = value_decoders
            self.model.device = resolve_torch_device(self.device)
            self.model.to(self.model.device)

    def _require_ready_model(self) -> DSModelMultiQ:
        if self.model is None or not self.rules_ready:
            raise RuntimeError("Rules are not ready. Call build_rule_base()/generate_rules() or load_rules() first.")
        return self.model

    @staticmethod
    def _make_optimizer(model: DSModelMultiQ, cfg: TrainConfig) -> torch.optim.Optimizer:
        name = str(getattr(cfg, "optimizer_name", "adamw") or "adamw").strip().lower()
        if name == "adam":
            return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def _snapshot_raw_baselines(
        self,
        X: np.ndarray,
        *,
        methods: tuple[str, ...],
    ) -> Dict[str, np.ndarray]:
        model = self._require_ready_model()
        return {
            method: np.asarray(model.predict_rule_baseline_proba(X, method=method), dtype=float)
            for method in methods
        }

    def build_rule_base(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        algo: str = "RIPPER",
        feature_names=None,
        value_decoders=None,
        rule_gen_params: Optional[Dict[str, Any]] = None,
        on_rule: Optional[Callable[[int], None]] = None,
    ) -> None:
        self._ensure_model(feature_names=feature_names, value_decoders=value_decoders, algo=algo)
        assert self.model is not None

        params = dict(rule_gen_params or {})
        if on_rule is not None:
            params["on_emit_rule"] = on_rule

        self.model.build_rule_base(
            X,
            y,
            feature_names=feature_names,
            algo=algo,
            **params,
        )
        self.rules_ready = True
        self.trained = False

    def generate_rules(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        algo: str = "RIPPER",
        feature_names=None,
        value_decoders=None,
        rule_gen_params: Optional[Dict[str, Any]] = None,
        on_rule: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Backward-compatible wrapper around build_rule_base()."""
        self.build_rule_base(
            X,
            y,
            algo=algo,
            feature_names=feature_names,
            value_decoders=value_decoders,
            rule_gen_params=rule_gen_params,
            on_rule=on_rule,
        )

    def load_rules(self, rules_path: str, *, device: Optional[str] = None) -> None:
        if device is not None:
            self.device = resolve_torch_device(device)
        self._ensure_model()
        assert self.model is not None
        self.model.load_rules_bin(rules_path)
        self.model.combination_rule = self.model._normalize_combination_rule(self.combination_rule)
        self.model.device = resolve_torch_device(self.device)
        self.model.to(self.model.device)
        self.rules_ready = True
        self.trained = False

    def save_rules(self, rules_path: str) -> None:
        assert self.model is not None and self.rules_ready
        self.model.save_rules_bin(rules_path)

    def load_model(self, model_path: str, *, device: Optional[str] = None) -> None:
        if device is not None:
            self.device = resolve_torch_device(device)
        self._ensure_model()
        assert self.model is not None
        self.model.load_rules_bin(model_path)
        self.model.combination_rule = self.model._normalize_combination_rule(self.combination_rule)
        self.model.device = resolve_torch_device(self.device)
        self.model.to(self.model.device)
        self.model.eval()
        self.rules_ready = True
        self.trained = True

    def save_model(self, model_path: str) -> None:
        assert self.model is not None and self.rules_ready
        self.model.save_rules_bin(model_path)

    @staticmethod
    def _nll_from_probs(
        probs: torch.Tensor,
        target: torch.Tensor,
        *,
        class_weights: Optional[torch.Tensor] = None,
        eps: float = 1e-9,
    ) -> torch.Tensor:
        """Stable NLL from probability outputs."""
        probs = torch.clamp(probs, eps, 1.0)
        return F.nll_loss(torch.log(probs), target, weight=class_weights, reduction="mean")

    @staticmethod
    def _objective_from_probs(
        probs: torch.Tensor,
        target: torch.Tensor,
        *,
        class_weights: Optional[torch.Tensor],
        eps: float = 1e-9,
    ) -> torch.Tensor:
        return DSClassifierMultiQ._nll_from_probs(probs, target, class_weights=class_weights, eps=eps)

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    @staticmethod
    def _is_better_checkpoint(val_obj: float, val_nll: float, best_obj: float, best_nll: float) -> bool:
        obj_eps = 1e-6
        nll_eps = 1e-6
        return (val_obj + obj_eps < best_obj) or (
            abs(val_obj - best_obj) <= obj_eps and val_nll + nll_eps < best_nll
        )

    @staticmethod
    def _binary_threshold_candidates(pos_proba: np.ndarray) -> np.ndarray:
        grid = np.linspace(0.05, 0.95, 181, dtype=float)
        quant = np.quantile(pos_proba, np.linspace(0.05, 0.95, 19, dtype=float))
        cand = np.unique(np.clip(np.concatenate([grid, quant, np.array([0.5])]), 1e-6, 1.0 - 1e-6))
        return cand

    @staticmethod
    def _regularize_binary_threshold(threshold: float, *, n_val: int, tau: float = 16.0) -> float:
        if n_val <= 0:
            return float(threshold)
        alpha = float(n_val) / float(n_val + max(1.0, tau))
        return float(np.clip(0.5 + alpha * (float(threshold) - 0.5), 1e-6, 1.0 - 1e-6))

    def _tune_binary_threshold(self, y_true: np.ndarray, proba: np.ndarray) -> float:
        pos_proba = np.asarray(proba[:, 1], dtype=float)
        thresholds = self._binary_threshold_candidates(pos_proba)
        best_thr = 0.5
        best_f1 = -1.0
        for thr in thresholds:
            y_pred = (pos_proba >= thr).astype(int)
            f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            if f1 > best_f1 + 1e-9 or (abs(f1 - best_f1) <= 1e-9 and abs(thr - 0.5) < abs(best_thr - 0.5)):
                best_f1 = f1
                best_thr = float(thr)
        return best_thr

    @staticmethod
    def _should_use_tuned_threshold(
        threshold: float,
        *,
        tuned_f1: float,
        argmax_f1: float,
        n_val: int,
        max_shift: float = 0.08,
    ) -> bool:
        # Small validation wins from aggressively shifted thresholds were unstable on
        # low-dimensional binary tasks. Keep threshold tuning only when the gain is
        # material and the selected threshold stays close to the natural 0.5 split.
        gain_margin = max(0.01, 3.0 / max(50.0, float(n_val)))
        if tuned_f1 <= argmax_f1 + gain_margin:
            return False
        return abs(float(threshold) - 0.5) <= float(max_shift)

    def _predict_labels_from_proba(self, proba: np.ndarray, *, use_tuned_threshold: bool = True) -> np.ndarray:
        p = np.asarray(proba, dtype=float)
        if (
            use_tuned_threshold
            and self.k == 2
            and self._decision_threshold is not None
            and p.ndim == 2
            and p.shape[1] == 2
        ):
            return (p[:, 1] >= float(self._decision_threshold)).astype(int)
        return p.argmax(axis=1)

    def _split_train_val(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.train_cfg
        idx = np.arange(len(y), dtype=int)
        if cfg.val_split <= 0.0 or len(y) <= 2:
            return idx, idx
        test_size = float(np.clip(cfg.val_split, 0.0, 0.5))
        try:
            return train_test_split(idx, test_size=test_size, random_state=int(cfg.seed), stratify=y)
        except Exception:
            return train_test_split(idx, test_size=test_size, random_state=int(cfg.seed), stratify=None)

    def _build_class_weights(self, y_train: np.ndarray) -> Optional[torch.Tensor]:
        pwr = float(max(0.0, self.train_cfg.class_weight_power))
        if pwr <= 0.0:
            return None
        counts = np.bincount(np.asarray(y_train, dtype=int), minlength=self.k).astype(np.float64)
        counts = np.clip(counts, 1.0, None)
        beta = float(np.clip(self.train_cfg.class_weight_beta, 0.9, 0.99999))
        eff_num = 1.0 - np.power(beta, counts)
        weights = ((1.0 - beta) / np.clip(eff_num, 1e-12, None)) ** pwr
        weights = weights / max(float(np.mean(weights)), 1e-12)
        assert self.model is not None
        return torch.tensor(weights, dtype=torch.float32, device=self.model.device)

    def _evaluate_validation(
        self,
        X_val_t: torch.Tensor,
        y_val: np.ndarray,
        y_val_t: torch.Tensor,
        *,
        class_weights: Optional[torch.Tensor],
        eps: float,
    ) -> Dict[str, float]:
        assert self.model is not None
        pv = self.model(X_val_t, combination_rule=self.model.combination_rule)
        val_obj = float(
            self._objective_from_probs(
                pv,
                y_val_t,
                class_weights=class_weights,
                eps=eps,
            ).item()
        )
        val_bal_nll = float(self._nll_from_probs(pv, y_val_t, class_weights=class_weights, eps=eps).item())
        val_nll = float(self._nll_from_probs(pv, y_val_t, class_weights=None, eps=eps).item())
        y_val_pred = self._to_numpy(pv.argmax(dim=1))
        val_f1 = float(f1_score(y_val, y_val_pred, average="macro", zero_division=0))
        act_v = self.model._activation_matrix(X_val_t).to(dtype=pv.dtype)
        masses_v = self.model.get_rule_masses()
        omega_v = masses_v[:, -1].view(1, -1)
        fired_v = act_v.sum(dim=1).clamp(min=1.0)
        active_omega_mean = float(((act_v * omega_v).sum(dim=1) / fired_v).mean().item())
        return {
            "val_obj": val_obj,
            "val_f1": val_f1,
            "val_nll": val_nll,
            "val_bal_nll": val_bal_nll,
            "active_omega_mean": active_omega_mean,
        }

    def train_rule_masses(self, X: np.ndarray, y: np.ndarray) -> None:
        model = self._require_ready_model()
        cfg = self.train_cfg
        train_rule = str(model.combination_rule)
        torch.manual_seed(cfg.seed)
        self._decision_threshold = None

        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        tr_idx, val_idx = self._split_train_val(X, y)

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        X_tr_t = model._prepare_numeric_tensor(X_tr)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=model.device)
        X_val_t = model._prepare_numeric_tensor(X_val)
        y_val_t = torch.tensor(y_val, dtype=torch.long, device=model.device)

        bs = max(1, min(int(cfg.batch_size), int(len(y_tr_t))))
        dl = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=bs, shuffle=False)

        class_weights = self._build_class_weights(y_tr)

        opt = self._make_optimizer(model, cfg)
        if cfg.verbose:
            print(
                "[DSGD] train_cfg: "
                f"optimizer={getattr(cfg, 'optimizer_name', 'adamw')} "
                f"lr={cfg.lr:.6f} weight_decay={cfg.weight_decay:.6f} "
                f"objective=balanced_nll stop=train_delta({cfg.min_train_obj_delta:.2e})"
            )

        eps = 1e-9
        model.prepare_for_mass_training(init_seed=cfg.seed, rule_uncertainty=self.rule_uncertainty)
        model.eval()
        with torch.no_grad():
            base_eval = self._evaluate_validation(
                X_val_t,
                y_val,
                y_val_t,
                class_weights=class_weights,
                eps=eps,
            )
        base_obj = float(base_eval["val_obj"])
        base_f1 = float(base_eval["val_f1"])
        base_nll = float(base_eval["val_nll"])
        base_bal_nll = float(base_eval["val_bal_nll"])
        base_active_omega = float(base_eval["active_omega_mean"])
        baseline_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        best_obj = base_obj
        best_f1 = base_f1
        best_nll = base_nll
        best_state = baseline_state
        best_omega = base_active_omega
        bad_epochs = 0
        best_epoch = 0
        prev_train_obj: Optional[float] = None
        history: list[Dict[str, float | int]] = []

        for epoch in range(1, cfg.max_epochs + 1):
            model.train()
            total = 0.0
            count = 0
            for xb, yb in dl:
                opt.zero_grad(set_to_none=True)
                p = model(xb, combination_rule=train_rule)
                loss = self._objective_from_probs(
                    p,
                    yb,
                    class_weights=class_weights,
                    eps=eps,
                )
                if not loss.requires_grad:
                    continue
                loss.backward()
                opt.step()
                model.project_rules_to_simplex()
                total += float(loss.detach().cpu())
                count += 1

            model.eval()
            with torch.no_grad():
                val_eval = self._evaluate_validation(X_val_t, y_val, y_val_t, class_weights=class_weights, eps=eps)
                val_obj = val_eval["val_obj"]
                val_f1 = val_eval["val_f1"]
                val_nll = val_eval["val_nll"]
                val_bal_nll = val_eval["val_bal_nll"]
                active_omega_mean = val_eval["active_omega_mean"]
            train_obj = float(total / max(count, 1))
            lr_now = float(opt.param_groups[0]["lr"])
            if cfg.record_history:
                history.append(
                    {
                        "epoch": int(epoch),
                        "train_obj": train_obj,
                        "val_obj": float(val_obj),
                        "val_nll": float(val_nll),
                        "val_bal_nll": float(val_bal_nll),
                        "val_macro_f1": float(val_f1),
                        "lr": lr_now,
                    }
                )

            if cfg.verbose and (epoch == 1 or epoch % 5 == 0 or epoch == cfg.max_epochs):
                print(
                    f"[DSGD] epoch {epoch:03d}/{cfg.max_epochs}  "
                    f"train_obj={train_obj:.4f}  "
                    f"val_obj={val_obj:.4f}  "
                    f"val_nll={val_nll:.4f}  val_bal_nll={val_bal_nll:.4f}  "
                    f"val_macro_f1={val_f1:.4f}  lr={lr_now:.6f}"
                )

            improved = self._is_better_checkpoint(val_obj, val_nll, best_obj, best_nll)
            if improved:
                best_obj = val_obj
                best_f1 = val_f1
                best_nll = val_nll
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                best_omega = active_omega_mean
                best_epoch = int(epoch)
                bad_epochs = 0
            else:
                bad_epochs += 1

            if prev_train_obj is not None and epoch > int(cfg.min_train_epochs):
                delta = float(prev_train_obj - train_obj)
                if delta < float(cfg.min_train_obj_delta):
                    if cfg.verbose:
                        print(
                            f"[DSGD] stop at epoch {epoch} "
                            f"(train_obj delta={delta:.6f} < {float(cfg.min_train_obj_delta):.6f}; "
                            f"best val_obj={best_obj:.4f}, best val_macro_f1={best_f1:.4f}, best val_nll={best_nll:.4f})"
                        )
                    break
            prev_train_obj = train_obj

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pv_best = self._to_numpy(model(X_val_t, combination_rule=train_rule))
        if cfg.enable_binary_threshold_tuning and self.k == 2 and len(np.unique(y_val)) == 2:
            tuned_thr = self._tune_binary_threshold(y_val, pv_best)
            tuned_thr = self._regularize_binary_threshold(tuned_thr, n_val=int(len(y_val)))
            tuned_pred = (pv_best[:, 1] >= tuned_thr).astype(int)
            tuned_f1 = float(f1_score(y_val, tuned_pred, average="macro", zero_division=0))
            argmax_pred = pv_best.argmax(axis=1)
            argmax_f1 = float(f1_score(y_val, argmax_pred, average="macro", zero_division=0))
            if self._should_use_tuned_threshold(
                tuned_thr,
                tuned_f1=tuned_f1,
                argmax_f1=argmax_f1,
                n_val=int(len(y_val)),
            ):
                self._decision_threshold = float(tuned_thr)
        if cfg.verbose:
            print(
                f"[DSGD] val_f1 baseline_dempster={base_f1:.4f} "
                f"best_dsgd={best_f1:.4f} best_val_nll={best_nll:.4f} "
                f"best_epoch={best_epoch if best_epoch > 0 else len(history)}"
            )
            if self._decision_threshold is not None:
                print(f"[DSGD] binary decision threshold={self._decision_threshold:.4f}")
        self.trained = True
        self._last_fit_meta = {
            "n_train": int(len(tr_idx)),
            "n_val": int(len(val_idx)),
            "class_weight_power": float(cfg.class_weight_power),
            "rule_uncertainty_init": float(self.rule_uncertainty),
            "optimizer_name": str(getattr(cfg, "optimizer_name", "adamw")),
            "epochs_ran": int(history[-1]["epoch"]) if history else 0,
            "best_epoch": int(best_epoch),
            "val_obj_best_dsgd": float(best_obj),
            "val_f1_baseline_dempster": float(base_f1),
            "val_f1_best_dsgd": float(best_f1),
            "val_nll_best_dsgd": float(best_nll),
            "decision_threshold": None if self._decision_threshold is None else float(self._decision_threshold),
            "history": history,
        }
        self._print_mass_summary(best_active_omega=best_omega)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        algo: Optional[str] = None,
        feature_names=None,
        value_decoders=None,
        rule_gen_params: Optional[Dict[str, Any]] = None,
        on_rule: Optional[Callable[[int], None]] = None,
        rules_path: Optional[str] = None,
        fallback_rules_paths: Optional[list[str]] = None,
        use_cached_rules: bool = False,
        save_rules_path: Optional[str] = None,
        verify_raw_on: Optional[np.ndarray] = None,
        verify_raw_methods: tuple[str, ...] = ("first_hit_laplace", "weighted_vote"),
    ) -> Dict[str, Any]:
        """Linear training pipeline: load/generate rules, then train masses."""
        algo_name = str(algo or self.default_rule_algo or "RIPPER").upper()
        save_target = str(save_rules_path or rules_path or "").strip() or None
        resolved_rule_source = "existing"
        loaded_from: Optional[str] = None

        candidates: list[Path] = []
        if rules_path:
            candidates.append(Path(rules_path))
        for candidate in fallback_rules_paths or []:
            if candidate:
                candidates.append(Path(candidate))

        loaded = False
        if use_cached_rules:
            for candidate in candidates:
                if candidate.exists():
                    self.load_rules(str(candidate))
                    loaded = True
                    loaded_from = str(candidate)
                    resolved_rule_source = "cache"
                    if save_target is not None and str(candidate) != str(save_target):
                        self.save_rules(save_target)
                    break

        if not loaded and (not self.rules_ready or feature_names is not None or value_decoders is not None or algo is not None):
            self.build_rule_base(
                X,
                y,
                algo=algo_name,
                feature_names=feature_names,
                value_decoders=value_decoders,
                rule_gen_params=rule_gen_params,
                on_rule=on_rule,
            )
            resolved_rule_source = "generated"
            if save_target is not None:
                self.save_rules(save_target)

        raw_before = None
        if verify_raw_on is not None and verify_raw_methods:
            raw_before = self._snapshot_raw_baselines(np.asarray(verify_raw_on), methods=tuple(verify_raw_methods))

        self.train_rule_masses(X, y)

        if raw_before is not None:
            raw_after = self._snapshot_raw_baselines(np.asarray(verify_raw_on), methods=tuple(verify_raw_methods))
            for method, before in raw_before.items():
                delta = float(np.max(np.abs(before - raw_after[method])))
                if delta > 1e-10:
                    raise RuntimeError(f"Raw baseline changed after DSGD training: method={method}, max_delta={delta:.3e}")

        if save_target is not None:
            self.save_rules(save_target)

        return {
            "rule_source": resolved_rule_source,
            "loaded_from": loaded_from,
            "saved_to": save_target,
        }

    def predict_proba(self, X: np.ndarray, *, combination_rule: str = "dempster") -> np.ndarray:
        model = self._require_ready_model()
        model.eval()
        with torch.no_grad():
            p = model(X, combination_rule=combination_rule)
            return self._to_numpy(p)

    def predict(self, X: np.ndarray, *, combination_rule: str = "dempster", use_tuned_threshold: bool = True) -> np.ndarray:
        proba = self.predict_proba(X, combination_rule=combination_rule)
        return self._predict_labels_from_proba(proba, use_tuned_threshold=use_tuned_threshold)

    def raw_predict_proba(self, X: np.ndarray, *, method: str = "first_hit_laplace") -> np.ndarray:
        model = self._require_ready_model()
        return model.predict_rule_baseline_proba(X, method=method)

    def raw_predict(self, X: np.ndarray, *, method: str = "first_hit_laplace") -> np.ndarray:
        return self.raw_predict_proba(X, method=method).argmax(axis=1)

    def _print_mass_summary(self, *, best_active_omega: float) -> None:
        if self.model is None or self.model.rule_mass_params is None or not self.model.rules:
            return

        masses = self._to_numpy(self.model.get_rule_masses())
        omega = np.asarray(masses[:, -1], dtype=float)
        labels = np.array([
            r.get("label", -1) if r.get("label", None) is not None else -1 for r in self.model.rules
        ], dtype=int)
        valid = (labels >= 0) & (labels < self.k)
        target_mass_mean = float(np.mean(masses[np.where(valid)[0], labels[valid]])) if valid.any() else float("nan")

        p10, p50, p90 = np.percentile(omega, [10, 50, 90]).tolist()
        print(
            "[DSGD] masses: "
            f"omega_mean={float(np.mean(omega)):.4f} "
            f"omega_p10={p10:.4f} omega_p50={p50:.4f} omega_p90={p90:.4f} "
            f"target_mass_mean={target_mass_mean:.4f} "
            f"best_active_omega_mean={best_active_omega:.4f}"
        )
