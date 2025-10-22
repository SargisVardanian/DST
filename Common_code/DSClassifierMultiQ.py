# -*- coding: utf-8 -*-
"""Sklearn-style wrapper around DSModelMultiQ with simple training loop (clean)."""

from __future__ import annotations

import copy
from typing import Dict, Optional, Sequence

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from DSModelMultiQ import DSModelMultiQ


def _as_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def masses_to_pignistic(m: torch.Tensor) -> torch.Tensor:
    """BetP(i) = m(i) + m(Ω)/k  for singletons+Ω."""
    k = m.shape[-1] - 1
    omega = m[:, -1:].expand(-1, k)
    return torch.clamp(m[:, :k] + omega / float(k), min=1e-12)


class DSClassifierMultiQ(ClassifierMixin):
    """Minimal training wrapper used throughout Common_code (simplified)."""

    def __init__(
        self,
        k: int,
        device: str = "cpu",
        algo: str = "STATIC",
        value_decoders: Optional[Dict[str, Dict[int, str]]] = None,
        feature_names: Optional[Sequence[str]] = None,
        max_iter: int = 50,
        batch_size: int = 512,
        lr: float = 5e-3,
        val_split: float = 0.2,
        seed: int = 42,
        print_every: int = 5,
        weight_decay: float = 2e-4,
        class_weight_power: float = 0.5,
        grad_clip: float = 1.0,
        early_stop_patience: int = 5,
        combination_rule: str = "dempster",
        debug: bool = False,
    ) -> None:
        self.k = int(k)
        self.device = str(device)
        self.algo = str(algo or "STATIC").upper()
        self.max_iter = int(max_iter)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.val_split = float(val_split)
        self.seed = int(seed)
        self.print_every = int(print_every)
        self.weight_decay = float(weight_decay)
        self.class_weight_power = float(max(0.0, class_weight_power))
        self.grad_clip = float(max(0.0, grad_clip))
        self.early_stop_patience = max(1, int(early_stop_patience))
        self.debug = bool(debug)

        self.model = DSModelMultiQ(
            k=self.k,
            algo=self.algo,
            device=self.device,
            feature_names=feature_names,
            value_decoders=value_decoders,
            combination_rule=combination_rule,
        )

        self.default_class_ = 0
        self.history_ = []
        self.best_epoch_ = None
        self.best_val_metrics_ = None
        self.last_uncertain_mask_ = None
        self.last_uncertain_scores_ = None

    # --------------------------- Rule generation ---------------------------

    def generate_raw(self, X, y=None, feature_names=None, algo: str = "STATIC", **kwargs):
        self.algo = str(algo or "STATIC").upper()
        self.model.algo = self.algo
        self.model.generate_raw(X, y, feature_names=feature_names, algo=self.algo, **kwargs)

    # --------------------------- Training ---------------------------

    def fit(self, X, y):  # type: ignore[override]
        if torch is None:
            print("[warn] PyTorch not available – skipping training")
            return self

        X = _as_numpy(X).astype(np.float32)
        y = _as_numpy(y).astype(int)
        if X.shape[0] < 2:
            print("[warn] need at least two samples to fit")
            return self

        rng = np.random.default_rng(self.seed)
        indices = np.arange(len(X))
        rng.shuffle(indices)

        val_count = max(1, int(round(self.val_split * len(X)))) if self.val_split > 0 else 0
        if val_count >= len(X):
            val_count = len(X) - 1
        val_idx = indices[:val_count]
        train_idx = indices[val_count:]
        if train_idx.size == 0:
            print("[warn] empty train split")
            return self

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = (X[val_idx], y[val_idx]) if val_count else (X[train_idx], y[train_idx])

        # убедимся, что массы инициализированы
        self.model._init_mass_params(reset=False)

        params=[p for p in [self.model.rule_mass_params] if (p is not None and getattr(p,"requires_grad",False))]
        if not params: return self

        device = torch.device(getattr(self.model, "device", self.device))
        X_train_t = torch.from_numpy(X_train)
        y_train_t = torch.from_numpy(y_train.astype(np.int64, copy=False))
        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        class_counts = np.bincount(y_train, minlength=self.k).astype(np.float32)
        class_counts[class_counts == 0.0] = 1.0
        inv_freq = class_counts.sum() / class_counts
        class_weights = inv_freq ** self.class_weight_power
        class_weights = class_weights / class_weights.mean()
        class_weights_t = torch.from_numpy(class_weights).to(device=device, dtype=torch.float32)

        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

        X_val_t = torch.from_numpy(X_val).to(device=device, dtype=torch.float32)
        y_val_t = torch.from_numpy(y_val.astype(np.int64, copy=False)).to(device=device)

        self.history_.clear()
        best_state = None
        best_val = float("inf")

        assert self.k > 0, "classifier requires at least one class"

        stale_epochs = 0
        for epoch in range(1, self.max_iter + 1):
            self.model.train()
            total_loss = 0.0
            batches = 0

            for xb, yb in loader:
                xb = xb.to(device=device, dtype=torch.float32)
                yb = yb.to(device=device, dtype=torch.int64)

                optimizer.zero_grad(set_to_none=True)
                masses = torch.clamp(self.model.forward(xb), min=1e-9)
                betp = torch.clamp(masses_to_pignistic(masses), min=1e-9)
                logp = torch.log(betp)
                idx = torch.arange(yb.size(0), device=device)
                loss = -(class_weights_t[yb] * logp[idx, yb]).mean()

                loss.backward()
                if self.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
                optimizer.step()
                self.model.project_masses()

                total_loss += float(loss.detach())
                batches += 1

            train_loss = total_loss / max(1, batches)

            # Validation
            self.model.eval()
            with torch.inference_mode():
                masses_val = torch.clamp(self.model.forward(X_val_t), min=1e-9)
                betp_val = torch.clamp(masses_to_pignistic(masses_val), min=1e-9)
                logp_val = torch.log(betp_val)
                idx_val = torch.arange(y_val_t.size(0), device=device)
                val_loss = -(class_weights_t[y_val_t] * logp_val[idx_val, y_val_t]).mean()
                preds_val = betp_val.argmax(dim=1)
                mean_unc = float(masses_val[:, -1].mean().cpu())

            val_loss_float = float(val_loss.detach().cpu())
            metrics = _metrics(y_val, preds_val.detach().cpu().numpy())
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss_float,
                "val_uncertainty": mean_unc,
                **metrics,
            }
            self.history_.append(record)

            improved = val_loss_float < best_val - 1e-6
            if improved:
                best_val = val_loss_float
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch_ = epoch
                self.best_val_metrics_ = {
                    **metrics,
                    "train_loss": train_loss,
                    "val_loss": val_loss_float,
                    "val_uncertainty": mean_unc,
                    "epoch": epoch,
                }
                stale_epochs = 0
            else:
                stale_epochs += 1

            if epoch == 1 or epoch % max(1, self.print_every) == 0 or improved:
                print(
                    f"[epoch {epoch:03d}] train_loss={train_loss:.5f} val_loss={val_loss_float:.5f} "
                    f"Acc={metrics['Accuracy']:.4f} F1={metrics['F1']:.4f} "
                    f"P={metrics['Precision']:.4f} R={metrics['Recall']:.4f} "
                    f"val_unc={mean_unc:.4f}"
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
        return self

    # --------------------------- Prediction helpers ---------------------------

    def predict_mass(self, X):
        return self.model.predict_mass(X)

    def predict(self, X):  # type: ignore[override]
        masses = self.predict_mass(X)
        if self.k == 0:
            return np.zeros(masses.shape[0], dtype=int)
        omega = masses[:, -1][:, None]
        betp = masses[:, : self.k] + omega / float(self.k)
        preds = betp.argmax(axis=1).astype(int)
        omega_flat = omega.reshape(-1)
        class_max = masses[:, : self.k].max(axis=1)
        self.last_uncertain_mask_ = omega_flat >= class_max
        self.last_uncertain_scores_ = omega_flat
        return preds

    # --------------------------- Persistence ---------------------------

    def save_model(self, path: str) -> None:
        self.model.save_rules_bin(path)

    def load_model(self, path: str) -> None:
        self.model.load_rules_bin(path)

    def prepare_rules_for_export(self, sample=None):
        return self.model.prepare_rules_for_export(sample)
