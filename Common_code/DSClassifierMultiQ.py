"""
DSClassifierMultiQ
------------------
Thin sklearn-like wrapper around `DSModelMultiQ`.

Responsibilities:
  • generate_raw(...) — build rules (STATIC/RIPPER/FOIL), keep textual decoders
  • fit(X, y, X_val, y_val) — batch training loop; prints validation metrics per epoch
  • predict_rules_only(X) — simple RAW voting by per-rule target classes
  • predict / predict_proba — inference via DST forward
"""
from typing import Optional, Dict, Any, List
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

try:
    from sklearn.base import ClassifierMixin
except Exception:  # fallback if sklearn is missing
    class ClassifierMixin:  # type: ignore
        pass

from DSModelMultiQ import DSModelMultiQ


class DSClassifierMultiQ(ClassifierMixin):
    def __init__(
        self,
        k: int,
        *,
        algo: str = "STATIC",        # expected by test_Ripper_DST.py
        device: str = "cpu",
        lr: float = 1e-2,
        batch_size: int = 512,
        max_iter: int = 200,
        optim: str = "adam",
        debug_print: bool = False,
        value_decoders: Optional[Dict[str, Dict[int, str]]] = None,
    ):
        self.k = int(k)
        self.algo = (algo or "STATIC").upper()
        self.device = device
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.max_iter = int(max_iter)
        self.optim_name = optim.lower()
        self.debug_print = bool(debug_print)

        self.model = DSModelMultiQ(k=self.k, device=self.device)
        if value_decoders:
            self.model.set_value_decoders(value_decoders)

        self.default_class_: int = 0  # majority fallback for RAW

    # ---------------- RAW ----------------
    def set_value_decoders(self, decoders: Optional[Dict[str, Dict[int, str]]]):
        if decoders:
            self.model.set_value_decoders(decoders)

    def generate_raw(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        column_names: Optional[List[str]] = None,
        algo: Optional[str] = None,
        **rip_kwargs: Any):
        """Build rules in the model (STATIC/RIPPER/FOIL)."""
        used_algo = (algo or self.algo or "STATIC").upper()
        self.algo = used_algo
        self.model.generate_raw(X, y=y, column_names=column_names, algo=used_algo, **rip_kwargs)
        if y is not None and y.size:
            binc = np.bincount(y.astype(int), minlength=self.k)
            if binc.size:
                self.default_class_ = int(binc.argmax())
        else:
            self.default_class_ = 0
        return self

    def predict_rules_only(self, X: np.ndarray):
        """Vote by counting active rules per class. If no rules fired → majority class."""
        A = self.model.activation_matrix(X)  # (N, R) uint8
        N, R = A.shape
        if R == 0:
            return np.full(N, self.default_class_, dtype=int)
        t = self.model.get_rule_targets()    # (R,), -1 if a rule has no target
        valid_idx = np.where(t >= 0)[0]
        if valid_idx.size == 0:
            return np.full(N, self.default_class_, dtype=int)
        K = self.k
        onehot = np.zeros((R, K), dtype=np.int32)
        onehot[valid_idx, t[valid_idx].astype(int)] = 1
        votes = A.astype(np.int32) @ onehot  # (N, K)
        sums = votes.sum(axis=1)
        y_pred = votes.argmax(axis=1)
        y_pred[sums == 0] = self.default_class_
        return y_pred.astype(int)

    # ---------------- Train / Infer ----------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        val_frac: float = 0.2,
        val_random_state: int = 42,
        stratified: bool = True,
    ):
        """
        Batch training loop.
        Validation is created internally from (X, y) via a held-out split.
        """
        device = torch.device(self.device)

        # 1) internal train/val split from (X, y)
        stratify_vec = y if (stratified and len(np.unique(y)) > 1) else None
        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=val_frac, random_state=val_random_state, stratify=stratify_vec
            )
        except ValueError:
            # fallback if some class too small for stratify
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=val_frac, random_state=val_random_state, stratify=None
            )

        # 2) tensors
        X_tr_t = torch.as_tensor(X_tr, dtype=torch.float32, device=device)
        y_tr_t = torch.as_tensor(y_tr, dtype=torch.long,   device=device)
        X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=device)
        y_val_np = np.asarray(y_val).astype(int)

        # 3) prior from TRAIN ONLY (no leakage)
        binc = np.bincount(y_tr_t.detach().cpu().numpy(), minlength=self.k).astype(np.float32)
        prior = binc / (binc.sum() + 1e-12)
        self.model.set_prior(prior)

        # 4) loader over TRAIN split
        ds = TensorDataset(X_tr_t, y_tr_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        # 5) optimizer / loss
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=self.lr) if self.optim_name == "adam" \
              else torch.optim.SGD(params, lr=self.lr, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()

        # 6) train loop + epoch-wise validation on held-out
        for epoch in range(1, self.max_iter + 1):
            self.model.train()
            loss_sum, n_batches = 0.0, 0
            for X_b, y_b in dl:
                opt.zero_grad(set_to_none=True)
                logits = self.model.forward(X_b)           # [B, K] logits
                loss = criterion(logits, y_b)              # targets are class indices
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
                loss_sum += float(loss.detach())
                n_batches += 1

            # validation on held-out split
            self.model.eval()
            with torch.inference_mode():
                logits_v = self.model.forward(X_val_t)     # [N_val, K]
                probs_v  = torch.softmax(logits_v, dim=1).cpu().numpy()
                y_hat_v  = probs_v.argmax(axis=1).astype(int)

            m = self._metrics(y_val_np, y_hat_v)
            print(f"[epoch {epoch:03d}] loss={loss_sum/max(1,n_batches):.5f} "
                  f"Acc={m['Accuracy']:.4f} F1={m['F1']:.4f} "
                  f"P={m['Precision']:.4f} R={m['Recall']:.4f}")

        # 7) majority fallback from TRAIN split only
        binc = np.bincount(y_tr_t.detach().cpu().numpy())
        if binc.size:
            self.default_class_ = int(binc.argmax())

        return self

    def predict(self, X: np.ndarray):
        self.model.eval()
        with torch.inference_mode():
            X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            logits = self.model.forward(X)
            return torch.argmax(logits, dim=1).cpu().numpy().astype(int)

    def predict_proba(self, X: np.ndarray):
        self.model.eval()
        with torch.inference_mode():
            X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            logits = self.model.forward(X)                 # [N, K]
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

    # ---------------- utils / sklearn API ----------------
    @staticmethod
    def _metrics(y_true: np.ndarray, y_pred: np.ndarray):
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        binary = (len(np.unique(y_true)) == 2)
        avg = "binary" if binary else "macro"
        return {
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "F1": float(f1_score(y_true, y_pred, average=avg)),
            "Precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
            "Recall": float(recall_score(y_true, y_pred, average=avg)),
        }

    def get_params(self, deep: bool = True):  # sklearn compatibility
        return {
            "k": self.k,
            "algo": self.algo,
            "device": self.device,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "max_iter": self.max_iter,
            "optim": self.optim_name,
            "debug_print": self.debug_print,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self
