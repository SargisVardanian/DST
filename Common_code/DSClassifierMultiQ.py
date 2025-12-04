"""
Sklearn-style wrapper around DSModelMultiQ with PGD training loop and Evidential Regularization.

This module provides a scikit-learn compatible classifier interface for the DST model,
including training via Projected Gradient Descent (PGD), early stopping, and class balancing.
"""
from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from DSModelMultiQ import DSModelMultiQ, masses_to_pignistic
from core import params_to_mass



def _as_numpy(x):
    """Convert input to numpy array if needed."""
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)



def _metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int = None) -> dict[str, float]:
    """
    Compute classification metrics with adaptive averaging.
    
    Args:
        y_true (ndarray): True labels
        y_pred (ndarray): Predicted labels
        k (int, optional): Number of classes
        
    Returns:
        dict: Dictionary with Accuracy, F1, Precision, Recall scores
    """
    # Adaptive averaging: binary for K=2, weighted for K>2
    if k is None:
        k = len(np.unique(y_true))
    avg = 'binary' if k == 2 else 'weighted'
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
    }



def masses_to_pignistic(m: torch.Tensor) -> torch.Tensor:
    """
    Convert masses to pignistic probabilities (pass-through for compatibility).
    
    The model now returns probabilities directly, so this is effectively a pass-through.
    
    Args:
        m (Tensor): Mass or probability tensor
        
    Returns:
        Tensor: Probabilities
    """
    # Model returns probabilities (N, K) directly from forward()
    return m


class DSClassifierMultiQ(ClassifierMixin):
    """
    Scikit-learn compatible DST classifier with uncertainty quantification.
    
    This classifier combines rule-based learning with Dempster-Shafer Theory
    to provide predictions with quantified uncertainty. It supports multiple
    rule generation algorithms (STATIC, RIPPER, FOIL) and trains mass functions
    using Projected Gradient Descent.
    
    Attributes:
        k (int): Number of classes
        device (str): PyTorch device ('cpu' or 'cuda')
        rule_algo (str): Rule generation algorithm
        model (DSModelMultiQ): Underlying DST model
    """
    
    def __init__(
        self,
        k,
        device="cpu",
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
    ) -> None:
        """
        Initialize the DST classifier.
        
        Args:
            k (int): Number of classes
            device (str): PyTorch device. Default: 'cpu'
            rule_algo (str): Rule algorithm ('STATIC', 'RIPPER', 'FOIL'). Default: 'STATIC'
            max_iter (int): Maximum training epochs. Default: 50
            batch_size (int): Training batch size. Default: 512
            lr (float): Learning rate. Default: 2e-3
            val_split (float): Validation split ratio. Default: 0.2
            seed (int): Random seed. Default: 42
            print_every (int): Print metrics every N epochs. Default: 5
            weight_decay (float): AdamW weight decay. Default: 2e-4
            class_weight_power (float): Class balancing exponent. Default: 0.5
            grad_clip (float): Gradient clipping threshold. Default: 1.0
            early_stop_patience (int): Early stopping patience. Default: 10
            combination_rule (str): DST combination rule. Default: 'dempster'
            debug (bool): Enable debug output. Default: False
            rule_uncertainty (float): Base uncertainty for mass init. Default: 0.8
            lossfn (str): Loss function ('MSE' or 'CE'). Default: 'MSE'
            uncertainty_rule_weight (float): Uncertainty regularization weight. Default: 0.1
            rule_gen_params (dict, optional): Additional rule generation parameters
        """
        self.k = int(k)
        self.device = str(device)
        self.rule_algo = str(rule_algo or "STATIC").upper()
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
        self.lossfn = str(lossfn or "MSE").upper()  # ["MSE", "CE"]
        self.uncertainty_rule_weight = float(max(0.0, uncertainty_rule_weight))
        self.rule_gen_params = rule_gen_params or {
            # More variety for undertrained setups while keeping redundancy in check
            "enable_diversity_filter": True,
            "diversity_threshold": 0.82,
            "pair_top": 12,
            "triple_top": 32,
        }
        self._trained = False

        # Model is initialized lazily in fit() or explicitly if needed, 
        # but we keep a placeholder here to avoid errors if accessed early.
        self.model = DSModelMultiQ(
            k=self.k,
            algo=self.rule_algo,
            device=self.device,
            rule_uncertainty=rule_uncertainty,
        )

        self.default_class_ = 0
        self.history_ = []
        self.best_epoch_ = None
        self.best_val_metrics_ = None

    def generate_rules(self, X, y=None, feature_names=None, rule_algo: str | None = None, **kwargs) -> None:
        """
        Explicitly generate classification rules.
        
        Usually called automatically by fit(), but can be called explicitly
        to regenerate rules with different parameters.
        
        Args:
            X: Input features
            y: Target labels (required for RIPPER/FOIL)
            feature_names (list, optional): Feature names
            rule_algo (str, optional): Override default rule algorithm
            **kwargs: Additional parameters for rule generation
        """
        algo = rule_algo or self.rule_algo
        self.model.algo = str(algo).upper()
        
        # Pass through verbose flags if present
        verbose_flag = bool(kwargs.pop("verbose", kwargs.pop("verbose_rules", False)))
        
        # Ensure 'algo' is not in kwargs to avoid duplicates
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
    def fit(
        self, 
        X, 
        y, 
        feature_names=None, 
        value_decoders=None, 
        rule_params: dict | None = None
    ):  # type: ignore[override]
        """
        Fit the DST model.
        
        1. If rules are not generated, generates them using `self.rule_algo`.
        2. Trains the mass weights using PGD.
        
        Args:
            X: Training data (array-like)
            y: Target labels (array-like)
            feature_names: Optional list of feature names.
            value_decoders: Optional dict mapping feature names to value decoders.
            rule_params: Dictionary of parameters to pass to the rule generator (e.g. {'grow_ratio': 0.7}).
        """
        X = _as_numpy(X).astype(np.float32)
        y = _as_numpy(y).astype(int)
        
        # Update model metadata if provided
        if feature_names is not None:
            self.model.feature_names = list(feature_names)
            self.model._feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        if value_decoders is not None:
            self.model.value_names = value_decoders

        # 1. Automatic Rule Generation
        # If we have no rules, OR if we explicitly want to regenerate (e.g. rule_params provided), do it.
        # For now, we'll generate if the rule set is empty.
        if not self.model.rules:
            if self.debug:
                print(f"[DSClassifier] Generating rules using {self.rule_algo}...")

            # Merge init params with fit params (fit params take precedence)
            gen_params = self.rule_gen_params.copy()
            if rule_params:
                gen_params.update(rule_params)

            # Encourage richer static rule pools by default
            if self.rule_algo == "STATIC":
                gen_params.setdefault("enable_diversity_filter", True)
                gen_params.setdefault("diversity_threshold", 0.82)
                gen_params.setdefault("pair_top", 12)
                gen_params.setdefault("triple_top", 32)

            # Pass feature_names explicitly to ensure generator uses the same names as the model
            fnames = feature_names if feature_names is not None else self.model.feature_names
            if self.debug:
                print(f"[DSClassifier] fnames passed to generate_rules: {fnames[:5] if fnames else 'None'} (len={len(fnames) if fnames else 0})")

            self.generate_rules(
                X,
                y,
                feature_names=fnames,
                rule_algo=self.rule_algo,
                **gen_params
            )

        # 2. Training Loop
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
        X_val, y_val = (X[train_idx], y[train_idx]) if val_count == 0 else (X[val_idx], y[val_idx])

        # Ensure masses are reset and valid before training
        if self.model.rule_mass_params is None:
            # Default to DSGD++ initialization if X, y are available, else random
            # We can add an init_mode param to fit() or __init__, but for now let's auto-detect or default.
            # The user requested "implement improvements", so let's use DSGD++ by default if possible.
            try:
                if self.debug: print("[DSClassifier] Initializing masses using DSGD++...")
                self.model.init_masses_dsgdpp(X_train, y_train)
            except Exception as e:
                print(f"[warn] DSGD++ init failed ({e}), falling back to random")
                self.model.reset_masses()
        else:
            self.model.project_rules_to_simplex()

        params = [p for p in [self.model.rule_mass_params] if p is not None and p.requires_grad]
        if not params:
            self._trained = True
            return self

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

        # Set class prior on the model to redistribute Omega mass toward minority classes
        inv_freq_prior = inv_freq  # already sum/ count
        exp = 0.5 + max(self.class_weight_power, 0.0)
        prior_raw = inv_freq_prior ** exp
        prior = prior_raw / (prior_raw.sum() + 1e-12)
        try:
            self.model.class_prior = torch.from_numpy(prior).to(device=device, dtype=torch.float32)
        except Exception:
            self.model.class_prior = None

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

            # Precompute length-based weights once
            rule_len = getattr(self.model, "_rule_len", None)
            len_weights = None
            if rule_len is not None and rule_len.numel() > 0:
                rl = rule_len.to(device=self.model.device, dtype=torch.float32)
                # Higher weight for short rules → push them to low uncertainty
                # w = (Lmax - L + 1)
                w = (rl.max() - rl + 1.0)
                # normalize to mean 1.0 to keep scale stable
                # len_weights = (w / (w.mean().clamp(min=1e-9))).detach()
                len_weights = (w / (w.mean().clamp(min=1e-9))).detach()

            for xb, yb in loader:
                xb = xb.to(device=device, dtype=torch.float32)
                yb = yb.to(device=device, dtype=torch.int64)

                optimizer.zero_grad(set_to_none=True)

                # Forward → probabilities over classes (N, K)
                probs = torch.clamp(self.model.forward(xb), min=1e-9)

                # ---------------- Data loss (aligned with DSGD-Enhanced) ----------------
                if self.lossfn == "CE":
                    # Cross-entropy loss on log-probabilities
                    logp = torch.log(probs)
                    # NLL loss
                    nll = torch.nn.functional.nll_loss(
                        logp,
                        yb,
                        reduction="none",
                    )
                    # Class weighting (boost rare classes)
                    if self.class_weight_power > 0.0:
                        nll = nll * class_weights_t[yb]
                    data_loss = nll.mean()
                else:
                    # MSE between predicted probabilities and one-hot (standard DSGD approach)
                    y_onehot = torch.nn.functional.one_hot(
                        yb, num_classes=self.k
                    ).to(device=device, dtype=torch.float32)
                    if self.class_weight_power > 0.0:
                        w = class_weights_t[yb].unsqueeze(1)  # (N, 1)
                        diff = probs - y_onehot
                        data_loss = (w * diff.pow(2)).mean()
                    else:
                        data_loss = torch.nn.functional.mse_loss(probs, y_onehot)



                loss = data_loss

                loss.backward()

                if self.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(params, self.grad_clip)

                optimizer.step()

                # --- Projected Gradient Descent Step ---
                # Force weights back to valid simplex immediately
                self.model.project_rules_to_simplex()

                total_loss += float(loss.detach())
                batches += 1

            train_loss = total_loss / max(1, batches)

            # Validation
            self.model.eval()
            with torch.inference_mode():
                probs_val = torch.clamp(self.model.forward(X_val_t), min=1e-9)
                masses_val = torch.as_tensor(self.model.predict_masses(X_val_t), device=device, dtype=torch.float32)
                masses_val = torch.clamp(masses_val, min=0.0)
                masses_val = masses_val / masses_val.sum(dim=1, keepdim=True).clamp_min(1e-9)

                if self.lossfn == "CE":
                    logp_val = torch.log(probs_val)
                    nll_val = torch.nn.functional.nll_loss(
                        logp_val,
                        y_val_t,
                        reduction="none",
                    )
                    if self.class_weight_power > 0.0:
                        nll_val = nll_val * class_weights_t[y_val_t]
                    val_loss = nll_val.mean()
                else:
                    y_val_onehot = torch.nn.functional.one_hot(
                        y_val_t, num_classes=self.k
                    ).to(device=device, dtype=torch.float32)
                    if self.class_weight_power > 0.0:
                        w_val = class_weights_t[y_val_t].unsqueeze(1)
                        diff_val = probs_val - y_val_onehot
                        val_loss = (w_val * diff_val.pow(2)).mean()
                    else:
                        val_loss = torch.nn.functional.mse_loss(probs_val, y_val_onehot)

                preds_val = probs_val.argmax(dim=1)
                # Uncertainty as average mass on Omega across activated rules
                unc_samples = self.model.sample_uncertainty(X_val_t)
                mean_unc = float(np.asarray(unc_samples, dtype=np.float32).mean())

            val_loss_float = float(val_loss.detach().cpu())
            metrics_val = _metrics(y_val, preds_val.detach().cpu().numpy(), k=self.k)
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss_float,
                "val_uncertainty": mean_unc,
                **metrics_val,
            }
            self.history_.append(record)

            improved = val_loss_float < best_val - 1e-6
            if improved:
                best_val = val_loss_float
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch_ = epoch
                self.best_val_metrics_ = {
                    **metrics_val,
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
                    f"Acc={metrics_val['Accuracy']:.4f} F1={metrics_val['F1']:.4f} "
                    f"P={metrics_val['Precision']:.4f} R={metrics_val['Recall']:.4f} "
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
        if self.model.rule_mass_params is not None:
            self.model.initial_rule_masses = self.model.rule_mass_params.detach().clone()
        self._trained = True
        return self

    # --------------------------- Prediction helpers ---------------------------
    @torch.no_grad()
    def predict(
        self,
        X,
        *,
        default_label=None,
        vote_mode: str = "majority",
        use_first_rule=False,
        use_initial_masses=None,
    ):
        """
        Predict class labels for input samples.
        
        Args:
            X: Input features
            default_label (int, optional): Default class when no rules fire
            vote_mode (str): Voting mode ('majority' or 'weighted'). Default: 'majority'
            use_first_rule (bool): Use first-rule prediction. Default: False
            use_initial_masses (bool, optional): Use untrained masses
            
        Returns:
            ndarray: Predicted class labels
        """
        base = self.default_class_ if default_label is None else int(default_label)
        X_np = _as_numpy(X).astype(np.float32)
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
        if self._trained:
            use_initial = bool(use_initial_masses) if use_initial_masses is not None else False
            return np.asarray(self.model.predict_dst_labels(X_np, use_initial_masses=use_initial), dtype=int)
        if use_first_rule:
            return np.asarray(self.model.predict_by_first_rule(X_np, default_label=base), dtype=int)
        if str(vote_mode).lower() == "weighted":
            return np.asarray(self.model.predict_by_weighted_rule_vote(X_np, default_label=base), dtype=int)
        return np.asarray(self.model.predict_by_rule_vote(X_np, default_label=base), dtype=int)

    # --------------------------- Persistence ---------------------------
    def save_model(self, path: str) -> None:
        """Save model rules and parameters to binary file."""
        self.model.save_rules_bin(path)

    def load_model(self, path: str) -> None:
        """Load model rules and parameters from binary file."""
        self.model.load_rules_bin(path)
        self.rule_algo = getattr(self.model, "algo", self.rule_algo)
        self._trained = self.model.rule_mass_params is not None

    def prepare_rules_for_export(self, sample=None):
        """Prepare rules and predictions for export."""
        return self.model.prepare_rules_for_export(sample)

