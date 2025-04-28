# DSClassifierMultiQ.py
import time
import numpy as np
import torch
from torch.autograd import Variable

from Common_code.DSModelMultiQ import DSModelMultiQ
from Common_code.DSRipper import DSRipper   # kept for back-compat

# -----------------------------------------------------------------------------


class DSClassifierMultiQ:
    """
    End-to-end Dempster–Shafer classifier.
    It can:
        * generate rules with RIPPER (default) or FOIL (`use_foil=True`)
        * learn masses via gradient descent (optional)
        * prune low-value rules (optional)
    Every *legacy* method from your former version is still present.
    """

    # ------------------------------------------------------------------ #
    def __init__(self,
                 num_classes: int,
                 lr: float = 0.005,
                 max_iter: int = 50,
                 min_iter: int = 2,
                 min_dloss: float = 1e-4,
                 optim: str = "adam",
                 lossfn: str = "MSE",
                 batch_size: int = 4000,
                 num_workers: int = 1,
                 precompute_rules: bool = False,
                 device: str = "cpu",
                 force_precompute: bool = False,
                 use_foil: bool = False):
        self.k           = num_classes
        self.lr          = lr
        self.max_iter    = max_iter
        self.min_iter    = min_iter
        self.min_dloss   = min_dloss
        self.optim       = optim.lower()
        self.lossfn      = lossfn.upper()
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.device      = torch.device(device)
        self.use_foil    = use_foil

        # main DS model (keeps all old functionality)
        self.model = DSModelMultiQ(
            k=num_classes,
            precompute_rules=precompute_rules,
            device=self.device,
            force_precompute=force_precompute,
            use_foil=use_foil
        ).to(self.device)

        # majority-vote fallback
        self.default_class_ = 0

        # deprecated attr kept for older code paths
        self.ripper = DSRipper(d=64, ratio=2/3.0, k=2)

    # ------------------------------------------------------------------ #
    #                              FIT                                   #
    # ------------------------------------------------------------------ #
    def fit(self, X, y,
            column_names=None,
            add_single_rules=False,
            single_rules_breaks=2,
            add_mult_rules=False,
            optimize_weights=True,
            prune_after_fit=False,
            prune_kwargs=None):
        """
        High-level training routine.
        1. (optional) add handcrafted rules
        2. generate RIPPER / FOIL rules *if* rule list is empty
        3. optimise DST masses (optional)
        4. prune (optional)
        """
        prune_kwargs = prune_kwargs or {}

        # –– handcrafted extra rules ––
        if add_single_rules:
            self.model.generate_statistic_single_rules(
                X, breaks=single_rules_breaks, column_names=column_names)
        if add_mult_rules:
            self.model.generate_mult_pair_rules(X, column_names=column_names)

        # –– automatic rule induction ––
        if not self.model.preds:
            print(f"Inducing {'FOIL' if self.use_foil else 'RIPPER'} rules …")
            if self.use_foil:
                rules = self.model.generate_foil_rules(
                    X, y, column_names, batch_size=self.batch_size)
            else:
                rules = self.model.generate_ripper_rules(
                    X, y, column_names, batch_size=self.batch_size)
            self.model.import_test_rules(rules)
        else:
            print("Existing rules detected – skipping induction.")

        self.default_class_ = int(np.bincount(y).argmax())

        # rule-only accuracy before optimisation
        acc0 = (self.predict_rules_only(X) == y).mean()
        print(f"Rule-only train accuracy: {acc0:.3f}")

        # –– DST mass optimisation ––
        losses, last_epoch = None, None
        if optimize_weights:
            opt_cls = torch.optim.Adam if self.optim == "adam" else torch.optim.SGD
            optimizer = opt_cls(self.model.parameters(), lr=self.lr)
            criterion = (torch.nn.CrossEntropyLoss()
                         if self.lossfn == "CE" else torch.nn.MSELoss())

            X_idx = np.insert(X, 0, values=np.arange(len(X)), axis=1)
            print("Starting mass optimisation …")
            losses, last_epoch = self._optimize(X_idx, y, optimizer, criterion)
            print(f"Finished – final loss {losses[-1]:.4f} @ epoch {last_epoch}")
        else:
            print("Skipping mass optimisation.")

        # –– pruning ––
        if prune_after_fit:
            X_idx = np.insert(X, 0, values=np.arange(len(X)), axis=1)
            kept = self.model.prune_rules(X_idx, **prune_kwargs)
            acc1 = (self.predict(X) == y).mean()
            print(f"Pruned → {kept} rules kept. Post-prune train acc: {acc1:.3f}")

        return losses, last_epoch

    # ------------------------------------------------------------------ #
    #              ORIGINAL optimise / predict methods stay              #
    # ------------------------------------------------------------------ #
    def _optimize(self, X, y, optimizer, criterion):
        """(unchanged except cosmetic)."""
        losses = []
        self.model.train(); self.model.clear_rmap()

        Xt = Variable(torch.Tensor(X).to(self.device))
        yt = (torch.LongTensor(y).to(self.device) if self.lossfn == "CE"
              else torch.nn.functional.one_hot(
                  torch.LongTensor(y).to(self.device), self.k).float())

        dataset = torch.utils.data.TensorDataset(Xt, yt)
        loader  = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers)

        for epoch in range(self.max_iter):
            epoch_loss = 0.0
            for Xi, yi in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(Xi), yi)
                loss.backward(retain_graph=True)
                optimizer.step()
                self.model.normalize()
                epoch_loss += loss.item() * len(yi) / len(dataset)
            losses.append(epoch_loss)
            print(f"\rEpoch {epoch+1}  loss={epoch_loss:.5f}", end="")
            if epoch > self.min_iter and abs(losses[-2] - epoch_loss) < self.min_dloss:
                break
        print()
        return losses, epoch

    # ................................................
    def predict(self, X, one_hot=False):
        """DST prediction."""
        self.model.eval(); self.model.clear_rmap()
        X_idx = np.insert(X, 0, values=np.arange(len(X)), axis=1)
        with torch.no_grad():
            out = self.model(torch.Tensor(X_idx).to(self.device))
            return (out.cpu().numpy() if one_hot
                    else torch.argmax(out, 1).cpu().numpy())

    # ................................................
    def predict_rules_only(self, X):
        """
        First-match rule prediction with majority-class fallback.
        """
        preds = np.full(len(X), self.default_class_, dtype=int)
        for i, row in enumerate(X):
            for rule in self.model.preds:
                if rule(row):
                    preds[i] = int(rule.caption.split()[1].rstrip(':'))
                    break
        return preds
