# benchmark_full.py
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from Common_code.Datasets_loader    import load_dataset
from Common_code.DSClassifierMultiQ import DSClassifierMultiQ
from Common_code.core               import rules_to_dsb


def metrics(y_true, y_pred):
    return OrderedDict([
        ("Accuracy",  accuracy_score(y_true, y_pred)),
        ("F1",        f1_score(y_true, y_pred)),
        ("Precision", precision_score(y_true, y_pred)),
        ("Recall",    recall_score(y_true, y_pred)),
    ])


if __name__ == "__main__":
    DATASET_ID = 6
    X, y, feat_names = load_dataset(DATASET_ID)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y)

    results = OrderedDict()

    # ================ RIPPER ==========================================
    rip_raw = DSClassifierMultiQ(2, use_foil=False, max_iter=0)
    rip_raw.default_class_ = int(np.bincount(y_tr).argmax())
    raw_pkl = f"ripper_rules_dataset{DATASET_ID}_raw.pkl"
    raw_dsb = f"ripper_rules_dataset{DATASET_ID}_raw.dsb"
    if not os.path.exists(raw_pkl):
        rules = rip_raw.model.generate_ripper_rules(
            X_tr, y_tr, feat_names, batch_size=1000)
        rip_raw.model.import_test_rules(rules)
        rip_raw.model.save_rules_bin(raw_pkl)
        rules_to_dsb(rip_raw.model.preds, rip_raw.model._params, raw_dsb)
    else:
        rip_raw.model.load_rules_bin(raw_pkl)
    results["R_raw"] = metrics(y_te, rip_raw.predict_rules_only(X_te))

    # — RIPPER + DST
    rip_dst = DSClassifierMultiQ(2, use_foil=False)
    rip_dst.model.import_test_rules(rip_raw.model.preds)
    dst_pkl = f"ripper_rules_dataset{DATASET_ID}.pkl"
    dst_dsb = f"ripper_rules_dataset{DATASET_ID}.dsb"
    if not os.path.exists(dst_pkl):
        rip_dst.fit(X_tr, y_tr, column_names=feat_names,
                    optimize_weights=True, prune_after_fit=False)
        rip_dst.model.save_rules_bin(dst_pkl)
        rules_to_dsb(rip_dst.model.preds, rip_dst.model._params, dst_dsb)
    else:
        rip_dst.model.load_rules_bin(dst_pkl)
    results["R_DST"] = metrics(y_te, rip_dst.predict(X_te))

    # — RIPPER + DST + prune
    rip_pr = DSClassifierMultiQ(2, use_foil=False)
    rip_pr.model.import_test_rules(rip_raw.model.preds)
    rip_pr.model.load_rules_bin(dst_pkl)
    rip_pr.fit(X_tr, y_tr, feat_names,
               optimize_weights=False, prune_after_fit=True,
               prune_kwargs=dict(max_unc=0.5, min_score=0.4, min_cov=10))
    pr_pkl = f"ripper_rules_dataset{DATASET_ID}_pruned.pkl"
    pr_dsb = f"ripper_rules_dataset{DATASET_ID}_pruned.dsb"
    rip_pr.model.save_rules_bin(pr_pkl)
    rules_to_dsb(rip_pr.model.preds, rip_pr.model._params, pr_dsb)
    results["R_pruned"] = metrics(y_te, rip_pr.predict(X_te))

    # ================ FOIL ============================================
    foil_raw = DSClassifierMultiQ(2, use_foil=True, max_iter=0)
    foil_raw.default_class_ = int(np.bincount(y_tr).argmax())
    fraw_pkl = f"foil_rules_dataset{DATASET_ID}_raw.pkl"
    fraw_dsb = f"foil_rules_dataset{DATASET_ID}_raw.dsb"
    if not os.path.exists(fraw_pkl):
        rules = foil_raw.model.generate_foil_rules(
            X_tr, y_tr, feat_names, batch_size=1000)
        foil_raw.model.import_test_rules(rules)
        foil_raw.model.save_rules_bin(fraw_pkl)
        rules_to_dsb(foil_raw.model.preds, foil_raw.model._params, fraw_dsb)
    else:
        foil_raw.model.load_rules_bin(fraw_pkl)
    results["F_raw"] = metrics(y_te, foil_raw.predict_rules_only(X_te))

    # — FOIL + DST
    foil_dst = DSClassifierMultiQ(2, use_foil=True)
    foil_dst.model.import_test_rules(foil_raw.model.preds)
    fdst_pkl = f"foil_rules_dataset{DATASET_ID}.pkl"
    fdst_dsb = f"foil_rules_dataset{DATASET_ID}.dsb"
    if not os.path.exists(fdst_pkl):
        foil_dst.fit(X_tr, y_tr, feat_names,
                     optimize_weights=True, prune_after_fit=False)
        foil_dst.model.save_rules_bin(fdst_pkl)
        rules_to_dsb(foil_dst.model.preds, foil_dst.model._params, fdst_dsb)
    else:
        foil_dst.model.load_rules_bin(fdst_pkl)
    results["F_DST"] = metrics(y_te, foil_dst.predict(X_te))

    # — FOIL + DST + prune
    foil_pr = DSClassifierMultiQ(2, use_foil=True)
    foil_pr.model.import_test_rules(foil_raw.model.preds)
    foil_pr.model.load_rules_bin(fdst_pkl)
    foil_pr.fit(X_tr, y_tr, feat_names,
                optimize_weights=False, prune_after_fit=True,
                prune_kwargs=dict(max_unc=0.5, min_score=0.4, min_cov=10))
    fpr_pkl = f"foil_rules_dataset{DATASET_ID}_pruned.pkl"
    fpr_dsb = f"foil_rules_dataset{DATASET_ID}_pruned.dsb"
    foil_pr.model.save_rules_bin(fpr_pkl)
    rules_to_dsb(foil_pr.model.preds, foil_pr.model._params, fpr_dsb)
    results["F_pruned"] = metrics(y_te, foil_pr.predict(X_te))

    # ------------ save & plot ----------------------------------------
    df = pd.DataFrame(results).T
    df.to_csv(f"benchmark_dataset{DATASET_ID}.csv")

    y_min = df.min().min()
    pad   = 0.02
    ax = df.plot(kind="bar",
                 figsize=(9, 4),
                 ylim=(max(0, y_min-pad), 1.0+pad),
                 rot=30)

    # передаём df.round(3) вторым позиционным аргументом
    tbl = pd.plotting.table(ax,               # ① axes
                            df.round(3),      # ② data  (теперь аргумент на месте!)
                            loc='bottom',
                            cellLoc='center')

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.3)

    plt.subplots_adjust(left=0.05, bottom=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(f"benchmark_dataset{DATASET_ID}.png", dpi=150)
