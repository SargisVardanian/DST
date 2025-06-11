# test_Ripper_DST.py

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import OrderedDict
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from Common_code.Datasets_loader    import load_dataset
from Common_code.DSClassifierMultiQ import DSClassifierMultiQ

sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")


def metrics(y_true, y_pred):
    return OrderedDict([
        ("Accuracy",  accuracy_score(y_true, y_pred)),
        ("F1",        f1_score(y_true, y_pred, zero_division=0)),
        ("Precision", precision_score(y_true, y_pred, zero_division=0)),
        ("Recall",    recall_score(y_true, y_pred, zero_division=0)),
    ])


# ── helpers ──────────────────────────────────────────────────
def set_test_usability(model, X_test):
    """
    Заполняет rule.usability (%) через готовую fire-матрицу, без Python-циклов
    и ошибок сравнения типов.
    """
    fires = model.fire_matrix(X_test)          # shape = (N_test, n_rules)
    cov   = fires.sum(axis=0)                  # len == n_rules
    N     = len(X_test)
    for r, c in zip(model.preds, cov):
        r.usability = float(c) / N * 100
def dump_rules(model, filename):
    """Сохраняет правила в .dsb (по строке = str(rule))."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        for r in model.preds:
            f.write(str(r) + "\n")

if __name__ == "__main__":
    # -------------------- параметры и загрузка --------------------
    DATASET = "/Users/sargisvardanyan/PycharmProjects/DST/german.csv"

    TAG = Path(DATASET).stem  # "adult"

    X, y, feat_names = load_dataset(DATASET)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = OrderedDict()

    # -------------------- RIPPER raw rules --------------------
    print("================ RIPPER ==========================================")
    rip_raw = DSClassifierMultiQ(2,
                                 lr=0.05,
                                 max_iter=50,
                                 min_iter=5,
                                 min_dloss=1e-4,
                                 optim="adam",
                                 lossfn="MSE",
                                 batch_size=4096,
                                 num_workers=0,       # важно для mps
                                 precompute_rules=False,
                                 device="mps",        # или "cpu"
                                 force_precompute=False,
                                 use_foil=False)

    rip_raw.default_class_ = int(np.bincount(y_tr).argmax())

    raw_pkl = f"pkl_rules/ripper_rules_dataset_{TAG}_raw.pkl"
    raw_dsb = f"dsb_rules/ripper_rules_dataset_{TAG}_raw.dsb"

    if not os.path.exists(raw_pkl):
        rules = rip_raw.model.generate_ripper_rules(X_tr, y_tr, feat_names)
        rip_raw.model.import_test_rules(rules)
        rip_raw.model.save_rules_bin(raw_pkl)
    else:
        rip_raw.model.load_rules_bin(raw_pkl)

    rip_raw.model.precompute_fire_matrix(X_tr)
    set_test_usability(rip_raw.model, X_te)
    dump_rules(rip_raw.model, raw_dsb)

    raw_rules_ripper = len(rip_raw.model.preds)
    results["R_raw"] = metrics(y_te, rip_raw.predict_rules_only(X_te))

    # -------------------- RIPPER + DST --------------------
    rip_dst = DSClassifierMultiQ(2,
                                 lr=0.05,
                                 max_iter=50,
                                 min_iter=5,
                                 min_dloss=1e-4,
                                 optim="adam",
                                 lossfn="MSE",
                                 batch_size=4096,
                                 num_workers=0,       # чтобы не падало на MPS
                                 precompute_rules=False,
                                 device="mps",
                                 force_precompute=False,
                                 use_foil=False,
                                 batches_per_epoch=3)

    rip_dst.model.import_test_rules(rip_raw.model.preds)

    dst_pkl = f"pkl_rules/ripper_rules_dataset_{TAG}.pkl"
    dst_dsb = f"dsb_rules/ripper_rules_dataset_{TAG}.dsb"

    if not os.path.exists(dst_pkl):
        rip_dst.fit(X_tr, y_tr, column_names=feat_names,
                    optimize_weights=True, prune_after_fit=False)
        rip_dst.model.sort_rules_by_quality(X_tr)
        rip_dst.model.save_rules_bin(dst_pkl)
    else:
        rip_dst.model.load_rules_bin(dst_pkl)

    results["R_DST"] = metrics(y_te, rip_dst.predict(X_te))

    # -------------------- RIPPER + DST + prune --------------------
    rip_pr = DSClassifierMultiQ(2,
                                lr=0.05,
                                max_iter=50,
                                min_iter=5,
                                min_dloss=1e-4,
                                optim="adam",
                                lossfn="MSE",
                                batch_size=4096,
                                num_workers=0,
                                device="mps",
                                use_foil=False)
    rip_pr.model.import_test_rules(rip_raw.model.preds)
    rip_pr.model.load_rules_bin(dst_pkl)

    rip_pr.fit(X_tr, y_tr, feat_names,
               optimize_weights=False, prune_after_fit=True,
               prune_kwargs=dict(max_unc=0.6, min_score=0.3, min_cov=10))
    pr_pkl = f"pkl_rules/ripper_rules_dataset_{TAG}_pruned.pkl"
    pr_dsb = f"dsb_rules/ripper_rules_dataset_{TAG}_pruned.dsb"
    rip_pr.model.save_rules_bin(pr_pkl)

    results["R_pruned"] = metrics(y_te, rip_pr.predict(X_te))
    pruned_rules_ripper = len(rip_pr.model.preds)

    # -------------------- FOIL raw rules --------------------
    print("================ FOIL ==========================================")
    foil_raw = DSClassifierMultiQ(2,
                                  lr=0.05,
                                  max_iter=50,
                                  min_iter=5,
                                  min_dloss=1e-4,
                                  optim="adam",
                                  lossfn="MSE",
                                  batch_size=4096,
                                  num_workers=0,
                                  device="mps",
                                  use_foil=True)
    foil_raw.default_class_ = int(np.bincount(y_tr).argmax())

    fraw_pkl = f"pkl_rules/foil_rules_dataset_{TAG}_raw.pkl"
    fraw_dsb = f"dsb_rules/foil_rules_dataset_{TAG}_raw.dsb"

    if not os.path.exists(fraw_pkl):
        rules = foil_raw.model.generate_ripper_rules(X_tr, y_tr, feat_names)
        foil_raw.model.import_test_rules(rules)
        foil_raw.model.save_rules_bin(fraw_pkl)
    else:
        foil_raw.model.load_rules_bin(fraw_pkl)

    foil_raw.model.precompute_fire_matrix(X_tr)
    set_test_usability(foil_raw.model, X_te)
    dump_rules(foil_raw.model, fraw_dsb)

    raw_rules_foil = len(foil_raw.model.preds)
    results["F_raw"] = metrics(y_te, foil_raw.predict_rules_only(X_te))

    # -------------------- FOIL + DST --------------------
    foil_dst = DSClassifierMultiQ(2,
                                  lr=0.05,
                                  max_iter=50,
                                  min_iter=5,
                                  min_dloss=1e-4,
                                  optim="adam",
                                  lossfn="MSE",
                                  batch_size=4096,
                                  num_workers=0,
                                  device="mps",
                                  use_foil=True,
                                  batches_per_epoch=3)
    foil_dst.model.import_test_rules(foil_raw.model.preds)

    fdst_pkl = f"pkl_rules/foil_rules_dataset_{TAG}.pkl"
    fdst_dsb = f"dsb_rules/foil_rules_dataset_{TAG}.dsb"
    if not os.path.exists(fdst_pkl):
        foil_dst.fit(X_tr, y_tr, feat_names,
                     optimize_weights=True, prune_after_fit=False)
        foil_dst.model.sort_rules_by_quality(X_tr)
        foil_dst.model.save_rules_bin(fdst_pkl)
    else:
        foil_dst.model.load_rules_bin(fdst_pkl)

    results["F_DST"] = metrics(y_te, foil_dst.predict(X_te))

    # -------------------- FOIL + DST + prune --------------------
    foil_pr = DSClassifierMultiQ(2,
                                 lr=0.05,
                                 max_iter=50,
                                 min_iter=5,
                                 min_dloss=1e-4,
                                 optim="adam",
                                 lossfn="MSE",
                                 batch_size=4096,
                                 num_workers=0,
                                 device="mps",
                                 use_foil=True)
    foil_pr.model.import_test_rules(foil_raw.model.preds)
    foil_pr.model.load_rules_bin(fdst_pkl)

    foil_pr.fit(X_tr, y_tr, feat_names,
                optimize_weights=False, prune_after_fit=True,
                prune_kwargs=dict(max_unc=0.6, min_score=0.3, min_cov=10))
    fpr_pkl = f"pkl_rules/foil_rules_dataset_{TAG}_pruned.pkl"
    fpr_dsb = f"dsb_rules/foil_rules_dataset_{TAG}_pruned.dsb"
    foil_pr.model.save_rules_bin(fpr_pkl)

    results["F_pruned"] = metrics(y_te, foil_pr.predict(X_te))
    pruned_rules_foil = len(foil_pr.model.preds)

    # ------------- сохраняем результаты в DataFrame -------------
    df = pd.DataFrame(results).T
    csv_path = f"results/benchmark_dataset_{TAG}.csv"
    df.to_csv(csv_path, float_format="%.4f")

    # -------------------- функции для построения графиков --------------------
    def plot_metrics(df_metrics: pd.DataFrame, savepath: str):
        dfm = df_metrics.reset_index().melt(
            id_vars="index", var_name="Metric", value_name="Score"
        )
        dfm.rename(columns={"index": "System"}, inplace=True)

        plt.figure(figsize=(8, 4.5))
        ax = sns.barplot(
            data=dfm, x="System", y="Score", hue="Metric", palette="Set2"
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Score")
        ax.set_xlabel("")
        ax.legend(
            frameon=False,
            loc="upper center",
            ncol=4,
            bbox_to_anchor=(0.5, 1.10),
        )

        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height):
                ax.annotate(
                    f"{height:.2f}",
                    (p.get_x() + p.get_width() / 2, height),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="black",
                )

        plt.tight_layout()
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_rule_counts(
        ripper_sizes,
        foil_sizes,
        savepath,
    ):
        systems = ["Ripper raw", "Ripper pruned", "FOIL raw", "FOIL pruned"]
        counts = [
            ripper_sizes[0],
            ripper_sizes[1],
            foil_sizes[0],
            foil_sizes[1],
        ]
        df_r = pd.DataFrame({"System": systems, "Rules": counts})

        plt.figure(figsize=(7, 3.5))
        ax = sns.barplot(data=df_r, x="System", y="Rules", palette="Pastel1")
        ax.set_ylabel("# of rules")
        ax.set_xlabel("")

        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close()

    # -------------------- строим и сохраняем графики --------------------
    png_metrics = f"results/benchmark_dataset_{TAG}_metrics.png"
    png_rules   = f"results/benchmark_dataset_{TAG}_rules.png"

    plot_metrics(df, png_metrics)
    plot_rule_counts(
        (raw_rules_ripper, pruned_rules_ripper),
        (raw_rules_foil, pruned_rules_foil),
        png_rules,
    )

    print(f"\nSaved table → {csv_path}")
    print(f"Saved metrics plot → {png_metrics}")
    print(f"Saved rules count plot → {png_rules}")
