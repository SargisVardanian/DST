# ─────────────────────────────────────────────────────────────
# test_Ripper_DST.py  •  universal experiment runner
# ─────────────────────────────────────────────────────────────
import os, json, pickle
from pathlib import Path
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from Common_code.Datasets_loader   import load_dataset
from Common_code.DSClassifierMultiQ import DSClassifierMultiQ
from Common_code.core               import rules_to_dsb

sns.set_context("paper", font_scale=1.2)
sns.set_style  ("whitegrid")


# ─────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    """Convenient metric bundle."""
    return OrderedDict([
        ("Accuracy",  accuracy_score (y_true, y_pred)),
        ("F1",        f1_score       (y_true, y_pred, zero_division=0)),
        ("Precision", precision_score(y_true, y_pred, zero_division=0)),
        ("Recall",    recall_score   (y_true, y_pred, zero_division=0)),
    ])


def set_test_usability(model_wrapper, X_test):
    """Stores %-coverage (“usability”) inside each DSRule for later export."""
    fires = model_wrapper.model.fire_matrix(X_test)  # (N, R)
    cov   = fires.sum(axis=0)
    N     = len(X_test)
    for r, c in zip(model_wrapper.model.preds, cov):
        r.usability = float(c) / N * 100


def dump_rules(model_wrapper, filename):
    """Exports rules in .dsb format with the already-trained DST masses."""
    rules_to_dsb(model_wrapper.model.preds,
                 model_wrapper.model._params,
                 filename)


def rule_class_counts(model_wrapper, num_classes):
    """
    Returns an np.array[len = num_classes] with the number of rules
    assigned to every class (parsed from rule.caption).
    """
    counts = np.zeros(num_classes, dtype=int)
    for r in model_wrapper.model.preds:
        try:
            lbl = int(r.caption.split()[1].rstrip(":"))
            counts[lbl] += 1
        except Exception:
            # safety – put undecodable captions into class 0
            counts[0] += 1
    return counts


# ─────────────────────────────────────────────────────────────
# main experiment
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATASET      = PROJECT_ROOT / "german.csv"      # <-- change as needed
    TAG          = DATASET.stem

    # 2.  load & split
    X, y, feat_names = load_dataset(DATASET)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    num_classes = len(np.unique(y))

    # result containers
    metrics_tbl           = OrderedDict()     # { system : {metric:val} }
    rule_counts_total     = OrderedDict()     # { system : int }
    rule_counts_per_class = OrderedDict()     # { system : np.array[k] }

    # run both inducers
    for algo_name, use_foil in [("ripper", False), ("foil", True)]:
        prefix = algo_name[0]  # "r" or "f"
        # ───────────────────────────────────────── 1. RAW rules
        clf_raw = DSClassifierMultiQ(
            num_classes=num_classes,
            device="cpu",
            use_foil=use_foil
        )
        clf_raw.default_class_ = int(np.bincount(y_tr).argmax())

        raw_pkl = f"pkl_rules/{algo_name}_{TAG}_raw.pkl"
        raw_dsb = f"dsb_rules/{algo_name}_{TAG}_raw.dsb"

        if not os.path.exists(raw_pkl):
            rules = (clf_raw.model.generate_foil_rules(X_tr, y_tr, feat_names)
                     if use_foil else
                     clf_raw.model.generate_ripper_rules(X_tr, y_tr, feat_names))
            clf_raw.model.import_test_rules(rules)
            os.makedirs(os.path.dirname(raw_pkl), exist_ok=True)
            clf_raw.model.save_rules_bin(raw_pkl)
        else:
            clf_raw.model.load_rules_bin(raw_pkl)

        clf_raw.model.precompute_fire_matrix(X_tr)
        set_test_usability(clf_raw, X_te)
        dump_rules(clf_raw, raw_dsb)

        sys_key = f"{prefix}_raw"
        rule_counts_total    [sys_key] = len(clf_raw.model.preds)
        rule_counts_per_class[sys_key] = rule_class_counts(clf_raw, num_classes)
        metrics_tbl[sys_key]           = metrics(y_te, clf_raw.predict_rules_only(X_te))

        # ───────────────────────────────────────── 2. DST-training
        clf_dst = DSClassifierMultiQ(
            num_classes       = num_classes,
            device            = "cpu",
            use_foil          = use_foil,
            batches_per_epoch = 3
        )
        clf_dst.default_class_ = clf_raw.default_class_
        clf_dst.model.import_test_rules(clf_raw.model.preds)

        dst_pkl = f"pkl_rules/{algo_name}_{TAG}_dst.pkl"
        dst_dsb = f"dsb_rules/{algo_name}_{TAG}_dst.dsb"

        if not os.path.exists(dst_pkl):
            _, _, sorted_rules = clf_dst.fit(
                X_tr, y_tr, column_names=feat_names,
                optimize_weights=True, prune_after_fit=False, verbose=True
            )
            os.makedirs(os.path.dirname(dst_pkl), exist_ok=True)
            clf_dst.model.save_rules_bin(dst_pkl)
            dump_rules(clf_dst, dst_dsb)
        else:
            clf_dst.model.load_rules_bin(dst_pkl)

        sys_key = f"{prefix}_dst"
        rule_counts_total    [sys_key] = len(clf_dst.model.preds)
        rule_counts_per_class[sys_key] = rule_class_counts(clf_dst, num_classes)
        metrics_tbl[sys_key]           = metrics(y_te, clf_dst.predict(X_te))

        # ───────────────────────────────────────── 3. SORTED list (pre-prune)
        sorted_pkl = f"pkl_rules/{algo_name}_{TAG}_sorted.pkl"
        sorted_dsb = f"dsb_rules/{algo_name}_{TAG}_sorted.dsb"

        if not os.path.exists(sorted_pkl):
            # sorted_rules was built during DST fit()
            clf_dst.model.import_rules_with_params(
                sorted_rules,
                [p.clone().detach() for p in clf_dst.model._params]
            )
            clf_dst.model.save_rules_bin(sorted_pkl)
            clf_dst.model.precompute_fire_matrix(X_tr)
            set_test_usability(clf_dst, X_te)
            dump_rules(clf_dst, sorted_dsb)
        else:
            clf_dst.model.load_rules_bin(sorted_pkl)

        sys_key = f"{prefix}_sorted"
        rule_counts_total    [sys_key] = len(clf_dst.model.preds)
        rule_counts_per_class[sys_key] = rule_class_counts(clf_dst, num_classes)
        metrics_tbl[sys_key]           = metrics(y_te, clf_dst.predict_rules_only(X_te))

        # ───────────────────────────────────────── 4. PRUNED
        clf_pr = DSClassifierMultiQ(num_classes=num_classes, device="cpu", use_foil=use_foil)
        clf_pr.default_class_ = clf_dst.default_class_
        clf_pr.model.import_test_rules(clf_dst.model.preds)
        clf_pr.model.load_rules_bin(dst_pkl)

        pruned_pkl = f"pkl_rules/{algo_name}_{TAG}_pruned.pkl"
        pruned_dsb = f"dsb_rules/{algo_name}_{TAG}_pruned.dsb"

        if not os.path.exists(pruned_pkl):
            clf_pr.fit(X_tr, y_tr, column_names=feat_names,
                       optimize_weights=False, prune_after_fit=True,
                       prune_kwargs=dict(max_unc=0.7, min_ratio=1))
            clf_pr.model.save_rules_bin(pruned_pkl)
            dump_rules(clf_pr, pruned_dsb)
        else:
            clf_pr.model.load_rules_bin(pruned_pkl)

        sys_key = f"{prefix}_pruned"
        rule_counts_total    [sys_key] = len(clf_pr.model.preds)
        rule_counts_per_class[sys_key] = rule_class_counts(clf_pr, num_classes)
        metrics_tbl[sys_key]           = metrics(y_te, clf_pr.predict(X_te))


        # NEW:   first-fired evaluation on the same pruned list
        sys_key_first = f"{prefix}_pruned_first"
        # rule counts are identical to “_pruned”
        rule_counts_total    [sys_key_first] = rule_counts_total[sys_key]
        rule_counts_per_class[sys_key_first] = rule_counts_per_class[sys_key]
        metrics_tbl          [sys_key_first] = metrics(
            y_te,
            clf_pr.predict_rules_only(X_te)          # ← first-fired
        )
    # ──────────────────────────────────────────────── save metrics table
    df_metrics = pd.DataFrame(metrics_tbl).T
    df_metrics.index.name = "System"

    # pull the index into an explicit column, prepend our dataset tag
    df_metrics = df_metrics.reset_index()
    df_metrics.insert(0, "Dataset", TAG)

    # save (no more index column)
    out_csv = f"results/benchmark_dataset_{TAG}_metrics.csv"
    os.makedirs(Path(out_csv).parent, exist_ok=True)
    df_metrics.to_csv(out_csv, index=False, float_format="%.4f")

    # ──────────────────────────────────────────────── metric bar-plot
    dfm = df_metrics.melt(
        id_vars=["Dataset","System"],
        var_name="Metric",
        value_name="Score",
    )
    plt.figure(figsize=(10,8))
    ax = sns.barplot(
        data=dfm,
        x="System", y="Score",
        hue="Metric",
        palette="Set2",
    )
    # добавляем заголовок с именем датасета
    ax.set_title(f"Dataset: {TAG}", fontsize=14, pad=12)
    ax.set_xlabel("System")
    ax.set_ylabel("Score")
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"results/benchmark_dataset_{TAG}_metrics.png", dpi=300)
    plt.close()

    # ──────────────────────────────────────────────── stacked rule-count plot
    # build long-form frame: System · Class · Rules
    rows = []
    for sys, arr in rule_counts_per_class.items():
        for cls_idx, cnt in enumerate(arr):
            rows.append(dict(System=sys, Class=f"Class {cls_idx}", Rules=cnt))
    df_rules = pd.DataFrame(rows)
    df_rules.insert(0, "Dataset", TAG)

    # ordered systems for prettier x-axis
    order_systems = [
        "r_raw", "r_pruned",
        "f_raw", "f_pruned"
    ]
    df_rules["System"] = pd.Categorical(df_rules["System"],
                                        categories=order_systems, ordered=True)

    plt.figure(figsize=(8, 4))
    ax = sns.barplot(data=df_rules,
                     x="System", y="Rules",
                     hue="Class", palette="Pastel1",
                     estimator=sum, ci=None)
    # добавляем заголовок с именем датасета
    ax.set_title(f"Dataset: {TAG}", fontsize=14, pad=12)
    ax.set_xlabel("System")
    ax.set_ylabel("Number of Rules")
    # annotate total on top of every bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='edge', padding=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"results/benchmark_dataset_{TAG}_rules.png", dpi=300)
    plt.close()

    print(f"✓  Metrics  → {out_csv}")
    print(f"✓  Plots    → results/benchmark_dataset_{TAG}_rules.png")
