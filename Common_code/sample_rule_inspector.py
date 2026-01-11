from __future__ import annotations

"""
Simple script to inspect a single sample using a saved rule model.
Loads a dataset and a pre-trained model, then runs prediction on a specific sample.
"""

import argparse
from pathlib import Path
import sys
import numpy as np

THIS = Path(__file__).resolve()
COMMON = THIS.parent
ROOT = COMMON.parent

# Add Common_code to the import path for local modules
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from Datasets_loader import load_dataset
from core import split_train_test
from DSClassifierMultiQ import DSClassifierMultiQ

def _decode_value(name: str, value: float, decoders: dict[str, dict[int, str]]) -> str:
    """Helper to decode categorical values for display."""
    mapping = decoders.get(name)
    if not mapping:
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    # Try exact match first
    if value in mapping:
        return mapping[value]
    try:
        ivalue = int(value)
    except Exception:
        return str(value)
    if ivalue in mapping:
        return mapping[ivalue]

    return str(value)

def resolve_dataset(path_or_name: str) -> Path:
    """Resolve a dataset path from a name or explicit CSV filename."""
    path = Path(path_or_name)
    if path.suffix.lower() == ".csv" and path.is_file():
        return path
    for base in (ROOT, COMMON, Path.cwd()):
        candidate = base / f"{path_or_name}.csv"
        if candidate.is_file():
            return candidate
    return path

def default_model_path(algo: str, dataset: Path, *, run_tag: str = "") -> Path:
    """Return a default model path for a given algorithm and dataset."""
    base = COMMON / "pkl_rules"
    tag = f"_{run_tag}" if str(run_tag).strip() else ""
    stem = f"{algo.lower()}_{dataset.stem}{tag}_dst.pkl"
    direct = base / stem
    if direct.exists():
        return direct
    # Check .dsb file as well since load_model expects binary/pkl
    stem_dsb = f"{algo.lower()}_{dataset.stem}{tag}_dst.dsb"
    direct_dsb = COMMON / "dsb_rules" / stem_dsb
    if direct_dsb.exists():
        return direct_dsb
    # Fallback to the untagged default names (legacy).
    legacy = base / f"{algo.lower()}_{dataset.stem}_dst.pkl"
    if legacy.exists():
        return legacy
    legacy_dsb = COMMON / "dsb_rules" / f"{algo.lower()}_{dataset.stem}_dst.dsb"
    if legacy_dsb.exists():
        return legacy_dsb
    return direct

def main() -> None:
    parser = argparse.ArgumentParser("Inspect a single sample using a saved rule model")
    parser.add_argument("--dataset", required=True, help="Dataset name or path")
    parser.add_argument("--algo", default="RIPPER", choices=["STATIC", "RIPPER", "FOIL"], help="Algorithm used")
    parser.add_argument("--idx", type=int, default=0, help="Sample index within chosen split")
    parser.add_argument("--row-index", type=int, default=None, help="Select by original dataset row index (after --split)")
    parser.add_argument("--model", default="", help="Path to the saved model (.pkl or .dsb)")
    parser.add_argument("--run-tag", default="", help="Optional run tag suffix used in saved model filenames")
    parser.add_argument("--split", choices=["full", "train", "test"], default="test", help="Subset to inspect (matches benchmark split)")
    parser.add_argument("--test-size", type=float, default=0.16, help="Test split fraction for train/test subset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test subset")
    parser.add_argument(
        "--combine-rule",
        default="dempster",
        choices=["dempster", "yager", "vote"],
        help="Combination rule to use (dempster, yager, vote).",
    )
    parser.add_argument("--show-combined", action="store_true", help="Show combined rule with weighted literals")
    parser.add_argument("--eval-combined", action="store_true", help="Evaluate combined rule on training data")
    args = parser.parse_args()

    # 1. Load Dataset
    csv_path = resolve_dataset(args.dataset)
    if not csv_path.exists():
        print(f"Error: Dataset not found at {csv_path}")
        return

    # Load dataset with stats to get original class labels
    X, y, feature_names, value_decoders, stats = load_dataset(csv_path, return_stats=True)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    idx_all = np.arange(len(X))
    X_view, y_view, idx_view = X, y, idx_all
    if args.split != "full":
        X_tr, X_te, y_tr, y_te, idx_tr, idx_te = split_train_test(
            X,
            y,
            test_size=float(args.test_size),
            seed=int(args.seed),
            stratify=True,
        )
        if args.split == "train":
            X_view, y_view, idx_view = X_tr, y_tr, idx_tr
        else:
            X_view, y_view, idx_view = X_te, y_te, idx_te
        print(f"[split] {args.split} n={len(X_view)} (seed={args.seed}, test_size={args.test_size})")

    # Get original class labels (before factorization)
    original_classes = stats.get('classes', [])  # e.g., [1, 2, 3, 4, 5, 6] for gas_drift

    # Create mapping: index -> original label
    index_to_original = {idx: label for idx, label in enumerate(original_classes)} if original_classes else {}

    # 2. Select Sample
    if args.row_index is not None:
        hits = np.where(idx_view == int(args.row_index))[0]
        if len(hits) == 0:
            raise SystemExit(f"row-index {args.row_index} not found in split '{args.split}'")
        sample_idx = int(hits[0])
    else:
        sample_idx = int(args.idx) % len(X_view)
    sample = X_view[sample_idx]
    true_label_idx = int(y_view[sample_idx])
    true_label_original = index_to_original.get(true_label_idx, true_label_idx)
    original_row_idx = int(idx_view[sample_idx])

    print(f"\nInspecting Sample #{sample_idx}")
    print("-" * 40)
    print(f"{'Row Index':<20}: {original_row_idx}")
    MAX_FEATS = 15
    if len(feature_names) > MAX_FEATS + 5:
        # Show first 10
        for i in range(10):
            name, value = feature_names[i], sample[i]
            display = _decode_value(name, value, value_decoders)
            print(f"{name:<20}: {display}")

        print(f"{'...':<20}: ... ({len(feature_names) - 15} more features) ...")

        # Show last 5
        for i in range(len(feature_names) - 5, len(feature_names)):
            name, value = feature_names[i], sample[i]
            display = _decode_value(name, value, value_decoders)
            print(f"{name:<20}: {display}")
    else:
        for name, value in zip(feature_names, sample):
            display = _decode_value(name, value, value_decoders)
            print(f"{name:<20}: {display}")
    if index_to_original:
        print(f"{'True Label':<20}: {true_label_original} (idx={true_label_idx})")
    else:
        print(f"{'True Label':<20}: {true_label_idx}")
    print("-" * 40)

    # 3. Load Model
    model_path = Path(args.model) if args.model else default_model_path(args.algo, csv_path, run_tag=args.run_tag)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    print(f"\nLoading model from: {model_path}")
    # Initialize classifier with correct signature
    k = int(np.unique(y).size)
    # NOTE: We pass rule_algo, not algo
    clf = DSClassifierMultiQ(
        k=k,
        rule_algo=args.algo,
        device="cpu"
    )

    # Load the trained rules and masses
    # Note: load_model expects the path to the .dsb file usually, or .pkl
    # DSClassifierMultiQ.load_model calls self.model.load_rules_bin(path)
    # If we passed a .pkl, we might need to adjust if load_rules_bin expects .dsb
    # But typically the save logic saves both. Let's try loading.
    try:
        clf.load_model(str(model_path))
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try changing extension to .dsb if .pkl was passed
        if model_path.suffix == '.pkl':
            dsb_path = model_path.with_suffix('.dsb')
            if dsb_path.exists():
                print(f"Retrying with {dsb_path.name}...")
                clf.load_model(str(dsb_path))
            else:
                return
        else:
            return

    # 4. Predict
    batch = sample.reshape(1, -1)
    combine_rule = str(args.combine_rule)

    # Use centralized model logic for all modes
    pred_label_idx = int(clf.predict(batch, combination_rule=combine_rule)[0])
    pred_label_original = index_to_original.get(pred_label_idx, pred_label_idx)

    # Get explanation via unified method
    combined_d = clf.model.get_combined_rule(
        sample, 
        return_details=True, 
        combination_rule=combine_rule,
    )
    
    betp = np.zeros(k)  # Placeholder for display
    if "error" not in combined_d:
        unc_stats = clf.model.uncertainty_stats(batch, combination_rule=combine_rule)
        unc_rule = float(unc_stats["unc_rule"][0]) if len(unc_stats["unc_rule"]) else float("nan")
        # We can also get masses from predict_masses for probability display
        betp = clf.model.forward(batch, combination_rule=combine_rule)[0].detach().cpu().numpy()
    else:
        unc_rule = float("nan")

    # 5. Show Activated Rules
    print("\nActivated Rules")
    print("-" * 40)
    export = clf.model.prepare_rules_for_export(sample)
    activated = export.get("activated_rules", [])

    if not activated:
        print("No rules fired (Uncertainty/Omega dominant).")
    else:
        # Calculate H-scores for sorting
        # H = 2 * U' * R' / (U' + R')
        # U' = 1 - m(Omega)
        # R = max(m) / 2nd_max(m)
        # R' = normalized R

        rule_metrics = []
        r_values = []

        for item in activated:
            mass = np.array(item.get("mass", []))
            if len(mass) < 2:
                u_prime, r_val = 0.0, 0.0
            else:
                omega = mass[-1]
                u_prime = 1.0 - omega

                # Compare class masses for contrast
                class_masses = mass[:-1]
                if class_masses.size > 0:
                    sorted_m = np.sort(class_masses)
                    max_m = sorted_m[-1]
                    sec_m = sorted_m[-2] if len(sorted_m) > 1 else 0.0
                    r_val = max_m / (sec_m + 1e-9)
                else:
                    r_val = 0.0

            r_values.append(r_val)
            rule_metrics.append({'item': item, 'u_prime': u_prime, 'r_raw': r_val})

        # Normalize R
        # Use log scale for R to handle huge differences (e.g. 100 vs 1e9)
        # R_val can be 0 or inf.
        # We clip R_val to a reasonable max (e.g. 1e6) to avoid inf issues in normalization

        r_values_log = []
        for rm in rule_metrics:
            r_val = rm['r_raw']
            # If r_val is inf or extremely large, clip it
            if np.isinf(r_val) or r_val > 1e6:
                r_val = 1e6
            # Use log1p to compress scale
            r_log = np.log1p(r_val)
            r_values_log.append(r_log)
            rm['r_log'] = r_log

        r_min = min(r_values_log) if r_values_log else 0.0
        r_max = max(r_values_log) if r_values_log else 1.0

        if r_max - r_min < 1e-9:
            r_denom = 1.0
        else:
            r_denom = r_max - r_min

        # Compute H and sort
        for rm in rule_metrics:
            r_prime = (rm['r_log'] - r_min) / r_denom
            u_prime = rm['u_prime']
            h = 0.0 if u_prime + r_prime < 1e-9 else 2 * u_prime * r_prime / (u_prime + r_prime)
            rm['h_score'] = h

        # Sort by H-score descending
        rule_metrics.sort(key=lambda x: x['h_score'], reverse=True)

        for rm in rule_metrics:
            item = rm['item']
            h = rm['h_score']
            rid = item.get('id')
            cond = item.get("condition", "???")
            cls = item.get("class")
            mass = item.get("mass")
            stats = item.get("stats", {})

            # Format mass vector
            mass_str = str(np.round(mass, 3)) if mass is not None else "N/A"

            # Try to infer class from mass if not explicit
            if cls is None and mass is not None and len(mass) > 1:
                cls = int(np.argmax(mass[:-1]))

            # Display class with original label if available
            cls_display = index_to_original.get(cls, cls) if cls is not None and index_to_original else cls

            header = f"Rule #{rid:<3} | Class: {cls_display} (idx={cls}) | H-score: {h:.3f} | Mass: {mass_str}"
            if not index_to_original or cls is None:
                header = f"Rule #{rid:<3} | Class: {cls} | H-score: {h:.3f} | Mass: {mass_str}"

            print(header)
            print(f"  Condition: {cond}")
            if stats:
                prec, cov = stats.get('precision', 'N/A'), stats.get('coverage', 'N/A')
                print(f"  Stats    : Precision={prec}, Coverage={cov}")
            print("-" * 20)

    # 6. Show Results
    print(f"\nPrediction Results")
    print("-" * 40)
    correct = (pred_label_idx == true_label_idx)
    pred_label_name = f"{combine_rule.capitalize()} Prediction"
    
    if index_to_original:
        pred_disp = f"{pred_label_original} (idx={pred_label_idx})"
        true_disp = f"{true_label_original} (idx={true_label_idx})"
    else:
        pred_disp = f"{pred_label_idx}"
        true_disp = f"{true_label_idx}"
    status = "CORRECT" if correct else f"WRONG (true: {true_disp})"
    print(f"{pred_label_name:<20}: {pred_disp} {status}")
    
    if combine_rule == "vote":
        print(f"Scores              : {np.round(betp, 4)}")
        print("Uncertainty (Omega) : N/A")
    else:
        print(f"DST Probs           : {np.round(betp, 4)}")
        unc_stats = clf.model.uncertainty_stats(batch, combination_rule=combine_rule)
        unc_rule = float(np.nanmean(unc_stats["unc_rule"]))
        print(f"Omega (rules avg)   : {unc_rule:.4f}" if np.isfinite(unc_rule) else "Omega (rules avg)   : N/A")
    print("-" * 40)

    # 7. Show Combined Rules (if --show-combined flag)
    if args.show_combined:
        if "error" in combined_d:
            print(f"Error: {combined_d['error']}")
        else:
            rule_name = "Vote" if combine_rule == "vote" else "DST Combination"
            print(f"\nCombined Rule ({rule_name})")
            print("=" * 50)
            
            pred_display = index_to_original.get(combined_d["predicted_class"], combined_d["predicted_class"]) if index_to_original else combined_d["predicted_class"]
            
            print(f"Predicted Class     : {pred_display} (idx={combined_d['predicted_class']})")
            if combine_rule != "vote":
                print(f"Rules Fired         : {combined_d['n_rules_fired']} (agreeing: {combined_d['n_agreeing']}, conflicting: {combined_d['n_conflicting']})")
            
            if combined_d.get("combined_condition"):
                print(f"Combined Condition (weighted literals):")
                print(f"  {combined_d['combined_condition']}")
            print()

            if combined_d.get("rule_contributions"):
                print("Rule Contributions (sorted by weight):")
                print(f"  [method={combine_rule}]")
                print()
                for rc in combined_d["rule_contributions"]:
                    agree_sym = "✓" if rc["agrees"] else "✗"
                    cls_lbl = index_to_original.get(rc['rule_class'], rc['rule_class']) if index_to_original else rc['rule_class']
                    
                    if combine_rule == "vote":
                        print(f"  {agree_sym} Rule #{rc['rule_id']:3d} | w={rc['weight']:+.4f} | class={cls_lbl}")
                    else:
                        print(f"  {agree_sym} Rule #{rc['rule_id']:3d} | w={rc['weight']:+.4f} | cert={rc.get('certainty', 0.0):.3f} | Ω={rc.get('omega', 0.0):.3f} | class={cls_lbl}")
                    print(f"      {rc['caption'][:80]}")
            print()

            if combine_rule != "vote":
                print("Combined Mass Distribution:")
                try:
                    masses = clf.model.predict_masses(batch, combination_rule=combine_rule)[0]
                    class_names = [f"C{i}" for i in range(len(masses) - 1)] + ["Ω"]
                    bar_width = 40
                    print(f"  {'Class':<8} {'Mass':<8} Bar")
                    for i, (name, m) in enumerate(zip(class_names, masses)):
                        bar_len = int(m * bar_width)
                        symbol = "█" if i == combined_d['predicted_class'] else "░"
                        if name == "Ω": symbol = "▒"
                        print(f"  {name:<8} {m:<8.4f} {symbol * bar_len}")
                except Exception as e:
                    print(f"  [Error computing masses: {e}]")
        print("=" * 50)

        # --- Evaluate combined rule on training data ---
        if args.eval_combined and combined_d.get("combined_literals"):
            print(f"\nEvaluating Combined Rule on Training Data")
            print("=" * 50)
            
            # Build a mask for all samples that satisfy the combined rule
            literals = combined_d["combined_literals"]
            pred_cls = combined_d["predicted_class"]
            
            # Create mask: which samples satisfy all literals
            mask = np.ones(len(X), dtype=bool)
            for lit in literals:
                name, op, val = lit["literal"]
                # Find feature index
                feat_idx = feature_names.index(name) if name in feature_names else -1
                if feat_idx < 0:
                    continue
                col = X[:, feat_idx]
                if op == "==":
                    lit_mask = np.isclose(col, float(val), atol=1e-9)
                elif op == ">":
                    lit_mask = col > float(val)
                elif op == "<":
                    lit_mask = col < float(val)
                else:
                    continue
                mask &= lit_mask
            
            n_matching = mask.sum()
            if n_matching > 0:
                y_matching = y[mask]
                precision = (y_matching == pred_cls).mean()
                coverage = n_matching / len(X)
                
                # Also check original model predictions on matching samples
                X_matching = X[mask]
                model_preds = clf.predict(X_matching)
                model_acc = (model_preds == y_matching).mean()
                
                print(f"  Samples Matching: {n_matching} ({coverage*100:.2f}%)")
                print(f"  Rule Precision: {precision*100:.2f}% (true label == predicted class)")
                print(f"  Model Accuracy: {model_acc*100:.2f}% (on matching subset)")
            else:
                print("  No samples match the combined rule on training data.")
            print("=" * 50)

if __name__ == "__main__":
    main()
