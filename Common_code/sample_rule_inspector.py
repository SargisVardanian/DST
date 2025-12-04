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
from DSClassifierMultiQ import DSClassifierMultiQ

def _decode_value(name: str, value: float, decoders: dict[str, dict[int, str]]) -> str:
    """Helper to decode categorical values for display."""
    mapping = decoders.get(name)
    if not mapping:
        return f"{value:.4f}" if isinstance(value, float) else str(value)
    
    # Try exact match first
    if value in mapping: return mapping[value]
    if int(value) in mapping: return mapping[int(value)]
    
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

def default_model_path(algo: str, dataset: Path) -> Path:
    """Return a default model path for a given algorithm and dataset."""
    base = COMMON / "pkl_rules"
    stem = f"{algo.lower()}_{dataset.stem}_dst.pkl"
    direct = base / stem
    if direct.exists():
        return direct
    # Check .dsb file as well since load_model expects binary/pkl
    stem_dsb = f"{algo.lower()}_{dataset.stem}_dst.dsb"
    direct_dsb = COMMON / "dsb_rules" / stem_dsb
    if direct_dsb.exists():
        return direct_dsb
    return direct

def main() -> None:
    parser = argparse.ArgumentParser("Inspect a single sample using a saved rule model")
    parser.add_argument("--dataset", default="adult", help="Dataset name or path")
    parser.add_argument("--algo", default="RIPPER", choices=["STATIC", "RIPPER", "FOIL"], help="Algorithm used")
    parser.add_argument("--idx", type=int, default=0, help="Index of the sample to inspect")
    parser.add_argument("--model", default="", help="Path to the saved model (.pkl or .dsb)")
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
    
    # Get original class labels (before factorization)
    original_classes = stats.get('classes', [])  # e.g., [1, 2, 3, 4, 5, 6] for gas_drift
    
    # Create mapping: index -> original label
    index_to_original = {idx: label for idx, label in enumerate(original_classes)} if original_classes else {}

    # 2. Select Sample
    sample_idx = int(args.idx) % len(X)
    sample = X[sample_idx]
    true_label_idx = int(y[sample_idx])
    true_label_original = index_to_original.get(true_label_idx, true_label_idx)
    
    print(f"\nInspecting Sample #{sample_idx}")
    print("-" * 40)
    for name, value in zip(feature_names, sample):
        display = _decode_value(name, value, value_decoders)
        print(f"{name:<20}: {display}")
    if index_to_original:
        print(f"{'True Label (orig)':<20}: {true_label_original}")
        print(f"{'True Label (index)':<20}: {true_label_idx}")
    else:
        print(f"{'True Label':<20}: {true_label_idx}")
    print("-" * 40)

    # 3. Load Model
    model_path = Path(args.model) if args.model else default_model_path(args.algo, csv_path)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    print(f"\nLoading model from: {model_path.name}")
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
    # Prepare batch (1 sample)
    batch = sample.reshape(1, -1)
    
    # Get DST output.
    # NOTE: DSModelMultiQ.forward (used by predict_with_dst) already returns
    # class-wise probabilities of shape (k,) – there is no explicit Omega
    # mass in this representation.
    dst_output = np.asarray(clf.model.predict_with_dst(batch)[0], dtype=float)
    if dst_output.shape[0] != k:
        # Fallback: if someone loads an older model that returns K+1 masses
        # (classes + Omega), convert to pignistic probabilities.
        if dst_output.shape[0] == k + 1:
            omega = float(dst_output[-1])
            dst_output = dst_output[:-1] + (omega / float(k))
        else:
            raise ValueError(
                f"Unexpected DST output shape {dst_output.shape}; "
                f"expected {k} (num_classes) or {k + 1} (classes + Omega)."
            )

    betp = dst_output
    pred_label_idx = int(np.argmax(betp))
    pred_label_original = index_to_original.get(pred_label_idx, pred_label_idx)
    # Per-sample model uncertainty (average Omega over active rules)
    try:
        unc_arr = clf.model.sample_uncertainty(batch)
        unc = float(np.asarray(unc_arr, dtype=float).reshape(-1)[0])
    except Exception:
        unc = float("nan")

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
                u_prime = 0.0
                r_val = 0.0
            else:
                omega = mass[-1]
                u_prime = 1.0 - omega
                
                # Sort masses to find max and 2nd max (excluding Omega if it's last? No, Omega is last element)
                # We care about class masses contrast.
                class_masses = mass[:-1]
                if len(class_masses) > 0:
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
            if u_prime + r_prime < 1e-9:
                h = 0.0
            else:
                h = 2 * u_prime * r_prime / (u_prime + r_prime)
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
                prec = stats.get('precision', 'N/A')
                cov = stats.get('coverage', 'N/A')
                print(f"  Stats    : Precision={prec}, Coverage={cov}")
            print("-" * 20)

    # 6. Show Results
    print(f"\nPrediction Results")
    print("-" * 40)
    correct = (pred_label_idx == true_label_idx)
    if index_to_original:
        print(f"DST Prediction      : {pred_label_original} (idx={pred_label_idx}) {'(CORRECT)' if correct else '(WRONG)'}")
    else:
        print(f"DST Prediction      : {pred_label_idx} {'(CORRECT)' if correct else '(WRONG)'}")
    print(f"DST Probs           : {np.round(betp, 4)}")
    print(f"Uncertainty (Omega) : {unc:.4f}" if np.isfinite(unc) else f"Uncertainty (Omega) : N/A")
    print("-" * 40)

if __name__ == "__main__":
    main()
