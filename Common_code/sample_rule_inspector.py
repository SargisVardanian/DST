# Common_code/sample_rule_inspector.py
# Коротко: грузит PKL, печатает активные правила только с mass(K+Ω),
# полагаясь на вспомогательные методы модели для вычислений. Всё сохраняет в JSON.
from __future__ import annotations
import argparse
import json
import sys
import numpy as np

# The ``dill`` module is used for serialising Python objects.  It may not
# always be available in constrained environments.  Attempt to import
# ``dill`` and fall back to the built‑in ``pickle`` module if it is not
# installed.  ``pickle`` provides similar functionality for basic data
# structures.
try:
    import dill  # type: ignore
except ImportError:
    import pickle as dill  # type: ignore
from pathlib import Path
from typing import Any, Dict, List, Optional

THIS = Path(__file__).resolve(); COMMON = THIS.parent; PROJECT_ROOT = COMMON.parent
sys.path.insert(0, str(COMMON))
from Datasets_loader import load_dataset
from DSClassifierMultiQ import DSClassifierMultiQ
from core import params_to_mass

# ---------- tiny utils ----------
_f = lambda x: [float(v) for v in np.asarray(x, float).ravel().tolist()]

def _pkl(algo: str, ds: str) -> Path:
    base = COMMON / "pkl_rules"
    stem = f"{algo.lower()}_{Path(ds).stem}_dst.pkl"
    direct = base / stem
    if direct.exists():
        return direct
    for prefix in ("stable_", "conserv_", "baseline_", "bench_"):
        candidate = base / f"{prefix}{stem}"
        if candidate.exists():
            return candidate
    return direct
def _decode(dec: Dict[Any,Dict[Any,Any]], names: List[str], j: int, v: float) -> Optional[str]:
    mp = (dec or {}).get(names[j]) or (dec or {}).get(j)
    if not isinstance(mp, dict): return None
    for k in (v, int(round(v)), float(v), str(v), str(int(round(v)))):
        if k in mp: return str(mp[k])
    for k, txt in mp.items():
        try:
            if abs(float(k)-float(v))<1e-9: return str(txt)
        except: pass
    return None

def _prior_mass_from_blob(blob: Any) -> Optional[List[float]]:
    if not isinstance(blob, dict):
        return None
    bias_payload = blob.get("bias_mass_params")
    if bias_payload is None:
        bias_payload = blob.get("bias_mass_logits")
    if bias_payload is not None:
        logits = np.asarray(bias_payload, dtype=np.float32).reshape(1, -1)
        mass = params_to_mass(logits)[0]
        return _f(mass)
    if blob.get("prior_mass") is not None:
        return _f(blob["prior_mass"])
    if blob.get("prior_logits") is not None:
        logits = np.asarray(blob["prior_logits"], dtype=np.float32).reshape(1, -1)
        mass = params_to_mass(logits)[0]
        return _f(mass)
    return None

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Inspect fired rules → masses/conf → JSON")
    ap.add_argument("--dataset", default="adult")
    ap.add_argument("--algo", default="RIPPER", choices=["STATIC","RIPPER","FOIL"])
    ap.add_argument("--idx", type=int, default=7)
    ap.add_argument("--pkl", default="")
    args = ap.parse_args()

    # data + sample
    csv = Path(args.dataset)
    if csv.suffix.lower() != ".csv":
        for c in (PROJECT_ROOT/f"{args.dataset}.csv", COMMON/f"{args.dataset}.csv", Path.cwd()/f"{args.dataset}.csv"):
            if c.is_file(): csv=c; break
    X,y,feat,dec = load_dataset(csv); y = np.asarray(y,int)
    print(f"Detected label column: 'labels'  →  classes: {sorted(np.unique(y).tolist())}")
    print(f"X shape = {X.shape},  y distribution = {np.bincount(y).tolist()}")

    i = int(args.idx)%len(X);
    x = np.asarray(X[i],float);
    y_true=int(y[i])

    print(f"\nSelected sample index: {i}")
    feats_disp={}
    for j,n in enumerate(feat):
        t=_decode(dec,list(feat),j,float(x[j]));
        if t is None: t=str(int(x[j]) if float(x[j]).is_integer() else float(x[j]))
        print(f"  - {n}={t}"); feats_disp[n]=t

    # model + PKL
    pkl = Path(args.pkl) if args.pkl else _pkl(args.algo, csv.stem)
    clf = DSClassifierMultiQ(k=int(np.unique(y).size), algo=args.algo, value_decoders=dec, feature_names=list(feat))
    clf.model.load_rules_bin(str(pkl))
    try:
        with open(pkl,"rb") as fh: blob=dill.load(fh)
    except Exception:
        blob={}

    # activated rules via модельный helper
    export = clf.model.prepare_rules_for_export(x)
    activated = export.get("activated_rules", [])
    # Sort by activation/mass weight for readability; keep top-10
    def _score(item: Dict[str, Any]) -> float:
        act = float(item.get("activation", 0.0) or 0.0)
        mass = item.get("mass")
        cls = item.get("class")
        w = 0.0
        if isinstance(mass, (list, tuple)) and cls is not None:
            try:
                w = float(mass[int(cls)])
            except Exception:
                w = 0.0
        return 0.7 * act + 0.3 * w
    activated = sorted(activated, key=_score, reverse=True)[:10]
    confidence = _f(export.get("confidence", []))
    mass_vector = _f(export.get("mass_vector", []))
    pred = int(export.get("predicted_class", 0))

    # print
    total_rules = len(getattr(clf.model, "rules", []) or [])
    print(f"\n[info] fired rules: {len(activated)} / {total_rules}")
    if activated:
        print("Activated rules (top-10 by activation/weight):")
        for item in activated:
            cap = item.get("condition", "<rule>")
            lbl = item.get("class")
            mass = item.get("mass")
            act = item.get("activation")
            stats = item.get("stats") if isinstance(item.get("stats"), dict) else None
            origin = (stats or {}).get("origin") if isinstance(stats, dict) else None
            head = f"(Class {lbl}) " if lbl is not None else ""
            extras = []
            if act is not None:
                extras.append(f"act={float(act):.3f}")
            if mass is not None:
                extras.append(f"mass={_f(mass)}")
            if stats:
                prec = stats.get("precision")
                rec = stats.get("recall")
                supp = stats.get("support")
                f1 = stats.get("f1")
                if prec is not None and rec is not None:
                    extras.append(f"pr={float(prec):.3f}/rc={float(rec):.3f}")
                if f1 is not None:
                    extras.append(f"f1={float(f1):.3f}")
                if supp is not None:
                    extras.append(f"supp={int(supp)}")
            if origin is not None:
                extras.append(f"origin={origin}")
            suffix = " -> " + ", ".join(extras) if extras else ""
            print(f" - {head}{cap}{suffix}")

    # Display the belief mass and derive a prediction based solely on the mass vector.
    # The mass vector includes one entry per class followed by the uncertainty
    # mass (Ω).  A sample is considered "uncertain" when the uncertainty
    # exceeds the largest class mass.
    display_pred: Any = None
    if mass_vector:
        print(f"\nBelief mass (classes + Ω): {mass_vector}")
        k = len(mass_vector) - 1
        if k > 0:
            class_masses = mass_vector[:-1]
            uncertainty = mass_vector[-1]
            best_class = int(np.argmax(class_masses))
            uncertainty = float(uncertainty)
            predicted_label = best_class
            dominates = uncertainty >= class_masses[best_class]
            display_pred = f"{predicted_label} (unc={uncertainty:.3f}{', dominates' if dominates else ''})"
        else:
            display_pred = "n/a"
        print(f"Prediction using belief mass: {display_pred}")
    # Always show pignistic probabilities as a reference (optional)
    if confidence:
        print(f"Class mass (per class): {confidence}")
    # Display true label and predicted label (via belief mass when available)
    if mass_vector:
        print(f"True label: {y_true} | Predicted: {display_pred}")
    else:
        print(f"True label: {y_true} | Predicted: {pred}")

    prior_mass = _prior_mass_from_blob(blob)

    # JSON
    out = COMMON/"results"; out.mkdir(parents=True, exist_ok=True)
    path = out / f"inspect_{args.algo.lower()}_{csv.stem}_idx{i}.json"
    payload = {
        "dataset": str(csv), "algo": args.algo, "pkl": str(pkl), "index": i,
        "true_label": y_true, "predicted_class": pred,
        "confidence": confidence,
        "mass_vector": mass_vector,
        "prior_mass": prior_mass,
        "prior_logits": (_f(blob["prior_logits"]) if isinstance(blob, dict) and blob.get("prior_logits") is not None else None),
        "counts": {"rules_total": int(total_rules), "rules_fired": int(len(activated))},
        "features_display": feats_disp,
        "activated_rules": [{
            "condition": item.get("condition", "<rule>"),
            "class": (int(item["class"]) if item.get("class") is not None else None),
            "activation": (float(item["activation"]) if item.get("activation") is not None else None),
            "mass": (_f(item.get("mass")) if item.get("mass") is not None else None),
            "stats": (item.get("stats") if isinstance(item.get("stats"), dict) else None),
        } for item in activated],
    }
    with open(path,"w",encoding="utf-8") as f: json.dump(payload,f,ensure_ascii=False,indent=2)
    print(f"\n✓ Saved report → {path}")

if __name__=="__main__":
    main()
