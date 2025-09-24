# Common_code/sample_rule_inspector.py
# Коротко: грузит PKL, печатает активные правила только с mass(K+Ω),
# комбинирует массы → final_mass, betP → confidence. Всё сохраняет в JSON.
from __future__ import annotations
import argparse, json, sys, dill, numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

THIS = Path(__file__).resolve(); COMMON = THIS.parent; PROJECT_ROOT = COMMON.parent
sys.path.insert(0, str(COMMON))
from Datasets_loader import load_dataset
from DSClassifierMultiQ import DSClassifierMultiQ

# ---------- tiny utils ----------
_f = lambda x: [float(v) for v in np.asarray(x, float).ravel().tolist()]
def _pkl(algo: str, ds: str) -> Path: return COMMON / "pkl_rules" / f"{algo.lower()}_{Path(ds).stem}_dst.pkl"
def _softmax_rows(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, float); m = W.max(axis=1, keepdims=True); E = np.exp(W - m); return E/(E.sum(axis=1, keepdims=True)+1e-12)
def _combine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    K = a.shape[0]-1; s = a[:K]*b[:K] + a[:K]*b[K] + a[K]*b[:K]; o = a[K]*b[K]; out = np.concatenate([s,[o]])
    return out/(out.sum()+1e-12)
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

def _get_K(y: np.ndarray, blob: Any, params: Optional[np.ndarray]) -> int:
    if isinstance(blob, dict) and isinstance(blob.get("prior_logits"), (list, np.ndarray)):
        return int(np.asarray(blob["prior_logits"]).size)
    if params is not None and params.ndim==2: return int(params.shape[1]-1)
    u = np.unique(y).astype(int); return int(u.size)

def _get_prior_mass(model, blob: Any, K: int) -> np.ndarray:
    try:
        pm = getattr(model, "_prior_mass", None)
        if pm is not None: return np.asarray(pm.detach().cpu().numpy(), float).ravel()
    except Exception: pass
    if isinstance(blob, dict) and blob.get("prior_mass") is not None:
        return np.asarray(blob["prior_mass"], float).ravel()
    if isinstance(blob, dict) and blob.get("prior_logits") is not None:
        pl = np.asarray(blob["prior_logits"], float).ravel()
        p = np.exp(pl-pl.max()); p/= (p.sum()+1e-12)
    else:
        p = np.full(K, 1.0/max(1,K), float)
    omega=0.02; return np.concatenate([p*(1.0-omega), [omega]])

def _params_from(blob: Any, model, K: int|None) -> Optional[np.ndarray]:
    # PKL dict variants
    if isinstance(blob, dict):
        for key in ("params","rule_params","weights","W"):
            arr = blob.get(key)
            if isinstance(arr, np.ndarray) and arr.ndim==2 and (K is None or arr.shape[1] in (K+1, arr.shape[1])):
                return arr.astype(float, copy=False)
            if isinstance(arr, list) and arr:
                vecs = [np.asarray(a, float).ravel() for a in arr if np.asarray(a).size>=2]
                if not vecs: break
                # выберем длину K+1 если знаем K, иначе модальную
                if K is not None:
                    vecs = [v for v in vecs if v.size==K+1] or vecs
                lengths = [v.size for v in vecs]; modal = max(set(lengths), key=lengths.count)
                vecs = [v for v in vecs if v.size==modal]
                if vecs: return np.stack(vecs, 0)
    # model._params
    arrs=[]
    for p in getattr(model, "_params", []) or []:
        try: arrs.append(np.asarray(p.detach().cpu().numpy(), float).ravel())
        except Exception: arrs.append(np.asarray(p, float).ravel())
    if arrs:
        if K is not None:
            arrs = [v for v in arrs if v.size==K+1] or arrs
        lengths=[v.size for v in arrs]; modal=max(set(lengths), key=lengths.count)
        arrs=[v for v in arrs if v.size==modal]
        if arrs: return np.stack(arrs,0)
    # as last resort, flatten all parameters and split by (K+1) if possible
    try:
        if K is not None:
            vecs=[np.asarray(p.detach().cpu().numpy(),float).ravel() for p in model.parameters()]
            flat=np.concatenate(vecs) if vecs else None
            if flat is not None and flat.size%(K+1)==0:
                R=flat.size//(K+1); return flat.reshape(R,K+1)
    except Exception: pass
    return None

def _rules_meta(blob: Any, model) -> List[Dict[str,Any]]:
    if isinstance(blob, dict) and isinstance(blob.get("rules"), list): return blob["rules"]
    return [{"caption": getattr(r,"caption","<rule>"),
             "_label": (int(getattr(r,"_label")) if hasattr(r,"_label") else None)}
            for r in (getattr(model,"rules",[]) or [])]

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
    i = int(args.idx)%len(X); x = np.asarray(X[i],float); y_true=int(y[i])
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

    # params/rules
    K_hint = _get_K(y, blob, None)
    params = _params_from(blob, clf.model, K_hint)          # [R?,K+1] или None
    K = _get_K(y, blob, params)
    masses = _softmax_rows(params) if isinstance(params, np.ndarray) else None
    rules = _rules_meta(blob, clf.model)
    R = (masses.shape[0] if masses is not None else len(rules))
    fired_all = clf.model._activation_matrix(x.reshape(1,-1)).astype(bool)[0]
    fired = [j for j in np.where(fired_all)[0] if j < R]

    # combine → final_mass/conf
    m = _get_prior_mass(clf.model, blob, K)
    if masses is not None:
        for j in fired:
            if j < masses.shape[0]: m = _combine(m, masses[j])
    # betP
    confidence = m[:K] + m[-1]/float(max(1,K))
    pred = int(np.argmax(confidence))

    # print
    print(f"\n[info] fired rules: {len(fired)} / {R}")
    if fired:
        print("Activated rules:")
        for j in fired:
            cap = rules[j].get("caption","<rule>") if j < len(rules) else "<rule>"
            lbl = rules[j].get("_label", None) if j < len(rules) else None
            mass_j = _f(masses[j]) if (isinstance(masses,np.ndarray) and j<masses.shape[0]) else None
            head = f"(Class {lbl}) " if lbl is not None else ""
            if mass_j is not None:
                print(f" - {head}{cap} -> mass={mass_j}")
            else:
                print(f" - {head}{cap}")

    print(f"\nModel confidence (per class): {_f(confidence)}")
    print(f"True label: {y_true} | Predicted: {pred}")

    # JSON
    out = COMMON/"results"; out.mkdir(parents=True, exist_ok=True)
    path = out / f"inspect_{args.algo.lower()}_{csv.stem}_idx{i}.json"
    payload = {
        "dataset": str(csv), "algo": args.algo, "pkl": str(pkl), "index": i,
        "true_label": y_true, "predicted_class": pred,
        "confidence": _f(confidence), "final_mass": _f(m),
        "prior_mass": _f(_get_prior_mass(clf.model, blob, K)),
        "prior_logits": (_f(blob["prior_logits"]) if isinstance(blob, dict) and blob.get("prior_logits") is not None else None),
        "counts": {"rules_total": int(R), "rules_fired": int(len(fired))},
        "features_display": feats_disp,
        "activated_rules": [{
            "condition": (rules[j].get("caption","<rule>") if j<len(rules) else "<rule>"),
            "class": (int(rules[j]["_label"]) if (j<len(rules) and rules[j].get("_label") is not None) else None),
            "mass": (_f(masses[j]) if (isinstance(masses,np.ndarray) and j<masses.shape[0]) else None),
        } for j in fired],
    }
    with open(path,"w",encoding="utf-8") as f: json.dump(payload,f,ensure_ascii=False,indent=2)
    print(f"\n✓ Saved report → {path}")

if __name__=="__main__":
    main()
