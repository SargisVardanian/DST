# -*- coding: utf-8 -*-
from __future__ import annotations
import os, importlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:
    torch, nn = None, object

_OPCODES = {"==": 0, "!=": 1, "<": 2, "<=": 3, ">": 4, ">=": 5}

def _safe_device(device: Optional[str]) -> str:
    if torch is None: return "cpu"
    d = str(device or "cpu").lower()
    return "cuda" if d.startswith("cuda") and torch.cuda.is_available() else "cpu"

def _lazy_pickle():
    try:
        return importlib.import_module("dill")
    except Exception:
        import pickle as p
        return p

class _RuleEvaluator:
    __slots__ = ("feature_names","name_to_idx","float_tol","value_decoders")
    def __init__(self, feature_names: Optional[Sequence[str]], value_decoders=None, *, float_tol=1e-6):
        self.float_tol = float_tol
        self.value_decoders = value_decoders or {}
        self.set_feature_names(feature_names)
    def set_feature_names(self, feature_names: Optional[Sequence[str]]):
        if feature_names is None:
            self.feature_names, self.name_to_idx = None, {}
        else:
            self.feature_names = list(feature_names)
            self.name_to_idx = {n:i for i,n in enumerate(self.feature_names)}
    def _to_numeric(self, feature_name: str, v: Any) -> float:
        if isinstance(v,(int,float,np.generic)): return float(v)
        if isinstance(v,str):
            dec = self.value_decoders.get(feature_name, {})
            for k,lbl in dec.items():
                if str(lbl)==v: return float(k)
            return float(v)
        return float(v)
    def compile(self, specs: Optional[Sequence[Tuple[str,str,Any]]]):
        if specs is None or not self.feature_names: return None
        idx,ops,vals=[],[],[]
        for name,op,thr in specs:
            if name not in self.name_to_idx: return None
            code=_OPCODES.get(op);
            if code is None: return None
            idx.append(self.name_to_idx[name]); ops.append(code); vals.append(self._to_numeric(name,thr))
        return (np.asarray(idx,np.int32), np.asarray(ops,np.int8), np.asarray(vals,np.float32))
    def evaluate_compiled(self, compiled, X: np.ndarray) -> np.ndarray:
        if compiled is None: return np.zeros(X.shape[0],dtype=bool)
        idx,ops,vals=compiled
        if idx.size==0: return np.ones(X.shape[0],dtype=bool)
        act=np.ones(X.shape[0],dtype=bool); sub=X[:,idx]
        for col,code,thr in zip(sub.T,ops,vals):
            if   code==0: cond=np.isclose(col,thr,atol=self.float_tol)
            elif code==1: cond=~np.isclose(col,thr,atol=self.float_tol)
            elif code==2: cond=col<thr
            elif code==3: cond=col<=thr
            elif code==4: cond=col>thr
            elif code==5: cond=col>=thr
            else: cond=np.zeros_like(col,dtype=bool)
            act &= cond
            if not act.any(): break
        return act
    def evaluate_all(self, compiled_specs, X: np.ndarray) -> np.ndarray:
        if not compiled_specs: return np.zeros((X.shape[0],0),dtype=bool)
        M=np.zeros((X.shape[0],len(compiled_specs)),dtype=bool)
        for j,c in enumerate(compiled_specs): M[:,j]=self.evaluate_compiled(c,X)
        return M

class DSModelMultiQ(nn.Module if torch is not None else object):
    def __init__(self, k:int, algo:str="STATIC", device:str="cpu",
                 feature_names:Optional[Sequence[str]]=None,
                 value_decoders:Optional[Dict[str,Dict[int,Any]]]=None,
                 rule_uncertainty:float=0.4, combination_rule:str="yager"):
        super().__init__()
        self.k=int(k); self.algo=str(algo).upper(); self.device=_safe_device(device)
        self.feature_names=list(feature_names) if feature_names is not None else None
        self.value_decoders=value_decoders or {}
        self.rules: List[Dict[str,Any]]=[]; self._compiled_specs: List[Any]=[]
        self.rule_mass_params: Optional[nn.Parameter]=None  # [N_rules, k+1]
        self._initial_rule_masses: Optional[np.ndarray]=None
        self._unc=float(max(0.0, min(rule_uncertainty, 0.95)))
        self._eq_tol=1e-6; self._min_mass=1e-9
        self._rule_evaluator=_RuleEvaluator(self.feature_names,self.value_decoders,float_tol=self._eq_tol)
        comb=str(combination_rule or "dempster").lower()
        if comb not in {"dempster","yager"}: raise ValueError("combination_rule in {'dempster','yager'}")
        self.combination_rule=comb
        if torch is not None: self.to(torch.device(self.device))

    # ---- formatting helpers ----
    def _literal_to_numeric(self, feature_name: str, value: Any) -> float:
        if isinstance(value, (int, float, np.generic)):
            return float(value)
        if isinstance(value, str):
            decoder = self.value_decoders.get(feature_name, {})
            for key, label in decoder.items():
                try:
                    if str(label) == value:
                        return float(key)
                except Exception:
                    continue
            try:
                return float(value)
            except Exception as exc:
                raise ValueError(f"Cannot convert literal '{value}' for feature '{feature_name}' to float") from exc
        try:
            return float(value)
        except Exception as exc:
            raise ValueError(f"Cannot convert literal '{value}' for feature '{feature_name}' to float") from exc

    def _value_display(self, feature_name: str, value: Any) -> str:
        decoder = self.value_decoders.get(feature_name)
        if decoder:
            try:
                fv = float(value)
            except Exception:
                fv = None
            if fv is not None:
                for key, label in decoder.items():
                    try:
                        if abs(float(key) - fv) < 1e-6:
                            return str(label)
                    except Exception:
                        continue
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        try:
            fv = float(value)
            if abs(fv - round(fv)) < 1e-6:
                return str(int(round(fv)))
            return f"{fv:.6g}"
        except Exception:
            return str(value)

    def _caption_from_specs(self, specs: Sequence[Tuple[str, str, Any]], label: Optional[int]) -> str:
        if not specs:
            return f"Class {label}" if label is not None else ""
        parts = []
        for name, op, val in specs:
            parts.append(f"{name} {op} {self._value_display(name, val)}")
        prefix = f"Class {label}: " if label is not None else ""
        return prefix + " & ".join(parts)

    def _normalize_specs(self, specs: Sequence[Tuple[str, str, Any]]) -> Tuple[Tuple[str, str, float], ...]:
        norm: List[Tuple[str, str, float]] = []
        for name, op, raw in specs:
            fname = str(name)
            fop = str(op)
            numeric = self._literal_to_numeric(fname, raw)
            norm.append((fname, fop, float(numeric)))
        return tuple(norm)

    # ---- rules ----
    def _set_feature_names(self,names:Optional[Sequence[str]]):
        self.feature_names=list(names) if names is not None else None
        self._rule_evaluator.set_feature_names(self.feature_names)
        if self.rules: self._compiled_specs=[self._rule_evaluator.compile(r.get("specs")) for r in self.rules]
    def _clear_rules(self): self.rules.clear(); self._compiled_specs.clear()
    def _add_rule_specs(self,specs:Sequence[Tuple[str,str,Any]],label:Optional[int]=None,
                        caption:str="",stats:Optional[Dict[str,Any]]=None):
        normalized = self._normalize_specs(specs)
        caption_txt = (caption or "").strip()
        if not caption_txt and normalized:
            caption_txt = self._caption_from_specs(normalized, label)
        record = {
            "specs": list(normalized),
            "label": (int(label) if label is not None else None),
            "caption": caption_txt,
            "stats": (dict(stats) if isinstance(stats, dict) else None),
        }
        self.rules.append(record)
        self._compiled_specs.append(self._rule_evaluator.compile(normalized))
    def _ingest_generated_rules(self, generated: Iterable[Dict[str,Any]])->None:
        seen=set()
        for e in generated:
            specs=e.get("specs") or [];
            if not specs: continue
            sig=tuple((str(n),str(o),round(float(v) if isinstance(v,(int,float,np.generic)) else 0.0,6)) for n,o,v in specs)
            if sig in seen: continue
            seen.add(sig)
            self._add_rule_specs(specs,label=e.get("label"),caption=str(e.get("caption","")).strip(),
                                 stats=(e.get("stats") if isinstance(e.get("stats"),dict) else None))

    # ---- rule generation ----
    def generate_raw(self,X:np.ndarray,y:Optional[np.ndarray]=None,feature_names:Optional[Sequence[str]]=None,
                     algo:str="STATIC",**kw)->None:
        from rule_generator import RuleGenerator, generate_static_rule_bundle
        self.algo=str(algo or "STATIC").upper(); self._clear_rules()
        X=np.asarray(X,np.float32)
        names=list(feature_names) if feature_names is not None else [f"X[{i}]" for i in range(X.shape[1])]
        self._set_feature_names(names)
        params=dict(kw); verbose=bool(params.pop("verbose_rules",False))
        if self.algo=="STATIC":
            generated=generate_static_rule_bundle(X,column_names=names,
                breaks=params.pop("breaks",3), value_decoders=self.value_decoders,
                generated_columns=params.pop("generated_mask",None),
                top_k_cats=params.pop("top_k_cats",12),
                include_pairs=params.pop("include_pairs",True),
                pair_top=params.pop("pair_top",4), verbose=verbose)
            self._ingest_generated_rules(generated)
        elif self.algo in {"RIPPER","FOIL"}:
            if y is None: raise ValueError("y required for RIPPER/FOIL")
            gen=RuleGenerator(algo=self.algo.lower(),verbose=verbose,**params); gen.fit(X,np.asarray(y).astype(int),feature_names=names)
            metrics=getattr(gen,"_rule_metrics",{}) if isinstance(getattr(gen,"_rule_metrics",None),dict) else {}
            seen=set()
            for label,cond in gen._ordered_rules:
                if not cond: continue
                specs=tuple((n,op,float(v)) for n,(op,v) in cond.items())
                if specs in seen: continue
                seen.add(specs)
                key="|".join(f"{n}{op}{round(v,6)}" for n,op,v in specs)
                self._add_rule_specs(specs,label=int(label),caption="",stats=metrics.get(key))
        else:
            raise ValueError(f"Unsupported generator '{algo}'")
        self._init_mass_params(reset=True)

    # ---- mass init / project ----
    def _init_mass_params(self, reset: bool=True)->None:
        if torch is None: return
        device=torch.device(self.device); n=len(self.rules); k=self.k; dtype=torch.float32
        def _new_table(n:int)->torch.Tensor:
            m=torch.zeros((n,k+1),dtype=dtype,device=device)
            for i in range(n):
                m[i,-1]=self._unc
                if k>0:
                    r=torch.rand(k,device=device,dtype=dtype); r=r/(r.sum().clamp_min(1e-9))
                    m[i,:k]=(1.0-self._unc)*r
                m[i]/=m[i].sum()
            return m
        if reset or (self.rule_mass_params is None):
            self.rule_mass_params=nn.Parameter(_new_table(n),requires_grad=True)
        else:
            cur=self.rule_mass_params.detach().clone()
            new=_new_table(n); keep=min(cur.shape[0],new.shape[0])
            if keep>0: new[:keep]=cur[:keep].to(device)
            self.rule_mass_params=nn.Parameter(new,requires_grad=True)
        self._initial_rule_masses=self.rule_mass_params.detach().cpu().numpy()
        self.project_masses()

    def project_masses(self)->None:
        if torch is None: return
        with torch.no_grad():
            if self.rule_mass_params is not None:
                p=self.rule_mass_params
                p.clamp_(min=self._min_mass)
                p/=p.sum(dim=-1,keepdim=True).clamp_min(self._min_mass)

    # ---- activations ----
    def _activation_matrix_np(self, X: np.ndarray)->np.ndarray:
        return self._rule_evaluator.evaluate_all(self._compiled_specs,np.asarray(X,np.float32)).astype(np.float32,copy=False)

    # ---- DS combine (torch) ----
    def _combine_all_torch(self, A: torch.Tensor, M: torch.Tensor)->torch.Tensor:
        k=self.k; dtype=M.dtype; device=M.device
        if k==0: return torch.ones((A.shape[0],1),dtype=dtype,device=device)
        eps=torch.tensor(max(self._min_mass, torch.finfo(dtype).tiny),dtype=dtype,device=device)
        q_cls=(M[:,:k]+M[:,-1:]).clamp_min(float(eps)); q_omg=M[:,-1:].clamp_min(float(eps))
        log_q_cls=q_cls.log(); log_q_omg=q_omg.log()
        W=A.to(dtype=dtype)                  # [B,N]
        log_Q_cls=W@log_q_cls                # [B,k]
        log_Q_omg=W@log_q_omg                # [B,1]
        Q_cls=log_Q_cls.exp(); Q_omg=log_Q_omg.exp()
        raw_cls=(Q_cls-Q_omg).clamp_min(0.0); raw_omg=Q_omg
        total=raw_cls.sum(1,keepdim=True)+raw_omg
        if self.combination_rule=="yager":
            leftover=(1.0-total).clamp_min(0.0)
            out=torch.cat([raw_cls, (raw_omg+leftover).clamp_min(0.0)],dim=1)
        else:
            out=torch.cat([raw_cls/total.clamp_min(float(eps)), raw_omg/total.clamp_min(float(eps))],dim=1)
        out/=out.sum(1,keepdim=True).clamp_min(float(eps))
        return out

    # ---- public ----
    def forward(self, X):  # torch-only path
        if torch is None: raise RuntimeError("PyTorch required")
        if self.rule_mass_params is None: self._init_mass_params(reset=True)
        device=self.rule_mass_params.device
        Xnp=np.asarray(X.detach().cpu().numpy() if isinstance(X,torch.Tensor) else X,np.float32)
        A_np=self._activation_matrix_np(Xnp)              # [B,N_rules] in {0,1}
        if A_np.shape[1]==0:  # нет правил: вернуть вакуум
            B=Xnp.shape[0]; out=torch.zeros((B,self.k+1),device=device); out[:,-1]=1.0; return out
        A=torch.from_numpy(A_np).to(device=device,dtype=torch.float32)
        M=self.rule_mass_params                            # [N,k+1]
        return self._combine_all_torch(A,M)

    def predict_mass(self, X)->np.ndarray:
        if torch is not None:
            self.eval()
            with torch.inference_mode():
                Xt=torch.from_numpy(np.asarray(X,np.float32)) if not isinstance(X,torch.Tensor) else X
                out=self.forward(Xt.to(device=self.rule_mass_params.device if self.rule_mass_params is not None else "cpu",
                                       dtype=torch.float32))
            return out.detach().cpu().numpy()
        # без torch — деградация в равномерную пигристику
        Xnp=np.asarray(X,np.float32); B=Xnp.shape[0]; out=np.zeros((B,self.k+1),np.float32); out[:,-1]=1.0; return out

    def predict_mass_initial(self, X)->np.ndarray:
        if torch is None or self._initial_rule_masses is None:
            return self.predict_mass(X)
        device=torch.device(self.device)
        M=torch.from_numpy(self._initial_rule_masses).to(device=device,dtype=torch.float32)
        Xnp=np.asarray(X,np.float32); A_np=self._activation_matrix_np(Xnp)
        if A_np.shape[1]==0:
            B=Xnp.shape[0]; out=torch.zeros((B,self.k+1),device=device); out[:,-1]=1.0; return out.detach().cpu().numpy()
        A=torch.from_numpy(A_np).to(device=device,dtype=torch.float32)
        with torch.inference_mode():
            out=self._combine_all_torch(A,M)
        return out.detach().cpu().numpy()

    def predict_by_rule_labels(self, X)->np.ndarray:
        A=self._activation_matrix_np(np.asarray(X,np.float32)).astype(bool,copy=False)
        labels=np.array([r.get("label",-1) for r in self.rules],dtype=int)
        preds=np.zeros(A.shape[0],dtype=int)
        for i,fire in enumerate(A):
            v=labels[fire]; v=v[v>=0]
            if v.size:
                vals,cnts=np.unique(v,return_counts=True); preds[i]=int(vals[np.argmax(cnts)])
        return preds

    # ---- export/persist ----
    def get_activated_rules(self, sample: Sequence[float])->Dict[str,Any]:
        v=np.asarray(sample,np.float32); a=self._activation_matrix_np(v.reshape(1,-1))[0]; fired=np.flatnonzero(a>0.5)
        items=[]
        rm=self.rule_mass_params.detach().cpu().numpy() if (torch is not None and self.rule_mass_params is not None) else None
        for rid in fired:
            mv = rm[rid].tolist() if (rm is not None and rid< len(rm)) else None
            r = self.rules[rid]
            items.append({"id":int(rid),"condition":r.get("caption",f"rule#{rid}"),"class":r.get("label",None),
                          "activation":float(a[rid]),"mass":mv,"initial_mass":None,"stats":(r.get("stats") if isinstance(r.get("stats"),dict) else None)})
        mass=self.predict_mass(v.reshape(1,-1))[0]; pred=int(np.argmax(mass[:self.k])) if self.k else None
        if not items: items.append({"id":-1,"condition":"<no_rule_fired>","class":pred,"activation":1.0,"mass":None,"initial_mass":None,"stats":None})
        return {"activated_rules":items,"mass_vector":mass.tolist(),"predicted_class":pred}

    def prepare_rules_for_export(self, sample:Optional[Sequence[float]]=None)->Dict[str,Any]:
        if sample is None:
            return {"algo":self.algo,"k":self.k,"rules":[{"id":i,"condition":r.get("caption",f"rule#{i}"),
                    "class":r.get("label",None),"initial_mass":None,"stats":(r.get("stats") if isinstance(r.get("stats"),dict) else None)}
                    for i,r in enumerate(self.rules)]}
        return self.get_activated_rules(sample)

    def save_rules_bin(self, path:str)->None:
        os.makedirs(os.path.dirname(str(path)),exist_ok=True)
        payload={"k":self.k,"algo":self.algo,"feature_names":self.feature_names,"value_decoders":self.value_decoders,
                 "rules":[{"caption":r.get("caption",""),"specs":r.get("specs"),"label":r.get("label"),"stats":(r.get("stats") if isinstance(r.get("stats"),dict) else None)} for r in self.rules],
                 "rule_mass_params": (self.rule_mass_params.detach().cpu().numpy() if (torch is not None and self.rule_mass_params is not None) else None),
                 "initial_rule_masses": (self._initial_rule_masses.tolist() if self._initial_rule_masses is not None else None)}
        pkl=_lazy_pickle()
        with open(path,"wb") as f: pkl.dump(payload,f)

    def load_rules_bin(self, path:str)->None:
        pkl=_lazy_pickle()
        with open(path,"rb") as f: payload=pkl.load(f)
        self.k=int(payload.get("k",self.k)); self.algo=str(payload.get("algo",self.algo)).upper()
        self._set_feature_names(payload.get("feature_names")); self.value_decoders=payload.get("value_decoders",{})
        self._clear_rules()
        for rec in payload.get("rules",[]):
            self._add_rule_specs(rec.get("specs") or [],label=rec.get("label"),caption=rec.get("caption",""),
                                 stats=(rec.get("stats") if isinstance(rec.get("stats"),dict) else None))
        self._initial_rule_masses = (np.asarray(payload.get("initial_rule_masses"),np.float32) if payload.get("initial_rule_masses") is not None else None)
        if torch is not None:
            rp=payload.get("rule_mass_params")
            if rp is not None:
                t=torch.from_numpy(np.asarray(rp)).to(device=torch.device(self.device),dtype=torch.float32)
                self.rule_mass_params=nn.Parameter(t,requires_grad=True)
            else:
                self.rule_mass_params=None
        self._init_mass_params(reset=(self.rule_mass_params is None))
    def save_rules_dsb(self,path:str)->None:
        os.makedirs(os.path.dirname(str(path)),exist_ok=True)
        rm = self.rule_mass_params.detach().cpu().numpy() if (torch is not None and self.rule_mass_params is not None) else None
        lines=[]
        for i,r in enumerate(self.rules):
            cap=r.get("caption",f"rule#{i}"); lbl=r.get("label",None)
            mv = (rm[i] if (rm is not None and i<len(rm)) else None)
            if mv is not None:
                cls=", ".join(f"{v:.4f}" for v in mv[:self.k]); lines.append(f"{'Class '+str(lbl)+': ' if lbl is not None else ''}{cap} || mass=[{cls}, unc={mv[self.k]:.4f}]")
            else:
                lines.append(f"{'Class '+str(lbl)+': ' if lbl is not None else ''}{cap}")
        with open(path,"w",encoding="utf-8") as f: f.write("\n".join(lines))
