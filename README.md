Thought for a few seconds


Below is a **drop‑in README.md**.
Simply copy‑paste the whole block (including the opening/closing \`\`\`markdown) into `README.md`, commit & push – GitHub will render the maths as SVG images via LaTeX Codecogs.


# DST – Rule‑Based Classification with RIPPER / FOIL + Dempster‑Shafer Theory

<p align="center">
  <img src="Common_code/benchmark_dataset6.png" width="600">
</p>

> • Automatic rule induction (RIPPER / FOIL)  
> • Dempster‑Shafer evidence aggregation & mass optimisation  
> • Post‑pruning that keeps the model tiny without hurting accuracy  

---

## 1 · Quick start

git clone https://github.com/SargisVardanian/DST.git
cd DST
python test_Ripper_DST.py          # full benchmark pipeline

`test_Ripper_DST.py` will

1. learn rules with **RIPPER** and **FOIL**;
2. convert them to Dempster‑Shafer rules;
3. optimise masses via gradient descent, optionally prune;
4. save everything as portable `.dsb` files and dump the plots / CSV metrics.

---

## 2 · Project layout

core.py                   # helper maths (mass initialisation, commonalities, …)
utils.py                  # misc helpers (I/O, plotting, …)
DSRule.py                 # light wrapper around a Python predicate + caption
DSRipper.py               # incremental RIPPER/FOIL rule inducer
DSModel.py                # single‑class  DST model
DSModelMultiQ.py          # multi‑class   DST model
DSClassifier.py           # sklearn‑style wrapper (binary)
DSClassifierMultiQ.py     # sklearn‑style wrapper (multi‑class)
Datasets_loader.py        # 6 toy+real binary datasets used in benchmarks
test_Ripper_DST.py        # end‑to‑end experiment / plotting script


---

## 3 · Rule induction with **DSRipper**

\### 3.1 Algorithm in a nutshell

Classical **RIPPER** loop \[Cohen 1995]:

1. **Grow**: add a literal that maximises information‑gain

   <img src="https://latex.codecogs.com/svg.image?\mathrm{Gain}(r)=p_{\text{new}}\Bigl(\log_2\frac{p_{\text{new}}}{p_{\text{new}}+n_{\text{new}}}-\log_2\frac{p_{\text{old}}}{p_{\text{old}}+n_{\text{old}}}\Bigr)" style="background:#f7f7f7;padding:4px 8px;border-radius:4px;" alt="RIPPER gain"/>

2. **Prune** the rule by reduced‑error pruning on a held‑out split.

3. **Delete** positives covered by the rule & repeat.

`DSRipper` supports two modes:

| `algo` value | Gain expression                                                                                                                                                                 |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `"ripper"`   | same as above                                                                                                                                                                   |
| `"foil"`     | <img  style="background-color:#f7f7f7; padding:4px 8px; border-radius:4px;"  src="https://latex.codecogs.com/svg.image?\mathrm{FOILGain}=p\bigl(\log_2t'-\log_2t\bigr)" style="background:#f7f7f7;padding:4px 8px;border-radius:4px;" alt="FOIL gain"/> |

where *p* = #positive samples covered, *t* = total covered.

The implementation processes the dataset **incrementally in mini‑batches**, so \~6 k wine rows fit into laptop RAM.

\### 3.2 Output

For each class *c* we get a Python `dict` description, e.g.


{'free sulfur dioxide': ('<', 36),
 'total sulfur dioxide': ('>=', 110)}

Converted to a `DSRule`:


lam  = lambda x, op='<', thr=36: x[idx] < thr
rule = DSRule(ld=lam,
              caption="Class 1: free SO₂ < 36 & total SO₂ ≥ 110")

At this stage a rule knows only **coverage** |{x : rule(x)}| – all other quality metrics are learned later by DST.

---

## 4 · Dempster‑Shafer model

\### 4.1 Mass Assignment (MAF)

Each rule $R_i$ carries a mass vector

<img
    style="background-color:#f7f7f7; padding:4px 8px; border-radius:4px;"
    src="https://latex.codecogs.com/svg.image?m^{(i)}=(m_1^{(i)},\dots,m_K^{(i)},m_{\mathrm{unc}}^{(i)}),\;\sum_{j=1}^Km_j^{(i)}+m_{\mathrm{unc}}^{(i)}=1" style="background:#f7f7f7;padding:4px 8px;border-radius:4px;" alt="mass vector"/>

Initialisation: either “uniform + 0.8 uncertainty” or smarter clustering (see **DSGD++**).

\### 4.2 Combination → Commonality space

For sample **x** let $\mathcal R(x)=\{i:R_i(x)\}$.

| Step              | Formula                                                                                                                                                                                            |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Commonalities** | <img src="https://latex.codecogs.com/svg.image?q_k^{(i)}=m_k^{(i)}+m_{\mathrm{unc}}^{(i)},\;\forall k" style="background:#f7f7f7;padding:4px 8px;border-radius:4px;"/>  |                                   |
| **Product rule**  | <img src="https://latex.codecogs.com/svg.image?q_k(x)=\prod_{i\in\mathcal{R}(x)}q_k^{(i)}" style="background:#f7f7f7;padding:4px 8px;border-radius:4px;"/>                                         |
| **Normalise**     | <img src="https://latex.codecogs.com/svg.image?\hat{m}_k(x)=\frac{q_k(x)}{\sum_{\ell=1}^Kq_\ell(x)},\;\hat{m}_{\mathrm{unc}}(x)=0" style="background:#f7f7f7;padding:4px 8px;border-radius:4px;"/> |

Prediction = $\arg\max_k\hat{m}_k(x)$.

\### 4.3 Mass optimisation (DST‑GD)

Treat every $\mathbf m^{(i)}$ as a learnable tensor and minimise **cross‑entropy** (or MSE) with Adam.
`DSModel*.normalize()` enforces $m\ge0$ and ∑=1.

---

## 5 · Rule quality & pruning

After optimisation each rule $R_i$ has:

| Metric      | Expression                                                                                                                                                                        |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Uncertainty | <img src="https://latex.codecogs.com/svg.image?u_i=m_{\mathrm{unc}}^{(i)}" style="background:#f7f7f7;padding:4px 8px;border-radius:4px;"/>                                        |
| Top‑2 ratio | <img src="https://latex.codecogs.com/svg.image?r_i=\dfrac{\max_j\,m_j^{(i)}}{\text{2nd-largest }m^{(i)}+10^{-3}}" style="background:#f7f7f7;padding:4px 8px;border-radius:4px;"/> |

Scale $r_i\to r'_i\in[0,1]$ (min‑max) and compute usefulness

<img src="https://latex.codecogs.com/svg.image?H_i=\frac{2(1-u_i)\,r'_i}{(1-u_i)+r'_i}" style="background:#f7f7f7;padding:4px 8px;border-radius:4px;"/>

Coverage $c_i=|\{x:R_i(x)\}|$.

**Drop** $R_i$ if simultaneously

* $u_i>0.7$
* $H_i<0.4$
* $c_i<10$

*Wine* dataset → rules **42 → 33** with identical F1.

---

## 6 · Benchmarks

| Variant       | Acc.  | F1    | Prec. | Rec.  |
| ------------- | ----- | ----- | ----- | ----- |
| **R\_raw**    | 0.786 | 0.766 | 1.000 | 0.621 |
| **R\_DST**    | 0.983 | 0.988 | 1.000 | 0.970 |
| **R\_pruned** | 0.983 | 0.985 | 1.000 | 0.970 |
| **F\_raw**    | 0.982 | 0.984 | 0.992 | 0.975 |
| **F\_DST**    | 0.989 | 0.990 | 0.992 | 0.989 |
| **F\_pruned** | 0.989 | 0.991 | 0.986 | 0.995 |

> **DST weighting** lifts plain RIPPER from mediocre to *SOTA*.
> **Pruning** shrinks the model by ≈ 20 % with zero loss in performance.

---

## 7 · Citing & further reading

* G. Shafer — *A Mathematical Theory of Evidence* (Princeton U Press, 1976)
* W. Cohen — *Fast Effective Rule Induction* (ICML 1995)
* R. Quinlan — *FOIL: A Midterm Report* (ICML tutorial 1990)
* S. Peñafiel et al. — *Applying Dempster‑Shafer Theory for Developing a Flexible, Accurate and Interpretable Classifier* (ESWA 2020)

---

## 8 · Licence

