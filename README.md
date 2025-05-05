````markdown
# DST – Rule‑Based Classification with RIPPER / FOIL + Dempster‑Shafer Theory  

<p align="center">
  <img src="Common_code/benchmark_dataset6.png" width="650">
</p>

> • **Automatic rule induction** (RIPPER / FOIL)  
> • **Dempster‑Shafer** evidence aggregation & mass optimisation  
> • **Post‑pruning** that keeps the model tiny without hurting accuracy  

---

## 1. Quick start  

```bash
git clone https://github.com/SargisVardanian/DST.git
cd DST
python test_Ripper_DST.py        # full benchmark pipeline
````

`test_Ripper_DST.py` will

1. learn rules with **RIPPER** and **FOIL**;
2. convert them to Dempster‑Shafer rules;
3. optimise masses via gradient descent, optionally prune;
4. save everything to portable `.dsb` files and plot the metrics you saw above.

---

\## 2. Project layout

```
core.py                   # helper maths (mass init, commonality, …)
utils.py                  # misc helpers (IO, formatting, …)
DSRule.py                 # tiny wrapper: predicate + caption
DSRipper.py               # incremental RIPPER/FOIL inducer
DSModel.py                # single‑class DST model
DSModelMultiQ.py          # multi‑class DST model
DSClassifier.py           # sklearn‑style wrapper (binary)
DSClassifierMultiQ.py     # sklearn‑style wrapper (multi‑class)
Datasets_loader.py        # six toy + real binary datasets
test_Ripper_DST.py        # end‑to‑end experiment script
```

---

\## 3. Rule induction with **DSRipper**

\### 3.1 Algorithm in a nutshell

We follow the classical *RIPPER* loop (Cohen 1995):

1. **Grow** – add literals that maximise information‑gain

   <img style="background-color:#f7f7f7; padding:4px 8px; border-radius:4px;"
     src="https://latex.codecogs.com/svg.image?\mathrm{Gain}(r)%20=%20p_{\text{new}}\!\left(\log_2\frac{p_{\text{new}}}{p_{\text{new}}+n_{\text{new}}}-\log_2\frac{p_{\text{old}}}{p_{\text{old}}+n_{\text{old}}}\right)" />

2. **Prune** the rule with reduced‑error pruning on a held‑out slice.

3. **Delete** covered positives; repeat until no positives remain.

`DSRipper` supports two modes

| `algo`      | Gain formula                                                                                                                                                     |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `"ripper"`  | classic gain above                                                                                                                                               |
| `"foil"`    | <img style="background-color:#f7f7f7; padding:4px 8px; border-radius:4px;" src="https://latex.codecogs.com/svg.image?\mathrm{FOILGain}=p(\log_2 t'-\log_2 t)" /> |

We run induction **incrementally in mini‑batches** – lets 6 k‑row Wine dataset fit into RAM.

\### 3.2 Output

For each class *𝑐* we get a list of dict‑rules, e.g.

```python
{'free sulfur dioxide': ('<', 36),
 'total sulfur dioxide': ('>=', 110)}
```

Then we build `DSRule` objects:

```python
lam  = lambda x, thr=36: x[idx] < thr          # compiled predicate
rule = DSRule(ld=lam,
              caption="Class 1: free SO₂ < 36 & total SO₂ ≥ 110")
```

At this point rule only knows **coverage** |{x : rule(x)}|; quality metrics come later from DST.

---

\## 4. Dempster–Shafer model

\### 4.1 Mass assignment (MAF)

Each rule *R*<sub>i</sub> carries a mass vector

<img style="background-color:#f7f7f7; padding:4px 8px; border-radius:4px;"
  src="https://latex.codecogs.com/svg.image?\mathbf{m}^{(i)}=(m_1^{(i)},\dots,m_K^{(i)},\,m_{\mathrm{unc}}^{(i)}),\quad\sum_{j=1}^K m_j^{(i)}+m_{\mathrm{unc}}^{(i)}=1" />

We initialise them **biased to uncertainty** (e.g. *m*<sub>unc</sub>=0.8) or by a smarter clustering‑based scheme (DSGD++).

\### 4.2 Combination → commonality space

For a sample **x** denote $\mathcal R(x)=\{i:R_i(x)=\text{True}\}$.

1. **Mass → commonality**

   <img style="background-color:#f7f7f7; padding:4px 8px; border-radius:4px;"
     src="https://latex.codecogs.com/svg.image?q_k^{(i)}=m_k^{(i)}+m_{\mathrm{unc}}^{(i)},\quad k=1..K" />

2. **Combine** by point‑wise product

   <img style="background-color:#f7f7f7; padding:4px 8px; border-radius:4px;"
     src="https://latex.codecogs.com/svg.image?q_k(x)=\prod_{i\in\mathcal{R}(x)}q_k^{(i)}" />

3. **Back to masses**

   <img style="background-color:#f7f7f7; padding:4px 8px; border-radius:4px;"
     src="https://latex.codecogs.com/svg.image?\hat{m}_k(x)=\frac{q_k(x)}{\sum_{\ell=1}^K q_\ell(x)},\quad\hat{m}_{\mathrm{unc}}(x)=0" />

Prediction = $\arg\max_k\hat m_k(x)$.

\### 4.3 Mass optimisation (DST‑GD)

Treat every $\mathbf m^{(i)}$ as a learnable tensor; minimise cross‑entropy (or MSE) with Adam.
`DSModel*.normalize()` enforces $m\ge0$ and $\sum=1$.

---

\## 5. Rule quality & pruning

After optimisation each rule *R*<sub>i</sub> has

* uncertainty   <img style="background-color:#f7f7f7; padding:4px 8px; border-radius:4px;" src="https://latex.codecogs.com/svg.image?u_i=m_{\mathrm{unc}}^{(i)}" />
* top‑2 ratio   <img style="background-color:#f7f7f7; padding:4px 8px; border-radius:4px;" src="https://latex.codecogs.com/svg.image?r_i=\frac{\max_j m_j^{(i)}}{\text{2nd‑largest }m^{(i)}+10^{-3}}" />

Scale $r_i\rightarrow r'_i\in[0,1]$; usefulness

<img style="background-color:#f7f7f7; padding:4px 8px; border-radius:4px;"
  src="https://latex.codecogs.com/svg.image?H_i=\frac{2(1-u_i)r'_i}{(1-u_i)+r'_i}" />

Coverage $c_i=|\{x:R_i(x)\}|$. We **drop** a rule when

* $u_i>0.7$ **and**
* $H_i<0.4$ **and**
* $c_i<10$

Wine‑quality dataset: rules 42 → 33, identical F1.

---

\## 6. Benchmarks

| Variant       |  Acc. |    F1 | Prec. |  Rec. |
| ------------- | ----: | ----: | ----: | ----: |
| **R\_raw**    | 0.786 | 0.766 | 1.000 | 0.621 |
| **R\_DST**    | 0.983 | 0.988 | 1.000 | 0.970 |
| **R\_pruned** | 0.983 | 0.985 | 1.000 | 0.970 |
| **F\_raw**    | 0.982 | 0.984 | 0.992 | 0.975 |
| **F\_DST**    | 0.989 | 0.990 | 0.992 | 0.989 |
| **F\_pruned** | 0.989 | 0.991 | 0.986 | 0.995 |

> **DST weighting** lifts plain RIPPER from mediocre to SOTA.
> **Pruning** cuts model size by ≈ 20 % with *no* accuracy loss.

---

\## 7. Citing & further reading

* G. Shafer — *A Mathematical Theory of Evidence* (1976)
* W. Cohen — *Fast Effective Rule Induction* (ICML 1995)
* R. Quinlan — *FOIL: A Midterm Report* (1990)
* S. Peñafiel et al. — *Applying DST for an Interpretable Classifier* (ESWA 2020)

