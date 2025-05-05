# DST – Rule–Based Classification with RIPPER / FOIL + Dempster‑Shafer Theory  

<p align="center">
  <img src="Common_code/benchmark_dataset6.png" width="600">
</p>

> • Automatic rule induction (RIPPER / FOIL)  
> • Dempster‑Shafer evidence aggregation & mass optimisation  
> • Post‑pruning that keeps the model tiny without hurting accuracy  

---

## 1. Quick start  

```bash
git clone https://github.com/SargisVardanian/DST.git
cd DST
python test_Ripper_DST.py          # full benchmark pipeline

test_Ripper_DST.py will:

learn rules with RIPPER and FOIL,
convert them to Dempster‑Shafer rules,
optimise masses via gradient descent, optionally prune,
save everything to portable .dsb files and plot metrics.

2. Project layout

core.py                   # helper maths (mass initialisation, commonalities, etc.)
utils.py                  # misc helpers (IO, formatting, …)
DSRule.py                 # light wrapper around a Python predicate + caption
DSRipper.py               # incremental RIPPER/FOIL rule inducer
DSModel.py                # single‑class DST model
DSModelMultiQ.py          # multi‑class DST model
DSClassifier.py           # sklearn‑style wrapper (binary)
DSClassifierMultiQ.py     # sklearn‑style wrapper (multi‑class)
Datasets_loader.py        # six binary toy + real datasets used in benchmarks
test_Ripper_DST.py        # end‑to‑end experiment script + plots


3. Rule induction with DSRipper

3.1 Algorithm in a nutshell
We follow the classical RIPPER loop 
C
o
h
e
n
1995
Cohen1995:

Grow: add literals that maximise information‑gain
<img style="background-color:#f7f7f7; padding:4px 8px; border-radius:4px;" src="https://latex.codecogs.com/svg.image?\mathrm{Gain}(r)=p_{\text{new}}\Bigl(\log_2\frac{p_{\text{new}}}{p_{\text{new}}+n_{\text{new}}}-\log_2\frac{p_{\text{old}}}{p_{\text{old}}+n_{\text{old}}}\Bigr)" alt="RIPPER gain formula" />
Prune the rule with Reduced‑Error pruning on a held‑out portion.
Delete covered positives from the training set and repeat until no positives remain.
DSRipper supports two modes:

algo="ripper" – standard RIPPER gain as above.
algo="foil" – FOIL‑style gain
<img style="background-color:#f7f7f7; padding:4px 8px; border-radius:4px;" src="https://latex.codecogs.com/svg.image?\mathrm{FOILGain}=p\bigl(\log_2t'-\log_2t\bigr)" alt="FOIL gain formula" />

where 
p
p is the number of positives covered, 
t
t the total examples covered.
We run induction incrementally in mini‑batches so that large datasets (e.g. 6 k wine samples) fit in memory.

3.2 Output
For each target class c we get a list of simple dict‑rules, e.g.:

{'free sulfur dioxide': ('<', 36),
 'total sulfur dioxide': ('>=', 110)}
These are converted to DSRule objects:

lam  = lambda x, op='<', thr=36: x[idx] < thr    # compiled predicate
rule = DSRule(ld=lam, caption="Class 1: free SO₂ < 36 & total SO₂ ≥ 110")
At this stage each rule only knows coverage 
∣{x:rule(x)}∣. All other quality metrics come later via DST.

## 4. Dempster–Shafer model

### 4.1 Mass Assignment (MAF)

Each rule \(R_i\) carries a mass vector

<img
  style="background-color:#f7f7f7; padding:4px 8px; border-radius:4px;"
  src="https://latex.codecogs.com/svg.image?m^{(i)}%20%3D%20%28m_1^{(i)}%2C%20\ldots%2C%20m_K^{(i)}%2C%20m_{\mathrm{unc}}^{(i)}%29%2C%20\sum_{j=1}^K%20m_j^{(i)}%20%2B%20m_{\mathrm{unc}}^{(i)}%20%3D%201"
  alt="mass vector" />

We initialise them biased to uncertainty (e.g. \(m_{\mathrm{unc}}=0.8\)) or via a smarter clustering‑based method (see DSGD++).

### 4.2 Combination ⇢ Commonality space

For sample **x**, let \(\mathcal R(x)=\{\,i:R_i(x)=\mathrm{True}\}\). Then:

Convert masses to commonalities

<img
  style="background-color:#f7f7f7; padding:4px 8px; border-radius:4px;"
  src="https://latex.codecogs.com/svg.image?q_k^{(i)}%20%3D%20m_k^{(i)}%20%2B%20m_{\mathrm{unc}}^{(i)}%2C%20%5Cquad%20\forall\,k=1\ldots K"
  alt="commonalities" />

Combine by pointwise product

<img
  style="background-color:#f7f7f7; padding:4px 8px; border-radius:4px;"
  src="https://latex.codecogs.com/svg.image?q_k(x)%20%3D%20\prod_{i\in\mathcal{R}(x)}%20q_k^{(i)}"
  alt="combination" />

Renormalise back to masses

<img
  style="background-color:#f7f7f7; padding:4px 8px; border-radius:4px;"
  src="https://latex.codecogs.com/svg.image?\hat{m}_k(x)%20%3D%20\frac{q_k(x)}{\sum_{\ell=1}^Kq_\ell(x)}%2C%20\quad%20\hat{m}_{\mathrm{unc}}(x)=0"
  alt="normalisation" />

**Prediction** = \(\displaystyle\arg\max_k\hat{m}_k(x)\).

### 4.3 Mass optimisation (DST‑GD)

Treat each \(\mathbf m^{(i)}\) as a learnable parameter tensor and minimise cross‑entropy (or MSE) on the training set via Adam. Constraints (\(m\ge0\), sum to 1) are enforced in `DSModel*.normalize()`.

---

## 5. Rule quality and pruning

After optimisation each rule \(R_i\) has:

- **uncertainty**  
  <img
    style="background-color:#f7f7f7; padding:4px 8px; border-radius:4px;"
    src="https://latex.codecogs.com/svg.image?u_i%20%3D%20m_{\mathrm{unc}}^{(i)}"
    alt="uncertainty" />

- **top‑2 ratio**  
  <img
    style="background-color:#f7f7f7; padding:4px 8px; border-radius:4px;"
    src="https://latex.codecogs.com/svg.image?r_i%20%3D%20\frac{\max_j\,m_j^{(i)}}{\text{2nd-largest }m^{(i)}%20%2B%2010^{-3}}"
    alt="top-2 ratio" />

We min‑max normalise \(r_i\to r'_i\in[0,1]\) and compute a harmonic “usefulness” score:

<img
  style="background-color:#f7f7f7; padding:4px 8px; border-radius:4px;"
  src="https://latex.codecogs.com/svg.image?H_i%20%3D%20\frac{2(1-u_i)\,r'_i}{(1-u_i)+r'_i}"
  alt="usefulness score" />

Re‑compute coverage \(c_i=|\{x:R_i(x)\}|\).  Drop rule \(R_i\) if **all three** hold:

- \(u_i > 0.7\)  
- \(H_i < 0.4\)  
- \(c_i < 10\)  

Wine example: rules ↘ **42 → 33** same F1.

## 6. Benchmarks

| Variant    | Acc.   | F1     | Prec.  | Rec.   |
|------------|--------|--------|--------|--------|
| **R_raw**      | 0.786  | 0.766  | 1.000  | 0.621  |
| **R_DST**      | 0.983  | 0.988  | 1.000  | 0.970  |
| **R_pruned**   | 0.983  | 0.985  | 1.000  | 0.970  |
| **F_raw**      | 0.982  | 0.984  | 0.992  | 0.975  |
| **F_DST**      | 0.989  | 0.990  | 0.992  | 0.989  |
| **F_pruned**   | 0.989  | 0.991  | 0.986  | 0.995  |

> *DST weighting lifts plain RIPPER from mediocre to SOTA.*  
> *Pruning shrinks the model by ≈ 20 % with no loss in performance.*

---

## 7. Citing & further reading

- G. Shafer — *A Mathematical Theory of Evidence* (Princeton, 1976)  
- W. Cohen — *Fast Effective Rule Induction* (ICML, 1995)  
- R. Quinlan — *FOIL: A Midterm Report* (ICML tutorial, 1990)  
- S. Peñafiel et al. — *Applying Dempster–Shafer Theory for Developing a Flexible, Accurate and Interpretable Classifier* (ESWA, 2020)  


