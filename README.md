# DST: Interpretable Rule-Based Classification via Dempster-Shafer Theory

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-green" alt="Python">
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange" alt="PyTorch">
</p>

## Abstract

This repository implements an interpretable classification framework that combines **symbolic rule-based models** with **Dempster-Shafer Theory (DST)** and gradient-based optimization.

Classic rule learners such as **RIPPER** and **FOIL** produce human-readable IF–THEN rules, but they are rigid and lack calibrated uncertainty. On the other hand, modern probabilistic models are accurate but often opaque.

Our framework bridges this gap:

- Logical rules are treated as **sources of evidence**
- Each rule has a learnable **mass function** over classes + uncertainty
- Evidence from multiple rules is aggregated via **Dempster's rule**
- Masses are optimized by **Dempster-Shafer Gradient Descent (DSGD)**

We support three rule generation strategies — **RIPPER**, **FOIL**, and **STATIC** — and show how DST turns them into robust, differentiable classifiers with explicit epistemic uncertainty.

A **representation-based mass initialization** is provided as an option: rule masses can be initialized using **class centroids, representativeness of samples, and rule purity**, which accelerates convergence and reduces initial uncertainty compared to purely random initialization.

---

## 1. Theoretical Framework

### 1.1 Dempster-Shafer Theory of Evidence

For a classification task with classes $\Theta = \{C_1, C_2, \dots, C_K\}$, DST defines a **mass assignment function** $m$ over subsets of $\Theta$:

$$\sum_{A \subseteq \Theta} m(A) = 1, \qquad m(\emptyset) = 0$$

Unlike Bayesian probability, DST allows mass on **non-singleton** sets:

- Mass on $\{C_1\}$ is belief specifically in class $C_1$
- Mass on $\Theta$ (denoted $\Omega$) models **pure ignorance**

In this project we follow a common simplification — we only assign masses to **singletons** and the **universal set** $\Omega$:

$$\mathbf{m} = [m_1, m_2, \dots, m_K, m_\Omega]$$

where $m_k = m(\{C_k\})$ and $m_\Omega = m(\Theta)$. For each rule $r_i$ we maintain such a vector $\mathbf{m}^{(i)}$ — these are the parameters the model learns.

### 1.2 Dempster's Rule of Combination

Given two rules with mass functions $m^{(1)}$ and $m^{(2)}$, Dempster's rule combines them:

$$(m^{(1)} \oplus m^{(2)})(A) = \frac{1}{1 - K} \sum_{B \cap C = A} m^{(1)}(B) \, m^{(2)}(C)$$

where the **conflict** $K$ is:

$$K = \sum_{B \cap C = \emptyset} m^{(1)}(B) \, m^{(2)}(C)$$

In our singletons+Ω setting:

- Mass on each class $C_k$ is proportional to both rules supporting $C_k$, or one rule supporting $C_k$ while the other is ignorant (Ω)
- Mass on Ω is proportional to both rules being ignorant
- Conflict $K$ arises when rules support **different classes simultaneously**; normalization by $1 - K$ redistributes this conflict

The code for batched pairwise combination is in `core.py` (`ds_combine_pair`), with a vectorized wrapper `_combine_active_masses` in `DSModelMultiQ`.

### 1.3 From Masses to Probabilities

We convert masses $\mathbf{m}$ to probabilities $p$ using:

1. **Pignistic transform** (uniformly spreading $m_\Omega$ across classes):
   $$p_k = m_k + \frac{m_\Omega}{K}$$

2. **Class-prior-based redistribution** (optional): distribute ignorance proportionally to class priors $\pi$:
   $$p_k = m_k + m_\Omega \cdot \frac{\pi_k}{\sum_j \pi_j}$$

This is implemented in `DSModelMultiQ._mass_to_prob` and `core.masses_to_pignistic`.

---

## 2. Model Architecture

The end-to-end classifier has three layers:

1. **Rule Generation** (`rule_generator.py`) — RIPPER / FOIL / STATIC learn symbolic rules from `(X, y)`
2. **DST Model** (`DSModelMultiQ.py`) — stores rules, evaluates activations, maintains per-rule mass vectors, combines masses via Dempster's rule, outputs class probabilities
3. **Training Loop** (`DSClassifierMultiQ.py`) — splits data, calls rule generator, initializes masses, trains via projected gradient descent

### 2.1 Rule Generation Strategies

All strategies operate on a shared representation:

- A **literal** is a simple condition `(feature, operator, threshold/value)`: numeric (`Vj < t`, `Vj > t`) or categorical (`Vj == value`)
- A **rule** is a conjunction of literals: IF $\ell_1 \land \ell_2 \land \dots \land \ell_L$ THEN class $c$

#### 2.1.1 FOIL (First Order Inductive Learner)

FOIL is a classic **top-down** rule learner (Quinlan, 1990):

1. Start with an empty rule (covers everything)
2. Iteratively add literals to maximize **FOIL-Gain**:
   $$Gain = t \cdot \left(\log_2\frac{p_1}{p_1 + n_1} - \log_2\frac{p_0}{p_0 + n_0}\right)$$
   where $p_0, n_0$ are positives/negatives before, $p_1, n_1$ after, and $t$ is covered positives
3. Stop when rule covers zero negatives
4. Remove covered positives; repeat

**Characteristics:**
- ✅ Generates **specific** rules with high precision
- ✅ Good at catching "edge cases"
- ❌ No pruning phase → prone to overfitting

#### 2.1.2 RIPPER (Repeated Incremental Pruning)

RIPPER (Cohen, 1995) adds critical **pruning**:

1. **Grow Phase:** On Grow Set, add literals with FOIL-gain until rule is pure
2. **Prune Phase:** On Prune Set, remove literals from end if it improves:
   $$Score = \frac{p - n}{p + n}$$
3. **Optimization Phase:** Rewrite/replace rules using MDL criterion

**Our Extension: Tiered Precision RIPPER**

```python
precision_tiers = [
    {"min_precision": 0.80, "max_literals": 8},
    {"min_precision": 0.60, "max_literals": 6},
    {"min_precision": 0.40, "max_literals": 5},
    {"min_precision": 0.20, "max_literals": 4},
]
```

- First grow very precise (but narrow) rules
- Then allow less precise rules with limited length
- For DST: high-precision rules provide strong evidence; lower-precision rules add weak but useful signals

#### 2.1.3 STATIC (Exhaustive Heuristic Search)

STATIC casts a "wide net" by combining simple candidates:

1. **Single literals:** Quantile thresholds for numeric, frequent values for categorical
2. **Pairs and triples:** Combine best literals into 2-3 condition rules
3. **Diversity filter:** Remove highly overlapping rules (Jaccard similarity)

**Result:** Hundreds (sometimes 1000+) rules including feature interactions. Noisy rules are naturally suppressed by DST — they learn high $m_\Omega \approx 1$ and contribute little.

#### 2.1.4 Strategy Comparison

| Aspect | RIPPER | FOIL | STATIC |
|:-------|:-------|:-----|:-------|
| **Strategy** | Grow-Prune-Optimize | Greedy FOIL-gain | Exhaustive heuristic |
| **Rule Count** | Low (20-150) | Medium (100-400) | High (400-1200) |
| **Overfitting** | Resistant | Prone | N/A |
| **Best For** | Clean, high-signal | Balanced data | Noisy, interaction-heavy |

### 2.2 DST Layer (DSModelMultiQ)

`DSModelMultiQ.py` implements:

1. **Feature encoding:** Numeric as-is, categorical via internal encoders
2. **Literal evaluation:** `_eval_literals(X)` → boolean masks for all literals
3. **Rule activation:** `_activation_matrix(X)` → matrix $A[n,r] = 1$ if rule $r$ fires on sample $n$
4. **Mass parameterization:** `rule_mass_params` matrix $[N_{rules}, K+1]$, projected to simplex via `params_to_mass`
5. **Combination:** `_combine_active_masses` applies Dempster's rule to active rules only
6. **Probabilities:** `_mass_to_prob` applies pignistic transform or class-prior redistribution

### 2.3 Mass Initialization: Random vs Representation-Based

**Random initialization** (`reset_masses`):
- Random masses on classes, normalized
- Fixed base uncertainty (e.g., 0.8) goes to Ω

**Representation-based initialization** (`init_masses_dsgdpp`):
1. Build class centroids from training data
2. Compute representativeness for each sample:
   $$R(x) = \frac{1}{1 + \|x - \mu_y\|}$$
3. For each rule:
   - Find covered samples
   - Calculate purity (majority class fraction)
   - Average representativeness of covered samples
   - Confidence: $Conf(r) = purity \times \mathbb{E}[R(x) \mid x \in D_r]$
   - Assign: $m(\text{majority class}) = Conf$, $m(\Omega) = 1 - Conf$

---

## 3. Training Loop (DSClassifierMultiQ)

### 3.1 Main Steps in `fit(X, y, ...)`

1. **Data preparation:** Convert X, y to numpy; set feature names and decoders
2. **Rule generation:** If `model.rules` empty, call `model.generate_rules(X, y, algo=...)`
3. **Train/Val split:** Stratified random split with `val_split`
4. **Mass initialization:** Try `init_masses_dsgdpp`, fallback to `reset_masses`
5. **Class weights:** Inverse-frequency weighting for imbalanced data
6. **Training loop:**
   - Optimize only `rule_mass_params` via AdamW
   - Each epoch: `probs = model.forward(xb)` → loss (MSE or CE) → backward → clip gradients → step → `project_rules_to_simplex()`
7. **Validation & early stopping:** Track val_loss, save best state, stop after `patience` epochs without improvement

### 3.2 Projected Gradient Descent vs. Softmax

To ensure valid masses ($\sum m = 1$, $m \ge 0$), we tested two approaches:

| Dataset | PGD F1 | Softmax F1 | Winner |
|:--------|:-------|:-----------|:-------|
| Adult | 0.846 | 0.846 | TIE |
| German | **0.706** | 0.699 | PGD |
| Breast Cancer | 0.946 | 0.946 | TIE |
| Wine | 0.703 | 0.705 | TIE |

**Conclusion:** PGD performs equally or better. We use explicit projection (clamp + normalize) for:
- No vanishing gradients
- Direct interpretability of mass weights
- Simpler debugging

### 3.3 No Explicit Uncertainty Penalties

The loss function contains **no penalty for high Ω**. Uncertainty decreases only where it helps classification; in noisy regions, high Ω correctly reflects model ignorance.

---

## 4. Interpretability: Rule Inspection & H-Score

### 4.1 Rule Inspection

`sample_rule_inspector.py` allows interactive exploration:

```bash
python sample_rule_inspector.py --algo RIPPER --dataset german --idx 10
```

Output includes:
- Original features of the sample
- All fired rules with conditions
- Mass vectors $[m_1, ..., m_K, m_\Omega]$
- Rule statistics (precision, coverage, support)
- Final DST prediction, probabilities, and Ω

### 4.2 H-Score: Ranking Rules

Raw uncertainty $m_\Omega$ isn't enough — a rule with $m_\Omega = 0$ but masses $[0.5, 0.5, 0, ...]$ isn't discriminative.

**H-Score** balances certainty and class separation:

1. Certainty: $U' = 1 - m_\Omega$
2. Class ratio: $R = \frac{m_{max_1}}{m_{max_2} + \varepsilon}$, normalized via $R' = \text{Norm}(\log(1+R))$
3. H-Score: $H = \frac{2 \cdot U' \cdot R'}{U' + R' + \varepsilon}$

High H-Score = rule is both **certain** (low Ω) and **discriminative** (prefers one class).

---

## 5. Empirical Results

Full results in `Common_code/results/benchmark_dataset_*_metrics.csv`.

### 5.0 RAW Rules vs DST-Enhanced: The Key Improvement

DST transforms rigid rules into "soft" evidence sources. The table below shows how DST improves upon raw rule-based predictions:

| Dataset | Algorithm | RAW Acc | DST Acc | RAW Recall | DST Recall | Improvement |
|:--------|:----------|:--------|:--------|:-----------|:-----------|:------------|
| **Bank Marketing** | RIPPER | 0.89 | **0.89** | 0.25 | **0.58** | +132% Recall |
| **Bank Marketing** | FOIL | 0.88 | **0.90** | 0.30 | **0.46** | +53% Recall |
| **Adult** | RIPPER | 0.83 | **0.85** | 0.55 | **0.68** | +24% Recall |
| **German Credit** | FOIL | 0.70 | **0.73** | 0.35 | **0.44** | +26% Recall |

**Key insight:** DST allows weaker rules to contribute evidence. Even if no single rule fires with high confidence, the **combination** of multiple partial signals produces accurate predictions with appropriate uncertainty.

### Why RIPPER/FOIL outperform STATIC:

| Aspect | RIPPER/FOIL | STATIC |
|:-------|:------------|:-------|
| Rule quality | Few, precise rules | Many noisy rules |
| Final Ω | Lower (0.3-0.6) | Higher (0.6-0.8) |
| Interpretation | Each rule meaningful | Many redundant rules |

---

### 5.1 Brain Tumor MRI (Binary, Near-Separable)

| Model | Accuracy | F1 | Ω |
|:------|:---------|:---|:--|
| RIPPER_DST | **0.990** | **0.989** | 0.48 |
| FOIL_DST | 0.988 | 0.987 | 0.36 |

**Interpretation:** Near-perfect separation with simple rules. DST adds minimal refinement.

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_Brain%20Tumor_metrics.png" width="85%">
</p>

### 5.2 Breast Cancer Wisconsin

| Model | Accuracy | F1 | Precision | Recall | Ω |
|:------|:---------|:---|:----------|:-------|:--|
| RIPPER_DST | **0.964** | **0.949** | 0.925 | 0.974 | 0.70 |

**Interpretation:** Clinically interpretable rules (Bare_Nuclei, Clump_Thickness) achieve near-ceiling performance.

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_breast-cancer-wisconsin_metrics.png" width="85%">
</p>

### 5.3 Adult Income

| Model | Accuracy | F1 | Precision | Recall | Ω |
|:------|:---------|:---|:----------|:-------|:--|
| FOIL_DST | **0.851** | 0.678 | 0.734 | 0.629 | 0.53 |
| RIPPER_DST | 0.849 | **0.692** | 0.705 | 0.679 | 0.62 |

**Interpretation:** FOIL gives broader coverage (higher recall); RIPPER is more compact (higher precision). DST combines weak signals (education, occupation, capital gain) into confident predictions.

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_adult_metrics.png" width="85%">
</p>

### 5.4 Bank Marketing (Highly Imbalanced — 11.7% positive)

| Model | Accuracy | F1 | Precision | Recall | Ω |
|:------|:---------|:---|:----------|:-------|:--|
| RIPPER_DST | **0.893** | **0.559** | 0.538 | 0.580 | 0.62 |
| FOIL_DST | 0.896 | 0.509 | 0.569 | 0.461 | 0.49 |

**Key insight:** Without DST, raw rules achieve ~0.25 recall on minority class. With DST combination of weak signals (poutcome, duration, contacts), recall doubles to ~0.58.

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_bank-full_metrics.png" width="85%">
</p>

### 5.5 Wine Quality

| Model | Accuracy | F1 | Recall | Ω |
|:------|:---------|:---|:-------|:--|
| RIPPER_DST | **0.723** | **0.769** | 0.818 | 0.78 |
| STATIC_DST | 0.704 | 0.744 | 0.789 | 0.80 |

**Interpretation:** Feature interactions (alcohol × sulphates) are key. High Ω reflects limited data and task complexity.

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_df_wine_metrics.png" width="85%">
</p>

### 5.6 Gas Drift (126 features, 6 classes)

| Model | Accuracy | F1 | Ω |
|:------|:---------|:---|:--|
| FOIL_DST | **0.982** | **0.982** | **0.03** |
| RIPPER_DST | 0.979 | 0.979 | 0.19 |
| STATIC_DST | 0.934 | 0.934 | 0.65 |

**Interpretation:** FOIL generates few but decisive rules → near-deterministic predictions (Ω→0). STATIC's dense rule set creates conflicts → high Ω correctly reflects noise.

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_gas_drift_metrics.png" width="85%">
</p>

### 5.7 German Credit

| Model | Accuracy | F1 | Ω |
|:------|:---------|:---|:--|
| RIPPER_DST | **0.744** | 0.506 | 0.75 |
| STATIC_DST | 0.731 | **0.574** | 0.80 |
| FOIL_DST | 0.731 | 0.442 | 0.61 |

**Interpretation:** When rules conflict ("good salary" vs "bad payment history"), DST outputs high Ω — a natural signal for manual review of borderline cases.

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_german_metrics.png" width="85%">
</p>

### 5.8 SAheart (Cardiovascular Risk)

| Model | Accuracy | F1 | Ω |
|:------|:---------|:---|:--|
| FOIL_DST | **0.649** | **0.552** | 0.71 |
| STATIC_DST | 0.608 | 0.525 | 0.80 |

**Interpretation:** Small dataset with high noise. FOIL_DST achieves best F1 while maintaining appropriate uncertainty.

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_SAheart_metrics.png" width="85%">
</p>

---

## 6. Usage

### 6.1 Installation

```bash
git clone https://github.com/SargisVardanian/DST.git
cd DST
pip install numpy pandas scikit-learn matplotlib seaborn torch
```

### 6.2 Running Benchmarks

```bash
cd Common_code

# Run experiment on gas_drift
python test_Ripper_DST.py --dataset gas_drift --experiment stable

# Other datasets: adult, bank-full, german, df_wine, SAheart, breast-cancer-wisconsin
python test_Ripper_DST.py --dataset adult --experiment stable

# Specify device and epochs
python test_Ripper_DST.py --dataset gas_drift --device cuda --epochs 100
```

Results saved to `Common_code/results/`.

### 6.3 Inspecting Individual Predictions

```bash
# Inspect sample #5 on gas_drift with RIPPER rules
python sample_rule_inspector.py --algo RIPPER --dataset gas_drift --idx 5

# Same with FOIL
python sample_rule_inspector.py --algo FOIL --dataset gas_drift --idx 5
```

### 6.4 Programmatic API

```python
from Common_code.DSClassifierMultiQ import DSClassifierMultiQ
from Common_code.Datasets_loader import load_dataset

# 1. Load dataset
X, y, feature_names, value_decoders = load_dataset('german.csv')

# 2. Create classifier
clf = DSClassifierMultiQ(
    k=len(set(y)),
    rule_algo="RIPPER",
    device="cpu",
    max_iter=50,
    lr=1e-3,
)

# 3. Fit: generates rules, initializes masses, trains via DSGD
clf.fit(X, y, feature_names=feature_names, value_decoders=value_decoders)

# 4. Predict
y_pred = clf.predict(X)
print("Train accuracy:", (y_pred == y).mean())

# 5. Inspect rules
clf.model.print_rules(top_n=5)
```

### 6.5 Key Files

| File | Purpose |
|:-----|:--------|
| `rule_generator.py` | RIPPER/FOIL/STATIC rule induction |
| `DSModelMultiQ.py` | DST model with mass assignment and Dempster combination |
| `core.py` | Dempster's rule, pignistic transform |
| `DSClassifierMultiQ.py` | Sklearn-compatible classifier with training loop |
| `sample_rule_inspector.py` | Interactive prediction inspection |
| `test_Ripper_DST.py` | Benchmark script |

---

## 7. References & Acknowledgments

**Core Theory:**
1. Shafer, G. (1976). *A Mathematical Theory of Evidence*. Princeton University Press.
2. Peñafiel, S., Baloian, N., et al. *Applying the Dempster-Shafer Theory for Detecting Inconsistencies*. Expert Systems with Applications.
3. DST + gradient-based optimization inspired by: A. Tarkhanyan, A. Harutyunyan, "DSGD++: Revised Methodology for Gradient-Based Mass Optimization in DST"

**Rule Algorithms:**
- Cohen, W. W. (1995). *Fast Effective Rule Induction (RIPPER)*. ICML.
- Quinlan, J. R. (1990). *Learning Logical Definitions from Relations (FOIL)*.

**Development:**
This project was developed by **S. Vardanian**, in collaboration with **A. Tarkhanyan** and **A. Harutyunyan**, extending evidential learning ideas to practical, interpretable rule-based classification.

---

_Last updated: 2024-12-03_
