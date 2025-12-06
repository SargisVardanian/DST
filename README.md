# DST: Interpretable Rule-Based Classification via Dempster-Shafer Theory

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-green" alt="Python">
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange" alt="PyTorch">
</p>

## Abstract

This repository implements an interpretable classification framework that combines **symbolic rule-based models** with **Dempster-Shafer Theory (DST)** and gradient-based optimization.

Classic rule learners such as **RIPPER** and **FOIL** produce human-readable IF–THEN rules, but they are rigid and lack calibrated uncertainty. On the other hand, modern probabilistic models are accurate but often opaque.

Our framework bridges this gap:

* Logical rules are treated as **sources of evidence**
* Each rule has a learnable **mass function** over classes + uncertainty
* Evidence from multiple rules is aggregated via **Dempster's rule**
* Masses are optimized by **Dempster-Shafer Gradient Descent (DSGD)**

We support three rule generation strategies — **RIPPER**, **FOIL**, and **STATIC** — and show how DST turns them into robust, differentiable classifiers with explicit epistemic uncertainty.

A **representation-based mass initialization** is provided as an option: rule masses can be initialized using **class centroids, representativeness of samples, and rule purity**, which accelerates convergence and reduces initial uncertainty compared to purely random initialization.

---

## 1. Theoretical Framework

### 1.1 Dempster-Shafer Theory of Evidence

For a classification task with classes $\Theta = {C_1, C_2, \dots, C_K}$, DST defines a **mass assignment function** $m$ over subsets of $\Theta$:

$$\sum_{A \subseteq \Theta} m(A) = 1, \qquad m(\emptyset) = 0$$

Unlike Bayesian probability, DST allows mass on **non-singleton** sets:

* Mass on ${C_1}$ is belief specifically in class $C_1$
* Mass on $\Theta$ (denoted $\Omega$) models **pure ignorance**

For brevity, we denote the universal set $\Theta$ by $\Omega$ in the remainder of this document.

In this project we follow a common simplification — we only assign masses to **singletons** and the **universal set** $\Omega$:

$$\mathbf{m} = [m_1, m_2, \dots, m_K, m_\Omega]$$

where $m_k = m({C_k})$ and $m_\Omega = m(\Theta) = m(\Omega)$. For each rule $r_i$ we maintain such a vector $\mathbf{m}^{(i)}$ — these are the parameters the model learns.

### 1.2 Dempster's Rule of Combination

Given two rules with mass functions $m^{(1)}$ and $m^{(2)}$, Dempster's rule combines them:

$$(m^{(1)} \oplus m^{(2)})(A) = \frac{1}{1 - \kappa} \sum_{B \cap C = A} m^{(1)}(B), m^{(2)}(C)$$

where the **conflict** $\kappa$ is:

$$\kappa = \sum_{B \cap C = \emptyset} m^{(1)}(B), m^{(2)}(C)$$

In our singletons+Ω setting:

* Mass on each class $C_k$ is proportional to both rules supporting $C_k$, or one rule supporting $C_k$ while the other is ignorant (Ω)
* Mass on Ω is proportional to both rules being ignorant
* Conflict $\kappa$ arises when rules support **different classes simultaneously**; normalization by $1 - \kappa$ redistributes this conflict

The code for batched pairwise combination is in `core.py` (`ds_combine_pair`), with a vectorized wrapper `_combine_active_masses` in `DSModelMultiQ`.

### 1.3 From Masses to Probabilities

We convert masses $\mathbf{m}$ to probabilities $p$ using:

1. **Pignistic transform** (uniformly spreading $m_\Omega$ across classes):

   $$p_k = m_k + \frac{m_\Omega}{K}, \quad K = |\Theta|$$

2. **Class-prior-based redistribution** (optional): distribute ignorance proportionally to class priors $\pi$:

   $$p_k = m_k + m_\Omega \cdot \frac{\pi_k}{\sum_j \pi_j}$$

This is implemented in `DSModelMultiQ._mass_to_prob` and `core.masses_to_pignistic`.

---

## 2. Model Architecture

The end-to-end classifier has three conceptual layers:

1. **Rule Generation** (`rule_generator.py`) — RIPPER / FOIL / STATIC learn symbolic rules from `(X, y)`
2. **DST Model** (`DSModelMultiQ.py`) — stores rules, evaluates activations, maintains per-rule mass vectors, combines masses via Dempster's rule, outputs class probabilities
3. **Training Loop** (`DSClassifierMultiQ.py`) — splits data, calls rule generator, initializes masses, trains via projected gradient descent

This separation lets you:

* Swap rule generators while keeping the evidential layer fixed
* Reuse the DST model with externally provided rule sets
* Plug the whole thing into an sklearn-style pipeline (`fit/predict`)

### 2.1 Rule Generation Strategies

All strategies operate on a shared representation:

* A **literal** is a simple condition `(feature, operator, threshold/value)`:

  * numeric: `V_j < t`, `V_j > t`
  * categorical: `V_j == value`
* A **rule** is a conjunction of literals: IF $\ell_1 \land \ell_2 \land \dots \land \ell_L$ THEN class $c$

#### 2.1.1 FOIL (First Order Inductive Learner)

FOIL is a classic **top-down** rule learner (Quinlan, 1990):

1. Start with an empty rule (covers everything)
2. Iteratively add literals to maximize **FOIL-Gain**:

   $$Gain = t \cdot \left(\log_2\frac{p_1}{p_1 + n_1} - \log_2\frac{p_0}{p_0 + n_0}\right)$$

   where $p_0, n_0$ are positives/negatives before, $p_1, n_1$ after, and $t$ is covered positives.
3. Stop when rule covers zero negatives
4. Remove covered positives; repeat

**Characteristics:**

* ✅ Generates **specific** rules with high precision
* ✅ Good at catching "edge cases"
* ❌ No pruning phase → prone to overfitting, especially on noisy datasets

#### 2.1.2 RIPPER (Repeated Incremental Pruning)

RIPPER (Cohen, 1995) adds critical **pruning**:

1. **Grow Phase:** On the Grow Set, add literals with FOIL-gain until the rule is (almost) pure

2. **Prune Phase:** On the Prune Set, remove literals from the end if it improves:

   $$Score = \frac{p - n}{p + n}$$

3. **Optimization Phase:** Rewrite/replace rules using an MDL-like criterion

Compared to FOIL, RIPPER trades some recall for more robust generalization via pruning and MDL-based rule set optimization.

**Our Extension: Tiered Precision RIPPER**

```python
precision_tiers = [
    {"min_precision": 0.80, "max_literals": 8},
    {"min_precision": 0.60, "max_literals": 6},
    {"min_precision": 0.40, "max_literals": 5},
    {"min_precision": 0.20, "max_literals": 4},
]
```

* First grow very precise (but narrow) rules
* Then allow less precise rules with limited length
* For DST: high-precision rules provide strong evidence; lower-precision rules add weak but useful signals, especially for recall

#### 2.1.3 STATIC (Exhaustive Heuristic Search)

STATIC casts a "wide net" by combining simple candidates:

1. **Single literals:** Quantile thresholds for numeric features, frequent values for categorical
2. **Pairs and triples:** Combine the best literals into 2–3 condition rules
3. **Diversity filter:** Remove highly overlapping rules (Jaccard similarity) to keep a diverse rule set

**Result:** Hundreds (sometimes 1000+) rules including feature interactions. Noisy rules are naturally suppressed by DST — they learn high $m_\Omega \approx 1$ and contribute little to final decisions.

#### 2.1.4 Strategy Comparison

| Aspect          | RIPPER              | FOIL             | STATIC                   |
| :-------------- | :------------------ | :--------------- | :----------------------- |
| **Strategy**    | Grow–Prune–Optimize | Greedy FOIL-gain | Exhaustive heuristic     |
| **Rule Count**  | Low (20–150)        | Medium (100–400) | High (400–1200)          |
| **Overfitting** | Resistant           | Prone            | Offloaded to DST layer   |
| **Best For**    | Clean, high-signal  | Balanced data    | Noisy, interaction-heavy |

In practice, RIPPER/FOIL produce compact rule sets that are easy to inspect; STATIC acts as a dense feature-interaction generator whose noise is filtered by DST.

### 2.2 DST Layer (DSModelMultiQ)

`DSModelMultiQ.py` is the core evidential model:

1. **Feature encoding:**

   * Numeric features are used as-is (optionally normalized upstream)
   * Categorical features are encoded via internal label encoders and stored in `value_decoders`
2. **Literal evaluation:** `_eval_literals(X)` → boolean masks for all literals
3. **Rule activation:** `_activation_matrix(X)` → matrix $A[n, r] = 1$ if rule $r$ fires on sample $n$
4. **Mass parameterization:**

   * Parameters: `rule_mass_params` matrix of shape `[N_{rules}, K+1]`
   * Projection: `params_to_mass` maps unconstrained params onto the probability simplex (classes + Ω)
5. **Combination:** `_combine_active_masses` applies Dempster's rule only over **active** rules for each sample (sparse combination)
6. **Probabilities:** `_mass_to_prob` applies the chosen redistribution (pignistic or prior-based) and returns class probabilities

This layer is **fully differentiable** w.r.t. the mass parameters, which enables gradient-based training while preserving interpretable rule structure.

### 2.3 Mass Initialization: Random vs Representation-Based

**Random initialization** (`reset_masses`):

* Draw random logits for $K$ classes
* Normalize to a distribution
* Allocate a fixed base uncertainty (e.g., 0.8) to Ω, rescale class masses accordingly

This yields high $m_\Omega$ initially, allowing the optimizer to "learn away" ignorance where the data is clean.

**Representation-based initialization** (DSGD++-style):

When enabled, we use a deterministic, data-driven initialization:

1. Build class centroids $\mu_y$ from training data (in the **normalized feature space**)
2. Compute representativeness for each sample:

   $$R(x) = \frac{1}{1 + \lVert x - \mu_y \rVert_2}$$

   where $\lVert \cdot \rVert_2$ is the Euclidean norm in the normalized feature space.
3. For each rule $r$:

   * Find covered samples $D_r$

   * Calculate purity (majority class fraction in $D_r$)

   * Compute average representativeness $\mathbb{E}[R(x) \mid x \in D_r]$

   * Define confidence:

     $$Conf(r) = purity \times \mathbb{E}[R(x) \mid x \in D_r]$$

   * Assign:

     * $m(\text{majority class}) = Conf(r)$
     * $m(\Omega) = 1 - Conf(r)$

Intuitively, **pure rules covering highly representative points** start with low Ω and strong class mass; ambiguous or "weird" rules start near full ignorance.

---

## 3. Training Loop (DSClassifierMultiQ)

### 3.1 Main Steps in `fit(X, y, ...)`

High-level logic in `DSClassifierMultiQ.fit`:

1. **Data preparation:**

   * Convert `X`, `y` to numpy arrays
   * Optionally infer/attach `feature_names` and `value_decoders`
2. **Rule generation:**

   * If `model.rules` is empty, call `model.generate_rules(X, y, algo=...)`
   * Rule algorithm is controlled by `rule_algo` (`"RIPPER"`, `"FOIL"`, `"STATIC"`)
3. **Train/Val split:**

   * Stratified random split with `val_split`
   * Same split is used for RAW vs DST comparisons
4. **Mass initialization:**

   * If representation-based initialization is enabled, compute centroids + purity-based masses (as in §2.3)
   * If it fails (e.g., degenerate coverage or numerical issues), fall back to `reset_masses` (random with high Ω)
5. **Class weights:**

   * Inverse-frequency weighting for imbalanced data
   * Used to reweight the loss (cross-entropy or MSE) per class
6. **Training loop:**

   * Optimize only `rule_mass_params` via AdamW
   * For each epoch:

     * Sample batches
     * `probs = model.forward(xb)`
     * Compute loss (CE/MSE) with class weights
     * Backpropagate, clip gradients
     * Optimizer step
     * `project_rules_to_simplex()` to enforce valid masses ($\sum m = 1$, $m \ge 0$)
7. **Validation & early stopping:**

   * Track `val_loss` (and optionally F1)
   * Keep the best model state (by val_loss)
   * Stop after `patience` epochs without improvement

The core idea: **rules are fixed**, we only learn how much evidence each rule should contribute to each class vs Ω.

### 3.2 Projected Gradient Descent vs Softmax

To ensure valid masses ($\sum m = 1$, $m \ge 0$), we tested two approaches:

| Dataset       | PGD F1    | Softmax F1 | Winner |
| :------------ | :-------- | :--------- | :----- |
| Adult         | 0.846     | 0.846      | TIE    |
| German        | **0.706** | 0.699      | PGD    |
| Breast Cancer | 0.946     | 0.946      | TIE    |
| Wine          | 0.703     | 0.705      | TIE    |

**Conclusion:** PGD performs equally or better. We use explicit projection (clamp + normalize) because:

* It avoids vanishing gradient issues tied to softmax saturation
* The parameters remain directly interpretable as masses
* It simplifies debugging and manual edits (you can literally look at the learned $[m_1,\dots,m_K,m_\Omega]$ vectors)

### 3.3 No Explicit Uncertainty Penalties

The loss function contains **no penalty for high Ω**.

* Uncertainty decreases only where it improves classification
* In noisy or contradictory regions, high Ω correctly reflects model ignorance
* For downstream decision-making, Ω can be used as a **reject option** or a trigger for human review

---

## 4. Interpretability: Rule Inspection & H-Score

### 4.1 Rule Inspection

`sample_rule_inspector.py` allows interactive exploration of individual predictions:

```bash
python sample_rule_inspector.py --algo RIPPER --dataset german --idx 10
```

The script prints:

* Original features of the selected sample
* All fired rules with human-readable conditions
* Mass vectors $[m_1, \dots, m_K, m_\Omega]$ for each fired rule
* Rule statistics (precision, coverage, support)
* Final DST prediction, class probabilities, and Ω for that sample

This makes it easy to answer **“Why did the model predict this?”** at the level of explicit conditions, not opaque neurons.

### 4.2 H-Score: Ranking Rules

Raw uncertainty $m_\Omega$ isn't enough — a rule with $m_\Omega = 0$ but masses $[0.5, 0.5, 0, \dots]$ isn't discriminative.

**H-Score** balances certainty and class separation:

1. Certainty: $U' = 1 - m_\Omega$

2. Class ratio:

   $$R = \frac{m_{\text{max}*1}}{m*{\text{max}_2} + \varepsilon}$$

   where $m_{\text{max}*1}$ and $m*{\text{max}_2}$ are the largest and second-largest class masses for that rule.

3. Normalized discrimination:

   $$R' = \text{Norm}(\log(1 + R))$$

4. Final H-Score:

   $$H = \frac{2 \cdot U' \cdot R'}{U' + R' + \varepsilon}$$

Here `Norm(.)` denotes **min–max normalization across all rules on the training set**:

$$
R' = \frac{\log(1+R) - \min_r \log(1+R_r)}{\max_r \log(1+R_r) - \min_r \log(1+R_r) + \varepsilon}
$$

High H-Score = rule is both:

* **Certain** (low Ω)
* **Discriminative** (clearly prefers one class over the others)

You can sort rules by H-Score to quickly find the most informative ones.

---

## 5. Empirical Results

Full results live in `Common_code/results/benchmark_dataset_*_metrics.csv`.

### 5.0 RAW Rules vs DST-Enhanced: The Key Improvement

DST transforms rigid rules into "soft" evidence sources. The table below shows how DST improves upon raw rule-based predictions.

Here **RAW** corresponds to the **same set of rules** evaluated with **hard majority voting** (no mass learning, no DST combination) on the **same train/validation split** as the DST-enhanced models.

| Dataset            | Algorithm | RAW Acc | DST Acc  | RAW Recall | DST Recall | Improvement  |
| :----------------- | :-------- | :------ | :------- | :--------- | :--------- | :----------- |
| **Bank Marketing** | RIPPER    | 0.89    | **0.89** | 0.25       | **0.58**   | +132% Recall |
| **Bank Marketing** | FOIL      | 0.88    | **0.90** | 0.30       | **0.46**   | +53% Recall  |
| **Adult**          | RIPPER    | 0.83    | **0.85** | 0.55       | **0.68**   | +24% Recall  |
| **German Credit**  | FOIL      | 0.70    | **0.73** | 0.35       | **0.44**   | +26% Recall  |

**Key insight:** DST allows weaker rules to contribute evidence. Even if no single rule fires with high confidence, the **combination** of multiple partial signals produces accurate predictions with calibrated uncertainty.

### RIPPER/FOIL vs STATIC: qualitative comparison

On our benchmarks, RIPPER/FOIL usually produce more compact and precise rule sets than STATIC. Qualitatively:

| Aspect         | RIPPER/FOIL             | STATIC                     |
| :------------- | :---------------------- | :------------------------- |
| Rule quality   | Few, precise rules      | Many noisy rules           |
| Final Ω        | Lower (0.3–0.6)         | Higher (0.6–0.8)           |
| Interpretation | Each rule is meaningful | Many redundant/overlapping |

STATIC is not "worse by design" — it trades interpretability and compactness for **coverage**. On some noisy, highly entangled tabular problems (e.g., **German Credit**), STATIC_DST achieves the best F1 due to its wide rule coverage, at the cost of higher Ω and a denser rule set. DST’s uncertainty then helps flag such borderline decisions for manual review instead of over-committing.

---

### 5.1 Brain Tumor MRI (Binary, Near-Separable)

| Model      | Accuracy  | F1        | Ω    |
| :--------- | :-------- | :-------- | :--- |
| RIPPER_DST | **0.990** | **0.989** | 0.48 |
| FOIL_DST   | 0.988     | 0.987     | 0.36 |

**Interpretation:** Near-perfect separation with simple rules. DST adds minimal refinement; most uncertainty comes from marginal cases and class overlap around decision boundaries.

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_Brain%20Tumor_metrics.png" width="85%">
</p>

### 5.2 Breast Cancer Wisconsin

| Model      | Accuracy  | F1        | Precision | Recall | Ω    |
| :--------- | :-------- | :-------- | :-------- | :----- | :--- |
| RIPPER_DST | **0.964** | **0.949** | 0.925     | 0.974  | 0.70 |

**Interpretation:** Clinically interpretable rules (e.g., `Bare_Nuclei`, `Clump_Thickness`) achieve near-ceiling performance. DST mainly provides a consistent way to express uncertainty for borderline lesions, which is critical for risk-aware medical decision support.

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_breast-cancer-wisconsin_metrics.png" width="85%">
</p>

### 5.3 Adult Income

| Model      | Accuracy  | F1        | Precision | Recall | Ω    |
| :--------- | :-------- | :-------- | :-------- | :----- | :--- |
| FOIL_DST   | **0.851** | 0.678     | 0.734     | 0.629  | 0.53 |
| RIPPER_DST | 0.849     | **0.692** | 0.705     | 0.679  | 0.62 |

**Interpretation:**

* FOIL provides broader coverage (higher recall) but with slightly lower precision
* RIPPER yields a more compact rule set with better F1
* DST combines weak signals (education, occupation, capital gain, marital status) into confident predictions and exposes uncertainty where the socio-economic signals conflict

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_adult_metrics.png" width="85%">
</p>

### 5.4 Bank Marketing (Highly Imbalanced — 11.7% positive)

| Model      | Accuracy  | F1        | Precision | Recall | Ω    |
| :--------- | :-------- | :-------- | :-------- | :----- | :--- |
| RIPPER_DST | **0.893** | **0.559** | 0.538     | 0.580  | 0.62 |
| FOIL_DST   | 0.896     | 0.509     | 0.569     | 0.461  | 0.49 |

**Key insight:** Without DST, raw rules achieve ~0.25 recall on the minority class. With DST, the combination of weak signals (poutcome, duration, number of contacts, campaign history) doubles recall to ~0.58 with manageable loss in precision.

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_bank-full_metrics.png" width="85%">
</p>

### 5.5 Wine Quality

| Model      | Accuracy  | F1        | Recall | Ω    |
| :--------- | :-------- | :-------- | :----- | :--- |
| RIPPER_DST | **0.723** | **0.769** | 0.818  | 0.78 |
| STATIC_DST | 0.704     | 0.744     | 0.789  | 0.80 |

**Interpretation:** Feature interactions (e.g., alcohol × sulphates, acidity × residual sugar) are key. High Ω reflects limited data and task complexity: many borderline samples are left with significant uncertainty rather than being forced into a hard class.

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_df_wine_metrics.png" width="85%">
</p>

### 5.6 Gas Drift (126 features, 6 classes)

| Model      | Accuracy  | F1        | Ω        |
| :--------- | :-------- | :-------- | :------- |
| FOIL_DST   | **0.982** | **0.982** | **0.03** |
| RIPPER_DST | 0.979     | 0.979     | 0.19     |
| STATIC_DST | 0.934     | 0.934     | 0.65     |

**Interpretation:** FOIL generates few but decisive rules → near-deterministic predictions (Ω → 0). STATIC's dense rule set creates many overlapping conditions and conflicts; DST responds by assigning high Ω, correctly reflecting drift and noise in the sensor array.

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_gas_drift_metrics.png" width="85%">
</p>

### 5.7 German Credit

| Model      | Accuracy  | F1        | Ω    |
| :--------- | :-------- | :-------- | :--- |
| RIPPER_DST | **0.744** | 0.506     | 0.75 |
| STATIC_DST | 0.731     | **0.574** | 0.80 |
| FOIL_DST   | 0.731     | 0.442     | 0.61 |

**Interpretation:** German Credit is noisy and multi-factorial (income, employment history, savings, past defaults, etc.). STATIC’s wide rule coverage yields the best F1 but at the cost of many conflicting rules. When rules disagree (e.g., "good salary" vs "bad payment history"), DST outputs high Ω — a natural signal for manual review of borderline applicants.

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_german_metrics.png" width="85%">
</p>

### 5.8 SAheart (Cardiovascular Risk)

| Model      | Accuracy  | F1        | Ω    |
| :--------- | :-------- | :-------- | :--- |
| FOIL_DST   | **0.649** | **0.552** | 0.71 |
| STATIC_DST | 0.608     | 0.525     | 0.80 |

**Interpretation:** Small dataset with high noise and correlated risk factors (cholesterol, tobacco use, family history, etc.). FOIL_DST achieves the best F1 while maintaining appropriate uncertainty, making it a good candidate for risk stratification where false certainty is dangerous.

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

You may also want to create a dedicated virtualenv/conda environment with an appropriate PyTorch build for your hardware.

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

Results (metrics CSV + plots) are saved to `Common_code/results/`.

### 6.3 Inspecting Individual Predictions

```bash
# Inspect sample #5 on gas_drift with RIPPER rules
python sample_rule_inspector.py --algo RIPPER --dataset gas_drift --idx 5

# Same with FOIL
python sample_rule_inspector.py --algo FOIL --dataset gas_drift --idx 5
```

This prints fired rules, mass vectors, and the final DST decision for a single sample.

### 6.4 Programmatic API

```python
from Common_code.DSClassifierMultiQ import DSClassifierMultiQ
from Common_code.Datasets_loader import load_dataset

# 1. Load dataset (path is relative to the repo root)
X, y, feature_names, value_decoders = load_dataset("Common_code/data/german.csv")

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

See `Common_code/Datasets_loader.py` for the list of available datasets and their default paths.

### 6.5 Key Files

| File                       | Purpose                                          |
| :------------------------- | :----------------------------------------------- |
| `rule_generator.py`        | RIPPER/FOIL/STATIC rule induction                |
| `DSModelMultiQ.py`         | DST model with mass assignment and combination   |
| `core.py`                  | Dempster's rule, pignistic transform, utilities  |
| `DSClassifierMultiQ.py`    | Sklearn-compatible classifier with training loop |
| `sample_rule_inspector.py` | Interactive prediction inspection                |
| `test_Ripper_DST.py`       | Benchmark script                                 |
| `Datasets_loader.py`       | Dataset loading utilities and metadata           |

### 6.6 Datasets

The benchmark scripts expect CSV files that are shipped with the repository (see `Common_code/Datasets_loader.py` for exact filenames). Typical dataset IDs and sources:

| Dataset id (CLI / code)   | Description                       | Original source (typical)          |
| :------------------------ | :-------------------------------- | :--------------------------------- |
| `adult`                   | Adult Income (binary, imbalanced) | UCI Adult                          |
| `bank-full`               | Bank Marketing (full)             | UCI Bank Marketing                 |
| `german`                  | German Credit                     | UCI German Credit                  |
| `df_wine`                 | Wine Quality (merged red/white)   | UCI Wine Quality                   |
| `breast-cancer-wisconsin` | Breast Cancer Wisconsin           | UCI Breast Cancer Wisconsin        |
| `gas_drift`               | Gas sensor array drift (6 gases)  | UCI Gas Drift                      |
| `Brain Tumor`             | Brain tumor MRI classification    | Public MRI datasets (e.g., Kaggle) |
| `SAheart`                 | South African heart disease       | `ElemStatLearn` / UCI variants     |

All CSVs are placed under the `Common_code/data/` directory in the repository.

---

## 7. References & Acknowledgments

**Core Theory:**

1. Shafer, G. (1976). *A Mathematical Theory of Evidence*. Princeton University Press.
2. Peñafiel, S., Baloian, N., et al. *Applying the Dempster-Shafer Theory for Detecting Inconsistencies*. Expert Systems with Applications.
3. DST + gradient-based optimization inspired by: A. Tarkhanyan, A. Harutyunyan, *"DSGD++: Revised Methodology for Gradient-Based Mass Optimization in DST"*.

**Rule Algorithms:**

* Cohen, W. W. (1995). *Fast Effective Rule Induction (RIPPER)*. ICML.
* Quinlan, J. R. (1990). *Learning Logical Definitions from Relations (FOIL)*.

**Development:**

This project was developed by **S. Vardanian**, in collaboration with **A. Tarkhanyan** and **A. Harutyunyan**, extending evidential learning ideas to practical, interpretable rule-based classification.
