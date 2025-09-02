# DST: Rule-Based Classification with RIPPER/FOIL + Dempster-Shafer Theory

This repository contains the official implementation for the paper **"Applying the Dempster-Shafer theory for a flexible, accurate and interpretable classifier"**, published in *Expert Systems with Applications*. The project provides a robust framework for building interpretable machine learning models by synergizing classic rule-induction algorithms (RIPPER, FOIL) with Dempster-Shafer Theory for intelligent evidence combination.

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_german_metrics.png" width="700" alt="Benchmark results on the German Credit dataset">
</p>

> • **Interpretable by Design**: Generates human-readable IF-THEN rules using RIPPER or FOIL, forming a transparent decision model.  
> • **Robust Evidence Combination**: Leverages Dempster-Shafer Theory to move beyond rigid "first-match" logic, enabling a flexible consensus from multiple rules.  
> • **Optimized and Compact**: Learns rule confidences via gradient descent and intelligently prunes the model to create classifiers that are small, fast, and accurate.

---

## 1 · Quick Start

Clone the repository and run the main benchmark script to replicate the core experiments.

```bash
# 1. Clone the repository
git clone https://github.com/SargisVardanian/DST.git
cd DST

# 2. (Optional) Create and activate a virtual environment
# python -m venv venv
# source venv/bin/activate 

# 3. Install dependencies
# pip install -r requirements.txt # Assuming you create a requirements.txt

# 4. Navigate to the main code directory
cd Common_code

# 5. Run the full experiment pipeline
python test_Ripper_DST.py
```


This script will automatically execute the full pipeline for the German Credit dataset:
1.  Induce rule sets using both **RIPPER** and **FOIL**.
2.  Train the Dempster-Shafer mass assignments for the rules.
3.  Prune the models to enhance simplicity and efficiency.
4.  Generate performance plots and save detailed metrics to CSV files in the `results/` folder.

---

## 2 · The Three-Stage Method: From Raw Data to a Refined Classifier

Our method transforms data into an accurate and interpretable classifier through a systematic, three-stage process.

### Stage 1: Rule Induction (RIPPER vs. FOIL)

The foundation of our model is a set of simple **IF-THEN rules** automatically discovered from the data. We employ two well-established algorithms, RIPPER and FOIL, which differ in their rule-building philosophy:

*   **RIPPER: The Careful Generalist**
    RIPPER (Repeated Incremental Pruning to Produce Error Reduction) constructs rules by iteratively adding conditions (`age > 30`, `status = 'single'`) to maximize information gain. After growing a rule, it immediately prunes it, removing any conditions that have become redundant. This "grow-then-prune" cycle produces rules that are concise and general.

*   **FOIL: The Greedy Specialist**
    FOIL (First-Order Inductive Learner) also adds conditions one by one, but without an immediate pruning step. It greedily seeks the best literal to add until the rule achieves high precision. This can result in more specific and numerous rules than RIPPER, capturing fine-grained patterns in the data.

**Initial Prediction Model:** At this stage, the model is a simple **decision list**. When classifying a new sample, the rules are checked in a fixed order. The prediction is made by the *first rule that matches*, and all other potentially matching rules are ignored. This is a fast but rigid baseline. An example rule:
`IF ('credit_history' == 'A34') AND ('duration' < 24) THEN Class = 1`

### Stage 2: Evidence Combination with Dempster-Shafer Theory (DST)

A decision list is inflexible. It cannot handle cases where multiple rules fire with conflicting advice, nor can it weigh the confidence of each rule. To overcome this, we introduce a **Dempster-Shafer Theory (DST)** layer.

Instead of a binary "yes/no" vote, each rule $R_i$ is assigned a learnable **mass vector** $\mathbf{m}^{(i)}$. This vector distributes the rule's belief across the $K$ possible classes and, crucially, includes a term for **uncertainty**, $m_{\text{unc}}^{(i)}$:

$$
\mathbf m^{(i)}= \bigl(m_1^{(i)}, \dots, m_K^{(i)}, m_{\text{unc}}^{(i)}\bigr), \qquad \text{where} \quad \sum_{j=1}^K m_j^{(i)} + m_{\text{unc}}^{(i)} = 1.
$$

When a new sample $\mathbf x$ is presented, **all matching rules** "cast their votes." We combine these votes using Dempster's rule of combination. Our implementation performs this efficiently in the **commonality space**, where evidence fusion becomes a simple element-wise product:

$$
q_k(\mathbf x) = \prod_{i \in \mathcal{R}(\mathbf{x})} \left( m_k^{(i)} + m_{\text{unc}}^{(i)} \right)
$$

The key innovation is that these mass vectors $\mathbf{m}^{(i)}$ are **trainable parameters**. We use an optimizer (e.g., Adam) to adjust them via gradient descent, minimizing the classification error on the training data. This process teaches the model:
*   **Which rules to trust**: Consistently correct rules learn to have low uncertainty ($m_{\text{unc}}$) and high mass for the correct class.
*   **Which rules are less reliable**: Ambiguous or noisy rules learn to assign more mass to uncertainty, reducing their influence on the final prediction.

This DST training stage transforms the rigid decision list into a flexible and robust classifier that learns to weigh and synthesize evidence from multiple sources.

### Stage 3: Model Simplification via Pruning

After training, the model may contain rules that are redundant, ambiguous, or irrelevant. The final stage is to **prune** these weak rules to create a model that is smaller, faster, and even easier to interpret.

A rule is removed if it meets **any** of the following criteria, which are derived from the learned masses and data statistics:

1.  **High Uncertainty**: Its learned uncertainty mass $m_{\text{unc}}^{(i)}$ exceeds a threshold (e.g., > 0.6). The model itself has flagged the rule as unreliable.
2.  **High Ambiguity**: It fails to strongly distinguish between classes. We measure this by the log-ratio of belief in the top class versus the second-best class. If the ratio is low, the rule is not discriminative.
3.  **Low Coverage**: It fires on very few training samples (e.g., < 10). Such rules are often noise-driven and lack generalizability.

This intelligent filtering process uses insights from the DST training to simplify the model, often achieving a significant reduction in complexity with little to no loss in predictive performance.

---

### Stage 3 — Pruning (Simplify without losing accuracy)

After training, rules with **high uncertainty**, **low discriminative power**, or **tiny coverage** are dropped. You keep the signal, ditch the noise.

---

## 3 · What the Learned Rules Look Like

After DST training, each rule carries masses for classes and an uncertainty term:

```
Class 1: age < 58.0 & capital.gain > 5013.0 & marital.status == Married-civ-spouse  ||  mass=[0.007, 0.973, unc=0.020]
Class 1: capital.loss > 1848.0 & education == Bachelors & marital.status == Married-civ-spouse & occupation == Exec-managerial  ||  mass=[0.028, 0.881, unc=0.091]
Class 1: education == Bachelors & hours.per.week > 40.0 & marital.status == Married-civ-spouse & workclass == Private  ||  mass=[0.041, 0.789, unc=0.170]
Class 1: education == Prof-school & hours.per.week > 40.0 & marital.status == Married-civ-spouse & occupation == Prof-specialty  ||  mass=[0.066, 0.585, unc=0.350]
```

Format:
`Class c: <conjunctive conditions>  ||  mass=[m0, m1, ..., unc=<u>]`

---

## 4 · Systems Compared (legend used in plots)

* `r_raw` / `r_dst` — RIPPER rules as raw decision list vs with DST training
* `f_raw` / `f_dst` — FOIL rules raw vs with DST training
* `s_dst` — “STATIC” heuristic rules with DST training
* `dt` / `rf` / `gb` — Decision Tree / Random Forest / Gradient Boosting baselines

---

## 5 · Benchmarks (public datasets)

Each dataset has a **metrics** bar chart (`Accuracy`, `F1`, `Precision`, `Recall`). Files live under `Common_code/results/` with names:

```
benchmark_dataset_<dataset>_metrics.png
```

### Adult (UCI Income)

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_adult_metrics.png" width="85%">
</p>

**Takeaway:** DST materially improves F1/Recall vs raw rule lists and is competitive with tree baselines while staying interpretable.

### Bank Marketing

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_bank-full_metrics.png" width="85%">
</p>

**Takeaway:** The DST layer lifts the rule-based models substantially; the final results are balanced across metrics.

### Brain Tumor MRI

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_Brain%20Tumor_metrics.png" width="85%">
</p>

**Takeaway:** Very high recall/accuracy across systems; DST keeps performance strong on a high-signal medical set.

### Breast Cancer (Wisconsin)

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_breast-cancer-wisconsin_metrics.png" width="85%">
</p>

**Takeaway:** Already-strong baselines; DST maintains near-ceiling metrics with rule interpretability.

### Wine Quality (Red)

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_df_wine_metrics.png" width="85%">
</p>

**Takeaway:** Evidence combination stabilizes performance and lifts recall vs raw rules.

### German Credit

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_german_metrics.png" width="85%">
</p>

**Takeaway:** DST improves F1/Recall over raw rule lists; pruning (when applied) keeps the model compact.

---

## 6 · Repo layout (key files)

```
Common_code/
  ├─ test_Ripper_DST.py      # main entry: end-to-end pipeline
  ├─ DSClassifierMultiQ.py   # training loop, optimizer, pruning logic
  ├─ DSModelMultiQ.py        # rule activation + DST fusion forward pass
  ├─ Datasets_loader.py      # dataset loading utilities
  └─ results/                # PNG plots + CSV metrics
```

**Outputs:**

* Bar charts per dataset in `results/benchmark_dataset_<name>_metrics.png`
* (If enabled) CSVs with per-system metrics in `results/`

---

## 7 · Citing & Further Reading

If you build on this work, please cite:

* Peñafiel, S., Bleda, A. L., Candelas, F., & Sempere, J. M. (2020). *Applying the Dempster-Shafer theory for a flexible, accurate and interpretable classifier*. **Expert Systems with Applications**, 141, 112953.

Background:

* G. Shafer (1976). *A Mathematical Theory of Evidence*. Princeton University Press.
* W. W. Cohen (1995). *Fast Effective Rule Induction (RIPPER)*. ICML.
* J. R. Quinlan (1990). *Learning Logical Definitions from Relations (FOIL)*. *Machine Learning*, 5(3).

```

**Sources I checked (your repo & result images) to ground the README**: :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}
