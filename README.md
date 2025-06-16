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

## 3 · Systematic Evaluation on Public Datasets

We evaluated our three-stage approach on six public datasets. The plots below show the test set performance and model complexity. In each figure:
*   **_raw**: The baseline model using only the initial RIPPER/FOIL decision list.
*   **_DST**: The model after training the Dempster-Shafer masses.
*   **_pruned**: The final, optimized model after pruning weak rules.

### 3.1 Adult (UCI Income)

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_adult_metrics.png" width="48%">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_adult_rules.png" width="48%">
</p>

*DST training dramatically boosts the F1-score and Recall. Pruning then halves the number of rules with negligible impact on performance, yielding a compact and accurate model.*

### 3.2 Bank-Marketing

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_bank-full_metrics.png" width="48%">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_bank-full_rules.png" width="48%">
</p>

*Once again, DST provides a significant F1-score improvement over the raw rule list. The final pruned model retains these gains while being substantially simpler.*

### 3.3 Brain-Tumour MRI

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_Brain%20Tumor_metrics.png" width="48%">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_Brain%20Tumor_rules.png" width="48%">
</p>

*Recall jumps significantly after DST training, a crucial improvement for medical diagnostics. Pruning creates a highly compact model of around 20 rules without sacrificing accuracy.*

### 3.4 Breast-Cancer (Wisconsin)

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_breast-cancer-wisconsin_metrics.png" width="48%">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_breast-cancer-wisconsin_rules.png" width="48%">
</p>

*While the baseline models are already strong on this dataset, DST closes the final gap to achieve a near-perfect F1-score, and pruning effectively removes redundant rules.*

### 3.5 Wine-Quality (Red)

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_df_wine_metrics.png" width="48%">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_df_wine_rules.png" width="48%">
</p>

*A clear success story for evidence combination: DST lifts a mediocre Recall of ~60% to over 95%, transforming an unreliable model into a highly effective one.*

### 3.6 German-Credit

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_german_metrics.png" width="48%">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_german_rules.png" width="48%">
</p>

*The F1-score improves by over 8 percentage points after DST training. The subsequent pruning trims about 30% of the rules without any loss in performance, demonstrating the effectiveness of the simplification stage.*

---

## 4 · Citing & Further Reading

If you use this work in your research, please cite the original paper:

*   Peñafiel, S., Bleda, A. L., Candelas, F., & Sempere, J. M. (2020). **Applying the Dempster-Shafer theory for a flexible, accurate and interpretable classifier**. *Expert Systems with Applications*, 141, 112953.

Additional foundational reading:
*   Shafer, G. (1976). **A Mathematical Theory of Evidence**. Princeton University Press.
*   Cohen, W. W. (1995). *Fast Effective Rule Induction*. In Proceedings of the 12th International Conference on Machine Learning (ICML).
*   Quinlan, J. R. (1990). *Learning logical definitions from relations*. Machine Learning, 5(3), 239-266.


