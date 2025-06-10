
# DST – Rule-Based Classification with RIPPER / FOIL + Dempster-Shafer Theory

This repository contains the official implementation for the paper **Applying the Dempster-Shafer theory for a flexible, accurate and interpretable classifier**, presented at ESWA. The project provides a framework for building interpretable machine learning models by combining classic rule-induction algorithms with Dempster-Shafer theory for robust evidence combination.

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_german_metrics.png" width="700" alt="Benchmark results on the German Credit dataset">
</p>

> • **Interpretable Models**: Generates simple IF-THEN rules using RIPPER or FOIL.  
> • **Powerful Combination**: Uses Dempster-Shafer Theory to intelligently combine evidence from multiple rules.  
> • **Optimized & Efficient**: Learns rule weights via gradient descent and prunes weak rules to create small, accurate models.

---

## 1 · Quick Start

Get started by cloning the repository and running the main benchmark script.

```bash
# 1. Clone the repository
git clone https://github.com/SargisVardanian/DST.git

# 2. Navigate to the directory
cd DST

# 3. Run the full experiment pipeline
python test_Ripper_DST.py
```

This script will automatically:
1.  Learn rule sets using both **RIPPER** and **FOIL**.
2.  Train the rules' Dempster-Shafer weights.
3.  Prune the models to simplify them.
4.  Generate performance plots and save metrics to CSV files in the `results/` folder.

---

## 2 · How It Works: From Data to Smart Rules

Our method turns data into an accurate and interpretable classifier in three main steps.

### Step 1: Learning Basic Rules (RIPPER vs. FOIL)

First, we automatically discover simple **IF-THEN rules** from the data. We provide two well-known algorithms for this, **RIPPER** and **FOIL**, which have a key philosophical difference:

*   **RIPPER: The Careful Builder**
    RIPPER works by "growing" a rule and then immediately "pruning" it. It adds conditions one by one (e.g., `age > 30`) to maximize information gain. After adding a condition, it checks if any existing conditions have become redundant and removes them. This careful, iterative process results in rules that are **general and robust** from the start.

*   **FOIL: The Fast Builder**
    FOIL is more aggressive. It also adds conditions one by one, but it doesn't have an immediate pruning step. It greedily adds the best condition it can find until the rule is highly precise. This can lead to **more specific and numerous** rules compared to RIPPER, as it might capture niche patterns in the data.

The output is a set of human-readable rules. For example:
`IF ('free sulfur dioxide' < 36) AND ('total sulfur dioxide' >= 110) THEN Class = 1`

At this stage, the model is a simple decision list: the first rule that matches a sample makes the prediction. As the benchmarks will show, this initial model is a good start but has limitations.

### Step 2: Getting Smarter with Dempster-Shafer Theory (DST)

A simple decision list is rigid. What if multiple rules apply? What if a rule is only right 80% of the time? This is where **Dempster-Shafer Theory (DST)** comes in.

Instead of a simple "yes/no" prediction, we give each rule a "vote" with learnable weights. For each rule $R_i$, we assign a mass vector $\mathbf{m}^{(i)}$ that distributes its belief across the possible classes and, crucially, includes a term for **uncertainty**:

$$
\mathbf m^{(i)}= \bigl(m_1^{(i)}, \dots, m_K^{(i)}, m_{\text{unc}}^{(i)}\bigr), \qquad \sum_{j=1}^K m_j^{(i)} + m_{\text{unc}}^{(i)} = 1.
$$

When a new data sample arrives, all matching rules "cast their votes." We combine these votes using Dempster's rule of combination. In our implementation, we do this efficiently in the **commonality space**, where evidence combination is a simple product:

$$
q_k(\mathbf x) = \prod_{i \in \mathcal{R}(\mathbf{x})} \left( m_k^{(i)} + m_{\text{unc}}^{(i)} \right)
$$

The key innovation is that these mass vectors $\mathbf{m}^{(i)}$ are **trained like a neural network**. We use the Adam optimizer to adjust the weights, minimizing the classification error on the training data. This process teaches the model:
*   **Which rules to trust**: Rules that are consistently correct get their uncertainty mass $m_{\text{unc}}$ lowered and their class-specific mass raised.
*   **Which rules are less reliable**: Ambiguous or often-incorrect rules learn to assign more mass to uncertainty, reducing their impact on the final prediction.

This DST training step makes the model far more flexible and accurate, as it learns to weigh and combine evidence from many rules instead of relying on just one.

### Step 3: Cleaning Up with Pruning

After training, our model may contain dozens of rules. Some of these might be redundant or not very useful. The final step is to **prune** the model to make it smaller, faster, and even easier to understand.

A rule is removed if it meets any of these criteria:
1.  **Too Uncertain**: Its uncertainty mass $m_{\text{unc}}^{(i)}$ is very high (e.g., > 0.6). The model has learned that this rule isn't trustworthy.
2.  **Too Ambiguous**: It doesn't strongly distinguish between classes. We measure this with the ratio of its belief in the top class versus the second-best class.
3.  **Too Specific**: It covers very few samples in the training data (e.g., < 10).

This pruning process intelligently filters out the least effective rules, often resulting in a significantly smaller model with almost no loss in performance.

---

## 3 · Benchmark Results

We evaluated our approach on six public datasets. The plots below show the test set performance for each. In each plot:
*   **\_raw**: The initial model using only the generated RIPPER/FOIL rules.
*   **\_DST**: The model after training the Dempster-Shafer weights.
*   **\_pruned**: The final, optimized model after pruning weak rules.

### 6.1 Adult (UCI Income)

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_adult_metrics.png" width="48%">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_adult_rules.png" width="48%">
</p>

*DST training dramatically boosts F1-score and Recall. Pruning halves the number of rules with almost no performance loss.*

### 6.2 Bank-Marketing

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_bank-full_metrics.png" width="48%">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_bank-full_rules.png" width="48%">
</p>

*Again, DST provides a significant F1-score improvement. The pruned model is much simpler and retains the performance gains.*

### 6.3 Brain-Tumour MRI

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_Brain%20Tumor_metrics.png" width="48%">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_Brain%20Tumor_rules.png" width="48%">
</p>

*Recall jumps significantly with DST, while precision remains high. Pruning creates a very compact model with around 20 rules.*

### 6.4 Breast-Cancer (Wisconsin)

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_breast-cancer-wisconsin_metrics.png" width="48%">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_breast-cancer-wisconsin_rules.png" width="48%">
</p>

*The baseline models are already strong, but DST closes the final gap to achieve a near-perfect F1-score.*

### 6.5 Wine-Quality (Red)

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_df_wine_metrics.png" width="48%">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_df_wine_rules.png" width="48%">
</p>

*A clear success story: DST lifts recall from around 60% to over 95%, transforming a mediocre model into a highly effective one.*

### 6.6 German-Credit

<p align="center">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_german_metrics.png" width="48%">
  <img src="https://github.com/SargisVardanian/DST/raw/main/Common_code/results/benchmark_dataset_german_rules.png" width="48%">
</p>

*The F1-score improves by ~8 points after DST training, and pruning trims about 30% of the rules without sacrificing accuracy.*

---

## 4 · Citing & Further Reading

If you use this work, please cite the original paper:

*   S. Peñafiel et al. — **Applying the Dempster-Shafer theory for a flexible, accurate and interpretable classifier**. *Expert Systems with Applications* (2020).

Additional foundational reading:
*   G. Shafer — **A Mathematical Theory of Evidence**. Princeton U Press (1976).
*   W. Cohen — *Fast Effective Rule Induction*. ICML (1995).
*   R. Quinlan — *FOIL: A Midterm Report*. ICML tutorial (1990).
