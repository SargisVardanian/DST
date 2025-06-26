# -*- coding: utf-8 -*-
"""
Global configuration for experiments and rule-learning.

Keep only simple, import-safe constants here. Avoid heavy imports.
"""

# ----------------------------- Rule confidence tweaks -----------------------------
# If True, lower the rule confidence (mass) proportionally to its support.
LOWER_CONFIDENCE_BY_PROPORTION: bool = True

# Threshold (in standard deviations) for filtering outliers when estimating splits.
OUTLIER_THRESHOLD_NUM_STD: int = 2

# ----------------------------- DSClassifierMultiQ -----------------------------
# Maximum number of optimization epochs for the DST classifier.
max_iter: int = 500

# ----------------------------- DBSCAN grid-search -----------------------------
# Initial epsilon value for DBSCAN.
EPS: float = 0.01
# Step size for epsilon sweep.
STEP: float = 0.05
# Maximum epsilon boundary for the sweep.
MAX_EPS: float = 20.0
# Minimum number of samples to form a core point (tune based on density).
MIN_SAMPLES: int = 2
# Desired number of clusters (excluding noise) for early-stopping the sweep.
TARGET_CLUSTERS: int = 2

# ----------------------------- Dataset split -----------------------------
# Train split size (e.g., 0.7 means 70% train / 30% test).
train_set_size: float = 0.7

SEED = 42
