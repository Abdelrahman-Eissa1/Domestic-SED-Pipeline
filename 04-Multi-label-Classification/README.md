# Phase 04: Multi-label Classification

## 📌 Module Overview
This module implements the core machine learning pipeline for detecting 15 overlapping domestic sound events. By moving from data engineering to predictive modeling, I developed and compared two distinct architectures—**Random Forest (RF)** and **Multi-Layer Perceptron (MLP)**—to establish a robust Sound Event Detection (SED) system.

## 🛠️ Technical Implementations

### 1. Leakage-Free Data Splitting
To ensure model generalization and prevent "room-acoustic memorization," I implemented a **Collector-Level Split**:
*   **Strategy:** Data was split into **70% Train / 15% Val / 15% Test** based on the `collector_id`.
*   **Stratification:** Collectors were binned by contribution size (small/medium/large) before splitting to maintain consistent class distributions across all sets.
*   **Result:** A dataset of ~168,000 segments with consistent long-tail imbalance handling.

### 2. Preprocessing & Feature Selection
*   **Normalization:** Applied Z-score normalization (StandardScaler) fit exclusively on the training set to prevent data leakage.
*   **Feature Set:** Retained 107 dimensions, including MFCCs (mean, delta, delta-delta), Spectral Centroid, Flux, and Energy.
*   **Redundancy Audit:** Mel-Spectrograms were excluded due to a high Pearson correlation ($r=0.80$) with MFCCs, reducing feature inflation.

### 3. Model Architectures & Hyperparameter Tuning
Both models were wrapped in a `MultiOutputClassifier` to handle the multi-label nature of the task.

| Model | Hyperparameters Swept | Best Configuration |
| :--- | :--- | :--- |
| **Random Forest** | `n_estimators`, `max_depth` | `n=100`, `max_depth=20` |
| **MLP** | `hidden_layer_sizes`, `learning_rate` | `(128, 64)`, `lr=0.001` |

*   **RF Insight:** Fully grown trees (`max_depth=None`) led to severe overfitting (F1 ~0.18), while a depth of 20 provided the best generalization.
*   **MLP Insight:** Smaller, stable learning rates were required for convergence on this noisy dataset; larger architectures yielded diminishing returns.

### 4. Final Evaluation & Baselines
The models were evaluated against a **Random Sampling Baseline** using **Macro F1-Score** to account for heavy class imbalance.

| Model | Test Macro F1 |
| :--- | :--- |
| Random Sampling Baseline | 0.0513 |
| Multi-Layer Perceptron | 0.3576 |
| **Random Forest** | **0.4019** |

## 📊 Technical Visualizations

| Metric | Visualization |
| :--- | :--- |
| **Class Distribution** | ![Class Distribution](visuals/class_distribution_splits.png) |
| **RF Tuning** | ![RF Sweep](visuals/rf_sweep.png) |
| **Model Comparison** | ![Final Comparison](visuals/final_comparison.png) |
| **Case Study** | ![Case Study](visuals/case_study.png) |

## 🔍 Key Findings & Reflection
*   **Top Performers:** Distinctive continuous sounds like `running_water` (F1=0.71) and `vacuum_cleaner` (F1=0.64) were detected with high reliability.
*   **Hard Classes:** Short, transient sounds like `light_switch` (F1=0.14) remain a challenge due to their brief acoustic footprint.
*   **Failure Modes:** The models struggle with "Acoustic Hallucination" (false positives triggered by background noise like dishwashers) and overlapping continuous sounds.
*   **Future Work:** Transitioning to sequential models (RNN/Transformer) to leverage temporal context instead of segment-level classification.
