# Phase 05: Production Deployment & Challenge

## 📌 Module Overview
The final phase of the project marks the transition from segment-level classification to full **Sound Event Detection (SED)**. The primary objective was to build a system capable of predicting the precise **onset and offset timestamps** for 15 domestic sound classes in continuous audio recordings. This module documents the high-dimensional optimization of an ensemble architecture and the implementation of a temporal refinement engine to ensure production-grade stability.

## 🛠️ Technical Implementations

### 1. Baseline Reproduction & Gap Analysis
*   **System:** 15 independent Decision Trees (Binary Relevance).
*   **Methodology:** Audio processed in 1s windows with a 0.5s hop size.
*   **Split Consistency:** As shown in the comparison below, the model demonstrated high generalization consistency between the validation and non-hidden test sets.
*   **Recall Challenges:** The detailed metrics reveal a "Recall Gap"—while the baseline maintains decent precision, it fails to capture transient events like `light_switch` and `window_open_close`, which are often diluted by silence in the 1s windows.

| Split Consistency | Detailed Baseline Metrics |
| :--- | :--- |
| ![Baseline F1](visuals/baseline_f1_scores.png) | ![Baseline Table](visuals/baseline_table.png) |

### 2. High-Dimensional Optimization ("Smart Limit" RF)
With the feature space expanded to **960 dimensions**, I moved away from the rigid depth limits used in Phase 4 to a regularization strategy based on evidence density:
*   **Regularization:** Used `min_samples_leaf=10` with `max_depth=None`. This allows trees to grow deeply where patterns are clear while automatically pruning branches with insufficient statistical support.
*   **Efficiency:** A sweep of `n_estimators` proved that 50 trees reached the performance ceiling (**Macro $F_1 = 0.4636$**) before hitting diminishing returns.

![RF Tuning](visuals/rf_hyperparameter_table.png)

### 3. Temporal Refinement (Median Filtering)
Qualitative analysis identified "salt-and-pepper" noise in raw predictions, where continuous events suffered from 1-second dropouts (flickering). I implemented an edge-preserving **Median Filter (Window=3)** to stabilize the output trajectories.
*   **The Engineering Trade-off:** The Delta Plot below highlights a critical finding: while the filter successfully stabilized continuous sounds like `keyboard_typing` (Green), it acted as a destructive low-pass filter for transient sounds (Red), erasing genuine short-duration detections.

![Median Filter Impact](visuals/f1_tradeoff_barchart.png)

## 📊 Qualitative Error Analysis (Case Studies)
To audit model behavior beyond the numbers, I generated high-resolution visualizations of complex kitchen scenes, aligning Mel-Spectrograms with Ground Truth and model outputs.

| File Audit | Visualization | Key Insight |
| :--- | :--- | :--- |
| **Case Study 1** | ![Spec1](visuals/rf_qualitative_analysis_1.png) | Successfully captured the `phone_ringing` sequence but identified temporal "flickering" in the `keychain` class. |
| **Case Study 2** | ![Spec2](visuals/rf_qualitative_analysis_2.png) | Revealed spectral confusion between `microwave` and `coffee_machine` hums due to overlapping steady-state frequencies. |

## 📈 Final Performance Summary

| Model Configuration | Macro $F_1$ Score |
| :--- | :--- |
| Decision Tree Baseline | 0.3130 |
| Optimized Random Forest (Raw) | **0.4721** |
| **Optimized RF + Median Filter (Final)** | **0.4668** |

## 🔍 Real-World Reflection & Roadmap
*   **Edge Constraints:** Passing 960 features per second through 50 deep trees is computationally heavy. Production deployment on smart home hardware would likely require model quantization or distillation.
*   **Stability vs. Sensitivity:** The Median Filter introduces a mandatory processing delay. In safety-critical scenarios (e.g., detecting a breaking window), this 1-second latency must be carefully weighed against the benefit of reduced false alarms.
*   **The Next Step:** This project proves that while segment-level features are strong, SED requires native temporal modeling. The next architectural iteration will move toward **Convolutional Recurrent Neural Networks (CRNNs)** to learn sound dynamics end-to-end.
