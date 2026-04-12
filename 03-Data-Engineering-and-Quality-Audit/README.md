# Phase 3: Technical Audit & Label Engineering Pipeline

## 📌 Executive Summary
This module focuses on the **Quality Assurance (QA)** and **Feature Engineering** phase of the SED pipeline. Before moving to model training, this stage was critical for resolving inter-annotator disagreement, identifying acoustic biases, and generating a mathematically consistent ground-truth label set.

## 🛠️ Technical Implementations

### 1. Inter-Annotator Reliability Audit
We quantified the consistency of crowdsourced human labels using the **Jaccard Similarity (Intersection over Union)** metric. 
*   **Metric:** $J = \frac{\text{Agreement}}{\text{Total Annotations}}$
*   **Result:** Achieved a mean reliability score of **71.0%**.
*   **Observation:** Continuous acoustic events (e.g., *Vacuuming*) showed significantly higher stability compared to transient, impulsive sounds (e.g., *Light switch*), informing the need for class-specific loss-weighting in Phase 4.

### 2. Label Derivation Engine
Developed a data-cleaning pipeline to collapse multi-annotator overlap data into unified binary prediction targets.
*   **Aggregation:** Calculated the arithmetic mean of proportional overlap across all annotators per 1-second segment.
*   **Binarization:** Implemented a **Majority-Rule Threshold (0.5)** to generate the final ground-truth matrix.
*   **Audit Result:** Cross-validated these derived labels against original collector metadata, yielding a **94.3% tag confirmation rate**.

### 3. Acoustic Feature Analysis
Performed a statistical audit of the precomputed feature space (250+ dimensions) to optimize the future training pipeline.
*   **Redundancy Check:** Identified a high Pearson correlation (**r=0.80**) between Mel-Spectrograms and MFCCs, suggesting potential for dimensionality reduction.
*   **Scale Normalization:** Calculated feature ranges (e.g., *Signal Power* $\approx$ 11,000 vs. *ZCR* $\approx$ 0) to establish standardization parameters for gradient-based learning.

### 4. Manifold Learning (t-SNE)
Utilized **t-SNE** (pre-reduced via PCA to 50D) to visualize the clusterability of high-dimensional acoustic signatures.
*   **Findings:** Proved that spectral features are driven by the source event rather than the environment, validating the dataset's potential for high cross-room generalization.

## 📊 Key Artifacts Generated
*   `inter_annotator_distribution.png`: Histogram of dataset consistency.
*   `class_wise_reliability.png`: Analysis of which sound types were hardest to agree on.
*   `label_derivation_piano_roll.png`: A "Before vs. After" visualization of the binarization process.
*   `feature_space_clustering.png`: 2D projection of the spectral characteristics.

---

## 🚀 Impact on Phase 4 (Classification)
The insights from this audit directly dictate the architecture of the upcoming classification models:
1.  **Metric Selection:** Shifted from Accuracy to **F1-Macro** to account for the class imbalance found in the frequency audit.
2.  **Temporal Resolution:** Identified "temporal shrinking" in short sounds, requiring the model to utilize finer-grained attention mechanisms or data augmentation for transient classes.
