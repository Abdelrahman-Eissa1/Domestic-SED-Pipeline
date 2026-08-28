# 🔊 Domestic Sound Event Detection (SED) Pipeline
**An End-to-End System for Smart Home Acoustic Monitoring**

This repository documents the development of a robust Sound Event Detection (SED) system for **Kepler Intelligent Audio Labs (KIAL)**. The project identifies 15 distinct domestic sound classes to enable context-aware automation, home safety, and resource awareness.

---

## 📌 Project Vision
The goal was to develop an AI-driven system capable of analyzing continuous audio streams to predict acoustic events of interest, including their precise **temporal onsets and offsets**.

### Target Sound Classes (15):
*   **Infrastructure:** Door open/close, Window open/close, Light switch.
*   **Appliances:** Vacuum cleaner, Microwave, Coffee machine, Dish handling.
*   **Safety/Alerts:** Phone ringing, Bell ringing, Toilet flushing, Running water.
*   **Activity:** Footsteps, Keyboard typing, Keychain jingle.

---

## 🏗️ Development Stages

The project was executed in five iterative phases, moving from data engineering to advanced temporal modeling.

### Phase 1: Crowdsourced Data Collection (Completed)
Curated a large-scale domestic audio dataset using diverse hardware and realistic environments. 
*   **Volume:** 3,600+ unique recordings (15s to 35s).

### Phase 2: Human-in-the-loop Annotation (Completed)
Established a temporal labeling pipeline using **Label Studio** where human annotators marked onsets and offsets for all target classes.

### Phase 3: Data Quality Audit & Feature Engineering (Completed)
*   **Reliability:** Quantified annotator consensus using **Jaccard Similarity (IoU)**, achieving 71.0% mean agreement.
*   **Label Engineering:** Implemented a **Mean-Thresholding engine** to derive binary ground-truth targets.

### Phase 4: Multi-label Classification (Completed)
Implemented segment-level multi-label architectures.
*   **Benchmarking:** Compared **Random Forest (RF)** vs. **Multi-Layer Perceptron (MLP)**.
*   **Optimization:** Achieved a peak Macro $F_1$ of **0.3620** (RF).

### Phase 5: Production Deployment & Challenge (Completed)
Transitioned to full **Sound Event Detection (SED)** and optimized for 960-dimensional feature spaces.
*   **Regularization:** Implemented a "Smart Limit" RF strategy to prevent overfitting.
*   **Temporal Refinement:** Developed a **Median Filtering** engine to resolve prediction flickering.
*   **Final Result:** Achieved a final optimized Macro $F_1$ of **0.4668**, a ~50% improvement over the baseline.

---

## 🛠️ Tech Stack
*   **Language:** Python 3.11+
*   **Data Science:** NumPy, Pandas, Scikit-learn, Joblib
*   **Audio Processing:** Librosa, Scipy (Signal)
*   **Visualization:** Matplotlib, Seaborn

---

## 🔐 Licensing & Data Privacy
**Note:** The raw audio dataset and feature files used in this project are proprietary and are not included in this repository due to university redistribution restrictions. The repository contains the source code, pipeline logic, and technical results only.
