# 🔊 Domestic Sound Event Detection (SED) Pipeline
**An End-to-End System for Smart Home Acoustic Monitoring**

This repository documents the development of a robust Sound Event Detection (SED) system for **Kepler Intelligent Audio Labs (KIAL)**. The project focuses on identifying 15 distinct domestic sound classes to enable context-aware automation, home safety, and resource awareness.

---

## 📌 Project Vision
The goal is to develop an AI-driven system capable of analyzing raw audio streams to predict acoustic events of interest, including their precise temporal onsets and offsets. 

### Target Sound Classes (15):
*   **Infrastructure:** Door open/close, Window open/close, Light switch.
*   **Appliances:** Vacuum cleaner, Microwave, Coffee machine, Dish handling.
*   **Safety/Alerts:** Phone ringing, Bell ringing, Toilet flushing, Running water.
*   **Activity:** Footsteps, Keyboard typing, Keychain jingle.

---

## 🏗️ Development Stages

The project is structured into five iterative phases. Each phase builds upon the data engineering established in the previous steps.

### Phase 1: Crowdsourced Data Collection (Completed)
Curated a large-scale domestic audio dataset using diverse recording hardware (90% iOS / 10% Android) and realistic domestic environments (Kitchen, Bedroom, Hallway).
*   **Diversity focus:** Captured audio across varied acoustic conditions to ensure model generalization.
*   **Volume:** 3,600+ unique recordings ranging from 15s to 35s.

### Phase 2: Human-in-the-loop Annotation (Completed)
Established a temporal labeling pipeline using **Label Studio**. 
*   **Process:** Human annotators marked onsets and offsets for all target classes.
*   **Verification:** Implemented a verification stage to correct temporal boundary errors and ensure adherence to the "one-second pause" rule.

### Phase 3: Data Quality Audit & Feature Engineering (Completed)
Developed a preprocessing and validation pipeline to ensure the dataset is model-ready.
*   **Reliability Metrics:** Quantified annotator consensus using **Jaccard Similarity (IoU)**, achieving a mean agreement of **71.0%**.
*   **Label Engineering:** Developed a **Mean-Thresholding engine** to derive binary ground-truth targets from multi-annotator overlap data.
*   **Exploratory Data Analysis (EDA):** Analyzed class imbalance, feature correlation ($r=0.80$), and clusterability via **t-SNE**.

### Phase 4: Multi-label Classification (Completed)
Implemented and optimized machine learning architectures to detect overlapping sounds.
*   **Leakage Prevention:** Applied a **Collector-Level Split** to ensure models generalize to new users/environments.
*   **Model Comparison:** Evaluated **Random Forest (RF)** vs. **Multi-Layer Perceptron (MLP)** using a `MultiOutputClassifier` wrapper.
*   **Optimization:** Conducted hyperparameter sweeps on tree depth and network architecture, using **Macro F1-Score** to handle the heavy long-tail class imbalance.
*   **Benchmark:** Achieved a peak Macro F1 of **0.3620** (RF), significantly outperforming the Random Sampling Baseline (0.0544).

### Phase 5: Production Deployment & Challenge (Current/Roadmap)
Evaluation of system performance on a non-public "secret" validation set to simulate real-world customer deployment and "Black-Box" testing.

---

## 🛠️ Tech Stack
*   **Language:** Python 3.11+
*   **Data Science:** NumPy, Pandas, Scikit-learn
*   **Audio Processing:** Librosa
*   **Visualization:** Matplotlib, Seaborn
*   **Annotation:** Label Studio

---

## 🔐 Licensing & Data Privacy
**Note:** The raw audio dataset and feature files used in this project are proprietary and are not included in this repository due to university redistribution restrictions. The repository contains the source code, pipeline logic, and technical results only.
