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
*   **Exploratory Data Analysis (EDA):** 
    *   Analyzed class imbalance (identifying `footsteps` as the dominant class).
    *   Performed feature correlation audits between Mel-Spectrograms and MFCCs (r=0.80).
    *   Utilized **t-SNE** manifold learning to visualize clusterability in 250+ feature dimensions.

### Phase 4: Multi-label Classification (Current)
Implementation of deep learning architectures to detect overlapping sounds.
*   **Architecture:** Investigating Convolutional Recurrent Neural Networks (CRNNs).
*   **Optimization:** Utilizing F1-Macro scores to address class imbalance identified in Phase 3.

### Phase 5: Production Deployment & Challenge (Roadmap)
Evaluating system performance on a non-public "secret" validation set to simulate real-world customer deployment.

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
