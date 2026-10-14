# Toward a Security Score for 6G Vehicular Networks: Concept and a Simulated Spoofing-Detection Pilot (NAGT-Score)

**Official Implementation Repository**

This repository accompanies the research paper:

**“Toward a Security Score for 6G Vehicular Networks: Concept and a Simulated Spoofing-Detection Pilot (NAGT-Score)”**

It provides the **reference implementation, corrected experimental results, and reproducibility materials** for a conceptual security-scoring framework and a simulated Vehicle-to-Everything (V2X) spoofing-detection pilot.

**Important Note:** NAGT-Score represents a **proposed security-scoring concept and a simulated detection study**. It is **not a validated operational 6G security metric**.

---

## 1. Overview

This research introduces two interconnected components designed to investigate security assessment and spoofing detection in future vehicular communication environments.

### 1.1. Conceptual Security Score

The proposed framework combines four security-related dimensions:

- **Reliability (ρ):** Reliability of communication and system operation.
- **Resilience (Res):** Ability to withstand and recover from disturbances.
- **Propagation Containment (π):** Capacity to limit the spread of security threats.
- **Response Capability (δ):** Ability to identify and respond to security events.

### 1.2. Graph-Temporal Spoofing Detection Framework

The detection architecture integrates:

- **Noise-Aware Feature Weighting:** Adjusts feature contributions using estimated message noise.
- **Graph Attention Network (GAT):** Models relationships between interacting vehicles.
- **Transformer Sequence Encoding:** Captures temporal dependencies in vehicle message sequences.
- **Multi-Task Prediction Heads:** Supports multiple security-related prediction tasks.

The framework is evaluated using a **simulated V2X benchmark dataset**.

---

## 2. Main Contributions

The repository provides:

- **Reverse-Mode Automatic Differentiation Engine:** A NumPy-based implementation supporting gradient computation.
- **Complete NAGT-Score Implementation:** The proposed graph-temporal, noise-aware, multi-task learning architecture.
- **Four Baseline Models:**
  - LSTM
  - Transformer
  - GNN
  - GNN + Transformer
- **Corrected Five-Seed Experimental Results:** Updated evaluation outputs following computational graph corrections.
- **Supplementary Feature-Dependence Analysis:** Investigation of the influence of dataset-specific input features.
- **Reproducibility Verification:** Repeatability and gradient-checking procedures performed after correcting computational graph ordering.

---

## 3. NAGT-Score Architecture

The proposed architecture consists of four main modules.

### 3.1. Noise-Aware Feature Gate

- Applies a **learnable feature-weighting mechanism**.
- Uses estimated message noise to adjust feature importance.
- Generates weighted feature representations before downstream processing.

### 3.2. Graph Attention Encoder

- Represents vehicle interactions through **communication graphs**.
- Constructs relationships using sender–receiver message exchanges.
- Applies **graph attention** to learn interaction-aware representations.

### 3.3. Transformer Sequence Encoder

- Processes **per-vehicle communication sequences**.
- Uses self-attention to model temporal relationships.
- Extracts sequence-level representations for downstream predictions.

### 3.4. Multi-Task Prediction Heads

The shared representation supports four prediction tasks:

- **Anomaly Detection:** Identification of potentially spoofed messages.
- **Degradation Prediction:** Estimation of degradation-related indicators.
- **Propagation-State Prediction:** Prediction of security propagation states.
- **Timing Estimation:** Estimation of timing-related outcomes.

---

## 4. Dataset

Experiments are conducted using the **NAGT-V2X Benchmark Dataset (NVBD)**.

### 4.1. Dataset Characteristics

- **Dataset Type:** Simulated VeReMi-style V2X dataset
- **Driving Environment:** Highway scenario
- **Attack Type:** Constant-Position-Offset spoofing
- **Record Type:** Message-level communication observations

### 4.2. Dataset Statistics

| **Dataset Attribute** | **Value** |
|---|---|
| **Total Messages** | 99,937 |
| **Total Columns** | 52 |
| **Unique Sender IDs** | 841 |
| **Scenario** | Simulated highway |
| **Attack Category** | Constant-Position-Offset spoofing |

### 4.3. Dataset Availability

**Dataset DOI:** https://doi.org/10.5281/zenodo.23278351

**Note:** The dataset is **not included directly in this repository**. It must be downloaded separately before running the experiments.

---

## 5. Requirements

The implementation requires the following Python packages:

- **NumPy**
- **Pandas**
- **Scikit-learn**

### Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

---

## 6. Repository Structure

The repository contains the following implementation and evaluation files:

| **File / Directory** | **Description** |
|---|---|
| **autograd.py** | Reverse-mode automatic differentiation engine |
| **models.py** | NAGT-Score architecture and baseline implementations |
| **data_prep.py** | Dataset loading and sequence preparation |
| **main.py** | Model training and evaluation pipeline |
| **summarize_results.py** | Multi-seed performance summarization |
| **check_eval_subset.py** | Evaluation subset verification |
| **supplementary_gb.py** | Supplementary feature-dependence analysis |
| **test_autograd.py** | Automatic differentiation gradient checks |
| **results_corrected/** | Corrected experimental results used in the manuscript |
| **requirements.txt** | Required Python packages |
| **LICENSE** | MIT License |

---

## 7. Running Experiments

### 7.1. Single-Seed Experiment

Run an experiment using a single random seed:

```bash
DATA_PATH=/path/to/dataset.csv EPOCHS=15 SEEDS=0 python3 main.py
```

### 7.2. Five-Seed Experiment

Run the corrected experimental pipeline using five seeds:

```bash
DATA_PATH=/path/to/dataset.csv SEEDS=0,1,2,3,4 EPOCHS=15 python3 main.py
```

**Experimental Settings:**

- **Training Epochs:** 15
- **Random Seeds:** 0, 1, 2, 3, 4
- **Dataset Path:** User-specified CSV location

---

## 8. Corrected Experimental Configurations

The final manuscript reports three corrected feature configurations.

### 8.1. Table X: All Features

**Configuration:**

```bash
DROP_LEAKY=0
DROP_ROADEDGE=0
```

- Includes all available input features.
- Serves as the full-feature reference experiment.

### 8.2. Table XI: Without Label-Derived Feature

**Configuration:**

```bash
DROP_LEAKY=1
DROP_ROADEDGE=0
```

**Removed Feature:**

- `susceptible_previous_window`

This experiment assesses model performance after removing a label-derived feature.

### 8.3. Table XII: Without Label-Derived and Road-Edge Features

**Configuration:**

```bash
DROP_LEAKY=1
DROP_ROADEDGE=1
```

**Removed Features:**

- `susceptible_previous_window`
- `sender_distance_to_road_edge`

This configuration investigates model behavior after excluding two potentially influential dataset-specific features.

---

## 9. Key Experimental Findings

The corrected evaluation produced several important observations:

- **Transformer Performance:** The Transformer baseline achieved the highest mean anomaly AUC and F1 score in the full-feature experiment.
- **NAGT-Score Performance:** The proposed NAGT-Score model demonstrated competitive predictive performance.
- **Feature Sensitivity:** Removing dataset-specific features substantially reduced anomaly-detection performance.
- **Scenario Dependence:** Detection performance was strongly influenced by characteristics of the simulated spoofing environment.
- **Comparative Interpretation:** The findings do not establish universal superiority of NAGT-Score over alternative architectures.

**Interpretation:** The reported outcomes are specific to the evaluated **simulated highway spoofing scenario** and should not be generalized to unseen attacks or operational 6G deployments without further validation.

---

## 10. Reproducibility Audit

A reproducibility audit identified and corrected a computational graph ordering issue in the original automatic differentiation engine.

### 10.1. Identified Issue

- The original implementation stored computational graph parents using an **unordered structure**.
- This could affect the required ordering of computational dependencies during reverse-mode differentiation.

### 10.2. Implemented Correction

- Unordered parent storage was replaced with **ordered computational graph dependencies**.
- The experimental results were regenerated using the corrected implementation.

### 10.3. Verification Procedures

The corrected implementation was evaluated through:

- **Four Gradient Checks:** Verification of automatic differentiation behavior.
- **Independent LSTM Repeatability Test:** Repeated testing using random seed 0.
- **Five-Seed Experiment Regeneration:** Reproduction of experimental outputs after applying the correction.

**Reproducibility Note:** The study does not claim **bitwise reproducibility across different computing platforms**.

---

## 11. Supplementary Feature-Dependence Analysis

Additional analysis was performed to investigate the dependence of anomaly-detection performance on individual dataset features.

### Principal Observations

- **Road-Edge Distance:** The feature `sender_distance_to_road_edge` contains a strong detectable signal associated with the simulated spoofing scenario.
- **Performance Reduction:** Removing this feature substantially decreases anomaly-detection performance.
- **Dataset-Specific Dependencies:** The benchmark contains scenario-specific patterns that influence model predictions.
- **Generalization Concerns:** Strong performance on the evaluated benchmark does not necessarily indicate equivalent performance under different attacks, traffic environments, or real-world conditions.

**Important Interpretation:** The reported results should not be interpreted as evidence of generalized V2X security effectiveness across previously unseen spoofing strategies.

---

## 12. Study Limitations

### 12.1. Scope of the Proposed Framework

The present study does **not** establish:

- A validated operational **6G security measurement framework**.
- Readiness for real-world vehicular network deployment.
- Validation of physical safety consequences.
- Generalization across multiple attack strategies or environments.
- Universal predictive superiority over competing architectures.

### 12.2. Experimental Limitations

The current evaluation is constrained by:

- **Single Attack Type:** Only Constant-Position-Offset spoofing is considered.
- **Single Driving Environment:** Experiments focus on one simulated highway scenario.
- **No Physical Safety Outcomes:** Real-world safety consequences are not evaluated.
- **Non-Vehicle-Disjoint Data Splits:** Evaluation does not ensure separation of vehicles across training and testing partitions.
- **Proxy-Based Security Indicators:** Security dimensions are represented through indirect indicators rather than validated operational measurements.

These limitations should be considered when interpreting the findings and determining directions for future research.

---

## 13. Citation

If you use the implementation, experimental materials, or conceptual framework provided in this repository, please cite the accompanying paper:

**Toward a Security Score for 6G Vehicular Networks: Concept and a Simulated Spoofing-Detection Pilot (NAGT-Score)**

---

## 14. License

This project is distributed under the **MIT License**.

For additional licensing information, refer to the repository's **LICENSE** file.
