

Toward a Security Score for 6G Vehicular Networks: Concept and a Simulated Spoofing-Detection Pilot (NAGT-Score)
Official implementation repository accompanying the paper:

"Toward a Security Score for 6G Vehicular Networks: Concept and a Simulated Spoofing-Detection Pilot (NAGT-Score)"

This repository provides the reference implementation, corrected experiments, and reproducibility materials for a conceptual security scoring framework and a simulated V2X spoofing-detection pilot.

Important note: NAGT-Score is a proposed security-scoring concept and a simulated detection study. It is not a validated operational 6G security metric.

Overview
This work introduces two related components:

A conceptual security score combining four dimensions:
Reliability (ρ)
Resilience (Res)
Propagation containment (π)
Response capability (δ)
A graph-temporal spoofing detection framework that combines:
Noise-aware feature weighting
Graph Attention Network (GAT)
Transformer sequence encoding
Multi-task prediction heads
The framework is evaluated on a simulated V2X benchmark dataset.

Main Contributions
This repository contains:

A NumPy-based reverse-mode automatic differentiation engine.
The complete NAGT-Score implementation.
Four baseline implementations:
LSTM
Transformer
GNN
GNN+Transformer
Corrected five-seed experimental results.
Supplementary feature-dependence analysis.
Reproducibility verification after correcting the computational graph ordering.
NAGT-Score Architecture
Noise-aware Feature Gate
A learnable gate uses message noise estimates to weight input features before model processing.

Graph Attention Encoder
Vehicle communication relationships are represented as interaction graphs using sender-receiver message exchanges.

Transformer Sequence Encoder
Per-vehicle message sequences are processed using self-attention.

Multi-task Heads
The shared representation predicts:

Anomaly detection
Degradation prediction
Propagation-state prediction
Timing estimation
Dataset
Experiments use the:

NAGT-V2X Benchmark Dataset (NVBD)

Dataset characteristics:

Simulated VeReMi-style V2X dataset
Highway scenario
Constant-Position-Offset spoofing attack
Message-level communication records
Statistics:

99,937 messages
52 columns
841 sender IDs
Dataset download:

https://doi.org/10.5281/zenodo.23278351

The dataset is not included in this repository.

Requirements
numpy
pandas
scikit-learn
Install:

pip install -r requirements.txt
Repository Structure
autograd.py              Reverse-mode autodiff engine
models.py                NAGT-Score and baseline models
data_prep.py             Dataset loading and sequence construction
main.py                  Training and evaluation pipeline
summarize_results.py     Multi-seed result summary
check_eval_subset.py     Evaluation subset verification
supplementary_gb.py      Feature dependence analysis
test_autograd.py         Gradient checking
results_corrected/       Corrected manuscript results
requirements.txt         Python dependencies
LICENSE                  MIT License
Running Experiments
Single seed:

DATA_PATH=/path/to/dataset.csv EPOCHS=15 SEEDS=0 python3 main.py
Five seeds:

DATA_PATH=/path/to/dataset.csv SEEDS=0,1,2,3,4 EPOCHS=15 python3 main.py
Corrected Experiments
The final manuscript reports three corrected configurations:

Table X: All Features
DROP_LEAKY=0
DROP_ROADEDGE=0
Table XI: Without Label-derived Feature
DROP_LEAKY=1
DROP_ROADEDGE=0
Removed feature:

susceptible_previous_window
Table XII: Without Label-derived and Road-edge Features
DROP_LEAKY=1
DROP_ROADEDGE=1
Removed features:

susceptible_previous_window
sender_distance_to_road_edge
Key Findings
The corrected experiments show:

Transformer achieved the highest mean anomaly AUC and F1 in the full-feature experiment.
NAGT-Score achieved competitive performance.
Removing dataset-specific features significantly reduced anomaly detection performance.
Results are specific to the evaluated simulated spoofing scenario.
No claim is made that NAGT-Score universally outperforms all other architectures.

Reproducibility Audit
The original automatic differentiation engine used unordered computational graph parents.

The implementation was corrected by replacing unordered parent storage with ordered computation graph dependencies.

Verification included:

Four gradient checks.
Independent seed-0 LSTM repeatability testing.
Corrected five-seed experiment regeneration.
Cross-platform bitwise reproducibility is not claimed.

Feature Dependence Analysis
Supplementary analysis investigated feature contributions.

Important observations:

sender_distance_to_road_edge contains strong detectable spoofing signal.
Removing this feature reduces performance substantially.
The simulated dataset contains strong scenario-specific cues.
Therefore, results should not be interpreted as general V2X security performance across unseen attacks.

Limitations
This study does not claim:

validated 6G security measurement
operational deployment readiness
physical safety validation
generalization to other attacks or scenarios
universal model superiority
The current evaluation is limited by:

one simulated attack type
one highway scenario
absence of physical safety outcomes
non-vehicle-disjoint splits
proxy-based security indicators
Citation
If you use this repository, please cite:

Toward a Security Score for 6G Vehicular Networks:
Concept and a Simulated Spoofing-Detection Pilot (NAGT-Score)
License
MIT License



