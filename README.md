# NAGT-Score: Noise-Aware Graph-Temporal Security Scoring for 6G Vehicular Networks

Reference implementation accompanying the paper *"A Unified Security Score for 6G Vehicular Networks: A Resilience-Based Framework and Its Learned Graph-Temporal Realization (NAGT-Score)."*

This repository contains:

* A minimal reverse-mode automatic-differentiation engine written in NumPy (`src/autograd.py`), gradient-checked against numerical derivatives (`src/test_autograd.py`).
* NAGT-Score's four architectural components — a learnable noise-aware feature-weighting gate, a graph-attention (GAT) encoder, a Transformer temporal encoder, and four multi-task prediction heads (`src/models.py`).
* Data loading, windowing, and per-vehicle sequence construction for the VeReMi-style highway V2X dataset (`data_prep.py`).
* The full training/evaluation pipeline for NAGT-Score and four baselines (LSTM, Transformer, GNN, GNN+Transformer) under an identical multi-task protocol (`main.py`).
* The per-seed results used in the paper (`results_multiseed.json`).

## Why NumPy instead of PyTorch/TensorFlow?

This reference implementation was developed and run on a CPU-only machine with no GPU and insufficient disk space for a full deep-learning framework install. Rather than skip gradient verification, a small autodiff engine was written from scratch and checked against finite-difference numerical gradients (see `src/test_autograd.py`).

All five models (NAGT-Score and the four baselines) use single-layer graph, attention, and recurrent modules for this reason — see the paper's Discussion and Limitations section. Deeper, multi-layer, GPU-trained variants are natural future work.

## Requirements

```text
numpy
pandas
scikit-learn
```

Install with:

```bash
pip install -r requirements.txt
```

## Data

The experiments expect a VeReMi-style message-level V2X CSV. The dataset should follow the schema described in the paper's Dataset section, including:

* sender/receiver kinematics
* per-field noise estimates
* `attacker`
* `successful_delivery`
* `susceptible_previous_window`
* `active_current_window`
* `delay_raw_time_units`
* `split_rho`
* `split_beta`
* `split_gamma`
* `split_R0`
* `split_pi`

Place the CSV in the repository and point `DATA_PATH` in `main.py` to it, or pass the path as an environment variable as supported by `main.py`.

## Running

Run these commands from the repository folder.

### Single seed

For the default single-seed configuration:

```bash
cd src
EPOCHS=15 python3 main.py
```

### Multiple seeds

For statistical variation across five seeds:

```bash
cd src
SEEDS=0,1,2,3,4 EPOCHS=15 python3 main.py
```

This computes mean ± standard deviation across runs, as reported in the paper's ablation/statistical-variation section.

Results are written to:

```text
./results_multiseed.json
```

by default.

The output contains per-model, per-seed test-set metrics:

* Accuracy, F1, and AUC for the anomaly head
* Accuracy, F1, and AUC for the degradation head
* Accuracy, F1, and AUC for the propagation head
* RMSE for the timing head

To write results to a different location, set:

```bash
OUT_PATH=/some/path.json
```

**Use `OUT_PATH` whenever running a quick or partial test**, so a short debugging run does not silently overwrite the full multi-seed result file.

For example:

```bash
OUT_PATH=debug_results.json EPOCHS=1 python3 main.py
```

## Repository Structure

```text
src/
  autograd.py          # Reverse-mode autodiff engine, Tensor, Adam optimizer, losses
  models.py            # NoiseGate, GATLayer, LSTM, SelfAttentionLayer, MultiHead
  data_prep.py         # CSV loading, windowing, per-vehicle sequence construction
  main.py              # Training/evaluation pipeline for all 5 models
  test_autograd.py     # Gradient check of the autodiff engine

results_multiseed.json # Per-seed results reported in the paper
requirements.txt       # Python dependencies
LICENSE                # MIT License
```

## Citation

If you use this code, please cite the accompanying paper:

> Full citation to be added upon publication.

## License

MIT License — see `LICENSE`.


## Corrected experiments (October 2026)

Tables X, XI and XII in the corrected manuscript use the five-seed,
15-epoch results stored in `results_corrected/`.

The earlier `results_multiseed.json` is historical and was generated
before the autograd reproducibility correction.

The corrected `autograd.py` preserves computation-graph parent order
using `tuple(_children)` instead of `set(_children)`.
Gradient checks passed, and independent LSTM seed-0 runs matched exactly.

To reproduce the experiments, set `DATA_PATH` to the NVBD CSV,
`SEEDS=0,1,2,3,4`, and `EPOCHS=15`.

- Table X: DROP_LEAKY=0, DROP_ROADEDGE=0
- Table XI: DROP_LEAKY=1, DROP_ROADEDGE=0
- Table XII: DROP_LEAKY=1, DROP_ROADEDGE=1

Use a unique OUT_PATH for each run to preserve existing results.

The dataset CSV is not included in this repository.
These experiments cover one simulated V2X spoofing scenario, not a
validated real-world 6G security deployment.