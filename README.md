# NAGT-Score: Noise-Aware Graph-Temporal Security Scoring for 6G Vehicular Networks

Reference implementation accompanying the paper *"A Unified Security Score for
6G Vehicular Networks: A Resilience-Based Framework and Its Learned
Graph-Temporal Realization (NAGT-Score)."*

This repository contains:
- A minimal reverse-mode automatic-differentiation engine written in NumPy
  (`src/autograd.py`), gradient-checked against numerical derivatives.
- NAGT-Score's four architectural components — a learnable noise-aware
  feature-weighting gate, a graph-attention (GAT) encoder, a Transformer
  temporal encoder, and four multi-task prediction heads (`src/models.py`).
- Data loading, windowing, and per-vehicle sequence construction for the
  VeReMi-style highway V2X dataset (`src/data_prep.py`).
- The full training/evaluation pipeline for NAGT-Score and four baselines
  (LSTM, Transformer, GNN, GNN+Transformer) under an identical multi-task
  protocol (`src/main.py`).

## Why NumPy instead of PyTorch/TensorFlow?

This reference implementation was developed and run on a CPU-only machine
with no GPU and insufficient disk space for a full deep-learning framework
install. Rather than skip gradient verification, a small autodiff engine was
written from scratch and checked against finite-difference numerical
gradients (see `tests/` or the gradient-check snippets referenced in the
paper's methodology). All five models (NAGT-Score and the four baselines)
use single-layer graph, attention, and recurrent modules for this reason —
see the paper's Discussion and Limitations section. Deeper, multi-layer,
GPU-trained variants are natural future work.

## Requirements

```
numpy
pandas
scikit-learn
```

Install with:
```bash
pip install -r requirements.txt
```

## Data

The experiments expect a VeReMi-style message-level V2X CSV (see the paper's
Section on Dataset for the exact schema: sender/receiver kinematics and
per-field noise estimates, `attacker`, `successful_delivery`,
`susceptible_previous_window`, `active_current_window`,
`delay_raw_time_units`, and split-level `split_rho`/`split_beta`/
`split_gamma`/`split_R0`/`split_pi` columns). Place the CSV and point
`DATA_PATH` (see `src/main.py`) at it, or pass the path as an environment
variable — see `main.py` for the exact variable name used.

## Running

Single seed, default settings:
```bash
cd src
EPOCHS=15 python3 main.py
```

Multiple seeds for statistical variation (mean ± std across runs, as reported
in the paper's ablation/statistical-variation section):
```bash
cd src
SEEDS=0,1,2,3,4 EPOCHS=15 python3 main.py
```

Results are written to `./results_multiseed.json` by default (per-model,
per-seed Test-set metrics: accuracy/F1/AUC for the anomaly, degradation, and
propagation heads, and RMSE for the timing head). Set `OUT_PATH=/some/path.json`
to write elsewhere — **do this whenever running a quick/partial test**, so a
short debugging run does not silently overwrite a full multi-seed result file
sitting at the default path. (An earlier version of this script used a
hard-coded output path, which caused exactly this problem during development;
`OUT_PATH` exists specifically to prevent it from happening to you too.)

## Repository structure

```
src/
  autograd.py     # reverse-mode autodiff engine (Tensor, Adam optimizer, losses)
  models.py       # NoiseGate, GATLayer, LSTM, SelfAttentionLayer, MultiHead
  data_prep.py    # CSV loading, windowing, per-vehicle sequence construction
  main.py         # training/evaluation pipeline for all 5 models, multi-seed driver
```

## Citation

If you use this code, please cite the accompanying paper (full citation to be
added upon publication).

## License

MIT License — see `LICENSE`.
