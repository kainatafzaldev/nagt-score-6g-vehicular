# NAGT-Score: Noise-Aware Graph-Temporal Detection for Simulated V2X Spoofing

Reference implementation accompanying the paper *"NAGT-Score: A Security Scoring Concept and Graph-Temporal Detection Framework for Simulated V2X Spoofing Scenarios."*

This repository contains:

* A minimal reverse-mode automatic-differentiation engine written in NumPy (`autograd.py`), gradient-checked against numerical derivatives (`test_autograd.py`).
* NAGT-Score's four architectural components - a learnable noise-aware feature-weighting gate, a graph-attention (GAT) encoder, a Transformer sequence encoder, and four multi-task prediction heads (`models.py`).
* Data loading, windowing, and per-vehicle sequence construction for the VeReMi-style highway V2X dataset (`data_prep.py`).
* The full training/evaluation pipeline for NAGT-Score and four baselines (LSTM, Transformer, GNN, GNN+Transformer) under an identical multi-task protocol (`main.py`).
* Analysis scripts used in the paper: result summary (`summarize_results.py`), evaluated-subset check (`check_eval_subset.py`), and the supplementary feature-dependence check (`supplementary_gb.py`).
* The per-seed results used in the paper (`results_multiseed.json`).

## Why NumPy instead of PyTorch/TensorFlow?

This reference implementation was developed and run on a CPU-only machine with no GPU and insufficient disk space for a full deep-learning framework install. Rather than skip gradient verification, a small autodiff engine was written from scratch and checked against finite-difference numerical gradients (see `test_autograd.py`).

All five models (NAGT-Score and the four baselines) use single-layer graph, attention, and recurrent modules for this reason - see the paper's Discussion and Limitations section. Deeper, multi-layer, GPU-trained variants are natural future work.

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

The experiments use the **NAGT-V2X Benchmark Dataset (NVBD)**, a *simulated*, VeReMi-style, message-level V2X dataset (one highway scenario, one Constant-Position-Offset attack). The dataset is **not stored in this repository** (about 68 MB).

* Dataset download: https://doi.org/10.5281/zenodo.23278351 (file `dataset.zip`; unzip it to get `dataset.csv`)
* Dataset license: CC BY 4.0

The CSV should follow the schema described in the paper's Dataset section, including:

* sender/receiver kinematics
* per-field noise estimates
* `sender_distance_to_road_edge`
* `attacker`
* `successful_delivery`
* `susceptible_previous_window`
* `active_current_window`
* `delay_raw_time_units`
* `split` (Train / Validation / Test)
* `split_rho`, `split_beta`, `split_gamma`, `split_R0`, `split_pi`

Download `dataset.zip`, unzip it to get `dataset.csv`, and point `DATA_PATH` to it, for example:

```bash
DATA_PATH=/path/to/dataset.csv
```

If `DATA_PATH` is not set, `main.py` looks for `./dataset.csv` in the folder you run it from.

## Running

Run these commands from the repository folder (on Windows, use `py` instead of `python3`).

### Single seed

```bash
DATA_PATH=/path/to/dataset.csv EPOCHS=15 SEEDS=0 python3 main.py
```

### Multiple seeds

For statistical variation across five seeds:

```bash
DATA_PATH=/path/to/dataset.csv SEEDS=0,1,2,3,4 EPOCHS=15 python3 main.py
```

`main.py` writes per-model, per-seed results to `./results_multiseed.json` (in the folder you run it from) by default:

* Accuracy, F1, and AUC for the anomaly, degradation, and propagation heads
* RMSE for the timing head

To write results elsewhere, set `OUT_PATH=/some/path.json`.

**Use `OUT_PATH` whenever running a quick or partial test**, so a short debugging run does not silently overwrite the full multi-seed result file:

```bash
OUT_PATH=debug_results.json EPOCHS=1 SEEDS=0 python3 main.py
```

### Mean +- standard deviation

To compute the mean +- sample standard deviation over seeds reported in the paper (this also writes `model_comparison.csv`):

```bash
python3 summarize_results.py results_multiseed.json
```

### Additional checks

```bash
DATA_PATH=/path/to/dataset.csv python3 check_eval_subset.py     # messages actually evaluated after sequence truncation
DATA_PATH=/path/to/dataset.csv python3 supplementary_gb.py      # feature-dependence check (gradient boosting)
```

## Evaluation notes

* Metrics are computed **per message**, pooled over the messages in each sender's sequence.
* Each sender's sequence is truncated to its **first 80 messages** within a split; later messages are not used for training or evaluation.
* The Validation split is evaluated and reported but is not used for training, early stopping, or tuning.
* The input feature `susceptible_previous_window` is derived from the attacker label (see the paper's limitations), and the splits are not vehicle-disjoint.
* The dataset is simulated and covers a single attack type; results should not be read as general V2X attack-detection performance.

## Repository Structure

```text
autograd.py              # Reverse-mode autodiff engine, Tensor, Adam optimizer, losses
models.py                # NoiseGate, GATLayer, LSTM, SelfAttentionLayer, MultiHead
data_prep.py             # CSV loading, windowing, per-vehicle sequence construction
main.py                  # Training/evaluation pipeline for all 5 models
summarize_results.py     # Mean +- std over seeds; writes model_comparison.csv
check_eval_subset.py     # Evaluated-message subset and baseline check
supplementary_gb.py      # Supplementary feature-dependence check
test_autograd.py         # Gradient check of the autodiff engine

results_multiseed.json   # Per-seed results reported in the paper
model_comparison.csv     # Summary table of the multi-seed results
requirements.txt         # Python dependencies
LICENSE                  # MIT License
```

## Citation

If you use this code, please cite the accompanying paper:

> Full citation to be added upon publication.

## License

MIT License - see `LICENSE`.