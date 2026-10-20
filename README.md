# Toward a Security Score for 6G Vehicular Networks: Concept and a Simulated Spoofing-Detection Pilot (NAGT-Score)

**Official implementation repository.** Reference code, corrected experimental results, and reproducibility materials for the paper above, including NAGT-Score, a noise-aware graph-temporal detector for simulated V2X spoofing.

> **Important note.** The paper proposes a security-scoring *concept* and reports a *simulated* spoofing-detection pilot. The score is **not** a validated operational 6G security metric, and the results are **not** evidence of general V2X attack-detection performance.

This repository contains:

- A minimal reverse-mode automatic-differentiation engine written in NumPy (`autograd.py`), gradient-checked against numerical derivatives (`test_autograd.py`).
- NAGT-Score's four architectural components (`models.py`): a learnable noise-aware feature-weighting gate, a graph-attention (GAT) encoder, a Transformer sequence encoder, and four multi-task prediction heads.
- Data loading, windowing, and per-vehicle sequence construction for the NVBD dataset (`data_prep.py`).
- The full training and evaluation pipeline for NAGT-Score and four baselines (LSTM, Transformer, GNN, GNN+Transformer) under an identical multi-task protocol (`main.py`).
- The analysis scripts used in the paper: result summary (`summarize_results.py`), evaluated-subset check (`check_eval_subset.py`), and the supplementary feature-dependence check (`supplementary_gb.py`).
- The corrected five-seed results used in the paper (`results_corrected/`). Earlier results (`results_original/`, `results_multiseed.json`) are historical.

## Why NumPy instead of PyTorch/TensorFlow?

The reference implementation was developed and run on a CPU-only machine with no GPU and insufficient disk space for a full deep-learning framework. Rather than skip gradient verification, a small autodiff engine was written from scratch and checked against finite-difference numerical gradients (see `test_autograd.py`).

All five models use single-layer graph, attention, and recurrent modules for this reason (see the paper's Discussion and Limitations). Deeper, multi-layer, GPU-trained variants are future work.

## Requirements

- numpy
- pandas
- scikit-learn

```bash
pip install -r requirements.txt
```

## Data: the NAGT-V2X Benchmark Dataset (NVBD)

The dataset is **not** stored in this repository (about 68 MB).

- Download: https://doi.org/10.5281/zenodo.23278351 (file `dataset.zip`; unzip it to get `dataset.csv`)
- Dataset license: see the Zenodo record. The repository's MIT license applies to the code.

### How NVBD was built

NVBD was assembled by the authors from three ground-truth partition files (Train, Validation, Test) of the InTAS Highway 2 Constant-Position-Offset scenario. InTAS is the Ingolstadt Traffic Scenario for SUMO (Lobo et al., arXiv:2011.11995). The three files are partitions of one scenario, not three independent datasets. They were consolidated into one message-level table (`dataset.csv`, 52 columns) with a `split` column, keeping the original Train/Validation/Test assignment.

A broadcast message can be logged by several receiving vehicles. The table stores each received copy as one row (receiver log in `source_vehicle_file`), so the 45,101 distinct source messages give 99,937 rows.

| Split | Source messages | Rows | Senders | Attacker rows |
|---|---|---|---|---|
| Train | 19,186 | 37,701 | 482 | 7,831 (20.8%) |
| Validation | 7,023 | 17,977 | 152 | 3,247 (18.1%) |
| Test | 18,892 | 44,259 | 476 | 8,710 (19.7%) |
| Total | 45,101 | 99,937 | 841 (unique IDs) | 19,788 (19.8%) |

**The attack is injected when the table is built.** In the source files every position is the recorded ground-truth position. In NVBD, the sender position of every attacker message is shifted by a constant two-dimensional offset (each axis between 20 and 70 m in absolute value, with either sign; vector magnitude 31.5-96.0 m), with one offset per attacker sender ID within a split. Legitimate messages keep their source position exactly, so `attacker = 1` is equivalent to "the position differs from the source file". There are 221 attacker sender IDs, about 20% of the sender IDs in each split.

### Columns

The CSV follows the schema described in the paper's Dataset section, including:

- sender/receiver kinematics and per-field noise estimates
- `sender_distance_to_road_edge`
- `attacker`
- `successful_delivery`, `delay_raw_time_units`
- `susceptible_previous_window`, `active_current_window`
- `split` (Train / Validation / Test)
- `split_rho`, `split_beta`, `split_gamma`, `split_R0`, `split_pi`

### Known properties of the dataset

These are analysed in the paper (Section X-A) and matter when interpreting results:

- `active_current_window` is identical to `attacker`, so the propagation head is the same task as the anomaly head.
- `successful_delivery` is a fixed threshold of `delay_raw_time_units` (threshold 1,901,477.5 units reproduces the label with zero errors).
- `susceptible_previous_window` equals 0 only for attacker messages, so it is label-derived.
- Nearly all detectable signal is in `sender_distance_to_road_edge`; the position offset places attackers off the road.
- The splits are **not vehicle-disjoint** (180 of the 476 Test senders also appear in Train, 89 in Validation) and **not time-disjoint** (the Test period 7,201-14,399 s overlaps the Train period 3,601-12,599 s). Sender IDs are not persistent identities, and message IDs are not unique across splits.
- The data cover a single attack type on a single simulated highway scenario.

## Running

Run these commands from the repository folder (on Windows, use `py` instead of `python3`). If `DATA_PATH` is not set, `main.py` looks for `./dataset.csv` in the folder you run it from.

### Single seed

```bash
DATA_PATH=/path/to/dataset.csv EPOCHS=15 SEEDS=0 python3 main.py
```

### Multiple seeds

```bash
DATA_PATH=/path/to/dataset.csv SEEDS=0,1,2,3,4 EPOCHS=15 python3 main.py
```

`main.py` writes per-model, per-seed results to `./results_multiseed.json` by default:

- Accuracy, F1, and AUC for the anomaly, degradation, and propagation heads
- RMSE for the timing head

To write results elsewhere, set `OUT_PATH=/some/path.json`. Use `OUT_PATH` for any quick or partial test, so a short debugging run does not overwrite the full multi-seed result file:

```bash
OUT_PATH=debug_results.json EPOCHS=1 SEEDS=0 python3 main.py
```

### Mean and standard deviation

To compute the mean and sample standard deviation over seeds reported in the paper (this also writes `model_comparison.csv`):

```bash
python3 summarize_results.py results_multiseed.json
```

### Additional checks

```bash
DATA_PATH=/path/to/dataset.csv python3 check_eval_subset.py     # messages actually evaluated after sequence truncation
DATA_PATH=/path/to/dataset.csv python3 supplementary_gb.py      # feature-dependence check (gradient boosting)
```

## Corrected experiments (October 2026)

The paper reports five models, five seeds (0-4), and 15 epochs for each of three feature configurations (Tables XI, XII, and XIII of the paper).

| Paper table | Configuration | Removed inputs | Result file in `results_corrected/` |
|---|---|---|---|
| Table XI | `DROP_LEAKY=0 DROP_ROADEDGE=0` | none (all features) | `repro_fixed_tableXI_5seeds_15epochs.json` |
| Table XII | `DROP_LEAKY=1 DROP_ROADEDGE=0` | `susceptible_previous_window` | `repro_fixed_tableXII_5seeds_15epochs.json` |
| Table XIII | `DROP_LEAKY=1 DROP_ROADEDGE=1` | `susceptible_previous_window`, `sender_distance_to_road_edge` | `repro_fixed_tableXIII_5seeds_15epochs.json` |

Example (Table XII configuration):

```bash
DATA_PATH=/path/to/dataset.csv DROP_LEAKY=1 DROP_ROADEDGE=0 SEEDS=0,1,2,3,4 EPOCHS=15 \
OUT_PATH=my_tableXII.json python3 main.py
```

Set a different `OUT_PATH` for each experiment to prevent overwriting.

### Main findings (Test split, mean over 5 seeds)

- With all features, the Transformer has the highest mean anomaly AUC (0.951 +/- 0.009) and F1 (0.768 +/- 0.161). NAGT-Score obtains AUC 0.944 +/- 0.049 and F1 0.733 +/- 0.131.
- With both the label-derived and road-edge inputs removed, mean anomaly AUC is between 0.446 and 0.523 for all five models, i.e. close to chance.
- These are rankings of five-seed means, not significance tests, and they do not show that NAGT-Score is better than the baselines.

## Evaluation notes

- Metrics are computed per message, pooled over the messages in each sender's sequence.
- Each sender's sequence is truncated to its first 80 messages within a split; later messages are not used for training or evaluation. Only 27,274 of the 44,259 Test messages (61.6%) are therefore scored.
- The Validation split is evaluated and reported but is not used for training, early stopping, or tuning.
- The input feature `susceptible_previous_window` is derived from the attacker label (see the paper's dataset properties), and the splits are not vehicle-disjoint.
- The dataset is simulated and covers a single attack type; results should not be read as general V2X attack-detection performance.

## Reproducibility audit

The original autodiff engine stored graph parents in an unordered Python `set`, which allowed variation in backpropagation order and floating-point gradient accumulation. The engine now uses ordered computation-graph parents (`tuple(_children)` instead of `set(_children)`). Four gradient checks passed, and independent LSTM seed-0 runs matched exactly in the tested environment. Cross-platform bitwise reproducibility is not claimed.

## Repository structure

```
autograd.py              # Reverse-mode autodiff engine, Tensor, Adam optimizer, losses
models.py                # NoiseGate, GATLayer, LSTM, SelfAttentionLayer, MultiHead
data_prep.py             # CSV loading, windowing, per-vehicle sequence construction
main.py                  # Training/evaluation pipeline for all 5 models
summarize_results.py     # Mean +- std over seeds; writes model_comparison.csv
check_eval_subset.py     # Evaluated-message subset and baseline check
supplementary_gb.py      # Supplementary feature-dependence check
compare_runs.py          # Paired comparisons on the historical (pre-correction) results
test_autograd.py         # Gradient check of the autodiff engine

results_corrected/       # Corrected five-seed results used in the paper
results_original/        # Historical pre-correction results
results_multiseed.json   # Historical results file
model_comparison.csv     # Summary table written by summarize_results.py
requirements.txt         # Python dependencies
LICENSE                  # MIT License
```

## Citation

If you use this code, please cite the accompanying paper:

> Syed Usman Jamil, M. Arif Khan, Md. Abdur Rahman, Syed Sadiqur Rahman, Rabia Khan. "Toward a Security Score for 6G Vehicular Networks: Concept and a Simulated Spoofing-Detection Pilot (NAGT-Score)."

Full citation details will be added upon publication. If you use the underlying traffic scenario, please also cite InTAS: S. C. Lobo, S. Neumeier, E. M. G. Fernandez, C. Facchi, "InTAS - The Ingolstadt Traffic Scenario for SUMO," arXiv:2011.11995, 2020.

## License

MIT License - see `LICENSE`.
