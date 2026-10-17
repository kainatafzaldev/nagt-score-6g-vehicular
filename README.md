# NAGT-Score: Security-Scoring Concept and Simulated V2X Spoofing-Detection Pilot

Reference code and corrected experimental results accompanying the October 2026 manuscript:

**“Toward a Security Score for 6G Vehicular Networks: Concept and a Simulated Spoofing-Detection Pilot (NAGT-Score).”**

> **Research scope:** This work proposes a *conceptual*, bounded four-indicator security score and separately evaluates a noise-aware graph-temporal spoofing detector on **one simulated V2X scenario**. The detector does **not** validate the operational security score, and neither the score nor the dataset measures security in deployed 6G networks.

## What the paper contributes

The paper has two distinct parts:

1. **Conceptual score:** a proposed score `S` in `[0, 1]` using four indicators—reliability (`rho`), resilience (`Res`), propagation containment (`pi`), and response (`delta`)—and four suggested interpretation bands (Critical, Degraded, Adequate, Target). The numerical illustration uses **dataset-derived proxies**, not measured component reliability, recovery curves, observed propagation, or detection-to-containment times. Its operator-facing bands and investment sensitivity rule are proposals, not calibrated or validated operational thresholds.
2. **Simulated detection pilot:** NAGT-Score uses a learnable noise-aware input gate, graph-attention encoding, sender-sequence attention, and multitask prediction heads. It is compared with four reference models (LSTM, Transformer, GNN, GNN+Transformer) under a matched protocol. These models predict message-level labels; their learned outputs are **not** used to operationally validate the conceptual score.

## Repository implementation

- `autograd.py`: lightweight NumPy reverse-mode automatic differentiation, optimizer, and losses; the corrected implementation uses **ordered parents**, `tuple(_children)`, rather than `set(_children)`.
- `test_autograd.py`: numerical-gradient checks of the autodiff engine.
- `models.py`: noise-aware gate, graph attention, recurrent/attention sequence components, and prediction heads.
- `data_prep.py`: data loading, 250-message windowing, graph construction, and per-sender sequences.
- `main.py`: matched training and evaluation of NAGT-Score plus LSTM, Transformer, GNN, and GNN+Transformer.
- `summarize_results.py`: per-seed summaries (mean and **sample** standard deviation).
- `check_eval_subset.py`: audit of the evaluated subset after sequence truncation.
- `supplementary_gb.py`: supplementary gradient-boosting feature-dependence analysis, outside the five-model experimental protocol.

The project uses a small, CPU-oriented NumPy implementation rather than PyTorch/TensorFlow. The paper evaluates single-layer modules; more complex GPU-trained variants were not assessed.

## Requirements

```bash
pip install -r requirements.txt
```

The dependencies listed in the repository are `numpy`, `pandas`, and `scikit-learn`.

## Dataset and scope

**NAGT-V2X Benchmark Dataset (NVBD)** is a *simulated*, VeReMi-style highway V2X message dataset covering **one Highway 2 Constant-Position-Offset spoofing scenario**. It is **not** measured operational 6G traffic.

| Property | Manuscript description |
|---|---|
| Messages | 99,937 |
| Columns | 52 |
| Sender IDs | 841 (not guaranteed persistent across splits) |
| Provided Train split | 37,701 messages |
| Provided Validation split | 17,977 messages |
| Provided Test split | 44,259 messages |
| Attack coverage | One position-offset spoofing scenario |

Dataset archive: https://doi.org/10.5281/zenodo.23278351 (`dataset.zip`; extract `dataset.csv`). The dataset is not bundled with this code repository. Consult the Zenodo record for the dataset's licensing terms; the repository's **MIT license applies to the code**, not automatically to externally hosted data.

Relevant fields include kinematics, noise estimates, `sender_distance_to_road_edge`, `attacker`, `successful_delivery`, `susceptible_previous_window`, `active_current_window`, `delay_raw_time_units`, `split`, and split-level proxy fields `split_rho`, `split_beta`, `split_gamma`, `split_R0`, and `split_pi`.

**Important label/feature caveats:** `susceptible_previous_window` is derived from attacker information; propagation and anomaly targets coincide in this dataset; the degradation target is related to delay-derived delivery; and splits are not vehicle-disjoint. These issues restrict interpretation.

## Reproduce the **corrected** Tables X, XI and XII

Use the **corrected `autograd.py`** (ordered computation-graph parents) and a consistent version of `main.py` supporting `DROP_LEAKY`, `DROP_ROADEDGE`, `SEEDS`, `EPOCHS`, `DATA_PATH`, and `OUT_PATH`. Check that the corrected files have actually been committed to your checkout before running these commands.

Each corrected experiment uses **five models × five seeds (`0,1,2,3,4`) × 15 epochs**, with the dataset's fixed Train/Validation/Test split. Set an **independent output path** for every configuration to avoid overwriting results.

### Linux / macOS (Bash)

```bash
# Point this to your extracted NVBD dataset
export DATA_PATH="/path/to/dataset.csv"
export SEEDS="0,1,2,3,4"
export EPOCHS="15"

# Table X: all inputs retained
DROP_LEAKY=0 DROP_ROADEDGE=0 OUT_PATH=results_corrected/repro_fixed_tableX_5seeds_15epochs.json python3 main.py

# Table XI: remove label-derived susceptible_previous_window
DROP_LEAKY=1 DROP_ROADEDGE=0 OUT_PATH=results_corrected/repro_fixed_tableXI_5seeds_15epochs.json python3 main.py

# Table XII: remove label-derived feature and mask road-edge distance
DROP_LEAKY=1 DROP_ROADEDGE=1 OUT_PATH=results_corrected/repro_fixed_tableXII_5seeds_15epochs.json python3 main.py
```

### Windows PowerShell

```powershell
$env:DATA_PATH = "C:\path\to\dataset.csv"
$env:SEEDS = "0,1,2,3,4"
$env:EPOCHS = "15"

# Table X
$env:DROP_LEAKY = "0"; $env:DROP_ROADEDGE = "0"
$env:OUT_PATH = "results_corrected/repro_fixed_tableX_5seeds_15epochs.json"
py main.py

# Table XI
$env:DROP_LEAKY = "1"; $env:DROP_ROADEDGE = "0"
$env:OUT_PATH = "results_corrected/repro_fixed_tableXI_5seeds_15epochs.json"
py main.py

# Table XII
$env:DROP_LEAKY = "1"; $env:DROP_ROADEDGE = "1"
$env:OUT_PATH = "results_corrected/repro_fixed_tableXII_5seeds_15epochs.json"
py main.py
```

> Create `results_corrected/` first if it does not already exist (`mkdir -p results_corrected` in Bash or `New-Item -ItemType Directory -Force results_corrected` in PowerShell). Do **not** replace the archived corrected JSON files with incomplete/debug runs. To check the pipeline quickly, use another path, e.g. `OUT_PATH=debug_results.json EPOCHS=1 SEEDS=0` in Bash.

**Input handling:** Table X uses both features. Table XI excludes the label-derived `susceptible_previous_window` input. Table XII also replaces standardized `sender_distance_to_road_edge` with constant zero **while preserving input dimensions**; it does not remove the column by changing model dimensions.

### Canonical corrected result files

```text
results_corrected/
  repro_fixed_tableX_5seeds_15epochs.json
  repro_fixed_tableXI_5seeds_15epochs.json
  repro_fixed_tableXII_5seeds_15epochs.json
```

These are the corrected result-artifact names identified in the manuscript's accompanying repository notes. **Confirm all three files are present in the public `main` branch** before describing the repository as fully reproducible. Older outputs such as `results_multiseed.json`, `model_comparison.csv`, or files in `results_original/` are **historical** and must not be used as the corrected Tables X–XII.

For local diagnostics, the repository also supplies:

```bash
python3 test_autograd.py
python3 check_eval_subset.py
python3 supplementary_gb.py
```

Set `DATA_PATH` in your shell for the checks requiring the dataset. The gradient-boosting feature-dependence check is supplementary and is not an additional neural baseline.

## Evaluation protocol

- The model compares message-level outputs, pooled over messages retained in each sender's sequence.
- Graphs are assembled from chronologically ordered, nonoverlapping windows of 250 messages within a split; short final windows need at least four messages.
- Each sender's sequence is restricted to its **first 80 messages** within the split; later messages do not enter model training or evaluation.
- Only **27,274 of 44,259 Test messages (61.6%)** are scored after sequence truncation. The attacker fraction of this evaluated subset is **21.3%** and majority-class accuracy is approximately **0.787**.
- Train features are standardized with Train-split statistics only. The provided Validation split is evaluated, but not used for fitting, tuning, or early stopping.
- Models share the implementation's common training budget: 15 epochs, batches of 120 vehicle sequences, Adam learning rate 0.02, and the five fixed seeds.
- Reported Test metrics include anomaly accuracy/F1/AUC, degradation F1/AUC, and RMSE of **standardized log-delay** prediction. The code's additional propagation output does **not** measure observed epidemic-style attack spreading.
- Mean ± values denote mean ± **sample standard deviation across the five seeds**, not confidence intervals or formal significance evidence.

## Selected findings from the final corrected manuscript

| Experiment | Finding (Test, five-seed mean ± sample SD) |
|---|---|
| **Table X** – all features | Transformer anomaly F1 **0.768 ± 0.161** and AUC **0.951 ± 0.009**; NAGT-Score F1 **0.733 ± 0.131**, AUC **0.944 ± 0.049**, and accuracy **0.883 ± 0.091**. |
| **Table XI** – label-derived feature removed | Transformer has highest mean anomaly AUC **0.949 ± 0.022**; LSTM highest mean anomaly F1 **0.779 ± 0.072**; NAGT-Score AUC **0.902 ± 0.125**, F1 **0.634 ± 0.333**. |
| **Table XII** – label-derived and road-edge inputs removed/masked | Across all five models, mean anomaly AUC ranges from **0.446 to 0.523**, near chance. NAGT-Score AUC **0.461 ± 0.052**. |

These are descriptive observed means, **not statistically significant superiority claims**. In the supplementary gradient-boosting check, anomaly AUC is **0.841** with kinematic, noise and road-edge features (excluding leakage-related and identifier/time fields), and falls to **0.514** without road-edge distance. The results indicate strong dependence on a road-edge plausibility cue particular to this simulated attack.

## Autograd correction and reproduction limits

An earlier version of the NumPy autodiff engine represented a node's parents with `set(_children)`, making traversal/gradient accumulation order unstable. The corrected implementation uses `tuple(_children)` and passed **four** numerical-gradient checks. In the tested environment, two independent corrected **LSTM seed-0** 15-epoch runs produced exactly matching first-minibatch gradients and Test-metric hashes (AUC `0.896709948`, F1 `0.789763861`). This observation does **not** establish bitwise reproducibility for every model, seed, hardware/software stack, or platform.

## Limitations and claims **not** made

- `S` is **not** a calibrated or validated measure of real 6G vehicular security. The four score inputs are proxies on this dataset, and four proposed score bands are not operationally validated.
- The experiments do **not** measure network component uptime/failure, recovery traces, observed spread between vehicles, actual intrusion detection-to-containment delay, or physical safety outcomes.
- NAGT-Score has **not** been demonstrated to outperform the Transformer or other baselines consistently or significantly; these comparisons do **not** independently establish an advantage for graph context or the noise-aware gate.
- Results cover **one simulated highway position-offset spoofing scenario**, not new vehicles, different attacks or intensities, physical radio/6G conditions, or deployed vehicular systems.
- Dataset leakage/dependence and non-vehicle-disjoint splits prevent claims of generalization.
- The **57 proposed additional `new_` variables** discussed in the review are **future data requirements**, not collected or generated variables in NVBD.

## License and citation

Code: MIT License; see `LICENSE`. For external NVBD data, check the terms attached to its Zenodo record.

If you use this code, cite the accompanying manuscript (publication bibliographic details pending):

> *Toward a Security Score for 6G Vehicular Networks: Concept and a Simulated Spoofing-Detection Pilot (NAGT-Score).* Manuscript, October 2026.

Repository: https://github.com/kainatafzaldev/nagt-score-6g-vehicular
