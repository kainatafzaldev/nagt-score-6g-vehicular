# Toward a Security Score for 6G Vehicular Networks: Concept and a Simulated Spoofing-Detection Pilot (NAGT-Score)

Code and corrected experiments accompanying the manuscript **“Toward a Security Score for 6G Vehicular Networks: Concept and a Simulated Spoofing-Detection Pilot (NAGT-Score)”** (October 2026).

The paper has two separate parts: (1) a proposed four-indicator security-scoring concept, and (2) a small spoofing-detection experiment on simulated V2X messages. **The detection experiment does not validate the security score or measure security in deployed 6G networks.**

## What is included

- `autograd.py`, `test_autograd.py` — CPU-oriented NumPy automatic differentiation and gradient checks.
- `models.py` — NAGT-Score's noise-aware gate, graph-attention encoder, sequence encoder, and four prediction heads; also baseline models.
- `data_prep.py`, `main.py` — dataset preparation, training, and evaluation.
- `summarize_results.py`, `check_eval_subset.py`, `supplementary_gb.py` — result summary and supporting checks.
- `results_corrected/` — corrected JSON results for manuscript Tables X, XI, and XII. **Earlier result files are historical, not the final paper results.**

The five compared models are **LSTM, Transformer, GNN, GNN+Transformer, and NAGT-Score**. All are evaluated under the same multi-task protocol.

## Dataset

**NAGT-V2X Benchmark Dataset (NVBD):** 99,937 simulated messages, 52 columns, 841 sender IDs, one Highway 2 **Constant-Position-Offset spoofing** scenario.

- Train: 37,701 messages
- Validation: 17,977 messages
- Test: 44,259 messages
- Download: https://doi.org/10.5281/zenodo.23278351 (`dataset.zip`, containing `dataset.csv`)

The CSV is **not included** in this repository. Extract it and set `DATA_PATH` to its location. The repository's MIT license applies to the **code**; refer to Zenodo for dataset licensing.

## Install and run

```bash
pip install -r requirements.txt
python test_autograd.py
```

The corrected paper reports **five models × five seeds (0–4) × 15 epochs** for each configuration:

| Paper table | Input configuration | `DROP_LEAKY` | `DROP_ROADEDGE` |
|---|---|---:|---:|
| X | All inputs | 0 | 0 |
| XI | Exclude label-derived `susceptible_previous_window` | 1 | 0 |
| XII | Also mask standardized `sender_distance_to_road_edge` to zero | 1 | 1 |

Run from the repository folder, using a **different output filename for each experiment** so the published JSON files are not overwritten.

### Windows PowerShell

```powershell
$env:DATA_PATH = "C:\path\to\dataset.csv"
$env:SEEDS = "0,1,2,3,4"
$env:EPOCHS = "15"

$env:DROP_LEAKY = "0"; $env:DROP_ROADEDGE = "0"
$env:OUT_PATH = "reproduced_tableX.json"
py main.py

$env:DROP_LEAKY = "1"; $env:DROP_ROADEDGE = "0"
$env:OUT_PATH = "reproduced_tableXI.json"
py main.py

$env:DROP_LEAKY = "1"; $env:DROP_ROADEDGE = "1"
$env:OUT_PATH = "reproduced_tableXII.json"
py main.py
```

### Linux / macOS

```bash
export DATA_PATH="/path/to/dataset.csv"
export SEEDS="0,1,2,3,4"
export EPOCHS="15"

DROP_LEAKY=0 DROP_ROADEDGE=0 OUT_PATH=reproduced_tableX.json python3 main.py
DROP_LEAKY=1 DROP_ROADEDGE=0 OUT_PATH=reproduced_tableXI.json python3 main.py
DROP_LEAKY=1 DROP_ROADEDGE=1 OUT_PATH=reproduced_tableXII.json python3 main.py
```

Corrected results archived with the paper:

```text
results_corrected/repro_fixed_tableX_5seeds_15epochs.json
results_corrected/repro_fixed_tableXI_5seeds_15epochs.json
results_corrected/repro_fixed_tableXII_5seeds_15epochs.json
```

The evaluation retains only the first **80 messages per sender per split**: **27,274 of 44,259 Test messages** are scored. Validation is reported but is not used for training or tuning. Reported values are Test-set **means ± sample standard deviations across five seeds**.

## Key results from the corrected paper

- **Table X (all inputs):** Transformer achieved the highest mean anomaly F1 (**0.768 ± 0.161**) and AUC (**0.951 ± 0.009**). NAGT-Score achieved F1 **0.733 ± 0.131**, AUC **0.944 ± 0.049**, and the highest mean accuracy (**0.883 ± 0.091**).
- **Table XI (label-derived input excluded):** Transformer had the highest mean anomaly AUC (**0.949 ± 0.022**), while LSTM had the highest mean anomaly F1 (**0.779 ± 0.072**). NAGT-Score had AUC **0.902 ± 0.125** and F1 **0.634 ± 0.333**.
- **Table XII (both inputs excluded/masked):** all five models' mean anomaly AUCs were **0.446–0.523**, close to chance.

These are descriptive results, **not evidence of significant or consistent superiority** for NAGT-Score. The supplementary feature-dependence test also found strong reliance on the road-edge plausibility feature.

## Security-score concept and limits

The paper proposes a bounded score `S` combining **reliability, resilience, propagation containment, and response**, with four proposed bands: Critical, Degraded, Adequate, and Target. Its illustrative Test-split value is **S = 0.931**, calculated from dataset-derived **proxies**, **not independently measured cyber-physical security indicators**. This value does **not** prove that the simulated network is secure.

Important limitations:

- Only **one simulated attack and scenario** were evaluated; there is no operational 6G-security or physical-safety validation.
- `susceptible_previous_window` is label-derived, and anomaly and propagation labels coincide.
- The delivery/degradation label is derived from communication delay; timing prediction is **not** intrusion detection or containment time.
- The splits are **not vehicle-disjoint**, and road-edge distance carries much of the detection signal.
- The model does not demonstrate independent benefits from graph context or the noise-aware gate, nor generalization to other vehicles, attacks, or environments.

The corrected NumPy autodiff engine uses ordered parents (`tuple(_children)` instead of `set(_children)`). **Four gradient checks passed**, and two independent LSTM seed-0 runs matched exactly **in the tested environment**; cross-platform bitwise reproducibility is not established.

## Citation and license

**Syed Usman Jamil, M. Arif Khan, Md. Abdur Rahman, Syed Sadiqur Rahman, and Rabia Khan.** *Toward a Security Score for 6G Vehicular Networks: Concept and a Simulated Spoofing-Detection Pilot (NAGT-Score).* Publication details to be added when available.

Code license: **MIT** (see `LICENSE`). Dataset distribution is separate (see the Zenodo record).
