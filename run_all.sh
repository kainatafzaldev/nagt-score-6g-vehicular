#!/usr/bin/env bash
# Reproduce Tables XI, XII and XIII (5 models x 5 seeds x 15 epochs each).
# Usage:  DATA_PATH=/path/to/dataset.csv bash run_all.sh
# Needs a few hours on one CPU core. Output goes to results_rerun/ and does not
# overwrite results_corrected/ or the committed CSV files.
set -euo pipefail
: "${DATA_PATH:?Set DATA_PATH to the path of dataset.csv}"
export DATA_PATH SEEDS=0,1,2,3,4 EPOCHS=15
mkdir -p results_rerun

run() {  # run <table> <DROP_LEAKY> <DROP_ROADEDGE>
  echo "=== Table $1 (DROP_LEAKY=$2 DROP_ROADEDGE=$3) ==="
  DROP_LEAKY="$2" DROP_ROADEDGE="$3" OUT_PATH="results_rerun/table$1.json" python3 main.py
  (cd results_rerun && python3 ../summarize_results.py "table$1.json" && mv model_comparison.csv "model_comparison_table$1.csv")
}

run XI   0 0
run XII  1 0
run XIII 1 1
echo "Done. Compare results_rerun/model_comparison_table*.csv with the committed CSV files."
