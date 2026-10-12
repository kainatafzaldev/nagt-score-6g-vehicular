"""Summarize results_multiseed.json (written by main.py).
Prints the Test-split table (mean +- sample std over seeds) and writes model_comparison.csv.
Usage: python3 summarize_results.py [path/to/results_multiseed.json]
"""
import csv
import json
import statistics
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "results_multiseed.json"
with open(path, encoding="utf-8") as f:
    data = json.load(f)

def ms(vals):
    return statistics.mean(vals), statistics.stdev(vals)

# 1) model_comparison.csv (all heads, Test split)
with open("model_comparison.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Head", "Metric", "Mean", "Std"])
    for model, seeds in data.items():
        for head in ["anomaly", "degradation", "propagation"]:
            for metric in ["acc", "f1", "auc"]:
                m, s = ms([seeds[k]["Test"][head][metric] for k in seeds])
                writer.writerow([model, head, metric, f"{m:.4f}", f"{s:.4f}"])
        m, s = ms([seeds[k]["Test"]["timing_rmse"] for k in seeds])
        writer.writerow([model, "timing", "RMSE", f"{m:.4f}", f"{s:.4f}"])

# 2) console table used in the paper
cols = [("anomaly", "acc"), ("anomaly", "f1"), ("anomaly", "auc"),
        ("degradation", "f1"), ("degradation", "auc"), ("timing_rmse", None)]
print("Test split, mean +- sample std over seeds")
print(f"{'Model':18s}" + "".join(f"{h + '_' + (k or ''):>20s}" for h, k in cols))
for model, seeds in data.items():
    row = []
    for h, k in cols:
        vals = [(seeds[s]["Test"][h][k] if k else seeds[s]["Test"][h]) for s in seeds]
        m, sd = ms(vals)
        row.append(f"{m:.3f}+-{sd:.3f}")
    print(f"{model:18s}" + "".join(f"{x:>20s}" for x in row) + f"   (n_seeds={len(seeds)})")
print("Saved to model_comparison.csv")