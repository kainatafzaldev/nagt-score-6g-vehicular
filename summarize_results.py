"""Summarize results_multiseed.json (written by main.py) as mean +/- sample std over seeds.
Usage: python3 summarize_results.py [path/to/results_multiseed.json]
"""
import json, sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "results_multiseed.json"
res = json.load(open(path))

def ms(vals):
    v = np.array(vals, dtype=float)
    return f"{np.nanmean(v):.3f}+-{np.nanstd(v, ddof=1):.3f}"

cols = [("anomaly", "acc"), ("anomaly", "f1"), ("anomaly", "auc"),
        ("degradation", "f1"), ("degradation", "auc"), ("timing_rmse", None)]
print("Test split, mean +- sample std over seeds")
print(f"{'Model':18s}" + "".join(f"{h+'_'+(k or ''):>20s}" for h, k in cols))
for model, by_seed in res.items():
    row = []
    for h, k in cols:
        vals = [(r["Test"][h][k] if k else r["Test"][h]) for r in by_seed.values()]
        row.append(ms(vals))
    print(f"{model:18s}" + "".join(f"{x:>20s}" for x in row) + f"   (n_seeds={len(by_seed)})")
