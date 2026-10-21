import json
import numpy as np
from scipy import stats

files = {
    "WITH leaky feature (original, Table X)": "archive/results_original/results_multiseed_with_leaky.json",
    "WITHOUT susceptible_previous_window": "archive/results_original/results_multiseed_no_leaky.json",
}
pairs = [
    ("NAGT-Score", "GNN+Transformer"),
    ("NAGT-Score", "Transformer"),
    ("Transformer", "GNN+Transformer"),
]

def get(d, model, metric):
    seeds = sorted(d[model].keys(), key=int)
    return np.array([d[model][s]["Test"]["anomaly"][metric] for s in seeds])

for label, path in files.items():
    d = json.load(open(path))
    print()
    print("=" * 70)
    print(label)
    print("=" * 70)
    print("Means +- sample std (anomaly, Test):")
    for m in d:
        f1 = get(d, m, "f1"); auc = get(d, m, "auc"); acc = get(d, m, "acc")
        print("  %-16s acc %.3f+-%.3f  F1 %.3f+-%.3f  AUC %.3f+-%.3f" % (
            m, acc.mean(), acc.std(ddof=1), f1.mean(), f1.std(ddof=1), auc.mean(), auc.std(ddof=1)))
    print()
    print("Paired comparisons (A minus B), per seed 0-4:")
    for a, b in pairs:
        for metric in ("f1", "auc", "acc"):
            x = get(d, a, metric); y = get(d, b, metric)
            diff = x - y
            t, p = stats.ttest_rel(x, y)
            print("  %s vs %s | %s: diffs %s | mean %+.3f | p = %.3f" % (
                a, b, metric, np.round(diff, 3).tolist(), diff.mean(), p))
