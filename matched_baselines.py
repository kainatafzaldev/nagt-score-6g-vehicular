import csv
import numpy as np
import data_prep as dp

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# Load the original dataset.
df = dp.load("./dataset.csv")

# Recreate the exact neural-model evaluation subset.
windows = dp.make_windows(df)
used = np.sort(np.unique(np.concatenate(
    [w["rows"] for w in windows]
)))
evaluated = df.loc[used].reset_index(drop=True)

seq_idx, mask, splits = dp.build_sequences(evaluated)
selected = splits == "Test"
rows = seq_idx[selected][mask[selected].astype(bool)]

assert len(rows) == 27274
assert len(np.unique(rows)) == len(rows)

test = evaluated.iloc[rows].copy()
assert (test["split"] == "Test").all()

y = test["attacker"].to_numpy().astype(int)

def measure(name, prediction, scores):
    return {
        "Model": name,
        "N_Test": len(y),
        "Accuracy": accuracy_score(y, prediction),
        "Precision": precision_score(
            y, prediction, zero_division=0
        ),
        "Recall": recall_score(
            y, prediction, zero_division=0
        ),
        "F1": f1_score(
            y, prediction, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y, scores)
    }

results = []

# Baseline 1: Predict all messages as legitimate.
majority_pred = np.zeros(len(y), dtype=int)
results.append(measure(
    "Majority-class",
    majority_pred,
    majority_pred
))

# Baseline 2: Fixed descriptive road-edge rule.
distance = test[
    "sender_distance_to_road_edge"
].to_numpy()

road_pred = (distance < -5).astype(int)
results.append(measure(
    "Road-edge threshold (-5 m)",
    road_pred,
    -distance
))

# Baseline 3: Gradient boosting.
# Train on the original full Train partition.
train = df[df["split"] == "Train"].copy()

features = (
    dp.RAW_COLS + dp.NOISE_COLS
    + ["sender_distance_to_road_edge"]
)

gb = HistGradientBoostingClassifier(random_state=0)
gb.fit(train[features], train["attacker"])

scores = gb.predict_proba(test[features])[:, 1]
gb_pred = (scores > 0.5).astype(int)

results.append(measure(
    "HistGradientBoosting",
    gb_pred,
    scores
))

# Compare against independently recorded outputs.
expected = {
    "Majority-class": (0.787, 0.0, 0.5),
    "Road-edge threshold (-5 m)": (
        0.9319865072963261,
        0.8103078024337866,
        0.844522311991883
    ),
    "HistGradientBoosting": (
        0.7714673315245288,
        0.5901229696850134,
        0.8635031956681553
    )
}

for r in results:
    print("\n", r["Model"])
    for key in [
        "N_Test", "Accuracy", "Precision",
        "Recall", "F1", "ROC_AUC"
    ]:
        print(f"  {key}: {r[key]}")

    a, f, u = expected[r["Model"]]
    tolerance = 0.001 if r["Model"] == "Majority-class" else 0.000001

    assert abs(r["Accuracy"] - a) < tolerance
    assert abs(r["F1"] - f) < tolerance
    assert abs(r["ROC_AUC"] - u) < tolerance

# Save CSV only after all verification checks pass.
with open(
    "baseline_comparison.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file, fieldnames=list(results[0].keys())
    )
    writer.writeheader()
    writer.writerows(results)

print("\nALL NUMERICAL CHECKS PASSED")
print("Created: baseline_comparison.csv")
