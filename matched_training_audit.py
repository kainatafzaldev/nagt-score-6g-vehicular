"""Gradient boosting trained on exactly the Train messages the neural models keep.

The neural models retain 26,297 of the 37,701 Train messages after the
80-message per-sender truncation. matched_baselines.py trains gradient boosting
on all 37,701 Train messages; this script repeats the fit on the 26,297 messages
(same window and sequence construction as main.py) and scores the same 27,274
Test messages.

Usage:  python3 matched_training_audit.py   (expects ./dataset.csv)
Result recorded in the paper (default HistGradientBoostingClassifier, random_state=0):
    Accuracy 0.790460, F1 0.607405, ROC_AUC 0.868069
"""
import numpy as np
import data_prep as dp

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

df = dp.load("./dataset.csv")
windows = dp.make_windows(df)

used = np.sort(np.unique(np.concatenate([w["rows"] for w in windows])))
evaluated = df.loc[used].reset_index(drop=True)
seq_idx, mask, splits = dp.build_sequences(evaluated)

train_rows = seq_idx[splits == "Train"][mask[splits == "Train"].astype(bool)]
test_rows = seq_idx[splits == "Test"][mask[splits == "Test"].astype(bool)]

assert len(np.unique(train_rows)) == len(train_rows)
assert len(np.unique(test_rows)) == len(test_rows)

train = evaluated.iloc[train_rows].copy()
test = evaluated.iloc[test_rows].copy()

assert (train["split"] == "Train").all()
assert (test["split"] == "Test").all()

features = dp.RAW_COLS + dp.NOISE_COLS + ["sender_distance_to_road_edge"]

model = HistGradientBoostingClassifier(random_state=0)
model.fit(train[features], train["attacker"])

y = test["attacker"].astype(int)
scores = model.predict_proba(test[features])[:, 1]
pred = (scores > 0.5).astype(int)

print("Matched Train messages:", len(train))
print("Matched Test messages:", len(test))
print("Accuracy:", accuracy_score(y, pred))
print("F1:", f1_score(y, pred))
print("ROC_AUC:", roc_auc_score(y, scores))
print("MATCHED TRAINING AUDIT COMPLETED")
