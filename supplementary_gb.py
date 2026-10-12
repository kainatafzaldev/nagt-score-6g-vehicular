"""Supplementary feature-dependence check (outside the five-model protocol).
Gradient boosting trained on Train rows, scored on Test rows (message level), WITHOUT
susceptible_previous_window, identifiers, or time columns.
Run from the repo folder:  DATA_PATH=./dataset.csv python3 supplementary_gb.py
Numbers may differ slightly from the instructor's check (feature choice / settings).
"""
import os
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import data_prep as dp

df = dp.load(os.environ.get("DATA_PATH", "./dataset.csv"))
tr, te = df[df["split"] == "Train"], df[df["split"] == "Test"]
RAW, NOISE, DIST = dp.RAW_COLS, dp.NOISE_COLS, ["sender_distance_to_road_edge"]
sets = {"all (raw+noise+road-edge)": RAW + NOISE + DIST,
        "without road-edge": RAW + NOISE,
        "kinematics only (raw)": RAW}
for name, cols in sets.items():
    m = HistGradientBoostingClassifier(random_state=0).fit(tr[cols], tr["attacker"])
    auc = roc_auc_score(te["attacker"], m.predict_proba(te[cols])[:, 1])
    print(f"{name:28s} Test AUC = {auc:.3f}")
best = max(max(roc_auc_score(te["attacker"], te[c]), 1 - roc_auc_score(te["attacker"], te[c])) for c in NOISE)
print(f"max single-feature AUC over the 10 noise columns = {best:.3f}")
print("mean road-edge distance: attackers = %.1f m, legitimate = %.1f m" % (
    df.loc[df.attacker == 1, DIST[0]].mean(), df.loc[df.attacker == 0, DIST[0]].mean()))
