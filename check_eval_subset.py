"""Check which messages are actually evaluated after the 80-message-per-sender truncation
in data_prep.build_sequences, and recompute the baselines on that subset.
Run from the repo folder:  DATA_PATH=./dataset.csv python3 check_eval_subset.py
"""
import os
import numpy as np
import data_prep as dp

df = dp.load(os.environ.get("DATA_PATH", "./dataset.csv"))
windows = dp.make_windows(df)
used = np.sort(np.unique(np.concatenate([w["rows"] for w in windows])))
df = df.loc[used].reset_index(drop=True)
seq_idx, mask, seq_splits = dp.build_sequences(df)

for split in ["Train", "Validation", "Test"]:
    sel = seq_splits == split
    rows = seq_idx[sel][mask[sel].astype(bool)]
    sub = df.loc[rows]
    n_all = int((df["split"] == split).sum())
    atk = float(sub["attacker"].mean())
    fail = float(1 - sub["successful_delivery"].mean())
    print(f"[{split}] messages in split={n_all}  evaluated={len(sub)} "
          f"({100*len(sub)/n_all:.1f}%)  senders={int(sel.sum())}")
    print(f"   attacker fraction (evaluated subset) = {atk:.3f} -> majority-class acc = {max(atk,1-atk):.3f}")
    print(f"   delivery-failure rate = {fail:.3f} -> F1 of 'predict all failures' = {2*fail/(1+fail):.3f}")
