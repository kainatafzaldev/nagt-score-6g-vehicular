import numpy as np
import pandas as pd

RAW_COLS = [
    "receiver_pos_x", "receiver_pos_y", "receiver_speed", "receiver_acceleration", "receiver_heading",
    "sender_pos_x", "sender_pos_y", "sender_speed", "sender_acceleration", "sender_heading",
]
NOISE_COLS = [
    "receiver_pos_noise_x", "receiver_pos_noise_y", "receiver_speed_noise", "receiver_acceleration_noise",
    "receiver_heading_noise",
    "sender_pos_noise_x", "sender_pos_noise_y", "sender_speed_noise", "sender_acceleration_noise",
    "sender_heading_noise",
]
OTHER_COLS = ["sender_distance_to_road_edge", "susceptible_previous_window"]

WINDOW_SIZE = 250
MAX_SEQ_LEN = 80


def load(path):
    df = pd.read_csv(path, low_memory=False)
    df = df.sort_values(["split", "rcvTime"]).reset_index(drop=True)
    return df


def build_features(df):
    raw = df[RAW_COLS].to_numpy(dtype=np.float64)
    noise = df[NOISE_COLS].to_numpy(dtype=np.float64)
    other = df[OTHER_COLS].to_numpy(dtype=np.float64)
    delay = np.log1p(df["delay_raw_time_units"].to_numpy(dtype=np.float64))

    # standardize (fit stats will be passed in for train-only fitting)
    return raw, noise, other, delay


def standardize(train_vals, *others):
    mu = train_vals.mean(axis=0, keepdims=True)
    sd = train_vals.std(axis=0, keepdims=True) + 1e-8
    out = [(train_vals - mu) / sd]
    for o in others:
        out.append((o - mu) / sd)
    return out, mu, sd


def make_windows(df):
    """Return list of dicts, one per (split, window) with node index arrays and edge list."""
    windows = []
    for split, sub in df.groupby("split", sort=False):
        idx = sub.index.to_numpy()
        n = len(idx)
        for start in range(0, n, WINDOW_SIZE):
            rows = idx[start:start + WINDOW_SIZE]
            if len(rows) < 4:
                continue
            senders = df.loc[rows, "sender_id"].to_numpy()
            receivers = df.loc[rows, "source_vehicle_file"].to_numpy()
            nodes = pd.unique(np.concatenate([senders, receivers]))
            node_index = {v: i for i, v in enumerate(nodes)}
            s_idx = np.array([node_index[v] for v in senders])
            r_idx = np.array([node_index[v] for v in receivers])
            windows.append({
                "split": split, "rows": rows, "nodes": nodes,
                "s_idx": s_idx, "r_idx": r_idx,
            })
    return windows


def build_sequences(df, max_len=MAX_SEQ_LEN):
    """Group row indices by sender_id, sorted by rcvTime (already globally sorted by split,rcvTime
    upstream, so we resort within-split groups by sender). Truncate/pad to max_len.
    Returns: seq_row_idx (num_seq, max_len) int array with -1 padding, mask (num_seq, max_len),
    and a mapping split-> list of sequence ids for later split-wise metric aggregation."""
    seqs = []
    seq_splits = []
    for split, sub in df.groupby("split", sort=False):
        for sender, g in sub.groupby("sender_id", sort=False):
            rows = g.sort_values("rcvTime").index.to_numpy()
            rows = rows[:max_len]
            seqs.append(rows)
            seq_splits.append(split)
    L = max(len(r) for r in seqs)
    L = min(L, max_len)
    n = len(seqs)
    seq_row_idx = -np.ones((n, L), dtype=np.int64)
    mask = np.zeros((n, L), dtype=np.float64)
    for i, rows in enumerate(seqs):
        k = min(len(rows), L)
        seq_row_idx[i, :k] = rows[:k]
        mask[i, :k] = 1.0
    return seq_row_idx, mask, np.array(seq_splits)
