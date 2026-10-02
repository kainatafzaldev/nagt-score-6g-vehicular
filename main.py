import json, time, gc
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, mean_squared_error

from autograd import Tensor, Adam, param
from models import (NoiseGate, GATLayer, LSTM, SelfAttentionLayer, MultiHead,
                     multitask_loss, linear_params, linear)
import data_prep as dp

np.random.seed(42)
t0 = time.time()

# ---------------- 1. load + window + filter ----------------
DATA_PATH = __import__("os").environ.get("DATA_PATH", "./dataset.csv")
df = dp.load(DATA_PATH)
windows = dp.make_windows(df)
used_rows = np.sort(np.unique(np.concatenate([w["rows"] for w in windows])))
df = df.loc[used_rows].reset_index(drop=True)
windows = dp.make_windows(df)  # rebuild on cleaned/reindexed df so ordering matches concat order
N = len(df)
print(f"[{time.time()-t0:.1f}s] rows kept: {N}, windows: {len(windows)}")

seq_row_idx, seq_mask, seq_splits = dp.build_sequences(df)
print(f"[{time.time()-t0:.1f}s] sequences: {seq_row_idx.shape}")

# ---------------- 2. features ----------------
raw, noise, other, delay = dp.build_features(df)
train_mask_rows = (df["split"] == "Train").to_numpy()

def fit_transform(train_vals, full_vals):
    mu = train_vals.mean(axis=0, keepdims=True)
    sd = train_vals.std(axis=0, keepdims=True) + 1e-8
    return (full_vals - mu) / sd

raw_s = fit_transform(raw[train_mask_rows], raw)
noise_s = fit_transform(noise[train_mask_rows], noise)
dist = other[:, [0]]
susc = other[:, [1]]
dist_s = fit_transform(dist[train_mask_rows], dist)
other_s = np.concatenate([dist_s, susc], axis=1)
delay_mu, delay_sd = delay[train_mask_rows].mean(), delay[train_mask_rows].std() + 1e-8
delay_s = ((delay - delay_mu) / delay_sd).reshape(-1, 1)

targets_anom = df["attacker"].to_numpy(dtype=np.float64).reshape(-1, 1)
targets_deg = (1 - df["successful_delivery"].to_numpy(dtype=np.float64)).reshape(-1, 1)
targets_prop = df["active_current_window"].to_numpy(dtype=np.float64).reshape(-1, 1)
targets_time = delay_s

adj_list = []
for w in windows:
    n = len(w["nodes"])
    A = np.eye(n)
    A[w["s_idx"], w["r_idx"]] = 1
    A[w["r_idx"], w["s_idx"]] = 1
    adj_list.append(A)

print(f"[{time.time()-t0:.1f}s] median nodes/window: {np.median([len(w['nodes']) for w in windows]):.0f}, "
      f"max: {max(len(w['nodes']) for w in windows)}")

RAW_HID = 16  # GAT hidden size


def compute_graph_ctx(gat):
    """Full forward pass of the shared GAT layer over every window; returns a
    Tensor (N, 2*RAW_HID) of concatenated [sender_emb, receiver_emb] per message,
    row-aligned with df (windows partition df in row order)."""
    pieces = []
    for w, A in zip(windows, adj_list):
        n = len(w["nodes"])
        recv_state = raw_s[w["rows"], :5]
        send_state = raw_s[w["rows"], 5:10]
        acc = np.zeros((n, 5))
        cnt = np.zeros((n, 1))
        np.add.at(acc, w["s_idx"], send_state)
        np.add.at(cnt, w["s_idx"], 1)
        np.add.at(acc, w["r_idx"], recv_state)
        np.add.at(cnt, w["r_idx"], 1)
        node_feat = acc / np.maximum(cnt, 1)
        node_x = Tensor(node_feat, requires_grad=False)
        node_emb = gat(node_x, A)                       # (n, RAW_HID)
        sender_emb = node_emb[w["s_idx"]]                # (len(rows), RAW_HID)
        receiver_emb = node_emb[w["r_idx"]]
        pieces.append(Tensor.concat([sender_emb, receiver_emb], axis=-1))
    return Tensor.concat(pieces, axis=0)                 # (N, 2*RAW_HID)


def gather_seq(x_np):
    """x_np: (N, d) numpy array -> (n_seq, L, d) numpy array via seq_row_idx, 0-padded."""
    safe_idx = np.where(seq_row_idx < 0, 0, seq_row_idx)
    return x_np[safe_idx]


seq_mask3 = seq_mask[:, :, None]
y_anom_seq = gather_seq(targets_anom)
y_deg_seq = gather_seq(targets_deg)
y_prop_seq = gather_seq(targets_prop)
y_time_seq = gather_seq(targets_time)
split_of_seq = seq_splits  # (n_seq,)


def build_model(name):
    """Returns dict of components and a forward() closure appropriate to the variant."""
    use_gate = name == "NAGT-Score"
    use_graph = name in ("GNN", "GNN+Transformer", "NAGT-Score")
    temporal = {"LSTM": "lstm", "Transformer": "attn", "GNN": "none",
                "GNN+Transformer": "attn", "NAGT-Score": "attn"}[name]

    comp = {}
    d = 0
    if use_gate:
        comp["gate"] = NoiseGate(10)
        d += 10  # gated raw
    else:
        d += 10  # raw (ungated) still available to non-NAGT graph/temporal models
    d += 10  # noise magnitudes always available as explicit features
    d += 2   # other (distance, susceptible_previous_window)
    if use_graph:
        comp["gat"] = GATLayer(5, RAW_HID)
        d += 2 * RAW_HID

    d_model = 32
    comp["proj_W"], comp["proj_b"] = linear_params(d, d_model)
    if temporal == "lstm":
        comp["temporal"] = LSTM(d_model, d_model)
    elif temporal == "attn":
        comp["temporal"] = SelfAttentionLayer(d_model)
    comp["heads"] = MultiHead(d_model)

    def all_params():
        ps = []
        for k, v in comp.items():
            if hasattr(v, "params"):
                ps += v.params()
        ps += [comp["proj_W"], comp["proj_b"]]
        return ps

    def forward(row_idx_2d, mask2d):
        """row_idx_2d/mask2d: (n_batch, L) slices of seq_row_idx / seq_mask for this mini-batch."""
        if use_graph:
            gctx = compute_graph_ctx(comp["gat"])  # (N, 2*RAW_HID), full dataset (cheap)
        raw_t = Tensor(raw_s, requires_grad=False)
        noise_t = Tensor(noise_s, requires_grad=False)
        other_t = Tensor(other_s, requires_grad=False)
        if use_gate:
            gated, _ = comp["gate"](raw_t, noise_t)
            feat_parts = [gated, noise_t, other_t]
        else:
            feat_parts = [raw_t, noise_t, other_t]
        if use_graph:
            feat_parts.append(gctx)
        X = Tensor.concat(feat_parts, axis=-1)  # (N, d)

        safe_idx = np.where(row_idx_2d < 0, 0, row_idx_2d)
        Xseq = X[safe_idx]  # (n_batch, L, d)
        Xproj = linear(Xseq, comp["proj_W"], comp["proj_b"]).relu()  # (n_batch, L, d_model)

        if temporal == "lstm":
            T = Xproj.shape[1]
            x_list = [Xproj[:, t, :] for t in range(T)]
            hs = comp["temporal"].forward_steps(x_list, None)
            H = Tensor.stack(hs, axis=1)
        elif temporal == "attn":
            H = comp["temporal"](Xproj, mask2d)
        else:
            H = Xproj  # per-position MLP only, no temporal mixing

        preds = comp["heads"](H)
        return preds

    return comp, all_params, forward


BATCH = 120


def batches(n, bs):
    idx = np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def evaluate(forward):
    n_seq = seq_row_idx.shape[0]
    preds_accum = {"anomaly": [], "degradation": [], "propagation": [], "timing": []}
    for b in batches(n_seq, 200):
        preds = forward(seq_row_idx[b], seq_mask[b])
        p_anom, p_deg, p_prop, y_time = preds
        preds_accum["anomaly"].append(p_anom.data)
        preds_accum["degradation"].append(p_deg.data)
        preds_accum["propagation"].append(p_prop.data)
        preds_accum["timing"].append(y_time.data)
        del preds
        gc.collect()
    p_anom = np.concatenate(preds_accum["anomaly"], axis=0)
    p_deg = np.concatenate(preds_accum["degradation"], axis=0)
    p_prop = np.concatenate(preds_accum["propagation"], axis=0)
    y_time = np.concatenate(preds_accum["timing"], axis=0)

    m = seq_mask.astype(bool)
    results = {}
    for split in ["Train", "Validation", "Test"]:
        sel_seq = split_of_seq == split
        mm = m[sel_seq]
        out = {}
        for hname, p, y in [("anomaly", p_anom, y_anom_seq), ("degradation", p_deg, y_deg_seq),
                             ("propagation", p_prop, y_prop_seq)]:
            pv = p[sel_seq][mm]
            yv = y[sel_seq][:, :, 0][mm]
            pred_bin = (pv[:, 0] > 0.5).astype(int)
            try:
                auc = roc_auc_score(yv, pv[:, 0]) if len(np.unique(yv)) > 1 else float("nan")
            except Exception:
                auc = float("nan")
            out[hname] = {
                "acc": accuracy_score(yv, pred_bin),
                "f1": f1_score(yv, pred_bin, zero_division=0),
                "auc": auc,
            }
        tv = y_time_seq[sel_seq][:, :, 0][mm]
        pv_t = y_time[sel_seq][mm][:, 0]
        out["timing_rmse"] = float(np.sqrt(mean_squared_error(tv, pv_t)))
        results[split] = out
    return results


def train_model(name, epochs=15, lr=0.02, seed=0):
    np.random.seed(seed)          # controls weight initialization (param() uses np.random.randn)
    comp, all_params, forward = build_model(name)
    opt = Adam(all_params(), lr=lr)
    train_seq_ids = np.where(split_of_seq == "Train")[0]
    rng = np.random.RandomState(seed)   # controls mini-batch shuffling order

    for ep in range(epochs):
        rng.shuffle(train_seq_ids)
        ep_losses = []
        for b in batches(len(train_seq_ids), BATCH):
            ids = train_seq_ids[b]
            opt.zero_grad()
            preds = forward(seq_row_idx[ids], seq_mask[ids])
            loss, parts = multitask_loss(
                preds,
                (y_anom_seq[ids], y_deg_seq[ids], y_prop_seq[ids], y_time_seq[ids]),
                seq_mask3[ids],
            )
            loss.backward()
            opt.step()
            ep_losses.append(loss.data.item())
            del preds, loss, parts
            gc.collect()
        if ep % 3 == 0 or ep == epochs - 1:
            print(f"  [{name}] epoch {ep:2d} mean_loss={np.mean(ep_losses):.4f} "
                  f"[{time.time()-t0:.1f}s]", flush=True)
    result = evaluate(forward)
    gc.collect()
    return result


if __name__ == "__main__":
    import os
    SEEDS = [int(s) for s in os.environ.get("SEEDS", "0").split(",")]
    EPOCHS = int(os.environ.get("EPOCHS", 15))
    all_results = {}  # all_results[model][seed] = result dict
    for name in ["LSTM", "Transformer", "GNN", "GNN+Transformer", "NAGT-Score"]:
        all_results[name] = {}
        for seed in SEEDS:
            print(f"=== Training {name} (seed={seed}) ===", flush=True)
            res = train_model(name, epochs=EPOCHS, seed=seed)
            all_results[name][seed] = res
            print(f"  -> Test anomaly F1={res['Test']['anomaly']['f1']:.4f} "
                  f"AUC={res['Test']['anomaly']['auc']:.4f} [{time.time()-t0:.1f}s]", flush=True)
    OUT_PATH = os.environ.get("OUT_PATH", "./results_multiseed.json")
    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"DONE in {time.time()-t0:.1f}s")
