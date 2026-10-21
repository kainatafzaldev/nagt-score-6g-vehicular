"""Attacker-density sweep for the proxy security score S (checklist item 13).

Resamples the Test messages of NVBD to a fixed size and a target attacker
fraction, recomputes the four proxy indicators exactly as in the paper, and
writes density_sweep.csv plus density_sweep.pdf/.png.

  rho   = 1 - attacker fraction
  Res   = mean over complete 250-message windows of the delivery rate
  pi    = 1 (no propagation is modelled)
  delta = exp(-mean_delay_ms / 200), delay unit assumed to be 1 ns
  S     = equal-weight geometric mean

Usage:  python3 density_sweep.py                      (expects ./dataset.csv)
        DATA_PATH=/path/to/dataset.csv N_TOTAL=10000 REPS=30 python3 density_sweep.py
Needs: numpy, pandas, matplotlib.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = os.environ.get("DATA_PATH", "./dataset.csv")
N_TOTAL = int(os.environ.get("N_TOTAL", 10000))   # messages per resample
REPS = int(os.environ.get("REPS", 30))            # resamples per attacker fraction
FRACTIONS = [0.05, 0.10, 0.20, 0.40, 0.60]
WINDOW = 250
T_SAFE_MS = 200.0

cols = ["split", "rcvTime", "attacker", "successful_delivery", "delay_raw_time_units"]
df = pd.read_csv(DATA_PATH, usecols=cols)
test = df[df["split"] == "Test"].sort_values("rcvTime").reset_index(drop=True)
att = test[test["attacker"] == 1]
leg = test[test["attacker"] == 0]
print(f"Test messages: {len(test)} (attackers {len(att)}, legitimate {len(leg)})")


def indicators(sample):
    p = sample["attacker"].mean()
    rho = 1.0 - p
    deliv = sample["successful_delivery"].to_numpy(dtype=float)
    n_win = len(deliv) // WINDOW
    res = deliv[: n_win * WINDOW].reshape(n_win, WINDOW).mean(axis=1).mean()
    mean_delay_ms = sample["delay_raw_time_units"].mean() / 1e6
    delta = float(np.exp(-mean_delay_ms / T_SAFE_MS))
    pi = 1.0
    s = (rho * res * pi * delta) ** 0.25
    return rho, res, pi, delta, s


def band(s):
    return "Critical" if s < 0.25 else "Degraded" if s < 0.50 else "Adequate" if s < 0.75 else "Target"


rows = []
for frac in FRACTIONS:
    n_att = int(round(frac * N_TOTAL))
    n_leg = N_TOTAL - n_att
    if n_att > len(att) or n_leg > len(leg):
        raise SystemExit(f"Not enough messages for fraction {frac}: lower N_TOTAL")
    vals = []
    for rep in range(REPS):
        rng = np.random.RandomState(1000 * rep + int(frac * 100))
        sample = pd.concat([att.iloc[rng.choice(len(att), n_att, replace=False)],
                            leg.iloc[rng.choice(len(leg), n_leg, replace=False)]])
        sample = sample.sort_values("rcvTime").reset_index(drop=True)
        vals.append(indicators(sample))
    v = np.array(vals)
    m, sd = v.mean(axis=0), v.std(axis=0, ddof=1)
    rows.append({
        "attacker_fraction": frac, "N_total": N_TOTAL, "reps": REPS,
        "rho": m[0], "Res": m[1], "pi": m[2], "delta": m[3],
        "S_mean": m[4], "S_std": sd[4], "band": band(m[4]),
        # formula-only curve: Res, pi, delta held at their full-Test values
    })

# formula-only reference curve (what the paper's 66% figure comes from)
full = indicators(test)
res0, pi0, delta0 = full[1], full[2], full[3]
grid = np.linspace(0, 0.9, 181)
s_formula = ((1 - grid) * res0 * pi0 * delta0) ** 0.25
crossing = 1 - 0.75 ** 4 / (res0 * pi0 * delta0)   # S = 0.75 boundary
print(f"Full Test: rho={full[0]:.3f} Res={res0:.3f} pi={pi0:.3f} delta={delta0:.3f} S={full[4]:.3f}")
print(f"Formula-only: S stays >= 0.75 up to attacker fraction {crossing:.3f}")

out = pd.DataFrame(rows)
out.to_csv("density_sweep.csv", index=False, float_format="%.4f")
print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

fig, ax = plt.subplots(figsize=(3.5, 2.5))
for lo, hi, c, lab in [(0, 0.25, "#f4c7c3", "Critical"), (0.25, 0.5, "#fce8b2", "Degraded"),
                       (0.5, 0.75, "#d9ead3", "Adequate"), (0.75, 1.0, "#b6d7a8", "Target")]:
    ax.axhspan(lo, hi, color=c, alpha=0.6, lw=0)
    ax.text(0.89, (lo + hi) / 2, lab, ha="right", va="center", fontsize=6)
ax.plot(grid, s_formula, "k--", lw=1, label="formula only")
ax.errorbar(out["attacker_fraction"], out["S_mean"], yerr=out["S_std"], fmt="o-", ms=3,
            lw=1, capsize=2, color="C0", label="resampled Test")
ax.set_xlabel("Attacker fraction of messages", fontsize=8)
ax.set_ylabel("Proxy score $S$", fontsize=8)
ax.set_xlim(0, 0.9); ax.set_ylim(0, 1)
ax.tick_params(labelsize=7); ax.legend(fontsize=6, loc="lower left")
fig.tight_layout()
fig.savefig("density_sweep.pdf"); fig.savefig("density_sweep.png", dpi=300)
print("Saved density_sweep.csv, density_sweep.pdf, density_sweep.png")
