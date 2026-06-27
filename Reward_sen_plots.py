"""
plot_exp5.py — Reward-Shaping Sensitivity Plots
================================================
Reads results/exp5_coord_sweep.json and produces two figures:

  Figure 1: Mean reward vs each weight (2x2 subplots)
  Figure 2: Dropped requests vs each weight (2x2 subplots)

Each subplot sweeps one weight while the others are held at nominal.
Error bands show ±1 std across episodes.

Usage:
    python plot_exp5.py
    python plot_exp5.py --input results/exp5_coord_sweep.json --out_dir plots/
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── config ────────────────────────────────────────────────────────────────────
WEIGHT_LABELS = {
    "alpha": r"$\alpha$ (server cost)",
    "beta":  r"$\beta$ (queue length)",
    "gamma": r"$\gamma$ (drop penalty)",
    "omega": r"$\omega$ (thrash penalty)",
}
WEIGHT_ORDER = ["alpha", "beta", "gamma", "omega"]

AGENT_STYLE = {
    "ppo":      {"color": "#2a78d6", "marker": "o", "ls": "-",  "label": "PPO"},
    "dqn":      {"color": "#e34948", "marker": "s", "ls": "-",  "label": "DQN"},
    "baseline": {"color": "#1baf7a", "marker": "^", "ls": "--", "label": "Rule-based"},
    "random":   {"color": "#888780", "marker": "x", "ls": ":",  "label": "Random"},
}

METRICS = [
    ("reward",    "Mean total reward",          False),
    ("dropped",   "Mean dropped requests",      False),
    ("cost",      "Mean operational cost",      False),
    ("queue_occ", "Mean queue occupancy (frac)", False),
]


# ── helpers ───────────────────────────────────────────────────────────────────
def extract(data: dict, agent: str, weight: str, metric: str):
    """Return (x_values, means, stds) arrays sorted by x."""
    rec = data.get(agent, {}).get(weight, {})
    xs, means, stds = [], [], []
    for str_val, m in rec.items():
        xs.append(float(str_val))
        means.append(m[metric]["mean"])
        stds.append(m[metric]["std"])
    order = np.argsort(xs)
    return (np.array(xs)[order],
            np.array(means)[order],
            np.array(stds)[order])


def make_figure(data: dict, metric_key: str, metric_label: str,
                agents: list[str], out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flatten()

    for ax, weight in zip(axes, WEIGHT_ORDER):
        for agent in agents:
            if agent not in data:
                continue
            xs, means, stds = extract(data, agent, weight, metric_key)
            if len(xs) == 0:
                continue
            st = AGENT_STYLE[agent]
            ax.plot(xs, means, color=st["color"], marker=st["marker"],
                    ls=st["ls"], lw=1.8, ms=5, label=st["label"])
            ax.fill_between(xs, means - stds, means + stds,
                            color=st["color"], alpha=0.12)

        ax.set_title(WEIGHT_LABELS[weight], fontsize=11, pad=6)
        ax.set_xlabel("Weight value", fontsize=9)
        ax.set_ylabel(metric_label, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, lw=0.4, alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)

        # log x-axis only when values span more than one order of magnitude
        xs_all = []
        for agent in agents:
            xs_tmp, _, _ = extract(data, agent, weight, metric_key)
            xs_all.extend(xs_tmp.tolist())
        if len(xs_all) and max(xs_all) / max(min(xs_all), 1e-9) >= 10:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    # shared legend below the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(agents),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Exp 5 — Reward-shaping sensitivity: {metric_label}",
                 fontsize=13, fontweight="medium", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",   default="results/exp5_coord_sweep.json")
    ap.add_argument("--out_dir", default="results/plots")
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)

    # agents present in the file, in preferred display order
    order = ["ppo", "dqn", "baseline", "random"]
    agents = [a for a in order if a in data]

    print(f"Agents found: {agents}")
    print(f"Generating plots in '{args.out_dir}/' ...\n")
    c=9
    for metric_key, metric_label, _ in METRICS:
        c+=1
        out_path = os.path.join(args.out_dir, f"plot_{c}_{metric_key}.png")
        make_figure(data, metric_key, metric_label, agents, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()