"""
Experiment 5 — Reward-Shaping Sensitivity (Train-time γ sweep)
==============================================================
Retrains PPO and DQN under each γ value, then evaluates all four agents
(PPO, DQN, Rule-based baseline, Random) in an env built with that same γ.

Baseline and Random don't need retraining — their behaviour is independent
of the reward signal — but they are evaluated under each γ so their raw
metric numbers (drops, cost, queue) are comparable on the same scale.

γ values: [5, 10, 50, 100, 200]   (nominal = 50)

Output
------
results/exp5_gamma_sweep.json   — full metrics per agent × γ
plots/exp5_gamma_*.png          — one figure per metric

Directory layout produced
-------------------------
models/gamma_{g}/
    ppo_best/best_model.zip
    ppo_vecnorm.pkl
    dqn_best/best_model.zip
    dqn_vecnorm.pkl

Usage
-----
    # full run
    python exp5_gamma_train_sweep.py

    # smoke test (tiny timesteps, 2 episodes)
    python exp5_gamma_train_sweep.py --timesteps 20000 --episodes 2

    # skip training if models already exist
    python exp5_gamma_train_sweep.py --eval_only
"""

import argparse
import json
import os
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from baseline_agent import RuleBasedBaseline
from cloud_env import CloudScalingEnv  # noqa: F401
from env_factory import make_env, make_vec_env

if "CloudScaling-v1" not in gym.envs.registry:
    gym.register(id="CloudScaling-v1", entry_point="cloud_env:CloudScalingEnv")

#experiment config
GAMMA_VALUES  = [5.0, 10.0, 20.0, 30.0, 50.0]
NOMINAL       = {"alpha": 1.0, "beta": 0.1, "gamma": 50.0, "omega": 5.0}

# PPO hyperparams (from your Optuna run)
PPO_KWARGS = dict(
    learning_rate = 1.4377739650640294e-05,
    n_steps       = 4096,
    batch_size    = 64,
    n_epochs      = 5,
    gamma         = 0.9660969058740851,   # discount factor — NOT reward weight
    gae_lambda    = 0.9175042883882351,
    clip_range    = 0.1291937132179491,
    ent_coef      = 0.012344366981406195,
    vf_coef       = 0.9315525456344808,
    max_grad_norm = 0.48170949108431693,
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
)

# DQN hyperparams — sensible defaults; swap in your tuned values if available
DQN_KWARGS = dict(
    learning_rate        = 1e-4,
    buffer_size          = 100_000,
    learning_starts      = 10_000,
    batch_size           = 64,
    tau                  = 1.0,
    gamma                = 0.99,
    train_freq           = 4,
    gradient_steps       = 1,
    target_update_interval = 1_000,
    exploration_fraction   = 0.1,
    exploration_final_eps  = 0.05,
    policy_kwargs        = dict(net_arch=[256, 256]),
)

AGENT_STYLE = {
    "ppo":      {"color": "#2a78d6", "marker": "o", "ls": "-",  "label": "PPO"},
    "dqn":      {"color": "#e34948", "marker": "s", "ls": "-",  "label": "DQN"},
    "baseline": {"color": "#1baf7a", "marker": "^", "ls": "--", "label": "Rule-based"},
    "random":   {"color": "#888780", "marker": "x", "ls": ":",  "label": "Random"},
}


#helpers
def reward_weights_for(g: float) -> tuple:
    return (NOMINAL["alpha"], NOMINAL["beta"], g, NOMINAL["omega"])


def model_dir(g: float) -> str:
    return f"./models/gamma_{int(g)}"


#training
def train_ppo(g: float, timesteps: int, device: str) -> tuple[str, str]:
    """Train PPO with reward weight γ=g. Returns (model_path, vecnorm_path)."""
    rw   = reward_weights_for(g)
    mdir = model_dir(g)
    best_path   = f"{mdir}/ppo_best"
    vecnorm_path = f"{mdir}/ppo_vecnorm.pkl"
    os.makedirs(best_path, exist_ok=True)

    print(f"\n  [PPO] γ={g}  timesteps={timesteps:,}")

    train_env = make_vec_env(n_envs=8, seed=0, use_subprocess=True,
                             norm_reward=True, reward_weights=rw)
    eval_env = VecNormalize(
        DummyVecEnv([make_env(rank=100, seed=0, reward_weights=rw)]),
        norm_obs=True, norm_reward=False, clip_obs=5.0, gamma=0.99)
    eval_env.training = False

    model = PPO(
        policy="MlpPolicy", env=train_env,
        tensorboard_log=f"./logs/ppo_gamma_{int(g)}/",
        device=device, verbose=0, **PPO_KWARGS
    )

    eval_cb = EvalCallback(
        eval_env, best_model_save_path=best_path,
        eval_freq=10_000, n_eval_episodes=5, deterministic=True, verbose=0,
    )
    model.learn(total_timesteps=timesteps, callback=eval_cb,
                reset_num_timesteps=True)

    train_env.save(vecnorm_path)
    train_env.close()
    eval_env.close()

    return f"{best_path}/best_model.zip", vecnorm_path


def train_dqn(g: float, timesteps: int, device: str) -> tuple[str, str]:
    """Train DQN with reward weight γ=g. Returns (model_path, vecnorm_path)."""
    rw   = reward_weights_for(g)
    mdir = model_dir(g)
    best_path    = f"{mdir}/dqn_best"
    vecnorm_path = f"{mdir}/dqn_vecnorm.pkl"
    os.makedirs(best_path, exist_ok=True)

    print(f"\n  [DQN] γ={g}  timesteps={timesteps:,}")

    # DQN works best with a single env (experience replay doesn't benefit
    # from SubprocVecEnv the same way on-policy PPO does)
    train_env = VecNormalize(
        DummyVecEnv([make_env(rank=0, seed=0, reward_weights=rw)]),
        norm_obs=True, norm_reward=True, clip_obs=5.0, gamma=0.99)
    eval_env = VecNormalize(
        DummyVecEnv([make_env(rank=100, seed=0, reward_weights=rw)]),
        norm_obs=True, norm_reward=False, clip_obs=5.0, gamma=0.99)
    eval_env.training = False

    model = DQN(
        policy="MlpPolicy", env=train_env,
        tensorboard_log=f"./logs/dqn_gamma_{int(g)}/",
        device=device, verbose=0, **DQN_KWARGS
    )

    eval_cb = EvalCallback(
        eval_env, best_model_save_path=best_path,
        eval_freq=10_000, n_eval_episodes=5, deterministic=True, verbose=0,
    )
    model.learn(total_timesteps=timesteps, callback=eval_cb,
                reset_num_timesteps=True)

    train_env.save(vecnorm_path)
    train_env.close()
    eval_env.close()

    return f"{best_path}/best_model.zip", vecnorm_path


# evaluation
def evaluate(agent, g: float, vecnorm_path: str | None,
             n_episodes: int, seed: int, ep_len: int = 1000) -> dict:
    """Evaluate *agent* in an env with reward weight γ=g."""
    rw     = reward_weights_for(g)
    env_fn = make_env(rank=99, seed=seed, reward_weights=rw)
    vec    = DummyVecEnv([env_fn])

    if vecnorm_path and os.path.exists(vecnorm_path):
        vec = VecNormalize.load(vecnorm_path, vec)
        vec.training  = False
        vec.norm_reward = False

    rewards, costs, drops, qocc = [], [], [], []

    for _ in range(n_episodes):
        obs  = vec.reset()
        done = [False]
        R = c = d = q = steps = 0
        while not done[0]:
            if agent == "random":
                action = np.array([vec.action_space.sample()])
            elif hasattr(agent, "predict"):
                raw    = obs[0]
                result = agent.predict(raw, deterministic=True)
                action_val = result[0] if isinstance(result, tuple) else result
                action = np.atleast_1d(np.array(action_val))
            obs, r, done, info = vec.step(action)
            R += r[0];  c += info[0]["active"]
            d += info[0]["dropped"];  q += info[0]["queue"]
            steps += 1
        rewards.append(R);  costs.append(c)
        drops.append(d);    qocc.append(q / (steps * 500))

    vec.close()
    agg = lambda x: {"mean": float(np.mean(x)), "std": float(np.std(x))}
    return {"reward": agg(rewards), "cost": agg(costs),
            "dropped": agg(drops), "queue_occ": agg(qocc)}


#plotting
PLOT_METRICS = [
    ("reward",    "Mean total reward"),
    ("dropped",   "Mean dropped requests"),
    ("cost",      "Mean operational cost"),
    ("queue_occ", "Mean queue occupancy (frac)"),
]

def plot_results(results: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    agents = [a for a in ["ppo", "dqn", "baseline", "random"] if a in results]

    for metric_key, metric_label in PLOT_METRICS:
        fig, ax = plt.subplots(figsize=(8, 5))

        for agent in agents:
            st   = AGENT_STYLE[agent]
            xs   = sorted(results[agent].keys(), key=float)
            ys   = [results[agent][g][metric_key]["mean"] for g in xs]
            errs = [results[agent][g][metric_key]["std"]  for g in xs]
            xs_f = [float(x) for x in xs]

            ax.plot(xs_f, ys, color=st["color"], marker=st["marker"],
                    ls=st["ls"], lw=2, ms=6, label=st["label"])
            ax.fill_between(xs_f,
                            np.array(ys) - np.array(errs),
                            np.array(ys) + np.array(errs),
                            color=st["color"], alpha=0.12)

        # mark nominal γ
        ax.axvline(x=50.0, color="#b4b2a9", ls="--", lw=1.2, label="Nominal γ=50")

        ax.set_xlabel(r"$\gamma$ (drop penalty weight)", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f"Exp 5 — Train-time γ sweep: {metric_label}", fontsize=12)
        ax.set_xscale("log")
        ax.set_xticks([float(g) for g in GAMMA_VALUES])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.tick_params(labelsize=9)
        ax.grid(True, lw=0.4, alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9, frameon=False)

        out = os.path.join(out_dir, f"exp5_gamma_{metric_key}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out}")


#main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_000_000,
                    help="Training timesteps per config (use 20000 for smoke test)")
    ap.add_argument("--episodes",  type=int, default=10,
                    help="Evaluation episodes per agent × γ")
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--device",    default="cpu", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--eval_only", action="store_true",
                    help="Skip training; load existing models and evaluate only")
    ap.add_argument("--out_dir",   default="plots")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    results = {}   # agent → gamma_str → metrics

    baseline = RuleBasedBaseline(n_max=10, q_max=500)

    t0 = time.perf_counter()

    for g in GAMMA_VALUES:
        g_key = str(g)
        mdir  = model_dir(g)

        print(f"\n{'='*60}")
        print(f"  γ = {g}")
        print(f"{'='*60}")

        # train
        ppo_model_path = f"{mdir}/ppo_best/best_model.zip"
        ppo_vecnorm    = f"{mdir}/ppo_vecnorm.pkl"
        dqn_model_path = f"{mdir}/dqn_best/best_model.zip"
        dqn_vecnorm    = f"{mdir}/dqn_vecnorm.pkl"

        if not args.eval_only:
            ppo_model_path, ppo_vecnorm = train_ppo(g, args.timesteps, args.device)
            dqn_model_path, dqn_vecnorm = train_dqn(g, args.timesteps, args.device)
        else:
            print(f"  [--eval_only] skipping training, loading existing models")

        # load trained models
        agents_eval = {
            "baseline": baseline,
            "random":   "random",
        }

        if os.path.exists(ppo_model_path):
            agents_eval["ppo"] = PPO.load(ppo_model_path)
            print(f" PPO loaded from {ppo_model_path}")
        else:
            print(f"PPO model not found at {ppo_model_path} — skipping")

        if os.path.exists(dqn_model_path):
            agents_eval["dqn"] = DQN.load(dqn_model_path)
            print(f"DQN loaded from {dqn_model_path}")
        else:
            print(f"DQN model not found at {dqn_model_path} — skipping")

        # evaluate
        vecnorm_map = {
            "ppo":      ppo_vecnorm,
            "dqn":      dqn_vecnorm,
            "baseline": None,
            "random":   None,
        }

        print(f"\n  Evaluating ({args.episodes} episodes each) ...")
        for agent_name, agent in agents_eval.items():
            m = evaluate(agent, g, vecnorm_map[agent_name],
                         args.episodes, args.seed)
            results.setdefault(agent_name, {})[g_key] = m
            print(f"    {agent_name:<10}  reward={m['reward']['mean']:>10.1f}  "
                  f"dropped={m['dropped']['mean']:>8.1f}  "
                  f"cost={m['cost']['mean']:>8.1f}")

    #save JSON 
    out_json = "results/exp5_gamma_sweep.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_json}")

    #plot
    print("\nGenerating plots ...")
    plot_results(results, args.out_dir)

    print(f"\nTotal wall time: {time.perf_counter()-t0:.1f}s "
          f"({(time.perf_counter()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()