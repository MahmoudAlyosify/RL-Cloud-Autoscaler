"""
Experiment 4 — Cold-Start Test
===============================
Sweeps boot_delay in {0, 1, 3, 5, 10} and evaluates:
  - PPO agent (best_model.zip + VecNormalize stats)
  - Rule-based baseline (RuleBasedBaseline)
  - Random agent

Key insight: proactive scaling only matters when the system cannot react
instantly. A large boot_delay forces the agent to scale-out *before* traffic
arrives; a reactive/random policy will suffer visible drops.

Usage:
    python exp4_cold_start.py [--episodes 10] [--timesteps 1000]
"""

import argparse
import json
import os
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO ,DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from cloud_env import CloudScalingEnv          # noqa: F401
from env_factory import make_env
from baseline_agent import RuleBasedBaseline

# ── register env ──────────────────────────────────────────────────────────────
if "CloudScaling-v1" not in gym.envs.registry:
    gym.register(id="CloudScaling-v1", entry_point="cloud_env:CloudScalingEnv")

BOOT_DELAYS = [0, 1, 3, 5, 10]


# ── evaluation helpers ─────────────────────────────────────────────────────────

def _make_eval_env(boot_delay: int, seed: int, vecnorm_path: str | None = None):
    """Single-env DummyVecEnv, optionally wrapped in frozen VecNormalize."""
    env_fn = make_env(rank=99, seed=seed, boot_delay=boot_delay)
    vec = DummyVecEnv([env_fn])
    if vecnorm_path and os.path.exists(vecnorm_path):
        vec = VecNormalize.load(vecnorm_path, vec)
        vec.training = False
        vec.norm_reward = False
    return vec


def evaluate(agent, boot_delay: int, n_episodes: int, seed: int,
             vecnorm_path: str | None = None, ep_len: int = 1000):
    """Run *agent* for n_episodes and return aggregated metrics."""
    env = _make_eval_env(boot_delay, seed, vecnorm_path)

    rewards, costs, drops, qocc, response_times = [], [], [], [], []

    for ep in range(n_episodes):
        obs = env.reset()
        done = [False]
        R = c = d = q = steps = 0

        while not done[0]:
            if agent == "random":
                action = np.array([env.action_space.sample()])
            elif hasattr(agent, "predict"):
                action, _ = agent.predict(obs, deterministic=True)
            else:
                raw_obs = obs[0]          # baseline expects raw obs
                action_scalar, _ = agent.predict(raw_obs, deterministic=True)
                action = np.array([action_scalar])

            obs, r, done, info = env.step(action)
            R += r[0]
            c += info[0]["active"]
            d += info[0]["dropped"]
            q += info[0]["queue"]
            steps += 1

        rewards.append(R)
        costs.append(c)
        drops.append(d)
        qocc.append(q / (steps * 500))      # fractional queue occupancy

    env.close()

    def agg(x):
        return {"mean": float(np.mean(x)), "std": float(np.std(x))}

    return {
        "reward":    agg(rewards),
        "cost":      agg(costs),
        "dropped":   agg(drops),
        "queue_occ": agg(qocc),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes",   type=int, default=10)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--ep_len",     type=int, default=1000)
    ap.add_argument("--dqn_model", default="./models/best_dqn/best_model.zip")
    ap.add_argument("--ppo_model",  default="./models/best_ppo/best_model.zip")
    ap.add_argument("--vecnorm",    default="./models/vecnormalize_ppo.pkl")
    args = ap.parse_args()

    # load PPO once (model weights are boot-delay agnostic)
    ppo_model = None
    if os.path.exists(args.ppo_model):
        ppo_model = PPO.load(args.ppo_model)
        print(f"Loaded PPO")
    else:
        print(f"PPO model not found — skipping PPO")
    dqn_model = None
    if os.path.exists(args.dqn_model):
        dqn_model = DQN.load(args.dqn_model)
        print(f"Loaded DQN")
    else:
        print(f"DQN model not found — skipping DQN")
    baseline = RuleBasedBaseline(n_max=10, q_max=500)

    agents = {
    "baseline": baseline,
    "random": "random"
}

    if ppo_model:
        agents["ppo"] = ppo_model

    if dqn_model:
        agents["dqn"] = dqn_model
    results = {}   # agent → boot_delay → metrics

    header = f"{'agent':<12} {'boot_delay':>10} {'reward':>12} {'dropped':>10} {'cost':>10} {'queue_occ':>12}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    for agent_name, agent in agents.items():
        results[agent_name] = {}
        if agent_name == "ppo":
            vecnorm = "./models/vecnormalize_ppo.pkl"

        elif agent_name == "dqn":
            vecnorm = "./models/vecnormalize_dqn.pkl"

        else:
            vecnorm = None
        for bd in BOOT_DELAYS:
            t0 = time.perf_counter()
            m = evaluate(agent, bd, args.episodes, args.seed,
                         vecnorm_path=vecnorm, ep_len=args.ep_len)
            elapsed = time.perf_counter() - t0
            results[agent_name][bd] = m

            print(
                f"{agent_name:<12} {bd:>10d} "
                f"{m['reward']['mean']:>12.1f} "
                f"{m['dropped']['mean']:>10.1f} "
                f"{m['cost']['mean']:>10.1f} "
                f"{m['queue_occ']['mean']:>12.4f}"
                f"   [{elapsed:.1f}s]"
            )

    print(sep + "\n")

    os.makedirs("results", exist_ok=True)
    out = "results/exp4_cold_start.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out}")

    # ── proactivity advantage summary ─────────────────────────────────────────
    if ppo_model and "baseline" in results:
        print("\n── Proactivity Advantage (PPO reward − baseline reward) ──")
        print(f"  {'boot_delay':>10} {'Δ reward':>12} {'Δ dropped':>12}")
        for bd in BOOT_DELAYS:
            dr = results["ppo"][bd]["reward"]["mean"] - results["baseline"][bd]["reward"]["mean"]
            dd = results["baseline"][bd]["dropped"]["mean"] - results["ppo"][bd]["dropped"]["mean"]
            print(f"  {bd:>10d} {dr:>+12.1f} {dd:>+12.1f}")
        print()


if __name__ == "__main__":
    main()
