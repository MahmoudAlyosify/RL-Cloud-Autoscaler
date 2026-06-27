"""
Experiment 5 — Reward-Shaping Sensitivity
==========================================
Coordinate sweep: vary one weight at a time (α, β, γ, ω), hold others at
nominal values from the project constitution.
Reveals which penalty term each algorithm is most sensitive to.

Usage:
    python exp5_reward_sensitivity.py
    python exp5_reward_sensitivity.py --episodes 10 --mode both
"""

import argparse
import json
import os
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env_factory import make_env
from baseline_agent import RuleBasedBaseline

if "CloudScaling-v1" not in gym.envs.registry:
    gym.register(id="CloudScaling-v1", entry_point="cloud_env:CloudScalingEnv")

# ── nominal weights (from project constitution / cloud_env defaults) ───────────
NOMINAL = {"alpha": 1.0, "beta": 0.1, "gamma": 50.0, "omega": 5.0}

# ── coordinate sweep grids ─────────────────────────────────────────────────────
COORD_GRIDS = {
    "alpha": [0.1, 0.5, 1.0, 2.0, 5.0],        # server-cost weight
    "beta":  [0.01, 0.05, 0.1, 0.5, 1.0],       # queue-length weight
    "gamma": [5.0, 10.0, 50.0, 100.0, 200.0],   # drop-penalty weight
    "omega": [0.0, 1.0, 5.0, 10.0, 20.0],       # thrash-penalty weight
}


# ── evaluation ─────────────────────────────────────────────────────────────────
def evaluate_config(agent, weights: dict, n_episodes: int, seed: int,
                    vecnorm_path: str | None, ep_len: int = 1000) -> dict:
    """Evaluate *agent* in an env instantiated with custom reward_weights."""
    reward_weights = (
        weights["alpha"],
        weights["beta"],
        weights["gamma"],
        weights["omega"],
    )

    env_fn = make_env(rank=99, seed=seed, reward_weights=reward_weights)
    vec = DummyVecEnv([env_fn])
    if vecnorm_path and os.path.exists(vecnorm_path):
        vec = VecNormalize.load(vecnorm_path, vec)
        vec.training = False
        vec.norm_reward = False

    rewards, costs, drops, qocc = [], [], [], []

    for _ in range(n_episodes):
        obs = vec.reset()
        done = [False]
        R = c = d = q = steps = 0
        while not done[0]:
            if agent == "random":
                action = np.array([vec.action_space.sample()])
            elif hasattr(agent, "predict"):
                raw = obs[0]
                result = agent.predict(raw, deterministic=True)
                action_val = result[0] if isinstance(result, tuple) else result
                action = np.atleast_1d(np.array(action_val))
            obs, r, done, info = vec.step(action)
            R += r[0];  c += info[0]["active"]
            d += info[0]["dropped"];  q += info[0]["queue"]
            steps += 1
        rewards.append(R);  costs.append(c);  drops.append(d)
        qocc.append(q / (steps * 500))

    vec.close()
    agg = lambda x: {"mean": float(np.mean(x)), "std": float(np.std(x))}
    return {"reward": agg(rewards), "cost": agg(costs),
            "dropped": agg(drops), "queue_occ": agg(qocc)}


# ── coordinate sweep ───────────────────────────────────────────────────────────
def run_coord_sweep(agents: dict, vecnorm_map: dict, n_episodes: int,
                    seed: int, ep_len: int) -> dict:
    """
    agents     : {name: model_or_"random"}
    vecnorm_map: {name: path_or_None}  — one entry per agent
    """
    results = {}

    for weight_name, grid in COORD_GRIDS.items():
        print(f"\n── Sweeping {weight_name} ──")
        for agent_name, agent in agents.items():
            results.setdefault(agent_name, {}).setdefault(weight_name, {})
            vn = vecnorm_map.get(agent_name)

            for val in grid:
                w = {**NOMINAL, weight_name: val}
                m = evaluate_config(agent, w, n_episodes, seed, vn, ep_len)
                results[agent_name][weight_name][val] = m
                print(f"  {agent_name:<10} {weight_name}={val:<8.3f} "
                      f"reward={m['reward']['mean']:>10.1f}  "
                      f"dropped={m['dropped']['mean']:>8.1f}")

    return results


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes",    type=int, default=5)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--ep_len",      type=int, default=1000)
    ap.add_argument("--ppo_model",   default="./models/best_ppo/best_model.zip")
    ap.add_argument("--ppo_vecnorm", default="./models/vecnormalize_ppo.pkl")
    ap.add_argument("--dqn_model",   default="./models/best_dqn/best_model.zip")
    ap.add_argument("--dqn_vecnorm", default="./models/vecnormalize_dqn.pkl")
    args = ap.parse_args()

    # ── load models ────────────────────────────────────────────────────────────
    agents: dict = {
        "baseline": RuleBasedBaseline(n_max=10, q_max=500),
        "random":   "random",
    }
    vecnorm_map: dict = {
        "baseline": None,
        "random":   None,
    }

    if os.path.exists(args.ppo_model):
        agents["ppo"] = PPO.load(args.ppo_model)
        vecnorm_map["ppo"] = args.ppo_vecnorm
        print(f"Loaded PPO")
    else:
        print(f"PPO model not found")

    if os.path.exists(args.dqn_model):
        agents["dqn"] = DQN.load(args.dqn_model)
        vecnorm_map["dqn"] = args.dqn_vecnorm
        print(f"Loaded DQN ")
    else:
        print(f"DQN model not found")

    os.makedirs("results", exist_ok=True)
    t0 = time.perf_counter()

    coord_res = run_coord_sweep(agents, vecnorm_map, args.episodes,
                                args.seed, args.ep_len)
    with open("results/exp5_coord_sweep.json", "w") as f:
        json.dump(coord_res, f, indent=2)
    print("\nCoord sweep saved → results/exp5_coord_sweep.json")
    print(f"Total wall time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()