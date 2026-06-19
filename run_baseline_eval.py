"""Evaluate the rule-based baseline over 10 episodes and save metrics.

The mean total reward here is the target threshold PPO/DQN must beat.
"""

import json
import os

import gymnasium as gym
import numpy as np

from cloud_env import CloudScalingEnv  # noqa: F401
from baseline_agent import RuleBasedBaseline

if "CloudScaling-v1" not in gym.envs.registry:
    gym.register(id="CloudScaling-v1", entry_point="cloud_env:CloudScalingEnv")


def main():
    n_episodes = 10
    ep_length = 1000
    base_seed = 1000

    baseline = RuleBasedBaseline(n_max=10, q_max=500)
    all_rewards, all_costs, all_drops, all_queues = [], [], [], []

    print("=" * 60)
    print("  BASELINE EVALUATION (10 episodes x 1000 steps)")
    print("=" * 60)

    for ep in range(n_episodes):
        seed = base_seed + ep
        env = gym.make("CloudScaling-v1")
        obs, _ = env.reset(seed=seed)

        ep_reward, ep_cost, ep_drops, ep_queue = 0.0, 0, 0, 0

        for _ in range(ep_length):
            action, _ = baseline.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            ep_cost += info["active"]
            ep_drops += info["dropped"]
            ep_queue += info["queue"]

            if truncated:
                break

        env.close()
        mean_q = ep_queue / ep_length

        all_rewards.append(ep_reward)
        all_costs.append(ep_cost)
        all_drops.append(ep_drops)
        all_queues.append(mean_q)

        print(f"  Episode {ep+1:2d}/{n_episodes} | seed={seed} | "
              f"reward={ep_reward:9.1f} | dropped={ep_drops:4d} | "
              f"cost={ep_cost:5d} | mean_queue={mean_q:6.1f}")

    # aggregate
    def agg(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    results = {
        "reward": agg(all_rewards),
        "cost": agg(all_costs),
        "dropped": agg(all_drops),
        "queue": agg(all_queues),
        "n_episodes": n_episodes,
        "episode_length": ep_length,
    }

    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Total Reward     : {results['reward']['mean']:10.2f} +/- {results['reward']['std']:.2f}")
    print(f"  Operational Cost : {results['cost']['mean']:10.2f} +/- {results['cost']['std']:.2f}")
    print(f"  Dropped Requests : {results['dropped']['mean']:10.2f} +/- {results['dropped']['std']:.2f}")
    print(f"  Mean Queue Length: {results['queue']['mean']:10.2f} +/- {results['queue']['std']:.2f}")
    print("=" * 60)

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "baseline_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
