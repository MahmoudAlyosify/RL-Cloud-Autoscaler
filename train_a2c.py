"""Train A2C on CloudScaling-v1.

A2C is used as a simple actor-critic reinforcement learning method.
It learns:
1. an actor policy that chooses scale out, hold, or scale in
2. a critic value function that estimates how good the current state is

This allows us to compare a simple actor-critic method against PPO,
DQN-based methods, and the rule-based baseline.
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from cloud_env import CloudScalingEnv  # noqa: F401
from env_factory import make_env, make_vec_env
from metrics_callback import MetricsCallback


def plot_eval_curve(eval_path, out_path):
    if not os.path.exists(eval_path):
        print(f"No eval file found at {eval_path}")
        return

    data = np.load(eval_path)
    timesteps = data["timesteps"]
    rewards = data["results"]

    mean_rewards = rewards.mean(axis=1)
    std_rewards = rewards.std(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, mean_rewards, label="A2C mean reward")
    plt.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.25,
    )

    plt.xlabel("Timesteps")
    plt.ylabel("Evaluation reward")
    plt.title("A2C Evaluation Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved plot to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train A2C on CloudScaling-v1")

    parser.add_argument(
        "--timesteps",
        type=int,
        default=2_000_000,
        help="Total number of training timesteps.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for training.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    for path in [
        "./models/best_a2c",
        "./checkpoints/a2c",
        "./logs/a2c",
        "./logs/a2c_eval",
        "./results/a2c",
    ]:
        os.makedirs(path, exist_ok=True)

    train_env = make_vec_env(
        n_envs=8,
        seed=args.seed,
        use_subprocess=False,
        norm_reward=True,
    )

    eval_env = VecNormalize(
        DummyVecEnv([make_env(rank=100, seed=args.seed)]),
        norm_obs=True,
        norm_reward=False,
        clip_obs=5.0,
        gamma=0.99,
    )
    eval_env.training = False

    model = A2C(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=512,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
        tensorboard_log="./logs/a2c/",
        device=args.device,
        seed=args.seed,
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_a2c/",
        log_path="./logs/a2c_eval/",
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=100_000,
        save_path="./checkpoints/a2c/",
    )

    metrics_cb = MetricsCallback()

    print("=" * 60)
    print("[START] A2C Training")
    print(f"Device: {args.device} | Timesteps: {args.timesteps:,}")
    print("=" * 60)

    start = time.perf_counter()

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[eval_cb, ckpt_cb, metrics_cb],
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving A2C model")

    model.save("./models/final_a2c")
    train_env.save("./models/vecnormalize_a2c.pkl")

    wall_time = time.perf_counter() - start

    plot_eval_curve(
        eval_path="./logs/a2c_eval/evaluations.npz",
        out_path="./results/a2c/a2c_eval_curve.png",
    )

    print("=" * 60)
    print("[DONE] A2C Training")
    print(f"Wall time: {wall_time:.1f}s")
    print("Final model: ./models/final_a2c.zip")
    print("Best model:  ./models/best_a2c/best_model.zip")
    print("VecNormalize: ./models/vecnormalize_a2c.pkl")
    print("=" * 60)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()