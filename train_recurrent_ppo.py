#pip install sb3-contrib


"""Train PPO with LSTM on CloudScaling-v1.

This file trains Recurrent PPO, also called PPO-LSTM.

PPO provides stable policy-gradient learning using clipped updates.
The LSTM policy adds memory, allowing the agent to use recent traffic
and queue history when choosing scaling actions.

This is useful for cloud autoscaling because server boot delay and
traffic spikes make the best action depend on previous timesteps,
not only the current observation.
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from sb3_contrib import RecurrentPPO
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
    plt.plot(timesteps, mean_rewards, label="PPO-LSTM mean reward")
    plt.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.25,
    )

    plt.xlabel("Timesteps")
    plt.ylabel("Evaluation reward")
    plt.title("PPO-LSTM Evaluation Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved plot to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO-LSTM on CloudScaling-v1")

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
        "./models/best_recurrent_ppo",
        "./checkpoints/recurrent_ppo",
        "./logs/recurrent_ppo",
        "./logs/recurrent_ppo_eval",
        "./results/recurrent_ppo",
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

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            lstm_hidden_size=128,
            n_lstm_layers=1,
            shared_lstm=False,
            enable_critic_lstm=True,
            net_arch=dict(pi=[256], vf=[256]),
        ),
        tensorboard_log="./logs/recurrent_ppo/",
        device=args.device,
        seed=args.seed,
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_recurrent_ppo/",
        log_path="./logs/recurrent_ppo_eval/",
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=100_000,
        save_path="./checkpoints/recurrent_ppo/",
    )

    metrics_cb = MetricsCallback()

    print("=" * 60)
    print("[START] PPO-LSTM Training")
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
        print("\n[INTERRUPTED] Saving PPO-LSTM model")

    model.save("./models/final_recurrent_ppo")
    train_env.save("./models/vecnormalize_recurrent_ppo.pkl")

    wall_time = time.perf_counter() - start

    plot_eval_curve(
        eval_path="./logs/recurrent_ppo_eval/evaluations.npz",
        out_path="./results/recurrent_ppo/recurrent_ppo_eval_curve.png",
    )

    print("=" * 60)
    print("[DONE] PPO-LSTM Training")
    print(f"Wall time: {wall_time:.1f}s")
    print("Final model: ./models/final_recurrent_ppo.zip")
    print("Best model:  ./models/best_recurrent_ppo/best_model.zip")
    print("VecNormalize: ./models/vecnormalize_recurrent_ppo.pkl")
    print("=" * 60)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()