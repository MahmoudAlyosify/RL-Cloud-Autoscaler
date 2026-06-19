"""Train DQN on CloudScaling-v1 with EvalCallback, checkpoints, and custom metrics."""

import argparse
import os
import time

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from cloud_env import CloudScalingEnv  # noqa: F401
from env_factory import make_env, make_vec_env
from metrics_callback import MetricsCallback


def parse_args():
    p = argparse.ArgumentParser(description="Train DQN on CloudScaling-v1")
    p.add_argument("--timesteps", type=int, default=2_000_000,
                   help="Total training timesteps (use 20000 for smoke test)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="Device for training")
    return p.parse_args()


def main():
    args = parse_args()

    for d in ["./models/best_dqn", "./checkpoints/dqn",
              "./logs/dqn", "./logs/dqn_eval"]:
        os.makedirs(d, exist_ok=True)

    # single env with DummyVecEnv -- never SubprocVecEnv for DQN
    train_env = make_vec_env(n_envs=1, seed=0, use_subprocess=False,
                             norm_reward=True)

    # eval env: no reward normalization, frozen stats
    eval_env = VecNormalize(
        DummyVecEnv([make_env(rank=100, seed=0)]),
        norm_obs=True, norm_reward=False, clip_obs=5.0, gamma=0.99)
    eval_env.training = False

    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=256,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log="./logs/dqn/",
        device=args.device,
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_dqn/",
        log_path="./logs/dqn_eval/",
        eval_freq=80_000,  # matches PPO's effective interval (10k × 8 envs)
        n_eval_episodes=5,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(save_freq=100_000,
                                 save_path="./checkpoints/dqn/")
    metrics_cb = MetricsCallback()

    print("=" * 60)
    print(f"  [START] DQN Training")
    print(f"  Device: {args.device} | Timesteps: {args.timesteps:,}")
    print("=" * 60)

    t0 = time.perf_counter()

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[eval_cb, ckpt_cb, metrics_cb],
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving model before exit ...")

    model.save("./models/final_dqn")
    train_env.save("./models/vecnormalize_dqn.pkl")

    wall_time = time.perf_counter() - t0

    print()
    print("=" * 60)
    print(f"  [DONE] DQN Training")
    print(f"  Wall-clock time: {wall_time:.1f}s ({wall_time/60:.1f} min)")
    print(f"  Model saved to: ./models/final_dqn.zip")
    print(f"  VecNormalize saved to: ./models/vecnormalize_dqn.pkl")
    print(f"  Best model (EvalCallback): ./models/best_dqn/best_model.zip")
    print(f"  Eval logs: ./logs/dqn_eval/evaluations.npz")
    print("=" * 60)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
