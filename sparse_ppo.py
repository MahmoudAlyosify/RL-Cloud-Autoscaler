import time
import os
import torch
import subprocess
import threading
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from metrics_callback import MetricsCallback
from env_factory import make_vec_env, make_env

class SparsePPO(PPO):
    """PPO that performs a gradient update only every K-th rollout.
    On skipped iterations the freshly collected on-policy rollout is
    discarded (it cannot be reused next iteration because it is on-policy).
    This trades sample usage for optimizer compute."""
    def __init__(self, *args, update_every_k=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_every_k = int(update_every_k)
        self._rollout_idx = 0

    def train(self) -> None:
        self._rollout_idx += 1
        if self.update_every_k > 1 and (self._rollout_idx % self.update_every_k != 0):
            # log that we skipped, then return WITHOUT touching the optimizer
            self.logger.record("sparse/skipped_update", 1)
            return
        self.logger.record("sparse/skipped_update", 0)
        super().train()   # the real gradient update

def sample_gpu_util(stop_evt, out):
    while not stop_evt.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2)
            out.append(int(r.stdout.strip().splitlines()[0]))
        except Exception:
            pass
        time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser(description="Run Sparse PPO experiments")
    parser.add_argument("--timesteps", type=int, default=2_000_000,
                        help="Timesteps per K value (matches DQN variant budget)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    for k in [1, 4, 8]:
        print(f"\n--- Running Sparse PPO with K={k} ---")

        best_model_dir = f"./models/best_sparse_ppo_k{k}/"
        eval_log_dir = f"./logs/sparse_ppo_k{k}_eval/"
        checkpoint_dir = f"./checkpoints/sparse_ppo_k{k}/"
        tb_log_dir = f"./logs/sparse_ppo_k{k}/"

        for d in [best_model_dir, eval_log_dir, checkpoint_dir]:
            os.makedirs(d, exist_ok=True)

        raw_train_env = DummyVecEnv(
            [make_env(rank=i, seed=args.seed) for i in range(8)]
        )
        train_env = VecNormalize.load(
            "./models/vecnormalize_ppo.pkl",
            raw_train_env,
        )
        train_env.training = True
        train_env.norm_reward = True

        raw_eval_env = DummyVecEnv([make_env(rank=100, seed=args.seed)])
        eval_env = VecNormalize.load(
            "./models/vecnormalize_ppo.pkl",
            raw_eval_env,
        )
        eval_env.training = False
        eval_env.norm_reward = False

        model = SparsePPO.load(
            "./models/best_ppo/best_model.zip",
            env=train_env,
            tensorboard_log=tb_log_dir,
            device=args.device,
        )
        model.update_every_k = k
        model._rollout_idx = 0

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=best_model_dir,
            log_path=eval_log_dir,
            eval_freq=10_000,
            n_eval_episodes=5,
            deterministic=True,
        )
        ckpt_cb = CheckpointCallback(save_freq=100_000, save_path=checkpoint_dir)
        metrics_cb = MetricsCallback()

        stop_evt, util = threading.Event(), []
        t = threading.Thread(target=sample_gpu_util, args=(stop_evt, util))
        t.start()

        start = time.perf_counter()
        model.learn(
            total_timesteps=args.timesteps,
            callback=[eval_cb, ckpt_cb, metrics_cb],  # ← was missing entirely
        )
        wall = time.perf_counter() - start

        stop_evt.set()
        t.join()

        peak_mem = 0.0
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1e9

        mean_gpu_util = sum(util) / max(1, len(util)) if util else 0

        print(f"K={model.update_every_k}  wall={wall:.1f}s  "
              f"peak_mem={peak_mem:.2f}GB  "
              f"mean_gpu_util={mean_gpu_util:.0f}%")

        model.save(f"./models/sparse_ppo_k{k}")
        train_env.save(f"./models/vecnormalize_sparse_ppo_k{k}.pkl")

        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
