import time
import torch
import subprocess
import threading
import argparse
from stable_baselines3 import PPO
from env_factory import make_vec_env

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
    parser.add_argument("--timesteps", type=int, default=500_000, help="Timesteps for timing study")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to use")
    args = parser.parse_args()

    for k in [1, 4, 8]:
        print(f"\n--- Running Sparse PPO with K={k} ---")
        train_env = make_vec_env(n_envs=8, seed=42, use_subprocess=True, norm_reward=True)

        model = SparsePPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            tensorboard_log=f"./logs/sparse_ppo_k{k}/",
            device=args.device,
            verbose=0,
            update_every_k=k
        )

        stop_evt, util = threading.Event(), []
        t = threading.Thread(target=sample_gpu_util, args=(stop_evt, util))
        t.start()

        start = time.perf_counter()
        model.learn(total_timesteps=args.timesteps)
        wall = time.perf_counter() - start

        stop_evt.set()
        t.join()

        peak_mem = 0.0
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1e9

        mean_gpu_util = sum(util)/max(1,len(util)) if util else 0

        print(f"K={model.update_every_k}  wall={wall:.1f}s  "
              f"peak_mem={peak_mem:.2f}GB  "
              f"mean_gpu_util={mean_gpu_util:.0f}%")
              
        model.save(f"./models/sparse_ppo_k{k}")
        train_env.save(f"./models/vecnormalize_sparse_ppo_k{k}.pkl")
        train_env.close()

if __name__ == "__main__":
    main()
