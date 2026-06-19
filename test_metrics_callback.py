"""Quick test: train PPO for 2k steps and verify custom metrics appear in TB."""

import os
import glob

from stable_baselines3 import PPO
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from env_factory import make_vec_env
from metrics_callback import MetricsCallback

LOG_DIR = "./logs/test_metrics/"

env = make_vec_env(n_envs=2, seed=0, use_subprocess=False)
model = PPO("MlpPolicy", env, n_steps=256, batch_size=64, verbose=0,
            tensorboard_log=LOG_DIR)

print("Training 2000 steps ...")
model.learn(total_timesteps=2000, callback=MetricsCallback())
env.close()
print("Done.\n")

# find event file
pattern = os.path.join(LOG_DIR, "PPO_*", "events.out.tfevents.*")
event_files = glob.glob(pattern)
assert event_files, f"No event files found: {pattern}"

ea = EventAccumulator(event_files[-1])
ea.Reload()
tags = ea.Tags().get("scalars", [])

expected = [
    "custom/dropped_requests_per_episode",
    "custom/active_servers_mean",
    "custom/queue_length_mean",
]

missing = [t for t in expected if t not in tags]
if missing:
    raise AssertionError(f"Missing from TensorBoard: {missing}")

for tag in expected:
    print(f"  {tag}: {len(ea.Scalars(tag))} points")

print("\n[OK] All custom metrics found in TensorBoard logs.")
