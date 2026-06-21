# Team Integration README

Internal engineering doc for Mohamed (PPO) and Shrouk (DQN + Sparse Updates).  
Everything below is ready to use — just import and build your training scripts on top.

---

## 1. Quick Start

```python
from env_factory import make_vec_env
from metrics_callback import MetricsCallback
from stable_baselines3 import PPO  # or DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# PPO: 8 parallel envs with SubprocVecEnv
train_env = make_vec_env(n_envs=8, seed=0, use_subprocess=True)

# DQN: 1 env with DummyVecEnv (never use SubprocVecEnv for DQN)
# train_env = make_vec_env(n_envs=1, seed=0, use_subprocess=False)

model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log="./logs/ppo/")

model.learn(
    total_timesteps=2_000_000,
    callback=[
        MetricsCallback(),          # <-- logs custom metrics every 1k steps
        # EvalCallback(...),        # add your eval callback here
        # CheckpointCallback(...),  # add your checkpoint callback here
    ],
)

model.save("./models/final_ppo")
train_env.save("./models/vecnormalize_ppo.pkl")  # CRITICAL — read gotcha #1
```

---

## 2. File Manifest

| File | Purpose | What you import |
|------|---------|-----------------|
| `traffic.py` | Poisson traffic generator (sinusoidal + spikes) | **Don't import** — used internally by `cloud_env.py` |
| `cloud_env.py` | Gymnasium env `"CloudScaling-v1"`, obs(5,), Discrete(3) | Only if you need `CloudScalingEnv` directly (rare) |
| `env_factory.py` | Vectorized env construction + VecNormalize | `make_env`, `make_vec_env` |
| `baseline_agent.py` | Rule-based heuristic (the target to beat) | `RuleBasedBaseline` |
| `run_baseline_eval.py` | Baseline evaluation script (already run) | Don't import — run as script |
| `metrics_callback.py` | TensorBoard custom metrics (drops, servers, queue) | `MetricsCallback` |
| `results/baseline_metrics.json` | Baseline numbers for comparison | Load with `json.load()` |

**Environment details at a glance:**

| Property | Value |
|----------|-------|
| Observation | `Box(0, 1, shape=(5,))` — `[active/10, booting/10, cpu_util, queue/500, lambda/240]` |
| Action | `Discrete(3)` — `0=Out, 1=Hold, 2=In` |
| Episode length | 1000 steps (truncation, never termination) |
| Reward | `-(1.0*C + 0.1*L + 50.0*D + 5.0*T)` |
| `info` keys | `dropped`, `active`, `queue`, `cpu_util`, `lambda`, `reward_components` |

---

## 3. The Target You Must Beat

Baseline results (10 episodes × 1000 steps, fixed seeds 1000–1009):

| Metric | Mean | Std |
|--------|------|-----|
| Total Reward | **-9122.69** | ±5287.15 |
| Operational Cost | 3213.60 | ±193.92 |
| Dropped Requests | 118.10 | ±107.69 |
| Mean Queue Length | 21.97 | ±2.65 |

**Why the variance is so high:** the baseline never looks at `arrival_rate` (obs[4]). It reacts *after* CPU/queue thresholds are crossed, but with a 3-step boot delay that's already too late during traffic spikes. Good episodes (no spikes) score ~-3200; bad episodes (multiple spikes) score ~-18000. An RL agent that learns to read `arrival_rate` and pre-warm servers should reduce both the mean *and* the variance.

---

## 4. Critical Gotchas

**1. Save VecNormalize stats after training.**  
Call `train_env.save("vecnormalize_ppo.pkl")` right after `model.save()`. At eval time, load with `VecNormalize.load(path, env)` and set `training=False`, `norm_reward=False`. **If you skip this, the agent sees observations on the wrong scale and appears to have forgotten everything.**

**2. DQN uses DummyVecEnv, never SubprocVecEnv.**  
```python
# correct
dqn_env = make_vec_env(n_envs=1, use_subprocess=False)
# wrong — adds overhead, no benefit for off-policy single-env learning
dqn_env = make_vec_env(n_envs=1, use_subprocess=True)
```
**SubprocVecEnv with DQN wastes IPC overhead for zero benefit.**

**3. Don't override terminated/truncated behavior.**  
The env already returns `terminated=False` always and `truncated=True` at step 1000. Don't wrap it in `TimeLimit` or manually set `terminated=True`. **Setting terminated=True at the time limit zeroes the value bootstrap and biases learning.**

**4. MetricsCallback handles logger.dump() internally.**  
Just pass `MetricsCallback()` to `model.learn()` — no extra flush calls needed. The three custom scalars (`custom/dropped_requests_per_episode`, `custom/active_servers_mean`, `custom/queue_length_mean`) will appear in TensorBoard automatically.

---

## 5. Who Owns What

| Owner | Files | Ping for |
|-------|-------|----------|
| **Mahmoud (Magnum)** | `cloud_env.py`, `traffic.py`, `env_factory.py`, `baseline_agent.py`, `metrics_callback.py`, `run_baseline_eval.py` | Env bugs, reward tuning, factory changes |
| **Mohamed** | `train_ppo.py` (to build) | PPO hyperparams, learning curves |
| **Shrouk** | `train_dqn.py`, `sparse_ppo.py` (to build) | DQN training, sparse update experiments |

### Still to build

- **Mohamed:** `train_ppo.py` — use `make_vec_env(n_envs=8, use_subprocess=True)`, PPO hyperparams from Constitution Section 6.2
- **Shrouk:** `train_dqn.py` — use `make_vec_env(n_envs=1, use_subprocess=False)`, DQN hyperparams from Constitution Section 6.3
- **Shrouk:** `sparse_ppo.py` — `SparsePPO(PPO)` subclass that overrides `train()`, per Constitution Section 7.2 (do NOT use the `set_training_mode` callback pattern — it doesn't actually skip the optimizer step)
