"""
Experiment 5 — Reward-Shaping Sensitivity (Train-time γ sweep)
==============================================================
Retrains PPO, DQN, A2C, Double DQN, Dueling DQN, Dueling Double DQN,
and Recurrent PPO under each γ value, then evaluates all agents
(including Rule-based baseline and Random) with that same γ.

Baseline and Random don't need retraining — their behaviour is independent
of the reward signal — but they are evaluated under each γ so their raw
metric numbers (drops, cost, queue) are comparable on the same scale.

γ values: [5, 10, 20, 30, 50]   (nominal = 50)

Output
------
results/exp5_gamma_sweep.json   — full metrics per agent × γ
plots/exp5_gamma_*.png          — one figure per metric

Directory layout produced
-------------------------
models/gamma_{g}/
    ppo_best/best_model.zip          ppo_vecnorm.pkl
    recurrent_ppo_best/best_model.zip  recurrent_ppo_vecnorm.pkl
    a2c_best/best_model.zip          a2c_vecnorm.pkl
    dqn_best/best_model.zip          dqn_vecnorm.pkl
    double_dqn_best/best_model.zip   double_dqn_vecnorm.pkl
    dueling_dqn_best/best_model.zip  dueling_dqn_vecnorm.pkl
    dueling_double_dqn_best/best_model.zip  dueling_double_dqn_vecnorm.pkl

Usage
-----
    # full run
    python exp5_gamma_train_sweep.py

    # smoke test (tiny timesteps, 2 episodes)
    python exp5_gamma_train_sweep.py --timesteps 20000 --episodes 2

    # skip training if models already exist
    python exp5_gamma_train_sweep.py --eval_only
"""

import argparse
import json
import os
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent import MlpLstmPolicy
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from baseline_agent import RuleBasedBaseline
from cloud_env import CloudScalingEnv  # noqa: F401
from dueling_dqn_policy import DuelingDQNPolicy
from env_factory import make_env, make_vec_env

if "CloudScaling-v1" not in gym.envs.registry:
    gym.register(id="CloudScaling-v1", entry_point="cloud_env:CloudScalingEnv")

# ── experiment config ──────────────────────────────────────────────────────────
GAMMA_VALUES = [5.0, 10.0, 20.0, 30.0, 50.0]
NOMINAL      = {"alpha": 1.0, "beta": 0.1, "gamma": 50.0, "omega": 5.0}

# ── hyperparameters ────────────────────────────────────────────────────────────
PPO_KWARGS = dict(
    learning_rate = 1.4377739650640294e-05,
    n_steps       = 4096,
    batch_size    = 64,
    n_epochs      = 5,
    gamma         = 0.9660969058740851,
    gae_lambda    = 0.9175042883882351,
    clip_range    = 0.1291937132179491,
    ent_coef      = 0.012344366981406195,
    vf_coef       = 0.9315525456344808,
    max_grad_norm = 0.48170949108431693,
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
)

# RecurrentPPO uses same on-policy kwargs; net_arch format differs slightly
RECURRENT_PPO_KWARGS = dict(
    learning_rate = 1.4377739650640294e-05,
    n_steps       = 4096,
    batch_size    = 64,
    n_epochs      = 5,
    gamma         = 0.9660969058740851,
    gae_lambda    = 0.9175042883882351,
    clip_range    = 0.1291937132179491,
    ent_coef      = 0.012344366981406195,
    vf_coef       = 0.9315525456344808,
    max_grad_norm = 0.48170949108431693,
)

A2C_KWARGS = dict(
    learning_rate = 1.4377739650640294e-05,
    n_steps       = 5,
    gamma         = 0.9660969058740851,
    gae_lambda    = 0.9175042883882351,
    ent_coef      = 0.012344366981406195,
    vf_coef       = 0.9315525456344808,
    max_grad_norm = 0.48170949108431693,
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
)

DQN_KWARGS = dict(
    learning_rate          = 1e-4,
    buffer_size            = 100_000,
    learning_starts        = 10_000,
    batch_size             = 64,
    tau                    = 1.0,
    gamma                  = 0.99,
    train_freq             = 4,
    gradient_steps         = 1,
    target_update_interval = 1_000,
    exploration_fraction   = 0.1,
    exploration_final_eps  = 0.05,
    policy_kwargs          = dict(net_arch=[256, 256]),
)

DOUBLE_DQN_KWARGS = {
    **DQN_KWARGS,
    "policy_kwargs": dict(net_arch=[256, 256]),
    # Double DQN is always-on in SB3's DQN via the target network —
    # no extra flag needed; this config is identical to DQN_KWARGS,
    # kept separate so it can diverge later if you tune it independently.
}

DUELING_DQN_KWARGS = {
    **DQN_KWARGS,
    "policy_kwargs": dict(net_arch=[256, 256]),
    # NOTE: dueling architecture is supplied via DuelingDQNPolicy
    # (see dueling_dqn_policy.py), not via a policy_kwargs flag —
    # SB3's DQNPolicy has no native `dueling` argument.
}

DUELING_DOUBLE_DQN_KWARGS = {
    **DQN_KWARGS,
    "policy_kwargs": dict(net_arch=[256, 256]),
    # Same dueling note as above; Double DQN behavior again comes for
    # free from SB3's target network.
}

# ── plot styles ────────────────────────────────────────────────────────────────
AGENT_STYLE = {
    "ppo":                {"color": "#2a78d6", "marker": "o", "ls": "-",  "label": "PPO"},
    "recurrent_ppo":      {"color": "#7b4fd4", "marker": "D", "ls": "-",  "label": "Recurrent PPO"},
    "a2c":                {"color": "#f5a623", "marker": "p", "ls": "-",  "label": "A2C"},
    "dqn":                {"color": "#e34948", "marker": "s", "ls": "-",  "label": "DQN"},
    "double_dqn":         {"color": "#c0392b", "marker": "P", "ls": "-",  "label": "Double DQN"},
    "dueling_dqn":        {"color": "#e67e22", "marker": "h", "ls": "-",  "label": "Dueling DQN"},
    "dueling_double_dqn": {"color": "#922b21", "marker": "*", "ls": "-",  "label": "Dueling Double DQN"},
    "baseline":           {"color": "#1baf7a", "marker": "^", "ls": "--", "label": "Rule-based"},
    "random":             {"color": "#888780", "marker": "x", "ls": ":",  "label": "Random"},
}

# ── helpers ────────────────────────────────────────────────────────────────────
def reward_weights_for(g: float) -> tuple:
    return (NOMINAL["alpha"], NOMINAL["beta"], g, NOMINAL["omega"])


def model_dir(g: float) -> str:
    return f"./models/exp5/gamma_{int(g)}"


def _make_train_env_onpolicy(g: float, seed: int, n_envs: int = 8):
    """SubprocVecEnv + VecNormalize for on-policy algorithms."""
    return make_vec_env(n_envs=n_envs, seed=seed, use_subprocess=True,
                        norm_reward=True, reward_weights=reward_weights_for(g))


def _make_train_env_offpolicy(g: float, seed: int):
    """Single DummyVecEnv + VecNormalize for off-policy algorithms."""
    return VecNormalize(
        DummyVecEnv([make_env(rank=0, seed=seed,
                              reward_weights=reward_weights_for(g))]),
        norm_obs=True, norm_reward=True, clip_obs=5.0, gamma=0.99)


def _make_eval_env(g: float, seed: int):
    """Frozen eval env (no reward normalization)."""
    env = VecNormalize(
        DummyVecEnv([make_env(rank=100, seed=seed,
                              reward_weights=reward_weights_for(g))]),
        norm_obs=True, norm_reward=False, clip_obs=5.0, gamma=0.99)
    env.training = False
    return env


def _train(cls, kwargs, tag, g, timesteps, device, train_env, eval_env,
           best_path, vecnorm_path, policy="MlpPolicy"):
    """Generic train loop shared by all algorithms."""
    print(f"\n  [{tag}] γ={g}  timesteps={timesteps:,}")
    os.makedirs(best_path, exist_ok=True)

    model = cls(policy=policy, env=train_env,
                tensorboard_log=f"./logs/exp5/{tag}_gamma_{int(g)}/",
                device=device, verbose=0, **kwargs)

    eval_cb = EvalCallback(
        eval_env, best_model_save_path=best_path,
        eval_freq=10_000, n_eval_episodes=5, deterministic=True, verbose=0)

    model.learn(total_timesteps=timesteps, callback=eval_cb,
                reset_num_timesteps=True)

    train_env.save(vecnorm_path)
    train_env.close()
    eval_env.close()
    return f"{best_path}/best_model.zip", vecnorm_path


# ── per-algorithm training functions ──────────────────────────────────────────
def train_ppo(g, timesteps, device):
    mdir = model_dir(g)
    return _train(PPO, PPO_KWARGS, "ppo", g, timesteps, device,
                  _make_train_env_onpolicy(g, seed=0),
                  _make_eval_env(g, seed=0),
                  f"{mdir}/ppo_best", f"{mdir}/ppo_vecnorm.pkl")


def train_recurrent_ppo(g, timesteps, device):
    mdir = model_dir(g)
    return _train(RecurrentPPO, RECURRENT_PPO_KWARGS, "recurrent_ppo", g,
                  timesteps, device,
                  _make_train_env_onpolicy(g, seed=0),
                  _make_eval_env(g, seed=0),
                  f"{mdir}/recurrent_ppo_best",
                  f"{mdir}/recurrent_ppo_vecnorm.pkl",policy="MlpLstmPolicy")


def train_a2c(g, timesteps, device):
    mdir = model_dir(g)
    return _train(A2C, A2C_KWARGS, "a2c", g, timesteps, device,
                  _make_train_env_onpolicy(g, seed=0),
                  _make_eval_env(g, seed=0),
                  f"{mdir}/a2c_best", f"{mdir}/a2c_vecnorm.pkl")


def train_dqn(g, timesteps, device):
    mdir = model_dir(g)
    return _train(DQN, DQN_KWARGS, "dqn", g, timesteps, device,
                  _make_train_env_offpolicy(g, seed=0),
                  _make_eval_env(g, seed=0),
                  f"{mdir}/dqn_best", f"{mdir}/dqn_vecnorm.pkl")


def train_double_dqn(g, timesteps, device):
    mdir = model_dir(g)
    return _train(DQN, DOUBLE_DQN_KWARGS, "double_dqn", g, timesteps, device,
                  _make_train_env_offpolicy(g, seed=0),
                  _make_eval_env(g, seed=0),
                  f"{mdir}/double_dqn_best", f"{mdir}/double_dqn_vecnorm.pkl")


def train_dueling_dqn(g, timesteps, device):
    mdir = model_dir(g)
    return _train(DQN, DUELING_DQN_KWARGS, "dueling_dqn", g, timesteps, device,
                  _make_train_env_offpolicy(g, seed=0),
                  _make_eval_env(g, seed=0),
                  f"{mdir}/dueling_dqn_best",
                  f"{mdir}/dueling_dqn_vecnorm.pkl",
                  policy=DuelingDQNPolicy)


def train_dueling_double_dqn(g, timesteps, device):
    mdir = model_dir(g)
    return _train(DQN, DUELING_DOUBLE_DQN_KWARGS, "dueling_double_dqn", g,
                  timesteps, device,
                  _make_train_env_offpolicy(g, seed=0),
                  _make_eval_env(g, seed=0),
                  f"{mdir}/dueling_double_dqn_best",
                  f"{mdir}/dueling_double_dqn_vecnorm.pkl",
                  policy=DuelingDQNPolicy)


# maps agent name → (train_fn, loader_cls)
TRAINABLE_AGENTS = {
    "ppo":                (train_ppo,              PPO),
    "recurrent_ppo":      (train_recurrent_ppo,    RecurrentPPO),
    "a2c":                (train_a2c,              A2C),
    "dqn":                (train_dqn,              DQN),
    "double_dqn":         (train_double_dqn,       DQN),
    "dueling_dqn":        (train_dueling_dqn,      DQN),
    "dueling_double_dqn": (train_dueling_double_dqn, DQN),
}


# ── evaluation ─────────────────────────────────────────────────────────────────
def evaluate(agent, g: float, vecnorm_path: str | None,
             n_episodes: int, seed: int, ep_len: int = 1000) -> dict:
    rw     = reward_weights_for(g)
    env_fn = make_env(rank=99, seed=seed, reward_weights=rw)
    vec    = DummyVecEnv([env_fn])

    if vecnorm_path and os.path.exists(vecnorm_path):
        vec = VecNormalize.load(vecnorm_path, vec)
        vec.training    = False
        vec.norm_reward = False

    rewards, costs, drops, qocc = [], [], [], []

    for _ in range(n_episodes):
        obs  = vec.reset()
        done = [False]
        R = c = d = q = steps = 0

        lstm_states    = None
        episode_starts = np.ones((1,), dtype=bool)

        while not done[0]:
            if agent == "random":
                action = np.array([vec.action_space.sample()])
            elif isinstance(agent, RecurrentPPO):
                action, lstm_states = agent.predict(
                    obs, state=lstm_states,
                    episode_start=episode_starts, deterministic=True)
                episode_starts = np.array(done)
            elif hasattr(agent, "predict"):
                raw        = obs[0]
                result     = agent.predict(raw, deterministic=True)
                action_val = result[0] if isinstance(result, tuple) else result
                action     = np.atleast_1d(np.array(action_val))

            obs, r, done, info = vec.step(action)
            R += r[0];  c += info[0]["active"]
            d += info[0]["dropped"];  q += info[0]["queue"]
            steps += 1

        rewards.append(R);  costs.append(c)
        drops.append(d);    qocc.append(q / (steps * 500))

    vec.close()
    agg = lambda x: {"mean": float(np.mean(x)), "std": float(np.std(x))}
    return {"reward": agg(rewards), "cost": agg(costs),
            "dropped": agg(drops), "queue_occ": agg(qocc)}


# ── plotting ───────────────────────────────────────────────────────────────────
PLOT_METRICS = [
    ("reward",    "Mean total reward"),
    ("dropped",   "Mean dropped requests"),
    ("cost",      "Mean operational cost"),
    ("queue_occ", "Mean queue occupancy (frac)"),
]


def plot_results(results: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    agent_order = ["ppo", "recurrent_ppo", "a2c", "dqn", "double_dqn",
                   "dueling_dqn", "dueling_double_dqn", "baseline", "random"]
    agents = [a for a in agent_order if a in results]

    for metric_key, metric_label in PLOT_METRICS:
        fig, ax = plt.subplots(figsize=(10, 6))

        for agent in agents:
            st   = AGENT_STYLE[agent]
            xs   = sorted(results[agent].keys(), key=float)
            ys   = [results[agent][g][metric_key]["mean"] for g in xs]
            errs = [results[agent][g][metric_key]["std"]  for g in xs]
            xs_f = [float(x) for x in xs]

            ax.plot(xs_f, ys, color=st["color"], marker=st["marker"],
                    ls=st["ls"], lw=2, ms=6, label=st["label"])
            ax.fill_between(xs_f,
                            np.array(ys) - np.array(errs),
                            np.array(ys) + np.array(errs),
                            color=st["color"], alpha=0.10)

        ax.axvline(x=50.0, color="#b4b2a9", ls="--", lw=1.2, label="Nominal γ=50")
        ax.set_xlabel(r"$\gamma$ (drop penalty weight)", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f"Exp 5 — Train-time γ sweep: {metric_label}", fontsize=12)
        ax.set_xscale("log")
        ax.set_xticks([float(g) for g in GAMMA_VALUES])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.tick_params(labelsize=9)
        ax.grid(True, lw=0.4, alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False, ncol=2)

        out = os.path.join(out_dir, f"exp5_gamma_{metric_key}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out}")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--episodes",  type=int, default=10)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--device",    default="cpu", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--out_dir",   default="plots")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    results  = {}
    baseline = RuleBasedBaseline(n_max=10, q_max=500)
    t0       = time.perf_counter()

    for g in GAMMA_VALUES:
        g_key = str(g)
        mdir  = model_dir(g)

        print(f"\n{'='*60}")
        print(f"  γ = {g}")
        print(f"{'='*60}")

        # ── train / locate models ──────────────────────────────────────────────
        model_paths   = {}   # agent_name → model zip path
        vecnorm_paths = {}   # agent_name → vecnorm pkl path

        for name, (train_fn, _) in TRAINABLE_AGENTS.items():
            best_path    = f"{mdir}/{name}_best/best_model.zip"
            vecnorm_path = f"{mdir}/{name}_vecnorm.pkl"

            if not args.eval_only:
                best_path, vecnorm_path = train_fn(g, args.timesteps, args.device)

            model_paths[name]   = best_path
            vecnorm_paths[name] = vecnorm_path

        # ── load models ────────────────────────────────────────────────────────
        agents_eval = {"baseline": baseline, "random": "random"}
        vecnorm_map = {"baseline": None, "random": None}

        for name, (_, loader_cls) in TRAINABLE_AGENTS.items():
            mp = model_paths[name]
            if os.path.exists(mp):
                agents_eval[name] = loader_cls.load(mp)
                vecnorm_map[name] = vecnorm_paths[name]
                print(f"  [✓] {name} loaded")
            else:
                print(f"  [!] {name} not found at {mp} — skipping")

        # ── evaluate ───────────────────────────────────────────────────────────
        print(f"\n  Evaluating ({args.episodes} episodes each) ...")
        for agent_name, agent in agents_eval.items():
            m = evaluate(agent, g, vecnorm_map[agent_name],
                         args.episodes, args.seed)
            results.setdefault(agent_name, {})[g_key] = m
            print(f"    {agent_name:<22}  reward={m['reward']['mean']:>10.1f}  "
                  f"dropped={m['dropped']['mean']:>8.1f}  "
                  f"cost={m['cost']['mean']:>8.1f}")

    # ── save & plot ────────────────────────────────────────────────────────────
    out_json = "results/exp5_gamma_sweep.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Results saved → {out_json}")

    print("\nGenerating plots ...")
    plot_results(results, args.out_dir)

    print(f"\n[✓] Total wall time: {time.perf_counter()-t0:.1f}s "
          f"({(time.perf_counter()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
