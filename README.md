# Autonomous Cloud Resource Provisioning via Reinforcement Learning ☁️

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Standard-brightgreen)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO%20%7C%20DQN-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 📌 Overview

This repository implements autonomous cloud resource provisioning using Deep Reinforcement Learning. The auto-scaling problem is modelled as a Markov Decision Process inside a custom Gymnasium environment.

The goal is to learn dynamic scaling policies that minimise infrastructure cost while keeping latency low and avoiding dropped requests under non-stationary traffic patterns.

---

## 🚀 Key Features

- **Custom Gymnasium Environment** — A mathematically bounded cloud simulator enforcing constraints such as minimum servers (N_min), boot-up latencies (cold start), and capacity limits.
- **Deep RL Agents** — PPO, Recurrent PPO (LSTM), DQN (Vanilla, Double, Dueling, Double+Dueling), and A2C.
- **Hyperparameter Tuning (Optuna)** — Optuna-based sweeps and fine-tuning scripts for DQN / dueling-family DQN variants.
- **Fine-tuning & Sparse-update Ablations** — Scripts to fine-tune pretrained models and to run the sparsity ablation (fewer gradient updates per environment step) for compute/quality trade-offs.
- **Proactive Scaling** — Cold-start delay forces agents to anticipate traffic rather than react to it.
- **Reproducible Experiments** — Seedable training and evaluation; all results serialised to JSON and NPZ for plotting.

---

## 📁 Repository Structure

```
custom_envs/                         — Gymnasium environment and wrappers
agents/                              — DQN, PPO, A2C agent implementations and helpers
experiments/                         — Training, evaluation, ablation, and utility scripts
notebooks/                           — Interactive experiments and visualisations
    RL_Cloud_Autoscaler_Complete.ipynb — Complete project notebook (tutorial + reproducible runs)
environment.yml                      — Conda environment specification
requirements.txt                     — pip dependencies
train_dqn_fine_tuning.py             — Optuna sweeps & fine-tuning for DQN
train_dueling_dqn_fine_tuning.py     — Optuna sweeps for dueling-family DQN
train_recurrent_ppo_variants.py      — Variant runner for recurrent PPO (LSTM)
train_a2c_variants.py                — Variant runner for A2C configurations
sparse_*.py                          — Sparse-update ablation scripts (ppo / recurrent_ppo / a2c)
plot_results.py                      — Aggregation and plotting utilities
README.md                            — This file
```

---

## 🛠️ Installation

**Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate rl-cloud-autoscaler
```

**pip:**
```bash
pip install -r requirements.txt
# or the minimal set:
pip install stable-baselines3 sb3-contrib gymnasium numpy matplotlib torch optuna
```

Note: `sb3-contrib` (for RecurrentPPO) and `optuna` (for hyperparameter sweeps) are optional but required if you intend to run LSTM agents or Optuna studies.

---

## 🏃 Training

### DQN Variants (Vanilla, Double, Dueling, Double+Dueling)

```bash
python train_dqn.py --variant vanilla
python train_dqn.py --variant double
python train_dqn.py --variant dueling
python train_dqn.py --variant double_dueling
```

You can also control the sparse-update ablation for DQN with `--update_frequency` (choices: 1,2,4,8).

### DQN Fine-tuning / Optuna Sweeps

```bash
python train_dqn_fine_tuning.py
python train_dueling_dqn_fine_tuning.py
```

These scripts run Optuna studies and save the best configurations and models.

### PPO

```bash
python train_ppo.py --timesteps 2000000 --device auto
```

### Recurrent PPO (LSTM)

```bash
python train_recurrent_ppo.py --timesteps 2000000 --device auto --seed 0
```

For variant-based recurrent runs (e.g., robust spike traffic), use:

```bash
python train_recurrent_ppo_variants.py --variant <variant_name>
```

### A2C

```bash
python train_a2c.py --timesteps 2000000 --device auto --seed 0
```

For A2C variants/configurations:

```bash
python train_a2c_variants.py --variant <variant_name>
```

---

## 🧪 Experiment 1: Sparsity Ablation

Tests whether sparse gradient updates (fewer updates per environment step) improve compute efficiency without hurting control quality.

> ⚠️ Important: run the default training command for each algorithm first (see Training above) to produce the pretrained model. The sparsity scripts fine-tune those pretrained models (they do not always train from scratch).

### DQN Sparsity (4 variants × 3 frequencies)

```bash
python train_dqn.py --variant <variant> --update_frequency 1
python train_dqn.py --variant <variant> --update_frequency 4
python train_dqn.py --variant <variant> --update_frequency 8
```

### PPO / Recurrent PPO / A2C Sparsity

```bash
python sparse_ppo.py
python sparse_recurrent_ppo.py
python sparse_a2c.py
```

### Generate Sparsity Plots

```bash
python sparsity_plotter.py
```

Six figures are saved to `./results/Experiments/plots_exp2/` by convention.

---

## ✅ Evaluation

Run these after training and sparsity scripts have completed.

```bash
# Baseline and agent evaluation
python run_baseline_eval.py
python eval_agent.py --episodes 10 --seed 42

# Select the best DQN variant at each update frequency
python select_best_dqn.py --freq 1
python select_best_dqn.py --freq 4
python select_best_dqn.py --freq 8

# Algorithm comparison and stress tests
python main_algorithm_comparison.py --eval-seeds 0,1,2,3,4 --episodes-per-seed 1
python traffic_stress_test.py --seeds 0,1,2,3,4

# Generate all plots
python plot_results.py
```

---

## 📊 Plots Produced

After running `plot_results.py`, the following figures are saved to `./results/plots/`:

- Learning curves at update frequency 1, 4, and 8 for all algorithms
- Final performance bar charts at update frequency 1, 4, and 8
- Convergence box plots and algorithm comparison charts

---

## 📝 Notes & Known issues (please read)

- Observation-space documentation: the notebook mentions a 6-D observation in some markdown cells, but the environment implementation currently returns a 5-D observation vector (active, booting, cpu_util, queue, arrival_ema). If you rely on a 6th feature (e.g. previous action or trend), either update `cloud_env.py` or change the docs. Recommended action: keep the env as 5-D and update the markdown to avoid mismatch.

- Reward-weight mismatch: global defaults in the notebook/configs use `gamma = 50.0` while `cloud_env.py` sets the default `reward_weights=(1.0, 0.1, 20.0, 5.0)` in the environment constructor. For reproducible experiments, unify these values. Recommended: set the environment's default gamma (drop penalty) to 50.0 or explicitly pass `reward_weights=` when creating envs.

- Optional dependencies: `sb3-contrib` and `optuna` are optional; the code checks for them and prints warnings if missing. If you want to use RecurrentPPO or run Optuna sweeps, install them.

- Results directory: README now references `./results/Experiments/plots_exp2/` for sparsity plots (corrected from the earlier typo `Experments`). Ensure downstream scripts write to expected locations or adjust `plot_results.py` accordingly.

If you want, I can open a small PR that:
- fixes the `observation_space` description in the notebook,
- aligns the default reward weight values across notebook and environment, and
- optionally adds sb3-contrib/optuna to `environment.yml` (recommended for reproducibility).

---

## 🔧 Tips

- Use `--timesteps 20000` for a quick smoke test before committing to a full run.
- If training is unstable, try lowering the learning rate or increasing batch size.
- All results are saved incrementally and many training scripts support resuming from checkpoints.

---

## 📚 Citation

If you use this code in research, please cite the repository and any relevant papers your work builds on.

## 🤝 Contributing

Contributions are welcome. Open an issue for major changes and submit PRs with clear descriptions.

## 📝 License

Specify your licence here (e.g. MIT).

## ✉️ Contact

Maintainer: MahmoudAlyosify
