# Autonomous Cloud Resource Provisioning via Reinforcement Learning ☁️

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Standard-brightgreen)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO%20%7C%20DQN-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 📌 Overview

This repository implements autonomous cloud resource provisioning using Deep Reinforcement Learning. The auto-scaling problem is formulated as a Markov Decision Process inside a custom Gymnasium environment.

The goal is to learn dynamic scaling policies that minimise infrastructure costs while keeping latency low and avoiding dropped requests under non-stationary traffic patterns.

---

## 🚀 Key Features

- **Custom Gymnasium Environment** — A mathematically bounded cloud simulator enforcing constraints such as minimum servers (N_min), boot-up latencies, and capacity limits.
- **Deep RL Agents** — PPO, Recurrent PPO (LSTM), DQN (Vanilla, Double, Dueling, Double+Dueling), and A2C.
- **Hyperparameter Tuning (Optuna)** — Optuna-based sweeps and fine-tuning scripts for DQN / Dueling families to find robust settings.
- **Fine-tuning & Sparse-update Ablations** — Scripts to fine-tune pretrained models and to run the sparsity ablation (fewer gradient updates per environment step) for compute/quality trade-offs.
- **Proactive Scaling** — Cold-start delay forces agents to anticipate traffic rather than react to it.
- **Sparsity Ablation** — Each algorithm is evaluated at update frequencies k = 1, 4, 8 to measure the compute-efficiency tradeoff.
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

---

## 🏃 Training

### DQN Variants (Vanilla, Double, Dueling, Double+Dueling)

```bash
python train_dqn.py --variant vanilla
python train_dqn.py --variant double
python train_dqn.py --variant dueling
python train_dqn.py --variant double_dueling
```

### DQN Fine-tuning / Optuna Sweeps

Use the Optuna-based scripts to run hyperparameter sweeps or trimmed fine-tuning for the dueling family:

```bash
python train_dqn_fine_tuning.py
python train_dueling_dqn_fine_tuning.py
```

(These scripts run optuna studies and return the best hyperparameters and saved models.)

### PPO

```bash
python train_ppo.py --timesteps 2000000 --device auto
```

### Recurrent PPO (LSTM)

```bash
python train_recurrent_ppo.py --timesteps 2000000 --device auto --seed 0
```

For variant-based recurrent runs (e.g., robust spike traffic, different traffic generators), use:

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

> ⚠️ **Important:** Run the default training command for each algorithm first (see Training above) to produce the best pretrained model. The sparsity scripts load that model and fine-tune under each frequency (they do not train from scratch).

### DQN Sparsity (12 runs: 4 variants × 3 frequencies)

```bash
python train_dqn.py --variant vanilla        --update_frequency 1
python train_dqn.py --variant vanilla        --update_frequency 4
python train_dqn.py --variant vanilla        --update_frequency 8
python train_dqn.py --variant double         --update_frequency 1
python train_dqn.py --variant double         --update_frequency 4
python train_dqn.py --variant double         --update_frequency 8
python train_dqn.py --variant dueling        --update_frequency 1
python train_dqn.py --variant dueling        --update_frequency 4
python train_dqn.py --variant dueling        --update_frequency 8
python train_dqn.py --variant double_dueling --update_frequency 1
python train_dqn.py --variant double_dueling --update_frequency 4
python train_dqn.py --variant double_dueling --update_frequency 8
```

### PPO Sparsity (k = 1, 4, 8 runs sequentially in one command)

```bash
python sparse_ppo.py
```

### Recurrent PPO Sparsity (k = 1, 4, 8 runs sequentially in one command)

```bash
python sparse_recurrent_ppo.py
```

### A2C Sparsity (k = 1, 4, 8 runs sequentially in one command)

```bash
python sparse_a2c.py
```

### 📈 Generate Sparsity Plots

After all sparsity training runs have completed:

```bash
python sparsity_plotter.py
```

Six figures are saved to `./results/Experiments/plots_exp2/` (note: corrected directory name from previous README):

| Plot | File | What it shows |
|---|---|---|
| Learning Curves (freq=1) | `learning_curves_freq1.png` | Learning speed when updating every step.
| Learning Curves (freq=4) | `learning_curves_freq4.png` | Balanced update schedule comparison.
| Learning Curves (freq=8) | `learning_curves_freq8.png` | Behaviour with least frequent updates.
| Final Performance (freq=1) | `final_performance_freq1.png` | Final performance bar chart (freq=1).
| Final Performance (freq=4) | `final_performance_freq4.png` | Final performance bar chart (freq=4).
| Final Performance (freq=8) | `final_performance_freq8.png` | Final performance bar chart (freq=8).

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

- Learning curves at update frequency 1, 4, and 8 for all the algorithms
- Final performance bar charts at update frequency 1, 4, and 8 for all the algorithms
- Convergence box plots and algorithm comparison charts

---

## 🔧 Tips

- Use `--timesteps 20000` for a quick smoke test before committing to a full run.
- If training is unstable, try lowering the learning rate or increasing batch size.
- All results are saved incrementally and a crashed run can be resumed without losing completed files.

---

## 📚 Citation

If you use this code in research, please cite the repository and any relevant papers your work builds on.

## 🤝 Contributing

Contributions are welcome. Open an issue for major changes and submit PRs with clear descriptions.

## 📝 License

Specify your licence here (e.g. MIT).

## ✉️ Contact

Maintainer: MahmoudAlyosify
