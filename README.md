# Autonomous Cloud Resource Provisioning via Reinforcement Learning ☁️

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Standard-brightgreen)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO%20%7C%20DQN-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 📌 Overview

This repository implements autonomous cloud resource provisioning using Deep Reinforcement Learning. The auto-scaling problem is formulated as a Markov Decision Process inside a custom Gymnasium environment that models realistic cloud behaviours: cold-start delays, minimum server counts, and stochastic workloads.

The goal is to learn dynamic scaling policies that minimise infrastructure costs while keeping latency low and avoiding dropped requests under non-stationary traffic patterns.

---

## 🚀 Key Features

- **Custom Gymnasium Environment** A mathematically bounded cloud simulator enforcing constraints such as minimum servers (N_min), boot-up latencies, and capacity limits.
- **Deep RL Agents** PPO, Recurrent PPO (LSTM), DQN (Vanilla, Double, Dueling, Double+Dueling), and A2C.
- **Proactive Scaling** Cold-start delay forces agents to anticipate traffic rather than react to it.
- **Sparsity Ablation** Each algorithm is evaluated at update frequencies k = 1, 4, 8 to measure the compute-efficiency tradeoff.
- **Reproducible Experiments** Seedable training and evaluation; all results serialised to JSON and NPZ for plotting.

---

## 📁 Repository Structure

```
custom_envs/        — Gymnasium environment and wrappers
agents/             — DQN, PPO, A2C agent implementations
experiments/        — Training, evaluation, and ablation scripts
notebooks/          — Interactive experiments and visualisations
tests/              — Unit and integration tests
environment.yml     — Conda environment specification
requirements.txt    — pip dependencies
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

### PPO

```bash
python train_ppo.py --timesteps 2000000 --device auto
```

### Recurrent PPO (LSTM)

```bash
python train_recurrent_ppo.py --timesteps 2000000 --device auto --seed 0
```

### A2C

```bash
python train_a2c.py --timesteps 2000000 --device auto --seed 0
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

Six figures are saved to `./results/Experments/plots_exp2/`:

| Plot | File | What it shows |
|---|---|---|
| Learning Curves (freq=1) | `learning_curves_freq1.png` | This shows how quickly each algorithm learns when we update its weights after every single step. This is the most aggressive update schedule you can use as you'll see the fastest learning here, but it also costs the most computation per timestep. Notice that algorithms which diverge or plateau early on are the ones that can't handle getting updated so frequently. |
| Learning Curves (freq=4) | `learning_curves_freq4.png` | This is where we compare algorithms under balanced conditions. It's the standard setup everyone uses to test different variants against each other. You can actually see meaningful differences here because the update schedule isn't extreme in either direction where it's not too sparse and not too aggressive. That's what makes it the best place to fairly judge which algorithm works better. |
| Learning Curves (freq=8) | `learning_curves_freq8.png` | This shows what happens when we update the least frequently: just once every eight steps. If an algorithm can still learn here, it proves you can train effectively without all that computation. Algorithms that break down at this sparsity level are fragile and can't handle infrequent updates. |
| Final Performance (freq=1) | `final_performance_freq1.png` | A bar chart showing how well each algorithm performed at the end of training when updating every single step. The error bars tell you the range of results across different test runs. You can compare this directly to the freq=4 and freq=8 charts to see how much performance drops when we update less often. |
| Final Performance (freq=4) | `final_performance_freq4.png` | The same type of chart, but with updates every four steps instead. Whichever algorithm has the tallest bar here wins the fair comparison. This is the main result the paper focuses on because this is the balanced, standard way everyone tests. |
| Final Performance (freq=8) | `final_performance_freq8.png` | The same chart again, but now updating only once every eight steps. When you compare this against freq=1 and freq=4, you get the answer to the big question: how much performance do you lose when you cut the updates down to one eighth? And is saving that much computation worth the performance hit? |

---

## ✅ Evaluation

Run these after all training and sparsity scripts have completed.

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
