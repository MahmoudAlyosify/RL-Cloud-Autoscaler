# Autonomous Cloud Resource Provisioning via Reinforcement Learning ☁️

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Standard-brightgreen)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO%20%7C%20DQN-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

## 📌 Project Overview
This repository contains an implementation of autonomous cloud resource provisioning using Deep Reinforcement Learning (DQN & PPO). The auto-scaling problem is formulated as a Markov Decision Process (MDP) inside a custom Gymnasium environment designed to model realistic cloud behaviours such as cold-start delays, minimum server counts and stochastic workloads.

The goal is to learn dynamic scaling policies that minimize infrastructure costs while keeping latency low and avoiding dropped requests under non-stationary traffic patterns.

## 🚀 Key Features
- **Custom Gymnasium Environment:** A mathematically-bounded cloud simulator that enforces constraints like minimum servers (N_min), boot-up latencies and capacity limits.
- **Deep RL Agents:** Implementations and benchmarks for Proximal Policy Optimization (PPO) and Deep Q-Networks (DQN), including common variants (double, dueling, double-dueling).
- **Proactive Scaling:** The environment models a cold-start delay (k timesteps) that forces agents to predict future traffic rather than purely react.
- **Vectorized Training:** Support for parallelized training using SubprocVecEnv to speed up data collection and learning.
- **Reproducible Experiments:** Scripts for training, evaluating and plotting results; seedable evaluation runs for reproducibility.

## 📁 Repository contents
- custom_envs/        - Gymnasium environment and wrappers
- agents/             - DQN, PPO and other agent implementations
- experiments/        - Training & evaluation scripts, ablation studies
- notebooks/          - Jupyter notebooks with experiments and visualizations
- tests/              - Unit and integration tests
- environment.yml     - Conda environment specification
- requirements.txt    - Python dependencies (lighter alternative to environment.yml)

(If your directory layout differs, update these paths accordingly.)

## 🛠️ Requirements
You can either use the provided Conda environment or pip install the required packages.

Conda (recommended):

```bash
conda env create -f environment.yml
conda activate rl-cloud-autoscaler
```

Or with pip (create a venv first):

```bash
pip install -r requirements.txt
# or the minimal set:
pip install stable-baselines3 sb3-contrib gymnasium numpy matplotlib torch
```

## 🏃 Running training scripts
Below are common entrypoints. Customize flags (timesteps, device, seeds) as needed.

- Train vanilla DQN (default):

```bash
python train_dqn.py
```

- DQN variants:

```bash
python train_dqn.py --variant double
python train_dqn.py --variant dueling
python train_dqn.py --variant double_dueling
```

- Sparsity / update frequency ablation (examples):

```bash
python train_dqn.py --variant vanilla --update_frequency 1
python train_dqn.py --variant vanilla --update_frequency 4
python train_dqn.py --variant vanilla --update_frequency 8
```

- A2C variants:

```bash
python train_a2c.py --variant default
python train_a2c.py --variant low_entropy
python train_a2c.py --variant high_entropy
python train_a2c.py --variant short_rollout
python train_a2c.py --variant long_rollout
```

- PPO (including recurrent PPO/LSTM variants):

```bash
python train_ppo.py --timesteps 2000000 --device auto
python train_recurrent_ppo.py --variant default --timesteps 2000000 --device auto
```

Adjust `--timesteps`, `--device` and `--seed` according to your hardware and experiment plan.

## ✅ Evaluation & Comparison
After training, run evaluation scripts to measure performance and generate plots:

```bash
python run_baseline_eval.py
python eval_agent.py --episodes 10 --seed 42
python main_algorithm_comparison.py --eval-seeds 0,1,2,3,4 --episodes-per-seed 1
python traffic_stress_test.py --seeds 0,1,2,3,4
python plot_results.py
```

## 🧪 Notebooks
The `notebooks/` directory contains interactive experiments and visualization pipelines. Use them to reproduce plots, inspect learning curves, and run quick experiments.

## 🔧 Tips & Troubleshooting
- Ensure consistent package versions across experiments by using `environment.yml`.
- Use smaller environments and fewer timesteps for quick debugging before full-scale runs.
- If you see unstable training, try lowering the learning rate, increasing batch size, or adjusting entropy/epsilon schedules depending on the algorithm.

## 📚 Citation / Acknowledgements
If you use this code in research, please cite the repository and any relevant papers you base your work on.

## 🤝 Contributing
Contributions are welcome. Please open an issue for major changes and submit PRs with tests and clear descriptions.

## 📝 License
Specify your license here (e.g., MIT). If you want me to add a LICENSE file, tell me which license to use and I'll add it.

## ✉️ Contact
Maintainer: MahmoudAlyosify

---

(README updated to fix formatting, expand installation and usage instructions, and add a clearer project overview. If you want the README in Arabic or you want specific details added to reflect exact code changes you made, tell me what changed and I'll update the README further.)
