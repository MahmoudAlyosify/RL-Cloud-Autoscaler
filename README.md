# Autonomous Cloud Resource Provisioning via Reinforcement Learning ☁️

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Standard-brightgreen)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO%20%7C%20DQN-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

## 📌 Project Overview
This project investigates the use of Deep Reinforcement Learning (DRL) for autonomous cloud resource provisioning. The auto-scaling problem is formulated as a Markov Decision Process (MDP), where an RL agent learns when to add, remove, or maintain server capacity based on the current state of the system. 

The goal is to learn a dynamic scaling policy that minimizes infrastructure cost while maintaining low latency and avoiding dropped requests under highly stochastic, non-stationary workloads (Poisson-distributed traffic).

## 🚀 Key Features
- **Custom Gymnasium Environment:** A rigorously mathematically bounded cloud simulator enforcing constraints like `$N_{min}$` servers and boot-up latencies.
- **Deep RL Agents:** Benchmarking Proximal Policy Optimization (PPO) and Deep Q-Networks (DQN).
- **Proactive Scaling:** The environment includes a cold-start delay (`k` timesteps), forcing the agent to predict future traffic rather than purely reacting.
- **Vectorized Training:** Utilizing `SubprocVecEnv` for massive parallel training acceleration.

## 🛠️ Installation & Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/RL-Cloud-Autoscaler.git](https://github.com/YOUR_USERNAME/RL-Cloud-Autoscaler.git)
   cd RL-Cloud-Autoscaler
   
2. Create and activate the conda environment:
    ```
    conda env create -f environment.yml
    conda activate rl-cloud-autoscaler

## 🏃 Running the Scripts
### Vanilla DQN (default)
    python train_dqn.py

### DQN variants
    # Just use default train_freq=1
    python train_dqn.py --variant double
    python train_dqn.py --variant dueling
    python train_dqn.py --variant double_dueling
    
### A2C variants
python train_a2c.py --variant default
python train_a2c.py --variant low_entropy
python train_a2c.py --variant high_entropy
python train_a2c.py --variant short_rollout
python train_a2c.py --variant long_rollout

### PPO-LSTM variants
python train_recurrent_ppo.py --variant default
python train_recurrent_ppo.py --variant hidden64
python train_recurrent_ppo.py --variant hidden256
python train_recurrent_ppo.py --variant short_sequence
python train_recurrent_ppo.py --variant long_sequence


pip install stable-baselines3 sb3-contrib gymnasium numpy matplotlib torch
Then run training one after the other:
python train_ppo.py --timesteps 2000000 --device auto
python sparse_ppo.py --timesteps 500000 --device auto
python train_dqn.py --variant double --timesteps 2000000 --device auto
python train_dqn.py --variant dueling --timesteps 2000000 --device auto
python train_dqn.py --variant double_dueling --timesteps 2000000 --device auto
python train_a2c.py --timesteps 2000000 --device auto --seed 0
python train_recurrent_ppo.py --timesteps 2000000 --device auto --seed 0
After all training finishes, run evaluation:
python run_baseline_eval.py
python eval_agent.py --episodes 10 --seed 42
If you added the comparison/stress-test files:
python main_algorithm_comparison.py --eval-seeds 0,1,2,3,4 --episodes-per-seed 1
python traffic_stress_test.py --seeds 0,1,2,3,4
Then generate plots:
python plot_results.py


## 🏃 Running Experiments' Scripts

### Experiment 1: Sparsity ablation studies

#### DQN Sparsity Ablation Training

    python train_dqn.py --variant vanilla --update_frequency 1
    python train_dqn.py --variant vanilla --update_frequency 4
    python train_dqn.py --variant vanilla --update_frequency 8
    python train_dqn.py --variant double --update_frequency 1
    python train_dqn.py --variant double --update_frequency 4
    python train_dqn.py --variant double --update_frequency 8
    python train_dqn.py --variant dueling --update_frequency 1
    python train_dqn.py --variant dueling --update_frequency 4
    python train_dqn.py --variant dueling --update_frequency 8
    python train_dqn.py --variant double_dueling --update_frequency 1
    python train_dqn.py --variant double_dueling --update_frequency 4
    python train_dqn.py --variant double_dueling --update_frequency 8

<p style="color: red; font-weight: bold;">⚠️ Note: For the upcoming algorithms you must run the default training command first to generate the best optimized model then test sparsity effects.</p>
    
#### PPO Sparsity Ablation Training

    python train_ppo.py --timesteps 2000000 --device auto
    python sparse_ppo.py

#### Recurrent PPO Sparsity Ablation Training

    python train_recurrent_ppo.py --timesteps 2000000 --device auto --seed 0
    python sparse_recurrent_ppo.py
    
#### AC2 Sparsity Ablation Training

    python train_a2c.py --timesteps 2000000 --device auto --seed 0
    python sparse_a2c.py
