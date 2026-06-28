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
   
2. Install dependencies:
   ```pip install -r requirements.txt```

## 🏃 Running the Scripts
### Vanilla DQN (default)
    python train_dqn.py

### DQN variants
    python train_dqn.py --variant double
    python train_dqn.py --variant dueling
    python train_dqn.py --variant double_dueling

### Sparsity ablation studies
    python train_dqn.py --variant vanilla --update_frequency 1
    python train_dqn.py --variant vanilla --update_frequency 8


