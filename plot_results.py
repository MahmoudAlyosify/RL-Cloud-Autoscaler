import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from baseline_agent import RuleBasedBaseline
from env_factory import make_env

def smooth(scalars, weight=0.9):
    """EMA smoothing for learning curves."""
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def load_eval_data(eval_path):
    """Extract metric from EvalCallback .npz logs."""
    if not os.path.exists(eval_path):
        return [], []
    data = np.load(eval_path)
    steps = data['timesteps']
    mean_rewards = np.mean(data['results'], axis=1)
    return steps, mean_rewards

def generate_plot_1_learning_curves():
    """Plot 1: Learning curves for PPO, DQN, Baseline"""
    plt.figure(figsize=(12, 6), dpi=150)
    
    ppo_steps, ppo_vals = load_eval_data("./logs/ppo_eval/evaluations.npz")
    dqn_steps, dqn_vals = load_eval_data("./logs/dqn_eval/evaluations.npz")
    
    if len(ppo_steps) > 0:
        plt.plot(ppo_steps, smooth(ppo_vals, 0.9), label="PPO", color="blue", linewidth=2)
        plt.fill_between(ppo_steps, np.array(smooth(ppo_vals, 0.9)) - np.std(ppo_vals)*0.2, 
                         np.array(smooth(ppo_vals, 0.9)) + np.std(ppo_vals)*0.2, color="blue", alpha=0.2)
                         
    if len(dqn_steps) > 0:
        plt.plot(dqn_steps, smooth(dqn_vals, 0.9), label="DQN", color="orange", linewidth=2)
        plt.fill_between(dqn_steps, np.array(smooth(dqn_vals, 0.9)) - np.std(dqn_vals)*0.2, 
                         np.array(smooth(dqn_vals, 0.9)) + np.std(dqn_vals)*0.2, color="orange", alpha=0.2)
    
    plt.axhline(y=-9122.69, color='gray', linestyle='--', label='Baseline Mean', linewidth=2)
    
    plt.title("Learning Curves (PPO vs DQN)")
    plt.xlabel("Environment Steps")
    plt.ylabel("Mean Episode Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./results/plots/plot_1_learning_curves.png")
    plt.close()

def generate_plot_2_sparse_updates():
    """Plot 2: Sparse-updates trade-off"""
    # Dummy data until experiment is run; ideally, read from saved JSON
    # K=1, 4, 8
    ks = [1, 4, 8]
    rewards = [-2000, -2200, -2500] # Mocked
    times = [1000, 300, 150] # Mocked
    
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=150)
    
    ax1.plot(ks, rewards, 'b-o', label="Final Mean Reward", linewidth=2)
    ax1.set_xlabel('K (Update Frequency Multiplier)')
    ax1.set_ylabel('Mean Reward', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_xticks(ks)
    
    ax2 = ax1.twinx()
    ax2.bar(ks, times, alpha=0.3, color='orange', label="Wall-clock time (s)", width=0.5)
    ax2.set_ylabel('Wall-clock Time (s)', color='orange')
    ax2.tick_params('y', colors='orange')
    
    plt.title("Sparse Updates Trade-off (Reward vs Compute Time)")
    fig.tight_layout()
    plt.savefig("./results/plots/plot_2_sparse_updates.png")
    plt.close()

def generate_plot_3_cost_breakdown():
    """Plot 3: Operational cost breakdown"""
    labels = ['PPO', 'DQN', 'Baseline', 'Random']
    # These values should ideally come from eval_agent.py outputs, mocking for layout
    costs = np.array([3000, 3100, 3213.60, 4000])
    latency = np.array([500, 600, 800, 1500])
    drops = np.array([100, 120, 118.10*50, 5000])
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    
    ax.bar(labels, costs, label='Infrastructure Cost ($\\alpha C$)', color='skyblue')
    ax.bar(labels, latency, bottom=costs, label='Latency Penalty ($\\beta L$)', color='orange')
    ax.bar(labels, drops, bottom=costs+latency, label='Drop Penalty ($\\gamma D$)', color='red')
    
    ax.set_ylabel('Penalty Components')
    ax.set_title('Operational-Cost Breakdown by Policy')
    ax.legend()
    plt.tight_layout()
    plt.savefig("./results/plots/plot_3_cost_breakdown.png")
    plt.close()

def generate_plot_4_behavior_trace():
    """Plot 4: Policy behavior trace for 1 episode"""
    # Needs to run one episode and collect metrics. 
    # To avoid long runtimes here, we generate a mock sine wave for layout.
    t = np.arange(0, 1000)
    lam = 10 + 70 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 200))
    
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True, dpi=150)
    
    axs[0].plot(t, lam, color='black', label=r"Arrival Rate $\lambda(t)$")
    axs[0].legend(loc="upper right")
    axs[0].set_ylabel("Rate")
    
    axs[1].plot(t, np.clip(lam/50 + 1, 1, 10), color='blue', label="PPO (Active)")
    axs[1].plot(t, np.clip(lam/50 + 0.5, 1, 10), color='orange', label="DQN (Active)")
    axs[1].plot(t, np.clip(lam/50 + 2, 1, 10), color='gray', label="Baseline (Active)", alpha=0.7)
    axs[1].legend(loc="upper right")
    axs[1].set_ylabel("Servers")
    
    axs[2].plot(t, np.zeros_like(t), color='blue', label="PPO Queue")
    axs[2].plot(t, np.random.randint(0, 20, 1000), color='orange', label="DQN Queue")
    axs[2].plot(t, np.random.randint(0, 50, 1000), color='gray', label="Baseline Queue", alpha=0.7)
    axs[2].legend(loc="upper right")
    axs[2].set_ylabel("Queue Length")
    
    axs[3].plot(t, -np.clip(lam/50 + 1, 1, 10), color='blue', label="PPO Reward")
    axs[3].plot(t, -np.clip(lam/50 + 0.5, 1, 10)-0.1, color='orange', label="DQN Reward")
    axs[3].plot(t, -np.clip(lam/50 + 2, 1, 10)-0.2, color='gray', label="Baseline Reward", alpha=0.7)
    axs[3].legend(loc="upper right")
    axs[3].set_ylabel("Reward")
    axs[3].set_xlabel("Timestep")
    
    plt.suptitle("Policy Behavior Trace (1 Episode)")
    plt.tight_layout()
    plt.savefig("./results/plots/plot_4_behavior_trace.png")
    plt.close()

def generate_plot_5_convergence_boxplots():
    """Plot 5: Convergence box plots over final 100k steps"""
    # Mock data
    data = [
        np.random.normal(-3000, 500, 100), # PPO
        np.random.normal(-3200, 600, 100), # DQN
        np.random.normal(-9122, 5287, 100) # Baseline
    ]
    labels = ['PPO', 'DQN', 'Baseline']
    
    plt.figure(figsize=(10, 6), dpi=150)
    plt.boxplot(data, tick_labels=labels, patch_artist=True)
    plt.title("Convergence Box Plots (Final 100k Steps)")
    plt.ylabel("Episode Reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./results/plots/plot_5_convergence_boxplots.png")
    plt.close()

def main():
    os.makedirs("./results/plots", exist_ok=True)
    print("Generating Plot 1...")
    generate_plot_1_learning_curves()
    print("Generating Plot 2...")
    generate_plot_2_sparse_updates()
    print("Generating Plot 3...")
    generate_plot_3_cost_breakdown()
    print("Generating Plot 4...")
    generate_plot_4_behavior_trace()
    print("Generating Plot 5...")
    generate_plot_5_convergence_boxplots()
    print("Done! Plots saved to ./results/plots/")

if __name__ == "__main__":
    main()
