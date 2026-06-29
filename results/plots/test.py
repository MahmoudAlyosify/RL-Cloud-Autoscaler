import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Load evaluation files
# ==========================

models = {
    "PPO": "logs/ppo_eval/evaluations.npz",
    "A2C": "logs/a2c_eval/evaluations.npz",
    "DQN": "logs/vanilla_dqn_freq4_eval/evaluations.npz",
    "Double DQN": "logs/double_dqn_freq4_eval/evaluations.npz",
    "Dueling Double DQN": "logs/double_dueling_dqn_freq4_eval/evaluations.npz",
    "Recurrent PPO": "logs/recurrent_ppo_eval/evaluations.npz"
}

plt.figure(figsize=(12, 7))

for name, path in models.items():

    data = np.load(path)

    timesteps = data["timesteps"]
    results = data["results"]

    mean_reward = results.mean(axis=1)
    std_reward = results.std(axis=1)

    # Mean curve
    plt.plot(
        timesteps,
        mean_reward,
        linewidth=2,
        label=name
    )

    # Standard deviation
    plt.fill_between(
        timesteps,
        mean_reward - std_reward,
        mean_reward + std_reward,
        alpha=0.15
    )

plt.title("Learning Curves Comparison", fontsize=16)
plt.xlabel("Training Timesteps", fontsize=13)
plt.ylabel("Mean Evaluation Reward", fontsize=13)

plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig("results/plots/full_learning_curves_comparison.png", dpi=300)
plt.show()