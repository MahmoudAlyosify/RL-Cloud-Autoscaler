import json
import os

notebook_path = "pipeline_runbook.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

new_markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Run Full Pipeline\n",
        "This cell runs the entire training and evaluation pipeline sequentially."
    ]
}

new_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Optuna sweep (Quick Test)\n",
        "!python run_optuna_sweep.py --trials 5\n",
        "\n",
        "# Train DQN\n",
        "!python train_dqn.py --timesteps 20000\n",
        "\n",
        "# Train PPO\n",
        "!python train_ppo.py --timesteps 20000\n",
        "\n",
        "# Run Sparse PPO (K-updates experiment)\n",
        "!python sparse_ppo.py --timesteps 20000\n",
        "\n",
        "# Evaluate all trained agents\n",
        "!python eval_agent.py --episodes 5\n",
        "\n",
        "# Generate the publication plots\n",
        "!python plot_results.py\n"
    ]
}

nb["cells"].extend([new_markdown_cell, new_code_cell])

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Cells successfully appended to notebook.")
