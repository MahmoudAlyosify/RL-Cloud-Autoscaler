"""
sparsity_plotter.py

Reads evaluations.npz files directly by path

Produces
--------
1) Learning curves at freq=1, all algorithms, one figure
2) Learning curves at freq=4, all algorithms, one figure
3) Learning curves at freq=8, all algorithms, one figure
4) Final performance bar chart at freq=1, all algorithms
5) Final performance bar chart at freq=4, all algorithms
6) Final performance bar chart at freq=8, all algorithms

"""

import os
import numpy as np
import matplotlib.pyplot as plt


class SparsityPlotter:
    """Produces six sparsity comparison charts directly from evaluations.npz paths.

    Parameters
    ----------
    eval_paths : dict[(str, int), str]
        Keys are (algorithm_label, frequency) tuples.
        Values are file paths to that run's evaluations.npz.
        Missing or unreadable files are skipped gracefully — that
        algorithm just won't appear on the corresponding chart.
    out_dir : str
        Directory where the six PNG files are saved.
    colors : dict[str, str] or None
        Optional fixed color per algorithm label, so the same
        algorithm has the same color across all six charts.
        If not given, a default palette is assigned automatically.
    """

    DEFAULT_PALETTE = [
        "darkorange", "green", "crimson", "purple",
        "royalblue", "mediumorchid", "darkcyan", "saddlebrown",
    ]

    def __init__(self, eval_paths: dict, out_dir: str = "./results/Experments/plots_exp2/",
                colors: dict = None):
        self.eval_paths = eval_paths
        self.out_dir    = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        # derive the algorithm list and frequency list from the keys
        self.algorithms = sorted({label for (label, freq) in eval_paths})
        self.frequencies = sorted({freq for (label, freq) in eval_paths})

        if colors is not None:
            self.colors = colors
        else:
            self.colors = {
                label: self.DEFAULT_PALETTE[i % len(self.DEFAULT_PALETTE)]
                for i, label in enumerate(self.algorithms)
            }

    # Load Files
    def _load(self, label: str, freq: int):
        """Load one evaluations.npz. Returns dict or None if missing/unreadable."""
        path = self.eval_paths.get((label, freq))
        if path is None or not os.path.exists(path):
            print(f"  [MISSING] {label} freq={freq} — {path}")
            return None

        try:
            data = np.load(path)
            steps = data["timesteps"]
            results = data["results"]
        except Exception as e:
            print(f"  [ERROR] {label} freq={freq} — could not read {path}: {e}")
            return None

        means = results.mean(axis=1)
        stds  = results.std(axis=1)
        return {"steps": steps, "means": means, "stds": stds}

    @staticmethod
    def _smooth(values, weight=0.9):
        """EMA smoothing for cleaner learning curve lines."""
        last, out = values[0], []
        for v in values:
            last = last * weight + (1 - weight) * v
            out.append(last)
        return np.array(out)

    def plot_learning_curves(self, freq: int, filename: str = None):
        """One figure: all algorithms' learning curves at a single frequency.

        Parameters
        ----------
        freq : int
            Which update frequency to plot (1, 4, or 8).
        filename : str or None
            Output PNG name. Defaults to f"learning_curves_freq{freq}.png".
        """
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        plotted_any = False

        for label in self.algorithms:
            curve = self._load(label, freq)
            if curve is None:
                continue

            s = self._smooth(curve["means"])
            d = curve["stds"]
            color = self.colors[label]

            ax.plot(curve["steps"], s, label=label, color=color, linewidth=2)
            ax.fill_between(curve["steps"], s - d * 0.3, s + d * 0.3,
                            color=color, alpha=0.15)
            plotted_any = True

        if not plotted_any:
            print(f"  [SKIP] No data available for freq={freq} — chart not saved")
            plt.close(fig)
            return

        ax.set_title(f"Learning Curves — All Algorithms (update_freq={freq})",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Mean Episode Reward")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out_name = filename or f"learning_curves_freq{freq}.png"
        out_path = os.path.join(self.out_dir, out_name)
        plt.savefig(out_path)
        plt.close(fig)
        print(f"  Saved {out_name}")

    def plot_final_performance(self, freq: int, filename: str = None):
        """One figure: bar chart of final mean reward, all algorithms, one frequency.

        Parameters
        ----------
        freq : int
            Which update frequency to plot (1, 4, or 8).
        filename : str or None
            Output PNG name. Defaults to f"final_performance_freq{freq}.png".
        """
        labels, finals, errs, colors = [], [], [], []

        for label in self.algorithms:
            curve = self._load(label, freq)
            if curve is None:
                continue
            labels.append(label)
            finals.append(curve["means"][-1])
            errs.append(curve["stds"][-1])
            colors.append(self.colors[label])

        if not labels:
            print(f"  [SKIP] No data available for freq={freq} — chart not saved")
            return

        fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
        x = np.arange(len(labels))

        ax.bar(x, finals, yerr=errs, color=colors, alpha=0.85, capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Final Mean Episode Reward")
        ax.set_title(f"Final Performance — All Algorithms (update_freq={freq})",
                     fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        out_name = filename or f"final_performance_freq{freq}.png"
        out_path = os.path.join(self.out_dir, out_name)
        plt.savefig(out_path)
        plt.close(fig)
        print(f"  Saved {out_name}")

    def plot_all(self):
        """Generate all six charts: 3 learning curve figures + 3 bar chart figures."""
        print("Generating sparsity comparison charts ...")

        for freq in self.frequencies:
            self.plot_learning_curves(freq)

        for freq in self.frequencies:
            self.plot_final_performance(freq)

        print(f"\nDone — charts saved to {self.out_dir}")


if __name__ == "__main__":
    eval_paths = {
        ("Vanilla DQN",        1): "./logs/vanilla_dqn_freq1_eval/evaluations.npz",
        ("Vanilla DQN",        4): "./logs/vanilla_dqn_freq4_eval/evaluations.npz",
        ("Vanilla DQN",        8): "./logs/vanilla_dqn_freq8_eval/evaluations.npz",
        ("Double DQN",         1): "./logs/double_dqn_freq1_eval/evaluations.npz",
        ("Double DQN",         4): "./logs/double_dqn_freq4_eval/evaluations.npz",
        ("Double DQN",         8): "./logs/double_dqn_freq8_eval/evaluations.npz",
        ("Dueling DQN",        1): "./logs/dueling_dqn_freq1_eval/evaluations.npz",
        ("Dueling DQN",        4): "./logs/dueling_dqn_freq4_eval/evaluations.npz",
        ("Dueling DQN",        8): "./logs/dueling_dqn_freq8_eval/evaluations.npz",
        ("Double+Dueling DQN", 1): "./logs/double_dueling_dqn_freq1_eval/evaluations.npz",
        ("Double+Dueling DQN", 4): "./logs/double_dueling_dqn_freq4_eval/evaluations.npz",
        ("Double+Dueling DQN", 8): "./logs/double_dueling_dqn_freq8_eval/evaluations.npz",
        ("PPO",                1): "./logs/sparse_ppo_k1_eval/evaluations.npz",
        ("PPO",                4): "./logs/sparse_ppo_k4_eval/evaluations.npz",
        ("PPO",                8): "./logs/sparse_ppo_k8_eval/evaluations.npz",
        ("PPO-LSTM",           1): "./logs/sparse_recurrent_ppo_k1_eval/evaluations.npz",
        ("PPO-LSTM",           4): "./logs/sparse_recurrent_ppo_k4_eval/evaluations.npz",
        ("PPO-LSTM",           8): "./logs/sparse_recurrent_ppo_k8_eval/evaluations.npz",
        ("A2C",                1): "./logs/sparse_a2c_k1_eval/evaluations.npz",
        ("A2C",                4): "./logs/sparse_a2c_k4_eval/evaluations.npz",
        ("A2C",                8): "./logs/sparse_a2c_k8_eval/evaluations.npz",
    }

    plotter = SparsityPlotter(eval_paths, out_dir="./results/Experments/plots_exp2/")
    plotter.plot_all()