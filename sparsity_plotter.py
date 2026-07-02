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
    """Produces six sparsity comparison charts from evaluations.npz paths.

    Parameters
    ----------
    eval_paths : dict[(str, int), list[str]]
        Keys are (algorithm_label, frequency) tuples.
        Values are LISTS of file paths to that run's evaluations.npz
        — one path per seed. Pass a single-item list for algorithms
        with only one seed (e.g. PPO/A2C baselines not yet swept
        over seeds). Missing or unreadable files are skipped
        gracefully; if ALL seeds for a cell are missing, that
        algorithm just won't appear on the corresponding chart.
        Curves from multiple seeds are averaged (aligned to the
        shortest common length), with the shaded band showing
        across-seed std rather than the within-run eval std.
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
                colors: dict = None, model_paths: dict = None):
        self.eval_paths  = eval_paths
        self.model_paths = model_paths or {}   # (label, freq) -> list of (model_zip, vecnorm_pkl) per seed
        self.out_dir     = out_dir
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
        """Load and average evaluations.npz across all seeds for (label, freq).

        Returns dict with "steps", "means" (across-seed mean of the
        per-checkpoint mean reward), "stds" (across-seed std at each
        checkpoint), and "n_seeds" (how many seed runs were actually
        found and used). Returns None if zero seeds were readable.
        """
        seed_paths = self.eval_paths.get((label, freq), [])
        if isinstance(seed_paths, str):
            # Backward-compat: allow a single path string instead of a list
            seed_paths = [seed_paths]

        per_seed_means = []
        steps = None

        for path in seed_paths:
            if path is None or not os.path.exists(path):
                print(f"  [MISSING] {label} freq={freq} — {path}")
                continue
            try:
                data = np.load(path)
                seed_steps = data["timesteps"]
                results = data["results"]
            except Exception as e:
                print(f"  [ERROR] {label} freq={freq} — could not read {path}: {e}")
                continue

            per_seed_means.append(results.mean(axis=1))  # mean over eval episodes
            if steps is None or len(seed_steps) < len(steps):
                steps = seed_steps  # keep the shortest steps array seen so far

        if not per_seed_means:
            print(f"  [SKIP] {label} freq={freq} — no readable seed runs")
            return None

        # Align all seeds to the shortest run (in case one was interrupted early)
        min_len = min(len(m) for m in per_seed_means)
        stacked = np.stack([m[:min_len] for m in per_seed_means])  # (n_seeds, n_evals)
        steps = steps[:min_len]

        means = stacked.mean(axis=0)
        # Across-seed std (variance between runs) rather than within-run
        # episode std — this is the more relevant uncertainty here since
        # each cell may now represent multiple independent seeds.
        stds = stacked.std(axis=0)

        return {"steps": steps, "means": means, "stds": stds, "n_seeds": len(per_seed_means)}

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

            legend_label = f"{label} (n={curve['n_seeds']} seed{'s' if curve['n_seeds'] != 1 else ''})"
            ax.plot(curve["steps"], s, label=legend_label, color=color, linewidth=2)
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
            labels.append(f"{label} (n={curve['n_seeds']})")
            finals.append(curve["means"][-1])   # across-seed mean at final checkpoint
            errs.append(curve["stds"][-1])      # across-seed std at final checkpoint
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

    def _evaluate_model(self, model_zip: str, vecnorm_pkl: str, n_episodes: int = 10):
        """Load one saved model and collect reward, cost, dropped, queue_occ.

        Uses evaluate_agent() from eval_agent.py — the same function
        already used by select_best_dqn.py, so no new evaluation logic
        is introduced here.

        Returns dict with keys: reward, cost, dropped, queue_occ.
        Returns None if model file is missing or evaluation fails.
        """
        from eval_agent import evaluate_agent
        from stable_baselines3 import DQN as SB3DQN
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from env_factory import make_env

        if not os.path.exists(model_zip + ".zip") and not os.path.exists(model_zip):
            print(f"  [MISSING model] {model_zip}")
            return None
        if not os.path.exists(vecnorm_pkl):
            print(f"  [MISSING vecnorm] {vecnorm_pkl}")
            return None

        try:
            model = SB3DQN.load(model_zip)
            eval_env = VecNormalize.load(
                vecnorm_pkl,
                DummyVecEnv([make_env(rank=999, seed=999)])
            )
            eval_env.training = False
            eval_env.norm_reward = False

            metrics = evaluate_agent(model, eval_env, n_episodes=n_episodes)
            eval_env.close()
            return metrics
        except Exception as e:
            print(f"  [ERROR evaluating] {model_zip}: {e}")
            return None

    def collect_domain_metrics(self, n_episodes: int = 10):
        """Re-evaluate all saved models and return a nested dict of domain metrics.

        Returns
        -------
        results : dict[(label, freq)] -> dict with keys:
            reward, cost, dropped, queue_occ — each a list of per-seed values.
            Missing seeds are omitted (not set to NaN) so plots stay clean.
        """
        results = {}
        for (label, freq), seed_model_paths in self.model_paths.items():
            per_seed = {"reward": [], "cost": [], "dropped": [], "queue_occ": []}
            for model_zip, vecnorm_pkl in seed_model_paths:
                metrics = self._evaluate_model(model_zip, vecnorm_pkl, n_episodes)
                if metrics is None:
                    continue
                for key in per_seed:
                    per_seed[key].append(metrics[key])
            if any(per_seed[k] for k in per_seed):
                results[(label, freq)] = per_seed
        return results

    def plot_domain_metrics(self, freq: int, domain_results: dict, filename: str = None):
        """Four-panel bar chart: reward, cost, dropped requests, queue occupancy.

        Each panel shows all algorithms at one update_frequency, with
        error bars representing across-seed std. This is the main
        "complete analysis" figure — reward alone can't distinguish
        a variant that over-provisions to avoid drops from one that
        does it efficiently; these four panels together can.

        Parameters
        ----------
        freq : int
            Which update frequency to plot.
        domain_results : dict
            Output of collect_domain_metrics().
        filename : str or None
            Output PNG name. Defaults to f"domain_metrics_freq{freq}.png".
        """
        METRICS = [
            ("reward",    "Mean Episode Reward",         False),  # (key, ylabel, lower_is_better)
            ("cost",      "Infrastructure Cost\n(cumulative active servers)", True),
            ("dropped",   "Dropped Requests\n(total per episode)",            True),
            ("queue_occ", "Queue Occupancy\n(fraction, 0–1)",                 True),
        ]

        fig, axes = plt.subplots(1, 4, figsize=(20, 6), dpi=150)
        fig.suptitle(f"Domain Metrics — All Algorithms (update_freq={freq})",
                     fontsize=13, fontweight="bold")

        for ax, (metric_key, ylabel, lower_is_better) in zip(axes, METRICS):
            labels_plot, means_plot, errs_plot, colors_plot = [], [], [], []

            for label in self.algorithms:
                data = domain_results.get((label, freq))
                if data is None or not data[metric_key]:
                    continue
                vals = np.array(data[metric_key])
                labels_plot.append(label)
                means_plot.append(vals.mean())
                errs_plot.append(vals.std())
                colors_plot.append(self.colors[label])

            if not labels_plot:
                ax.set_visible(False)
                continue

            x = np.arange(len(labels_plot))
            ax.bar(x, means_plot, yerr=errs_plot, color=colors_plot,
                   alpha=0.85, capsize=4)
            ax.set_xticks(x)
            ax.set_xticklabels(labels_plot, rotation=25, ha="right", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")

            # Annotate direction so reader knows which way is better
            direction = "↓ lower is better" if lower_is_better else "↑ higher is better"
            ax.set_title(direction, fontsize=8, color="gray")

        plt.tight_layout()
        out_name = filename or f"domain_metrics_freq{freq}.png"
        out_path = os.path.join(self.out_dir, out_name)
        plt.savefig(out_path)
        plt.close(fig)
        print(f"  Saved {out_name}")

    def plot_all(self, n_eval_episodes: int = 10):
        """Generate all charts: learning curves, final performance bars,
        and (if model_paths were provided) domain metric panels."""
        print("Generating sparsity comparison charts ...")

        for freq in self.frequencies:
            self.plot_learning_curves(freq)

        for freq in self.frequencies:
            self.plot_final_performance(freq)

        # Domain metrics — only runs if model_paths were provided
        if self.model_paths:
            print("\nRe-evaluating saved models for domain metrics ...")
            domain_results = self.collect_domain_metrics(n_episodes=n_eval_episodes)
            for freq in self.frequencies:
                self.plot_domain_metrics(freq, domain_results)
        else:
            print("\n[INFO] No model_paths provided — skipping domain metrics plots.")
            print("       Pass model_paths to SparsityPlotter() to enable them.")

        print(f"\nDone — charts saved to {self.out_dir}")


if __name__ == "__main__":
    from train_dqn import VARIANT_MAP

    FREQUENCIES = [1, 4, 8]
    SEEDS = [0, 1, 2]  # must match the seeds you actually trained — edit if different

    eval_paths = {}

    # DQN variants: build paths dynamically via each AgentClass's get_paths(),
    # so this stays in sync automatically if PATHS/get_paths change again
    # (e.g. a different seed naming scheme) instead of drifting out of date
    # like the old hardcoded strings did.
    for variant_key, AgentClass in VARIANT_MAP.items():
        label = AgentClass.LABEL
        for freq in FREQUENCIES:
            paths_per_seed = []
            for seed in SEEDS:
                paths = AgentClass.get_paths(freq, seed)
                paths_per_seed.append(os.path.join(paths["eval_log"], "evaluations.npz"))
            eval_paths[(label, freq)] = paths_per_seed

    # Non-DQN baselines (PPO / PPO-LSTM / A2C) aren't trained through
    # train_dqn.py / VARIANT_MAP, so their paths are still hardcoded here.
    # Wrapped in single-item lists since _load() now always expects a list.
    # Add more entries to each list if/when these get a seed sweep too.
    baseline_paths = {
        ("PPO",                1): ["./logs/sparse_ppo_k1_eval/evaluations.npz"],
        ("PPO",                4): ["./logs/sparse_ppo_k4_eval/evaluations.npz"],
        ("PPO",                8): ["./logs/sparse_ppo_k8_eval/evaluations.npz"],
        ("PPO-LSTM",           1): ["./logs/sparse_recurrent_ppo_k1_eval/evaluations.npz"],
        ("PPO-LSTM",           4): ["./logs/sparse_recurrent_ppo_k4_eval/evaluations.npz"],
        ("PPO-LSTM",           8): ["./logs/sparse_recurrent_ppo_k8_eval/evaluations.npz"],
        ("A2C",                1): ["./logs/sparse_a2c_k1_eval/evaluations.npz"],
        ("A2C",                4): ["./logs/sparse_a2c_k4_eval/evaluations.npz"],
        ("A2C",                8): ["./logs/sparse_a2c_k8_eval/evaluations.npz"],
    }
    eval_paths.update(baseline_paths)

    # Model paths for domain metric re-evaluation.
    # Each cell is a list of (model_zip, vecnorm_pkl) tuples, one per seed.
    # Built dynamically via get_paths() for the same reason as eval_paths above.
    model_paths = {}
    for variant_key, AgentClass in VARIANT_MAP.items():
        label = AgentClass.LABEL
        for freq in FREQUENCIES:
            seed_model_paths = []
            for seed in SEEDS:
                paths = AgentClass.get_paths(freq, seed)
                seed_model_paths.append((
                    paths["best_model"] + "best_model",   # SB3 saves best_model.zip here
                    paths["vecnorm"],
                ))
            model_paths[(label, freq)] = seed_model_paths

    # Baselines: point to their own saved model locations.
    # Update these paths to match wherever your PPO/A2C models are saved.
    # Set to an empty list for any baseline you don't want to re-evaluate.
    baseline_model_paths = {
        ("PPO",                1): [("./models/best_sparse_ppo_k1/best_model",     "./models/vecnormalize_sparse_ppo_k1.pkl")],
        ("PPO",                4): [("./models/best_sparse_ppo_k4/best_model",     "./models/vecnormalize_sparse_ppo_k4.pkl")],
        ("PPO",                8): [("./models/best_sparse_ppo_k8/best_model",     "./models/vecnormalize_sparse_ppo_k8.pkl")],
        ("PPO-LSTM",           1): [("./models/best_sparse_recurrent_ppo_k1/best_model", "./models/vecnormalize_sparse_recurrent_ppo_k1.pkl")],
        ("PPO-LSTM",           4): [("./models/best_sparse_recurrent_ppo_k4/best_model", "./models/vecnormalize_sparse_recurrent_ppo_k4.pkl")],
        ("PPO-LSTM",           8): [("./models/best_sparse_recurrent_ppo_k8/best_model", "./models/vecnormalize_sparse_recurrent_ppo_k8.pkl")],
        ("A2C",                1): [("./models/best_sparse_a2c_k1/best_model",     "./models/vecnormalize_sparse_a2c_k1.pkl")],
        ("A2C",                4): [("./models/best_sparse_a2c_k4/best_model",     "./models/vecnormalize_sparse_a2c_k4.pkl")],
        ("A2C",                8): [("./models/best_sparse_a2c_k8/best_model",     "./models/vecnormalize_sparse_a2c_k8.pkl")],
    }
    model_paths.update(baseline_model_paths)

    plotter = SparsityPlotter(
        eval_paths=eval_paths,
        model_paths=model_paths,
        out_dir="./results/Experments/plots_exp2/",
    )
    plotter.plot_all(n_eval_episodes=10)