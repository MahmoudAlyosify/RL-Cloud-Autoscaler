"""Threshold-based auto-scaler baseline for cloud RL project."""
import numpy as np


class RuleBasedBaseline:
    """Rule-based heuristic that mimics production auto-scalers.
    Returns (action, None) to match SB3's model.predict() signature.

    Note: this heuristic never uses arrival_rate (obs[4]), so it
    can't anticipate traffic spikes -- that's the gap RL should exploit.
    """

    def __init__(self, n_max=10, q_max=500):
        self.n_max = n_max
        self.q_max = q_max

    def predict(self, obs, deterministic=True):
        if len(obs.shape) == 2:
            obs = obs[0]
        # denormalize
        booting = obs[1] * self.n_max
        cpu_util = obs[2]
        queue = obs[3] * self.q_max

        if cpu_util > 0.80 and queue > 100:       # urgent
            action = 0
        elif cpu_util > 0.65 and queue > 50:      # proactive
            action = 0
        elif cpu_util < 0.30 and queue == 0 and booting == 0:
            action = 2                             # scale in
        else:
            action = 1                             # hold

        return np.array([action]), None
