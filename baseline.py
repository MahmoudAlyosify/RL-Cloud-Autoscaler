""" The baseline agent implements a threshold-based autoscaling policy.
 It observes the normalized environment state, converts the needed values back to
  their original scale, and applies fixed CPU/queue rules to decide whether to scale out,
   hold, or scale in. This provides a simple non-learning benchmark for evaluating PPO and DQN."""


"""Threshold-based auto-scaler baseline for the cloud RL project."""

import numpy as np


class RuleBasedBaseline:
    """Rule-based heuristic that mimics a traditional cloud auto-scaler.

    This baseline does not learn. It uses fixed CPU and queue thresholds
    to decide whether to scale out, hold, or scale in.

    Action mapping:
        0 = scale out
        1 = hold
        2 = scale in

    The predict() method follows the Stable-Baselines3 interface so this
    baseline can be evaluated using the same code as PPO and DQN.
    """

    def __init__(
        self,
        n_max=10,
        q_max=500,
        urgent_cpu=0.80,
        urgent_queue=100,
        moderate_cpu=0.65,
        moderate_queue=50,
        low_cpu=0.30,
    ):
        # Maximum number of servers and maximum queue length.
        # These are needed because the environment observation is normalized.
        self.n_max = n_max
        self.q_max = q_max

        # Urgent scale-out thresholds.
        # If CPU and queue are both very high, the baseline adds a server.
        self.urgent_cpu = urgent_cpu
        self.urgent_queue = urgent_queue

        # Moderate scale-out thresholds.
        # These make the baseline scale out before the urgent threshold.
        self.moderate_cpu = moderate_cpu
        self.moderate_queue = moderate_queue

        # Scale-in threshold.
        # If CPU is very low, queue is empty, and no server is booting,
        # the baseline removes one server.
        self.low_cpu = low_cpu

    def predict(self, obs, deterministic=True):
        """Choose a scaling action from the current observation."""

        # Stable-Baselines3/vectorized environments may pass observations
        # with shape (1, 5). A raw Gymnasium environment passes shape (5,).
        is_vectorized = len(obs.shape) == 2
        if is_vectorized:
            obs = obs[0]

        # The environment gives normalized values.
        # Convert the values needed by the rules back to real scale.
        booting = obs[1] * self.n_max
        cpu_util = obs[2]
        queue = obs[3] * self.q_max

        # Rule 1: urgent scale out.
        # The system is under strong pressure, so request one more server.
        if cpu_util > self.urgent_cpu and queue > self.urgent_queue:
            action = 0

        # Rule 2: moderate scale out.
        # The system is becoming busy, so scale out earlier.
        elif cpu_util > self.moderate_cpu and queue > self.moderate_queue:
            action = 0

        # Rule 3: scale in.
        # Only remove a server when the system is clearly underloaded.
        # We also require no booting servers to avoid unnecessary oscillation.
        elif cpu_util < self.low_cpu and queue == 0 and booting == 0:
            action = 2

        # Rule 4: hold.
        # If none of the scale-out or scale-in conditions are met,
        # keep the current capacity.
        else:
            action = 1

        # Return action in the format expected by the caller.
        # Vectorized envs expect an array; raw envs can use an integer.
        if is_vectorized:
            return np.array([action]), None

        return int(action), None