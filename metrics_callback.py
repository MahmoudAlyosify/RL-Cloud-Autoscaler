"""Shared TensorBoard callback for PPO and DQN training.

logger.record() only buffers values -- you MUST call logger.dump()
to actually write them to the event file, otherwise custom scalars
silently never appear.
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MetricsCallback(BaseCallback):
    """Logs dropped requests, active servers, and queue length every 1k steps."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos and self.num_timesteps % 1000 == 0:
            self.logger.record("custom/dropped_requests_per_episode",
                               np.mean([i.get("dropped", 0) for i in infos]))
            self.logger.record("custom/active_servers_mean",
                               np.mean([i.get("active", 0) for i in infos]))
            self.logger.record("custom/queue_length_mean",
                               np.mean([i.get("queue", 0) for i in infos]))
            self.logger.dump(self.num_timesteps)
        return True
