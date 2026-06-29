"""

Why combining both helps beyond either alone
---------------------------------------------
DuelingDQN alone: learns state values faster (V(s) updated every
step), but the TD target is still overestimated because argmax
is applied to the target network's own Q-values.

DoubleDQN alone: fixes the overestimation bias in the TD target,
but still wastes learning signal on steps where the taken action
is "Hold" and the other two actions' Q-values sit frozen.

What this class overrides
--------------------------
- DoubleDuelingDQNPolicy → identical to DuelingDQNPolicy in effect, separate class for naming clarity
- train()
"""

from stable_baselines3 import DQN as SB3DQN
from stable_baselines3.dqn.policies import DQNPolicy

from base_dqn import BaseDQN
from dueling_dqn import DuelingQNetwork   # reuse — no duplication




# Policy that builds both online and target networks as DuelingQNetwork.
class DoubleDuelingDQNPolicy(DQNPolicy):
    q_net_class = DuelingQNetwork

# Dueling architecture + Double-Q target.
class DoubleDuelingDQN(BaseDQN):
    """
    Parameters
    ----------
    env : GymEnv
        Training environment wrapped in VecNormalize.
    update_frequency : int
        Steps between gradient updates. 1/2/4/8 for the ablation.
        Default 4 matches the existing train_dqn.py configuration.
    **kwargs
        Forwarded to BaseDQN → SB3's DQN.__init__ unchanged.
        Pass SHARED_HYPERPARAMS here as **SHARED_HYPERPARAMS.
    """

    LABEL = "Double + Dueling DQN"
    SLUG  = "double_dueling_dqn"

    PATHS = {
        "log_dir": "./logs/{slug}_freq{freq}/",
        "eval_log": "./logs/{slug}_freq{freq}_eval/",
        "best_model": "./models/best_{slug}_freq{freq}/",
        "checkpoint": "./checkpoints/{slug}_freq{freq}/",
        "final_model": "./models/final_{slug}_freq{freq}",
        "vecnorm": "./models/vecnormalize_{slug}_freq{freq}.pkl",
    }

    def __init__(self, env, update_frequency: int = 4, **kwargs):
        super().__init__(
            policy=DoubleDuelingDQNPolicy,  # Dueling architecture injected
            env=env,
            update_frequency=update_frequency,
            **kwargs,
        )


    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        if self.num_timesteps % self.update_frequency != 0:
            return
        original_forward = self.q_net_target.forward

        def _online_forward(obs):
            """Return Dueling online Q-values in place of Dueling target Q-values."""
            return self.q_net.forward(obs)

        self.q_net_target.forward = _online_forward
        try:
            SB3DQN.train(self, gradient_steps, batch_size)
        finally:
            self.q_net_target.forward = original_forward

    # Return fully formatted output paths for a given update_frequency.
    @classmethod
    def get_paths(cls, update_frequency: int) -> dict:
        return {
            key: template.format(slug=cls.SLUG, freq=update_frequency)
            for key, template in cls.PATHS.items()
        }