"""
Double DQN target:
    y = r + γ · Q_target(s', argmax_a Q_online(s', a))
                          └─── online net selects ────┘
                └─────── target net evaluates ─────────┘
    Fix: the online network picks the action (less biased
    because it is being actively trained and regularized),
    the target network scores it (stable, updated slowly).
    The two networks disagree slightly — this disagreement
    cancels the upward bias.

Why this kind of varient is expected to matter
----------------------------------------------
The problem :: At the start of training, the Q-network is a randomly initialized neural network. It has never seen a real episode.
So its Q-values for any state are essentially noise where some actions get slightly high estimates by pure chance, some get slightly low ones.

The reward function has extreme outliers.
At a given timestep where requests are dropped gives reward ≈ -1500 while a normal traffic timestep gives ≈ -3.
Vanilla DQN sees a few lucky "hold" actions during light traffic that happen to avoid drops and overestimates how good "hold" actually is.
This makes it slow to scale out when a spike starts.

On the other hand Double DQN is more learns that "hold" is only good when traffic is low not good at all situations.
The difference is which network does which job.
Vanilla DQN: the target network both selects the best action and evaluates its value. Same network, same weights, same biases.
Double DQN: the online network selects which action looks best while another network (target network) evaluates how good that action actually is.

What this class overrides
--------------------------
- _compute_td_target() → Implements the Double-Q formula
- train() → A mechanism that makes _compute_td_target() take effect inside SB3's training loop

"""

import torch
from base_dqn import BaseDQN


class DoubleDQN(BaseDQN):
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

    LABEL = "Double DQN"
    SLUG  = "double_dqn"

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
            policy="MlpPolicy",
            env=env,
            update_frequency=update_frequency,
            **kwargs,
        )

    #  Use the Double-Q formula to override the TD target computation
    def _compute_td_target(
        self,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Double DQN TD target.

        Parameters
        ----------
        next_observations : torch.Tensor, shape (batch, obs_dim)
            Next states s' sampled from the replay buffer.
        rewards : torch.Tensor, shape (batch,)
            Immediate rewards r.
        dones : torch.Tensor, shape (batch,)
            Terminal flags (1.0 if terminal, else 0.0).
        """
        with torch.no_grad():
            # Step 1 online network scores all actions at s'
            online_next_q = self.q_net(next_observations)
            # Step 2 online network selects the best action
            best_actions = online_next_q.argmax(dim=1)
            # Step 3 target network scores all actions at s'
            target_next_q = self.q_net_target(next_observations)
            # Step 4 target network evaluates the online-selected action and gather() picks one value per row using best_actions as column index
            best_next_q = target_next_q.gather(dim=1, index=best_actions.unsqueeze(1)).squeeze(1)
            # Step 5 mask terminals and discount
            best_next_q = best_next_q * (1.0 - dones)

        return rewards + self.gamma * best_next_q

    #  Override the training to make _compute_td_target work
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """Apply sparse-update gate, then run Double-Q patched training.

        The mechanism used here is a monkey-patch:
            1. Save the target network's original forward() method.
            2. Replace it temporarily with a function that returns the online network Q-values instead.
            3. SB3's loop now calls the patched forward() when computing next_q_values for action selection & argmax picks from the online net.
            4. SB3 then calls the target network in the 2nd time for evaluation.
            5. Always restore the original forward() in finally block.

        """
        # Sparse-update ablation
        if self.num_timesteps % self.update_frequency != 0:
            return
        # Swap target forward with online forward
        original_forward = self.q_net_target.forward

        def _online_forward(obs):
            """Return online net Q-values in place of target net Q-values."""
            return self.q_net.forward(obs)

        self.q_net_target.forward = _online_forward

        try:
            from stable_baselines3 import DQN as SB3DQN
            SB3DQN.train(self, gradient_steps, batch_size)
        finally:
            # Restore original forward even on exception or interrupt
            self.q_net_target.forward = original_forward


    @classmethod
    def get_paths(cls, update_frequency: int) -> dict:
        return {
            key: template.format(slug=cls.SLUG, freq=update_frequency)
            for key, template in cls.PATHS.items()
        }