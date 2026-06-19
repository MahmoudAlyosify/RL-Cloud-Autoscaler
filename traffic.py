"""
traffic.py — Poisson Traffic Generator for Cloud Auto-Scaling RL Environment
=============================================================================
CISC 856 · Queen's University · Spring 2026

Generates synthetic web-request arrivals following a sinusoidal day/night
cycle with superimposed random spikes (Black-Friday simulation).
Arrivals are drawn from a Poisson distribution, consistent with the M/M/c
queueing assumption documented in the Project Constitution (Section 4).
"""

import numpy as np


class PoissonTrafficGenerator:
    """Generates Poisson-distributed request arrivals with a sinusoidal
    base rate and stochastic traffic spikes.

    Parameters
    ----------
    base_rate_min : int
        Mean arrival rate during quiet hours (trough of the sine wave).
    base_rate_max : int
        Mean arrival rate during peak hours (crest of the sine wave).
    spike_probability : float
        Per-timestep probability of initiating a new traffic spike.
    spike_multiplier : float
        Multiplicative factor applied to the base rate during a spike.
    period_length : int
        Number of timesteps per full sinusoidal "day" cycle.
    mode : str
        "stochastic" — full Poisson sampling with random spikes.
        "deterministic" — returns rounded base rate, no spikes, no noise.
    seed : int
        Seed for the numpy random generator (reproducibility).
    """

    def __init__(
        self,
        base_rate_min: int = 10,
        base_rate_max: int = 80,
        spike_probability: float = 0.05,
        spike_multiplier: float = 3.0,
        period_length: int = 200,
        mode: str = "stochastic",
        seed: int = 0,
    ):
        self.rmin = base_rate_min
        self.rmax = base_rate_max
        self.p_spike = spike_probability
        self.k_spike = spike_multiplier
        self.period = period_length
        self.mode = mode
        self.rng = np.random.default_rng(seed)

        # Tracks how many timesteps remain in the current spike (0 = no spike)
        self._spike_remaining = 0

    def peek_lambda(self, t: int) -> float:
        """Return the base sinusoidal arrival rate at timestep *t*.

        This is the deterministic component only — no spike multiplier
        and no Poisson sampling. Useful for initializing the EMA in the
        environment's reset() without side effects.

        Formula:
            λ_base(t) = rmin + (rmax - rmin) * (0.5 + 0.5 * sin(2πt / period))

        This keeps λ_base ∈ [rmin, rmax] and provides a learnable
        periodic structure the RL agent can anticipate.
        """
        return self.rmin + (self.rmax - self.rmin) * (
            0.5 + 0.5 * np.sin(2 * np.pi * t / self.period)
        )

    def generate(self, t: int) -> tuple:
        """Generate arrivals for timestep *t*.

        Returns
        -------
        arrivals : int
            Number of requests arriving this timestep.
        current_lambda : float
            The effective arrival rate used (after any spike multiplier).
        """
        # --- 1. Compute the base sinusoidal rate ---
        lam = self.peek_lambda(t)

        if self.mode == "stochastic":
            # --- 2a. Spike logic (stochastic mode only) ---
            if self._spike_remaining > 0:
                # Currently inside a spike: amplify rate, decrement counter
                lam *= self.k_spike
                self._spike_remaining -= 1
            elif self.rng.random() < self.p_spike:
                # No active spike, but we just triggered a new one.
                # Duration is uniformly drawn from [5, 15] timesteps.
                self._spike_remaining = int(self.rng.integers(5, 16))
                lam *= self.k_spike

            # --- 3a. Poisson sample from the (possibly spiked) rate ---
            arrivals = int(self.rng.poisson(lam))
        else:
            # --- 2b/3b. Deterministic mode: no spikes, no noise ---
            arrivals = int(round(lam))

        return arrivals, float(lam)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gen = PoissonTrafficGenerator(seed=42)
    all_arrivals = []
    spike_steps = 0

    for t in range(500):
        arrivals, lam = gen.generate(t)
        all_arrivals.append(arrivals)
        # A spike is active when λ exceeds the theoretical max base rate
        if lam > gen.rmax:
            spike_steps += 1

    arr = np.array(all_arrivals)
    print(f"Arrivals over 500 steps  —  min: {arr.min()}  max: {arr.max()}  "
          f"mean: {arr.mean():.1f}")
    print(f"Spike steps: {spike_steps} / 500")
