---
title: "Project Constitution"
subtitle: "Autonomous Cloud Resource Provisioning via Deep Reinforcement Learning"
author: "CISC 856 — Reinforcement Learning · Queen's University · Spring 2026"
date: "Single source of truth · Drafted June 17, 2026"
---

\newpage

# 1 — Executive Summary & Project Philosophy

Cloud platforms must continuously decide how many server instances to keep running. The dominant production approach is the static, threshold-based auto-scaler: a rule such as *"if average CPU > 70% for 5 minutes, add an instance."* These heuristics fail under stochastic, bursty traffic for two structural reasons. First, they are **reactive** — they only respond after utilization has already crossed a threshold, by which point the request queue may already be growing and the new servers they request will not be ready for several seconds (cold start). Second, they are **myopic** — a single threshold cannot encode the trade-off between the cost of an idle server and the cost of a dropped request, and it cannot anticipate a recurring traffic pattern (a daily peak, a Black-Friday spike) before it arrives.

The core thesis of this project is that a Deep Reinforcement Learning (DRL) agent can learn a **proactive, cost-optimal scaling policy** that dominates any fixed heuristic, because the agent observes the full system state, optimizes a long-horizon discounted return rather than an instantaneous rule, and can learn to provision *ahead* of demand to hide cold-start latency.

We pursue a deliberate **two-algorithm strategy** to make the comparison scientifically meaningful rather than anecdotal. **Proximal Policy Optimization (PPO)** is our primary policy-gradient method; **Deep Q-Network (DQN)** is the value-based comparison. Both are compatible with our discrete action space, which makes the comparison fair: any performance gap is attributable to the *learning paradigm* (on-policy actor-critic versus off-policy value iteration with replay), not to a difference in action representation.

The **research novelty** is a *Sparse Update* mechanism. Standard training performs a gradient update at every fixed training interval. We investigate whether updating network weights only once every $K$ training iterations ($K \in \{1, 4, 8\}$) reduces wall-clock and GPU compute *without* materially degrading the learned policy — the hypothesis being that cloud auto-scaling dynamics evolve slowly enough that per-iteration updates are wasteful. This is operationalized precisely in Section 7, including a correction to a common but non-functional implementation pattern.

**Grading philosophy (internalize this).** This project is graded on *process rigor and mathematical honesty*, not on whether the agent achieves perfect performance. A failed agent with a deep, correct diagnostic analysis scores higher than a black-box agent that happens to win. Therefore every design decision, every failure mode, and every dead end must be **documented and explained**. Negative results are first-class deliverables.

**Academic-integrity commitment.** Every AI-assisted action is logged in a shared tracking sheet with columns `Date | Task | Tool | Prompt | Result`. Per the course's stated rules, AI may assist with code, debugging, and clarity editing, but the report's theoretical framing, the choice of research idea, and the analysis of *our own* results must be authored by the team. **This constitution itself is an AI-assisted planning artifact and must be logged as such.** It is a scaffold for implementation, not a substitute for the team's own reasoning in the final report.

> **Critical:** Treat the deadline of **June 25, 2026, 23:59 EST** as immovable. There is no grace period. Section 12 builds in a one-day debugging buffer; do not spend it early.

\newpage

# 2 — Mathematical Formulation: The MDP

We model the system as a discrete-time Markov Decision Process $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$, where $\gamma$ is the discount factor (distinct from the reward weight $\gamma$ introduced in Section 2.3 — we disambiguate the two explicitly below to avoid a notation collision). One timestep represents one control interval of the simulated cluster.

## 2.1 State Space $\mathcal{S}$ (5-dimensional continuous vector)

The raw state at time $t$ is

$$S_t = \big[\,N^{\text{active}}_t,\; N^{\text{boot}}_t,\; U_t,\; Q_t,\; \lambda_t\,\big] \in \mathbb{R}^5.$$

| Dim | Symbol | Raw range | Physical meaning | Normalization |
|-----|--------|-----------|------------------|---------------|
| $s_1$ | `active_servers` | $[1, 10]$ | Servers running and accepting requests | $/\,N_{\max}$ |
| $s_2$ | `booting_servers` | $[0, 10]$ | Servers in cold-start (active after `boot_delay`) | $/\,N_{\max}$ |
| $s_3$ | `cpu_utilization` | $[0, 1]$ | Mean CPU load over active servers | already in $[0,1]$ |
| $s_4$ | `queue_length` | $[0, 500]$ | Backlogged requests awaiting processing | $/\,Q_{\max}$ |
| $s_5$ | `arrival_rate` $\lambda$ | $[0, \lambda_{\max}]$ | EMA of last 5 timesteps' Poisson arrivals | $/\,\lambda_{\max}$ |

CPU utilization is defined as

$$U_t = \min\!\Big(1.0,\; \frac{Q_t + A_t}{N^{\text{active}}_t \cdot c}\Big),$$

where $A_t$ is the current arrival count and $c = 50$ is per-server capacity (requests per timestep).

**Why each feature.** The pair $(N^{\text{active}}_t, N^{\text{boot}}_t)$ is essential and non-trivial: without $N^{\text{boot}}_t$ the process is *not* Markov, because the future capacity of the system depends on pending boots that are invisible in the active count. Exposing $N^{\text{boot}}_t$ removes that hidden state and lets the agent avoid double-ordering capacity that is already on its way. $U_t$ and $Q_t$ together describe present load (instantaneous pressure and accumulated backlog), while $\lambda_t$ describes *recent demand trend* and is what enables a proactive — rather than reactive — policy. We deliberately exclude any feature the simulator could not measure in a real deployment (e.g., future arrivals), to avoid information leakage that would make results un-reproducible on real infrastructure.

> **Critical — normalizing $\lambda$:** The proposal writes $\lambda \in [0, \infty)$, but an unbounded feature cannot be scaled to $[0,1]$. Fix a finite cap
$\lambda{\max} = \texttt{base\_rate\_max} \times \texttt{spike\_multiplier}$ (here $80 \times 3 = 240$) and clip before scaling. This is a deliberate modelling choice and must be reported.

**Normalization strategy.** All five dimensions are min–max scaled into $[0,1]$ inside the environment, and we additionally wrap the vectorized environment in Stable-Baselines3's `VecNormalize` (`norm_obs=True`, `clip_obs=5.0`, `gamma=0.99`). `VecNormalize` maintains a running mean/variance and standardizes observations to roughly zero-mean unit-variance, which stabilizes early training; the in-environment min–max scaling is a deterministic, deployment-portable fallback. The two are complementary, not redundant: min–max gives interpretable bounded features, `VecNormalize` adapts to the empirical distribution actually visited.

## 2.2 Action Space $\mathcal{A}$ — `Discrete(3)`

$$\mathcal{A} = \{\,0:\text{Scale OUT } (+1),\;\; 1:\text{HOLD } (0),\;\; 2:\text{Scale IN } (-1)\,\}.$$

Hard constraints enforced inside `step()`:

- The provisioned total must satisfy $N_{\min} \le N^{\text{active}} + N^{\text{boot}} + a_t \le N_{\max}$ with $[N_{\min}, N_{\max}] = [1, 10]$.
- A **scale-out** that would exceed $N_{\max}$, or a **scale-in** that would drop below $N_{\min}$, is **clamped to HOLD** ($a_t = 1$).
- **Anti-thrashing:** if the previous applied action and the current applied action are opposite non-zero actions ($+1$ then $-1$, or $-1$ then $+1$), a thrashing flag is raised and penalized in the reward (Section 2.3).

**Why `Discrete(3)`.** A finite action set is *required* by DQN, whose output layer emits one Q-value per action. Restricting the cardinality to three keeps the policy search space compact and — crucially — keeps PPO and DQN on identical footing, since PPO handles discrete actions natively. A continuous action ("set the cluster to $x$ servers") would (i) break the DQN comparison, (ii) make oscillation harder to regularize, and (iii) be unrealistic: real auto-scalers add or remove instances incrementally, and a $\pm 1$ step naturally damps aggressive oscillation. The literature supports this: incremental discrete scaling has been reported to improve SLA compliance and reduce resource consumption relative to default Kubernetes autoscaling.

## 2.3 Reward Function $R(s, a, s')$ — Full Specification

> **Critical correction (mathematical honesty).** The drafting notes contained a double-counting bug: the drop term had a $\times 50$ baked into $D(t)$ *and* a weight $\gamma=50$; the thrash term was valued at $5.0$ *and* weighted by $\delta=5$. We define each component as a **bare quantity** and apply the multiplier **once** through its weight. Document this fix in the report; it is exactly the kind of formulation rigor that is rewarded.

The reward is the negative weighted sum of four penalties:

$$R(t) = -\Big[\,\alpha\, C(t) + \beta\, L(t) + \gamma\, D(t) + \delta\, T(t)\,\Big],$$

with components

$$
\begin{aligned}
C(t) &= N^{\text{active}}_t && \text{(infrastructure cost, linear in server count)}\\
L(t) &= \Big(\tfrac{Q_t}{Q_{\max}}\Big)^2 && \text{(latency penalty, quadratic in normalized backlog)}\\
D(t) &= \max\!\big(0,\; (Q_t + A_t - P_t) - Q_{\max}\big) && \text{(count of dropped requests)}\\
T(t) &= \mathbb{1}[\text{action oscillated}] \in \{0,1\} && \text{(thrash indicator)}
\end{aligned}
$$

where $P_t$ is the number of requests processed this step. Default weights:

$$\alpha = 1.0,\quad \beta = 0.1,\quad \gamma = 50.0,\quad \delta = 5.0.$$

**Rationale for each weight.**

- $\alpha = 1.0$ anchors the scale: one running server costs one unit per timestep.
- $\beta = 0.1$ scales latency relative to cost. We normalize the queue by $Q_{\max}$ *before* squaring. The original draft used the raw $Q^2$, which for $Q_{\max}=500$ reaches $250{,}000$ — six orders of magnitude above the cost term and a near-certain source of reward-scale pathology. Normalizing keeps $L(t) \in [0,1]$ so $\beta$ is interpretable, while the quadratic still punishes large backlogs disproportionately more than small ones.
- $\gamma = 50.0$ makes a single dropped request as costly as running 50 extra server-timesteps. Dropping requests is an SLA violation and must dominate.
- $\delta = 5.0$ discourages oscillation, which consumes boot cost and churn without delivering steady capacity.

**Why this resists reward hacking.** The agent cannot trivially game the objective:

1. *Scale-everything-out* is bounded by $C(t)$, which grows linearly with the server count — there is no free capacity.
2. *Ignore the queue* is punished by the quadratic $L(t)$ and, beyond $Q_{\max}$, by the catastrophic $\gamma D(t)$ — the degenerate "drop all requests to keep cost at zero" strategy is precisely what the large $\gamma$ forecloses.
3. *Flip-flop to chase the load* is punished by $\delta T(t)$.

The remaining failure mode to watch is *over-provisioning to avoid all risk of drops*; the linear $C(t)$ is what keeps that in check, and the balance between $\gamma$ and $\alpha$ is the single most important quantity to sensitivity-test (Section 9).

## 2.4 Environment Dynamics

- **Episode length:** $1000$ timesteps; episodes are time-`truncated`, not `terminated`, unless a catastrophic-failure condition is added (see Section 3.4).
- **Boot delay:** a requested server enters $N^{\text{boot}}$; a per-server counter decrements each step and the server is promoted to $N^{\text{active}}$ after `boot_delay = 3` steps. This is the cold-start latency the proactive policy must learn to hide.
- **Traffic model:** arrivals are Poisson, $A_t \sim \text{Poisson}(\lambda_{\text{base}}(t))$, where $\lambda_{\text{base}}(t)$ follows a sinusoidal day/night cycle with superimposed random spikes (Section 4).
- **Service:** each active server processes up to $c = 50$ requests/timestep.
- **Queue:** FIFO with $Q_{\max} = 500$; overflow becomes dropped requests.

The transition $P(s' \mid s, a)$ is stochastic solely through the Poisson arrivals and the spike process; all other dynamics (boot promotion, queue update, capacity) are deterministic given the arrival draw.

\newpage

# 3 — Environment Implementation Specification (`cloud_env.py`)

This section specifies a runnable Gymnasium environment. Treat the pseudocode as authoritative for ordering; the exact numeric defaults are in the constructor.

## 3.1 Class definition and init parameters

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CloudScalingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 max_servers=10,        # N_max, upper provisioning bound
                 min_servers=1,         # N_min, >=1 keeps the cluster alive
                 server_capacity=50,    # c, requests/timestep/server
                 max_queue=500,         # Q_max, FIFO capacity; overflow -> drops
                 boot_delay=3,          # cold-start steps before a server is active
                 episode_length=1000,   # steps per episode (truncation horizon)
                 traffic_mode="stochastic",   # "stochastic" | "deterministic"
                 reward_weights=(1.0, 0.1, 50.0, 5.0),  # (alpha, beta, gamma, delta)
                 lambda_max=240.0,      # cap for normalizing arrival rate
                 seed=None):
        super().__init__()
        self.N_max, self.N_min = max_servers, min_servers
        self.c, self.Q_max = server_capacity, max_queue
        self.boot_delay, self.ep_len = boot_delay, episode_length
        self.traffic_mode = traffic_mode
        self.alpha, self.beta, self.gamma_w, self.delta = reward_weights
        self.lambda_max = lambda_max
        # observation = [active, booting, cpu_util, queue, arrival_rate], all in [0,1]
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0=out, 1=hold, 2=in
        self._rng = np.random.default_rng(seed)
        self.traffic = None  # injected/created in reset()
```

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `max_servers` | 10 | $\ge$`min_servers` | Hard upper bound $N_{\max}$ |
| `min_servers` | 1 | $\ge 1$ | Keeps $\ge 1$ live server |
| `server_capacity` | 50 | $>0$ | Throughput per server |
| `max_queue` | 500 | $>0$ | FIFO depth; overflow = drops |
| `boot_delay` | 3 | $\ge 0$ | Cold-start latency |
| `episode_length` | 1000 | $>0$ | Truncation horizon |
| `traffic_mode` | `stochastic` | enum | Validation vs. full eval |
| `reward_weights` | (1,0.1,50,5) | $\ge 0$ | $(\alpha,\beta,\gamma,\delta)$ |
| `lambda_max` | 240 | $>0$ | Arrival-rate normalizer |

## 3.2 Observation and action spaces

The observation space is `Box(low=0, high=1, shape=(5,), dtype=float32)` — bounded because every raw quantity is divided by its cap (or, for CPU, already bounded). The action space is `Discrete(3)`. Using `float32` (not `float64`) matches SB3's default network dtype and avoids silent casting overhead.

## 3.3 `reset()` — complete logic

```python
def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    if seed is not None:
        self._rng = np.random.default_rng(seed)
    self.t = 0
    self.active = 2            # start with 2 active servers
    self.boot_timers = []      # list of remaining boot counters
    self.queue = 0
    self.last_action = 1       # HOLD, so step-1 thrash check is well-defined
    self.traffic = PoissonTrafficGenerator(seed=int(self._rng.integers(1e9)),
                                           mode=self.traffic_mode)
    self.arrival_ema = self.traffic.peek_lambda(0)
    obs = self._build_obs(cpu_util=0.0)
    info = {"active": self.active, "queue": self.queue}
    return obs.astype(np.float32), info
```

## 3.4 `step(action)` — complete transition with all edge cases

The order of operations is load-bearing; deviating from it produces subtle bugs (e.g., counting a server as active before its boot completes). Follow exactly:

```python
def step(self, action):
    # --- 1. Validate action and apply hard constraints ---
    applied = action
    provisioned = self.active + len(self.boot_timers)
    if action == 0 and provisioned >= self.N_max:
        applied = 1                       # clamp illegal scale-out to HOLD
    if action == 2 and self.active <= self.N_min:
        applied = 1                       # clamp illegal scale-in to HOLD
    if applied == 0:
        self.boot_timers.append(self.boot_delay)   # new server starts booting
    elif applied == 2:
        self.active -= 1                  # immediate removal (no shutdown delay)

    # --- 2. Generate this step's traffic ---
    arrivals, current_lambda = self.traffic.generate(self.t)

    # --- 3. Compute processed requests (capacity-limited) ---
    capacity = self.active * self.c
    backlog = self.queue + arrivals
    processed = min(backlog, capacity)

    # --- 4. Update queue ---
    self.queue = max(0, backlog - processed)

    # --- 5. Detect dropped requests (overflow beyond Q_max) ---
    dropped = max(0, self.queue - self.Q_max)
    self.queue = min(self.queue, self.Q_max)

    # --- 6. Advance boot queue: decrement, promote ready servers ---
    still_booting = []
    for timer in self.boot_timers:
        timer -= 1
        if timer <= 0:
            self.active = min(self.N_max, self.active + 1)
        else:
            still_booting.append(timer)
    self.boot_timers = still_booting

    # --- 7-9. Reward components and total ---
    cpu_util = min(1.0, backlog / max(1, self.active * self.c))
    thrash = int((self.last_action == 0 and applied == 2) or
                 (self.last_action == 2 and applied == 0))
    C = self.active
    L = (self.queue / self.Q_max) ** 2
    D = dropped
    T = thrash
    reward = -(self.alpha * C + self.beta * L +
               self.gamma_w * D + self.delta * T)

    # --- 10-11. Bookkeeping ---
    self.last_action = applied
    self.arrival_ema = 0.8 * self.arrival_ema + 0.2 * current_lambda
    self.t += 1

    # --- 12. Observation ---
    obs = self._build_obs(cpu_util).astype(np.float32)

    # --- 13. Termination / truncation ---
    truncated = self.t >= self.ep_len
    terminated = False  # no catastrophic absorbing state by default
    info = {"dropped": dropped, "active": self.active, "queue": self.queue,
            "cpu_util": cpu_util, "lambda": current_lambda,
            "reward_components": {"C": C, "L": L, "D": D, "T": T}}

    # --- 14. Return ---
    return obs, float(reward), terminated, truncated, info
```

```python
def _build_obs(self, cpu_util):
    return np.array([
        self.active / self.N_max,
        len(self.boot_timers) / self.N_max,
        cpu_util,
        self.queue / self.Q_max,
        min(self.arrival_ema, self.lambda_max) / self.lambda_max,
    ], dtype=np.float32)
```

> **Critical edge cases:** (i) a scale-in removes an *active* server immediately but a scale-out only adds to the boot queue — this asymmetry is the cold-start cost the agent must reason about; (ii) `cpu_util` is computed from `backlog` (pre-service demand) so it can momentarily reflect overload even when the queue is later cleared; (iii) `terminated` stays `False` so Gymnasium/SB3 bootstrap the value at the horizon correctly — using `terminated=True` at the time limit would wrongly zero the bootstrap and bias the value function.

## 3.5 `render()` and `close()`

```python
def render(self):
    print(f"t={self.t:4d} active={self.active} boot={len(self.boot_timers)} "
          f"queue={self.queue:3d} lambda={self.arrival_ema:6.1f}")

def close(self):
    pass
```

## 3.6 Registration

```python
gym.register(id="CloudScaling-v1", entry_point="cloud_env:CloudScalingEnv")
```

Registering by string lets training scripts construct the env via `gym.make("CloudScaling-v1", **kwargs)` and keeps `make_env` factories clean for vectorization (Section 6.1).

\newpage

# 4 — Traffic Generator Specification (`traffic.py`)

## 4.1 Poisson traffic model

```python
import numpy as np

class PoissonTrafficGenerator:
    def __init__(self,
                 base_rate_min=10,       # quiet-hours mean arrivals
                 base_rate_max=80,       # peak-hours mean arrivals
                 spike_probability=0.05, # P(start a spike) per timestep
                 spike_multiplier=3.0,   # lambda multiplier during a spike
                 period_length=200,      # timesteps per simulated "day"
                 mode="stochastic",
                 seed=0):
        self.rmin, self.rmax = base_rate_min, base_rate_max
        self.p_spike, self.k_spike = spike_probability, spike_multiplier
        self.period = period_length
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self._spike_remaining = 0
```

## 4.2 Traffic patterns

**Base sinusoidal load.** The mean arrival rate follows a day/night cycle:

$$\lambda_{\text{base}}(t) = r_{\min} + (r_{\max} - r_{\min})\cdot\Big(0.5 + 0.5\sin\!\big(\tfrac{2\pi t}{\text{period}}\big)\Big).$$

This keeps $\lambda_{\text{base}}(t) \in [r_{\min}, r_{\max}] = [10, 80]$ and gives the agent a *learnable* periodic structure it can anticipate.

**Random spikes (Black-Friday simulation).** Independently at each step, with probability `spike_probability`, a spike begins and lasts a random $5$–$15$ timesteps; during a spike the base rate is multiplied by `spike_multiplier`:

$$\lambda(t) = \lambda_{\text{base}}(t) \cdot \big(1 + (k_{\text{spike}}-1)\,\mathbb{1}[\text{in spike}]\big).$$

Spikes are the *unpredictable* component that defeats reactive thresholds and rewards a policy that keeps a small safety margin of warm capacity.

**Poisson sampling.** The realized arrivals are

$$A_t \sim \text{Poisson}\big(\lambda(t)\big).$$

## 4.3 Why Poisson? (M/M/c justification)

Web-request arrivals are well modelled as a Poisson process: inter-arrival times are memoryless (exponential) and events are independent, the standard assumption underlying M/M/c queueing analysis and reflected in mainstream cloud auto-scaling guidance. The M/M/c lens also gives us a sanity check: with $c$ servers each of rate $\mu$ (here $\mu = 50$), the system is stable only while the offered load $\rho = \lambda / (c\mu) < 1$. At peak $\lambda \approx 80$ a single server ($c\mu = 50$) is already overloaded ($\rho = 1.6$), so the agent *must* scale out; at $r_{\min} = 10$ a single server suffices ($\rho = 0.2$), so holding extra servers is pure waste. This analytic boundary is what the learned policy should approximately recover, and comparing the policy's server count against $\lceil \lambda / (c\,\rho_{\text{target}}) \rceil$ is a useful diagnostic.

## 4.4 `generate(t)` method

```python
def peek_lambda(self, t):
    return self.rmin + (self.rmax - self.rmin) * (0.5 + 0.5*np.sin(2*np.pi*t/self.period))

def generate(self, t):
    lam = self.peek_lambda(t)
    if self.mode == "stochastic":
        if self._spike_remaining > 0:
            lam *= self.k_spike
            self._spike_remaining -= 1
        elif self.rng.random() < self.p_spike:
            self._spike_remaining = int(self.rng.integers(5, 16))
            lam *= self.k_spike
    arrivals = int(self.rng.poisson(lam)) if self.mode == "stochastic" else int(round(lam))
    return arrivals, float(lam)
```

`deterministic` mode disables spikes and Poisson noise (returns the rounded mean), which is the MVP traffic used to validate that the RL pipeline learns *anything* before stochasticity is introduced.

\newpage

# 5 — Baseline Agent Specification (`baseline_agent.py`)

## 5.1 Rule-based heuristic agent

The baseline is the incumbent we must beat — a threshold scaler representative of production auto-scalers. It consumes the **normalized** observation and denormalizes internally.

```python
class RuleBasedBaseline:
    def __init__(self, n_max=10, q_max=500):
        self.n_max, self.q_max = n_max, q_max

    def predict(self, obs, deterministic=True):
        active = obs[0] * self.n_max
        booting = obs[1] * self.n_max
        cpu_util = obs[2]
        queue = obs[3] * self.q_max
        # arrival_rate = obs[4] * lambda_max   # available, unused by the heuristic

        if cpu_util > 0.80 and queue > 100:        # Rule 1: urgent scale-out
            action = 0
        elif cpu_util > 0.65 and queue > 50:       # Rule 2: proactive scale-out
            action = 0
        elif cpu_util < 0.30 and queue == 0 and booting == 0:  # Rule 3: scale-in
            action = 2
        else:                                       # Rule 4: hold
            action = 1
        return action, None   # mirror SB3's (action, state) return signature
```

The heuristic is intentionally *reasonable* — it has urgent and proactive tiers and a conservative scale-in guard — so that beating it is a meaningful result and not a straw man. Its blind spot is structural: it never uses $\lambda_t$, so it cannot anticipate the recurring peak or pre-warm capacity ahead of a spike.

## 5.2 Evaluating the baseline

Run the baseline for **10 independent episodes** of 1000 timesteps each (fixed seeds for reproducibility). Record per episode and report mean $\pm$ std:

- **Mean total reward** — this becomes the *target threshold* the RL agents must exceed.
- **Operational cost** — total server-timesteps, $\sum_t N^{\text{active}}_t$.
- **Dropped requests** — $\sum_t D(t)$ (the SLA-violation count).
- **Mean queue length** — average backlog over the episode.

The success criterion is that an RL agent's mean total reward exceeds the baseline's. If it does not, Section 10.3 governs how to diagnose and report the shortfall — *do not* discard the result.

\newpage

# 6 — RL Agents Training Specification

## 6.1 Vectorization setup

```python
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

def make_env(rank, seed=0):
    def _init():
        env = CloudScalingEnv()
        env.reset(seed=seed + rank)
        return env
    return _init

N_ENVS = 8
train_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                         clip_obs=5.0, gamma=0.99)
```

**`SubprocVecEnv` vs `DummyVecEnv`.** `DummyVecEnv` runs all environments sequentially in the main process; `SubprocVecEnv` runs each in its own process, so rollout collection proceeds in true parallel on multiple CPU cores. For a CPU-cheap environment like ours the inter-process communication overhead can rival the step cost, so for small `N_ENVS` `DummyVecEnv` is sometimes faster — benchmark both. We use `SubprocVecEnv` for PPO because PPO collects long rollouts where the parallelism pays off.

> **Honest framing of the "8 envs for the GPU" claim:** the number of parallel environments is primarily a *CPU rollout-throughput* knob, not a GPU one. The indirect GPU benefit is that 8 environments produce a batch of 8 observations per inference call, so the forward pass during rollout is better-batched and the policy network's GPU utilization is higher than with a single env. Eight is a sound default that matches typical core counts; do not claim it is tuned to a specific GPU SKU unless you measure it. Report measured GPU utilization (Section 7.4) rather than asserting it.

## 6.2 PPO agent — full hyperparameters

```python
from stable_baselines3 import PPO

model = PPO(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=3e-4,    # Adam LR; lower if value loss is unstable
    n_steps=2048,          # rollout length per env; 2048 * 8 = 16384 samples/update
    batch_size=256,        # minibatch for SGD
    n_epochs=10,           # optimization epochs per rollout
    gamma=0.99,            # discount factor
    gae_lambda=0.95,       # GAE bias/variance trade-off
    clip_range=0.2,        # PPO trust-region clip
    ent_coef=0.01,         # entropy bonus (exploration)
    vf_coef=0.5,           # value-loss weight
    max_grad_norm=0.5,     # gradient clipping
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    tensorboard_log="./logs/ppo/",
    verbose=1,
)
```

**Justification.**

- $\gamma = 0.99$ gives an effective horizon of $\sim\!1/(1-\gamma) = 100$ steps — long enough to value the multi-step payoff of pre-warming capacity, short enough to remain well-conditioned over a 1000-step episode.
- `gae_lambda = 0.95` is the conventional bias–variance sweet spot for GAE; it trades a little bias for substantially lower advantage variance.
- `clip_range = 0.2` is the canonical PPO clip that bounds the policy update to a trust region and is the dominant source of PPO's training stability.
- `ent_coef = 0.01` sustains exploration; if entropy collapses to zero early (Section 10.3) raise it.
- `n_steps = 2048` with 8 envs yields 16,384 transitions per update — large enough for low-variance gradient estimates.
- `net_arch = [256, 256]` separate actor/critic trunks give ample capacity for a 5-D input without overfitting.

## 6.3 DQN agent — full hyperparameters

```python
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

dqn_env = DummyVecEnv([make_env(0)])
dqn_env = VecNormalize(dqn_env, norm_obs=True, norm_reward=True,
                       clip_obs=5.0, gamma=0.99)

model = DQN(
    policy="MlpPolicy",
    env=dqn_env,
    learning_rate=1e-4,
    buffer_size=100_000,        # replay capacity
    learning_starts=10_000,     # warmup of random transitions before learning
    batch_size=256,
    tau=1.0,                    # hard target update
    gamma=0.99,
    train_freq=4,               # gradient step every 4 env steps
    gradient_steps=1,
    target_update_interval=1000,
    exploration_fraction=0.1,   # fraction of training spent annealing epsilon
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[256, 256]),
    tensorboard_log="./logs/dqn/",
    verbose=1,
)
```

> **Critical:** DQN in SB3 should be run on a **single** environment (`DummyVecEnv([make_env(0)])`); the `SubprocVecEnv` machinery exists for on-policy methods like PPO/A2C and buys DQN little, since DQN learns from a replay buffer one minibatch at a time. Keep `VecNormalize` so the observation scaling matches PPO.

**Key architectural difference.** PPO is an **on-policy actor–critic**: it maintains an explicit stochastic policy network $\pi_\theta$ and a separate value network $V_\phi$, and updates from freshly collected on-policy rollouts inside a clipped trust region. DQN is an **off-policy value method**: a single Q-network estimates $Q(s,a)$, actions are $\arg\max_a Q$, and learning reuses past transitions from a replay buffer with a slowly-updated target network. PPO is *theoretically better suited* to our setting because the reward distribution is **non-stationary** (traffic shifts within and across episodes): on-policy updates always reflect the current data distribution, whereas DQN's replay buffer mixes transitions gathered under stale conditions, and its $\max$-operator bootstrapping is prone to value overestimation when the target is a moving one. We *expect* PPO to be more stable; whether it is more sample-efficient is exactly what the experiment will tell us.

## 6.4 Training script (`train_ppo.py` / `train_dqn.py`)

```python
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import sync_envs_normalization

eval_env = VecNormalize(DummyVecEnv([make_env(100)]),
                        norm_obs=True, norm_reward=False,  # NEVER norm reward at eval
                        clip_obs=5.0, gamma=0.99)
eval_env.training = False                                  # freeze running stats

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/best_ppo/",
    log_path="./logs/ppo_eval/",
    eval_freq=10_000,
    n_eval_episodes=5,
    deterministic=True,
)

model.learn(
    total_timesteps=2_000_000,             # minimum 1e6, preferred 2e6
    callback=[eval_callback,
              CheckpointCallback(save_freq=100_000, save_path="./checkpoints/")],
    reset_num_timesteps=False,
)

model.save("./models/final_ppo")
train_env.save("./models/vecnormalize_ppo.pkl")   # CRITICAL for deployment/eval
```

> **Critical:** the `VecNormalize` statistics are part of the model. If you save the policy but not `vecnormalize_*.pkl`, the agent will see observations on the wrong scale at evaluation time and appear to have "forgotten" everything. Always save *and* reload both. The eval env must (a) set `norm_reward=False` so reported returns are real, (b) set `training=False` so it does not update its own stats, and (c) be kept in sync with the training stats via `sync_envs_normalization(train_env, eval_env)` when not loading from disk.

\newpage

# 7 — Research Novelty: Sparse Updates

## 7.1 Motivation and research question

Standard training performs a gradient update at every fixed training interval. The hypothesis: because cloud auto-scaling dynamics evolve slowly (on minute-level timescales relative to the control step), the policy may not need to be refreshed every interval. Updating only once per $K$ intervals should reduce optimizer compute by roughly a factor $(K-1)/K$, and — *if the hypothesis holds* — should not materially degrade the final policy.

**Research question.** *Does a sparse-update schedule ($K = 4$ or $K = 8$) significantly degrade final policy performance in cloud auto-scaling, and what is the resulting performance-versus-compute trade-off?*

## 7.2 Correct SB3 implementation

> **Critical — the naive callback does not work.** A common pattern is a `BaseCallback` that, in `_on_rollout_end`, calls `self.model.policy.set_training_mode(False)` to "skip" the update. This is incorrect: `set_training_mode` only toggles `train()`/`eval()` mode on the modules (affecting dropout/batch-norm); it does **not** prevent SB3 from running the optimizer. The gradient step still executes. There is also no public callback hook positioned to *cancel* the upcoming `train()` call. The robust solution is to **subclass the algorithm and override `train()`** so the gradient update itself is conditionally skipped. For PPO (on-policy), `train()` is invoked exactly once per collected rollout, so a rollout counter gives clean control:

```python
from stable_baselines3 import PPO

class SparsePPO(PPO):
    """PPO that performs a gradient update only every K-th rollout.
    On skipped iterations the freshly collected on-policy rollout is
    discarded (it cannot be reused next iteration because it is on-policy).
    This trades sample usage for optimizer compute."""
    def __init__(self, *args, update_every_k=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_every_k = int(update_every_k)
        self._rollout_idx = 0

    def train(self) -> None:
        self._rollout_idx += 1
        if self.update_every_k > 1 and (self._rollout_idx % self.update_every_k != 0):
            # log that we skipped, then return WITHOUT touching the optimizer
            self.logger.record("sparse/skipped_update", 1)
            return
        self.logger.record("sparse/skipped_update", 0)
        super().train()   # the real gradient update
```

For **DQN**, sparse updates are native: the update cadence is controlled directly by `train_freq` (and `gradient_steps`). To realize $K\times$ sparser updates, set `train_freq = 4*K` (relative to the baseline `train_freq=4`). No subclass is needed; document that the DQN and PPO sparsity knobs are mechanically different and report each on its own axis.

**Why this is the honest operationalization.** In SB3, neither PPO nor DQN updates "every environment step" out of the box — PPO updates once per `n_steps` rollout and DQN once per `train_freq` steps. "Every $K$ steps" is therefore ambiguous; we define $K$ as *the multiplier on the existing update interval*, which is unambiguous, measurable, and directly controls optimizer compute. State this definition explicitly in the report.

## 7.3 Experimental design

Run the matrix below; **3 random seeds** per $K$ value gives 9 runs total:

- $K \in \{1, 4, 8\}$ (where $K=1$ is the standard control).
- 3 seeds each $\Rightarrow$ report mean $\pm$ std.
- Measure: final mean eval reward, training wall-clock time, peak GPU memory, and mean GPU utilization.
- **Statistical test:** a Mann–Whitney U test (non-parametric, no normality assumption, appropriate for $n=3$ as a directional indicator rather than a strong claim) comparing the $K=4$ and $K=8$ reward distributions against $K=1$.
- **Report the trade-off even if sparse is worse:** quantify *by how much* reward drops and *how much* compute is saved. The deliverable is the trade-off curve, not a binary verdict. With only 3 seeds, frame significance cautiously and treat effect size as primary.

## 7.4 GPU/compute timing measurement

```python
import time, torch, subprocess, threading

def sample_gpu_util(stop_evt, out):
    while not stop_evt.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2)
            out.append(int(r.stdout.strip().splitlines()[0]))
        except Exception:
            pass
        time.sleep(1.0)

stop_evt, util = threading.Event(), []
t = threading.Thread(target=sample_gpu_util, args=(stop_evt, util)); t.start()

start = time.perf_counter()
model.learn(total_timesteps=500_000)         # shorter budget for the timing study
wall = time.perf_counter() - start

stop_evt.set(); t.join()
print(f"K={model.update_every_k}  wall={wall:.1f}s  "
      f"peak_mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB  "
      f"mean_gpu_util={sum(util)/max(1,len(util)):.0f}%")
```

Use a fixed, shorter budget (e.g., 500k steps) for the timing comparison so the wall-clock numbers are directly comparable across $K$. Report both wall-clock and the theoretical $(K-1)/K$ savings, and explain any gap (rollout collection time is unaffected by $K$, so the realized speedup is always *less* than $(K-1)/K$ — a point worth analyzing).

\newpage

# 8 — Evaluation & Visualization Pipeline

## 8.1 `eval_agent.py` — deterministic evaluation

```python
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def evaluate_agent(model, vecnorm_path=None, n_episodes=10, seed=42):
    """Deterministic policy evaluation. Returns mean/std of total_reward,
    operational_cost, dropped_requests, queue_occupancy_rate."""
    env = DummyVecEnv([make_env(seed)])
    if vecnorm_path:                       # load TRAINING normalization stats
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False               # do not update stats
        env.norm_reward = False            # report real rewards
    rewards, costs, drops, qocc = [], [], [], []
    for ep in range(n_episodes):
        obs = env.reset(); done = [False]
        R = c = d = q = 0; steps = 0
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, info = env.step(action)
            R += r[0]
            c += info[0]["active"]
            d += info[0]["dropped"]
            q += info[0]["queue"]; steps += 1
        rewards.append(R); costs.append(c); drops.append(d)
        qocc.append(q / (steps * 500))     # mean fractional queue occupancy
    agg = lambda x: (float(np.mean(x)), float(np.std(x)))
    return {"reward": agg(rewards), "cost": agg(costs),
            "dropped": agg(drops), "queue_occ": agg(qocc)}
```

Evaluate four policies on identical seeds: **PPO** (best model), **DQN** (best model), **Baseline** (rule-based), and **Random** (uniform over the three actions, as a lower bound). The Random lower bound contextualizes how much of the baseline's performance is "free" structure versus genuine control.

## 8.2 Required plots (publication quality)

Generate all five with `matplotlib`/`seaborn`, `figsize=(12,6)`, `dpi=150`, and save to `results/plots/` as PNG.

**Plot 1 — Learning curves.** PPO vs DQN vs Baseline. X: environment steps (0–2M). Y: mean episode reward, smoothed with a Gaussian kernel ($\sigma=10$ over the logged points). Overlay a horizontal dashed line at the baseline mean. Plot 3 seeds per agent with a shaded $\pm 1$ std band. This is the headline figure.

**Plot 2 — Sparse-updates trade-off.** X: $K \in \{1,4,8\}$. Left Y (blue line): final mean reward. Right Y (orange bars): training wall-clock seconds. This is the novelty figure; annotate the percentage reward change and the percentage time saved between $K=1$ and $K=4$.

**Plot 3 — Operational-cost breakdown.** Grouped/stacked bars for PPO, DQN, Baseline, Random, decomposing total penalty into its three substantive components: infrastructure cost ($\alpha C$), latency ($\beta L$), and drops ($\gamma D$). Shows *where* each policy spends versus wastes.

**Plot 4 — Policy behavior trace.** A single representative episode (0–1000). Four stacked subplots: (a) arrival rate $\lambda(t)$, (b) active servers, (c) queue length, (d) instantaneous reward. Three lines per subplot — PPO (blue), DQN (orange), Baseline (gray). This is where a reader *sees* proactive versus reactive behavior.

**Plot 5 — Convergence box plots.** Over the final 100k steps, box plots of the reward distribution across the 3 seeds for each agent — a compact view of convergence stability and seed sensitivity.

## 8.3 TensorBoard metrics

Log every 1000 steps via a small custom callback that calls `self.logger.record(key, value)`:

- `train/reward_mean`, `train/ep_len_mean`
- PPO: `train/value_loss`, `train/policy_gradient_loss`, `train/entropy_loss`
- DQN: `train/exploration_rate`
- Custom: `custom/dropped_requests_per_episode`, `custom/active_servers_mean`, `custom/queue_length_mean`

```python
from stable_baselines3.common.callbacks import BaseCallback

class MetricsCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos and self.num_timesteps % 1000 == 0:
            drop = np.mean([i.get("dropped", 0) for i in infos])
            act  = np.mean([i.get("active", 0) for i in infos])
            q    = np.mean([i.get("queue", 0) for i in infos])
            self.logger.record("custom/dropped_requests_per_episode", drop)
            self.logger.record("custom/active_servers_mean", act)
            self.logger.record("custom/queue_length_mean", q)
            self.logger.dump(self.num_timesteps)   # REQUIRED to flush custom keys
        return True
```

> **Critical:** custom scalars only appear in TensorBoard after `self.logger.dump(step)` — `record()` alone buffers them.

\newpage

# 9 — Hyperparameter Tuning (Optuna Sweep)

## 9.1 Optuna integration

```python
import optuna
from stable_baselines3 import PPO

def objective(trial):
    lr       = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    n_steps  = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.05)
    # reward weights: alpha fixed at 1.0 (the scale anchor); tune the rest.
    # NOTE: w_drop below is the reward weight gamma from Section 2.3 --
    # it is NOT the discount factor gamma, which stays fixed at 0.99.
    beta    = trial.suggest_float("beta",    0.05, 0.5)   # latency weight
    w_drop  = trial.suggest_float("w_drop",  10.0, 100.0) # drop-penalty weight
    delta   = trial.suggest_float("delta",   1.0, 10.0)   # thrash weight

    env = make_vecnormalized_env(reward_weights=(1.0, beta, w_drop, delta))
    model = PPO("MlpPolicy", env, learning_rate=lr, n_steps=n_steps,
                ent_coef=ent_coef, gamma=0.99, verbose=0)
    model.learn(total_timesteps=200_000)          # short trial budget
    mean_reward = quick_eval(model, n_episodes=5) # eval with norm_reward=False
    return mean_reward

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print("Best:", study.best_params, study.best_value)
```

> **Critical naming:** the reward-component weight in Section 2.3 is written $\gamma$, which collides with the MDP discount factor $\gamma=0.99$. In code, name the reward weight `w_drop` (or `gamma_w`) and keep the discount as `gamma`. Tuning the reward weights inside the objective means you are partly co-designing the objective and the policy; be transparent about this in the report — the "best" reward weights are a modelling decision, while `lr`/`n_steps`/`ent_coef` are pure optimization choices.

Each trial trains for only 200k steps (a quick proxy); take the best configuration and run the full 2M-step training for the final reported numbers. Optionally add Optuna's `MedianPruner` to terminate hopeless trials early and stretch the trial budget.

\newpage

# 10 — Report Writing Guide (IEEE Format)

The professor's explicit instruction: **do not** explain what PPO or DQN *are*. Spend the page budget on **your** MDP, **your** architecture, **your** problems, and **your** results.

## 10.1 Report structure

- **Abstract (about 150 words):** problem, approach, single headline result (e.g., "PPO reduced operational cost by X% over the threshold baseline at Y% SLA compliance").
- **I. Introduction:** the auto-scaling problem, why threshold heuristics fail, why RL, and a one-sentence statement of contributions (the MDP design, the PPO/DQN comparison, the sparse-update study).
- **II. Problem Formulation:** Section 2 rewritten as academic prose — state, action, reward, dynamics, with the M/M/c justification.
- **III. Methodology:** environment and traffic design, the baseline, network architectures, the sparse-update mechanism.
- **IV. Experiments:** training setup, hyperparameters (in a table), seeds, compute.
- **V. Results:** the five plots with captions and quantitative analysis.
- **VI. Discussion:** the *why* — this section carries the grade.
- **VII. Conclusion:** limitations and future work.
- **Appendix A:** GenAI disclosure (the tracking sheet).

## 10.2 Discussion section guide (most important for grading)

For each agent, answer concretely and with evidence from your plots:

1. Did it beat the baseline, and by how much (reward, cost, drops separately)?
2. Which training phase showed the most learning — early or late? (Read it off Plot 1's slope.)
3. Did it **over-scale** (waste cost — visible as high $\alpha C$ in Plot 3) or **under-scale** (drop requests — high $\gamma D$)?
4. What does the entropy curve (PPO) say about the exploration/exploitation balance over time?
5. Did sparse updates ($K=4$) degrade performance, and was the compute saving worth it? Quote Plot 2's numbers.

## 10.3 How to handle negative results

If PPO underperforms the baseline, **do not fabricate** — diagnose, and the diagnosis *is* the contribution:

- Plot `value_loss`: divergence suggests the value target is too non-stationary (consider reward normalization or a shorter horizon).
- Plot policy entropy: an early collapse to zero means premature exploitation — raise `ent_coef`.
- Inspect the Plot 4 trace: visible $+1/-1$ oscillation means thrashing — raise $\delta$.
- Re-examine reward scale: if returns collapse toward $-\infty$, the quadratic latency or the drop penalty likely dominates — revisit the component normalization from Section 2.3 and check `VecNormalize` clipping.

A well-diagnosed failure, with the corrective experiments you ran and what they showed, is explicitly valued: the best grades have gone to projects that failed *for understood reasons*.

\newpage

# 11 — Academic Integrity & AI Disclosure

## 11.1 What AI may assist with

Writing and debugging Python; explaining error messages; suggesting library functions; grammar and clarity editing.

## 11.2 What AI must not do

Author the report's theoretical sections (Introduction, related-work framing); generate the novel research idea (the sparse-update study and MDP design are the team's own, as evidenced by the original proposal); fabricate or "clean up" experimental results.

## 11.3 Tracking-sheet format

Maintain a shared sheet and submit it as Appendix A:

| Date | Task Description | Tool Used | Exact Prompt | Result / Solution |
|------|------------------|-----------|--------------|-------------------|
| ...  | ...              | ...       | ...          | ...               |

> **Critical — log this document.** This constitution was produced with AI assistance as a *planning and code-scaffolding* artifact. Record it in the tracking sheet (date, the generating prompt, and that the output is an implementation plan). The report's prose, the analysis of your real results, and the final interpretation must be written by the team. Keeping this entry honest protects the whole submission.

\newpage

# 12 — Execution Timeline (June 17–25, 2026)

Critical-path logic: training gates evaluation, which gates the results plots, which gate the Results/Discussion sections. Traffic + environment + baseline can finish day one; PPO and DQN training run in parallel; the Optuna sweep runs alongside long training. Reserve June 24 as the debugging buffer — assume something breaks.

- [ ] **June 17 (today):** finalize `traffic.py`; integrate into `cloud_env.py`; smoke-test with `check_env`; run the rule-based baseline (10 episodes) and record target metrics.
- [ ] **June 18:** launch PPO 2M-step run (roughly 8–16h); launch DQN run in parallel; implement and unit-test `SparsePPO`.
- [ ] **June 19:** training in progress; write `eval_agent.py` and `plot_results.py`; dry-run plotting on partial checkpoints.
- [ ] **June 20:** run the Optuna sweep (50 × 200k); begin report Sections I–II.
- [ ] **June 21:** training complete; full evaluation of all four policies; generate all five plots; write Sections III–IV.
- [ ] **June 22:** write Sections V–VI (Results + Discussion) — **the most important day.**
- [ ] **June 23:** write Abstract, Conclusion, Appendix A; full team review.
- [ ] **June 24:** debugging buffer + final polish; compile to IEEE PDF; verify figures, captions, references.
- [ ] **June 25:** submit by 18:00 EST as a buffer against the 23:59 hard deadline.

**Parallelizable:** traffic/env vs. plotting code; PPO vs. DQN training; Optuna vs. long training. **Critical path:** training → evaluation → results plots → Results/Discussion. **Partial-result plots:** Plots 1 and 5 can be drafted from checkpoints; Plots 2–4 need final models.

\newpage

# 13 — File Structure & Deliverables

```
cloud_rl_project/
├── cloud_env.py              # Gymnasium environment (Section 3)
├── traffic.py                # Poisson traffic generator (Section 4)
├── baseline_agent.py         # Rule-based heuristic (Section 5)
├── train_ppo.py              # PPO training script (Section 6)
├── train_dqn.py              # DQN training script (Section 6)
├── sparse_ppo.py             # SparsePPO subclass (Section 7)
├── eval_agent.py             # Evaluation & comparison (Section 8)
├── plot_results.py           # All five publication plots (Section 8)
├── optuna_sweep.py           # Hyperparameter tuning (Section 9)
├── models/
│   ├── best_ppo/             # best PPO checkpoint (EvalCallback)
│   ├── best_dqn/             # best DQN checkpoint
│   └── vecnormalize_*.pkl    # normalization stats (MUST save)
├── logs/
│   ├── ppo/                  # TensorBoard logs
│   └── dqn/
├── results/
│   ├── plots/                # five figures as .png
│   └── metrics.json          # all numerical results
└── report/
    └── CISC856_Team10_FinalReport.pdf
```

\newpage

# 14 — Common Pitfalls & Debugging Guide

1. **`VecNormalize` not saved.** The model loads but rewards/observations are on the wrong scale and the agent appears amnesiac. *Fix:* always `env.save("vecnormalize_*.pkl")` after training and `VecNormalize.load(path, env)` before eval, with `training=False` and `norm_reward=False`.

2. **Eval normalization out of sync.** With `EvalCallback`, the eval `VecNormalize` keeps its own stats; if they drift from training, eval rewards are misleading. *Fix:* call `sync_envs_normalization(train_env, eval_env)`, set `eval_env.training=False`, and never normalize reward at eval.

3. **DQN with the wrong vec env.** DQN is single-environment in practice; wrapping it in `SubprocVecEnv` adds overhead without benefit. *Fix:* use `DummyVecEnv([make_env(0)])` (still `VecNormalize`-wrapped).

4. **Reward divergence to $-\infty$.** Usually the drop penalty or the (un-normalized) quadratic latency dominates. *Fix:* normalize the queue before squaring (Section 2.3), verify `clip_obs`, and lower the learning rate if value loss is unstable.

5. **Sparse updates silently doing nothing.** The `set_training_mode(False)` callback pattern does **not** skip the optimizer step. *Fix:* use the `SparsePPO` subclass that overrides `train()` (Section 7.2); for DQN, change `train_freq`. Do not mutate internal update counters by hand.

6. **Custom TensorBoard metrics missing.** `logger.record()` buffers; the scalars never appear. *Fix:* call `self.logger.dump(self.num_timesteps)` after recording.

7. **Thrashing agent.** Persistent $+1/-1$ oscillation. *Fix:* raise $\delta$, confirm the anti-thrash indicator is wired into the reward, and raise `ent_coef` if the policy has collapsed onto a degenerate alternation.

8. **Cold-start blindness.** The agent ignores `boot_delay` and under-provisions before spikes. *Fix:* confirm `booting_servers` ($s_2$) is in the observation and non-zero during boots; the proactive behavior must emerge from valuing future capacity, so verify $\gamma=0.99$ (the discount) and that the value function is actually learning (Plot 1 slope, falling `value_loss`).

9. **Time-limit treated as termination.** Returning `terminated=True` at the 1000-step horizon zeroes the value bootstrap and biases learning. *Fix:* return `truncated=True`, `terminated=False` at the horizon (Section 3.4).

10. **`check_env` failures.** Run `from stable_baselines3.common.env_checker import check_env; check_env(CloudScalingEnv())` before any training — it catches dtype, bound, and return-signature bugs in minutes that would otherwise cost hours mid-run.

---

*End of Project Constitution. This document is a planning and implementation scaffold; the final report's analysis and prose are the team's own work and must be logged per Section 11.*
