import numpy as np
import torch as th
from numba import njit
from stable_baselines3.common.buffers import NStepReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class PrioritizedReplayBufferSamples(ReplayBufferSamples):
    """
    Extends ReplayBufferSamples (NamedTuple) with PER-specific fields.

    NamedTuple itself has __slots__ = () — no __dict__.
    A subclass without __slots__ gets __dict__ automatically, so we can
    set extra attributes freely in __new__.
    (Nonempty __slots__ on a NamedTuple subtype is explicitly forbidden.)
    """

    def __new__(cls, *args, weights=None, indices=None, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        instance.weights = weights   # IS correction weights, shape (B, 1)
        instance.indices = indices   # buffer indices, numpy — for _update_priorities
        return instance


class SumTree:
    """
    Binary tree where each leaf stores a priority p_i and each internal node
    stores the sum of its children.  Root = total priority.

    Capacity is rounded up to the next power of 2 to guarantee a complete
    binary tree — required for the vectorized traversal in get().

    Layout (capacity=4):
        index:  0          <- root (sum of all)
               1    2
              3  4  5  6   <- leaves (buffer indices 0..3)

    Leaf for buffer index i lives at tree position i + capacity - 1.
    """

    def __init__(self, capacity: int):
        # Next power of 2 -> complete binary tree, no out-of-bounds in traversal
        self.capacity = 1 << int(np.ceil(np.log2(max(capacity, 2))))
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)

    def update(self, idx: int, priority: float) -> None:
        """Update leaf at buffer index idx and propagate delta to root."""
        tree_idx = idx + self.capacity - 1
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += delta

    def get(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Vectorized traversal: for each value in [0, total], walk the tree
        left if value <= left-child sum, else subtract left sum and go right.
        Returns (buffer_indices, priorities) both shape (B,).
        """
        idx   = np.zeros(len(values), dtype=np.int64)
        depth = int(np.log2(self.capacity))   # exact depth for power-of-2 capacity
        for _ in range(depth):
            left     = 2 * idx + 1
            go_right = values > self.tree[left]
            values  -= self.tree[left] * go_right
            idx      = np.where(go_right, left + 1, left)
        data_idx = idx - (self.capacity - 1)
        return data_idx, self.tree[idx]

    @property
    def total(self) -> float:
        return float(self.tree[0])


class PrioritizedReplayBuffer(NStepReplayBuffer):
    """
    N-step replay buffer with Prioritized Experience Replay (PER).

    Transitions are sampled proportionally to |TD error|^alpha instead of
    uniformly.  Importance-sampling weights w_i correct for the resulting bias.

    Flow per training step:
      1. samples = buffer.sample(B)          # stratified PER sampling
      2. loss = (samples.weights * f(samples)).mean()
      3. buffer._update_priorities(samples.indices, td_errors)
    """

    def __init__(
        self,
        *args,
        alpha: float = 0.5,               # priority exponent  (0 = uniform, 1 = full PER)
        beta: float = 0.4,                # initial IS correction exponent (annealed -> 1)
        total_samples: int = 200_000_000, # total env steps — determines annealing rate
        epsilon: float = 0.01,            # floor added to |TD error| before exponentiation
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta_0 = beta
        self.beta = beta
        self.beta_annealing = (1.0 - beta) / total_samples
        self.epsilon = epsilon
        self.max_priority = 1.0
        self.sum_tree = SumTree(self.buffer_size)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def add(self, obs, next_obs, action, reward, done, infos) -> None:
        # Capture write position BEFORE super() increments self.pos
        idx = self.pos
        super().add(obs, next_obs, action, reward, done, infos)
        # New transitions get max priority so they are sampled at least once
        self.sum_tree.update(idx, self.max_priority ** self.alpha)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, batch_size: int, env=None) -> PrioritizedReplayBufferSamples:
        """
        Stratified PER sampling: divide [0, total] into B equal segments,
        draw one uniform sample per segment.  Stratification reduces variance
        compared to i.i.d. draws.
        """
        total   = self.sum_tree.total
        segment = total / batch_size

        offsets = np.random.uniform(0.0, segment, size=batch_size)
        values  = segment * np.arange(batch_size, dtype=np.float64) + offsets
        values  = np.clip(values, 0.0, total - 1e-8)

        batch_inds, _ = self.sum_tree.get(values)
        batch_inds    = np.clip(batch_inds, 0, self.size() - 1)  # safety for edge cases

        self.beta = min(1.0, self.beta + self.beta_annealing * batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> PrioritizedReplayBufferSamples:
        # n-step returns from parent
        base = super()._get_samples(batch_inds, env=env)

        # Look up current priorities directly from tree leaves
        tree_indices = batch_inds + self.sum_tree.capacity - 1
        priorities   = self.sum_tree.tree[tree_indices].astype(np.float32)

        # IS weights: w_i = (N * P(i))^{-beta},  normalised so max weight = 1
        N               = self.size()
        sampling_probs  = priorities / self.sum_tree.total
        weights         = (N * sampling_probs) ** (-self.beta)
        weights        /= weights.max()

        return PrioritizedReplayBufferSamples(
            observations      = base.observations,
            actions           = base.actions,
            next_observations = base.next_observations,
            dones             = base.dones,
            rewards           = base.rewards,
            discounts         = base.discounts,
            weights           = th.tensor(weights.reshape(-1, 1), dtype=th.float32,
                                          device=base.observations.device),
            indices           = batch_inds,
        )

    # ------------------------------------------------------------------
    # Priority update (called by training loop after TD-error computation)
    # ------------------------------------------------------------------

    def _update_priorities(self, batch_inds: np.ndarray, td_errors: np.ndarray) -> None:
        """
        td_errors: numpy array of shape (B,) or (B, 1), raw signed errors.
        """
        td_errors  = np.abs(td_errors).flatten()
        priorities = (td_errors + self.epsilon) ** self.alpha
        for idx, p in zip(batch_inds, priorities):
            self.sum_tree.update(int(idx), float(p))
        self.max_priority = max(self.max_priority, float(priorities.max()))


# ── λ-return calculation (standalone, JIT-compiled) ───────────────────────────

@njit(cache=True)
def calculate_lambda_returns(
    rewards: np.ndarray,  # (B, T)   float32
    max_q:   np.ndarray,  # (B, T+1) float32
    dones:   np.ndarray,  # (B, T)   float32
    mask:    np.ndarray,  # (B, T)   float32
    gamma:   float,
    lambd:   float,
) -> np.ndarray:           # (B, T)   float32
    """
    Vectorised backward λ-recurrence, JIT-compiled over the time axis.

    The Python for-loop over t is the only bottleneck — @njit eliminates its
    overhead while keeping numpy-vectorised operations over the B axis intact.

      Init:     G[:, t] = r[:, t] + γ · maxQ[:, t+1] · (1−done[:, t])
      Backward: G[:, t] += (1−done[:, t]) · γ · λ · mask[:, t] · (G[:, t+1] − maxQ[:, t+1])

    Terminal steps (done=1) zero the bootstrap and the correction term.
    Watkins cut (mask=0) collapses to 1-step TD at that step.
    """
    G = rewards + gamma * max_q[:, 1:] * (1.0 - dones)
    for t in range(G.shape[1] - 2, -1, -1):
        G[:, t] += (
            (1.0 - dones[:, t])
            * gamma * lambd * mask[:, t]
            * (G[:, t + 1] - max_q[:, t + 1])
        )
    return G.astype(np.float32)


# Warm up JIT on import (avoids cold-start latency on first refresh_cache call)
_w = np.zeros((1, 2), dtype=np.float32)
calculate_lambda_returns(_w, np.zeros((1, 3), np.float32), _w, _w, 0.99, 0.8)
del _w


# ── LambdaReplayBuffer ────────────────────────────────────────────────────────

class LambdaReplayBuffer(NStepReplayBuffer):
    """
    NStepReplayBuffer extended with Watkins' Q(λ) returns via a two-tier cache,
    with Daley & Amato (2018) median-based priority sampling.

    Architecture (Daley & Amato, 2018  arXiv:1810.09967):

      Tier 1 — Buffer : raw transitions               (standard ring buffer)
      Tier 2 — Cache  : pre-computed λ-returns, rebuilt at target-update frequency

    Priority sampling (§3.3 — Directly Prioritized Replay):
      δᵢ = |G^λᵢ − Q(sᵢ, aᵢ)|  for each cached slot i ∈ {0..S-1}
      med = median(δ₀, ..., δ_{S-1})
      P(xᵢ) = (1/S)(1+p)  if δᵢ > med        ← high-error slots
             = (1/S)       if δᵢ = med        ← boundary (exact ties)
             = (1/S)(1-p)  if δᵢ < med        ← low-error slots

      p is linearly annealed from per_priority → 0 over per_total_samples
      steps (Daley & Amato §3.3, following PER [Schaul 2015]).

      sample() uses stratified sampling over _slot_probs:
        divide [0,1] into batch_size equal segments,
        draw one uniform per segment, map through CDF.
        Lower variance than i.i.d., same expected distribution.

    Watkins mask is recomputed at refresh time (Daley & Amato §3):
      mask[t] = 1 if stored_action[t] == argmax Q_current(s_t) else 0

    sample() returns:
      rewards   = λ-return   (precomputed target)
      discounts = zeros       (kills bootstrap in RainbowDQN.train())

    Parameters
    ----------
    lambd            : float  λ ∈ [0,1].  0 → n-step TD,  1 → Monte Carlo.
    block_size       : int    Sequence length for backward λ-recurrence.
    cache_size       : int    Pre-computed transitions ready for sampling.
                              Must be divisible by block_size.
    use_watkins      : bool   True = Watkins' Q(λ),  False = Peng's Q(λ).
    per_priority     : float  Initial p for median split (annealed to 0).
                              0 → uniform;  0.5 → 3:1 ratio high:low.
    per_total_samples: int    Total env steps over which p anneals to 0.
    """

    def __init__(
        self,
        *args,
        lambd:             float = 0.8,
        block_size:        int   = 200,
        cache_size:        int   = 10_000,
        use_watkins:       bool  = True,
        per_priority:      float = 0.5,
        per_total_samples: int   = 1_000_000,
        **kwargs,
    ):
        assert cache_size % block_size == 0, \
            f"cache_size ({cache_size}) must be divisible by block_size ({block_size})"
        super().__init__(*args, **kwargs)

        self.lambd             = lambd
        self.block_size        = block_size
        self.cache_size        = cache_size
        self.use_watkins       = use_watkins
        self.per_priority      = per_priority
        self.per_total_samples = per_total_samples

        self._cache_obs     = np.zeros((cache_size, *self.obs_shape), dtype=np.float32)
        self._cache_actions = np.zeros((cache_size,),                  dtype=np.int64)
        self._cache_returns = np.zeros((cache_size,),                  dtype=np.float32)
        self._slot_probs    = np.full(cache_size, 1.0 / cache_size,   dtype=np.float64)
        self._cache_filled  = False
        self._priority_samples_seen = 0

    # ── cache management ──────────────────────────────────────────────────────

    def _sample_block_starts(self, n_blocks: int):
        """
        Sample (starts, env_indices) for n_blocks sequences.

        Valid start range excludes block_size positions on BOTH sides of the
        write head (self.pos) to prevent blocks from wrapping through stale data:
          n_valid = buffer_size - 2*block_size - 1
          starts  = randint(0, n_valid) shifted by (pos + block_size)
          → block ends at most at pos - 2  (never touches pos)

        When not full: simple randint up to pos - block_size - 1 (no wrap risk).
        """
        if self.full:
            n_valid = self.buffer_size - 2 * self.block_size - 1
            if n_valid <= 0:
                raise RuntimeError(
                    f"buffer_size ({self.buffer_size}) must be > 2*block_size+1 ({2*self.block_size+1})."
                )
            starts = np.random.randint(0, n_valid, size=n_blocks)
            starts = (starts + self.pos + self.block_size) % self.buffer_size
        else:
            max_start = self.pos - self.block_size - 1
            if max_start <= 0:
                raise RuntimeError(
                    f"Buffer has {self.pos} transitions, need ≥ {self.block_size + 2}."
                )
            starts = np.random.randint(0, max_start, size=n_blocks)

        env_indices = np.random.randint(0, self.n_envs, size=n_blocks)
        return starts, env_indices

    def refresh_cache(self, q_net: th.nn.Module, device: th.device) -> None:
        """
        Rebuild all cache slots in a single batched forward pass, then compute
        Daley & Amato median-based slot probabilities.

        Extraction (identical pattern to NStepReplayBuffer._get_samples):
          steps   = arange(block_size+1)                               # (T+1,)
          indices = (starts[:, None] + steps) % buffer_size            # (B, T+1)
          obs     = observations[indices, env_indices[:, None]]         # (B, T+1, obs_dim)

        Q-values: one batched forward pass over all (n_blocks × (block_size+1)) states.
        Watkins mask: refresh-time argmax recheck.
        λ-recurrence: calculate_lambda_returns (njit, vectorised over B).

        Priority (Daley & Amato §3.3):
          δᵢ = |G^λᵢ − Q(sᵢ, aᵢ)|
          p_now = per_priority × max(0, 1 − samples_seen / per_total_samples)
          _slot_probs ∝ (1±p_now) above/below median(δ)
        """
        n_blocks            = self.cache_size // self.block_size
        starts, env_indices = self._sample_block_starts(n_blocks)

        # ── vectorised extraction (NStepReplayBuffer pattern) ─────────────────
        steps   = np.arange(self.block_size + 1)                          # (T+1,)
        indices = (starts[:, None] + steps[None, :]) % self.buffer_size   # (B, T+1)

        obs     = self.observations[indices,       env_indices[:, None]]           # (B, T+1, obs_dim)
        actions = self.actions[indices[:, :-1],    env_indices[:, None], 0]        # (B, T)
        rewards = self.rewards[indices[:, :-1],    env_indices[:, None]]           # (B, T)
        dones   = self.dones[indices[:, :-1],      env_indices[:, None]]           # (B, T)

        obs     = obs.astype(np.float32)
        actions = actions.astype(np.int64)
        rewards = rewards.astype(np.float32)
        dones   = dones.astype(np.float32)

        # ── single batched Q-net forward pass ─────────────────────────────────
        obs_flat = obs.reshape(n_blocks * (self.block_size + 1), *self.obs_shape)
        q_net.eval()
        with th.no_grad():
            q_vals = q_net(
                th.tensor(obs_flat, dtype=th.float32, device=device)
            ).cpu().numpy()                                                # (B*(T+1), n_actions)
        q_net.train()

        q_vals = q_vals.reshape(n_blocks, self.block_size + 1, -1)        # (B, T+1, n_actions)
        max_q  = q_vals.max(axis=2).astype(np.float32)                    # (B, T+1)

        # ── Watkins mask (refresh-time recheck — Daley & Amato §3) ────────────
        if self.use_watkins:
            greedy = q_vals[:, :-1].argmax(axis=2)                         # (B, T)
            mask   = (actions == greedy).astype(np.float32)                # (B, T)
        else:
            mask = np.ones((n_blocks, self.block_size), dtype=np.float32)

        # ── λ-recurrence (njit, vectorised over B) ────────────────────────────
        returns = calculate_lambda_returns(rewards, max_q, dones, mask,
                                           self.gamma, self.lambd)        # (B, T)

        # ── Daley & Amato §3.3 — median-based slot probabilities ──────────────
        # Q(s_t, a_t): index into q_vals[:, :-1] with stored actions
        S          = self.cache_size
        q_states   = q_vals[:, :-1].reshape(S, -1)                        # (S, n_actions)
        acts_flat  = actions.reshape(S)
        q_at_act   = q_states[np.arange(S), acts_flat]                    # (S,)
        td_errors  = np.abs(returns.reshape(S) - q_at_act)                # (S,)

        p_now  = self.per_priority * max(
            0.0, 1.0 - self._priority_samples_seen / self.per_total_samples
        )
        med    = np.median(td_errors)
        probs  = np.where(td_errors > med, (1.0 + p_now) / S,
                 np.where(td_errors < med, (1.0 - p_now) / S,
                                           1.0 / S))
        self._slot_probs = probs / probs.sum()   # renormalise for float precision

        # ── write to cache ────────────────────────────────────────────────────
        self._cache_obs[:]     = obs[:, :-1].reshape(S, *self.obs_shape)
        self._cache_actions[:] = acts_flat
        self._cache_returns[:] = returns.reshape(S)
        self._cache_filled     = True

    def _calculate_lambda_returns(self, rewards, max_q, dones, mask) -> np.ndarray:
        """Thin wrapper — delegates to standalone calculate_lambda_returns."""
        return calculate_lambda_returns(
            rewards.astype(np.float32), max_q.astype(np.float32),
            dones.astype(np.float32),   mask.astype(np.float32),
            self.gamma, self.lambd,
        )

    # ── SB3 sampling interface ────────────────────────────────────────────────

    def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
        """
        Stratified sampling over _slot_probs (Daley & Amato §3.3).

        Divides [0, 1] into batch_size equal segments, draws one uniform sample
        per segment, then maps through the CDF of _slot_probs.  Lower variance
        than i.i.d. draws while preserving the same expected distribution.
        """
        if not self._cache_filled:
            raise RuntimeError("Call refresh_cache(q_net, device) before sampling.")

        # Stratified sampling over slot probabilities
        cumprobs   = np.cumsum(self._slot_probs)
        segment    = 1.0 / batch_size
        offsets    = np.random.uniform(0.0, segment, size=batch_size)
        thresholds = segment * np.arange(batch_size, dtype=np.float64) + offsets
        idx        = np.searchsorted(cumprobs, thresholds)
        idx        = np.clip(idx, 0, self.cache_size - 1)

        self._priority_samples_seen += batch_size

        obs     = self._cache_obs[idx]
        actions = self._cache_actions[idx]
        returns = self._cache_returns[idx]

        if env is not None:
            obs = self._normalize_obs(obs, env)

        dev       = self.device
        obs_t     = th.tensor(obs,     dtype=th.float32).to(dev)
        actions_t = th.tensor(actions, dtype=th.int64).unsqueeze(1).to(dev)
        returns_t = th.tensor(returns, dtype=th.float32).unsqueeze(1).to(dev)
        zero      = th.zeros(batch_size, 1, dtype=th.float32, device=dev)

        return ReplayBufferSamples(
            observations      = obs_t,
            actions           = actions_t,
            next_observations = obs_t,      # dummy — zeroed by discounts=0
            dones             = zero,
            rewards           = returns_t,  # λ-return IS the Bellman target
            discounts         = zero,       # kills bootstrap in train()
        )

    @property
    def priority_p(self) -> float:
        """Current (annealed) value of p for median priority split."""
        return self.per_priority * max(
            0.0, 1.0 - self._priority_samples_seen / self.per_total_samples
        )
