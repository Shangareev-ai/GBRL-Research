import numpy as np
import torch as th
import torch.nn.functional as F

from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.common.torch_layers import create_mlp

from replay_buffer import PrioritizedReplayBuffer, LambdaReplayBuffer


# ── Dueling network ───────────────────────────────────────────────────────────

class DuelingQNetwork(QNetwork):
    """
    Dueling Q-network — drop-in replacement for SB3's QNetwork.

    Splits the final layer into two heads:
      V(s)          — scalar state value
      A(s, a)       — per-action advantage
    Combined as:
      Q(s, a) = V(s) + A(s, a) − mean_a A(s, a)

    The mean-subtraction centres advantages at zero, making V and A
    identifiable and stabilising training.

    net_arch is interpreted as:
      net_arch[:-1]  — hidden layers shared between both heads
      net_arch[-1]   — width of the shared trunk output (feeds both heads)
    Requires len(net_arch) >= 1.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.net_arch) >= 1, "net_arch must have ≥ 1 layer for dueling"
        n_actions = self.action_space.n
        trunk_dim = self.net_arch[-1]

        trunk = create_mlp(self.features_dim, trunk_dim, self.net_arch[:-1], self.activation_fn)
        trunk.append(self.activation_fn())
        self.q_net      = th.nn.Sequential(*trunk)
        self.value_head = th.nn.Linear(trunk_dim, 1)
        self.adv_head   = th.nn.Linear(trunk_dim, n_actions)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        latent = self.q_net(self.extract_features(obs, self.features_extractor))
        value  = self.value_head(latent)                # (B, 1)
        adv    = self.adv_head(latent)                  # (B, n_actions)
        return value + adv - adv.mean(dim=1, keepdim=True)


class DuelingDQNPolicy(DQNPolicy):
    """DQNPolicy that uses DuelingQNetwork for both q_net and q_net_target."""

    def make_q_net(self) -> DuelingQNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return DuelingQNetwork(**net_args).to(self.device)


# ── RainbowDQN ────────────────────────────────────────────────────────────────

class RainbowDQN(DQN):
    """
    Rainbow DQN base class — composes four of the six Rainbow components:

      1. Double DQN        (use_double_dqn=True)
         Online net selects the bootstrap action; target net evaluates it.
         Eliminates maximisation bias present in standard DQN.

      2. Dueling networks  (pass DuelingDQNPolicy as policy arg)
         Drop-in: train() is unchanged; dueling is in the network architecture.

      3. Prioritized Experience Replay  (replay_buffer_class=PrioritizedReplayBuffer)
         IS-weighted Huber loss + priority feedback after each gradient step.
         Auto-activates via isinstance check — zero overhead on standard buffer.

      4. Multi-step returns  (n_steps kwarg, forwarded to NStepReplayBuffer)
         Hard n-step bootstrap. Watkins Q(λ) would replace this in a future sprint.

    Components not included (see notes):
      - Noisy Networks  — conflicts with domain-specific Markov ε-exploration in StickyDQN
      - Distributional RL (C51) — prefer QR-DQN for non-stationary financial returns

    Subclass hook API (override in subclasses for domain-specific logic):
      _bellman_next_q(next_q, data) → next_q
          Modify next-state Q-values before argmax/max.
          Default: identity. StickyDQN uses this for switching-cost penalty.

      _compute_loss(current_q, target_q, data) → scalar tensor
          Default: IS-weighted Huber when PER is active, plain Huber otherwise.

      _after_gradient_step(data, current_q, target_q) → None
          Default: updates PER priorities when PER is active.
    """

    def __init__(
        self,
        *args,
        use_double_dqn:       bool           = False,
        per_alpha:            float          = 0.6,
        per_beta:             float          = 0.4,
        per_epsilon:          float          = 1e-6,
        per_total_samples:    int            = 10_000_000,
        replay_buffer_class                  = None,
        replay_buffer_kwargs: dict | None    = None,
        **kwargs,
    ):
        if replay_buffer_class is PrioritizedReplayBuffer:
            per_defaults = dict(
                alpha         = per_alpha,
                beta          = per_beta,
                epsilon       = per_epsilon,
                total_samples = per_total_samples,
            )
            replay_buffer_kwargs = {**per_defaults, **(replay_buffer_kwargs or {})}

        super().__init__(
            *args,
            replay_buffer_class  = replay_buffer_class,
            replay_buffer_kwargs = replay_buffer_kwargs,
            **kwargs,
        )
        self.use_double_dqn = use_double_dqn

    # ── subclass hooks ────────────────────────────────────────────────────────

    def _bellman_next_q(self, next_q: th.Tensor, data) -> th.Tensor:
        """Modify next-state Q-values before argmax/max. Identity by default."""
        return next_q

    def _compute_loss(self, current_q: th.Tensor, target_q: th.Tensor, data) -> th.Tensor:
        """IS-weighted Huber loss when PER is active; plain Huber otherwise."""
        element_loss = F.smooth_l1_loss(current_q, target_q, reduction="none")
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            return (data.weights * element_loss).mean()
        return element_loss.mean()

    def _after_gradient_step(self, data, current_q: th.Tensor, target_q: th.Tensor) -> None:
        """Update PER priorities from TD errors after each gradient step."""
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            td_errors = (target_q - current_q).abs().detach().cpu().numpy().flatten()
            self.replay_buffer._update_priorities(data.indices, td_errors)

    # ── Bellman update ────────────────────────────────────────────────────────

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            data      = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = data.discounts if data.discounts is not None else self.gamma

            with th.no_grad():
                next_q_target = self.q_net_target(data.next_observations)

                if self.use_double_dqn:
                    # Online net selects; target net evaluates — no maximisation bias
                    next_q_online = self.q_net(data.next_observations)
                    next_q_online = self._bellman_next_q(next_q_online, data)
                    next_actions  = next_q_online.argmax(dim=1, keepdim=True)
                    next_q_max    = next_q_target.gather(1, next_actions)
                else:
                    next_q_target = self._bellman_next_q(next_q_target, data)
                    next_q_max    = next_q_target.max(dim=1).values.reshape(-1, 1)

                target_q = data.rewards + (1 - data.dones) * discounts * next_q_max

            current_q = th.gather(
                self.q_net(data.observations), dim=1, index=data.actions.long()
            )

            loss = self._compute_loss(current_q, target_q, data)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            self._after_gradient_step(data, current_q, target_q)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


# ── StickyDQN ─────────────────────────────────────────────────────────────────

class StickyDQN(RainbowDQN):
    """
    RainbowDQN + two trading-specific extensions:

      1. Markov ε-exploration
         Replaces uniform random exploration with a persistence-biased transition
         matrix. Actions tend to repeat for ~mean_hold bars.
             T[flat]  = [1-p, p/2, p/2]
             T[long]  = [p,   1-p, 0  ]
             T[short] = [p,   0,   1-p]
         p = 1/mean_hold. Exploration rate ε is still managed by SB3 (annealed).

      2. Switching cost (Bellman regularization)
         Penalises next-state Q-values for actions that differ from the position
         currently held (obs[0], float32 encoded):
             Q_reg(s', a) = Q(s', a) − c · 𝟙(a ≠ pos_t)
         Applied to the online net (action selection) in Double DQN mode,
         so the agent learns to value holding over churning.

    Inherits from RainbowDQN:
      Double DQN, Dueling (via policy), PER, multi-step returns, train() + hooks.
    """

    _T = {
        0: lambda p: (1 - p,   p / 2, p / 2),
        1: lambda p: (p,       1 - p, 0      ),
        2: lambda p: (p,       0,     1 - p  ),
    }

    def __init__(
        self,
        *args,
        mean_hold:      int   = 50,
        switching_cost: float = 0.01,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.p              = 1.0 / mean_hold
        self.switching_cost = switching_cost
        self._last_action: np.ndarray | None = None

    # ── exploration ───────────────────────────────────────────────────────────

    def _explore(self, n: int, episode_start=None) -> np.ndarray:
        if self._last_action is None:
            self._last_action = np.zeros(n, dtype=np.int64)
        if episode_start is not None:
            self._last_action[np.asarray(episode_start, dtype=bool)] = 0
        actions = np.empty(n, dtype=np.int64)
        for i in range(n):
            probs = self._T[int(self._last_action[i])](self.p)
            actions[i] = np.random.choice(3, p=probs)
        return actions

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        """Structured ε-exploration at all times — bypasses SB3's uniform warm-up."""
        actions, _ = self.predict(self._last_obs, deterministic=False)
        return actions, actions

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        n = observation.shape[0] if hasattr(observation, "shape") else 1
        if self._last_action is None or self._last_action.shape != (n,):
            self._last_action = np.zeros(n, dtype=np.int64)
        if episode_start is not None:
            self._last_action[np.asarray(episode_start, dtype=bool)] = 0
        if not deterministic and np.random.rand() < self.exploration_rate:
            actions = self._explore(n, episode_start)
            self._last_action = actions.copy()
            return actions, state
        actions, _ = self.policy.predict(observation, state, episode_start, True)
        self._last_action = np.asarray(actions, dtype=np.int64)
        return actions, state

    # ── LambdaReplayBuffer hook ───────────────────────────────────────────────

    def _on_step(self) -> None:
        super()._on_step()   # polyak update + ε schedule (increments self._n_calls)
        if isinstance(self.replay_buffer, LambdaReplayBuffer):
            if self._n_calls % max(self.target_update_interval // self.n_envs, 1) == 0:
                self.replay_buffer.refresh_cache(self.q_net, self.device)

    # ── Bellman regularization ────────────────────────────────────────────────

    def _bellman_next_q(self, next_q: th.Tensor, data) -> th.Tensor:
        """Subtract switching cost for actions that differ from the action taken at s.

        data.actions is the action executed at s, which equals the position held at s'.
        This is the correct reference: we penalise next-state actions that differ from
        the position the agent is now in (after the transition).
        """
        # data.actions: int64 (B, 1)  {0=flat, 1=long, 2=short}
        is_switch = (
            th.arange(self.action_space.n, device=self.device).unsqueeze(0) != data.actions
        ).float()                                                         # (B, n_actions)
        return next_q - self.switching_cost * is_switch
