import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN


class StickyDQN(DQN):
    """
    DQN with two extensions:

    1. Structured ε-exploration via Markov transition matrix (no uniform random).
       When ε fires, action is sampled from:
           T = [[1-p, p/2, p/2],   flat  → stay / long / short
                [p,   1-p, 0  ],   long  → flat / stay / no flip
                [p,   0,   1-p]]   short → flat / no flip / stay
       p = 1/mean_hold  →  E[T] = mean_hold bars per commitment.

    2. Switching cost in the Bellman target.
       The argmax over next Q-values is penalised for actions that differ
       from the action currently stored in the replay buffer:
           target = r + γ · max_a [Q(s', a) − c · 𝟙(a ≠ a_current)]
       The agent must clear a Q-value margin of c to justify switching.
       c is in Q-value units — tune relative to observed Q-value spread.

    Parameters
    ----------
    mean_hold      : int    Expected bars per committed position. p = 1/mean_hold.
    switching_cost : float  Bellman penalty for action change (Q-value units).
    """

    _T = {
        0: lambda p: (1 - p,   p / 2, p / 2),
        1: lambda p: (p,       1 - p, 0      ),
        2: lambda p: (p,       0,     1 - p  ),
    }

    def __init__(self, *args, mean_hold: int = 50, switching_cost: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.p               = 1.0 / mean_hold
        self.switching_cost  = switching_cost
        self._last_action: np.ndarray | None = None

    # ── exploration ───────────────────────────────────────────────────────────

    def _explore(self, n: int) -> np.ndarray:
        actions = np.empty(n, dtype=np.int64)
        for i in range(n):
            probs = self._T[int(self._last_action[i])](self.p)
            actions[i] = np.random.choice(3, p=probs)
        return actions

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        """Use structured exploration from step 0, bypassing uniform warm-up."""
        actions, _ = self.predict(self._last_obs, deterministic=False)
        return actions, actions

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        n = observation.shape[0] if hasattr(observation, "shape") else 1

        if self._last_action is None:
            self._last_action = np.zeros(n, dtype=np.int64)

        if episode_start is not None:
            self._last_action[np.asarray(episode_start, dtype=bool)] = 0

        if not deterministic and np.random.rand() < self.exploration_rate:
            self._last_action = self._explore(n)
            return self._last_action.copy(), state

        actions, _ = self.policy.predict(observation, state, episode_start, True)
        self._last_action = np.asarray(actions, dtype=np.int64)
        return actions, state

    # ── Bellman with switching cost ───────────────────────────────────────────
    # TODO: n-step approximation — `replay_data.actions` is the action at step t,
    #       but `next_observations` is the state at t+n.  The switching cost therefore
    #       compares the action the agent will choose at t+n against the action stored
    #       at t, which may be n steps stale.  Correct fix: store action at t+n-1 in
    #       the buffer (requires a custom ReplayBuffer).  Accepted approximation for now.
    #
    # TODO: Bellman regularization — extend switching cost to directional restrictions:
    #       trend-following (penalise counter-trend entries) and counter-trend
    #       (penalise trend-following entries).  Revisit when regime detection is in place.

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts   = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                next_q = self.q_net_target(replay_data.next_observations)   # (B, 3)

                # penalise switching: subtract c from every action ≠ a_current
                a_cur     = replay_data.actions                              # (B, 1)
                n_actions = self.action_space.n
                is_switch = (th.arange(n_actions, device=self.device).unsqueeze(0)
                             != a_cur).float()                               # (B, 3)
                next_q    = next_q - self.switching_cost * is_switch

                next_q_max = next_q.max(dim=1).values.reshape(-1, 1)
                target_q   = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_max

            current_q = self.q_net(replay_data.observations)
            current_q = th.gather(current_q, dim=1, index=replay_data.actions.long())

            loss = F.smooth_l1_loss(current_q, target_q)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
