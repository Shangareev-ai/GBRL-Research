import numpy as np
import torch as th

# ── Exploration Schedule ──────────────────────────────────────────────────────

class PiecewiseLinearSchedule:
    """
    Schedule defined by a sequence of (step, value) breakpoints.

    Between breakpoints the value is linearly interpolated; beyond the last
    breakpoint the final value is held constant.

    Parameters
    ----------
    breakpoints : list[tuple[int, float]]
        Ordered list of (t, v) pairs.  The first breakpoint's t is normally 0.

    Example
    -------
    >>> s = PiecewiseLinearSchedule([(0, 1.0), (5000, 0.1), (20000, 0.01)])
    >>> s.value(0)      # 1.0
    >>> s.value(2500)   # 0.55
    >>> s.value(20000)  # 0.01
    >>> s.value(99999)  # 0.01  (clamped)
    """

    def __init__(self, breakpoints: list[tuple[int, float]]):
        assert len(breakpoints) >= 2, "Need at least two breakpoints"
        self.breakpoints = breakpoints

    def value(self, t: int) -> float:
        pts = self.breakpoints
        if t <= pts[0][0]:
            return pts[0][1]
        for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
            if t <= t1:
                frac = (t - t0) / (t1 - t0)
                return v0 + frac * (v1 - v0)
        return pts[-1][1]


class LinearEpsilonSchedule(PiecewiseLinearSchedule):
    """Linear decay from *start* to *end* over *total_steps*, then constant."""

    def __init__(self, start: float, end: float, total_steps: int):
        self.start       = start
        self.end         = end
        self.total_steps = total_steps
        super().__init__([(0, start), (total_steps, end)])


# ── Evaluation helpers ────────────────────────────────────────────────────────

def eval_agent_generic(agent, env, collect_rewards=True):
    """
    Run a greedy episode.

    Parameters
    ----------
    collect_rewards : bool
        If False, skip building the rewards list (saves memory when only
        the final asset value is needed, e.g. inside on_update callbacks).

    Returns
    -------
    rewards : list[float] | None
    final_asset : float
    """
    obs, _ = env.reset()
    done   = False
    rewards = [] if collect_rewards else None
    while not done:
        state_t = agent._to_tensor(np.array(obs, dtype=np.float32)[np.newaxis])
        with th.no_grad():
            q_vals = agent._forward(state_t, requires_grad=False)
        action_idx = q_vals.argmax(dim=1).item()
        env_action = (agent.action_map[action_idx]
                      if agent.action_map is not None else action_idx)
        obs, reward, terminated, truncated, _ = env.step(env_action)
        if collect_rewards:
            rewards.append(reward)
        done = terminated or truncated
    return rewards, env.asset_memory[-1]


def collect_metrics(agent, env):
    """
    Run a greedy episode and return a dict with:
        assets    – full portfolio value curve  (len = T+1, includes t=0)
        positions – target fraction at each step (len = T, 0.0=Flat / 1.0=Long)
        rewards   – log-return at each step     (len = T)
    """
    rewards, _ = eval_agent_generic(agent, env, collect_rewards=True)
    return {
        'assets':    np.array(env.asset_memory,   dtype=np.float64),
        'positions': np.array(env._target_memory, dtype=np.float32),
        'rewards':   np.array(rewards,            dtype=np.float32),
    }