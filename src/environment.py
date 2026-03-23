"""
environment.py — single-asset RL trading environment.

Classes
-------
AssetEpisodeLog   numba jitclass — sparse signal buffer per episode
AssetEnvState     numba jitclass — leveraged single-asset state machine
AssetVecEnv       SB3 VecEnv    — vectorised wrapper with Discrete(3) actions

Helpers
-------
log_to_dataframe(log) → pd.DataFrame   sparse log → DatetimeIndex DataFrame
"""

import numpy as np
import pandas as pd
from numba import float32, int8, int32, int64, types, njit, optional
from numba.experimental import jitclass
from numba.typed import Dict
from typing import Any, List, Optional, Type
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv


# ── AssetEpisodeLog ───────────────────────────────────────────────────────────
# Single-asset trajectory buffer. Field layout identical to RG/logging_client.py
# so that assemble_logs_to_dataframe() works on frozen_copy() directly.

asset_episode_log_spec = [
    ('timestamps',     int64[:]),
    ('signal_types',   int8[:]),     # 0=pass, 1=open_long, 2=open_long+,
                                     # -1=open_short, -2=open_short+,
                                     #  4=close_long, -4=close_short
    ('leverage',       float32[:]),
    ('position_sizes', float32[:]),  # coin held (negative for short)
    ('cash',           float32[:]),
    ('debt',           float32[:]),
    ('size',           int32),
    ('capacity',       int32),
]

@jitclass(asset_episode_log_spec)
class AssetEpisodeLog:
    def __init__(self, initial_size=256):
        self.capacity       = initial_size
        self.size           = 0
        self.timestamps     = np.empty(initial_size, dtype=np.int64)
        self.signal_types   = np.empty(initial_size, dtype=np.int8)
        self.leverage       = np.empty(initial_size, dtype=np.float32)
        self.position_sizes = np.empty(initial_size, dtype=np.float32)
        self.cash           = np.empty(initial_size, dtype=np.float32)
        self.debt           = np.empty(initial_size, dtype=np.float32)

    def _resize(self):
        new_cap = self.capacity * 2

        new_ts = np.empty(new_cap, dtype=np.int64)
        new_ts[:self.size] = self.timestamps[:self.size]
        self.timestamps = new_ts

        new_st = np.empty(new_cap, dtype=np.int8)
        new_st[:self.size] = self.signal_types[:self.size]
        self.signal_types = new_st

        new_lev = np.empty(new_cap, dtype=np.float32)
        new_lev[:self.size] = self.leverage[:self.size]
        self.leverage = new_lev

        new_ps = np.empty(new_cap, dtype=np.float32)
        new_ps[:self.size] = self.position_sizes[:self.size]
        self.position_sizes = new_ps

        new_cash = np.empty(new_cap, dtype=np.float32)
        new_cash[:self.size] = self.cash[:self.size]
        self.cash = new_cash

        new_debt = np.empty(new_cap, dtype=np.float32)
        new_debt[:self.size] = self.debt[:self.size]
        self.debt = new_debt

        self.capacity = new_cap

    def push_signal(self, ts, signal_type, leverage, position_size, cash, debt):
        if self.size >= self.capacity:
            self._resize()
        i = self.size
        self.timestamps[i]     = ts
        self.signal_types[i]   = signal_type
        self.leverage[i]       = leverage
        self.position_sizes[i] = position_size
        self.cash[i]           = cash
        self.debt[i]           = debt
        self.size += 1

    def get_logged_signals(self):
        """Compatible with assemble_logs_to_dataframe() from logging_client.py."""
        return (
            self.timestamps[:self.size],
            self.signal_types[:self.size],
            self.leverage[:self.size],
            self.position_sizes[:self.size],
            self.cash[:self.size],
            self.debt[:self.size],
        )

    def frozen_copy(self):
        """
        Return a trimmed copy with capacity = size + 1.
        The extra slot is reserved for the end marker written by
        AssetEnvState.extract_log() — no resize triggered on that push.
        """
        copy = AssetEpisodeLog(self.size + 1)
        copy.timestamps[:-1]     = self.timestamps[:self.size]
        copy.signal_types[:-1]   = self.signal_types[:self.size]
        copy.leverage[:-1]       = self.leverage[:self.size]
        copy.position_sizes[:-1] = self.position_sizes[:self.size]
        copy.cash[:-1]           = self.cash[:self.size]
        copy.debt[:-1]           = self.debt[:self.size]
        copy.size     = self.size
        copy.capacity = self.size + 1
        return copy


# ── state dict factory ────────────────────────────────────────────────────────
# Dict.empty() cannot be called directly inside a jitclass method because numba
# cannot resolve module-level aliases in the nopython compiled scope.
# Wrapping in @njit is the canonical fix.

@njit
def _make_state_dict():
    """Return an empty Dict[unicode_type → float32] for AssetEnvState.step()."""
    return Dict.empty(key_type=types.unicode_type, value_type=types.float32)


# ── AssetEnvState ─────────────────────────────────────────────────────────────
# Single-asset leveraged state machine. Direct port of RG/logging_client.py.
#
# Sentinel convention: entry_price = collateral = liquidation_price = None when no position.
# leverage argument: pass None to open_long / open_short to use max_leverage.
#
# Episode clock (idx) belongs to AssetVecEnv, not here.
# Dict.empty() must be called via the @njit helper _make_state_dict().

_asset_log_type = AssetEpisodeLog.class_type.instance_type

asset_env_state_spec = [
    ('cash',               float32),
    ('initial_cash',       float32),
    ('coin',               float32),           # > 0 = long, < 0 = short
    ('debt',               float32),           # > 0 = borrowed (long); < 0 = locked (short)
    ('entry_price',        optional(float32)),
    ('collateral',         optional(float32)),
    ('liquidation_price',  optional(float32)),
    ('position',           int8),              # 1 = long, -1 = short, 0 = flat
    ('leverage',           float32),
    ('max_leverage',       float32),
    ('allow_shorts',       int8),
    ('commission_rate',    float32),
    ('total_asset',        float32),
    ('last_ts',            int64),
    ('log',                _asset_log_type),
]

@jitclass(asset_env_state_spec)
class AssetEnvState:
    def __init__(self, initial_cash, commission_rate, max_leverage=1.0, allow_shorts=0):
        self.cash              = float32(initial_cash)
        self.initial_cash      = float32(initial_cash)
        self.coin              = float32(0.0)
        self.debt              = float32(0.0)
        self.collateral        = None
        self.entry_price       = None
        self.liquidation_price = None
        self.position          = int8(0)
        self.leverage          = float32(0.0)
        self.max_leverage      = float32(max_leverage)
        self.allow_shorts      = int8(allow_shorts)
        self.commission_rate   = float32(commission_rate)
        self.total_asset       = float32(initial_cash)
        self.log               = AssetEpisodeLog(int32(256))
        self.last_ts           = int64(0)

    # ── private helpers ───────────────────────────────────────────

    def _portfolio_value(self, price):
        """
        Unified equity formula:
          Flat:  equity = cash
          Long:  equity = coin × price − debt   (coin > 0, debt > 0)
          Short: equity = coin × price − debt   (coin < 0, debt < 0)
        """
        if self.position == int8(0):
            return self.cash
        return self.coin * price - self.debt

    def _update_liquidation_price(self):
        """
        liq_price = debt / coin  (positive for both long and short)
          Long:  liquidate when price ≤ liq_price
          Short: liquidate when price ≥ liq_price
        Equity = 0 exactly at liq_price (before closing commission).
        """
        if self.coin != float32(0.0):
            self.liquidation_price = self.debt / self.coin
        else:
            self.liquidation_price = None

    # ── trading helpers ───────────────────────────────────────────

    def open_long(self, price, ts, leverage=None):
        """
        Open or increase a long position.
        Ignores if: already short / no cash / requested lev ≤ current lev.

        New long:
          borrowed = cash × (lev − 1)
          coin     = cash × lev × (1 − comm) / price
          debt     = cash × (lev − 1);  cash = 0

        Increase leverage:
          Δcoin = collateral × (lev − old_lev) × (1 − comm) / price
          Δdebt = collateral × (lev − old_lev)
          entry_price updated as leverage-weighted average
        """
        if self.position == -1:
            return
        elif self.position == 0 and self.cash <= 0:
            return

        if leverage is None:
            lev = self.max_leverage
        elif leverage > self.max_leverage:
            lev = self.max_leverage
        else:
            lev = leverage

        if self.leverage >= lev:
            return

        if self.position == int8(0):
            self.entry_price = price
            self.collateral  = self.cash
            signal_type      = int8(1)
            borrowed_funds   = self.cash * (lev - float32(1.0))
            total_funds      = self.cash + borrowed_funds
            coin_bought      = total_funds * (float32(1.0) - self.commission_rate) / price
        else:
            old_lev          = self.leverage
            self.entry_price = (old_lev / lev) * self.entry_price + ((lev - old_lev) / lev) * price
            signal_type      = int8(2)
            borrowed_funds   = self.collateral * (lev - old_lev)
            coin_bought      = borrowed_funds * (float32(1.0) - self.commission_rate) / price

        self.cash  = float32(0.0)
        self.coin += coin_bought
        self.debt += borrowed_funds
        self.position = int8(1)
        self.leverage = lev
        self._update_liquidation_price()
        self.log.push_signal(ts, signal_type, self.leverage, self.coin, self.cash, self.debt)

    def open_short(self, price, ts, leverage=None):
        """
        Open or increase a short position.
        Ignores if: shorts disabled / already long / no cash / requested lev ≤ current lev.

        New short:
          coin_borrowed = cash × lev / price
          cash_obtained = coin_borrowed × price × (1 − comm)
          locked        = cash_obtained + cash   (proceeds + original collateral)
          coin −= coin_borrowed;  debt −= locked;  cash = 0

        Increase leverage:
          Δcoin_borrowed = collateral × (lev − old_lev) / price
          Δlocked        = Δcoin_borrowed × price × (1 − comm)
          (original collateral already locked; only new proceeds added)
        """
        if not self.allow_shorts:
            return
        elif self.position == 1:
            return
        elif self.position == 0 and self.cash <= 0:
            return

        if leverage is None:
            lev = self.max_leverage
        elif leverage > self.max_leverage:
            lev = self.max_leverage
        else:
            lev = leverage

        if self.leverage >= lev:
            return

        if self.position == int8(0):
            self.entry_price  = price
            self.collateral   = self.cash
            signal_type       = int8(-1)
            coin_borrowed     = (self.cash * lev) / price
            cash_obtained     = coin_borrowed * price * (float32(1.0) - self.commission_rate)
            locked            = cash_obtained + self.cash
        else:
            old_lev           = self.leverage
            self.entry_price  = (old_lev / lev) * self.entry_price + ((lev - old_lev) / lev) * price
            signal_type       = int8(-2)
            coin_borrowed     = self.collateral * (lev - old_lev) / price
            cash_obtained     = coin_borrowed * price * (float32(1.0) - self.commission_rate)
            locked            = cash_obtained

        self.cash  = float32(0.0)
        self.coin -= coin_borrowed
        self.debt -= locked
        self.position = int8(-1)
        self.leverage = lev
        self._update_liquidation_price()
        self.log.push_signal(ts, signal_type, self.leverage, self.coin, self.cash, self.debt)

    def close_long(self, price, ts):
        """proceeds = coin × price × (1 − comm);  cash += proceeds − debt"""
        if self.position != int8(1):
            return
        cash_obtained  = self.coin * price * (float32(1.0) - self.commission_rate)
        self.cash     += cash_obtained - self.debt
        self.coin               = float32(0.0)
        self.debt               = float32(0.0)
        self.collateral         = None
        self.entry_price        = None
        self.liquidation_price  = None
        self.position           = int8(0)
        self.leverage           = float32(0.0)
        self.log.push_signal(ts, int8(4), self.leverage, self.coin, self.cash, self.debt)

    def close_short(self, price, ts):
        """
        coins_to_buy = −coin;  cash_used = coins_to_buy × price × (1 + comm)
        unlocked = −debt;  cash += unlocked − cash_used
        """
        if self.position != int8(-1):
            return
        coins_to_buy = -self.coin
        cash_used    = coins_to_buy * price * (float32(1.0) + self.commission_rate)
        self.cash   += -self.debt - cash_used
        self.coin               = float32(0.0)
        self.debt               = float32(0.0)
        self.collateral         = None
        self.entry_price        = None
        self.liquidation_price  = None
        self.position           = int8(0)
        self.leverage           = float32(0.0)
        self.log.push_signal(ts, int8(-4), self.leverage, self.coin, self.cash, self.debt)

    def check_liquidation(self, price, ts):
        if self.liquidation_price is None:
            return
        if self.position == int8(1) and price <= self.liquidation_price:
            self.close_long(price, ts)
        elif self.position == int8(-1) and price >= self.liquidation_price:
            self.close_short(price, ts)

    # ── RL interface ──────────────────────────────────────────────

    def step(self, raw_action, price, turb, ts):
        """
        Target-position step.

        raw_action : float32   in [-1, 1]
                     > 0   → long  at raw_action × max_leverage
                     < 0   → short at |raw_action| × max_leverage (if allow_shorts)
                     ≈ 0   → flat  (dead-zone ±0.05)
        price      : float32   current bar close price
        turb       : int8      1 = turbulence event → force flat
        ts         : int64     nanosecond timestamp of the current bar

        Returns
        -------
        state      : Dict[unicode → float32]   {'position': float32}
        reward     : float32   log(equity_t / equity_{t−1}); −10.0 on bankruptcy
        terminated : int8      1 = bankrupt (total_asset ≤ 0), 0 = alive
        """
        if ts < self.last_ts:
            raise ValueError('timestamp not monotonic')
        self.last_ts = ts

        prev_value = self.total_asset

        FLAT_ZONE = float32(0.05)
        if raw_action > FLAT_ZONE:
            target_dir = int8(1)
            target_lev = raw_action * self.max_leverage
        elif raw_action < -FLAT_ZONE and self.allow_shorts:
            target_dir = int8(-1)
            target_lev = -raw_action * self.max_leverage
        else:
            target_dir = int8(0)
            target_lev = float32(0.0)

        self.check_liquidation(price, ts)

        if turb:
            target_dir = int8(0)

        if self.position != int8(0):
            reduce = target_dir == self.position and target_lev < self.leverage - float32(0.01)
            if target_dir == int8(0) or target_dir != self.position or reduce:
                if self.position == int8(1):
                    self.close_long(price, ts)
                else:
                    self.close_short(price, ts)

        if target_dir == int8(1):
            self.open_long(price, ts, target_lev)
        elif target_dir == int8(-1):
            self.open_short(price, ts, target_lev)

        self.total_asset = self._portfolio_value(price)
        terminated       = int8(1) if self.total_asset <= float32(0.0) else int8(0)

        state = _make_state_dict()
        state['position'] = float32(self.position)

        reward = float32(-10.0) if terminated else float32(np.log(self.total_asset / prev_value))
        return state, reward, terminated

    def reset(self, price, ts):
        self.cash              = self.initial_cash
        self.coin              = float32(0.0)
        self.debt              = float32(0.0)
        self.collateral        = None
        self.entry_price       = None
        self.liquidation_price = None
        self.position          = int8(0)
        self.leverage          = float32(0.0)
        self.total_asset       = self.initial_cash
        self.log               = AssetEpisodeLog(int32(256))
        self.log.push_signal(ts, int8(0), self.leverage, self.coin, self.cash, self.debt)
        self.last_ts = ts

    def extract_log(self):
        """Frozen log with end-of-episode pass signal appended at last_ts."""
        copy = self.log.frozen_copy()  # capacity = size + 1, no resize on push
        copy.push_signal(
            self.last_ts, int8(0), self.leverage, self.coin, self.cash, self.debt
        )
        return copy


# ── log_to_dataframe ─────────────────────────────────────────────────────────

def log_to_dataframe(log) -> pd.DataFrame:
    """
    Convert a frozen AssetEpisodeLog to a DataFrame with DatetimeIndex.
    First half of assemble_logs_to_dataframe() (RG/logging_client.py) — sparse
    activity only, no price data, no equity calculation.
    Duplicate timestamps are preserved (one row per signal event).

    For episode-end activity use s.extract_log() — appends end marker at last_ts.
    """
    dates, signal_types, leverage, position_sizes, cash, debt = log.get_logged_signals()
    df = pd.DataFrame({
        'signal_type':   signal_types,
        'leverage':      leverage,
        'position_size': position_sizes,
        'cash':          cash,
        'debt':          debt,
    }, index=pd.to_datetime(dates, unit='ns'))
    df.index.name = 'Date'
    return df


# ── AssetVecEnv ───────────────────────────────────────────────────────────────
# Actions: Discrete(3)
#   0 = neutral  →  flat              (raw_action =  0.0)
#   1 = long     →  full max_leverage (raw_action =  1.0)
#   2 = short    →  full max_leverage (raw_action = -1.0, requires allow_shorts)
#
# Observation: [position | tech[0] … tech[F-1]]   shape = 1 + F
#   'position' mirrors AssetEnvState.step() state dict
#   tech = config["tech_array"] row at the current bar
#
# config keys
#   price_array      : float32[T]        close prices
#   tech_array       : float32[T × F]    technical features
#   timestamp_array  : int64[T]          bar timestamps in ns   (highest priority)
#   date             : DatetimeIndex/array                       (second priority)
#   (neither)        : synthetic arange  0, 1, 2 … T-1 ns

class AssetVecEnv(VecEnv):

    metadata   = {"render_modes": []}
    render_mode = None

    _RAW = [np.float32(0.0), np.float32(1.0), np.float32(-1.0)]

    def __init__(
        self,
        config: dict,
        n_envs: int = 1,
        initial_cash: float = 1e6,
        commission_rate: float = 1e-3,
        max_leverage: float = 1.0,
        allow_shorts: bool = False,
        episode_len: Optional[int] = None,
    ):
        # ── data ─────────────────────────────────────────────────
        price_raw      = config["price_array"].astype(np.float32)
        self.price_ary = price_raw.reshape(-1) if price_raw.ndim == 2 else price_raw
        self.tech_ary  = config["tech_array"].astype(np.float32)

        if "timestamp_array" in config:
            self.ts_ary = config["timestamp_array"].astype(np.int64)
        elif "date" in config:
            self.ts_ary = pd.DatetimeIndex(config["date"]).asi8
        else:
            self.ts_ary = np.arange(len(self.price_ary), dtype=np.int64)

        self.T        = len(self.price_ary)
        self.tech_dim = self.tech_ary.shape[1]

        # ── spaces ────────────────────────────────────────────────
        obs_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(1 + self.tech_dim,),
            dtype=np.float32,
        )
        super().__init__(n_envs, obs_space, spaces.Discrete(3))

        # ── per-env state ─────────────────────────────────────────
        self._initial_cash = np.float32(initial_cash)
        self.states: List[AssetEnvState] = [
            AssetEnvState(initial_cash, commission_rate, max_leverage, int(allow_shorts))
            for _ in range(n_envs)
        ]
        self.idx         = np.zeros(n_envs, dtype=np.int32)
        self._actions: Optional[np.ndarray] = None
        self.reset_infos: List[dict] = [{} for _ in range(n_envs)]

        # ── random-window mode ────────────────────────────────────
        # episode_len=None → full series, episode_len=int → random start + fixed window
        self._episode_len = episode_len
        self._end_idx     = np.full(n_envs, self.T, dtype=np.int32)

    # ── helpers ───────────────────────────────────────────────────

    def _draw_start(self) -> int:
        max_start = self.T - self._episode_len
        return int(np.random.randint(0, max_start + 1))

    def _obs(self, env_idx: int, i: int) -> np.ndarray:
        s   = self.states[env_idx]
        obs = np.empty(self.observation_space.shape[0], dtype=np.float32)
        obs[0]  = np.float32(s.position)
        obs[1:] = self.tech_ary[i]
        return obs

    def _all_obs(self) -> np.ndarray:
        out = np.empty((self.num_envs, self.observation_space.shape[0]), dtype=np.float32)
        for n in range(self.num_envs):
            out[n] = self._obs(n, int(self.idx[n]))
        return out

    # ── VecEnv interface ──────────────────────────────────────────

    def reset(self) -> np.ndarray:
        for n, s in enumerate(self.states):
            if self._episode_len is not None:
                start = self._draw_start()
                self.idx[n]      = start
                self._end_idx[n] = start + self._episode_len
            else:
                self.idx[n] = 0
            s.reset(self.price_ary[self.idx[n]], np.int64(self.ts_ary[self.idx[n]]))
            self.reset_infos[n] = {}
        return self._all_obs()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        obs     = np.empty((self.num_envs, self.observation_space.shape[0]), dtype=np.float32)
        rewards = np.empty(self.num_envs, dtype=np.float32)
        dones   = np.zeros(self.num_envs, dtype=np.bool_)
        infos: List[dict] = []

        for n, s in enumerate(self.states):
            i   = int(self.idx[n])
            raw = self._RAW[int(self._actions[n])]
            _, rew, term = s.step(raw, self.price_ary[i], np.int8(0), np.int64(self.ts_ary[i]))

            self.idx[n] += 1
            rewards[n]   = rew
            done         = self.idx[n] >= int(self._end_idx[n]) or bool(term)
            dones[n]     = done

            info: dict = {}
            if done:
                info["terminal_observation"] = self._obs(n, min(int(self.idx[n]), self.T - 1))
                info["episode"] = {
                    "r":        float(s.total_asset / self._initial_cash - 1.0),
                    "l":        int(s.log.size),
                    "activity": log_to_dataframe(s.extract_log()),
                }
                if bool(term):
                    info["terminated"] = True
                elif self._episode_len is not None:
                    info["TimeLimit.truncated"] = True
                if self._episode_len is not None:
                    start = self._draw_start()
                    self.idx[n]      = start
                    self._end_idx[n] = start + self._episode_len
                else:
                    self.idx[n] = 0
                s.reset(self.price_ary[self.idx[n]], np.int64(self.ts_ary[self.idx[n]]))

            obs[n] = self._obs(n, int(self.idx[n]))
            infos.append(info)

        return obs, rewards, dones, infos

    def close(self) -> None:
        pass

    # ── utility ───────────────────────────────────────────────────

    def get_attr(self, attr_name: str, indices=None) -> List[Any]:
        indices = self._get_indices(indices)
        if hasattr(type(self), attr_name):
            states = getattr(self, 'states', None)
            if states is None or not hasattr(states[0], attr_name):
                val = getattr(self, attr_name)
                return [val] * len(indices)
        return [getattr(self.states[i], attr_name) for i in indices]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        for i in self._get_indices(indices):
            setattr(self.states[i], attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> List[Any]:
        return [
            getattr(self.states[i], method_name)(*method_args, **method_kwargs)
            for i in self._get_indices(indices)
        ]

    def env_is_wrapped(self, wrapper_class: Type, indices=None) -> List[bool]:
        return [False for _ in self._get_indices(indices)]

    def _get_indices(self, indices) -> List[int]:
        if indices is None:
            return list(range(self.num_envs))
        return [indices] if isinstance(indices, int) else list(indices)
