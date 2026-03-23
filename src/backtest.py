"""
backtest.py — on-demand accounting and reporting utilities.

All functions are stateless and operate on data already produced by
AssetVecEnv / AssetEnvState.  Nothing here runs during training.

Entry points
------------
price_series_from_env(env)          build price pd.Series from VecEnv arrays
infer_periods(price)                infer bars-per-year from median bar width
activity_to_returns(activity, price) sparse log → pct-change return Series
metrics(returns, equity, periods)   dict of scalar QuantStats metrics
tearsheet(returns, output, ...)     QuantStats HTML tearsheet
"""

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

import numpy as np
import pandas as pd
import quantstats as qs


# ── data helpers ──────────────────────────────────────────────────────────────

def price_series_from_env(env) -> pd.Series:
    """
    Build a price pd.Series with DatetimeIndex from a fitted AssetVecEnv.
    Index is derived from env.ts_ary (int64 ns) — the same timestamps used
    during training, so activity reindex will always align perfectly.
    """
    return pd.Series(
        env.price_ary,
        index=pd.to_datetime(env.ts_ary, unit='ns'),
        name='price',
        dtype='float32',
    )


def infer_periods(price: pd.Series) -> int:
    """
    Estimate bars-per-year from the median inter-bar interval.
    Returns an int suitable for QuantStats `periods` parameter.

    Examples
    --------
    daily  bars → 252
    hourly bars → 8_760
    15-min bars → 35_040
    """
    median_ns = int(price.index.to_series().diff().dropna().median().value)
    return int(round(365.25 * 24 * 3600 * 1e9 / median_ns))


# ── core bridge ───────────────────────────────────────────────────────────────

def activity_to_equity(activity: pd.DataFrame, price: pd.Series) -> pd.Series:
    """
    Reconstruct bar-by-bar equity from a sparse activity log.

        equity[t] = position_size * price[t] + cash - debt

    Uses the reindex + ffill pattern from assemble_logs_to_dataframe
    (RG/logging_client.py).  Duplicate timestamps (same-bar signals) are
    resolved by keeping the last state, matching `keep='last'` behaviour.

    Parameters
    ----------
    activity : pd.DataFrame  output of log_to_dataframe(s.extract_log())
    price    : pd.Series     full-resolution price, DatetimeIndex aligned
                             to activity (use price_series_from_env)

    Returns
    -------
    pd.Series  float32, same DatetimeIndex as price
    """
    deduped = activity[~activity.index.duplicated(keep='last')]
    dense   = deduped.reindex(price.index)
    dense[['position_size', 'cash', 'debt']] = (
        dense[['position_size', 'cash', 'debt']].ffill().fillna(0.0)
    )
    equity = (dense['position_size'] * price
              + dense['cash']
              - dense['debt']).astype('float32')
    equity.name = 'equity'
    return equity


def activity_to_returns(activity: pd.DataFrame, price: pd.Series) -> pd.Series:
    """
    Sparse activity log → pct-change return Series for QuantStats.
    First row (NaN from pct_change) is dropped.
    No timezone conversion required by QuantStats.
    """
    equity = activity_to_equity(activity, price)
    return equity.pct_change().dropna().rename('returns')


# ── metrics ───────────────────────────────────────────────────────────────────

def metrics(
    returns: pd.Series,
    equity:  pd.Series,
    periods: int,
) -> dict:
    """
    Compute a flat dict of scalar metrics via QuantStats.
    Pass both returns (for ratio metrics) and equity (for drawdown).
    `periods` should come from infer_periods().

    Returns
    -------
    dict[str, float]  — suitable for JSON serialisation or pd.Series
    """
    return {
        'sharpe':        float(qs.stats.sharpe(returns, periods=periods)),
        'sortino':       float(qs.stats.sortino(returns, periods=periods)),
        'calmar':        float(qs.stats.calmar(returns, periods=periods)),
        'cagr':          float(qs.stats.cagr(returns, periods=periods)),
        'max_drawdown':  float(qs.stats.max_drawdown(equity)),
        'volatility':    float(qs.stats.volatility(returns, periods=periods)),
        'omega':         float(qs.stats.omega(returns, periods=periods)),
        'profit_factor': float(qs.stats.profit_factor(returns)),
        'win_rate':      float(qs.stats.win_rate(returns)),
        'var_95':        float(qs.stats.value_at_risk(returns)),
    }


# ── reporting ─────────────────────────────────────────────────────────────────

def tearsheet(
    returns:    pd.Series,
    output:     str,
    benchmark:  pd.Series | None = None,
    periods:    int | None = None,
    title:      str = 'Strategy Tearsheet',
) -> None:
    """
    Write a self-contained QuantStats HTML tearsheet to `output`.

    Parameters
    ----------
    returns   : pct-change Series (from activity_to_returns)
    output    : file path, e.g. 'runs/exp_001/tearsheet.html'
    benchmark : optional buy-and-hold return Series, same index
    periods   : bars-per-year; inferred from returns index if None
    title     : HTML page title
    """
    if periods is None:
        periods = infer_periods(returns)

    qs.reports.html(
        returns,
        benchmark=benchmark,
        periods_per_year=periods,
        output=output,
        title=title,
    )
