# GBRL-Research · Sprint 1: VecEnv + Rainbow DQN

Research project applying value-based reinforcement learning to single-asset cryptocurrency trading (ETH/USDT hourly). This sprint establishes the core infrastructure and Rainbow DQN baseline.

## What's here

### `src/` — Core library

| File | Contents |
|---|---|
| `environment.py` | `AssetVecEnv` — SB3-compatible vectorized trading environment; random episode starts; position tracking; log-return rewards |
| `rainbow_dqn.py` | `RainbowDQN` base class (Double DQN, Dueling, PER, hook architecture); `DuelingQNetwork/Policy`; `StickyDQN` (Markov ε-exploration + Bellman switching cost) |
| `replay_buffer.py` | `PrioritizedReplayBuffer` (SumTree, IS weights, β annealing); `LambdaReplayBuffer` (TD(λ) weighted sampling); `NStepReplayBuffer` |
| `backtest.py` | QuantStats tearsheet, equity reconstruction, hourly-annualized Sharpe/drawdown metrics |
| `preprocessing.py` | Feature engineering pipeline (86 technical indicators via pandas-ta) |
| `discrete_critic.py` | Earlier standalone `StickyDQN` implementation (superseded by `rainbow_dqn.py`) |
| `utils.py` | Shared utilities |

### `tests/` — Experiments

| File | Contents |
|---|---|
| `optuna_hparam_search.ipynb` | Stage 1 (ASHA, 80 trials, TPE multivariate) + Stage 2 (20-seed Mann-Whitney U, Bonferroni) hyperparameter search |
| `optuna_stage1_trials.csv` | Stage 1 trial results (201 trials, best val_r ≈ 1.05) |

### `notebooks-p1/` — Development notebooks

Exploratory notebooks: PER/n-step ablations, Lambda replay testing, parallel env validation, GBRL Q-function sketch, earlier CV experiments.

### `notes/` — Phase documentation

- `PHASE_SUMMARY.md` — full phase retrospective: goals, implemented components, experiment results, conclusion, deferred ideas review
- `2026-03-22.md`, `2026-03-23.md` — sprint session notes
- `optuna_hparam_search.md` — Optuna study reference

## Algorithm stack

```
StickyDQN(RainbowDQN(DQN))
├── Double DQN          — online net selects, target net evaluates
├── Dueling networks    — V(s) + A(s,a) − mean(A) decomposition
├── PER                 — SumTree prioritized sampling, IS-weighted Huber loss
├── N-step returns      — hard bootstrap over n steps
├── Markov ε-exploration — persistence-biased transition matrix (mean_hold parameter)
└── Bellman switching cost — penalises position changes in next-Q before argmax
```

## Key results

**Stage 1 hyperparameter search** (201 trials, ASHA pruner):
- Best config: val_r ≈ 1.05 on 6-month ETH/USDT val window
- Top importances: `exploration_fraction` (0.26) > `replay_ratio` (0.20) > `mean_hold`/`total_samples` (0.10 each)

**Stage 2 statistical test** (top-10 configs × 20 seeds):
- Kruskal-Wallis H=7.65, p=0.5698 — no significant differences between configs
- Conclusion: hyperparameters are not the bottleneck; policy convergence is

## Setup

```bash
pip install stable-baselines3 optuna pandas-ta quantstats torch
```

Data not included (large file). Expected at `data/ETHUSDT_1h_2018-01-01_to_2025-12-31.csv`.

## Next: Sprint 2

Phase 2 focus: **Distributional RL + Sequential VecEnv**
- IQN / QR-DQN / C51 / Laplace code
- Staircase parallelization for non-stationarity coverage
- Action encoding modes (target position vs delta)
- Convergence gate before distributional upgrade