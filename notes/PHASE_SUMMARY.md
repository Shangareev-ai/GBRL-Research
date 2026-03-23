# Phase Summary: VecEnv + Rainbow DQN

## Goals

Build a stable, reproducible deep Q-learning baseline for single-asset crypto trading (ETH/USDT hourly) that could be used as a foundation for further algorithmic research. Specifically:
- Implement the full Rainbow DQN stack on top of SB3's DQN
- Integrate a custom vectorized environment (`AssetVecEnv`) with structured ε-exploration
- Establish a principled training/evaluation protocol (VecNormalize, epoch loop, val rollout)
- Run a hyperparameter search to find a working configuration

## Implemented

### Infrastructure
- `AssetVecEnv` — custom SB3-compatible vectorized environment; random episode starts; episode length control; position tracking; log-return rewards
- `AssetEpisodeLog` / `AssetEnvState` — per-environment state bookkeeping
- `VecNormalize` integration — fixed Q-collapse from unscaled 86-feature obs
- `backtest.py` — QuantStats tearsheet, equity reconstruction, Sharpe/drawdown metrics (hourly annualization)
- Repository structure: `src/`, `notebooks/`, `tests/`, `data/`, `archive/`

### Rainbow DQN stack (`src/rainbow_dqn.py`)
- `RainbowDQN(DQN)` base class — Double DQN, hook architecture (`_bellman_next_q`, `_compute_loss`, `_after_gradient_step`)
- `DuelingQNetwork` / `DuelingDQNPolicy` — shared trunk → value + advantage heads, mean-centred
- `PrioritizedReplayBuffer` — SumTree, stratified sampling, IS weights, β annealing, `_update_priorities`
- `StickyDQN(RainbowDQN)` — Markov ε-exploration (transition matrix, `mean_hold` parameter), Bellman switching cost
- `LambdaReplayBuffer` hook — `_on_step()` refresh at target update frequency
- N-step returns via SB3 `NStepReplayBuffer`

### Bug fixes
- **Switching cost bug** — `_bellman_next_q` read `obs[:,0]` (current position); self-reinforcing flat bias caused degenerate all-flat policy. Fixed to `data.actions`. Validated via `switching_cost=0` ablation.
- **Episode boundary bug** — `_last_action` not reset on exploitation path; `IndexError` when n_envs switched between train (16) and val (1). Fixed with shape guard `_last_action.shape != (n,)`.

### Hyperparameter search (`tests/optuna_hparam_search.ipynb`)
- Stage 1: 1 seed, ASHA (HyperbandPruner, rungs 10→30→40), 80 trials, `TPESampler(multivariate=True)`
- Stage 2: top-10 configs × 20 seeds, Kruskal-Wallis + Mann-Whitney U, Bonferroni α=0.05/9

## Experiment Results

### Stage 1 (201 trials, 47 complete, 153 pruned)
Best trial: `val_r ≈ 1.05`, params: `gradient_steps=20, replay_ratio=1.09, n_steps=8, total_samples=69k, lr=1.76e-4, target_update=3117, exploration_fraction=0.318, mean_hold=70, switching_cost=0.023`

Hyperparameter importances:

| Param | Importance |
|---|---|
| `exploration_fraction` | 0.26 |
| `replay_ratio` | 0.20 |
| `mean_hold` / `total_samples` | 0.10 each |
| `n_steps` | 0.08 |
| `switching_cost` / `learning_rate` | 0.07 each |
| `gradient_steps` | 0.05 |

### Stage 2 (top-10 configs × 20 seeds)
- Kruskal-Wallis: H=7.65, **p=0.5698** — no significant differences between configs
- Best config: median val_r ≈ 1.05, std=0.384 over 20 seeds
- Consistent with ACM Computing Surveys meta-analysis (p=0.640 between all RL families)

## Conclusion

The Rainbow stack is correctly implemented and search is well-specified. The null Stage 2 result is not a failure — it reflects that **the policy has not converged**, not that hyperparameters are wrong. Std=0.38 over seeds on a 6-month window signals the agent is still exploring at evaluation time. The bottleneck is convergence and training regime, not the algorithm.

Exploration fraction (0.26) and replay ratio (0.20) dominate importance, pointing to **sample diversity and training regime** as the key levers — consistent with the convergence hypothesis.

## Deferred Ideas: Potential Review

| Idea | Verdict | Notes |
|---|---|---|
| **Stage 2 narrowed re-run** (fix `n_steps=8`, narrow `expl_frac`) | Low priority | Stage 2 null result means narrowing the search space won't help until convergence is achieved |
| **Walk-forward validation** | High — next phase | Blocked on convergence; do after policy stabilizes |
| **LambdaReplayBuffer ablation** | Medium | Hook implemented; run on stable baseline |
| **Rainbow ablation** (toggle components) | Medium | Useful once there's a converging reference; deferred to next phase |
| **Bellman directional regularization** | Low — blocked | Requires regime detection, which is not yet scoped |
| **DuelingDQNPolicy ablation** | Low | Dueling is architecturally sound; deprioritise vs convergence work |
| **`total_samples` dedicated sweep** | Low | PER annealing is secondary to the core non-convergence problem |
| **Noisy Networks** | Dropped | Conflicts with Markov ε-exploration; the structured exploration is a differentiator |
| **C51 distributional** | Superseded | IQN is strictly better; picked up in next phase |