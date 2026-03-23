# Optuna Hyperparameter Search — StickyDQN on ETH/USDT

## Setup

- **Environment:** `AssetVecEnv` — ETH/USDT 1h, 18-month train / 6-month val (2018-01 → 2019-12)
- **Model:** `StickyDQN(RainbowDQN)` — Markov ε-exploration + switching cost Bellman penalty + PER
- **Fixed:** `gamma=0.999`, `net_arch=[64,64]`, `train_freq=168` (1 week), `n_envs=16`, `replay_buffer_class=PrioritizedReplayBuffer`
- **Objective:** val net return `r = total_asset / initial_cash − 1`, deterministic rollout on full 6-month val window

---

## Protocols

### Stage 1 — ASHA (done)
- 1 fixed seed (`seed=42`), 80 trials, epoch checkpoints (1 epoch = 13 026 env steps)
- **Pruner:** `HyperbandPruner(min_resource=10, max_resource=40, reduction_factor=3)`
  - Rungs at epoch 10 → 30 → 40; ~55% compute saving vs unpruned
  - `min_resource=10` chosen to avoid pruning before configs show meaningful differentiation
- **Sampler:** `TPESampler(multivariate=True, n_startup_trials=15)` — multivariate TPE captures `gradient_steps` ↔ `replay_ratio` dependency
- **Storage:** `sqlite:///optuna_stage1.db`

### Stage 2 — seed robustness (pending)
- Top configs from Stage 1, 5 seeds each, median val return, no pruning
- Objective: confirm Stage 1 ranking survives seed variance (Henderson et al. 2018)

---

## Search Space (9 params, 1 derived)

| Parameter | Type | Range |
|---|---|---|
| `learning_rate` | log-float | [1e-5, 1e-3] |
| `target_update_interval` | log-int | [500, 5 000] |
| `exploration_fraction` | float | [0.2, 0.8] |
| `mean_hold` | int | [10, 200] |
| `switching_cost` | log-float | [1e-3, 0.2] |
| `gradient_steps` | log-int | [5, 64] |
| `replay_ratio` | float | [0.5, 4.0] |
| `n_steps` | int | [1, 10] |
| `total_samples` | log-int | [50k, 2M] |

`batch_size` derived: `round_down32(replay_ratio × 2688 / gradient_steps)` — stored as `trial.user_attr`.

---

## Done

### Stage 1 results (80 trials, 2026-03-22)

**Study:** `sqlite:///optuna_stage1.db`, study name `stage1_sticky_dqn_eth`

**Plots:** `tests/optuna_hparam_search.ipynb` §1 — Intermediate Values, Parallel Coordinates, Hyperparameter Importances

**Winner:** one trial dominates from epoch ~20, converging to `r ≈ 1.4` (140% net return on val). Most configs remain noisy — exploration schedule is the differentiator.

**Hyperparameter importances:**

| Rank | Parameter | Importance | Interpretation |
|---|---|---|---|
| 1 | `exploration_fraction` | 0.26 | Exploration length dominates — insufficient exploration = regime memorisation |
| 2 | `replay_ratio` | 0.20 | Data utilisation efficiency; validates searching it as first-class param |
| 3 | `mean_hold` | 0.10 | Markov commitment length is a real signal |
| 3 | `total_samples` | 0.10 | PER beta annealing speed matters as much as exploration |
| 5 | `n_steps` | 0.08 | Multi-step returns have modest effect; `n_steps=1` in winner |
| 6 | `switching_cost` | 0.07 | |
| 6 | `learning_rate` | 0.07 | |
| 8 | `target_update_interval` | 0.06 | |
| 9 | `gradient_steps` | 0.05 | Least important alone; effect absorbed into `replay_ratio` |

**Key observations:**
- Best config: high `exploration_fraction` (~0.65), low `n_steps` (1), `replay_ratio` ~2–3, low `switching_cost`
- `gradient_steps` least important alone but co-determines `batch_size` via `replay_ratio` — the coupling is correctly captured by multivariate TPE
- Parallel coordinates confirm: winning configs avoid extreme `switching_cost` and prefer longer exploration

---

## Possible ToDos

- **Stage 2** — run top 5–10 Stage 1 configs with 5 seeds, median return; confirm ranking is not seed-lucky
- **Narrow Stage 1 search** — fix `n_steps=1` (winner value, 0.08 importance), narrow `exploration_fraction` to [0.5, 0.8], re-run 40 trials on remaining 7 params
- **`total_samples` sweep** — 0.10 importance suggests PER annealing schedule is underexplored; dedicated sweep with `total_samples` ∈ [50k, 500k] on fixed other params
- **Longer training run** — winner found best policy at epoch ~20; train winning config to 80–100 epochs to check if it continues improving or plateaus
- **Walk-forward validation** — the 6-month val window is one regime (ETH 2019 bear). Re-evaluate winner on 2020, 2021, 2022 windows to test out-of-sample robustness
- **LambdaReplayBuffer** — `StickyDQN._on_step` hook already implemented; run ablation Stage 1 vs Stage 1 + Lambda to see if TD(λ) improves on the best config
- **`DuelingDQNPolicy`** — currently using `MlpPolicy`; add `DuelingDQNPolicy` as a categorical param or run separate Stage 1 with dueling fixed
- **RainbowDQN base class ablation** — per `project_todos_sprint1.md`: systematic ablation of Double DQN / Dueling / PER / n-step components on the winning hyperparameters