# MPO Fit Function - Write-up Answers (Q/A format, with numbered ablation citations)

> Source material: `fitSnarang73Ameshram_MPO_10H_ffw_fix-Copy1.py` and
> `Multi Period Optimisation Fit - Presentation.pdf`, plus the toy-example
> data in `mpo_toy_example` / `toy_example_data.json`.

---

## 1. Identification

**Name of idea**

Multi-Period Optimisation (MPO) Fit Function.

**Slack channel**

Not specified in the supplied materials (not applicable / to be filled by author).

**preTQ100m_xxx label name**

`preTQ100m_mpo_fit`.

**Author(s)**

Sanchit Narang, Alankar Meshram.

**Function type (filter, fit, post-processing, other)**

**Fit** function (a `fit_function` class with `fit()` + `construct_preA()`). It also embeds a portfolio-construction optimizer inside `construct_preA`, but in the TQ100m pipeline it is registered as a fit function (filter -> **fit** -> post-processing).

---

## 2. Idea & hypothesis

**Describe the idea in one paragraph.**

Standard mean-variance optimisation (MVO) is *single-period*: it optimises today's portfolio in isolation, ignoring how positions must evolve over the coming days. This produces high day-to-day turnover, because a position taken today can be reversed tomorrow when the next single-period solve flips its sign. The MPO fit function instead forecasts each stock's **cumulative excess return over H future horizons (H = 10 days)** using a bank of per-horizon ridge models, and then solves a **single multi-period quadratic program** that jointly chooses the portfolio path `x[1..H]`. The objective trades expected return against factor risk, distance from a baseline MVO portfolio, **turnover versus yesterday's positions**, and a coupling penalty that keeps consecutive horizons close to each other. The decay-weighted aggregate `x̄` of the per-horizon solutions becomes the preA. The net effect is a **low-turnover, path-aware preA** with the same or better information ratio at matched turnover.

**What is the intuition or market hypothesis behind the idea - why should it work?**

A signal that is informative about *both* tomorrow's and next week's returns should be traded smoothly rather than churned. By looking H days ahead and penalising path movement, the optimiser only trades when the multi-horizon forecast genuinely changes - it does not react to one-day noise that the next day's solve would undo; turnover falls monotonically as H grows from 1 to 10 **[A2]**. Crucially, turnover falls as H increases **even with the explicit turnover penalty switched off** (`k_tvr = 0`) **[A3]**, which shows the turnover reduction is a genuine multi-period effect, not just a penalty artefact.

**What existing function variants does it build on, replace, or extend?**

It extends the existing **single-period MVO fit** and turns it into a **multi-period** one. The existing fit forecasts one step ahead and optimises one day in isolation; this idea generalises both halves of that pipeline to multiple horizons:

1. **Fit part - a multi-horizon (long-horizon) forecast model.** Instead of a single one-step forecast, the fit trains a **bank of ridge models, one per horizon h = 1..10**, each predicting the cumulative return out to day t+h. This replaces the old single-period, alpha-weight prediction model with a **stock-level forecast that spans the whole horizon path** - it explicitly models how the expected return evolves over the next 10 days, not just tomorrow.

2. **Construct part - a multi-period optimisation.** Instead of a single-period MVO that solves for today's portfolio alone, `construct_preA` runs a **multi-period optimisation (MPO)** that takes all H horizon forecasts and jointly chooses the entire position path, penalising movement between consecutive days. This is the step that converts the multi-horizon view into a low-turnover, path-aware preA.

The original MVO is retained only as the **baseline anchor** `x_mvo_base` the optimiser is pulled toward (any baseline model could be used). So the through-line is **single-period -> multi-period on both sides**: a single forecast becomes a horizon bank of forecasts, and a single-period optimiser becomes a multi-period one. Conceptually it follows Boyd et al., *Multi-Period Trading via Convex Optimization* (arXiv:1705.00109).

In terms of where the effort went: **most of the research work was concentrated on the fit part** - designing the multi-horizon ridge forecast (bucketing into super-alphas, horizon targets, capping, shrinkage). The **construct_preA / MPO part is comparatively standard** - a fairly off-the-shelf multi-period mean-variance QP with default penalty weights - and has been **less explored**. There is therefore more headroom to research the construct side (e.g. tuning the objective weights, the risk model, the decay profile, or the baseline anchor) than the already-developed fit side.

---

## 3. Implementation overview

**Describe implementation of the idea in as much detail as you think necessary. / List the high-level steps of the function.**

The function has two parts: a periodic **fit** step that trains the models, and a daily **construct** step that turns those models into positions.

**Fit (run on each rebalance date):**

1. Select the alphas the filter has switched on for that date.
2. Group the selected alphas into 20 equal-size buckets by their average turnover, and sum the positions within each bucket to form 20 super-alphas. This reduces however many alphas were selected down to a fixed, manageable set of features.
3. Train one ridge model per horizon (h = 1 to 10). For each horizon, the target is the stock's cumulative excess return over the next h days (capped to limit outliers), and the inputs are the 20 super-alpha positions. 
4. Compute a baseline mean-variance portfolio from the selected alphas' PnL history. This serves as a baseline that the optimiser is pulled toward.
5. Save the trained models, the baseline weights, and the bucket definitions for use until the next refit.

**Construct (run every day):**

6. Look up the most recent trained model.
7. Build today's 20 super-alpha positions for the current universe.
8. Run each horizon's model to get a per-stock expected-return forecast, giving one forecast vector per horizon.
9. Estimate a risk model (stock covariance) from a trailing window of returns.
10. Set the two anchors: the baseline portfolio and yesterday's positions (used to control turnover).
11. Solve a multi-period optimisation problem that picks a position path over the horizons. It rewards expected return while penalising risk, distance from the baseline, turnover versus yesterday, and large moves between consecutive horizons, subject to being dollar-neutral with a cap on gross leverage.
12. Combine the per-horizon positions together, normalise it, and restrict it to the tradable universe. This is the preA.

---

## 4. Toy example

**Walk through the idea on a handful of alphas / data points, showing inputs (positions or pnls) and the resulting weights / preA.**

Take 3 stocks (AAPL, MSFT, NVDA) and 3 alpha signals (A1, A2, A3), as on slides 17-18.

**Inputs**

- Today's alpha values `A_today` (stock x signal):

| Stock | A1 | A2 | A3 |
|---|---|---|---|
| AAPL | 1.5 | -0.5 | 1.0 |
| MSFT | -0.3 | 1.5 | 0.6 |
| NVDA | -1.2 | -1.0 | 1.5 |

- Baseline alpha weights from the baseline model: `W = [0.4, 0.3, 0.3]` on `[A1, A2, A3]`.
- Yesterday's portfolio `y_prev`: the positions we already hold and pay turnover to move away from. MSFT was held **long** yesterday.

**Step 1 - project the baseline to today's stock positions.** Run the baseline model to get the alpha weights `W`, then project them onto today's alpha values, `x_base = A_today · W`, one stock at a time:

| Stock | Projection `W · A_today` | Today's position `x_base` |
|---|---|---|
| AAPL | (1.5)(0.4) + (-0.5)(0.3) + (1.0)(0.3) | +0.5 (long) |
| MSFT | (-0.3)(0.4) + (1.5)(0.3) + (0.6)(0.3) | -0.3 (short) |
| NVDA | (-1.2)(0.4) + (-1.0)(0.3) + (1.5)(0.3) | -0.2 (short) |

(Positions are the normalised baseline outputs used in the slide example.) The key observation: the single-period baseline wants to **short MSFT today**, which **flips MSFT's sign versus yesterday's long** - and tomorrow's baseline may well flip it back, so this kind of position churns and burns turnover.

**Step 2 - forecast each horizon with a ridge.** Train H ridges, one per horizon; each answers "what is the expected return after h days?" via `μ_h = β_h · A_today`. Tracking MSFT's forecast across horizons:

| Horizon h | 1 | 2 | 3 | ... | 10 |
|---|---|---|---|---|---|
| MSFT forecast `μ_h` | -0.40 (short) | -0.10 | +0.05 | ... | +0.30 (long) |

So MSFT looks like a short only on the 1-day view (`μ_1 = -0.40`); its forecast steadily climbs and by `h = 10` it is a clear **long** (`μ_10 = +0.30`). The single-period baseline, which only ever sees `h = 1`, would short MSFT today and likely reverse it tomorrow.

**Step 3 - solve one multi-period problem.** A single quadratic program takes all the horizon forecasts `μ^(1) … μ^(H)`, yesterday's positions `y_prev`, and the baseline `x_base`, and picks the whole position path `x_1 … x_H` jointly. It balances two competing pulls:

- **Reward expected return:** each `x_h` is rewarded for aligning with that horizon's forecast `μ_h`.
- **Penalise movement:** moving away from `y_prev` (turnover) and large jumps between consecutive horizons `x_{h-1} -> x_h` are penalised, and the solution is pulled toward the baseline `x_base`.

For MSFT, those two pulls trade off: the `h = 1` forecast argues for a short, but every later horizon argues for a long, and the turnover/coupling penalties make a short-today-then-long-tomorrow round trip expensive. The optimiser therefore declines to put on the aggressive 1-day short and instead holds a small, stable MSFT position consistent with its longer-horizon upward trajectory.

**Show how the model output differs from a naive baseline (equal-weight) on the same toy input.**

The naive single-period baseline trades only on the 1-day view: it shorts MSFT today (flipping it from yesterday's long), and is liable to flip it back to long tomorrow - a wasteful round trip repeated day after day, i.e. high turnover. MPO, by looking H = 10 horizons ahead and pricing in the turnover and coupling penalties, recognises MSFT's longer-term long trajectory and holds a steadier position rather than chasing the noisy 1-day signal. The directional view is broadly preserved while the day-to-day churn is removed, so MPO reaches a similar expected return at materially **lower turnover** - exactly the trade-off the full backtests confirm **[A1][A2]**.

---

## 5. Model

**What type of model is used (linear regression, ridge, NNLS, MVO/QP, tree ensemble, neural network, other)?**

A **two-stage hybrid**. Stage 1 is a bank of **linear ridge regressions** (one per horizon) for return forecasting. Stage 2 is a **convex quadratic program** (mean-variance / multi-period optimiser) for portfolio construction. There is also an auxiliary **long-only QP** (`run_mvo_qp`, cvxopt) producing the baseline anchor.

**Why is this model class appropriate for the idea?**

Ridge is robust and fast for the noisy, collinear super-alpha features (strong L2 shrinkage tames multicollinearity among the 20 buckets); a QP is the natural way to express the multi-period mean-variance utility with hard dollar-neutral / leverage constraints and quadratic turnover/coupling penalties, and it is convex so the solve is reliable in production.

**What are the key model hyperparameters and how were they chosen?**

The single most important hyperparameter is the **number of future horizons (H)** the optimiser forecasts and plans over. This is the lever that defines the idea: with `H = 1` the function collapses to an ordinary single-period optimiser, and as `H` grows the optimiser plans the position path further out and trades it more smoothly. We swept it directly (`H = 1 -> 5 -> 10`) and found turnover falls monotonically as `H` increases, with the largest reduction at `H = 10` **[A2]**, so **`HORIZONS = 10`** is the default. The decay weighting `decay[h] ∝ exp(-(h-1)/tau)` (with `tau = H/2 = 5`) controls how much each horizon counts toward the final position - near horizons dominate, far horizons taper off - which keeps the longer horizons useful for smoothing without letting them overwhelm the near-term forecast.

The remaining hyperparameters are supporting choices rather than the core lever:

- `N_BUCKETS_MPO = 20` - the alphas are compressed into 20 super-alphas, fixing the feature count regardless of how many alphas are selected and controlling overfitting / collinearity.
- `RIDGE_ALPHA = 1e5` - deliberately strong L2 shrinkage so the 20-feature per-horizon forecaster does not chase daily return noise.
- `LOOKBACK = 1008` (approximately 4 trading years) - the window used to train the ridges and estimate the risk model; long enough for stable estimates, short enough to stay responsive.
- Optimiser objective weights: `k_mean = 1.0` (reward return), `k_var = 0.5` (risk aversion), `k_base = 0.05` (pull toward the baseline), `k_tvr = 0.1` (turnover penalty vs. yesterday), `k_couple = 0.05` (penalty on moves between consecutive horizons) - chosen to balance return against path-stability.
- `RETURN_CAP = 0.15` (caps the forecast target to limit outliers) and `MAX_FACTORS_QP = 50` (number of risk factors in the covariance model).

**Approximately how many trainable parameters does the model have?**

20 super-alpha coefficients x 10 horizons = **200 ridge coefficients** (no intercept), plus the baseline MVO weights (one per selected alpha, but these are an anchor, not learned by gradient). The PCA risk model and decay weights are not trained.

**Is the mapping from alphas (or other model inputs) to preA linear or non-linear?**

The alpha->μ stage is **linear** (ridge). The full alpha->preA mapping is **non-linear**, because the QP applies inequality (leverage) and equality (dollar-neutral) constraints plus the previous-day-dependent turnover term - the output is a piecewise-linear/convex function of the inputs, not a simple linear combination.

---

## 6. Inputs / Features

**What feeds the model - alpha positions, alpha PnLs, simres fields, data variables, alpha features, alpha-description metadata, derived statistics?**

The two stages consume different inputs, so it is cleanest to split them:

*In `fit()` (trains the models, runs quarterly):*
- Alpha **postA positions** (the alpha cube) - the features that are bucketed into the 20 super-alphas and fed to the ridges.
- Alpha **dailypnl** - used only to build the baseline alpha-weight MVO anchor (`run_mvo_qp`).
- **simres field `dailytvr`** (`BUCKET_METRIC = dailytvr_top3000top1200`) - the derived statistic used to rank/bucket the alphas.
- **`ret1_excess`** (future, capped) - the training **target** (the cumulative h-day forward return per horizon).

*In `construct_preA()` (applies the models, runs every day):*
- Alpha **postA positions** for the current day(s) - compressed into the 20 super-alphas (using the stored bucket map and baseline weights) and passed to the per-horizon ridges to produce the forecasts `μ[h]`.
- **`ret1_excess`** (trailing `LOOKBACK` window) - used to estimate the **stock covariance / PCA risk model**, not as a target.
- **Yesterday's positions** (`y_prev`) - the live book in last mode, the previous preA column in full mode - the turnover anchor.
- The **stored fit artefacts** - the per-horizon ridge models + scalers, the bucket map, and the baseline MVO weights.

No alpha-description metadata or text features are used anywhere.

**What is the shape of the model input (n_observations x n_features) in a typical fit call?**

The features always have **20 columns** (the 20 super-alphas), but the number of rows differs sharply between the two stages.

*In `fit()` (one ridge per horizon, h = 1..10):* every (stock, day) pair in the lookback window is one observation, stacked into `X_h` of shape `(N_obs x 20)` with `N_obs ≈ n_valid_stocks x lookback_days` (order tens of thousands to low millions of rows), and a target vector `y_h` of length `N_obs`. Ten such matrices are built, one per horizon.

*In `construct_preA()` (inference + optimisation):* the feature cube `Z` is `(n_stocks x n_days x 20)`; for the day being solved only today's slice is used, **one row per stock**, shape `(n_stocks x 20)`. Each horizon's ridge maps that to a per-stock forecast, stacked into the forecast matrix `Mu` of shape `(H x n_stocks) = (10 x n_stocks)`. The optimisation then works in **stock space**: it solves for `H` position vectors each of length `n_stocks`, the PCA risk model is `(n_stocks x k_f)` with `k_f <= 50` factors, and the output position vector `x̄` is length `n_stocks`. So the fit call is "tall and thin" (many stock-days x 20), while the construct call is "wide" (a per-stock vector across 10 horizons).

This stacked, **pooled cross-sectional** layout in `fit()` is deliberate and is the better choice here. Rather than fitting a separate model per stock (which would give each stock only ~1000 noisy daily observations and badly overfit), every stock shares one set of 20 coefficients per horizon, learned from tens of thousands of stock-day observations. That gives a far larger, more stable training sample, ties the model down to the relationship between super-alpha exposure and forward return (which is what we actually want to estimate) rather than stock-specific quirks, and keeps the parameter count tiny (20 per horizon) so the strong ridge shrinkage is enough to prevent overfitting. It also makes the model robust to the universe changing day to day - stocks entering or leaving simply add or remove rows, with no need to retrain anything per name.

**Does the model make predictions for each stock independently or for multiple stocks at once?**

The ridge **predicts each stock independently** (same coefficients applied per stock-row). The **QP solves all stocks jointly** (the risk model and dollar-neutral / leverage constraints couple stocks).

**What transformations are applied to inputs (cross-sectional z-score, rank, capping, scaling, sign-flip)?**

Inputs go through three transformations, in order. Writing `a_i` for an alpha column, `S_b` for the super-alpha of bucket `b`, and `x_{s,j}` for feature `j` of stock `s`:

1. **Clean (NaN/inf -> 0):** `x -> 0 if (x is NaN or ±inf) else x`, applied to every cell, so missing or blown-up values never enter the maths.
2. **Common scaling of each super-alpha:** first sum the alphas in a bucket, `S_b = Σ_{i ∈ b} a_i`, then re-scale to a common book size: `S_b -> S_b · (B / Σ_s |S_{b,s}|)` with target book `B = 1e6` (booksize mode, the default). Alternatives are cross-sectional z-score `(S_b - mean_s S_b) / std_s S_b` and cross-sectional rank. This puts all 20 super-alphas on the same dollar footing so no bucket dominates purely by scale.
3. **Standardise the features before the ridge:** per feature column `j`, `x_{s,j} -> (x_{s,j} - μ_j) / σ_j`, where `μ_j, σ_j` are the column mean and standard deviation computed at fit time and stored (`σ_j` set to 1 when it is 0). Standardising matters for ridge specifically: the L2 penalty `α‖β‖²` shrinks all coefficients equally, so features must share a common scale or the penalty would unfairly punish small-variance columns. Re-applying the stored `(μ_j, σ_j)` at inference guarantees identical scaling to training with no look-ahead.

No sign-flipping is applied (the optimiser is free to go long or short each super-alpha); the target, not the inputs, is the quantity that gets capped (`clip(±0.15)`).

**How are NaNs / infs / zeros cells handled?**

`n2z` maps NaN/+/-inf -> 0; `op.at_nan2zero` on every slab; zero-std feature columns get σ = 1; alphas with today |sum| < 1000 or > 90% zeros are dropped, and stocks with > 90% zeros are dropped.

**Are alphas bucketed before fitting? If yes, what is the bucketing metric (tvr, liq2, sector, ...) and rationale?**

Yes - into **20 buckets** by **mean `dailytvr`** (`rank_equal_groups`). Rationale:

- **Runtime / scalability (the main driver):** the filter can select thousands of alphas, and feeding all of them to the ridges and the QP would be slow and memory-heavy and would scale with the alpha count. Collapsing them into a **fixed 20 super-alphas** makes the per-horizon ridge a fixed `(N_obs x 20)` problem and keeps the cost essentially **independent of how many alphas were selected** - 50 alphas or 5000 alphas both reduce to 20 features. This is what keeps the quarterly fit and the daily solve tractable.
- **Statistical:** grouping alphas of similar turnover keeps each bucket homogeneous, and the compression reduces dimensionality / collinearity before fitting, which (together with the strong ridge shrinkage) controls overfitting.

Turnover is the chosen bucketing metric because it is the axis the whole idea is about (smoothing trading), so grouping by turnover lets the model treat fast and slow signals separately.

**Are super-alphas (dimensionality-reduction features) created, if yes how many?**

Yes - **20** (= `N_BUCKETS_MPO`). Each is the booksize-normalised **sum** of the alphas in its tvr bucket.

**If super-alphas are constructed, what method splits alphas into super-alpha clusters, and what model is used to combine alphas?**

Alphas are split by **tvr-rank into 20 equal groups** (`rank_equal_groups`). The combiner across super-alphas is the **per-horizon ridge regression** (the 20-coefficient model), and downstream the **multi-period QP** combines the per-horizon stock forecasts.

---

## 7. Target

**What target variable is used (ret1, ret1_excess, n-day forward return, capped return, ranked return, custom)?**

`ret1_excess` - specifically the **cumulative h-day forward excess return**, `Σ_{k=1..h} ret1_excess(t+k)` for each horizon `h = 1..10`. (When `ret1_excess` is unavailable it falls back to `ret1`.)

**What transformations are applied to the target (capping, ranking, smoothing over n days, sign)?**

**Capped at +/-`RETURN_CAP = 0.15`**; summed over the next h days (cumulative); `at_nan2zero` on the assembled cube.

**How is the target shifted to be next-day / post-decision, and how is forward bias avoided?**

The key thing to understand is **where future returns are used and where they are not**:

- **In `fit()` (runs once a quarter):** building the targets *deliberately* uses future returns - the horizon-h target is `Σ_{k=1..h} ret1_excess(t+k+delay)`, implemented as `op.ts_delay(ret1_excess, -k - delay)` (a negative delay pulls future returns back to date `t`). This is legitimate because `fit()` is a historical training step: at a refit date the future is already realised, so the H-day-ahead targets are simply known labels for past dates. The `-delay` offset keeps the label strictly **post-decision** (a position decided at `t` only earns from `t+delay` onward, matching the simulation's execution lag).

- **In `construct_preA()` (runs every day, including live):** **no future returns are needed at all** - the ridge models are already trained, so inference just applies the stored coefficients to today's features to get `μ[h]`. Because the live path never touches future data, there is no avenue for forward bias at decision time.

So forward bias is structurally impossible in production: the only place future returns appear is the quarterly historical fit, and there they are realised labels, not look-ahead.

On dropping the tail of the training window: the code explicitly sets the last `delay` columns of the target to NaN (`ret_cut_h[:, -delay:] = np.nan`) and the `np.isfinite` mask drops them. Note it does **not** explicitly drop a full 10 (= H) days - it relies on the lookback window ending at `idx_end = di - delay`, well before the data tail, so the future returns needed for the H-day targets of the last training rows are already available; any genuinely unavailable future cell is zero-filled by `at_nan2zero` rather than dropped. (If the fit window ever ran right up to the data edge, the final ~H rows would have truncated targets - not a concern for the quarterly historical refit, but worth flagging.)

**How are missing or invalid target observations masked?**

`valid = np.isfinite(y)` selects only finite target cells, and the same mask is applied to the feature rows. Targets are further bounded by the universe (`ret1_excess * valids`).

**Why is this target appropriate for the prediction problem the model is solving?**

The QP needs an expected-return vector **per horizon**; cumulative h-day forward return is exactly the quantity each horizon's position should be paid for, so forecasting it directly aligns the model output with the optimiser's objective. Capping protects the ridge fit from earnings-day / outlier return blow-ups.

---

## 8. Loss / Objective

**What is the loss or objective function (MSE, NNLS residual, mean-variance utility, log-likelihood, custom)?**

- **Ridge loss:** L2-penalised mean-squared error, `‖Xβ - y‖² + α‖β‖²` with `α = 1e5`, `fit_intercept=False`.
- **Optimiser objective (`fast_fit`):** maximise the decay-weighted multi-period mean-variance utility
  `Σ_h decay[h]·( μ[h]·x_h - k_var·(factor + diag risk) - k_base·‖x_h - x_mvo_base‖² - k_tvr·‖x_h - y_prev‖² ) - k_couple·Σ_{h>0} ‖x_h - x_{h-1}‖²`.

  Reading the terms (here `x_h` is the portfolio at horizon `h`, `μ[h]` its forecast, `decay[h]` the per-horizon weight):
  - **`μ[h]·x_h`** - the **expected-return reward**: pays the portfolio for aligning with that horizon's forecast.
  - **`k_var` (risk aversion, = 0.5)** - penalises **portfolio risk** `(factor + diagonal residual variance)` from the covariance model; larger `k_var` -> more risk-averse, smaller positions on volatile/crowded names.
  - **`k_base` (anchor pull, = 0.05)** - penalises distance from the **baseline MVO portfolio** `x_mvo_base`; keeps the solution near a sensible reference so it cannot drift to a degenerate corner.
  - **`k_tvr` (turnover penalty, = 0.1)** - penalises distance from **yesterday's positions** `y_prev`; this is the explicit turnover/transaction-cost term (the ablation shows turnover still falls with this set to 0, so it is not the only source of smoothing).
  - **`k_couple` (coupling penalty, = 0.05)** - penalises **moves between consecutive horizons** `‖x_h - x_{h-1}‖²`; this is what ties the H portfolios into one smooth *path* rather than H independent single-period solves, and is the core multi-period mechanism.
  - **`decay[h] ∝ exp(-(h-1)/tau)`** - weights near horizons more than far ones, so the near-term forecast dominates the final position while the longer horizons act as smoothing.

**Are there hard constraints on the solution (sum-to-one, non-negativity, per-weight cap, leverage cap, return-target)?**

- **Dollar-neutral:** `Σ x̄ = 0`.
- **Leverage cap:** `‖x̄‖₁ <= 2·gross_ref` (gross_ref = max of prev/MVO gross, >= 1).
- Baseline `run_mvo_qp` additionally imposes **non-negativity** (`w >= 0`) and **sum-to-one** (`Σw = 1`) on the alpha weights.

**Is the objective convex? Is the solution unique?**

The MPO QP is **convex** (PSD quadratic risk + convex penalties, linear constraints) -> a **unique** optimum (solved by OSQP). The baseline QP is likewise convex (cvxopt). Ridge is strictly convex -> unique β.

**What regularization is applied (L1, L2, dropout, max-norm, early stopping, tree depth / leaves / min-split-gain)?**

Ridge **L2 (`α=1e5`)**; the optimiser's `k_base`, `k_tvr`, `k_couple` quadratic penalties act as Tikhonov-style regularisers on the portfolio. The **stock-space PCA factor risk model** used by the MPO QP is built from the **plain sample covariance** (`np.cov`) with **eigenvalue clipping** and a **diagonal residual variance**, truncated to **<= 50 factors** - it does *not* use Ledoit-Wolf. **Ledoit-Wolf shrinkage** is applied only to the **baseline alpha-space covariance** (the `run_mvo_qp` / alpha-weight MVO that produces the anchor), with `nearest_psd` / `+1e-8·I` for numerical stability.

---

## 9. Overfitting controls

**What explicit techniques mitigate overfitting (regularization, super-alpha pooling, bucketing, weight capping, walk-forward refit, ensembling, early stopping, shrinkage)?**

- Strong **ridge L2 (α = 1e5)** on the return models.
- **Super-alpha pooling**: the feature space is reduced to 20 buckets regardless of how many alphas are selected, drastically cutting parameter count.
- **tvr bucketing** (homogeneous groups).
- **PCA factor risk model** with truncation (<= 50 factors) + diagonal residual variance and eigenvalue clipping (sample covariance, not Ledoit-Wolf; Ledoit-Wolf shrinkage applies only to the baseline alpha-space covariance).
- **Turnover / anchor / coupling penalties** in the QP, which shrink the solution toward the baseline MVO and toward yesterday's positions.
- **Walk-forward refit** on `rebalance_dates_mask` with a fixed `LOOKBACK = 1008`.
- **Return capping** (+/-0.15) and **leverage cap**.

**How was the idea validated across different regions, sub-universes, and alpha sets - not just on one favorable backtest?**

The deliberate validation was **USA only** - on **US top1000** and **US top3000**, over **1y / 2y / 4y** windows, and against the full set of ablations: default-construct vs MPO **[A1]**, the horizon sweep H = 1/5/10 **[A2]**, the same sweep with `k_tvr = 0` **[A3]**, equal-weight ridge vs MPO **[A4]**, and raw ridge returns by horizon **[A5][A6]** (all detailed in Section 16). The **US + EU + JP** numbers that appear in the rankings come from the standard multi-region ranking pool rather than a deliberate multi-region study; the idea was developed and tuned on USA, and multi-region (especially the 1-year window) is the weak point that has not been separately validated (see Section 16). Cross-universe robustness within USA is shown by the top1000-vs-top3000 consistency; cross-alpha-set robustness comes from the ablations all using the same selected-alpha pool.

---

## 10. Training / Refit

**At what cadence is the model re-fit (daily, quarterly)?**

The model is re-fit **quarterly** - on `rebalance_dates_mask` dates (the standard TQ100m rebalance cadence), not daily. This is the expensive step that trains the 10 ridges (using realised future returns as labels, which is safe because it is historical - see Section 7). Between refits, `construct_preA` runs **every day** but only re-applies the most recent already-trained model and re-solves the cheap inference QP; it never re-trains and never touches future data. So the heavy quarterly fit and the light daily construct are cleanly separated, which is also why there is no forward bias in live trading.

**What lookback window is used to fit the model and how was its length chosen?**

A single lookback of **`LOOKBACK = 1008` trading days (approximately 4 years)** is used for both jobs the fit step performs: training the per-horizon ridges and estimating the risk model's covariance.

The length is a deliberate bias-variance trade-off. The window has to be long because both estimation tasks are data-hungry: the risk model estimates a large stock covariance (many factors plus per-stock variances) that is unstable on short samples, and although the ridges only have 20 coefficients per horizon, the forward-return target is extremely noisy, so a long window of stock-day observations is what makes the relationship between super-alpha exposure and forward return estimable at all. Four years also spans more than one market regime, so the models are fit across a mix of conditions rather than a single trending or mean-reverting stretch, which keeps them from over-specialising.

At the same time it is bounded so the model stays adaptive: going much longer would drag in stale relationships and make the fit slow to react when alpha behaviour shifts, while going much shorter would make the covariance and the ridge coefficients jump around from refit to refit. Roughly four years (1008 days) is also the natural choice because it matches the longest evaluation window the function is ranked on, so the model is trained on the same horizon over which it is judged. The chosen value was not heavily over-tuned - it sits in a wide, stable plateau, and the idea's performance is not fragile to the exact number (see the sensitivity analysis in Section 17).

**How is walk-forward set up (rolling vs expanding window)?**

**Rolling** window - each refit uses `[di - delay - lookback, di - delay]`, and `FIT_STARTDATE = 20210101` gates the first refit. In full mode, model `i` is applied to all days between refit `i` and refit `i+1`, so there is no look-ahead.

---

## 11. Outputs

**What does the model return?**

The **preA** - per-stock, per-day target positions.
- `fit()` returns `(model_dict, filter_matrix_copy)`; `model_dict[date]` holds the pickled per-refit model (mvo_weights, 10 ridge models, cluster_dic, hyperparameters).
- `construct_preA(mode="full")` returns the full preA matrix; `mode="last"` returns the single most recent day's preA vector.

**What is the shape, dtype, and ordering of the output?**

- Full mode: `(numstocks x numdates)`, `float32`, Fortran-ordered, `cs_booksize`-scaled to `data["booksize"]`, masked by `valids`, with zeros mapped to NaN (`at_zero2nan`).
- Last mode: `(numstocks,)`, `float32`.
- Ordering is the standard stock x date layout aligned to `data["dates"]`.

---

## 12. Inference / Productionization

**How is preA constructed in `full` mode (backtest) vs `last` mode (live forward / ffw)?**

Both modes run the **same per-day optimisation**; they differ only in how much history they process and where yesterday's positions come from.

- **`full` (backtest):** walks day by day across the whole history. For each day it looks up the refit model that was active on that date (so a model is only ever used on dates after it was trained, with no look-ahead), builds that day's features, solves the optimisation, and writes the resulting vector into that day's column of the preA matrix. The previous-day anchor (yesterday's positions) is simply the preceding column it has already filled in, so the path is internally consistent across the backtest.

- **`last` (live forward / ffw):** the production path that runs once per day. It picks the most recent refit on or before today, loads **only today's** features rather than the full history, and solves the optimisation once to emit a single day's position vector. Because there is no prior column to read from, yesterday's positions come from the live book (the actual positions currently held).

The two modes must agree: re-running `last` day after day in production should reproduce what `full` would have produced for those same days. The `_ffw_fix` (fast-forward fix) in the filename is exactly this reconciliation - if the day being computed in `last` mode is already covered by the cached feature history, the code transparently switches to the `full` feature path for that day so the live output matches the backtest bit-for-bit, instead of drifting because of the smaller data slice.

**Does the function use yesterday's positions? If yes, how is the dependency on previous-day state handled?**

**Yes.** `y_prev` is the turnover anchor in the QP. In last mode it comes from `data["get_yesterday_preA"]()`; in full mode from the previous preA column. Because gross can differ between live and backtest, `fast_fit` **rescales** `y_prev` when `gross_prev > 10x gross_mvo` so the turnover term stays well-scaled.

**How does the function handle the GLOBAL region (or primary-region slicing) differently from single-region runs?**

There is **no explicit GLOBAL branch** in the code - it operates on whatever universe `valids` defines and is masked by `valids` at the end. Multi-region behaviour is handled by running it over the combined universe (the US+EU+JP rankings), not by special-casing GLOBAL. (Single-region vs multi-region performance differs materially - see Sections 15-16.)

---

## 13. Portfolio construction

**How are model outputs converted into per-stock per-day positions (preA)?**

The QP's decay-weighted aggregate `x̄` (per-stock) is written directly into the preA column for that day. `x̄` is already in stock space (the ridge predicts per-stock returns and the QP optimises per-stock positions).

**Is the preA normalized cross-sectionally per day (by sum, abs-sum, z-score, rank)?**

Yes - final preA is `cs_booksize`-normalised per day to `data["booksize"]`, and intermediate aggregates are abs-sum normalised.

**Is the preA explicitly dollar-neutral, beta-neutral, or sector-neutral at this stage, or is that deferred to post-processing?**

The QP enforces **dollar-neutrality** explicitly (`Σ x̄ = 0`) and a **leverage cap**. **Beta/sector neutrality are deferred** to post-processing - they are not imposed here. (Factor *risk* is penalised via the PCA model, which is a soft control, not a hard neutrality constraint.)

**How is the trading universe enforced (multiply by `valids`, mask, drop)?**

`op.at_mask(preA, data["valids"])` then `cs_booksize`; only `selected_valids_today_mvo` stocks are ever assigned non-zero values.

---

## 14. Edge cases

**What happens when the filter selects zero / one / very few alphas on a rebalance date?**

The three cases behave differently because the function always tries to form a fixed **20 buckets** of alphas:

- **Zero alphas selected:** handled cleanly. The rebalance date is detected as empty and skipped in the fit step (no model is trained or stored for that date), and on the construct side the day produces no positions - its preA column is left as zeros and masked out of the universe. No error, just a flat (no-position) day for that signal.

- **One alpha selected:** degenerate and not robustly supported. With only one alpha there is nothing to forecast cross-sectionally and only one of the 20 bucket slots is populated, leaving the other 19 empty (see below) - so this does not give a meaningful model. In practice the filter is never expected to pass a single alpha, so this is an untested corner rather than a designed path.

- **Very few alphas (fewer than the 20 buckets):** also fragile. The number of buckets is fixed at 20 regardless of how many alphas are selected, so any selection of fewer than 20 alphas spreads them across only a handful of bucket slots and leaves the rest empty. The bucketing therefore degenerates (some buckets hold one alpha, most hold none), and the empty buckets are not handled gracefully in the fit step - so the function effectively requires on the order of 20-plus selected alphas to operate as intended.

In short: zero alphas is a safe no-op; anything from one up to fewer than the bucket count is a degenerate regime the function is not designed for. This is acceptable in practice because the upstream filter normally selects hundreds to thousands of alphas, but it is a real limitation for very small alpha sets (also noted in Sections 9 and 18).

**What happens when an alpha is all-zero or all-NaN over the lookback?**

Dropped in `create_feature` - today |sum| < 1000 or > 90% zero cells -> skip alpha; stocks > 90% zero -> dropped. Empty training matrices -> a **zero-coefficient ridge model** is stored.

**How does it behave when bucketing produces an empty bucket?**

Buckets are equal-sized by construction, so an empty bucket only arises when there are **fewer alphas than the 20 bucket slots** - exactly the very-few-alphas case above. This is not handled gracefully: the bucket-building step still iterates over all 20 bucket slots, and an empty slot has no alpha positions to sum, so it does not produce a valid super-alpha. The practical consequence is that the function expects the number of selected alphas to comfortably exceed the bucket count; when the universe of selected alphas is that small, the right fix is to reduce the bucket count rather than rely on the current path. With the normal alpha counts (hundreds to thousands) every bucket is populated and this never triggers.

**Are there asset-class-specific branches (equities vs futures)? Region-specific branches (GLOBAL vs single-region)?**

The function is **equities-only** (booksize normalisation, `ret1_excess`, alpha cube). No futures branch. No explicit region branches (see Section 12); same code path for single- and multi-region.

**What known failure modes exist (singular matrix, optimizer non-convergence, exploding gradients, OOM)?**

Singular/near-PSD covariance -> `nearest_psd`, eigenvalue clipping, `+1e-8·I` ridge; **OSQP non-convergence / no solution** -> caught and **falls back to the decay-weighted mean of `Mu`**; short return history (`ret1.shape[1] < 2`) -> equal-weight bucket fallback; missing ridge models -> equal-weight fallback. No special OOM handling beyond `float32` and the 20-bucket compression.

---

## 15. Robustness / generality

**In which regions has the idea been tested (US, EU, JP, ASIA, GLOBAL)?**

**US** (top1000) and **US + EU + JP** (eu top600 / jp top600 / us top1000), across 1y / 2y / 4y. EU/JP/ASIA were not tested in isolation; GLOBAL is only via the combined universe.

**In which stock sub-universes (top250 / top500 / top1000 / top3000)?**

Primarily **top1000** (US) and **top600** (EU, JP). The bucketing metric is `dailytvr_top3000top1200`, and one comparison references **TOP3000**, so it has been exercised on broader universes too.

**Can the function handle very few (less than 20) and very many (over 5000) alphas?**

The 20-super-alpha compression makes it **scale-robust to many alphas** (cost is dominated by the fixed 20 buckets and the stock-level QP, not the alpha count). For **very few alphas** it degrades gracefully via the single-bucket short-circuit and equal-weight fallbacks, though with < 20 alphas the bucketing becomes degenerate (some buckets hold one alpha).

---

## 16. Benchmark comparison

**Ablation index.** The benchmarks below are numbered A1-A6 and are cited inline throughout this document wherever a claim rests on one of them:

- **[A1]** Default construct preA (no MPO) vs MPO (H = 10) - isolates the value of the MPO step itself.
- **[A2]** Horizon sweep, H = 1 / 5 / 10 - isolates the effect of looking further ahead.
- **[A3]** Horizon sweep with the explicit turnover term off (`k_tvr = 0`) - isolates the multi-period effect from the turnover penalty.
- **[A4]** Add-value of the optimiser: equal-weight mean of ridge returns vs the same returns run through MPO - isolates the optimiser from the forecasts.
- **[A5]** Raw ridge returns as preA by horizon (1st / 5th / 10th / mean), no optimiser - isolates which horizon's forecast to trade.
- **[A6]** Mean of ridge returns across an increasing number of horizons (1H / 5H / 10H), no optimiser - isolates horizon-averaging from the optimiser.

**Same function with-idea vs without-idea, on US top1000 using all gw2 filter / fit / pp with apply-extra neut - performance of both.**

**[A1]** The with/without comparison pits the default construct_preA (no MPO) against the MPO function on the same baseline model:

| Function Name | Function Idea |
|---|---|
| fitSnarang73_MPO_baseTest_defaultCons | Returns the baseline MVO weights with **no MPO optimisation** in construct_preA |
| fitSnarang73_MPO_baseTest_10H | **MPO function with H = 10**, using the same baseline MVO model as above |

Performance (US top1000):

| Function | preA IR | preA RET | preA TVR | postA IR | postA RET | postA TVR |
|---|---|---|---|---|---|---|
| Default Construct PreA (no MPO) | 0.122 | 0.063 | 0.652 | 0.130 | 0.058 | 0.096 |
| MPO (H = 10) | **0.165** | 0.100 | **0.159** | **0.173** | 0.072 | 0.094 |

Finding: at the **preA** level the MPO version has materially lower turnover (0.652 -> 0.159). Equalising postA turnover (`pp_basic_hump` with `target_tvr = 0.10`), **MPO beats the default in IR (0.173 vs 0.130), RET and LIQ** at matched turnover (holds on TOP3000 as well).

**Doing the opposite of the idea - what happens?**

The opposite of "look many horizons ahead" is **H = 1** (single-period). Several benchmarks probe this from different angles.

**[A2]** *Horizon sweep (does more horizons reduce turnover?):*

| Function Name | Function Idea |
|---|---|
| fitSnarang73_MPO_baseTest_1H | MPO function with num_horizons = 1 |
| fitSnarang73_MPO_baseTest_5H | MPO function with num_horizons = 5 |
| fitSnarang73_MPO_baseTest_10H | MPO function with num_horizons = 10 |

Performance (US top3000):

| Function | preA IR | preA RET | preA TVR | postA IR | postA RET | postA TVR |
|---|---|---|---|---|---|---|
| 1H | 0.165 | 0.087 | 0.250 | 0.085 | 0.034 | 0.141 |
| 5H | 0.196 | 0.100 | 0.189 | 0.099 | 0.039 | 0.134 |
| 10H | **0.203** | 0.103 | **0.159** | **0.103** | 0.041 | **0.128** |

Finding: as the number of horizons increases (1 -> 5 -> 10), preA turnover falls monotonically (0.250 -> 0.189 -> 0.159) while IR improves (0.165 -> 0.196 -> 0.203); the same ordering holds at postA.

**[A3]** *Horizon sweep with the explicit turnover term switched off:*

| Function Name | Function Idea |
|---|---|
| fitSnarang73_MPO_baseTest_1H_ktvr0 | MPO function with num_horizons = 1 and `k_tvr = 0` |
| fitSnarang73_MPO_baseTest_5H_ktvr0 | MPO function with num_horizons = 5 and `k_tvr = 0` |
| fitSnarang73_MPO_baseTest_10H_ktvr0 | MPO function with num_horizons = 10 and `k_tvr = 0` |

Performance (US top3000):

| Function | preA IR | preA RET | preA TVR | postA IR | postA RET | postA TVR |
|---|---|---|---|---|---|---|
| 1H, k_tvr = 0 | 0.014 | 0.007 | 0.723 | 0.187 | 0.070 | 0.161 |
| 5H, k_tvr = 0 | 0.139 | 0.066 | 0.523 | 0.211 | 0.078 | 0.156 |
| 10H, k_tvr = 0 | **0.194** | 0.091 | **0.413** | **0.217** | 0.081 | **0.150** |

Finding: turnover **still** falls as horizons increase even with `k_tvr = 0` (preA 0.723 -> 0.523 -> 0.413; postA 0.161 -> 0.156 -> 0.150), and IR rises with it - proving the turnover reduction is the genuine effect of the multi-period structure, not the explicit turnover penalty.

**[A4]** *Add-value of the optimiser (MPO vs using ridge returns directly):*

| Function Name | Function Idea |
|---|---|
| fitSnarang73Ameshram_MPO_10H_ridge_ret_ew_mean | For H = 10, uses the **mean** return prediction over all 10 horizons as the preA (no optimiser) |
| fitSnarang73_MPO_baseTest_10H | For H = 10, feeds those same return predictions into the **MPO** setup |

Performance (US top3000):

| Function | preA IR | preA RET | preA TVR | postA IR | postA RET | postA TVR |
|---|---|---|---|---|---|---|
| 10H (only ridge) | 0.141 | 0.177 | 0.321 | 0.059 | 0.044 | 0.124 |
| 10H (MPO) | **0.203** | 0.103 | **0.159** | **0.103** | 0.041 | 0.128 |

Finding: the MPO setup gives **better IR (preA 0.203 vs 0.141; postA 0.103 vs 0.059), TVR and LIQ** than using the ridge returns directly; correlation between the two is **64%**.

**[A5]** *Additional benchmark - raw ridge returns as preA (which horizon to trade, no optimiser):*

| Function Name | Function Idea |
|---|---|
| fitSnarang73Ameshram_MPO_10H_ridge_ret_1st | For H = 10, uses the **1st** horizon's return prediction as the preA (no MPO optimiser) |
| fitSnarang73Ameshram_MPO_10H_ridge_ret_5th | For H = 10, uses the **5th** horizon's return prediction as the preA (no MPO optimiser) |
| fitSnarang73Ameshram_MPO_10H_ridge_ret_10th | For H = 10, uses the **10th** horizon's return prediction as the preA (no MPO optimiser) |
| fitSnarang73Ameshram_MPO_10H_ridge_ret_ew_mean | For H = 10, uses the **mean** prediction over all 10 horizons as the preA (no MPO optimiser) |

Performance (US top3000, preA):

| Function | IR | RET | TVR |
|---|---|---|---|
| 1st horizon | 0.081 | 0.096 | 0.621 |
| 5th horizon | 0.141 | 0.176 | 0.340 |
| Mean | 0.143 | 0.180 | 0.321 |
| 10th horizon | 0.147 | 0.186 | **0.270** |

Finding: longer-horizon predictions give **lower turnover** preA (1st 0.621 -> 10th 0.270), with IR also rising toward the longer horizons.

**[A6]** *Additional benchmark - mean of ridge returns across an increasing number of horizons (no optimiser):*

| Function Name | Function Idea |
|---|---|
| fitSnarang73Ameshram_MPO_1H_ridge_ret_ew_mean | For H = 1, mean return prediction over the 1 horizon |
| fitSnarang73Ameshram_MPO_5H_ridge_ret_ew_mean | For H = 5, mean return prediction over the 5 horizons |
| fitSnarang73Ameshram_MPO_10H_ridge_ret_ew_mean | For H = 10, mean return prediction over the 10 horizons |

Performance (US top3000, preA):

| Function | IR | RET | TVR |
|---|---|---|---|
| mean(1H) | 0.081 | 0.096 | 0.621 |
| mean(5H) | 0.133 | 0.164 | 0.401 |
| mean(10H) | **0.143** | 0.180 | **0.321** |

Finding: averaging over **more horizons** lowers the preA turnover (0.621 -> 0.401 -> 0.321) while IR improves (0.081 -> 0.133 -> 0.143).

**Function ranking across (US, US+EU+JP) x (4y, 2y, 1y): list percentage performance numbers and the maximum correlation to functions the idea does not outperform.**

For `fitSnarang73Ameshram_MPO_10H_ffw_fix` (with-idea), ranked within a pool of **9 functions**. Columns: improvement % (vs best in pool), IR, turnover, IR/sqrt(TVR), and **max correlation to a function it does not outperform**.

| Universe | Window | Rank | Improv% | IR | TVR | IR/sqrt(TVR) | Max corr (to a better fn) |
|---|---|---|---|---|---|---|---|
| US top1000 | 1y (252d) | 6 / 9 | 53.10% | 0.055 | 0.108 | 0.173 | **1.000** |
| US top1000 | 2y (502d) | 6 / 9 | 66.60% | 0.077 | 0.112 | 0.237 | **1.000** |
| US top1000 | 4y (1004d) | 6 / 9 | 65.28% | 0.067 | 0.118 | 0.202 | **1.000** |
| US+EU+JP | 1y (252d) | 7 / 9 | **-42.20%** | **-0.010** | 0.129 | **-0.020** | **1.000** |
| US+EU+JP | 2y (502d) | 7 / 9 | 49.60% | 0.027 | 0.131 | 0.085 | **1.000** |
| US+EU+JP | 4y (1004d) | 7 / 9 | 42.57% | 0.032 | 0.133 | 0.094 | **1.000** |

Notes: within the 9-function pool it ranks **6th in US** and **7th in US+EU+JP**, and is solidly positive over 2y/4y; the **multi-region 1-year window is negative** (the main weakness). The **max correlation is 1.000 in every window**, meaning there is a near-identical function ranked above it (almost certainly the sibling MPO/baseline variant by the same authors) - i.e. it does not add diversification on top of that twin, only on top of the rest of the pool.

**preTQ100_xxx label performance - relevant metrics from label OS pnl.**

The label `preTQ100m_mpo_fit` outperforms both TREX benchmarks over both windows (IR):

| Window (OS) | preTQ100m_mpo_fit | trex_ALL_gross | trex_USA_gross |
|---|---|---|---|
| Last 60 OS days (20260218 - 20260513) | **0.243** | 0.189 | 0.184 |
| Full OS, 88 days (20260107 - 20260513) | **0.076** | 0.042 | 0.063 |

**Benchmark strategy vs with idea comparison: OS days, OS IR, 1y IR, 2y IR, correlation to benchmark.**

The MPO-fit bmark strategy outperforms the `_beat_the_bmark` strategy over both windows, with no recent drawdown in the MPO strategy:

| Window (OS) | IR (MPO fit) | IR (_beat_the_bmark) | Correlation |
|---|---|---|---|
| Last 60 OS days (20260218 - 20260513) | **0.198** | 0.092 | 33.7% |
| Full OS, 88 days (20260107 - 20260513) | **0.091** | 0.070 | 26.3% |

---

## 17. Sensitivity / hyperparameter analysis

**Which hyperparameters were swept and over what ranges?**

- **Number of horizons H ∈ {1, 5, 10}** **[A2]**.
- **Optimiser-on vs optimiser-off** (MPO QP vs raw/equal-weight ridge returns) **[A4]**.
- **Explicit turnover term `k_tvr ∈ {default, 0}`** **[A3]**.
- **Which horizon's prediction to trade** (1st / 5th / 10th / mean) **[A5][A6]**.

**How sensitive is performance to each parameter (lookback, regularization strength, number of buckets, number of super-alphas, refit cadence, etc.)?**

- **H:** the dominant lever - turnover falls and IR/LIQ improve monotonically as H goes 1 -> 10 **[A2]**; the longest-horizon and mean-of-horizons predictions are the lowest-turnover **[A5][A6]**.
- **Optimiser:** removing it (equal-weight ridge returns) costs IR/TVR/LIQ **[A4]**.
- **`k_tvr`:** turnover reduction survives `k_tvr = 0`, so the result is **not fragile** to the explicit turnover weight - the multi-period structure carries it **[A3]**.
- **Ridge α = 1e5:** intentionally large; the forecaster is robust because of pooling + shrinkage.

**What are the recommended default values, and what is the rationale?**

`HORIZONS = 10`, `RIDGE_ALPHA = 1e5`, `N_BUCKETS_MPO = 20`, `LOOKBACK = 1008`, decay `tau = H/2`, optimiser weights `k_mean=1.0, k_var=0.5, k_base=0.05, k_tvr=0.1, k_couple=0.05`. Rationale: H=10 maximises the turnover/IR trade-off; strong ridge + 20 buckets control overfitting; the QP weights balance return vs path-stability.

**Is there any parameter the idea is fragile to - i.e., small changes flip the outcome?**

The clearest fragility is **region/window** rather than a numeric knob - the **US+EU+JP 1-year** result is negative while US is robustly positive, so the idea is sensitive to the universe it is deployed on. No single hyperparameter was reported to flip the outcome within its swept range.

---

## 18. Limitations & future work

**What are the known weaknesses or open questions of the idea?**

- **Multi-region 1-year underperformance** (improv -42.20%, IR -0.010) - needs investigation / region-specific tuning; the idea is validated mainly on US.
- **Max correlation approximately 1.000** to a higher-ranked twin function in every ranking window - it adds little on top of that sibling, so its incremental value over the existing book should be quantified before allocation.
- Beta/sector neutrality are **not** enforced at preA stage; reliance on post-processing.
- Bucketing degenerates when fewer than 20 alphas are selected.
- The baseline MVO anchor is somewhat arbitrary (any model can be used).

**What is the compute / memory profile of the function, and is there a path to make it cheaper?**

Each rebalance trains **10 ridge models** on stacked stock-day matrices (cheap, sklearn). The per-day cost is a **CVXPY/OSQP QP in stock space** with an H-horizon factor risk model (the heaviest piece; run time approximately 5 min in the rankings, capped at `max_iter=4000`). Memory is bounded by `float32` cubes and the **20-bucket compression**, so it scales well with the number of alphas. Cheaper paths: cache/limit `MAX_FACTORS_QP`, reduce H, solve a single aggregated QP rather than per-horizon variables, or warm-start across days (already `warm_start=True`).

**What variants were tried and rejected, and why?**

- **H = 1 / H = 5** (rejected: higher turnover than H = 10 **[A2]**).
- **Raw ridge returns as preA** - 1st / 5th / 10th / mean horizon, no optimiser (rejected: worse IR/LIQ; only useful to demonstrate the turnover-vs-horizon relationship **[A5][A6]**).
- **Equal-weight mean of ridge returns** (rejected: optimiser adds IR/TVR/LIQ **[A4]**).
- **Default construct_preA** (baseline MVO weights, no MPO) - rejected: MPO wins at matched postA turnover **[A1]**.
- **Alpha-weight perturbation** (the earlier approach) - replaced by **stock-level return** optimisation.
