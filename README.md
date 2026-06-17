# MPO Fit Function 

## 1. Identification

**Name of idea**

Multi-Period Optimisation (MPO) Fit Function

**Slack channel**

snarang73_ameshram_mpo_fit

**preTQ100m_xxx label name**

`preTQ100m_mpo_fit`

**Author(s)**

Sanchit Narang, Alankar Meshram

**Function type (filter, fit, post-processing, other)**

Fit function 

---

## 2. Idea & hypothesis 

**Describe the idea in one paragraph.**

Standard mean-variance optimisation (MVO) is *single-period*: it optimises today's portfolio in isolation, ignoring how positions must evolve over the coming days. This produces high day-to-day turnover, because a position taken today can be reversed tomorrow when the next single-period solve flips its sign. The MPO fit function instead forecasts each stock's **cumulative excess return over H future horizons (H = 10 days)** using a bank of per-horizon ridge models, and then solves a **single multi-period quadratic program** that jointly chooses the portfolio path `x[1..H]`. The objective trades expected return against factor risk, distance from a baseline MVO portfolio, **turnover versus yesterday's positions**, and a coupling penalty that keeps consecutive horizons close to each other. The decay-weighted aggregate `x̄` of the per-horizon solutions becomes the preA. The net effect is a **low-turnover, path-aware preA**.

**What is the intuition or market hypothesis behind the idea - why should it work?**

A signal that is informative about *both* tomorrow's and next week's returns should be traded smoothly rather than churned. By looking H days ahead and penalising path movement, the optimiser only trades when the multi-horizon forecast genuinely changes - it does not react to one-day noise that the next day's solve would undo; turnover falls monotonically as H grows from 1 to 10 (refer to Section 16, A2). Crucially, turnover falls as H increases **even with the explicit turnover penalty switched off** (`k_tvr = 0`) (refer to Section 16, A3), which shows the turnover reduction is a genuine multi-period effect, not just a penalty artefact.

**What existing function variants does it build on, replace, or extend?**

#####

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

Take 3 stocks (AAPL, MSFT, NVDA) and 3 alpha signals (A1, A2, A3).

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

The key observation: the single-period baseline wants to **short MSFT today**, which **flips MSFT's sign versus yesterday's long** - and tomorrow's baseline may well flip it back, so this kind of position churns and burns turnover.

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

The naive single-period baseline trades only on the 1-day view: it shorts MSFT today (flipping it from yesterday's long), and is liable to flip it back to long tomorrow - a wasteful round trip repeated day after day, i.e. high turnover. MPO, by looking H = 10 horizons ahead and pricing in the turnover and coupling penalties, recognises MSFT's longer-term long trajectory and holds a steadier position rather than chasing the noisy 1-day signal. The directional view is broadly preserved while the day-to-day churn is removed, so MPO reaches a similar expected return at materially **lower turnover** - exactly the trade-off the full backtests confirm (refer to Section 16, A1-A2).

---

## 5. Model

**What type of model is used (linear regression, ridge, NNLS, MVO/QP, tree ensemble, neural network, other)?**

A **two-stage hybrid**. Stage 1 is a bank of **linear ridge regressions** (one per horizon) for return forecasting. Stage 2 is a **convex quadratic program** (mean-variance / multi-period optimiser) for portfolio construction. 

**Why is this model class appropriate for the idea?**

Ridge is robust and fast for the noisy, collinear super-alpha features (strong L2 shrinkage tames multicollinearity among the 20 buckets); a QP is the natural way to express the multi-period mean-variance utility with hard dollar-neutral / leverage constraints and quadratic turnover/coupling penalties, and it is convex so the solve is reliable in production.

**What are the key model hyperparameters and how were they chosen?**

The single most important hyperparameter is the **number of future horizons (H)** the optimiser forecasts and plans over. This is the lever that defines the idea: with `H = 1` the function collapses to an ordinary single-period optimiser, and as `H` grows the optimiser plans the position path further out and trades it more smoothly. We swept it directly (`H = 1 -> 5 -> 10`) and found turnover falls monotonically as `H` increases, with the largest reduction at `H = 10` (refer to Section 16, A2), so **`HORIZONS = 10`** is the default. 

The remaining hyperparameters are :

- `N_BUCKETS_MPO = 20` - the alphas are compressed into 20 super-alphas, fixing the feature count regardless of how many alphas are selected and controlling overfitting / collinearity.
- `RIDGE_ALPHA = 1e5` - deliberately strong L2 shrinkage so the 20-feature per-horizon forecaster does not chase daily return noise.
- `LOOKBACK = 1008` (approximately 4 trading years) - the window used to train the ridges and estimate the risk model; long enough for stable estimates, short enough to stay responsive.
- Optimiser objective weights: `k_mean = 1.0` (reward return), `k_var = 0.5` (risk aversion), `k_base = 0.05` (pull toward the baseline), `k_tvr = 0.1` (turnover penalty vs. yesterday), `k_couple = 0.05` (penalty on moves between consecutive horizons) - chosen to balance return against path-stability.
- `RETURN_CAP = 0.15` (caps the forecast target to limit outliers)

**Approximately how many trainable parameters does the model have?**

20 super-alpha coefficients x 10 horizons = **200 ridge coefficients** (no intercept), plus the baseline MVO weights (one per selected alpha, but these are an anchor, not learned by gradient). 

**Is the mapping from alphas (or other model inputs) to preA linear or non-linear?**

The alpha->μ stage is **linear** (ridge). The full alpha->preA mapping is **non-linear**, because the QP applies inequality (leverage) and equality (dollar-neutral) constraints plus the previous-day-dependent turnover term - the output is a piecewise-linear/convex function of the inputs, not a simple linear combination.

---

## 6. Inputs / Features

**What feeds the model - alpha positions, alpha PnLs, simres fields, data variables, alpha features, alpha-description metadata, derived statistics?**

*In `fit()` (runs quarterly):*
- Alpha **postA positions** (the alpha cube) - the features that are bucketed into the 20 super-alphas and fed to the ridges.
- Alpha **dailypnl** - used only to build the baseline alpha-weight MVO anchor (`run_mvo_qp`).
- **simres field `dailytvr`** (`BUCKET_METRIC = dailytvr_top3000top1200`) - the derived statistic used to rank/bucket the alphas.
- **`ret1_excess`** (future, capped) - the training **target** (the cumulative h-day forward return per horizon).

*In `construct_preA()` (runs every day):*
- Alpha **postA positions** for the current day(s) - compressed into the 20 super-alphas (using the stored bucket map and baseline weights) and passed to the per-horizon ridges to produce the forecasts `μ[h]`.
- **Yesterday's positions** (`y_prev`) - the live book in last mode, the previous preA column in full mode - the turnover anchor.
- The **stored fit artefacts** - the per-horizon ridge models + scalers, the bucket map, and the baseline MVO weights.


**What is the shape of the model input (n_observations x n_features) in a typical fit call?**

The features always have **20 columns** (the 20 super-alphas), but the number of rows differs sharply between the two stages.

*In `fit()` (one ridge per horizon, h = 1..10):* every (stock, day) pair in the lookback window is one observation, stacked into `X_h` of shape `(N_obs x 20)` with `N_obs ≈ n_valid_stocks x lookback_days`, and a target vector `y_h` of length `N_obs`. Ten such matrices are built, one per horizon.

*In `construct_preA()` (inference + optimisation):* the feature cube `Z` is `(n_stocks x n_days x 20)`; for the day being solved only today's slice is used, **one row per stock**, shape `(n_stocks x 20)`. Each horizon's ridge maps that to a per-stock forecast, stacked into the forecast matrix `Mu` of shape `(H x n_stocks) = (10 x n_stocks)`. The optimisation then works in **stock space**: it solves for `H` position vectors each of length `n_stocks`. 

**Does the model make predictions for each stock independently or for multiple stocks at once?**

The ridge **predicts each stock independently** (same coefficients applied per stock-row). The **QP solves all stocks jointly** (the risk model and dollar-neutral / leverage constraints couple stocks).

**What transformations are applied to inputs (cross-sectional z-score, rank, capping, scaling, sign-flip)?**

Inputs go through three transformations, in order. Writing `a_i` for an alpha column, `S_b` for the super-alpha of bucket `b`, and `x_{s,j}` for feature `j` of stock `s`:

1. **Clean (NaN/inf -> 0):** `x -> 0 if (x is NaN or ±inf) else x`, applied to every cell, so missing or blown-up values never enter the maths.
2. **Common scaling of each super-alpha:** first sum the alphas in a bucket, `S_b = Σ_{i ∈ b} a_i`, then re-scale to a common book size: `S_b -> S_b · (B / Σ_s |S_{b,s}|)` with target book `B = 1e6` (booksize mode, the default). This puts all 20 super-alphas on the same dollar footing so no bucket dominates purely by scale.
3. **Standardise the features before the ridge:** per feature column `j`, `x_{s,j} -> (x_{s,j} - μ_j) / σ_j`, where `μ_j, σ_j` are the column mean and standard deviation computed at fit time and stored (`σ_j` set to 1 when it is 0). Standardising matters for ridge specifically: the L2 penalty `α‖β‖²` shrinks all coefficients equally, so features must share a common scale or the penalty would unfairly punish small-variance columns. Re-applying the stored `(μ_j, σ_j)` at inference guarantees identical scaling to training with no look-ahead.

No sign-flipping is applied (the optimiser is free to go long or short each super-alpha); the target, not the inputs, is the quantity that gets capped (`clip(±0.15)`).

**How are NaNs / infs / zeros cells handled?**

`n2z` maps NaN/+/-inf -> 0; `op.at_nan2zero` on every slab; zero-std feature columns get σ = 1; alphas with today |sum| < 1000 or > 90% zeros are dropped, and stocks with > 90% zeros are dropped.

**Are alphas bucketed before fitting? If yes, what is the bucketing metric (tvr, liq2, sector, ...) and rationale?**

Yes - into **20 buckets** by **mean `dailytvr`** (`rank_equal_groups`). Rationale:

- **Statistical:** grouping alphas of similar turnover keeps each bucket homogeneous, and the compression reduces dimensionality / collinearity before fitting, which (together with the strong ridge shrinkage) controls overfitting.
- **Runtime / scalability:** the filter can select thousands of alphas, and feeding all of them to the ridges and the QP would be slow and memory-heavy and would scale with the alpha count. Collapsing them into a **fixed 20 super-alphas** makes the per-horizon ridge a fixed `(N_obs x 20)` problem and keeps the cost essentially **independent of how many alphas were selected** - 50 alphas or 5000 alphas both reduce to 20 features. This is what keeps the quarterly fit and the daily solve tractable.

Turnover is the chosen bucketing metric because it is the axis the whole idea is about (smoothing trading), so grouping by turnover lets the model treat fast and slow signals separately.

**Are super-alphas (dimensionality-reduction features) created, if yes how many?**

Yes - **20** (= `N_BUCKETS_MPO`). Each is the booksize-normalised **sum** of the alphas in its tvr bucket.

**If super-alphas are constructed, what method splits alphas into super-alpha clusters, and what model is used to combine alphas?**

Alphas are split by **tvr-rank into 20 equal groups** (`rank_equal_groups`). An equal weight model is used to combine the alphas to make superalphas.

---

## 7. Target

**What target variable is used (ret1, ret1_excess, n-day forward return, capped return, ranked return, custom)?**

`ret1_excess` - specifically the **cumulative h-day forward excess return**, `Σ_{k=1..h} ret1_excess(t+k)` for each horizon `h = 1..10`. 

**What transformations are applied to the target (capping, ranking, smoothing over n days, sign)?**

**Capped at +/-`RETURN_CAP = 0.15`**; summed over the next h days (cumulative); `at_nan2zero` on the assembled cube.

**How is the target shifted to be next-day / post-decision, and how is forward bias avoided?**

The key thing to understand is **where future returns are used and where they are not**:

#### 


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

Ridge **L2** regularization is applied.

---

## 9. Overfitting controls

**What explicit techniques mitigate overfitting (regularization, super-alpha pooling, bucketing, weight capping, walk-forward refit, ensembling, early stopping, shrinkage)?**

- Strong **ridge L2** on the return models.
- **Super-alpha pooling**: the feature space is reduced to 20 buckets regardless of how many alphas are selected, drastically cutting parameter count.
- **tvr bucketing** (homogeneous groups).
- **Turnover / anchor / coupling penalties** in the QP, which shrink the solution toward the baseline MVO and toward yesterday's positions.
- **Walk-forward refit** on `rebalance_dates_mask` with a fixed `LOOKBACK = 1008`.
- **Return capping** (+/-0.15) and **leverage cap**.

**How was the idea validated across different regions, sub-universes, and alpha sets - not just on one favorable backtest?**

The validation was done in USA region - on US top1000 and US top3000 universes, over **1y / 2y / 4y** windows, and against the full set of ablations: default-construct vs MPO **[A1]**, the horizon sweep H = 1/5/10 **[A2]**, the same sweep with `k_tvr = 0` **[A3]**, equal-weight ridge vs MPO **[A4]**, and raw ridge returns by horizon **[A5][A6]** (all detailed in Section 16). 

---

## 10. Training / Refit

**At what cadence is the model re-fit (daily, quarterly)?**

The model is re-fit quarterly, only on dates where rebalance_dates_mask is true. Between refits, construct_preA runs daily, using the most recently trained model to generate forecasts and solves the QP (multi-period optimisation runs daily inside the construct_preA).

**What lookback window is used to fit the model and how was its length chosen?**

A lookback of LOOKBACK = 1008 trading days, or roughly 4 years, is used to fit the model.
Four years is a reasonable window because it is long enough to include multiple market regimes and provide enough data to estimate noisy forward-return relationships and covariance reliably. 

**How is walk-forward set up (rolling vs expanding window)?**

**Rolling** window.

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


**Does the function use yesterday's positions? If yes, how is the dependency on previous-day state handled?**

**Yes.** `y_prev` is the turnover anchor in the QP. In last mode it comes from `data["get_yesterday_preA"]()`; in full mode from the previous preA column. Because gross can differ between live and backtest, `fast_fit` **rescales** `y_prev` when `gross_prev > 10x gross_mvo` so the turnover term stays well-scaled.

**How does the function handle the GLOBAL region (or primary-region slicing) differently from single-region runs?**

The function is not GLOBAL region compatible currently.

---

## 13. Portfolio construction

**How are model outputs converted into per-stock per-day positions (preA)?**

The QP's decay-weighted aggregate `x̄` (per-stock) is written directly into the preA column for that day. `x̄` is already in stock space (the ridge predicts per-stock returns and the QP optimises per-stock positions).

**Is the preA normalized cross-sectionally per day (by sum, abs-sum, z-score, rank)?**

Yes - final preA is `cs_booksize`-normalised per day to `data["booksize"]`, and intermediate aggregates are abs-sum normalised.

**Is the preA explicitly dollar-neutral, beta-neutral, or sector-neutral at this stage, or is that deferred to post-processing?**

The QP enforces **dollar-neutrality** explicitly (`Σ x̄ = 0`) and a **leverage cap**. **Beta/sector neutrality are deferred** to post-processing - they are not imposed here. 

**How is the trading universe enforced (multiply by `valids`, mask, drop)?**

`op.at_mask(preA, data["valids"])` then `cs_booksize`; only `selected_valids_today_mvo` stocks are ever assigned non-zero values. 

---

## 14. Edge cases

**What happens when the filter selects zero / one / very few alphas on a rebalance date?**

The function forms a fixed **20 buckets** of alphas, so it is designed to run on selections of around 20 or more alphas, where each bucket is well populated and the super-alpha features are meaningful. This is the normal operating regime, since the upstream filter typically selects hundreds to thousands of alphas. Running on a handful of alphas falls outside this intended design point.

**What happens when an alpha is all-zero or all-NaN over the lookback?**

Dropped in `create_feature` - today |sum| < 1000 or > 90% zero cells -> skip alpha; stocks > 90% zero -> dropped. Empty training matrices -> a **zero-coefficient ridge model** is stored.

**How does it behave when bucketing produces an empty bucket?**

Bucketing is done by `rank_equal_groups`, which assigns the selected alphas to the 20 buckets by their turnover rank in equal-sized groups. Because the groups are sized as `(rank * 20) // n_alphas`, every one of the 20 buckets is guaranteed to be populated as long as there are at least 20 selected alphas - which is the function's normal operating regime, since the upstream filter typically selects hundreds to thousands. In that regime each bucket holds a healthy group of alphas whose positions are summed and booksize-normalised into a well-defined super-alpha, so no empty bucket ever arises. An empty bucket would only be possible on a very small alpha set (fewer than the 20 bucket slots) - the around-20-alpha design point already covered above and in Section 18 - and the natural extension is to let the bucket count scale with the number of selected alphas so the buckets stay fully populated on smaller selections too.

**Are there asset-class-specific branches (equities vs futures)? Region-specific branches (GLOBAL vs single-region)?**

The function is **equities-only**. No explicit region branches are there.

**What known failure modes exist (singular matrix, optimizer non-convergence, exploding gradients, OOM)?**

Singular/near-PSD covariance -> `nearest_psd`, eigenvalue clipping, `+1e-8·I` ridge; **OSQP non-convergence / no solution** -> caught and **falls back to the decay-weighted mean of `Mu`**; short return history (`ret1.shape[1] < 2`) -> equal-weight bucket fallback; missing ridge models -> equal-weight fallback. No special OOM handling beyond `float32` and the 20-bucket compression.

---

## 15. Robustness / generality

**In which regions has the idea been tested (US, EU, JP, ASIA, GLOBAL)?**

**US** across 1y / 2y / 4y.

**In which stock sub-universes (top250 / top500 / top1000 / top3000)?**

Primarily top1000(US) and top3000(US).

**Can the function handle very few (less than 20) and very many (over 5000) alphas?**

The function always compresses the selected alphas into a fixed set of 20 super-alphas before any model is trained, so the downstream cost (10 ridge fits + the stock-level QP) is independent of how many alphas were selected - whether it is 200 or 5000, the model still sees 20 features with no blow-up in parameter count or solve time, which makes it naturally suited to large alpha pools. For a handful of alphas the natural extension is to let the bucket count scale with the size of the selected set, so the same machinery operates comfortably on smaller selections too.

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
| fitSnarang73_MPO_baseTest_defaultCons | Returns the baseline MVO weights with no MPO optimisation in construct_preA |
| fitSnarang73_MPO_baseTest_10H | MPO function with H = 10, using the same baseline MVO model as above |

Performance (US top1000):

| Function | preA IR | preA RET | preA TVR | preA Liq | postA IR | postA RET | postA TVR | postA Liq |
|---|---|---|---|---|---|---|---|---|
| Default Construct PreA (no MPO) | 0.122 | 0.063 | 0.652 | 1620.48 | 0.130 | 0.058 | 0.096 | 2239.99 |
| MPO (H = 10) | 0.165 | 0.100 | 0.159 | 4785.16 | 0.173 | 0.072 | 0.094 | 4455.57 |

Finding: at the **preA** level the MPO version has materially lower turnover (0.652 -> 0.159). Equalising postA turnover (`pp_basic_hump` with `target_tvr = 0.10`), **MPO beats the default in IR (0.173 vs 0.130), RET and LIQ (postA 4455.57 vs 2239.99)** at matched turnover (holds on TOP3000 as well).

**Doing the opposite of the idea - what happens?**

The opposite of "look many horizons ahead" is **H = 1** (single-period). Several benchmarks probe this from different angles.

**[A2]** *Horizon sweep (does more horizons reduce turnover?):*

| Function Name | Function Idea |
|---|---|
| fitSnarang73_MPO_baseTest_1H | MPO function with num_horizons = 1 |
| fitSnarang73_MPO_baseTest_5H | MPO function with num_horizons = 5 |
| fitSnarang73_MPO_baseTest_10H | MPO function with num_horizons = 10 |

Performance (US top3000):

| Function | preA IR | preA RET | preA TVR | preA Liq | postA IR | postA RET | postA TVR | postA Liq |
|---|---|---|---|---|---|---|---|---|
| 1H | 0.165 | 0.087 | 0.250 | 662.23 | 0.085 | 0.034 | 0.141 | 1502.99 |
| 5H | 0.196 | 0.100 | 0.189 | 839.23 | 0.099 | 0.039 | 0.134 | 1620.48 |
| 10H | 0.203 | 0.103 | 0.159 | 949.10 | 0.103 | 0.041 | 0.128 | 1699.83 |

Finding: as the number of horizons increases (1 -> 5 -> 10), preA turnover falls monotonically (0.250 -> 0.189 -> 0.159) while IR improves (0.165 -> 0.196 -> 0.203); the same ordering holds at postA.

**[A3]** *Horizon sweep with the explicit turnover term switched off:*

| Function Name | Function Idea |
|---|---|
| fitSnarang73_MPO_baseTest_1H_ktvr0 | MPO function with num_horizons = 1 and `k_tvr = 0` |
| fitSnarang73_MPO_baseTest_5H_ktvr0 | MPO function with num_horizons = 5 and `k_tvr = 0` |
| fitSnarang73_MPO_baseTest_10H_ktvr0 | MPO function with num_horizons = 10 and `k_tvr = 0` |

Performance (US top3000):

| Function | preA IR | preA RET | preA TVR | preA Liq | postA IR | postA RET | postA TVR | postA Liq |
|---|---|---|---|---|---|---|---|---|
| 1H, k_tvr = 0 | 0.014 | 0.007 | 0.723 | 266.27 | 0.187 | 0.070 | 0.161 | 1329.04 |
| 5H, k_tvr = 0 | 0.139 | 0.066 | 0.523 | 360.11 | 0.211 | 0.078 | 0.156 | 1361.08 |
| 10H, k_tvr = 0 | 0.194 | 0.091 | 0.413 | 432.59 | 0.217 | 0.081 | 0.150 | 1397.71 |

Finding: turnover **still** falls as horizons increase even with `k_tvr = 0` (preA 0.723 -> 0.523 -> 0.413; postA 0.161 -> 0.156 -> 0.150), and IR rises with it - proving the turnover reduction is the genuine effect of the multi-period structure, not the explicit turnover penalty.

**[A4]** *Add-value of the optimiser (MPO vs using ridge returns directly):*

| Function Name | Function Idea |
|---|---|
| fitSnarang73Ameshram_MPO_10H_ridge_ret_ew_mean | For H = 10, uses the mean return prediction over all 10 horizons as the preA (no optimiser) |
| fitSnarang73_MPO_baseTest_10H | For H = 10, feeds those same return predictions into the MPO setup |

Performance (US top3000):

| Function | preA IR | preA RET | preA TVR | preA Liq | postA IR | postA RET | postA TVR | postA Liq |
|---|---|---|---|---|---|---|---|---|
| 10H (only ridge) | 0.141 | 0.177 | 0.321 | 463.87 | 0.059 | 0.044 | 0.124 | 1403.81 |
| 10H (MPO) | 0.203 | 0.103 | 0.159 | 949.10 | 0.103 | 0.041 | 0.128 | 1699.83 |

Finding: the MPO setup gives **better IR (preA 0.203 vs 0.141; postA 0.103 vs 0.059), TVR and LIQ (preA 949.10 vs 463.87; postA 1699.83 vs 1403.81)** than using the ridge returns directly; correlation between the two is **64%**.

**[A5]** *Additional benchmark - raw ridge returns as preA (which horizon to trade, no optimiser):*

| Function Name | Function Idea |
|---|---|
| fitSnarang73Ameshram_MPO_10H_ridge_ret_1st | For H = 10, uses the 1st horizon's return prediction as the preA (no MPO optimiser) |
| fitSnarang73Ameshram_MPO_10H_ridge_ret_5th | For H = 10, uses the 5th horizon's return prediction as the preA (no MPO optimiser) |
| fitSnarang73Ameshram_MPO_10H_ridge_ret_10th | For H = 10, uses the 10th horizon's return prediction as the preA (no MPO optimiser) |
| fitSnarang73Ameshram_MPO_10H_ridge_ret_ew_mean | For H = 10, uses the mean prediction over all 10 horizons as the preA (no MPO optimiser) |

Performance (US top3000, preA):

| Function | IR | RET | TVR | Liq |
|---|---|---|---|---|
| 1st horizon | 0.081 | 0.096 | 0.621 | 251.77 |
| 5th horizon | 0.141 | 0.176 | 0.340 | 445.56 |
| Mean | 0.143 | 0.180 | 0.321 | 463.87 |
| 10th horizon | 0.147 | 0.186 | 0.270 | 521.85 |

Finding: longer-horizon predictions give **lower turnover** preA (1st 0.621 -> 10th 0.270), with IR also rising toward the longer horizons.

**[A6]** *Additional benchmark - mean of ridge returns across an increasing number of horizons (no optimiser):*

| Function Name | Function Idea |
|---|---|
| fitSnarang73Ameshram_MPO_1H_ridge_ret_ew_mean | For H = 1, mean return prediction over the 1 horizon |
| fitSnarang73Ameshram_MPO_5H_ridge_ret_ew_mean | For H = 5, mean return prediction over the 5 horizons |
| fitSnarang73Ameshram_MPO_10H_ridge_ret_ew_mean | For H = 10, mean return prediction over the 10 horizons |

Performance (US top3000, preA):

| Function | IR | RET | TVR | Liq |
|---|---|---|---|---|
| mean(1H) | 0.081 | 0.096 | 0.621 | 251.77 |
| mean(5H) | 0.133 | 0.164 | 0.401 | 384.34 |
| mean(10H) | 0.143 | 0.180 | 0.321 | 463.87 |

Finding: averaging over **more horizons** lowers the preA turnover (0.621 -> 0.401 -> 0.321) while IR improves (0.081 -> 0.133 -> 0.143).

**Function ranking across (US, US+EU+JP) x (4y, 2y, 1y): list percentage performance numbers and the maximum correlation to functions the idea does not outperform.**

The MPO fit ranks in the middle of the fit ranking pool (9 functions) and is highly correlated to a higher-ranked sibling function it does not outperform.

| Region | Horizon | Rank | Max corr to functions it does not outperform |
|---|---|---|---|
| USA | 1y | 6th | 64.0% |
| USA | 2y | 6th | 61.2% |
| USA | 4y | 6th | 61.6% |
| USA + EU + JP | 1y | 7th | 64.6% |
| USA + EU + JP | 2y | 7th | 64.1% |
| USA + EU + JP | 4y | 7th | 63.6% |


**preTQ100_xxx label performance - relevant metrics from label OS pnl.**

The label `preTQ100m_mpo_fit` outperforms both TREX benchmarks over the entire os period.

| Window (OS) | preTQ100m_mpo_fit | trex_ALL_gross | trex_USA_gross |
|---|---|---|---|
| Full OS (20260107 - 20260617) | 0.081 | 0.053 | 0.071 |

**Benchmark strategy vs with idea comparison: OS days, OS IR, 1y IR, 2y IR, correlation to benchmark.**

The MPO-fit bmark strategy outperforms the `_beat_the_bmark` strategy over both windows, with no recent drawdown in the MPO strategy:

| Window (OS) | IR (MPO fit) | IR (_beat_the_bmark) | Correlation |
|---|---|---|---|
| Last 60 OS days (20260218 - 20260513) | 0.198 | 0.092 | 33.7% |
| Full OS, 88 days (20260107 - 20260513) | 0.091 | 0.070 | 26.3% |

---

## 17. Sensitivity / hyperparameter analysis

**Which hyperparameters were swept and over what ranges?**

- **Number of horizons H ∈ {1, 5, 10}** (refer to Section 16, A2).
- **Optimiser-on vs optimiser-off** (MPO QP vs raw/equal-weight ridge returns) (refer to Section 16, A4).
- **Explicit turnover term `k_tvr ∈ {default, 0}`** (refer to Section 16, A3).
- **Which horizon's prediction to trade** (1st / 5th / 10th / mean) (refer to Section 16, A5-A6).

**How sensitive is performance to each parameter (lookback, regularization strength, number of buckets, number of super-alphas, refit cadence, etc.)?**

- **H:** the dominant lever - turnover falls and IR/LIQ improve monotonically as H goes 1 -> 10 (refer to Section 16, A2); the longest-horizon and mean-of-horizons predictions are the lowest-turnover (refer to Section 16, A5-A6).
- **Optimiser:** removing it (equal-weight ridge returns) costs IR/TVR/LIQ (refer to Section 16, A4).
- **`k_tvr`:** turnover reduction survives `k_tvr = 0`, so the result is **not fragile** to the explicit turnover weight - the multi-period structure carries it (refer to Section 16, A3).
- **Ridge α = 1e5:** intentionally large; the forecaster is robust because of pooling + shrinkage.

**What are the recommended default values, and what is the rationale?**

`HORIZONS = 10`, `RIDGE_ALPHA = 1e5`, `N_BUCKETS_MPO = 20`, `LOOKBACK = 1008`, decay `tau = H/2`, optimiser weights `k_mean=1.0, k_var=0.5, k_base=0.05, k_tvr=0.1, k_couple=0.05`. Rationale: H=10 maximises the turnover/IR trade-off; strong ridge + 20 buckets control overfitting; the QP weights balance return vs path-stability.

**Is there any parameter the idea is fragile to - i.e., small changes flip the outcome?**

The clearest fragility is **region/window** rather than a numeric knob - the **US+EU+JP 1-year** result is negative while US is robustly positive, so the idea is sensitive to the universe it is deployed on. No single hyperparameter was reported to flip the outcome within its swept range.

---

## 18. Limitations & future work

**What are the known weaknesses or open questions of the idea?**

- **Not yet GLOBAL-compatible.** Extending it to run natively across the GLOBAL universe is a clear next step.
- **Requires around 20 alphas in the fit.** The fit builds a fixed 20 buckets of alphas, so the function is designed to run on selections of roughly 20 or more alphas, where each bucket is well populated and the super-alpha features are meaningful. This is the normal operating regime (the upstream filter usually selects hundreds to thousands of alphas); a future extension could make the bucket count adapt to the size of the selected alpha set so it also runs comfortably on smaller selections.


**What is the compute / memory profile of the function, and is there a path to make it cheaper?**

It runs comfortably within the default permutator setup, no special compute or memory needed

**What variants were tried and rejected, and why?**

- **H = 1 / H = 5** (rejected: higher turnover than H = 10 (refer to Section 16, A2)).
- **Raw ridge returns as preA** - 1st / 5th / 10th / mean horizon, no optimiser (rejected: worse IR/LIQ; only useful to demonstrate the turnover-vs-horizon relationship (refer to Section 16, A5-A6)).
- **Equal-weight mean of ridge returns** (rejected: optimiser adds IR/TVR/LIQ (refer to Section 16, A4)).
- **Default construct_preA** (baseline MVO weights, no MPO) - rejected: MPO wins at matched postA turnover (refer to Section 16, A1).
- **Alpha-weight perturbation** (the earlier approach) - replaced by **stock-level return** optimisation.
