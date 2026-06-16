# MPO Fit Function - Write-up Answers (Q/A format)

> Source material: `fitSnarang73Ameshram_MPO_10H_ffw_fix-Copy1.py` and
> `Multi Period Optimisation Fit - Presentation.pdf`, plus the toy-example
> data in `mpo_toy_example` / `toy_example_data.json`.

---

## 1. Identification

**Name of idea**

Multi-Period Optimisation (MPO) Fit Function.

**Slack channel**

snarang73_ameshram_mpo_fit

**preTQ100m_xxx label name**

preTQ100m_mpo_fit

**Author(s)**

Sanchit Narang, Alankar Meshram.

**Function type (filter, fit, post-processing, other)**

Fit function 

---

## 2. Idea & hypothesis

**Describe the idea in one paragraph.**

Standard mean-variance optimisation (MVO) is single-period: it optimises today's portfolio in isolation, ignoring how positions must evolve over the coming days. This produces high day-to-day turnover, because a position taken today can be reversed tomorrow when the next single-period solve flips its sign. The MPO fit function instead forecasts each stock's **cumulative excess return over H future horizons (H = 10 days)** using a bank of per-horizon ridge models, and then solves a **single multi-period quadratic program** that jointly chooses the portfolio path `x[1..H]`. The objective trades expected return against factor risk, distance from a baseline MVO portfolio, **turnover versus yesterday's positions**, and a coupling penalty that keeps consecutive horizons close to each other. The decay-weighted aggregate `x̄` of the per-horizon solutions becomes the preA. The net effect is a **low-turnover, path-aware preA**.

**What is the intuition or market hypothesis behind the idea - why should it work?**

A signal that is informative about *both* tomorrow's and next week's returns should be traded smoothly rather than churned. By looking H days ahead and penalising path movement, the optimiser only trades when the multi-horizon forecast genuinely changes - it does not react to one-day noise that the next day's solve would undo. Crucially, turnover falls as H increases **even with the explicit turnover penalty switched off** (`k_tvr = 0`), which shows the turnover reduction is a genuine multi-period effect, not just a penalty artefact.

**What existing function variants does it build on, replace, or extend?**

It extends a simple single-period MVO fit. The standard MVO perturbs alpha weights and re-solves today's portfolio in isolation, this approach replaces that perturbation with stock-level forecasting of cumulative excess return over the H-day horizon via the per-horizon ridge bank, then wraps the result in the multi-period QP that jointly chooses the portfolio path. The original MVO weights are still computed, but their role is reduced to a single input: they serve as the baseline anchor x_mvo_base that the path-distance term pulls toward (and any model can be substituted here). So the multi-period machinery sits on top of the existing single-period solve, reusing it as a reference point while adding the forward-looking, path-aware structure that delivers the low-turnover preA.

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

**Step 3 - solve multi-period optimisation problem.** A single quadratic program takes all the horizon forecasts `μ^(1) … μ^(H)`, yesterday's positions `y_prev`, and the baseline `x_base`, and picks the whole position path `x_1 … x_H` jointly. It balances two competing pulls:

- **Reward expected return:** each `x_h` is rewarded for aligning with that horizon's forecast `μ_h`.
- **Penalise movement:** moving away from `y_prev` (turnover) and large jumps between consecutive horizons `x_{h-1} -> x_h` are penalised, and the solution is pulled toward the baseline `x_base`.

For MSFT, those two pulls trade off: the `h = 1` forecast argues for a short, but every later horizon argues for a long, and the turnover/coupling penalties make a short-today-then-long-tomorrow round trip expensive. The optimiser therefore declines to put on the aggressive 1-day short and instead holds a small, stable MSFT position consistent with its longer-horizon upward trajectory.

**Show how the model output differs from a naive baseline (equal-weight) on the same toy input.**

The naive single-period baseline trades only on the 1-day view: it shorts MSFT today (flipping it from yesterday's long), and is liable to flip it back to long tomorrow - a wasteful round trip repeated day after day, i.e. high turnover. MPO, by looking H = 10 horizons ahead and pricing in the turnover and coupling penalties, recognises MSFT's longer-term long trajectory and holds a steadier position rather than chasing the noisy 1-day signal. The directional view is broadly preserved while the day-to-day churn is removed, so MPO reaches a similar expected return at materially **lower turnover** - exactly the trade-off the full backtests confirm.

---

## 5. Model

**What type of model is used (linear regression, ridge, NNLS, MVO/QP, tree ensemble, neural network, other)?**

A **two-stage hybrid**. Stage 1 is a bank of **linear ridge regressions** (one per horizon) for return forecasting. Stage 2 is a **convex quadratic program** (multi-period optimiser) for portfolio construction. 

**Why is this model class appropriate for the idea?**

Ridge is robust and fast for the noisy, collinear super-alpha features (strong L2 shrinkage tames multicollinearity among the 20 buckets); a QP is the natural way to express the multi-period mean-variance utility with hard dollar-neutral / leverage constraints and quadratic turnover/coupling penalties, and it is convex so the solve is reliable in production.

**What are the key model hyperparameters and how were they chosen?**

The single most important hyperparameter is the **number of future horizons (H)** the optimiser forecasts and plans over. This is the lever that defines the idea: with `H = 1` the function collapses to an ordinary single-period optimiser, and as `H` grows the optimiser plans the position path further out and trades it more smoothly. We swept it directly (`H = 1 -> 5 -> 10`) and found turnover falls monotonically as `H` increases, with the largest reduction at `H = 10`, so **`HORIZONS = 10`** is the default.

**Approximately how many trainable parameters does the model have?**

20 super-alpha coefficients x 10 horizons = 200 ridge coefficients, plus the baseline MVO weights (one per selected alpha, but these are an anchor, not learned by gradient).

**Is the mapping from alphas (or other model inputs) to preA linear or non-linear?**

The alpha->μ stage is **linear** (ridge). The full alpha->preA mapping is **non-linear**, because the QP applies inequality (leverage) and equality (dollar-neutral) constraints plus the previous-day-dependent turnover term - the output is a piecewise-linear/convex function of the inputs, not a simple linear combination.

---

## 6. Inputs / Features

**What feeds the model - alpha positions, alpha PnLs, simres fields, data variables, alpha features, alpha-description metadata, derived statistics?**

- Alpha positions - for the ridge models and the QP
- Alpha dailypnl - for the baseline MVO
- simres field `dailytvr` (`BUCKET_METRIC = dailytvr_top3000top1200`) - for bucketing
- `ret1_excess` - the target

**What is the shape of the model input (n_observations x n_features) in a typical fit call?**

The features always have 20 columns (the 20 super-alphas). For training, every (stock, day) pair in the lookback window is treated as one observation and stacked into a single matrix of shape `(N_obs x 20)`, where `N_obs ≈ n_valid_stocks x lookback_days`, with a matching target vector of length `N_obs`. At inference the model is applied to just today's slice, one row per stock, shape `(n_stocks x 20)`.

**Does the model make predictions for each stock independently or for multiple stocks at once?**

The ridge predicts each stock independently (same coefficients applied per stock-row). The **QP solves all stocks jointly** (the risk model and dollar-neutral / leverage constraints couple stocks).

**What transformations are applied to inputs (cross-sectional z-score, rank, capping, scaling, sign-flip)?**

The super-alphas are normalised by cs_booksize and the input features are standardised (subtract mean / divide std) before ridge.

**How are NaNs / infs / zeros cells handled?**

`n2z` maps NaN/+/-inf -> 0; `op.at_nan2zero` on every slab; zero-std feature columns get σ = 1; alphas with today |sum| < 1000 or > 90% zeros are dropped, and stocks with > 90% zeros are dropped.

**Are alphas bucketed before fitting? If yes, what is the bucketing metric (tvr, liq2, sector, ...) and rationale?**

Yes - into **20 buckets** by **mean `dailytvr`** (`rank_equal_groups`). Rationale: group alphas of similar turnover so the optimiser treats homogeneous-turnover signals together.

**Are super-alphas (dimensionality-reduction features) created, if yes how many?**

Yes, 20 super-alphas are created. 

**If super-alphas are constructed, what method splits alphas into super-alpha clusters, and what model is used to combine alphas?**

Alphas are split by tvr into 20 equal groups. An equal weight model is used to combine the alphas.

---

## 7. Target

**What target variable is used (ret1, ret1_excess, n-day forward return, capped return, ranked return, custom)?**

`ret1_excess` - specifically the **cumulative h-day forward excess return**, `Σ_{k=1..h} ret1_excess(t+k)` for each horizon `h = 1..10`. 

**What transformations are applied to the target (capping, ranking, smoothing over n days, sign)?**

Capped at +/-`RETURN_CAP = 0.15`

**How is the target shifted to be next-day / post-decision, and how is forward bias avoided?**

Uses `op.ts_delay(ret1_excess, -k - delay)` so the target is strictly post-decision (accounts for the simulation `delay`). The last `delay` columns are explicitly set to NaN and dropped, so no future information leaks into training.

**How are missing or invalid target observations masked?**

`valid = np.isfinite(y)` selects only finite target cells, and the same mask is applied to the feature rows. Targets are further bounded by the universe (`ret1_excess * valids`).

**Why is this target appropriate for the prediction problem the model is solving?**

The QP needs an expected-return vector **per horizon**; cumulative h-day forward return is exactly the quantity each horizon's position should be paid for, so forecasting it directly aligns the model output with the optimiser's objective. Capping protects the ridge fit from outlier return blow-ups.

---

## 8. Loss / Objective

**What is the loss or objective function (MSE, NNLS residual, mean-variance utility, log-likelihood, custom)?**

- **Ridge loss:** L2-penalised mean-squared error, `‖Xβ - y‖² + α‖β‖²` with `α = 1e5`, `fit_intercept=False`.
- **Optimiser objective (`fast_fit`):** maximise the multi-period mean-variance utility `Σ_h decay[h]·( μ[h]·x_h - k_var·(factor + diag risk) - k_base·‖x_h - x_mvo_base‖² - k_tvr·‖x_h - y_prev‖² ) - k_couple·Σ_{h>0} ‖x_h - x_{h-1}‖²`.

**Are there hard constraints on the solution (sum-to-one, non-negativity, per-weight cap, leverage cap, return-target)?**

- **Dollar-neutral:** `Σ x̄ = 0`.
- **Leverage cap:** `‖x̄‖₁ <= 2·gross_ref` (gross_ref = max of prev/MVO gross, >= 1).
- Baseline `run_mvo_qp` additionally imposes **non-negativity** (`w >= 0`) and **sum-to-one** (`Σw = 1`) on the alpha weights.

**Is the objective convex? Is the solution unique?**

The MPO QP is **convex** (PSD quadratic risk + convex penalties, linear constraints) -> a **unique** optimum (solved by OSQP). The baseline QP is likewise convex (cvxopt). Ridge is strictly convex -> unique β.

**What regularization is applied (L1, L2, dropout, max-norm, early stopping, tree depth / leaves / min-split-gain)?**

Ridge L2 regularization is applied.

---

## 9. Overfitting controls

**What explicit techniques mitigate overfitting (regularization, super-alpha pooling, bucketing, weight capping, walk-forward refit, ensembling, early stopping, shrinkage)?**

- Strong **ridge L2** on the return models.
- **Super-alpha pooling**: the feature space is reduced to 20 buckets regardless of how many alphas are selected, drastically cutting parameter count.
- **tvr bucketing** (homogeneous groups).
- **Factor risk model** with PCA truncation (<= 50 factors) + **Ledoit-Wolf** shrinkage in the baseline covariance.
- **Turnover / anchor / coupling penalties** in the QP, which shrink the solution toward the baseline MVO and toward yesterday's positions.
- **Walk-forward refit** on `rebalance_dates_mask` with a fixed `LOOKBACK = 1008`.
- **Return capping** (+/-0.15) and **leverage cap**.

**How was the idea validated across different regions, sub-universes, and alpha sets - not just on one favorable backtest?**

The idea was validated on **US top1000** over **1y / 2y / 4y** windows, and against multiple ablations (H = 1/5/10; default-construct vs MPO; equal-weight ridge vs MPO; raw ridge returns; `k_tvr = 0`). 

---

## 10. Training / Refit

**At what cadence is the model re-fit (daily, quarterly)?**

The model is re-fit quarterly. Between refits, `construct_preA` runs the QP each day.

**What lookback window is used to fit the model and how was its length chosen?**

A single lookback of **`LOOKBACK = 1008` trading days (approximately 4 years)** is used for both jobs the fit step performs: training the per-horizon ridges and estimating the risk model's covariance.

The length is a deliberate bias-variance trade-off. The window has to be long because both estimation tasks are data-hungry: the risk model estimates a large stock covariance (many factors plus per-stock variances) that is unstable on short samples, and although the ridges only have 20 coefficients per horizon, the forward-return target is extremely noisy, so a long window of stock-day observations is what makes the relationship between super-alpha exposure and forward return estimable at all. Four years also spans more than one market regime, so the models are fit across a mix of conditions rather than a single trending or mean-reverting stretch, which keeps them from over-specialising.

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

**Does the function use yesterday's positions? If yes, how is the dependency on previous-day state handled?**

**Yes.** `y_prev` is the turnover anchor in the QP. In last mode it comes from `data["get_yesterday_preA"]()`; in full mode from the previous preA column. Because gross can differ between live and backtest, `fast_fit` **rescales** `y_prev` when `gross_prev > 10x gross_mvo` so the turnover term stays well-scaled.

**How does the function handle the GLOBAL region (or primary-region slicing) differently from single-region runs?**

The function is not GLOBAL compatible currently.

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

If `selected_idx_di.size == 0` the date is skipped in `fit()` and returns zeros in `construct_preA`; if features end up empty, `fast_fit` returns a zero vector. With a single bucket, `_compress_by_buckets` short-circuits.

**What happens when an alpha is all-zero or all-NaN over the lookback?**

Dropped in `create_feature` - today |sum| < 1000 or > 90% zero cells -> skip alpha; stocks > 90% zero -> dropped. Empty training matrices -> a **zero-coefficient ridge model** is stored.

**How does it behave when bucketing produces an empty bucket?**

`rank_equal_groups` produces contiguous equal-size groups, so a bucket can only be empty if there are fewer alphas than buckets; the per-cluster loop simply produces no slab for an empty cluster and the QP runs on the remaining buckets.

**Are there asset-class-specific branches (equities vs futures)? Region-specific branches (GLOBAL vs single-region)?**

The function is expected to run in equities only.

**What known failure modes exist (singular matrix, optimizer non-convergence, exploding gradients, OOM)?**

Singular/near-PSD covariance -> `nearest_psd`, eigenvalue clipping, `+1e-8·I` ridge; **OSQP non-convergence / no solution** -> caught and **falls back to the decay-weighted mean of `Mu`**; short return history (`ret1.shape[1] < 2`) -> equal-weight bucket fallback; missing ridge models -> equal-weight fallback. No special OOM handling beyond `float32` and the 20-bucket compression.

---

## 15. Robustness / generality

**In which regions has the idea been tested (US, EU, JP, ASIA, GLOBAL)?**

**US** (top1000) across 1y / 2y / 4y. 

**In which stock sub-universes (top250 / top500 / top1000 / top3000)?**

Primarily **top1000** (US). 

**Can the function handle very few (less than 20) and very many (over 5000) alphas?**

The 20-super-alpha compression makes it **scale-robust to many alphas** (cost is dominated by the fixed 20 buckets and the stock-level QP, not the alpha count). For **very few alphas** it degrades gracefully via the single-bucket short-circuit and equal-weight fallbacks, though with < 20 alphas the bucketing becomes degenerate.

---

## 16. Benchmark comparison

**Same function with-idea vs without-idea, on US top1000 using all gw2 filter / fit / pp with apply-extra neut - performance of both.**

`fitSnarang73_MPO_baseTest_defaultCons` returns the **baseline MVO weights with no MPO optimisation**; `fitSnarang73_MPO_baseTest_10H` is the **same baseline + MPO (H=10)**. At the preA level the MPO version has materially **lower turnover**; after equalising postA turnover (`pp_basic_hump`, target_tvr = 0.10), **MPO beats the default in IR, RET and LIQ** (also on TOP3000).

**Doing the opposite of the idea - what happens?**

The opposite of "look many horizons ahead" is **H = 1** (single-period). Sweeping H = 1 -> 5 -> 10 shows turnover **falls monotonically** as horizons increase. Decisively, with the explicit turnover term **switched off (`k_tvr = 0`)**, turnover *still* falls as H increases - so removing the multi-period structure (the thing the idea adds) is what hurts, confirming the effect is genuine and not a penalty artefact. Equal-weighting the ridge returns instead of using the optimiser (`..._ridge_ret_ew_mean`) gives **worse IR/TVR/LIQ** (corr 64% to MPO), showing the optimiser itself adds value.

**Function ranking across (US, US+EU+JP) x (4y, 2y, 1y): list percentage performance numbers and the maximum correlation to functions the idea does not outperform.**

For `fitSnarang73Ameshram_MPO_10H_ffw_fix` (with-idea). Columns: improvement % (vs best in pool), IR, turnover, IR/sqrt(TVR), and **max correlation to a function it does not outperform**.

| Universe | Window | Rank | Improv% | IR | TVR | IR/sqrt(TVR) | Max corr (to a better fn) |
|---|---|---|---|---|---|---|---|
| US top1000 | 1y (252d) | ~6 / 66 | 53.1% | 0.055 | 0.108 | 0.173 | **1.000** |
| US top1000 | 2y (502d) | ~6 / 66 | 66.6% | 0.077 | 0.112 | 0.237 | **1.000** |
| US top1000 | 4y (1004d) | ~6 / 66 | 65.3% | 0.065 | 0.118 | 0.202 | **1.000** |
| US+EU+JP | 1y (252d) | ~6 / 198 | **-42.2%** | **-0.002** | 0.129 | **-0.020** | **1.000** |
| US+EU+JP | 2y (502d) | ~6 / 198 | 49.6% | 0.027 | 0.131 | 0.072 | **1.000** |
| US+EU+JP | 4y (1004d) | ~7 / 198 | 57.7% | 0.043 | 0.133 | 0.094 | **1.000** |

Notes: it sits **approximately top 9%** of functions in US and is solidly positive over 2y/4y; the **multi-region 1-year window is negative** (the main weakness). The **max correlation is approximately 1.000 in every window**, meaning there is a near-identical function ranked above it (almost certainly the sibling MPO/baseline variant by the same authors) - i.e. it does not add diversification on top of that twin, only on top of the rest of the pool.

**preTQ100_xxx label performance - relevant metrics from label OS pnl.**

preTQ100m_mpo_fit label OS pnl:
- Last **60 OS days** (20260218-20260513): **IR = 0.243** vs `trex_ALL_gross` 0.189, `trex_USA_gross` 0.184 - **outperforms both**.
- Full OS (**88 OS days**, from 20260107): **IR = 0.076** vs `trex_ALL_gross` 0.042, `trex_USA_gross` 0.063 - **outperforms both**.

**Benchmark strategy vs with idea comparison: OS days, OS IR, 1y IR, 2y IR, correlation to benchmark.**

MPO-fit bmark strategy vs `_beat_the_bmark`:
- 60 OS days (20260218-20260513): **IR_MPO = 0.198** vs 0.092; **correlation 33.7%**.
- Full OS (88 days, from 20260107): **IR_MPO = 0.091** vs 0.070; **correlation 26.3%**.
- No recent drawdown in the MPO strategy over either window.

---

## 17. Sensitivity / hyperparameter analysis

**Which hyperparameters were swept and over what ranges?**

- **Number of horizons H ∈ {1, 5, 10}**.
- **Optimiser-on vs optimiser-off** (MPO QP vs raw/equal-weight ridge returns).
- **Explicit turnover term `k_tvr ∈ {default, 0}`**.
- **Which horizon's prediction to trade** (1st / 5th / 10th / mean).

**How sensitive is performance to each parameter (lookback, regularization strength, number of buckets, number of super-alphas, refit cadence, etc.)?**

- **H:** the dominant lever - turnover falls and IR/LIQ improve monotonically as H goes 1 -> 10; the longest-horizon and mean-of-horizons predictions are the lowest-turnover.
- **Optimiser:** removing it (equal-weight ridge returns) costs IR/TVR/LIQ.
- **`k_tvr`:** turnover reduction survives `k_tvr = 0`, so the result is **not fragile** to the explicit turnover weight - the multi-period structure carries it.
- **Ridge α = 1e5:** intentionally large; the forecaster is robust because of pooling + shrinkage.

**What are the recommended default values, and what is the rationale?**

`HORIZONS = 10`, `RIDGE_ALPHA = 1e5`, `N_BUCKETS_MPO = 20`, `LOOKBACK = 1008`, decay `tau = H/2`, optimiser weights `k_mean=1.0, k_var=0.5, k_base=0.05, k_tvr=0.1, k_couple=0.05`. Rationale: H=10 maximises the turnover/IR trade-off; strong ridge + 20 buckets control overfitting; the QP weights balance return vs path-stability.

**Is there any parameter the idea is fragile to - i.e., small changes flip the outcome?**

The clearest fragility is **region/window** rather than a numeric knob - the **US+EU+JP 1-year** result is negative while US is robustly positive, so the idea is sensitive to the universe it is deployed on. No single hyperparameter was reported to flip the outcome within its swept range.

---

## 18. Limitations & future work

**What are the known weaknesses or open questions of the idea?**

- **Multi-region 1-year underperformance** (improv -42.2%, IR -0.002) - needs investigation / region-specific tuning; the idea is validated mainly on US.
- **Max correlation approximately 1.000** to a higher-ranked twin function in every ranking window - it adds little on top of that sibling, so its incremental value over the existing book should be quantified before allocation.
- Beta/sector neutrality are **not** enforced at preA stage; reliance on post-processing.
- Bucketing degenerates when fewer than 20 alphas are selected.
- The baseline MVO anchor is somewhat arbitrary (any model can be used).

**What is the compute / memory profile of the function, and is there a path to make it cheaper?**

Each rebalance trains **10 ridge models** on stacked stock-day matrices (cheap, sklearn). The per-day cost is a **CVXPY/OSQP QP in stock space** with an H-horizon factor risk model (the heaviest piece; run time approximately 5 min in the rankings, capped at `max_iter=4000`). Memory is bounded by `float32` cubes and the **20-bucket compression**, so it scales well with the number of alphas. Cheaper paths: cache/limit `MAX_FACTORS_QP`, reduce H, solve a single aggregated QP rather than per-horizon variables, or warm-start across days (already `warm_start=True`).

**What variants were tried and rejected, and why?**

- **H = 1 / H = 5** (rejected: higher turnover than H = 10).
- **Raw ridge returns as preA** - 1st / 5th / 10th / mean horizon, no optimiser (rejected: worse IR/LIQ; only useful to demonstrate the turnover-vs-horizon relationship).
- **Equal-weight mean of ridge returns** (rejected: optimiser adds IR/TVR/LIQ).
- **Default construct_preA** (baseline MVO weights, no MPO) - rejected: MPO wins at matched postA turnover.
- **Alpha-weight perturbation** (the earlier approach) - replaced by **stock-level return** optimisation.
