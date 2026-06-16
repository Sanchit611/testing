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

Standard mean-variance optimisation (MVO) is *single-period*: it optimises today's portfolio in isolation, ignoring how positions must evolve over the coming days. This produces high day-to-day turnover, because a position taken today can be reversed tomorrow when the next single-period solve flips its sign. The MPO fit function instead forecasts each stock's **cumulative excess return over H future horizons (H = 10 days)** using a bank of per-horizon ridge models, and then solves a **single multi-period quadratic program** that jointly chooses the portfolio path `x[1..H]`. The objective trades expected return against factor risk, distance from a baseline MVO portfolio, **turnover versus yesterday's positions**, and a coupling penalty that keeps consecutive horizons close to each other. The decay-weighted aggregate `x̄` of the per-horizon solutions becomes the preA. The net effect is a **low-turnover, path-aware preA** with the same or better information ratio at matched turnover.

**What is the intuition or market hypothesis behind the idea - why should it work?**

A signal that is informative about *both* tomorrow's and next week's returns should be traded smoothly rather than churned. By looking H days ahead and penalising path movement, the optimiser only trades when the multi-horizon forecast genuinely changes - it does not react to one-day noise that the next day's solve would undo. Crucially, turnover falls as H increases **even with the explicit turnover penalty switched off** (`k_tvr = 0`), which shows the turnover reduction is a genuine multi-period effect, not just a penalty artefact.

**What existing function variants does it build on, replace, or extend?**

It extends the existing single-period **MVO fit** (baseline `run_mvo_qp`) by (a) replacing the alpha-weight perturbation with **stock-level return forecasting** via ridge, and (b) wrapping the result in a **multi-period QP** (`fast_fit`). The MVO weights are still computed and used only as the *baseline anchor* `x_mvo_base` (any model can be used here). Conceptually it follows Boyd et al., *Multi-Period Trading via Convex Optimization* (arXiv:1705.00109).

---

## 3. Implementation overview

**Describe implementation of the idea in as much detail as you think necessary. / List the high-level steps of the function.**

High-level pipeline:

**`fit()`, run per rebalance date:**

1. Select alphas where `filter_matrix[:, col_i] > 0`.
2. **Bucket -> super-alphas:** `_make_training_matrices` ranks the selected alphas by their mean `BUCKET_METRIC` (`dailytvr_top3000top1200`) into `N_BUCKETS_MPO = 20` equal-size groups, sums the (booksize-normalised) alpha positions inside each group -> 20 "super-alpha" position slabs (stock x date).
3. **Build H = 10 ridge models:** for each horizon `h = 1..10`, the target is the cumulative `ret1_excess` over the next `h` days (capped +/-0.15), the features are the 20 super-alpha positions standardised per-column, and the model is `Ridge(alpha=1e5, fit_intercept=False)`. Stores model + scaler per horizon.
4. **Baseline MVO:** `run_mvo_qp` on the selected alphas' `dailypnl` over the lookback -> long-only, sum-to-one alpha weights `mvo_weights` = baseline anchor.
5. Pickle everything (`mvo_weights`, `ridge_models`, `cluster_dic`, horizon, hyperparameters) into `model_dict[date]`.

**`construct_preA()`, run per day:**

6. Map each day to the most recent prior refit model.
7. **Build today's features** (`create_feature`): assemble the alpha cube for the selected alphas, then `_compress_by_buckets` compresses it into the 20 super-alpha columns using the previous MVO weights as within-bucket weights -> `Z` (stock x n_days x 20).
8. **Per-horizon μ:** for each horizon, standardise today's super-alpha positions with the stored scaler and `ridge_h.predict` -> per-stock forecast `μ[h]`; stack into `Mu` (H x S).
9. **Risk model:** PCA factor model from the trailing `LOOKBACK` window of `ret1_excess` (top `k_f <= MAX_FACTORS_QP = 50` eigenvectors + diagonal residual variance).
10. **Anchors:** `x_mvo_base` = today's super-alpha positions projected through `w_prev`; `y_prev` = yesterday's preA (turnover anchor).
11. **Solve MPO QP** (CVXPY/OSQP): maximise decay-weighted `Σ_h decay[h]·(μ[h]·x_h - k_var·risk(x_h) - k_base·‖x_h - x_mvo_base‖² - k_tvr·‖x_h - y_prev‖²) - k_couple·Σ_h ‖x_h - x_{h-1}‖²`, subject to dollar-neutrality `Σ x̄ = 0` and an L1 leverage cap `‖x̄‖₁ <= 2·gross_ref`.
12. `x̄ = Σ_h decay[h]·x_h` -> write into preA, then `cs_booksize` normalise and mask by `valids`.

---

## 4. Toy example

**Walk through the idea on a handful of alphas / data points, showing inputs (positions or pnls) and the resulting weights / preA.**

Using `toy_example_data.json`: 4 stocks (AAPL, MSFT, GOOG, NVDA), 3 alphas (α1 momentum, α2 reversal, α3 quality), H = 3.

**Inputs**

- Today's alpha matrix `A_today` (4 stocks x 3 alphas): `AAPL=[1.5,-0.5,1.0]`, `MSFT=[-0.3,1.5,0.6]`, `GOOG=[0.6,0.2,-1.2]`, `NVDA=[-1.2,-1.0,1.5]`.
- Baseline MVO alpha weights `mvo_w = [0.357, 0.319, 0.325]` (approximately equal but tilted by pnl covariance).
- Yesterday's positions `y_prev = [0.3, -0.2, 0.2, -0.3]`.
- Per-horizon ridge coefficients `β[h]`, giving per-stock forecasts `Mu = β·A_today` (3 horizons x 4 stocks).

**MSFT walk-through.** Its forecast *changes sign across horizons*: `μ1(MSFT) = -0.15` (short), `μ2 = +0.39`, `μ3 = +0.81` (strong long). A single-period model would short MSFT today, then likely flip it long tomorrow -> wasted turnover. The MPO optimiser sees the whole path and avoids the round-trip.

**Baseline projection** `x_mvo_base = A_today · mvo_w`, normalised: `[0.291, 0.209, -0.205, -0.295]`.

**MPO solution** (with `k_mean=1, k_var=0.2, k_base=0.5, k_tvr=2.0, k_couple=0.5`, decay `[0.563, 0.289, 0.148]`): per-horizon `x[h]` are nearly identical (the coupling + turnover terms pull them together), and the decay-weighted aggregate normalises to `x̄ = [0.479, -0.156, 0.021, -0.344]`.

**Show how the model output differs from a naive baseline (equal-weight) on the same toy input.**


| | turnover vs y_prev | realised return |
|---|---|---|
| Raw baseline (single-period) | `tvr_raw = 0.827` | `ret_x_mvo = 0.2458` |
| MPO aggregate `x̄` | `tvr_mpo = 0.447` | `ret_x_bar = 0.2608` |

MPO **roughly halves turnover** (0.83 -> 0.45) while **slightly increasing** expected return (0.2458 -> 0.2608) - the core selling point of the idea on a toy scale.

---

## 5. Model

**What type of model is used (linear regression, ridge, NNLS, MVO/QP, tree ensemble, neural network, other)?**

A **two-stage hybrid**. Stage 1 is a bank of **linear ridge regressions** (one per horizon) for return forecasting. Stage 2 is a **convex quadratic program** (mean-variance / multi-period optimiser) for portfolio construction. There is also an auxiliary **long-only QP** (`run_mvo_qp`, cvxopt) producing the baseline anchor.

**Why is this model class appropriate for the idea?**

Ridge is robust and fast for the noisy, collinear super-alpha features (strong L2 shrinkage tames multicollinearity among the 20 buckets); a QP is the natural way to express the multi-period mean-variance utility with hard dollar-neutral / leverage constraints and quadratic turnover/coupling penalties, and it is convex so the solve is reliable in production.

**What are the key model hyperparameters and how were they chosen?**

- `HORIZONS = 10` - swept 1 -> 5 -> 10; 10 gives the largest turnover reduction.
- `RIDGE_ALPHA = 1e5` - deliberately strong shrinkage to prevent the 20-bucket forecaster from overfitting daily return noise.
- `N_BUCKETS_MPO = 20` - dimensionality reduction of the (potentially thousands of) selected alphas.
- `LOOKBACK = 1008` (approximately 4 trading years) - fit/risk window.
- Optimiser weights (`fast_fit` defaults): `k_mean=1.0`, `k_var (gamma)=0.5`, `k_base (lambda_preA)=0.05`, `k_tvr (lambda_turnover)=0.1`, `k_couple (epsilon)=0.05`.
- `TAU_DECAY`/decay: `tau = max(1, H/2) = 5`, `decay[h] ∝ exp(-(h-1)/tau)`.
- `RETURN_CAP = 0.15`, `MAX_FACTORS_QP = 50`.

**Approximately how many trainable parameters does the model have?**

20 super-alpha coefficients x 10 horizons = **200 ridge coefficients** (no intercept), plus the baseline MVO weights (one per selected alpha, but these are an anchor, not learned by gradient). The PCA risk model and decay weights are not trained.

**Is the mapping from alphas (or other model inputs) to preA linear or non-linear?**

The alpha->μ stage is **linear** (ridge). The full alpha->preA mapping is **non-linear**, because the QP applies inequality (leverage) and equality (dollar-neutral) constraints plus the previous-day-dependent turnover term - the output is a piecewise-linear/convex function of the inputs, not a simple linear combination.

---

## 6. Inputs / Features

**What feeds the model - alpha positions, alpha PnLs, simres fields, data variables, alpha features, alpha-description metadata, derived statistics?**

- Alpha **postA positions** (the alpha cube) - the primary features for the ridge models and the QP.
- Alpha **dailypnl** - only for the baseline `run_mvo_qp`.
- **simres field `dailytvr`** (`BUCKET_METRIC = dailytvr_top3000top1200`) - for bucketing.
- **`ret1_excess`** - the target and the source of the PCA risk model.

**What is the shape of the model input (n_observations x n_features) in a typical fit call?**

For each horizon, `_build_Xy_for_horizon` reshapes the 20 super-alpha slabs over the lookback into `X_h` of shape `(N_obs x 20)` where `N_obs ≈ n_valid_stocks x lookback_days` (stacked stock-days), and `y_h` of length `N_obs`. At inference the per-day feature is `Z[:, -1, :]` of shape `(S_kept x 20)`.

**Does the model make predictions for each stock independently or for multiple stocks at once?**

The ridge **predicts each stock independently** (same coefficients applied per stock-row). The **QP solves all stocks jointly** (the risk model and dollar-neutral / leverage constraints couple stocks).

**What transformations are applied to inputs (cross-sectional z-score, rank, capping, scaling, sign-flip)?**

`at_nan2zero` everywhere; super-alphas normalised by `cs_booksize` to 1e6 (`NORMALIZATION_MODE = "booksize"`; `zscore`/`rank` are alternative modes); features standardised (subtract mean / divide std, stored scaler) before ridge.

**How are NaNs / infs / zeros cells handled?**

`n2z` maps NaN/+/-inf -> 0; `op.at_nan2zero` on every slab; zero-std feature columns get σ = 1; alphas with today |sum| < 1000 or > 90% zeros are dropped, and stocks with > 90% zeros are dropped.

**Are alphas bucketed before fitting? If yes, what is the bucketing metric (tvr, liq2, sector, ...) and rationale?**

Yes - into **20 buckets** by **mean `dailytvr`** (`rank_equal_groups`). Rationale: group alphas of similar turnover so the optimiser treats homogeneous-turnover signals together, and reduce dimensionality / collinearity before fitting.

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

Uses `op.ts_delay(ret1_excess, -k - delay)` so the target is strictly post-decision (accounts for the simulation `delay`). The last `delay` columns are explicitly set to NaN and dropped, so no future information leaks into training.

**How are missing or invalid target observations masked?**

`valid = np.isfinite(y)` selects only finite target cells, and the same mask is applied to the feature rows. Targets are further bounded by the universe (`ret1_excess * valids`).

**Why is this target appropriate for the prediction problem the model is solving?**

The QP needs an expected-return vector **per horizon**; cumulative h-day forward return is exactly the quantity each horizon's position should be paid for, so forecasting it directly aligns the model output with the optimiser's objective. Capping protects the ridge fit from earnings-day / outlier return blow-ups.

---

## 8. Loss / Objective

**What is the loss or objective function (MSE, NNLS residual, mean-variance utility, log-likelihood, custom)?**

- **Ridge loss:** L2-penalised mean-squared error, `‖Xβ - y‖² + α‖β‖²` with `α = 1e5`, `fit_intercept=False`.
- **Optimiser objective (`fast_fit`):** maximise the decay-weighted multi-period mean-variance utility `Σ_h decay[h]·( μ[h]·x_h - k_var·(factor + diag risk) - k_base·‖x_h - x_mvo_base‖² - k_tvr·‖x_h - y_prev‖² ) - k_couple·Σ_{h>0} ‖x_h - x_{h-1}‖²`.

**Are there hard constraints on the solution (sum-to-one, non-negativity, per-weight cap, leverage cap, return-target)?**

- **Dollar-neutral:** `Σ x̄ = 0`.
- **Leverage cap:** `‖x̄‖₁ <= 2·gross_ref` (gross_ref = max of prev/MVO gross, >= 1).
- Baseline `run_mvo_qp` additionally imposes **non-negativity** (`w >= 0`) and **sum-to-one** (`Σw = 1`) on the alpha weights.

**Is the objective convex? Is the solution unique?**

The MPO QP is **convex** (PSD quadratic risk + convex penalties, linear constraints) -> a **unique** optimum (solved by OSQP). The baseline QP is likewise convex (cvxopt). Ridge is strictly convex -> unique β.

**What regularization is applied (L1, L2, dropout, max-norm, early stopping, tree depth / leaves / min-split-gain)?**

Ridge **L2 (`α=1e5`)**; the optimiser's `k_base`, `k_tvr`, `k_couple` quadratic penalties act as Tikhonov-style regularisers on the portfolio; risk model uses **Ledoit-Wolf shrinkage** in the baseline and eigenvalue clipping / `nearest_psd` for stability; PCA truncation to <= 50 factors.

---

## 9. Overfitting controls

**What explicit techniques mitigate overfitting (regularization, super-alpha pooling, bucketing, weight capping, walk-forward refit, ensembling, early stopping, shrinkage)?**

- Strong **ridge L2 (α = 1e5)** on the return models.
- **Super-alpha pooling**: the feature space is reduced to 20 buckets regardless of how many alphas are selected, drastically cutting parameter count.
- **tvr bucketing** (homogeneous groups).
- **Factor risk model** with PCA truncation (<= 50 factors) + **Ledoit-Wolf** shrinkage in the baseline covariance.
- **Turnover / anchor / coupling penalties** in the QP, which shrink the solution toward the baseline MVO and toward yesterday's positions.
- **Walk-forward refit** on `rebalance_dates_mask` with a fixed `LOOKBACK = 1008`.
- **Return capping** (+/-0.15) and **leverage cap**.

**How was the idea validated across different regions, sub-universes, and alpha sets - not just on one favorable backtest?**

The idea was validated on **US top1000** and on **US + EU + JP** (eu top600 / jp top600 / us top1000), over **1y / 2y / 4y** windows, and against multiple ablations (H = 1/5/10; default-construct vs MPO; equal-weight ridge vs MPO; raw ridge returns; `k_tvr = 0`). The idea is strongest and most consistent in **US**; multi-region 1-year is the weak point (see Section 16).

---

## 10. Training / Refit

**At what cadence is the model re-fit (daily, quarterly)?**

The model is re-fit on **`rebalance_dates_mask`** dates - i.e. the standard TQ100m rebalance cadence (quarterly), not daily. Between refits, `construct_preA` re-uses the most recent model and only re-runs the (cheap) inference QP each day.

**What lookback window is used to fit the model and how was its length chosen?**

`LOOKBACK = 1008` trading days (approximately 4 years) for both ridge training and the PCA risk window. Chosen to match the 4-year ranking horizon and to give the ridge enough stock-day observations while staying responsive.

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

- **`full`** (backtest): iterate every day, map to the active refit model, build the alpha cube for the refit's date span, run `fast_fit`, and write `preA[:, delaydi]`. Uses `preA[:, delaydi-1]` as the previous-day anchor.
- **`last`** (live / ffw): take the most recent refit <= today, load **only the single current day's** alpha cube (`sind = eind = delaydi`), and run one `fast_fit` to emit today's vector. The `_ffw_fix` in the filename refers to the fast-forward path: `fast_fit` will opportunistically use `feature_mode="full"` if the cached alpha cube already covers today to reconcile last-mode with full-mode output.

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

If `selected_idx_di.size == 0` the date is skipped in `fit()` and returns zeros in `construct_preA`; if features end up empty, `fast_fit` returns a zero vector. With a single bucket, `_compress_by_buckets` short-circuits.

**What happens when an alpha is all-zero or all-NaN over the lookback?**

Dropped in `create_feature` - today |sum| < 1000 or > 90% zero cells -> skip alpha; stocks > 90% zero -> dropped. Empty training matrices -> a **zero-coefficient ridge model** is stored.

**How does it behave when bucketing produces an empty bucket?**

`rank_equal_groups` produces contiguous equal-size groups, so a bucket can only be empty if there are fewer alphas than buckets; the per-cluster loop simply produces no slab for an empty cluster and the QP runs on the remaining buckets.

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
