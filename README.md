# MPO Fit Function — Write-up Answers

> Source material: `fitSnarang73Ameshram_MPO_10H_ffw_fix-Copy1.py` and
> `Multi Period Optimisation Fit - Presentation.pdf` (33 slides), plus the toy-example
> data in `mpo_toy_example` / `toy_example_data.json`.
>
> Throughout, code references use `file:line` style against the `.py` file above.

---

## 1. Identification

- **Name of idea:** Multi-Period Optimisation (MPO) Fit Function.
- **Slack channel:** Not specified in the supplied materials (not applicable / to be filled by author).
- **preTQ100m_xxx label name:** `preTQ100m_mpo_fit` (slides 5–7).
- **Author(s):** Sanchit Narang, Alankar Meshram (title slide).
- **Function type:** **Fit** function (a `fit_function` class with `fit()` + `construct_preA()`; file `fit_function` at line 874). It also embeds a portfolio-construction optimizer inside `construct_preA`, but in the TQ100m pipeline it is registered as a fit function (filter → **fit** → post-processing).

---

## 2. Idea & hypothesis

**One-paragraph description.** Standard mean-variance optimisation (MVO) is *single-period*: it
optimises today's portfolio in isolation, ignoring how positions must evolve over the coming days.
This produces high day-to-day turnover, because a position taken today can be reversed tomorrow when
the next single-period solve flips its sign. The MPO fit function instead forecasts each stock's
**cumulative excess return over H future horizons (H = 10 days)** using a bank of per-horizon ridge
models, and then solves a **single multi-period quadratic program** that jointly chooses the portfolio
path `x[1..H]`. The objective trades expected return against factor risk, distance from a baseline MVO
portfolio, **turnover versus yesterday's positions**, and a coupling penalty that keeps consecutive
horizons close to each other. The decay-weighted aggregate `x̄` of the per-horizon solutions becomes
the preA. The net effect is a **low-turnover, path-aware preA** with the same or better information
ratio at matched turnover.

**Intuition / market hypothesis.** A signal that is informative about *both* tomorrow's and next
week's returns should be traded smoothly rather than churned. By looking H days ahead and penalising
path movement, the optimiser only trades when the multi-horizon forecast genuinely changes — it does
not react to one-day noise that the next day's solve would undo. Crucially (slide 26), turnover falls
as H increases **even with the explicit turnover penalty switched off** (`k_tvr = 0`), which shows the
turnover reduction is a genuine multi-period effect, not just a penalty artefact.

**What it builds on / extends.** It extends the existing single-period **MVO fit** (baseline
`run_mvo_qp`, file:528) by (a) replacing the alpha-weight perturbation with **stock-level return
forecasting** via ridge (slide 15), and (b) wrapping the result in a **multi-period QP** (`fast_fit`,
file:545). The MVO weights are still computed and used only as the *baseline anchor* `x_mvo_base`
(slide 11 — "any model can be used here"). Conceptually it follows Boyd et al., *Multi-Period Trading
via Convex Optimization* (arXiv:1705.00109, cited on slide 8).

---

## 3. Implementation overview

High-level pipeline:

**`fit()` (file:911), run per rebalance date:**
1. Select alphas where `filter_matrix[:, col_i] > 0` (file:954).
2. **Bucket → super-alphas:** `_make_training_matrices` (file:447) ranks the selected alphas by their
   mean `BUCKET_METRIC` (`dailytvr_top3000top1200`) into `N_BUCKETS_MPO = 20` equal-size groups,
   sums the (booksize-normalised) alpha positions inside each group → 20 "super-alpha" position
   slabs (stock × date).
3. **Build H = 10 ridge models** (file:982): for each horizon `h = 1..10`, the target is the
   cumulative `ret1_excess` over the next `h` days (capped ±0.15), the features are the 20 super-alpha
   positions standardised per-column, and the model is `Ridge(alpha=1e5, fit_intercept=False)`.
   Stores model + scaler per horizon.
4. **Baseline MVO** (file:1039): `run_mvo_qp` on the selected alphas' `dailypnl` over the lookback →
   long-only, sum-to-one alpha weights `mvo_weights` = baseline anchor.
5. Pickle everything (`mvo_weights`, `ridge_models`, `cluster_dic`, horizon, hyperparameters) into
   `model_dict[date]` (file:1045).

**`construct_preA()` (file:1067), run per day:**
6. Map each day to the most recent prior refit model (file:1171 full mode / file:1085 last mode).
7. **Build today's features** (`create_feature`, file:336): assemble the alpha cube for the selected
   alphas, then `_compress_by_buckets` (file:417) compresses it into the 20 super-alpha columns using
   the previous MVO weights as within-bucket weights → `Z` (stock × n_days × 20).
8. **Per-horizon μ** (file:664): for each horizon, standardise today's super-alpha positions with the
   stored scaler and `ridge_h.predict` → per-stock forecast `μ[h]`; stack into `Mu` (H × S).
9. **Risk model** (file:702): PCA factor model from the trailing `LOOKBACK` window of `ret1_excess`
   (top `k_f ≤ MAX_FACTORS_QP = 50` eigenvectors + diagonal residual variance).
10. **Anchors:** `x_mvo_base` = today's super-alpha positions projected through `w_prev` (file:734);
    `y_prev` = yesterday's preA (turnover anchor, file:741).
11. **Solve MPO QP** (CVXPY/OSQP, file:781–844): maximise decay-weighted
    `Σ_h decay[h]·(μ[h]·x_h − k_var·risk(x_h) − k_base·‖x_h − x_mvo_base‖² − k_tvr·‖x_h − y_prev‖²)
    − k_couple·Σ_h ‖x_h − x_{h-1}‖²`, subject to dollar-neutrality `Σ x̄ = 0` and an L1 leverage cap
    `‖x̄‖₁ ≤ 2·gross_ref`.
12. `x̄ = Σ_h decay[h]·x_h` → write into preA, then `cs_booksize` normalise and mask by `valids`
    (file:1238).

---

## 4. Toy example

Using `toy_example_data.json` (slides 16–18): 4 stocks (AAPL, MSFT, GOOG, NVDA), 3 alphas
(α1 momentum, α2 reversal, α3 quality), H = 3.

**Inputs**
- Today's alpha matrix `A_today` (4 stocks × 3 alphas):
  `AAPL=[1.5,−0.5,1.0]`, `MSFT=[−0.3,1.5,0.6]`, `GOOG=[0.6,0.2,−1.2]`, `NVDA=[−1.2,−1.0,1.5]`.
- Baseline MVO alpha weights `mvo_w = [0.357, 0.319, 0.325]` (≈ equal but tilted by pnl covariance).
- Yesterday's positions `y_prev = [0.3, −0.2, 0.2, −0.3]`.
- Per-horizon ridge coefficients `β[h]`, giving per-stock forecasts
  `Mu = β·A_today` (3 horizons × 4 stocks).

**MSFT walk-through (slide 18).** Its forecast *changes sign across horizons*:
`μ1(MSFT) = −0.15` (short), `μ2 = +0.39`, `μ3 = +0.81` (strong long). A single-period model would
short MSFT today, then likely flip it long tomorrow → wasted turnover. The MPO optimiser sees the whole
path and avoids the round-trip.

**Baseline projection** `x_mvo_base = A_today · mvo_w`, normalised:
`[0.291, 0.209, −0.205, −0.295]`.

**MPO solution** (with `k_mean=1, k_var=0.2, k_base=0.5, k_tvr=2.0, k_couple=0.5`, decay
`[0.563, 0.289, 0.148]`): per-horizon `x[h]` are nearly identical (the coupling + turnover terms pull
them together), and the decay-weighted aggregate normalises to
`x̄ = [0.479, −0.156, 0.021, −0.344]`.

**MPO vs naive baseline.**
| | turnover vs y_prev | realised return |
|---|---|---|
| Raw baseline (single-period) | `tvr_raw = 0.827` | `ret_x_mvo = 0.2458` |
| MPO aggregate `x̄` | `tvr_mpo = 0.447` | `ret_x_bar = 0.2608` |

MPO **roughly halves turnover** (0.83 → 0.45) while **slightly increasing** expected return
(0.2458 → 0.2608) — the core selling point of the idea on a toy scale.

---

## 5. Model

- **Model type:** a **two-stage hybrid**. Stage 1 is a bank of **linear ridge regressions** (one per
  horizon) for return forecasting. Stage 2 is a **convex quadratic program** (mean-variance /
  multi-period optimiser) for portfolio construction. There is also an auxiliary **long-only QP**
  (`run_mvo_qp`, cvxopt) producing the baseline anchor.
- **Why appropriate:** ridge is robust and fast for the noisy, collinear super-alpha features (strong
  L2 shrinkage tames multicollinearity among the 20 buckets); a QP is the natural way to express the
  multi-period mean-variance utility with hard dollar-neutral / leverage constraints and quadratic
  turnover/coupling penalties, and it is convex so the solve is reliable in production.
- **Key hyperparameters & how chosen:**
  - `HORIZONS = 10` — swept 1 → 5 → 10 (slides 20, 26); 10 gives the largest turnover reduction.
  - `RIDGE_ALPHA = 1e5` — deliberately strong shrinkage to prevent the 20-bucket forecaster from
    overfitting daily return noise.
  - `N_BUCKETS_MPO = 20` — dimensionality reduction of the (potentially thousands of) selected alphas.
  - `LOOKBACK = 1008` (~4 trading years) — fit/risk window.
  - Optimiser weights (`fast_fit` defaults, file:545): `k_mean=1.0`, `k_var (gamma)=0.5`,
    `k_base (lambda_preA)=0.05`, `k_tvr (lambda_turnover)=0.1`, `k_couple (epsilon)=0.05`.
  - `TAU_DECAY`/decay: `tau = max(1, H/2) = 5`, `decay[h] ∝ exp(−(h−1)/tau)` (file:675).
  - `RETURN_CAP = 0.15`, `MAX_FACTORS_QP = 50`.
- **Approx. # trainable parameters:** 20 super-alpha coefficients × 10 horizons = **200 ridge
  coefficients** (no intercept), plus the baseline MVO weights (one per selected alpha, but these are
  an anchor, not learned by gradient). The PCA risk model and decay weights are not trained.
- **Linear or non-linear mapping?** The alpha→μ stage is **linear** (ridge). The full alpha→preA
  mapping is **non-linear**, because the QP applies inequality (leverage) and equality (dollar-neutral)
  constraints plus the previous-day-dependent turnover term — the output is a piecewise-linear/convex
  function of the inputs, not a simple linear combination.

---

## 6. Inputs / Features

- **What feeds the model:**
  - Alpha **postA positions** (the alpha cube) — the primary features for the ridge models and the QP.
  - Alpha **dailypnl** — only for the baseline `run_mvo_qp` (file:938, 1039).
  - **simres field `dailytvr`** (`BUCKET_METRIC = dailytvr_top3000top1200`) — for bucketing (file:453).
  - **`ret1_excess`** — the target and the source of the PCA risk model.
- **Input shape in a typical fit call:** for each horizon, `_build_Xy_for_horizon` (file:510) reshapes
  the 20 super-alpha slabs over the lookback into `X_h` of shape `(N_obs × 20)` where
  `N_obs ≈ n_valid_stocks × lookback_days` (stacked stock-days), and `y_h` of length `N_obs`. At
  inference the per-day feature is `Z[:, -1, :]` of shape `(S_kept × 20)`.
- **Per-stock or joint?** The ridge **predicts each stock independently** (same coefficients applied
  per stock-row, file:668). The **QP solves all stocks jointly** (the risk model and dollar-neutral /
  leverage constraints couple stocks).
- **Input transformations:** `at_nan2zero` everywhere; super-alphas normalised by `cs_booksize` to 1e6
  (`NORMALIZATION_MODE = "booksize"`, file:489; `zscore`/`rank` are alternative modes); features
  standardised (subtract mean / divide std, stored scaler) before ridge (file:1013, 668).
- **NaN/inf/zero handling:** `n2z` maps NaN/±inf → 0 (file:43); `op.at_nan2zero` on every slab;
  zero-std feature columns get σ = 1 (file:1015); alphas with today |sum| < 1000 or > 90% zeros are
  dropped, and stocks with > 90% zeros are dropped (file:367–392).
- **Bucketing:** yes — into **20 buckets** by **mean `dailytvr`** (file:460, `rank_equal_groups`).
  Rationale: group alphas of similar turnover so the optimiser treats homogeneous-turnover signals
  together, and reduce dimensionality / collinearity before fitting.
- **Super-alphas:** yes — **20** (= `N_BUCKETS_MPO`). Each is the booksize-normalised **sum** of the
  alphas in its tvr bucket (file:480–490).
- **Split method + combiner:** alphas are split by **tvr-rank into 20 equal groups**
  (`rank_equal_groups`, file:462). The combiner across super-alphas is the **per-horizon ridge
  regression** (the 20-coefficient model), and downstream the **multi-period QP** combines the
  per-horizon stock forecasts.

---

## 7. Target

- **Target variable:** `ret1_excess` — specifically the **cumulative h-day forward excess return**,
  `Σ_{k=1..h} ret1_excess(t+k)` for each horizon `h = 1..10` (file:511–514). (When `ret1_excess` is
  unavailable it falls back to `ret1`, file:930.)
- **Transformations:** **capped at ±`RETURN_CAP = 0.15`** (file:935); summed over the next h days
  (cumulative); `at_nan2zero` on the assembled cube.
- **Forward shift / look-ahead control:** uses `op.ts_delay(ret1_excess, −k − delay)` (file:513) so the
  target is strictly post-decision (accounts for the simulation `delay`). The last `delay` columns are
  explicitly set to NaN (file:515) and dropped, so no future information leaks into training.
- **Masking invalid observations:** `valid = np.isfinite(y)` selects only finite target cells
  (file:518), and the same mask is applied to the feature rows (file:524). Targets are further bounded
  by the universe (`ret1_excess * valids`, file:927).
- **Why appropriate:** the QP needs an expected-return vector **per horizon**; cumulative h-day
  forward return is exactly the quantity each horizon's position should be paid for, so forecasting it
  directly aligns the model output with the optimiser's objective. Capping protects the ridge fit from
  earnings-day / outlier return blow-ups.

---

## 8. Loss / Objective

- **Ridge loss:** L2-penalised mean-squared error, `‖Xβ − y‖² + α‖β‖²` with `α = 1e5`,
  `fit_intercept=False` (file:1021).
- **Optimiser objective (`fast_fit`, file:798–815):** maximise the decay-weighted multi-period
  mean-variance utility
  `Σ_h decay[h]·( μ[h]·x_h − k_var·(factor + diag risk) − k_base·‖x_h − x_mvo_base‖²
  − k_tvr·‖x_h − y_prev‖² ) − k_couple·Σ_{h>0} ‖x_h − x_{h−1}‖²`.
- **Hard constraints (file:821–827):**
  - **Dollar-neutral:** `Σ x̄ = 0`.
  - **Leverage cap:** `‖x̄‖₁ ≤ 2·gross_ref` (gross_ref = max of prev/MVO gross, ≥ 1).
  - Baseline `run_mvo_qp` additionally imposes **non-negativity** (`w ≥ 0`) and **sum-to-one**
    (`Σw = 1`) on the alpha weights (file:535–538).
- **Convexity / uniqueness:** the MPO QP is **convex** (PSD quadratic risk + convex penalties, linear
  constraints) → a **unique** optimum (solved by OSQP). The baseline QP is likewise convex (cvxopt).
  Ridge is strictly convex → unique β.
- **Regularization:** ridge **L2 (`α=1e5`)**; the optimiser's `k_base`, `k_tvr`, `k_couple` quadratic
  penalties act as Tikhonov-style regularisers on the portfolio; risk model uses **Ledoit-Wolf
  shrinkage** (file:73) in the baseline and eigenvalue clipping / `nearest_psd` for stability; PCA
  truncation to ≤ 50 factors.

---

## 9. Overfitting controls

- **Explicit techniques:**
  - Strong **ridge L2 (α = 1e5)** on the return models.
  - **Super-alpha pooling**: the feature space is reduced to 20 buckets regardless of how many alphas
    are selected, drastically cutting parameter count.
  - **tvr bucketing** (homogeneous groups).
  - **Factor risk model** with PCA truncation (≤ 50 factors) + **Ledoit-Wolf** shrinkage in the
    baseline covariance.
  - **Turnover / anchor / coupling penalties** in the QP, which shrink the solution toward the baseline
    MVO and toward yesterday's positions.
  - **Walk-forward refit** on `rebalance_dates_mask` with a fixed `LOOKBACK = 1008`.
  - **Return capping** (±0.15) and **leverage cap**.
- **Cross-validation breadth:** the deck validates the idea on **US top1000** and on
  **US + EU + JP** (eu top600 / jp top600 / us top1000), over **1y / 2y / 4y** windows (slides 29–32),
  and against multiple ablations (H = 1/5/10; default-construct vs MPO; equal-weight ridge vs MPO;
  raw ridge returns; `k_tvr = 0`). The idea is strongest and most consistent in **US**; multi-region
  1-year is the weak point (see §16).

---

## 10. Training / Refit

- **Cadence:** the model is re-fit on **`rebalance_dates_mask`** dates (file:947) — i.e. the standard
  TQ100m rebalance cadence (quarterly), not daily. Between refits, `construct_preA` re-uses the most
  recent model and only re-runs the (cheap) inference QP each day.
- **Lookback:** `LOOKBACK = 1008` trading days (~4 years) for both ridge training (file:964) and the
  PCA risk window (file:698). Chosen to match the 4-year ranking horizon and to give the ridge enough
  stock-day observations while staying responsive.
- **Walk-forward setup:** **rolling** window — each refit uses `[di − delay − lookback, di − delay]`
  (file:964–965), and `FIT_STARTDATE = 20210101` gates the first refit. In full mode, model `i` is
  applied to all days between refit `i` and refit `i+1` (file:1171–1181), so there is no look-ahead.

---

## 11. Outputs

- **What it returns:** the **preA** — per-stock, per-day target positions.
  - `fit()` returns `(model_dict, filter_matrix_copy)`; `model_dict[date]` holds the pickled per-refit
    model (mvo_weights, 10 ridge models, cluster_dic, hyperparameters).
  - `construct_preA(mode="full")` returns the full preA matrix; `mode="last"` returns the single most
    recent day's preA vector.
- **Shape / dtype / ordering:**
  - Full mode: `(numstocks × numdates)`, `float32`, Fortran-ordered (file:1167), `cs_booksize`-scaled
    to `data["booksize"]`, masked by `valids`, with zeros mapped to NaN (`at_zero2nan`).
  - Last mode: `(numstocks,)`, `float32` (file:1160).
  - Ordering is the standard stock × date layout aligned to `data["dates"]`.

---

## 12. Inference / Productionization

- **`full` vs `last` mode (file:1067):**
  - **`full`** (backtest): iterate every day, map to the active refit model, build the alpha cube for
    the refit's date span (file:1204), run `fast_fit`, and write `preA[:, delaydi]`. Uses
    `preA[:, delaydi-1]` as the previous-day anchor.
  - **`last`** (live / ffw): take the most recent refit ≤ today, load **only the single current day's**
    alpha cube (`sind = eind = delaydi`, file:1112), and run one `fast_fit` to emit today's vector. The
    `_ffw_fix` in the filename refers to the fast-forward path: `fast_fit` will opportunistically use
    `feature_mode="full"` if the cached alpha cube already covers today (file:570–584) to reconcile
    last-mode with full-mode output.
- **Uses yesterday's positions?** **Yes.** `y_prev` is the turnover anchor in the QP. In last mode it
  comes from `data["get_yesterday_preA"]()` (file:1129); in full mode from the previous preA column
  (file:753). Because gross can differ between live and backtest, `fast_fit` **rescales** `y_prev` when
  `gross_prev > 10×gross_mvo` (file:766–779) so the turnover term stays well-scaled.
- **GLOBAL / region slicing:** there is **no explicit GLOBAL branch** in the code — it operates on
  whatever universe `valids` defines and is masked by `valids` at the end (file:1238). Multi-region
  behaviour is handled by running it over the combined universe (the US+EU+JP rankings, slides 31–32),
  not by special-casing GLOBAL. (Single-region vs multi-region performance differs materially — see
  §15–16.)

---

## 13. Portfolio construction

- **Model output → preA:** the QP's decay-weighted aggregate `x̄` (per-stock) is written directly into
  the preA column for that day (file:1236). `x̄` is already in stock space (the ridge predicts
  per-stock returns and the QP optimises per-stock positions).
- **Cross-sectional normalisation:** yes — final preA is `cs_booksize`-normalised per day to
  `data["booksize"]` (file:1239), and intermediate aggregates are abs-sum normalised (file:332, 626).
- **Neutrality at this stage:** the QP enforces **dollar-neutrality** explicitly (`Σ x̄ = 0`,
  file:822) and a **leverage cap** (file:827). **Beta/sector neutrality are deferred** to
  post-processing — they are not imposed here. (Factor *risk* is penalised via the PCA model, which is
  a soft control, not a hard neutrality constraint.)
- **Universe enforcement:** `op.at_mask(preA, data["valids"])` then `cs_booksize` (file:1239); only
  `selected_valids_today_mvo` stocks are ever assigned non-zero values (file:1236).

---

## 14. Edge cases

- **Zero / one / very few alphas selected:** if `selected_idx_di.size == 0` the date is skipped in
  `fit()` (file:955) and returns zeros in `construct_preA`; if features end up empty, `fast_fit`
  returns a zero vector (file:602). With a single bucket, `_compress_by_buckets` short-circuits
  (file:418).
- **All-zero / all-NaN alpha over lookback:** dropped in `create_feature` — today |sum| < 1000 or
  > 90% zero cells → skip alpha (file:367–369); stocks > 90% zero → dropped (file:387–389). Empty
  training matrices → a **zero-coefficient ridge model** is stored (file:1001–1010).
- **Empty bucket:** `rank_equal_groups` produces contiguous equal-size groups, so a bucket can only be
  empty if there are fewer alphas than buckets; the per-cluster loop simply produces no slab for an
  empty cluster and the QP runs on the remaining buckets.
- **Asset-class branches:** the function is **equities-only** (booksize normalisation, `ret1_excess`,
  alpha cube). No futures branch.
- **Region branches:** none explicit (see §12); same code path for single- and multi-region.
- **Known failure modes & guards:** singular/near-PSD covariance → `nearest_psd` (file:47),
  eigenvalue clipping (file:709), `+1e-8·I` ridge (file:76); **OSQP non-convergence / no solution** →
  caught and **falls back to the decay-weighted mean of `Mu`** (file:867–869); short return history
  (`ret1.shape[1] < 2`) → equal-weight bucket fallback (file:636); missing ridge models → equal-weight
  fallback (file:641). No special OOM handling beyond `float32` and the 20-bucket compression.

---

## 15. Robustness / generality

- **Regions tested:** **US** (top1000) and **US + EU + JP** (eu top600 / jp top600 / us top1000),
  across 1y / 2y / 4y (slides 29–32). EU/JP/ASIA were not tested in isolation in the deck; GLOBAL is
  only via the combined universe.
- **Sub-universes:** primarily **top1000** (US) and **top600** (EU, JP). The bucketing metric is
  `dailytvr_top3000top1200`, and one comparison link references **TOP3000** (slide 21), so it has been
  exercised on broader universes too.
- **Very few / very many alphas:** the 20-super-alpha compression makes it **scale-robust to many
  alphas** (cost is dominated by the fixed 20 buckets and the stock-level QP, not the alpha count). For
  **very few alphas** it degrades gracefully via the single-bucket short-circuit and equal-weight
  fallbacks, though with < 20 alphas the bucketing becomes degenerate (some buckets hold one alpha).

---

## 16. Benchmark comparison

**(a) With-idea vs without-idea (default construct_preA), US, matched turnover (slide 21).**
`fitSnarang73_MPO_baseTest_defaultCons` returns the **baseline MVO weights with no MPO optimisation**;
`fitSnarang73_MPO_baseTest_10H` is the **same baseline + MPO (H=10)**. At the preA level the MPO
version has materially **lower turnover**; after equalising postA turnover (`pp_basic_hump`,
target_tvr = 0.10), **MPO beats the default in IR, RET and LIQ** (links S15721 / S15725, also TOP3000).

**(b) Doing the opposite of the idea.** The opposite of "look many horizons ahead" is **H = 1**
(single-period). Sweeping H = 1 → 5 → 10 (slide 20, S15719/S15720) shows turnover **falls
monotonically** as horizons increase. Decisively, with the explicit turnover term **switched off
(`k_tvr = 0`, slide 26, S15783/S15784)**, turnover *still* falls as H increases — so removing the
multi-period structure (the thing the idea adds) is what hurts, confirming the effect is genuine and
not a penalty artefact. Equal-weighting the ridge returns instead of using the optimiser
(`..._ridge_ret_ew_mean`, slide 22, S15727/S15728) gives **worse IR/TVR/LIQ** (corr 64% to MPO),
showing the optimiser itself adds value.

**(c) Function rankings — `fitSnarang73Ameshram_MPO_10H_ffw_fix` (with-idea) (slides 29–32).**
Columns: improvement % (vs best in pool), IR, turnover, IR/√TVR, and **max correlation to a function
it does not outperform**.

| Universe | Window | Rank | Improv% | IR | TVR | IR/√TVR | Max corr (to a better fn) |
|---|---|---|---|---|---|---|---|
| US top1000 | 1y (252d) | ~6 / 66 | 53.1% | 0.055 | 0.108 | 0.173 | **1.000** |
| US top1000 | 2y (502d) | ~6 / 66 | 66.6% | 0.077 | 0.112 | 0.237 | **1.000** |
| US top1000 | 4y (1004d) | ~6 / 66 | 65.3% | 0.065 | 0.118 | 0.202 | **1.000** |
| US+EU+JP | 1y (252d) | ~6 / 198 | **−42.2%** | **−0.002** | 0.129 | **−0.020** | **1.000** |
| US+EU+JP | 2y (502d) | ~6 / 198 | 49.6% | 0.027 | 0.131 | 0.072 | **1.000** |
| US+EU+JP | 4y (1004d) | ~7 / 198 | 57.7% | 0.043 | 0.133 | 0.094 | **1.000** |

Notes: it sits **~top 9%** of functions in US and is solidly positive over 2y/4y; the **multi-region
1-year window is negative** (the main weakness). The **max correlation is ~1.000 in every window**,
meaning there is a near-identical function ranked above it (almost certainly the sibling MPO/baseline
variant by the same authors) — i.e. it does not add diversification on top of that twin, only on top
of the rest of the pool.

**(d) preTQ100m_mpo_fit label OS pnl (slides 6–7).**
- Last **60 OS days** (20260218–20260513): **IR = 0.243** vs `trex_ALL_gross` 0.189, `trex_USA_gross`
  0.184 — **outperforms both**.
- Full OS (**88 OS days**, from 20260107): **IR = 0.076** vs `trex_ALL_gross` 0.042, `trex_USA_gross`
  0.063 — **outperforms both**.

**(e) Benchmark strategy vs with-idea (slides 3–4).** MPO-fit bmark strategy vs `_beat_the_bmark`:
- 60 OS days (20260218–20260513): **IR_MPO = 0.198** vs 0.092; **correlation 33.7%**.
- Full OS (88 days, from 20260107): **IR_MPO = 0.091** vs 0.070; **correlation 26.3%**.
- No recent drawdown in the MPO strategy over either window.

---

## 17. Sensitivity / hyperparameter analysis

- **Swept parameters:**
  - **Number of horizons H ∈ {1, 5, 10}** (slides 20, 24, 25, 26).
  - **Optimiser-on vs optimiser-off** (MPO QP vs raw/equal-weight ridge returns, slides 22, 24).
  - **Explicit turnover term `k_tvr ∈ {default, 0}`** (slide 26).
  - **Which horizon's prediction to trade** (1st / 5th / 10th / mean, slide 24).
- **Sensitivity:**
  - **H:** the dominant lever — turnover falls and IR/LIQ improve monotonically as H goes 1 → 10; the
    longest-horizon and mean-of-horizons predictions are the lowest-turnover (slides 24–25).
  - **Optimiser:** removing it (equal-weight ridge returns) costs IR/TVR/LIQ (slide 22).
  - **`k_tvr`:** turnover reduction survives `k_tvr = 0`, so the result is **not fragile** to the
    explicit turnover weight — the multi-period structure carries it (slide 26).
  - **Ridge α = 1e5:** intentionally large; the forecaster is robust because of pooling + shrinkage.
- **Recommended defaults:** `HORIZONS = 10`, `RIDGE_ALPHA = 1e5`, `N_BUCKETS_MPO = 20`,
  `LOOKBACK = 1008`, decay `tau = H/2`, optimiser weights `k_mean=1.0, k_var=0.5, k_base=0.05,
  k_tvr=0.1, k_couple=0.05`. Rationale: H=10 maximises the turnover/IR trade-off; strong ridge +
  20 buckets control overfitting; the QP weights balance return vs path-stability.
- **Fragility:** the clearest fragility is **region/window** rather than a numeric knob — the
  **US+EU+JP 1-year** result is negative while US is robustly positive, so the idea is sensitive to the
  universe it is deployed on. No single hyperparameter was reported to flip the outcome within its
  swept range.

---

## 18. Limitations & future work

- **Known weaknesses / open questions:**
  - **Multi-region 1-year underperformance** (improv −42.2%, IR −0.002) — needs investigation /
    region-specific tuning; the idea is validated mainly on US.
  - **Max correlation ≈ 1.000** to a higher-ranked twin function in every ranking window — it adds
    little on top of that sibling, so its incremental value over the existing book should be quantified
    before allocation.
  - Beta/sector neutrality are **not** enforced at preA stage; reliance on post-processing.
  - Bucketing degenerates when fewer than 20 alphas are selected.
  - The baseline MVO anchor is somewhat arbitrary (slide 11 explicitly says "any model can be used").
- **Compute / memory profile:** each rebalance trains **10 ridge models** on stacked stock-day
  matrices (cheap, sklearn). The per-day cost is a **CVXPY/OSQP QP in stock space** with an H-horizon
  factor risk model (the heaviest piece; RT ≈ 5 min in the rankings, file:836 caps `max_iter=4000`).
  Memory is bounded by `float32` cubes and the **20-bucket compression**, so it scales well with the
  number of alphas. Cheaper paths: cache/limit `MAX_FACTORS_QP`, reduce H, solve a single aggregated
  QP rather than per-horizon variables, or warm-start across days (already `warm_start=True`).
- **Variants tried and rejected:**
  - **H = 1 / H = 5** (rejected: higher turnover than H = 10).
  - **Raw ridge returns as preA** — 1st / 5th / 10th / mean horizon, no optimiser (rejected: worse
    IR/LIQ; only useful to demonstrate the turnover-vs-horizon relationship — slides 24–25).
  - **Equal-weight mean of ridge returns** (rejected: optimiser adds IR/TVR/LIQ — slide 22).
  - **Default construct_preA** (baseline MVO weights, no MPO) — rejected: MPO wins at matched postA
    turnover (slide 21).
  - **Alpha-weight perturbation** (the earlier approach) — replaced by **stock-level return**
    optimisation (slide 15).
