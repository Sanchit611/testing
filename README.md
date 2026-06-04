# Fit Function Review Answers

## 1. Identification

**Q: Name of idea**
CNN based Fit Function

**Q: Author(s)**
Sanchit Narang, Harsh Singhal

**Q: Function type (filter, fit, post processing, other)**
Fit function.

## 2. Idea & hypothesis

**Q: Describe the idea in one paragraph.**
This fit assigns each alpha a weight by reading its full PnL trajectory over a trailing window. A 1D CNN extracts shape features from each alpha's PnL curve on its own, an attention step learns which parts of the window matter most, and a mixing layer lets the alphas interact so correlated ones are not all rewarded at once. The weights are trained to maximize adjusted returns over the next, held out window while penalizing turnover and concentration. The final weights are normalized and capped, then combined with the alpha positions to form `preA`.

**Q: What is the intuition or market hypothesis behind the idea, why should it work?**
The idea rests on shape. An alpha's PnL curve can be grinding steadily, spiking and about to revert, or recovering off a drawdown, and these behave differently going forward even when their recent average looks similar. A convolution is built to pick up shape (a `[-1, 0, 1]` kernel, for instance, detects a local trend), so the model learns from the data which shapes tend to lead to strong forward PnL rather than assuming any single pattern.

**Q: What existing function variants does it build on, replace, or extend?**
It builds on the non linear, class based template (`fit_function_nonlinear.py`).

## 3. Implementation overview

**Q: Describe implementation of the idea in as much detail as you think necessary.**
The implementation has two phases: a training phase (`fit`) at each rebalance date, and a construction phase (`construct_preA`) that turns the trained model into daily positions.

In training, the fit function loads PnL for the filter selected alphas, applies a quality pre filter that drops persistently losing and inconsistent alphas (keeping all if fewer than five survive), builds rolling windows each paired with a forward PnL target, and trains the CNN under the custom loss. Training only ever uses windows whose target ends before the refit date, so no future data leaks in. Each refit saves the CNN weights, the selected alpha set, and the weight vector the model predicts on the latest window.

In construction, `construct_preA` runs inference on recent PnL: in daily mode (the default) it feeds the most recent window through the saved model each day to get that day's weights; in quarterly mode it reuses the saved weight vector. The per day weights are contracted with the alpha cube `(stocks × days × alphas)` via `einsum` so each stock's position is the weighted blend of the selected alphas' positions, then masked to the universe and scaled to booksize to form `preA`.

**Q: List the high level steps of the function (e.g., load alphas → bucket → build features → fit model → combine buckets → construct preA).**

select alphas → load data → quality filter → normalize + build windows/target → train CNN → predict & post process weights → construct preA (combine with cube, mask, booksize)

*Training phase (`fit`), repeated at each rebalance date:*

1. **Iterate over rebalance dates.** Skip dates before `LOOKBACK` (1000), before `FIT_STARTDATE` (20230101), or where `data['rebalance_dates_mask']` is False.
2. **Select alphas** from the filter matrix at that date.
3. **Load data:** pull the selected alphas' PnL and turnover over the lookback window and clean the PnL.
4. **Quality filter:** score each alpha on its recent performance and consistency, and drop the persistently losing and unreliable ones so the model trains on a cleaner pool.
5. **Prepare data:**
   * Normalize the PnL.
   * Build rolling windows paired with a forward PnL target, and split chronologically into train and validation.
6. **Train the CNN** under the custom loss, with early stopping on the validation split.
7. **Predict** the weights on the most recent window, then normalize and cap them so no single alpha dominates.
8. **Persist** `{selected_idx, alpha_weights, model_state}` per refit date (pickled into `model_dict`), and return `(model_dict, copy(filter_matrix))`, the filter matrix doubling as alpha attribution.

*Construction phase (`construct_preA`):*

9. **`construct_preA`** walks refit date intervals, generates weights per day (daily mode) or tiles cached weights (quarterly mode), and applies `einsum('sda,da->sd', cube, weights)` → `at_mask(valids)` → `cs_booksize` → `at_zero2nan`.

## 4. Toy example

**Q: Walk through the idea on a handful of alphas / data points, showing inputs (positions or pnls) and the resulting weights / preA.**

Toy setup: 3 alphas (A, B, C) over a 6 day PnL window. (The real pipeline also normalizes the PnL first; omitted here for clarity.)

|Day| 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| A | 1 | 2 | 3 | 4 | 5 | 6 |
| B | 1 | 3 | 5 | 4 | 2 | 1 |
| C | 3 | 3 | 3 | 3 | 3 | 3 |

A is in a clean uptrend, B peaks and reverts, C is flat.

**Step 1, depthwise temporal conv.** A learned kernel slides across each alpha independently (no cross alpha mixing in this stage). For illustration take kernel `[−1, 0, 1]`, a trend detector. Applied across each alpha it gives per alpha feature sequences:
* A → `[2, 2, 2, 2]`, consistently trending up
* B → `[4, 1, −3, −3]`, sign flip, reversal detected
* C → `[0, 0, 0, 0]`, no signal

**Step 2, temporal attention.** Per alpha softmax weights over the 4 timesteps decide which moments matter, then collapse each alpha to one scalar. With recent focused weights for A, reversal focused for B, and uniform for C: A → `2`, B → `−1.9`, C → `0`.

**Step 3, cross alpha mixing + output head.** The mixing layer combines all three alphas' shape reads into a shared summary, and the output head then decodes one weight per alpha from that summary. So a weight reflects each alpha's shape relative to the others, not in isolation: here the network tilts toward the trending alpha and away from the reverting and flat ones, giving a raw tilt of roughly **A = 0.70, B = 0.24, C = 0.06**.

**Step 4, post process.** The raw weights are normalized and capped at `MAX_WEIGHT = 0.08`. The cap exists to stop any single alpha from dominating, so on a large pool it flattens extreme tilts while keeping the ordering the network found (trend > reversal > flat). On this 3 alpha toy the cap pulls the top weights down toward each other, but A still ranks above B above C.

**Q: Show how the model output differs from a naive baseline (equal weight) on the same toy input.**
Equal weight assigns `A = B = C ≈ 0.33`, blind to the shapes. The CNN instead reads the trajectories and leans toward the trending alpha (A) and away from the reverting (B) and flat (C) ones. The hard weight cap then keeps the final book from over concentrating, but the tilt the network found, favoring the trend, is preserved, which is exactly the distinction equal weight cannot make.

## 5. Model

**Q: What type of model is used?**
A CNN, specifically a 1D temporal convolutional network with a temporal attention layer and a cross alpha mixing layer. Its four stages are:

1. a stack of **depthwise dilated conv blocks** that read each alpha's PnL curve on its own at several time scales;
2. a **temporal attention** layer that pools each alpha's window down to a single feature;
3. a **cross alpha mixing** layer that mixes across alphas into a small shared state;
4. a **linear head** that expands that state back to one weight per alpha.

**Q: Why is this model class appropriate for the idea?**
The CNN is the right class because each part matches what the idea needs. The idea is to read each alpha's PnL shape, and a convolution is the natural tool for reading shape, so it extracts shape features from each alpha's curve on its own. The idea cares about which moments in the history matter, so the attention step learns which parts of the window matter most. And the idea needs the alphas weighted jointly, not in isolation, so the mixing layer lets them interact and avoids rewarding correlated ones twice.

**Q: What are the key model hyperparameters and how were they chosen?**
The ones that carry design intent:
* `CNN_WIN = 252`, one trading year of PnL per sample.
* `CNN_FORWARD_HORIZON = 21`, a one month forward horizon, long enough to average out daily noise but short enough to stay relevant.
* `MAX_WEIGHT = 0.08`, the per alpha weight cap that keeps any one alpha from dominating.

The remaining settings are standard training knobs (Adam learning rate and weight decay, batch size, dropout, epochs with early stopping, kernel size and dilations). The forward horizon was tuned by a sweep (see §17); the rest are hand set defaults.

**Q: Approximately how many trainable parameters does the model have?**
Around 20k parameters for a typical alpha pool.

**Q: Is the mapping from alphas to preA linear or non linear?**
Non linear: the CNN maps PnL to weights non linearly, then those weights combine with the alpha positions in a linear step to form `preA`.

## 6. Inputs / Features

**Q: What feeds the model?**
The network is fed only the alphas' **daily PnL**. Turnover also goes in, but into the loss rather than the network.

**Q: What is the shape of the model input in a typical fit call?**
`(batch, 252, A)`: a batch of 252 day windows across `A` alphas (batch size 64 in training, 1 at inference).

**Q: Does the model make predictions for each stock independently or for multiple stocks at once?**
Neither: outputs are at the **alpha** level (one weight per alpha). Stock positions are formed afterward by combining the alpha cube with the weights.

**Q: What transformations are applied to inputs?**
The PnL is first scaled and winsorized to clip extreme days. Each window's daily PnL is then turned into a cumulative curve (a running sum) and divided by its own running standard deviation, a causal normalization so no day uses future information. The effect is that the network sees the normalized *shape* of each alpha's cumulative PnL curve rather than its raw magnitude.

**Q: How are NaNs / infs / zeros cells handled?**
NaNs and infs are turned into 0 wherever data is loaded or normalized, and again before the weights are combined into `preA`. Zeros are handled carefully: a zero PnL is treated as a real no profit day, not as missing data, and anywhere we divide, a zero denominator is avoided (a small floor is added, or the step is skipped) so a flat or empty series can't break the math. Turnover is also kept inside a safe range.

**Q: Are alphas bucketed before fitting?**
No. All kept alphas go into one network together.

**Q: Are super alphas (dimensionality reduction features) created, if yes how many?**
No, no super alphas are constructed. The model outputs one weight per raw alpha.

**Q: If super alphas are constructed, what method splits alphas into clusters, and what model combines them?**
Not applicable, no super alphas are constructed.

## 7. Target

**Q: What target variable is used?**
Custom: each alpha's own forward PnL, averaged over the next few weeks.

**Q: What transformations are applied to the target?**
The future PnL is scaled and winsorized like the inputs, then averaged over the forward window to give one smoothed target value per alpha.

**Q: How is the target shifted to be next day / post decision, and how is forward bias avoided?**
The target window always sits strictly after the feature window, so a training sample never sees its own future. When loading PnL, the function stops a few days before the refit date, leaving a buffer for the trade delay and for the latest day's PnL that is not yet known. The within window normalization is causal, using only past days. And the filter selection it reads is already delay shifted before it reaches the fit. Together these mean there is no forward bias.

**Q: How are missing or invalid target observations masked?**
There are no invalid targets to mask: the target is the alpha's own PnL, which is always defined. Missing PnL is set to 0 before the data is built, and if the forward window runs past the end of the data it falls back to the last available day.

**Q: Why is this target appropriate for the prediction problem the model is solving?**
The premise is that alpha quality drifts over time and that an alpha's recent PnL shape says something about its near term PnL. Scoring the weights against forward PnL lines the training objective up with how the strategy is actually traded, and averaging over a forward window instead of a single day cuts the noise in the target, which helps when there are only a limited number of windows per refit.

## 8. Loss / Objective

**Q: What is the loss or objective function?**
A custom loss, `f(adjusted_returns, turnover_penalty, sparsity_penalty)`, minimized, combining three terms:
* **adjusted_returns:** rewards weights that earn forward PnL, adjusted down for putting too much weight on alphas that tend to move together, so the book stays diversified.
* **turnover_penalty:** penalizes weight on high turnover alphas, nudging the book toward cheaper to trade signals.
* **sparsity_penalty:** a regularization term that keeps most weights near zero to prevent overfitting and keep the book robust.

**Q: Are there hard constraints on the solution?**
The constraints are applied after the network, in post processing: tiny weights are zeroed, the weights are normalized, and each is capped so no single alpha dominates. The network output itself is unconstrained, including sign, so the model can both add and subtract alpha exposure.

**Q: Is the objective convex? Is the solution unique?**
Like any neural network, the overall training objective is non convex in the model's parameters.

**Q: What regularization is applied?**
Weight decay, dropout, early stopping on a held out validation split, gradient clipping, the L1 style turnover and sparsity penalties in the loss, and the post processing weight cap.

## 9. Overfitting controls

**Q: What explicit techniques mitigate overfitting?**
Several layers of protection: dropout, weight decay, gradient clipping, and early stopping during training; the turnover and sparsity penalties plus a hard weight cap so no single alpha dominates; a quality pre filter that removes weak alphas before training; and walk forward refitting, so each refit only ever sees past data. Together these keep the model from concentrating on a few alphas or overfitting the training window.

Because the training windows overlap heavily, the number of effectively independent observations per refit is modest relative to the parameter count, which is exactly why this regularization and the small model size matter. The strongest evidence that the model captures real signal rather than overfitting is the opposite of idea test and the out of sample results (§16): a pure overfit would not flip cleanly from positive to negative when the weights are reversed.

**Q: How was the idea validated across different regions, sub universes, and alpha sets?**
The model was developed and tested on US. To check it generalizes, the fit ranking was run across regions (US, and US + EU + JP), and the function held up well in those rankings (§16), which suggests the idea is not specific to one region.

## 10. Training / Refit

**Q: At what cadence is the model re fit?**
The model is retrained on the rebalance schedule (quarterly). Between retrains, with daily updates on (the default), the weights are re predicted every day by running the most recent trained model on that day's trailing window. So the model trains quarterly but the weights refresh daily. In quarterly update mode, the last refit's weights are simply held until the next retrain.

**Q: What lookback window is used to fit the model and how was its length chosen?**
`LOOKBACK = 1000`, a standard template choice of about four years of history, the usual range for this kind of fit.

**Q: How is walk forward set up (rolling vs expanding window)?**
Rolling window: each refit uses a fixed length window that slides forward (older history drops off as new data comes in), so it is rolling, not expanding. Within a refit, the data is split chronologically into training and validation to drive early stopping. Daily inference always uses the latest trailing window.

## 11. Outputs

**Q: What does the model return?**
`fit()` returns the trained models (one per refit date, holding the selected alphas and the model weights) plus an attribution matrix that credits the alphas used. `construct_preA()` returns the stock level `preA`.

**Q: What is the shape, dtype, and ordering of the output?**
`(numstocks, numdates)` float32, Fortran ordered, in `full` mode; a 1D `(numstocks,)` float32 array in `last` mode. Standard stock × date ordering, with NaN meaning no position.

## 12. Inference / Productionization

**Q: How is preA constructed in `full` mode (backtest) vs `last` mode (live forward / ffw)?**
* **`full` (backtest):** the date range is split into intervals between refits. For each interval the saved model produces a weight per day, those weights are combined with the alpha positions, and the result is masked to the universe and scaled to book size to fill in `preA` over the whole history.
* **`last` (live / ffw):** only the latest model is available, so it produces weights for the most recent day, combines them with that day's alpha positions, and returns a single day of `preA`.

**Q: Does the function use yesterday's positions? If yes, how is the dependency handled?**
No.

**Q: How does the function handle the GLOBAL region differently from single region runs?**
GLOBAL is not implemented.

## 13. Portfolio construction

**Q: How are model outputs converted into per stock per day positions (preA)?**
Each day's weight vector is combined with that day's alpha positions, giving each stock a weighted sum of the alphas' positions as its raw position. Daily mode uses a fresh weight vector per day; quarterly mode reuses one weight vector across the interval.

**Q: Is the preA normalized cross sectionally per day?**
Yes. Each day is scaled to the target book size, and exact zeros are turned into NaN (no position). The alpha weights themselves are separately normalized and capped.

**Q: Is the preA explicitly dollar/beta/sector neutral at this stage, or is that deferred to post processing?**
The book size scaling makes the preA dollar neutral and sets its gross exposure; beta and sector neutralization are deferred to post processing.

**Q: How is the trading universe enforced?**
Positions for stocks outside the valid universe are masked out before the book size scaling, so only tradable stocks carry weight.

## 14. Edge cases

**Q: What happens when the filter selects zero / one / very few alphas on a rebalance date?**
If no alphas are selected, that date is skipped (no model). If the quality filter would leave fewer than five, it keeps all of them instead, so the model never trains on an over thinned pool. With just a few alphas the network still builds and produces weights normally.

**Q: What happens when an alpha is all zero or all NaN over the lookback?**
An all NaN alpha becomes all zeros once missing values are filled, so it looks flat to the network. The model gives it little to no weight, and the sparsity step tends to zero out such tiny weights entirely, so a dead alpha contributes effectively nothing.

**Q: How does it behave when bucketing produces an empty bucket?**
Not applicable.

**Q: Are there asset class specific or region specific branches?**
None.

**Q: What known failure modes exist?**
Standard neural network risks, each handled: gradient clipping prevents unstable training, early stopping prevents overfitting, and the per alpha weight cap prevents over concentration on any single alpha. If a refit has too little history, it produces no position instead of a wrong one.

## 15. Robustness / generality

**Q: In which regions has the idea been tested (US, EU, JP, ASIA, GLOBAL)?**
US.

**Q: In which stock sub universes (top250 / top500 / top1000 / top3000)?**
top1000.

**Q: Can the function handle very few (<20) and very many (>5000) alphas?**
Yes to both. A keep all fallback handles the very thin case when fewer than five alphas survive the quality filter.

## 16. Benchmark comparison

**Q: Same function with idea vs without idea, on US top1000 using all gw2 filter/fit/pp with apply extra neut, performance of both.**
On US top1000 with the same filter and pp, the CNN fit clearly beats the without idea baselines (equal weight and linear regression on the same alphas):

| Fit | 1y IR | 2y IR |
|---|---|---|
| **CNN fit (with idea)** | **0.104** | **0.127** |
| Equal weight | 0.013 | 0.025 |
| Linear regression (PnL) | 0.011 | 0.027 |

The CNN fit is the top function on the same filter output at both horizons, well above either baseline.

**Q: Doing the opposite of the idea, what happens?**
We built an opposite variant that takes the trained model's weights and reverses their signed ranking, so the alpha the model is most bullish on becomes the most bearish and vice versa, keeping book size, cap and sparsity identical. Inverting the model flips performance from clearly positive to negative, which confirms the weights carry real signal rather than noise:

| Function | 1y IR | 2y IR |
|---|---|---|
| CNN fit | 0.127 | 0.170 |
| Opposite (signed rank reversed) | -0.029 | -0.050 |

**Q: Function ranking across (US, US+EU+JP) × (4y, 2y, 1y): percentage numbers and max correlation to functions it does not outperform.**
The CNN fit ranks at or near the top of the fit ranking.

| Region | Horizon | Rank | Max corr to functions it does not outperform |
|---|---|---|---|
| USA | 1y | 1st | 54% |
| USA | 2y | 1st | 54% |
| USA | 4y | 5th | ~54% |
| USA + EU + JP | 1y | 4th | 52% |
| USA + EU + JP | 2y | 3rd | 52% |
| USA + EU + JP | 4y | 6th | ~53% |

On USA it ranks first on both 1y and 2y, and the maximum correlation to the functions it does not outperform stays around 52 to 54%, so it is adding largely orthogonal performance rather than tracking an existing function.

**Q: preTQ100_xxx label performance, relevant metrics from label OS pnl.**
The label `preTQ100m_VAPL_CNN` beats Trex over both windows in true OS:

| Window | Label IR | Trex (ALL_GROSS) IR |
|---|---|---|
| 120 day | 0.055 | 0.041 |
| 60 day | 0.241 | 0.229 |

**Q: Benchmark strategy vs with idea comparison: OS days, OS IR, 1y IR, 2y IR, correlation to benchmark.**
Strategies built with the CNN fit beat the benchmark (Joey filter + Tsingh fit) in true OS and stay low correlated to it. Both on USA top1000 with extra neut.

*CNN Filter + CNN Fit vs benchmark:*

| Strategy | OS days | OS IR | 1y IR | 2y IR | Correlation to benchmark |
|---|---|---|---|---|---|
| CNN Filter + CNN Fit | 99 | 0.339 | 0.248 | 0.200 | 22.2% |
| Benchmark | same window | 0.120 | 0.076 | 0.124 | — |

*Joey filter + CNN Fit vs benchmark:*

| Strategy | OS days | OS IR | 1y IR | 2y IR | Correlation to benchmark |
|---|---|---|---|---|---|
| Joey filter + CNN Fit | 74 | 0.167 | 0.128 | 0.138 | 30.1% |
| Benchmark | same window | 0.150 | 0.076 | 0.124 | — |

The CNN Filter + CNN Fit strategy in particular shows no YTD drawdown in true OS.

## 17. Sensitivity / hyperparameter analysis

**Q: Which hyperparameters were swept and over what ranges?**
The forward target horizon (the N day return target) was swept over N = 1, 5, 10, 21, 30, 40, 50, 100, measuring 2y mean IR at each setting.

**Q: How sensitive is performance to each parameter?**
The forward horizon sweep shows a clear, smooth optimum at N = 21:

| N day target | 1 | 5 | 10 | 21 | 30 | 40 | 50 | 100 |
|---|---|---|---|---|---|---|---|---|
| 2y mean IR | 0.100 | 0.129 | 0.148 | **0.152** | 0.144 | 0.143 | 0.141 | 0.135 |

Performance rises sharply from very short horizons (N=1 is the worst, too noisy), peaks at N=21 (~one month), and decays gently for longer horizons as the target goes stale. The curve is smooth, so the choice is robust rather than a sharp spike. For the other knobs: `LOOKBACK` / `CNN_WIN` trade history depth against adaptivity, `MAX_WEIGHT` trades concentration against diversification, and `UPDATE_FREQUENCY` daily reacts faster within a quarter while quarterly is cheaper and steadier.

**Q: What are the recommended default values, and what is the rationale?**
The defaults are the working values, with the forward horizon set to N = 21 because it is the peak of the sweep above (one month forward target). The remaining settings follow the one trading year window and standard regularization heuristics.

**Q: Is there any parameter the idea is fragile to?**
No. The forward horizon sweep is smooth around the optimum, so the result is not fragile to that choice; performance degrades gently rather than flipping with small changes.

## 18. Limitations & future work

**Q: What are the known weaknesses or open questions of the idea?**
Currently it does not support the GLOBAL region.

**Q: What is the compute / memory profile, and is there a path to make it cheaper?**
It runs comfortably within the default permutator setup, no special compute or memory needed. Memory is modest because the model works on alpha PnL rather than full positions.

**Q: What variants were tried and rejected, and why?**
We tried feeding the alphas' full position matrices into the CNN instead of their daily PnL, but it did not work as expected, so we kept the PnL based input.


### Planned next steps

1. Using Claude to generate new orthogonal versions of the idea.
2. Building a variant that uses the alphas' full positions instead of their daily PnL.
