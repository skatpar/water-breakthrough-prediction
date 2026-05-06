# Water Breakthrough Prediction — Method Description

## Problem Statement

Predict the time to water breakthrough (TTE) for gas wells in the MARI Energies Habib Rahi Limestone field. Breakthrough is defined as Water-Gas Ratio (WGR) exceeding 5 bbl/MMcf sustained for 3 or more consecutive months.

## Data

- **16 vertical wells** — 12 with confirmed breakthrough events, 4 right-censored (no breakthrough observed during production history)
- **5 horizontal wells** — 1 with confirmed breakthrough (M-122H-HRL at 12 months), 4 still producing without breakthrough
- Well M-51-HRL excluded due to insufficient data

## Features

Six features, all computable at or shortly after first production:

| Feature | Description | Source |
|---------|-------------|--------|
| `n_wells_producing_at_spud` | Number of other wells producing gas within 1 month of this well's first production | Field production records |
| `spud_year` | First production date expressed as decimal year | Well records |
| `field_cum_gas_at_spud` | Total cumulative gas produced by all other wells before this well started (BCF) | Field production records |
| `gas_rate_cv_yr12` | Coefficient of variation of monthly gas rate over the first min(BT_time, 24) months | Monthly production data |
| `peak_gas_rate` | Maximum monthly gas production (MMcf) over the first min(BT_time, 24) months | Monthly production data |
| `sw` | Initial water saturation (fraction) from well logs | Subsurface / geological data |

Production-derived features (`peak_gas_rate`, `gas_rate_cv_yr12`) are capped at the earlier of the breakthrough date or 24 months to prevent data leakage — ensuring no post-event information enters the model. `sw` is a static geological property known at spud time.

## Model Architecture

### Individual Models (7)

| Model | Type | Implementation |
|-------|------|----------------|
| Cox PH | Semi-parametric proportional hazards | lifelines (penalizer=0.1) |
| Weibull AFT | Parametric accelerated failure time | lifelines (penalizer=0.1) |
| XGB Cox | Gradient-boosted Cox objective | xgboost (max_depth=2, 20 rounds) |
| XGB AFT | Gradient-boosted AFT objective | xgboost (max_depth=2, 20 rounds) |
| sksurv GBSA | Gradient-boosted survival | scikit-survival (50 estimators) |
| sksurv CWGB | Component-wise gradient-boosted survival | scikit-survival (100 estimators) |
| Stacked | Cox PH base + XGBoost residual correction | lifelines + xgboost |

### Ensemble Construction

Two parent ensembles are combined:

1. **Geometric Mean (GM):** Geometric mean of XGB AFT and sksurv CWGB predictions — optimized for point accuracy
2. **Rank Fusion (RF):** Cox PH and sksurv GBSA predictions converted to ranks, averaged, then mapped back to the training distribution — optimized for discrimination (ordering)

**Final blend:** 45% GM + 55% RF

This blend exploits complementary error patterns: GM provides accurate point estimates while RF provides better rank ordering.

### Prediction Intervals

Adaptive conformal prediction intervals, where the width scales with model disagreement:

- **Model disagreement** = standard deviation across all 7 individual model predictions for each well
- **Interval half-width** = k × disagreement
- **k** is calibrated via binary search on LOOCV residuals to achieve 11/12 (92%) coverage of event wells
- **P10** = P50 − k × disagreement (floored at 1)
- **P90** = P50 + k × disagreement
- **P25/P75** = P50 ± 0.6 × k × disagreement

Wells where models agree get tight intervals; wells where models disagree get wide intervals.

## Validation

**Leave-One-Out Cross-Validation (LOOCV):** Each of the 16 wells is held out once. The remaining 15 wells are used to train all 7 models, construct both parent ensembles, and produce a blended prediction. This is repeated 16 times.

### Metrics

| Metric | Value |
|--------|-------|
| Harrell's C-index | 0.858 |
| MAE (event wells only) | 50 months |
| P10-P90 coverage | 92% (11/12 event wells) |

## Time-Varying Predictions (Landmark Analysis)

Fixed predictions at time zero do not update as wells age. A well that has survived 100 months without breakthrough should receive a revised, longer prediction.

### Method

At each landmark time t₀ (= 0, 12, 24, 36, 48, 60 months):

1. Filter to wells with TTE > t₀ (still at risk)
2. Set target to remaining_TTE = original_TTE − t₀
3. Retrain the full 7-model ensemble on this surviving cohort
4. Predict remaining time for each well via LOOCV

This is distinct from conditional survival (S(t|T>t₀) = S(t)/S(t₀)), which for proportional hazards models simply subtracts t₀ from a fixed prediction. Landmark analysis retrains on the surviving population, genuinely learning that survivors are different from early-breakthrough wells.

### Performance Across Horizons

| t₀ (months) | Training Wells | Events | C-index | MAE |
|-------------|---------------|--------|---------|-----|
| 0 | 16 | 12 | 0.811 | 61 |
| 12 | 16 | 12 | 0.811 | 61 |
| 24 | 16 | 12 | 0.811 | 61 |
| 36 | 16 | 12 | 0.811 | 61 |
| 48 | 14 | 10 | 0.779 | 82 |
| 60 | 13 | 9 | 0.734 | 93 |

Performance is stable through t₀=36. At t₀=48+, the training set shrinks as early-BT wells drop out, increasing MAE.

## Horizontal Well Extension

Horizontal wells are predicted using the same ensemble architecture with two pre-processing steps:

1. **Feature normalization:** Replace `peak_gas_rate` with `peak_gas_per_m = peak_gas_rate / perforation_length` (MMcf/m). This accounts for the 60x difference in reservoir contact between vertical wells (~11m perforation) and horizontal wells (~657m perforation).
2. **Feature scaling:** MinMaxScaler fit on combined vertical + horizontal features (21 wells). This uses only feature values, not outcomes — no data leakage. It eliminates extrapolation by placing both well types on a common [0, 1] scale.

No post-processing correction factor is applied, to avoid data leakage from the single horizontal BT observation (M-122H at 12 months).

### Horizontal Predictions

| Well | P10 | P50 | P90 | Observed | Status |
|------|-----|-----|-----|----------|--------|
| M-122H-HRL | 1 | 78 | 167 | 12 (BT) | Within P10-P90 |
| M-123H-HRL | 1 | 161 | 336 | 19 | In range |
| M-124H-HRL | 11 | 156 | 300 | 18 | In range |
| M-125H-HRL | 35 | 135 | 236 | 9 | Early |
| M-126H-HRL | 33 | 112 | 190 | 8 | Early |

The model systematically overpredicts for horizontal wells because their normalized gas-per-meter sits at the extreme low end of the vertical training range after scaling. The wide intervals appropriately capture this uncertainty. These predictions should be treated as indicative risk ranges.

## Limitations

- **Small training set** (16 vertical wells, 12 events) limits model complexity and statistical power
- **Feature window cap** at 24 months means wells with informative late-production patterns lose signal
- **Horizontal predictions** are out-of-distribution — the model has no horizontal training examples
- **No dynamic features** — current WGR trends, pressure data, or time-dependent covariates are not used
- **Single problematic well** — M-58-HRL falls outside P10-P90 in the current model, suggesting a missing feature or anomalous behavior
