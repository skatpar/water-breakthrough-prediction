# Observation Log — MARI Water Breakthrough Prediction POC

## 2026-04-24 — Phase 1: Load and Consolidate

### Source file structure
1. **BHP.csv** — 315 rows, 3 cols. Well names are bare numbers (11, 122H, E-2). Mixed date formats. 1 null BHP (M-126H-HRL).
2. **gas_gravity.csv** — 22 rows, canonical M-XX-HRL names. Tab characters before numeric values (stripped during load). Values rounded to 2 dp.
3. **Subsurface data.xlsx** — Single sheet, header on row 2 (rows 0-1 blank). 22 wells × 9 properties. Net pay column header contains embedded newline.
4. **STIXOR Sharing Data.xlsx [Rates]** — Header on row 1, heavy leading whitespace in column names. 6652 rows, 22 wells, 1978-03 to 2025-06. Dates are proper timestamps.
5. **STIXOR Sharing Data.xlsx [Pressures]** — Header on row 1. 5354 rows, 22 wells, 1990-06 to 2025-06.
6. **Pressure data.xlsx [Sheet2]** — Multi-column side-by-side layout (3 groups). 314 BHP records, 21 wells (M-126H absent). Also contains gas gravity (more precise than csv) and GWC note.
7. **Pressure data.xlsx [Well Completion Sketch]** — Embedded completion diagram images for M-122H and M-124H. Not parseable as tabular data.

### Well-name reconciliation
- Three naming conventions found: bare numbers (BHP.csv, Pressure xlsx), M-XX-HRL (subsurface, gas gravity, STIXOR Sharing Data)
- All 22 wells successfully mapped via `WELL_NAME_MAP` in config.py
- No ambiguous or conflicting names encountered

### Anomalies and flags
1. **Analog-copied static properties**: M-122H/M-124H/M-126H share identical (φ=0.22, k=34, Sw=0.46). M-123H/M-125H share (φ=0.24, k=22.5, Sw=0.30). These are not independent measurements.
2. **M-123H chlorides = 1215 ppm** — order of magnitude lower than all other wells (next lowest is M-125H at 2500, then ≥7000). Possible data entry error or different measurement context.
3. **Horizontal net_pay (475–802 m)** represents completed lateral length, not net pay thickness. Incomparable with vertical values (6–15 m). Must not mix in feature engineering.
4. **M-126H-HRL**: null BHP in BHP.csv, entirely absent from Pressure xlsx Sheet2. Least data of any well.
5. **Gas gravity discrepancy**: gas_gravity.csv is rounded; Pressure xlsx has 3+ dp. Max Δ = 0.005 (M-51, M-58, M-63, M-75). Using csv as primary source.
6. **Line pressure sparse**: 3339/5354 nulls in Pressures sheet. Almost no line pressure data before ~2005.
7. **Choke and prod_days**: not recorded before ~1993 (4560/6652 nulls).
8. **Water_bbl nulls**: 1969/6652. Ambiguous — could be "not measured" or "zero water". Critical for WGR.
9. **BHP.csv vs Pressure xlsx overlap**: 315 vs 314 records (the extra is M-126H with null BHP). Same underlying data.
10. **M-22-HRL and M-E-2-HRL** have positive skin (5.3 and 3.8) while all other verticals have negative skin — possible formation damage or partial completion effects.
11. **M-41-HRL** has anomalously low Sw = 0.13 (next lowest is M-65 at 0.19). Highest permeability among verticals at 37.1 mD.
12. **M-56-HRL** has highest chlorides = 23000 ppm (next is M-57 at 18000). Could indicate proximity to formation water.
13. **Rates sheet Notes column** (rows 2-5) contains metadata about how to interpret the data: gas/water are volumes not rates, ratios should use volumes with consistent units, WHFP and line pressure are separate and both affect well performance.

### Decisions confirmed by user
- **BHP source**: BHP.csv confirmed as primary.
- **Gas gravity source**: gas_gravity.csv (rounded) confirmed.
- **Water_bbl imputation**: Filled 1,884 nulls with zero where gas_mmcf > 0. Rationale: in a gas field, blank water entries in production logs mean "no water produced" rather than "not measured." Remaining 298 nulls are months where gas_mmcf is also null/zero (well not producing — WGR undefined). Logged as an imputation decision.
- **M-123H chlorides = 1215 ppm**: Kept as-is. Flagged as suspicious but not overwritten without MARI confirmation. Phase 4 will run chlorides analyses with and without M-123H.
- **RKB elevation**: 70 m applied uniformly. Added `rkb_elevation_m` column to well_static.parquet for future per-well corrections.

### Effective sample size (critical for modeling)
- Horizontal feature space has effectively **2 distinct rock samples, not 5**: Group A (M-122H/124H/126H: φ=0.22, k=34, Sw=0.46) and Group B (M-123H/125H: φ=0.24, k=22.5, Sw=0.30). Must not treat as 5 independent observations in any model. Phase 6 should collapse analog groups or explicitly note effective N.

### Phase 6 decision point
- **line_pressure_psig**: 62% null, concentrated in pre-2005 (oldest wells, strongest depletion signal). Likely not rescuable as a feature. Decision deferred to Phase 6 analysis.

### Questions for MARI
1. M-123H chlorides = 1215 ppm — is this a real measurement, data entry error, or sample-collection artifact?
2. Can you confirm rotary table elevation for each well? We've assumed 70 m uniformly based on M-11.
3. Do the horizontal wells have actual measured porosity/permeability, or are 122H/124H/126H and 123H/125H using analog values from nearby verticals?
4. **Critical**: Are the horizontal "Top Perf" and "Bottom Perf" values in MD or TVD? M-122H bottom_perf = 1550m MD implies the lateral extends far below the 754m RKB GWC. If this is MD (measured along the wellbore), what is the TVD at the toe? This determines whether the well was drilled through the gas-water contact.

## 2026-04-24 — Phase 2: Column Profiling

### Findings (15 items)
1. **No implausible negatives**: zero negative water_bbl, zero porosity > 0.4, zero perm < 0.1, zero Sw outside [0,1]
2. **gas_mmcf == 0**: 101 rows — legitimate shut-in months, not encoded nulls
3. **water_bbl == 0**: 1,969 rows after imputation — consistent with dry gas field
4. **Analog-copied statics confirmed** in strip plots: horizontal groups appear as identical points
5. **M-123H chlorides outlier** visible in strip plot — isolated from cluster
6. **Horizontal net_pay** clearly bimodal: 475-802m vs vertical 6-15m
7. **Permeability distribution**: Not cleanly log-normal (Shapiro p_raw=0.016, p_log=0.118). Bimodal: verticals 3-23 mD, horizontals 22-37 mD
8. **No WHFP zeros** — no encoded missingness in pressure data
9. **BHP very sparse**: 314/6865 non-null (4.6%). Annual surveys only.
10. **Positive skin**: M-22 (5.3), M-50 (1.4), M-75 (1.1), M-E-2 (3.8) — 4 wells, not 2 as initially noted
11. **gas_mmcf range**: 0.0–475.6 MMcf/mo. Peak from M-124H (horizontal, 2025-03)
12. **water_bbl range**: 0.0–6383 bbl/mo. Peak from M-41 (2024-03)
13. **WHFP range**: 51.3–1026 psig. Full range plausible.
14. **prod_days**: min 1.4 (M-123H partial month), max 31. Mean 29.9 — wells mostly produce full months.
15. **choke_64ths**: range 12–128 (1/64"). No zeros or implausible values.

## 2026-04-24 — Phase 3: Per-Well Stories

### Breakthrough detection results
- **14 wells** with sustained WGR > 5 bbl/MMcf for ≥ 3 consecutive months
- **13 events** after excluding M-51 (flowback from month 0)
- **M-122H**: breakthrough at month 12 (2023-12) — only horizontal event, validation anchor
- **4 dry verticals**: M-13, M-22, M-57, M-E-2 (censored, still producing without breakthrough)
- **Fastest verticals**: M-67 (month 45), M-82 (month 48), M-81 (month 57)
- **Slowest vertical**: M-11 (month 409 — 34 years of production before breakthrough)

### Key observations
- Vertical breakthrough times range from 45 to 409 months — enormous spread
- Newer wells (post-2000) break through faster: M-67 (45mo), M-81 (57mo), M-82 (48mo) vs M-11 (409mo), M-41 (329mo) — strong vintage effect
- M-13, M-22 have >500 months of production without breakthrough despite being among the oldest wells
- Horizontal wells have higher initial rates than verticals (300+ vs 100-200 MMcf/mo)

## 2026-04-24 — Phase 4: Cross-Variable Exploration

### Critical finding: vintage dominates
- **Spud date vs TTE: Spearman ρ = −0.860, p < 0.0001** — by far the strongest predictor
- All static properties have weaker correlations than spud date
- This means newer wells break through faster, consistent with cumulative field depletion

### Sw wrong sign confirmed
- Spearman ρ(Sw, TTE) = +0.133 (p=0.62) — positive but weak
- Direction is WRONG for physics (higher Sw should → faster BT)
- This is a vintage confounder: older wells drilled higher in gas column (low Sw) have had more time for depletion

### Perf depth vs Sw
- ρ = 0.001, p = 0.996 — **no transition zone signal** in the data
- Surprising: expected deeper perfs → higher Sw if in capillary transition zone
- Possible explanation: all verticals are well above GWC (25-53m), none in transition zone

### Chlorides sensitivity
- With M-123H: ρ(chlorides, TTE) = 0.299, p=0.26
- Without M-123H: ρ = 0.299, p=0.26 (Δρ = 0.000)
- M-123H outlier does NOT drive spurious correlations (it falls on the regression line)

### Correlation redundancy
- No pairs with |ρ| > 0.8 among static properties (verticals only)

## 2026-04-24 — Phase 5: Temporal Features

### Features computed
- Rate-based: mean_gas_yr1/2/3, cum_gas_yr1/2/5, peak_gas, time_to_peak, arps_di
- Pressure-based: initial_whfp, whfp_decline_rate, whfp_at_bt, drawdown_proxy, whfp_std
- Field-state: cum_field_gas_at_spud, cum_field_water_at_spud, active_wells_at_spud
- Volatility: gas_cov_2yr

### Null patterns
- initial_whfp: 12 nulls (wells with first production before WHFP data starts 1990)
- drawdown_proxy: 16 nulls (needs both initial WHFP and month-24 WHFP)
- arps_di: 4 nulls (horizontals with <24 months of data, insufficient for decline fitting)

## 2026-04-24 — Phase 6: Feature Selection

### Univariate Cox results (top 5 by C-index)
1. **active_wells_at_spud**: C=0.836, p=0.002, sign +, matches physics ✓
2. **cum_field_water_at_spud**: C=0.832, p=0.039, sign +, matches physics ✓
3. **cum_field_gas_at_spud**: C=0.828, p=0.003, sign +, matches physics ✓
4. **gas_cov_2yr**: C=0.746, p=0.015, sign −, unclear physics
5. **peak_gas**: C=0.697, p=0.121, sign −, WRONG physics sign (confounder)

### Redundancy analysis
- active_wells_at_spud, cum_field_water, cum_field_gas are all ρ > 0.94 with each other — all proxies for vintage/depletion
- Kept active_wells_at_spud (highest C-index), dropped the others
- chlorides_ppm and initial_whfp also redundant with active_wells_at_spud (ρ > 0.8)

### EPV constraint
- 13 events → max 1 feature unregularized, 4-6 features with L2 penalty

### Final shortlist
active_wells_at_spud, gas_cov_2yr, peak_gas, gas_gravity, time_to_peak_months, net_pay_m

### Physics gate
- peak_gas (sign −) and net_pay_m (sign +) have WRONG physics signs → flagged as confounder proxies
- active_wells_at_spud and time_to_peak_months match expectations

### line_pressure_psig: DROPPED
- 62% null, concentrated pre-2005. Not rescuable.

## 2026-04-24 — Phase 7: Benchmarks

### Critical finding: S-C and Papatzacos inapplicable
- **Sobocinski-Cornelius** predicts <1 month for ALL 16 verticals (actual: 45–409 months)
- Wells produce at 300–80,000× critical coning rate due to thin pay (10-15m) and high gas rates
- **Classical radial coning is NOT the breakthrough mechanism**
- Real mechanism: **field-wide GWC rise** from cumulative depletion over decades

### M-122H geometry discovery
- Bottom perf at 1550m MD, GWC at 754m RKB → lateral extends **below** GWC
- This is measured depth along the horizontal wellbore, not TVD
- But even in TVD, the horizontal section likely contacts or approaches the transition zone
- Explains 12-month breakthrough: not classical cresting, but near-immediate water contact
- **Papatzacos is inapplicable**: model assumes well above GWC, not through it

### Benchmark comparison (M-122H, actual = 12 months)
| Method | Predicted | Error | Comments |
|--------|-----------|-------|----------|
| KM median | 284 mo | 272 | Naive baseline |
| S-C | <1 mo | ~12 | Wrong mechanism |
| Papatzacos | N/A | N/A | Well below GWC |
| OLS (3 feats) | 148 mo | 136 | Trained on verticals |
| GWC position | <12 mo | <12 | Best physics match |

### Distance from bottom perf to GWC (verticals)
- Range: 18.2m (M-E-2) to 53.8m (M-56)
- Wells with d_gwc < 30m: M-65 (27.3m), M-67 (28.6m), M-75 (25.5m), M-82 (28.7m) — all broke through relatively fast
- **d_to_gwc is likely the most physically meaningful predictor** but was not in the subsurface data file as a derived feature. Should be added for Phase 6 modeling.

### KM field-wide median: 284 months
- This is the naive baseline. Any model must beat it.
- Heavily influenced by right-censored long-lived verticals (M-13 at 563 mo, M-22 at 525 mo still dry)

## 2026-04-24 — Phase 7 Addendum: d_to_gwc Feature

### d_to_gwc added to feature pipeline
- Computed as `GWC_RKB (754m) - bottom_perf_md` for verticals (where MD ≈ TVD)
- Set to NaN for horizontals (MD ≠ TVD along lateral)
- Added `compute_d_to_gwc()` to `src/features.py`, rebuilt `well_features.parquet`

### Univariate Cox results
- **C-index = 0.703**, Spearman ρ = 0.596, p = 0.015
- **Correct physics sign** (negative Cox coef → closer to GWC → faster breakthrough)
- This is the only static feature with both a strong C-index AND the correct physics sign

### d_to_gwc values (sorted)
| Well | d_to_gwc (m) | TTE (mo) | Event |
|------|-------------|----------|-------|
| M-E-2 | 18.2 | 148 | censored |
| M-75 | 25.5 | 127 | BT |
| M-65 | 27.3 | 259 | BT |
| M-67 | 28.6 | 45 | BT |
| M-82 | 28.7 | 48 | BT |
| M-61 | 31.0 | 284 | BT |
| M-63 | 31.8 | 275 | BT |
| M-22 | 33.6 | 525 | censored |
| M-81 | 38.0 | 57 | BT |
| M-41 | 45.8 | 329 | BT |
| M-50 | 45.8 | 379 | BT |
| M-58 | 46.2 | 88 | BT |
| M-11 | 48.5 | 409 | BT |
| M-13 | 52.2 | 563 | censored |
| M-57 | 52.7 | 384 | censored |
| M-56 | 53.8 | 299 | BT |

### Key observations
- Wells with d_gwc < 30m (M-75, M-67, M-82) are among the fastest breakthroughs
- M-E-2 has the smallest d_gwc (18.2m) but is censored — possible anomaly or formation damage (positive skin = 3.8)
- d_to_gwc does NOT fully explain TTE (M-81 at 38m broke through at 57 mo, while M-61 at 31m took 284 mo) — vintage/depletion still matters
- Added to `final_feature_set.json` as 7th feature
- Figure saved: `07_d_to_gwc_vs_tte.png`
