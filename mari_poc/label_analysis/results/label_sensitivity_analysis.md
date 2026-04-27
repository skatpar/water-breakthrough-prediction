# Label Sensitivity Analysis — WGR Threshold & Sustained Duration Sweep

**Date:** 2026-04-26
**Notebook:** `label_analysis/notebooks/label_sensitivity.ipynb`
**Scope:** 16 vertical wells, 25 label definitions (5 WGR thresholds × 5 sustained durations), 13 methods per definition
**Total experiments:** 325 (25 × 13)

---

## 1. Executive Summary

We tested 25 alternative breakthrough label definitions across all 7 individual models and 6 ensemble strategies. The key finding:

**WGR > 5 bbl/MMcf, sustained 2 months** is the optimal label definition — it improves C-index for most models compared to the baseline (WGR > 5, 3 months) while maintaining the same event count (12 events). However, the improvements are modest (+0.03 to +0.07 C-index), suggesting that the baseline definition is already near-optimal.

Lowering the WGR threshold (1-2 bbl/MMcf) dramatically reduces MAE but destroys ranking ability (C-index drops to 0.57-0.68). This is because low thresholds compress the TTE range — earlier breakthroughs are detected, making all wells look more similar.

---

## 2. Event Counts Across the Grid

All 25 combinations produce **12 events** — the same wells experience breakthrough regardless of threshold or duration. This is because the 12 event wells have strong, persistent water production that exceeds even WGR > 5 for 6+ months. The 4 censored wells (M-E-2, M-57, M-22, M-13) never sustain high water rates at any threshold.

**Implication:** Changing the label definition does NOT change which wells are labeled as events — it only changes **when** the event is detected (the TTE). Lower thresholds and shorter sustained periods detect breakthrough earlier, compressing the TTE range.

---

## 3. Best Label Definition by Method

### 3.1 Best by C-index

| Method | Best Label | C-index | MAE | vs Baseline C |
|---|---|---|---|---|
| Cox PH | WGR>5, 2mo | **0.819** | 125 | +0.031 |
| Weibull AFT | WGR>5, 2mo | **0.795** | 198 | +0.054 |
| XGB Cox | WGR>5, 2mo | **0.771** | 146 | +0.068 |
| XGB AFT | WGR>3, 6mo | **0.720** | 106 | +0.078 |
| sksurv GBSA | WGR>5, 3mo | **0.844** | 179 | +0.000 |
| sksurv CWGB | WGR>5, 2mo | **0.810** | 144 | +0.046 |
| Stacked | WGR>5, 2mo | **0.790** | 105 | +0.016 |
| Simple Mean | WGR>5, 2mo | **0.838** | 110 | +0.046 |
| Trimmed Mean | WGR>5, 2mo | **0.838** | 103 | +0.036 |
| Median | WGR>5, 2mo | **0.805** | 110 | +0.003 |
| Geometric Mean | WGR>5, 2mo | **0.829** | 80 | +0.046 |
| Inv-MAE Weighted | WGR>5, 2mo | **0.838** | 101 | +0.036 |
| Rank Fusion | WGR>5, 2mo | **0.810** | 101 | +0.027 |

**WGR > 5, 2 months** wins for 11 of 13 methods on C-index. This is a clear signal.

### 3.2 Best by MAE

| Method | Best Label | MAE | C-index | vs Baseline MAE |
|---|---|---|---|---|
| Cox PH | WGR>5, 2mo | **125** | 0.819 | -13 |
| Weibull AFT | WGR>1, 4mo | **183** | 0.594 | -22 |
| XGB Cox | WGR>1, 2mo | **62** | 0.580 | -91 |
| XGB AFT | WGR>1, 1mo | **51** | 0.571 | -67 |
| sksurv GBSA | WGR>1, 1mo | **93** | 0.679 | -87 |
| sksurv CWGB | WGR>4, 2mo | **126** | 0.766 | -30 |
| Stacked | WGR>5, 6mo | **97** | 0.783 | -7 |
| Simple Mean | WGR>5, 6mo | **104** | 0.802 | -7 |
| Trimmed Mean | WGR>5, 6mo | **99** | 0.811 | -4 |
| Geometric Mean | WGR>5, 6mo | **79** | 0.792 | -5 |
| Rank Fusion | WGR>5, 6mo | **80** | 0.792 | -7 |

**MAE-optimal labels split into two camps:**
- **WGR > 1, 1-2mo** for tree models (XGB, sksurv GBSA) — dramatic MAE drops but C-index collapses
- **WGR > 5, 6mo** for ensembles — modest MAE improvement, C-index maintained

---

## 4. The WGR > 5, 2-Month Finding

**Why does reducing sustained from 3 to 2 months help?**

Detecting breakthrough 1 month earlier for some wells shifts their TTE slightly, which can:
- Break ties between wells that had similar 3-month-sustained TTEs
- Give the model more spread in the target variable
- Reduce noise from the 3rd-month criterion (sometimes the 3rd consecutive month is a borderline call)

**Magnitude of improvement:**
- Cox PH: C=0.788 → 0.819 (+0.031), MAE=138 → 125 (-13 months)
- Ensembles (Simple Mean, Trimmed Mean, Inv-MAE Weighted): C ≈ 0.838 (+0.036-0.046)

These are meaningful improvements, especially the ensemble C-index reaching 0.838.

---

## 5. Why Low Thresholds Improve MAE but Destroy C-index

| Threshold | Range of TTEs | Spread | Effect |
|---|---|---|---|
| WGR > 1 | All wells BT in 20-380 months | Compressed | Models predict ~median, low MAE but can't rank |
| WGR > 5 | TTEs span 45-409 months | Wide | Models can distinguish early vs late |

At WGR > 1, even the "late" wells are detected earlier, compressing the TTE distribution. XGB AFT achieves MAE = 51 months at WGR > 1 because it predicts ~median and nobody is far from the median. But C-index = 0.571 (near random) — it can't tell which wells break through first.

**This is a classic bias-variance tradeoff applied to label definitions.**

---

## 6. Improvement Over Baseline — All Methods

| Method | Baseline C | Best C | Delta C | Baseline MAE | Best MAE | Delta MAE |
|---|---|---|---|---|---|---|
| Cox PH | 0.788 | 0.819 | **+0.031** | 138 | 125 | **-13** |
| Weibull AFT | 0.741 | 0.795 | **+0.054** | 205 | 183 | **-22** |
| XGB Cox | 0.703 | 0.771 | **+0.068** | 153 | 62 | -91* |
| XGB AFT | 0.642 | 0.720 | **+0.078** | 118 | 51 | -67* |
| sksurv GBSA | 0.844 | 0.844 | +0.000 | 179 | 93 | -87* |
| sksurv CWGB | 0.764 | 0.810 | **+0.046** | 156 | 126 | **-30** |
| Stacked | 0.774 | 0.790 | +0.016 | 104 | 97 | -7 |
| Simple Mean | 0.792 | 0.838 | **+0.046** | 111 | 104 | -7 |
| Trimmed Mean | 0.802 | 0.838 | **+0.036** | 103 | 99 | -4 |
| Median | 0.802 | 0.805 | +0.003 | 125 | 110 | -15 |
| Geometric Mean | 0.783 | 0.829 | **+0.046** | 84 | 79 | -5 |
| Inv-MAE Weighted | 0.802 | 0.838 | **+0.036** | 101 | 92 | -9 |
| Rank Fusion | 0.783 | 0.810 | **+0.027** | 87 | 80 | -7 |

*MAE reductions marked with * come from WGR>1 labels that sacrifice C-index — not recommended.

**Consistent improvements (same label = WGR>5, 2mo for C-index):**
- Cox PH improved on both C-index and MAE simultaneously
- All ensemble strategies improved C-index by +0.03 to +0.05
- Only sksurv GBSA and Stacked show negligible improvement

---

## 7. Cox PH Label Sensitivity

Out of 25 label definitions:
- **6/25** beat the baseline C-index (0.788)
- **2/25** beat the baseline MAE (138 months)
- **1/25** beats both simultaneously: **WGR > 5, 2 months**

The best 5 labels for Cox PH:

| Label | C-index | MAE | Median Error |
|---|---|---|---|
| WGR>5, 2mo | **0.819** | **125** | 43 |
| WGR>4, 2mo | 0.813 | 132 | 58 |
| WGR>5, 3mo (baseline) | 0.788 | 138 | 74 |
| WGR>5, 4mo | 0.788 | 138 | 74 |
| WGR>5, 1mo | 0.788 | 144 | 63 |

---

## 8. Recommendations for MARI

### Primary recommendation: Keep WGR > 5, 3 months as the standard

- It is the industry-standard threshold for gas wells
- The improvement from switching to 2 months (+0.031 C-index) is within LOOCV noise for n=12
- The 3-month sustained requirement provides more confidence that the water is a real breakthrough, not a transient event

### Secondary recommendation: Report WGR > 5, 2 months as a sensitivity finding

- Present as: "Relaxing the sustained criterion from 3 to 2 months marginally improves model ranking (C=0.819 vs 0.788)"
- This suggests that the 3rd consecutive month adds noise rather than signal for some wells
- Could be relevant for operational early warning systems where faster detection matters

### Do NOT recommend low WGR thresholds

- WGR > 1-2 bbl/MMcf produces impressive MAE numbers (51-62 months) but this is misleading
- Low thresholds compress the TTE range so all predictions cluster near the median
- C-index drops to 0.57-0.68 (near random) — the model can no longer rank wells
- These are noise-dominated labels for gas wells

### Additional finding: WGR > 5, 6 months for ensemble MAE

- Several ensemble strategies achieve their lowest MAE at WGR > 5, 6 months
- The longer sustained requirement is more conservative and may produce cleaner labels
- Worth noting as an alternative for conservative planning scenarios

---

## 9. Output Files

| File | Description |
|---|---|
| `label_analysis/notebooks/label_sensitivity.ipynb` | Full notebook with code and outputs (27 cells) |
| `label_analysis/results/label_sensitivity_results.xlsx` | Excel workbook (7 sheets) |
| `label_analysis/results/label_sensitivity_all.csv` | Flat CSV of all 325 experiments |
| `label_analysis/figures/01_event_count_grid.png` | Event count heatmap across grid |
| `label_analysis/figures/02_cindex_heatmaps.png` | C-index heatmaps for 6 key methods |
| `label_analysis/figures/03_mae_heatmaps.png` | MAE heatmaps for 6 key methods |
| `label_analysis/figures/04_all_labels_scatter.png` | C-index vs MAE scatter (all 325 points) |
| `label_analysis/figures/05_baseline_vs_best.png` | Bar chart comparing baseline vs best label |

### Excel sheets:
1. **All_Results** — Full 325-row results table
2. **Event_Counts** — Events per label definition
3. **Best_Per_Method** — Best label for each of 13 methods
4. **Improvement_vs_Baseline** — Delta from WGR>5, 3mo
5. **Cindex_Cox_PH / Rank_Fusion / sksurv_GBSA** — C-index pivot tables
6. **MAE_Cox_PH / Rank_Fusion / sksurv_GBSA** — MAE pivot tables

---

*Generated from 325 experiments (25 label definitions × 13 methods). All results are LOOCV.*
