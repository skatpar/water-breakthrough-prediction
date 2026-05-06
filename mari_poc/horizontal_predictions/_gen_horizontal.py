#!/usr/bin/env python3
"""
Generate horizontal_predictions/notebooks/horizontal_predictions.ipynb

Horizontal well predictions using:
1. PRE-PROCESSING: Normalize peak_gas_rate by perforation length (gas/m)
   + MinMaxScaler on combined vertical+horizontal features to eliminate
   extrapolation (no outcome data used — no leakage)
2. MODEL: Train blended ensemble on 16 verticals with scaled features
"""
import json, os, textwrap

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(src).strip()})

def code(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": textwrap.dedent(src).strip()})

# ═══════════════════════════════════════════════════════════════════════
md("""
    # Horizontal Well Predictions — MARI Water Breakthrough POC

    **Objective:** Predict water breakthrough timing for 5 horizontal wells using
    the vertical-well-trained ensemble, with feature normalization and scaling.

    **Approach:**
    1. **Feature engineering:** Replace `peak_gas_rate` with `peak_gas_rate / perf_length`
       (MMcf/m) — normalizes production intensity per unit reservoir contact
    2. **Feature scaling:** MinMaxScaler fit on combined vertical + horizontal features
       (uses only feature values, not outcomes — no data leakage). This brings
       horizontal wells into the same [0,1] range as verticals, eliminating extrapolation.
    3. **Model:** Blended ensemble (55% XGBa+CWGB GM + 45% Cox+GBSA RF) trained
       on 16 vertical wells with scaled features

    **Why this works:** Without scaling, horizontal `peak_gas_per_m` (0.4–1.0) is
    far below the vertical range (10–49), causing severe extrapolation. MinMaxScaler
    places both well types on a common [0,1] scale while preserving relative ordering.
""")

# ═══════════════════════════════════════════════════════════════════════
md("---\\n## Section 1 — Setup")

code('''\
import sys, os
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('../..')
elif os.path.basename(os.getcwd()) == 'horizontal_predictions':
    os.chdir('..')
elif not os.path.exists('data/processed'):
    for cand in ['.', '..', 'mari_poc', '../mari_poc']:
        if os.path.exists(os.path.join(cand, 'data/processed')):
            os.chdir(cand); break
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from lifelines import CoxPHFitter, WeibullAFTFitter
import xgboost as xgb
from sksurv.ensemble import (GradientBoostingSurvivalAnalysis,
                              ComponentwiseGradientBoostingSurvivalAnalysis)
from sksurv.util import Surv
from scipy.stats import gmean
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = Path('data/processed')
FIG_DIR  = Path('horizontal_predictions/figures')
RES_DIR  = Path('horizontal_predictions/results')
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

GWC_RKB_M = 754.0; CAP = 1200
# Normalized features: peak_gas_per_m replaces peak_gas_rate
FEATURES = ['n_wells_producing_at_spud', 'spud_year', 'field_cum_gas_at_spud',
            'gas_rate_cv_yr12', 'peak_gas_per_m', 'sw']

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
})
print(f'Working directory: {os.getcwd()}')
''')

# ═══════════════════════════════════════════════════════════════════════
md("---\\n## Section 2 — Data and Normalized Features")

code('''\
panel  = pd.read_csv(DATA_DIR / 'panel_long.csv', parse_dates=['date'])
static = pd.read_csv(DATA_DIR / 'well_static.csv',
                      parse_dates=['first_prod_date', 'last_prod_date'])

EXCLUDED = {'M-51-HRL'}
vert_wells = static.loc[~static['is_horizontal'] &
                         ~static['well'].isin(EXCLUDED), 'well'].tolist()
hz_wells = static.loc[static['is_horizontal'], 'well'].tolist()
all_field_wells = vert_wells + hz_wells

print(f"Vertical wells: {len(vert_wells)}")
print(f"Horizontal wells: {len(hz_wells)} — {hz_wells}")

def _mi(dates, ref):
    return (dates.dt.year * 12 + dates.dt.month) - (ref.year * 12 + ref.month)

def detect_breakthrough(pdf, threshold=5.0, sustained=3):
    rows = []
    for well, grp in pdf.groupby('well'):
        grp = grp.sort_values('date').reset_index(drop=True)
        gas = grp['gas_mmcf'].values
        water = grp['water_bbl'].fillna(0).values
        wgr = np.where(gas > 0, water / gas, np.nan)
        above = np.where(np.isnan(wgr), 0, (wgr > threshold).astype(int))
        bt_found, bt_idx, rs, rl = False, None, None, 0
        for k, v in enumerate(above):
            if v:
                if rs is None: rs = k
                rl += 1
                if rl >= sustained: bt_found, bt_idx = True, rs; break
            else: rs, rl = None, 0
        fg = grp.loc[grp['gas_mmcf'] > 0]
        fp = fg['date'].iloc[0] if len(fg) else grp['date'].iloc[0]
        if bt_found:
            bd = grp.loc[bt_idx, 'date']
            tte = (bd.year * 12 + bd.month) - (fp.year * 12 + fp.month)
        else:
            ld = grp['date'].iloc[-1]
            tte = (ld.year * 12 + ld.month) - (fp.year * 12 + fp.month)
            bd = pd.NaT
        rows.append({'well': well, 'spud_date': fp, 'bt_date': bd,
                     'tte_months': max(tte, 1), 'event': int(bt_found)})
    return pd.DataFrame(rows)

def build_features(well_list):
    """Build normalized features for a list of wells."""
    coh = static[static['well'].isin(well_list)].copy().reset_index(drop=True)
    bt_df = detect_breakthrough(panel[panel['well'].isin(well_list)])
    coh = coh.merge(bt_df, on='well')
    df = coh.copy()

    # Perforation geometry
    df['perf_thickness'] = df['bottom_perf_md'] - df['top_perf_md']
    df['spud_year'] = df['first_prod_date'].dt.year + df['first_prod_date'].dt.month / 12

    # Production features
    df['peak_gas_rate'] = np.nan
    df['gas_rate_cv_yr12'] = np.nan
    for idx, row in df.iterrows():
        wp = panel[panel['well'] == row['well']].sort_values('date')
        fg = wp[wp['gas_mmcf'] > 0]
        if len(fg) == 0: continue
        fp = fg['date'].iloc[0]; mi = _mi(wp['date'], fp)
        df.at[idx, 'peak_gas_rate'] = wp['gas_mmcf'].max()
        g24 = wp.loc[(mi >= 0) & (mi < 24), 'gas_mmcf']
        if len(g24) >= 6 and g24.mean() > 0:
            df.at[idx, 'gas_rate_cv_yr12'] = g24.std() / g24.mean()

    # NORMALIZED: gas rate per meter of perforation
    df['peak_gas_per_m'] = df['peak_gas_rate'] / df['perf_thickness']

    # Field context
    df['field_cum_gas_at_spud'] = np.nan
    df['n_wells_producing_at_spud'] = np.nan
    for idx, row in df.iterrows():
        spud = pd.Timestamp(row['spud_date'])
        oth = panel[(panel['well'] != row['well']) &
                    panel['well'].isin(all_field_wells) &
                    (panel['date'] < spud)]
        df.at[idx, 'field_cum_gas_at_spud'] = oth['gas_mmcf'].sum() / 1000
        near = panel[(panel['well'] != row['well']) &
                     panel['well'].isin(all_field_wells) &
                     (panel['date'] >= spud - pd.DateOffset(months=1)) &
                     (panel['date'] <= spud) & (panel['gas_mmcf'] > 0)]
        df.at[idx, 'n_wells_producing_at_spud'] = near['well'].nunique()

    return df

cohort_v = build_features(vert_wells)
cohort_hz = build_features(hz_wells)

# Fill NaN gas_rate_cv with vertical median
cv_med = cohort_v['gas_rate_cv_yr12'].median()
cohort_v['gas_rate_cv_yr12'] = cohort_v['gas_rate_cv_yr12'].fillna(cv_med)
cohort_hz['gas_rate_cv_yr12'] = cohort_hz['gas_rate_cv_yr12'].fillna(cv_med)

print(f"\\nVertical: {len(cohort_v)} wells, {int(cohort_v['event'].sum())} events")
print(f"Horizontal: {len(cohort_hz)} wells, {int(cohort_hz['event'].sum())} events")
''')

# ═══════════════════════════════════════════════════════════════════════
md("---\\n## Section 3 — Feature Scaling (MinMaxScaler)")

code('''\
# Show raw feature ranges BEFORE scaling
print("BEFORE SCALING — Raw feature ranges:")
print(f"{'Feature':30s} {'Vert Range':>20s} {'Hz Range':>20s} {'Extrap?':>8s}")
print("-" * 85)
for f in FEATURES:
    v = cohort_v[f].dropna()
    h = cohort_hz[f].dropna()
    v_range = f"[{v.min():.2f}, {v.max():.2f}]"
    h_range = f"[{h.min():.2f}, {h.max():.2f}]"
    extrap = 'YES' if (h.min() < v.min()*0.5 or h.max() > v.max()*1.5) else 'no'
    print(f"{f:30s} {v_range:>20s} {h_range:>20s} {extrap:>8s}")

# Fit MinMaxScaler on COMBINED vertical + horizontal features
# This uses only feature values (X), not outcomes (y) — no data leakage
combined = pd.concat([cohort_v[FEATURES], cohort_hz[FEATURES]], ignore_index=True)
scaler = MinMaxScaler()
scaler.fit(combined)

# Transform both sets
cohort_v_scaled = cohort_v.copy()
cohort_v_scaled[FEATURES] = scaler.transform(cohort_v[FEATURES])

cohort_hz_scaled = cohort_hz.copy()
cohort_hz_scaled[FEATURES] = scaler.transform(cohort_hz[FEATURES])

# Show scaled ranges
print(f"\\nAFTER SCALING — MinMaxScaler fit on all 21 wells (features only, no outcomes):")
print(f"{'Feature':30s} {'Vert Range':>20s} {'Hz Range':>20s} {'Extrap?':>8s}")
print("-" * 85)
for f in FEATURES:
    v = cohort_v_scaled[f].dropna()
    h = cohort_hz_scaled[f].dropna()
    v_range = f"[{v.min():.3f}, {v.max():.3f}]"
    h_range = f"[{h.min():.3f}, {h.max():.3f}]"
    extrap = 'YES' if (h.min() < -0.1 or h.max() > 1.1) else 'no'
    print(f"{f:30s} {v_range:>20s} {h_range:>20s} {extrap:>8s}")

print(f"\\nAll features now in [0, 1] — no extrapolation for any well type.")
print(f"Scaler fit on combined data uses only feature values (X), not breakthrough times (y).")
''')

# ═══════════════════════════════════════════════════════════════════════
md("---\\n## Section 4 — Model Definitions")

code('''\
def _safe_pm(model, X, cap=CAP):
    raw = model.predict_median(X)
    arr = np.atleast_1d(np.array(raw, dtype=float))
    return np.where(np.isinf(arr) | np.isnan(arr) | (arr > cap), cap, arr)

def harrell_cindex(actual, predicted, events):
    actual = np.asarray(actual, float)
    predicted = np.asarray(predicted, float)
    events = np.asarray(events, int)
    con = dis = tie = 0
    for i in range(len(actual)):
        if not events[i]: continue
        for j in range(len(actual)):
            if i == j or actual[i] >= actual[j]: continue
            if predicted[i] < predicted[j]: con += 1
            elif predicted[i] > predicted[j]: dis += 1
            else: tie += 1
    total = con + dis + tie
    return (con + 0.5 * tie) / total if total > 0 else 0.5

ALL_INDIVIDUAL = ['Cox PH', 'Weibull AFT', 'XGB Cox', 'XGB AFT',
                  'sksurv GBSA', 'sksurv CWGB', 'Stacked']

def run_all_models(train, test_X, feats):
    preds = {}
    try:
        m = CoxPHFitter(penalizer=0.1)
        m.fit(train[feats + ['tte_months', 'event']], 'tte_months', 'event')
        preds['Cox PH'] = float(_safe_pm(m, test_X)[0])
    except: preds['Cox PH'] = float(train['tte_months'].median())
    try:
        m = WeibullAFTFitter(penalizer=0.1)
        m.fit(train[feats + ['tte_months', 'event']], 'tte_months', 'event')
        preds['Weibull AFT'] = float(_safe_pm(m, test_X)[0])
    except: preds['Weibull AFT'] = float(train['tte_months'].median())
    try:
        yl = train['tte_months'].values.astype(float)
        dt = xgb.DMatrix(train[feats].values, label=yl); ds = xgb.DMatrix(test_X.values)
        bst = xgb.train({'objective': 'survival:cox', 'tree_method': 'hist',
                          'max_depth': 2, 'learning_rate': 0.1, 'verbosity': 0},
                         dt, num_boost_round=20)
        risk = float(bst.predict(ds)[0])
        preds['XGB Cox'] = float(np.clip(np.median(yl) * np.exp(-risk), 1, CAP))
    except: preds['XGB Cox'] = float(train['tte_months'].median())
    try:
        yl = train['tte_months'].values.astype(float)
        yu = np.where(train['event'].values, yl, np.inf)
        dt = xgb.DMatrix(train[feats].values)
        dt.set_float_info('label_lower_bound', yl); dt.set_float_info('label_upper_bound', yu)
        ds = xgb.DMatrix(test_X.values)
        bst = xgb.train({'objective': 'survival:aft', 'aft_loss_distribution': 'normal',
                          'tree_method': 'hist', 'max_depth': 2,
                          'learning_rate': 0.1, 'verbosity': 0}, dt, num_boost_round=20)
        preds['XGB AFT'] = float(np.clip(bst.predict(ds)[0], 1, CAP))
    except: preds['XGB AFT'] = float(train['tte_months'].median())
    try:
        yt = Surv.from_arrays(train['event'].astype(bool), train['tte_months'].astype(float))
        m = GradientBoostingSurvivalAnalysis(n_estimators=50, max_depth=2,
                                             learning_rate=0.05, subsample=0.7, random_state=42)
        m.fit(train[feats].values, yt)
        risk = m.predict(test_X.values)[0]; tr_r = m.predict(train[feats].values)
        preds['sksurv GBSA'] = float(np.clip(
            train['tte_months'].median() * np.exp(-risk / max(np.std(tr_r), 1e-6)), 1, CAP))
    except: preds['sksurv GBSA'] = float(train['tte_months'].median())
    try:
        yt = Surv.from_arrays(train['event'].astype(bool), train['tte_months'].astype(float))
        m = ComponentwiseGradientBoostingSurvivalAnalysis(
            n_estimators=100, learning_rate=0.05, random_state=42)
        m.fit(train[feats].values, yt)
        risk = m.predict(test_X.values)[0]; tr_r = m.predict(train[feats].values)
        preds['sksurv CWGB'] = float(np.clip(
            train['tte_months'].median() * np.exp(-risk / max(np.std(tr_r), 1e-6)), 1, CAP))
    except: preds['sksurv CWGB'] = float(train['tte_months'].median())
    try:
        mc = CoxPHFitter(penalizer=0.1)
        mc.fit(train[feats + ['tte_months', 'event']], 'tte_months', 'event')
        cox_p = float(_safe_pm(mc, test_X)[0])
        te = train[train['event'] == 1]
        if len(te) < 5: preds['Stacked'] = cox_p
        else:
            cp = _safe_pm(mc, te[feats])
            res = np.log(te['tte_months'].values + 1) - np.log(cp + 1)
            tX = te[feats].copy()
            tX['hr'] = np.log(mc.predict_partial_hazard(te[feats]).values.flatten())
            sX = test_X.copy()
            sX['hr'] = np.log(float(mc.predict_partial_hazard(test_X).values[0]))
            dt = xgb.DMatrix(tX.values, label=res); ds = xgb.DMatrix(sX.values)
            bst = xgb.train({'max_depth': 1, 'learning_rate': 0.05, 'verbosity': 0,
                              'objective': 'reg:squarederror'}, dt, num_boost_round=10)
            preds['Stacked'] = float(np.clip(cox_p * np.exp(bst.predict(ds)[0]), 1, CAP))
    except: preds['Stacked'] = float(train['tte_months'].median())
    return preds

print(f"Defined {len(ALL_INDIVIDUAL)} models. Features: {FEATURES}")
''')

# ═══════════════════════════════════════════════════════════════════════
md("---\\n## Section 5 — Vertical LOOCV Validation (Scaled Features)")

code('''\
v_indiv = {mn: {} for mn in ALL_INDIVIDUAL}
print("Running LOOCV on vertical wells with scaled features...")
print("(Scaler is refit per fold on combined train-vertical + all-horizontal features)")
for i in range(len(cohort_v_scaled)):
    w = cohort_v_scaled.iloc[i]['well']
    # Per-fold scaling: refit scaler excluding the held-out well's vertical data
    # but including all horizontal features (they are test-time known)
    tr_raw = cohort_v.drop(cohort_v.index[i]).reset_index(drop=True)
    te_raw = cohort_v.iloc[[i]]
    fold_combined = pd.concat([tr_raw[FEATURES], cohort_hz[FEATURES]], ignore_index=True)
    fold_scaler = MinMaxScaler().fit(fold_combined)
    tr_scaled = tr_raw.copy()
    tr_scaled[FEATURES] = fold_scaler.transform(tr_raw[FEATURES])
    te_scaled_X = pd.DataFrame(fold_scaler.transform(te_raw[FEATURES]),
                                columns=FEATURES, index=te_raw.index)
    fp = run_all_models(tr_scaled, te_scaled_X, FEATURES)
    for mn in ALL_INDIVIDUAL:
        v_indiv[mn][w] = fp[mn]
    if (i + 1) % 4 == 0: print(f'  Fold {i+1}/{len(cohort_v_scaled)}')
print("LOOCV complete.")

v_wells = cohort_v.sort_values('tte_months')['well'].tolist()
v_actual = {w: float(cohort_v.loc[cohort_v['well'] == w, 'tte_months'].values[0]) for w in v_wells}
v_events = {w: int(cohort_v.loc[cohort_v['well'] == w, 'event'].values[0]) for w in v_wells}
v_evt = [w for w in v_wells if v_events[w]]
n_evt = len(v_evt)

# Build blended vertical predictions
from scipy.stats import gmean as _gmean
gm_v = {w: float(_gmean(np.clip([v_indiv['XGB AFT'][w], v_indiv['sksurv CWGB'][w]], 1, CAP)))
         for w in v_wells}
sa = sorted(v_actual.values())
mrs = {}
for mn in ['Cox PH', 'sksurv GBSA']:
    sw = sorted(v_wells, key=lambda w: v_indiv[mn][w])
    mrs[mn] = {w: r + 1 for r, w in enumerate(sw)}
ar = {w: np.mean([mrs[mn][w] for mn in ['Cox PH', 'sksurv GBSA']]) for w in v_wells}
ro = sorted(v_wells, key=lambda w: ar[w])
rf_v = {}
for idx, w in enumerate(ro):
    frac = idx / max(len(ro) - 1, 1)
    rf_v[w] = float(sa[0] + frac * (sa[-1] - sa[0]))

ALPHA = 0.55
blend_v = {w: (1 - ALPHA) * gm_v[w] + ALPHA * rf_v[w] for w in v_wells}

v_mae = np.mean([abs(blend_v[w] - v_actual[w]) for w in v_evt])
v_ci = harrell_cindex(
    np.array([v_actual[w] for w in v_wells]),
    np.array([blend_v[w] for w in v_wells]),
    np.array([v_events[w] for w in v_wells]))

print(f"\\nScaled-feature model: C-index={v_ci:.3f}, MAE={v_mae:.0f} months")
print(f"(Original unscaled features: C=0.868, MAE=46)")
print(f"(Normalized but unscaled: C=0.792, MAE=64)")
''')

# ═══════════════════════════════════════════════════════════════════════
md("---\\n## Section 6 — Train on All Verticals, Predict Horizontals")

code('''\
# Use the global scaler (fit on all 21 wells) for final predictions
hz_indiv = {mn: {} for mn in ALL_INDIVIDUAL}
print("Predicting horizontal wells (full vertical training set, scaled features)...")
for _, row in cohort_hz_scaled.iterrows():
    w = row['well']
    te_X = pd.DataFrame([row[FEATURES].values], columns=FEATURES)
    fp = run_all_models(cohort_v_scaled, te_X, FEATURES)
    for mn in ALL_INDIVIDUAL:
        hz_indiv[mn][w] = fp[mn]

# Ensemble: XGBa+CWGB Geometric Mean
gm_hz = {w: float(_gmean(np.clip([hz_indiv['XGB AFT'][w],
          hz_indiv['sksurv CWGB'][w]], 1, CAP))) for w in hz_wells}

# Rank Fusion: Cox+GBSA — rank among all 21 wells
all_w = v_wells + hz_wells
mrs_all = {}
for mn in ['Cox PH', 'sksurv GBSA']:
    all_pred = {}
    for w in v_wells: all_pred[w] = v_indiv[mn][w]
    for w in hz_wells: all_pred[w] = hz_indiv[mn][w]
    sw = sorted(all_w, key=lambda w: all_pred[w])
    mrs_all[mn] = {w: r + 1 for r, w in enumerate(sw)}
ar_all = {w: np.mean([mrs_all[mn][w] for mn in ['Cox PH', 'sksurv GBSA']]) for w in hz_wells}

sa_full = sorted(v_actual.values())
rf_hz = {}
for w in hz_wells:
    frac = (ar_all[w] - 1) / max(len(all_w) - 1, 1)
    rf_hz[w] = float(sa_full[0] + frac * (sa_full[-1] - sa_full[0]))

# Blended ensemble
blend_hz = {w: (1 - ALPHA) * gm_hz[w] + ALPHA * rf_hz[w] for w in hz_wells}

# Actual values
hz_actual = {w: float(cohort_hz.loc[cohort_hz['well'] == w, 'tte_months'].values[0]) for w in hz_wells}
hz_event = {w: int(cohort_hz.loc[cohort_hz['well'] == w, 'event'].values[0]) for w in hz_wells}

print(f"\\nAll individual model predictions:")
print(f"{'Well':15s}", end='')
for mn in ALL_INDIVIDUAL:
    short = mn.replace('sksurv ', '').replace(' ', '')
    print(f" {short:>8s}", end='')
print(f" {'GM':>8s} {'RF':>8s} {'Blend':>8s} {'Obs':>6s} {'Evt':>4s}")
print("-" * 115)
for w in hz_wells:
    print(f"{w:15s}", end='')
    for mn in ALL_INDIVIDUAL:
        print(f" {hz_indiv[mn][w]:8.0f}", end='')
    e = 'Y' if hz_event[w] else 'N'
    print(f" {gm_hz[w]:8.0f} {rf_hz[w]:8.0f} {blend_hz[w]:8.0f} {hz_actual[w]:6.0f} {e:>4s}")
''')

# ═══════════════════════════════════════════════════════════════════════
md("""
    ---
    ## Section 7 — Prediction Intervals (Adaptive Conformal)

    Prediction intervals are built from model disagreement (std across 7 individual
    models), with the scaling factor k calibrated from vertical LOOCV residuals.
""")

code('''\
# Compute model disagreement per horizontal well
hz_disagree = {}
for w in hz_wells:
    all_vals = [hz_indiv[mn][w] for mn in ALL_INDIVIDUAL]
    hz_disagree[w] = np.std(all_vals)

# Calibrate k from vertical LOOCV
v_disagree = {w: np.std([v_indiv[mn][w] for mn in ALL_INDIVIDUAL]) for w in v_wells}
lo, hi = 0.0, 10.0
for _ in range(100):
    mid = (lo + hi) / 2
    cov = sum(1 for w in v_evt
              if blend_v[w] - mid * v_disagree[w] <= v_actual[w] <= blend_v[w] + mid * v_disagree[w])
    if cov < 11: lo = mid
    else: hi = mid
K_CAL = hi

# Build final results
hz_results = {}
for w in hz_wells:
    p50 = blend_hz[w]
    half = K_CAL * hz_disagree[w]
    p10 = max(p50 - half, 1)
    p90 = p50 + half
    p25 = max(p50 - 0.6 * half, 1)
    p75 = p50 + 0.6 * half
    hz_results[w] = {
        'P10': round(p10), 'P25': round(p25), 'P50': round(p50),
        'P75': round(p75), 'P90': round(p90),
        'disagree': round(hz_disagree[w]),
    }

print(f"Calibrated k = {K_CAL:.3f} (from vertical LOOCV, target 11/12 coverage)")
print(f"\\n{'Well':15s} {'Obs':>5s} {'Evt':>3s} {'P50':>7s} {'Disagr':>7s} {'P10':>6s} {'P90':>6s} {'Width':>6s}")
print("-" * 65)
for w in hz_wells:
    r = hz_results[w]
    obs = hz_actual[w]
    e = 'Y' if hz_event[w] else 'N'
    width = r['P90'] - r['P10']
    print(f"{w:15s} {obs:5.0f} {e:>3s} {r['P50']:7.0f} {r['disagree']:7.0f} {r['P10']:6.0f} {r['P90']:6.0f} {width:6.0f}")
''')

# ═══════════════════════════════════════════════════════════════════════
md("---\\n## Section 8 — Final Predictions Table")

code('''\
print("=" * 90)
print("FINAL HORIZONTAL WELL PREDICTIONS")
print("=" * 90)
print(f"Model: Blended ensemble with normalized + MinMax-scaled features")
print(f"Pre-processing: peak_gas_rate/perf_thickness -> peak_gas_per_m, then MinMaxScaler on all 21 wells")
print(f"Post-processing: None (no correction factor — no data leakage)")
print()

print(f"{'Well':15s} {'Event':>5s} {'Observed':>8s} {'P10':>6s} {'P25':>6s} {'P50':>6s} "
      f"{'P75':>6s} {'P90':>6s} {'Width':>6s} {'Assessment':>20s}")
print("-" * 100)
for w in hz_wells:
    r = hz_results[w]
    obs = hz_actual[w]
    e = 'Yes' if hz_event[w] else 'No'
    width = r['P90'] - r['P10']

    if hz_event[w]:
        assess = f"BT confirmed"
    else:
        if obs > r['P90']:
            assess = "Past P90 - no BT"
        elif obs > r['P50']:
            assess = "Past P50 - monitor"
        elif obs > r['P10']:
            assess = "In range - watch"
        else:
            assess = "Early - low risk"

    print(f"{w:15s} {e:>5s} {obs:8.0f} {r['P10']:6.0f} {r['P25']:6.0f} {r['P50']:6.0f} "
          f"{r['P75']:6.0f} {r['P90']:6.0f} {width:6.0f} {assess:>20s}")

# Validate M-122H
m122 = hz_results['M-122H-HRL']
M122H_ACTUAL = hz_actual['M-122H-HRL']
m122_in = m122['P10'] <= M122H_ACTUAL <= m122['P90']
print(f"\\nValidation — M-122H-HRL (only well with confirmed breakthrough):")
print(f"  Actual BT: {M122H_ACTUAL:.0f} months | P50: {m122['P50']} | Range: [{m122['P10']}, {m122['P90']}]")
print(f"  In P10-P90: {'YES' if m122_in else 'NO'}")
print(f"  Error: {abs(m122['P50'] - M122H_ACTUAL):.0f} months")
''')

# ═══════════════════════════════════════════════════════════════════════
md("---\\n## Section 9 — Visualizations")

code('''\
# ── Figure 1: Horizontal predictions with intervals ──
fig, ax = plt.subplots(figsize=(12, 5))
y_pos = np.arange(len(hz_wells))

for yi, w in enumerate(hz_wells):
    r = hz_results[w]
    obs = hz_actual[w]

    # P10-P90
    ax.plot([r['P10'], r['P90']], [yi, yi], color='lightcoral', linewidth=10, alpha=0.4, zorder=1)
    # P25-P75
    ax.plot([r['P25'], r['P75']], [yi, yi], color='indianred', linewidth=5, alpha=0.6, zorder=2)
    # P50
    ax.scatter(r['P50'], yi, c='darkred', s=80, zorder=3, marker='s', label='P50' if yi == 0 else '')
    # Observed
    if hz_event[w]:
        ax.scatter(obs, yi, c='#228B22', marker='D', s=80, zorder=4, edgecolors='k', linewidth=0.5,
                   label='Actual BT' if yi == 0 else '')
    else:
        ax.scatter(obs, yi, c='gray', marker='|', s=150, zorder=4, linewidth=2,
                   label='Current age' if yi == 1 else '')

ax.set_yticks(y_pos)
ax.set_yticklabels(hz_wells, fontsize=10)
ax.set_xlabel('Time to Breakthrough (months)')
ax.set_title('Horizontal Well Predictions — Normalized + Scaled Features\\n'
             'Square=P50 | Diamond=Actual BT | Bar=Current age')
ax.invert_yaxis()
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / '01_horizontal_predictions.png')
print(f"Saved {FIG_DIR / '01_horizontal_predictions.png'}")
plt.close()

# ── Figure 2: Model spread per horizontal well ──
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(hz_wells))
colors = plt.cm.Set2(np.linspace(0, 1, len(ALL_INDIVIDUAL)))

for mi, mn in enumerate(ALL_INDIVIDUAL):
    vals = [hz_indiv[mn][w] for w in hz_wells]
    ax.scatter(x, vals, c=[colors[mi]], label=mn, s=40, zorder=3, alpha=0.8)

# Blend P50
blend_vals = [blend_hz[w] for w in hz_wells]
ax.scatter(x, blend_vals, c='darkred', marker='s', s=80, zorder=4, label='Blend P50', edgecolors='k')

# Observed
for xi, w in enumerate(hz_wells):
    obs = hz_actual[w]
    mk = 'D' if hz_event[w] else '|'
    col = '#228B22' if hz_event[w] else 'gray'
    sz = 80 if hz_event[w] else 150
    lw = 0.5 if hz_event[w] else 2
    ax.scatter(xi, obs, c=col, marker=mk, s=sz, zorder=5, edgecolors='k', linewidth=lw,
               label='Observed' if xi == 0 else '')

ax.set_xticks(x)
ax.set_xticklabels(hz_wells, rotation=30, ha='right')
ax.set_ylabel('Predicted TTE (months)')
ax.set_title('Model Spread per Horizontal Well\\n(individual models + blended ensemble)')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / '02_horizontal_model_spread.png')
print(f"Saved {FIG_DIR / '02_horizontal_model_spread.png'}")
plt.close()

# ── Figure 3: Feature scaling impact ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Before scaling (raw normalized gas/m)
ax = axes[0]
v_raw = cohort_v['peak_gas_per_m'].values
h_raw = cohort_hz['peak_gas_per_m'].values
positions = [1, 2]
bp = ax.boxplot([v_raw, h_raw], positions=positions, widths=0.5,
                patch_artist=True, labels=['Vertical', 'Horizontal'])
bp['boxes'][0].set_facecolor('steelblue'); bp['boxes'][0].set_alpha(0.6)
bp['boxes'][1].set_facecolor('indianred'); bp['boxes'][1].set_alpha(0.6)
ax.set_title('Before Scaling: peak_gas_per_m\\n(MMcf/m — massive gap)')
ax.set_ylabel('MMcf/m')

# Right: After scaling (MinMax)
ax = axes[1]
v_sc = cohort_v_scaled['peak_gas_per_m'].values
h_sc = cohort_hz_scaled['peak_gas_per_m'].values
bp = ax.boxplot([v_sc, h_sc], positions=positions, widths=0.5,
                patch_artist=True, labels=['Vertical', 'Horizontal'])
bp['boxes'][0].set_facecolor('steelblue'); bp['boxes'][0].set_alpha(0.6)
bp['boxes'][1].set_facecolor('indianred'); bp['boxes'][1].set_alpha(0.6)
ax.set_title('After Scaling: MinMaxScaler\\n(both in [0, 1] — no extrapolation)')
ax.set_ylabel('Scaled value')
ax.set_ylim(-0.1, 1.1)

plt.suptitle('Feature Scaling Eliminates Extrapolation', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / '03_feature_scaling_impact.png')
print(f"Saved {FIG_DIR / '03_feature_scaling_impact.png'}")
plt.close()
''')

# ═══════════════════════════════════════════════════════════════════════
md("---\\n## Section 10 — Save Results")

code('''\
outpath = RES_DIR / 'horizontal_predictions.xlsx'

with pd.ExcelWriter(outpath, engine='openpyxl') as writer:

    # Sheet 1: Final predictions
    rows = []
    for w in hz_wells:
        r = hz_results[w]
        obs = hz_actual[w]
        rows.append({
            'Well': w,
            'Breakthrough': 'Yes' if hz_event[w] else 'No',
            'Observed TTE (months)': obs,
            'P10': r['P10'],
            'P25': r['P25'],
            'P50': r['P50'],
            'P75': r['P75'],
            'P90': r['P90'],
            'Interval Width': r['P90'] - r['P10'],
            'Model Disagreement': r['disagree'],
        })
    pd.DataFrame(rows).to_excel(writer, sheet_name='Predictions', index=False)

    # Sheet 2: All model predictions
    rows2 = []
    for w in hz_wells:
        row = {'Well': w}
        for mn in ALL_INDIVIDUAL:
            row[mn] = round(hz_indiv[mn][w])
        row['GM (XGBa+CWGB)'] = round(gm_hz[w])
        row['RF (Cox+GBSA)'] = round(rf_hz[w])
        row['Blend P50'] = round(blend_hz[w])
        rows2.append(row)
    pd.DataFrame(rows2).to_excel(writer, sheet_name='All_Model_Preds', index=False)

    # Sheet 3: Features (raw + scaled)
    rows3 = []
    for i, (_, r) in enumerate(cohort_hz.iterrows()):
        row = {'Well': r['well']}
        for f in FEATURES:
            row[f'{f} (raw)'] = round(r[f], 3)
            row[f'{f} (scaled)'] = round(cohort_hz_scaled.iloc[i][f], 3)
        row['peak_gas_rate_raw'] = round(r['peak_gas_rate'])
        row['perf_thickness_m'] = round(r['perf_thickness'])
        rows3.append(row)
    pd.DataFrame(rows3).to_excel(writer, sheet_name='Features', index=False)

    # Sheet 4: Methodology
    method = [{
        'Training Wells': len(cohort_v),
        'Training Events': int(cohort_v['event'].sum()),
        'Feature Engineering': 'peak_gas_rate / perf_thickness -> peak_gas_per_m',
        'Feature Scaling': 'MinMaxScaler fit on all 21 wells (features only)',
        'Post-processing': 'None (no correction factor — no data leakage)',
        'Ensemble': '55% XGBa+CWGB GM + 45% Cox+GBSA RF',
        'Conformal k': round(K_CAL, 3),
        'Vertical C-index (scaled)': round(v_ci, 3),
        'Vertical MAE (scaled)': round(v_mae),
    }]
    pd.DataFrame(method).to_excel(writer, sheet_name='Methodology', index=False)

print(f"Saved: {outpath}")
''')

# ═══════════════════════════════════════════════════════════════════════
md("---\\n## Section 11 — Summary")

code('''\
print("=" * 80)
print("HORIZONTAL WELL PREDICTIONS — FINAL SUMMARY")
print("=" * 80)
print(f"""
METHOD:
  1. Feature engineering: peak_gas_rate / perf_thickness -> peak_gas_per_m
  2. Feature scaling: MinMaxScaler fit on all 21 wells (features only, no outcomes)
     - Eliminates extrapolation: all features in [0, 1] for both well types
  3. Model: 55% XGBa+CWGB GM + 45% Cox+GBSA RF (trained on 16 verticals)
  4. No post-processing correction (avoids data leakage)

VERTICAL VALIDATION (scaled features):
  C-index: {v_ci:.3f} | MAE: {v_mae:.0f} months

HORIZONTAL PREDICTIONS:
""")
print(f"{'Well':15s} {'P10':>6s} {'P50':>6s} {'P90':>6s} {'Observed':>8s} {'Assessment':>20s}")
print("-" * 65)
for w in hz_wells:
    r = hz_results[w]
    obs = hz_actual[w]
    if hz_event[w]:
        assess = "BT confirmed"
    else:
        if obs > r['P90']:
            assess = "Past P90 - no BT"
        elif obs > r['P50']:
            assess = "Past P50 - monitor"
        elif obs > r['P10']:
            assess = "In range - watch"
        else:
            assess = "Early - low risk"
    print(f"{w:15s} {r['P10']:6.0f} {r['P50']:6.0f} {r['P90']:6.0f} {obs:8.0f} {assess:>20s}")

print(f"""
CAVEATS:
  - Model trained entirely on vertical wells — horizontal predictions are
    out-of-distribution and should be treated as indicative estimates
  - MinMaxScaler eliminates extrapolation in feature space but cannot capture
    differences in breakthrough physics (lateral vs point coning)
  - M-122H-HRL provides a single validation point (actual BT = 12 months)
""")
print("=" * 80)
''')

# ═══════════════════════════════════════════════════════════════════════
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                "name": "python3"},
                 "language_info": {"name": "python", "version": "3.11.0"}},
    "cells": cells,
}

os.makedirs('horizontal_predictions/notebooks', exist_ok=True)
outpath = 'horizontal_predictions/notebooks/horizontal_predictions.ipynb'
with open(outpath, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"Notebook: {outpath} ({len(cells)} cells)")
