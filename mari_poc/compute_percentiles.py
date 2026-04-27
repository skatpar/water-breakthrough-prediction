#!/usr/bin/env python3
"""
Compute P10, P25, P50, P75, P90 prediction intervals for all individual models
and ensemble strategies via bootstrap-perturbed LOOCV.

For each LOO fold:
  - Add Gaussian noise (5% of feature std) to training features (200 reps)
  - Refit all 7 models on each perturbed training set
  - Predict the held-out well → 200 predictions per model per well
  - Combine predictions into ensemble strategies → 200 ensemble predictions per well
  - Take P10, P25, P50, P75, P90 from each distribution

Output: results/percentile_predictions.xlsx with sheets:
  - Individual_P10 through Individual_P90
  - Ensemble_P10 through Ensemble_P90
  - Summary (one row per well, all models, all percentiles)
"""

import sys, os
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')
elif not os.path.exists('data/processed'):
    for cand in ['.', '..', 'mari_poc', '../mari_poc']:
        if os.path.exists(os.path.join(cand, 'data/processed')):
            os.chdir(cand); break
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from lifelines import CoxPHFitter, WeibullAFTFitter
from pathlib import Path
import xgboost as xgb
from sksurv.ensemble import (GradientBoostingSurvivalAnalysis,
                              ComponentwiseGradientBoostingSurvivalAnalysis)
from sksurv.util import Surv
from scipy.stats import gmean

# ── Config ──────────────────────────────────────────────────────────────
N_BOOTSTRAP = 200
NOISE_FRAC  = 0.05        # 5% Gaussian noise on features
CAP         = 1200
GWC_RKB_M   = 754.0
WGR_THRESHOLD = 5.0
BT_SUSTAINED  = 3
FEATURES = ['n_wells_producing_at_spud', 'spud_year', 'field_cum_gas_at_spud',
            'gas_rate_cv_yr12', 'peak_gas_rate']
MODEL_NAMES = ['Cox PH', 'Weibull AFT', 'XGB Cox', 'XGB AFT',
               'sksurv GBSA', 'sksurv CWGB', 'Stacked']
ENS_NAMES   = ['Simple Mean', 'Trimmed Mean', 'Median', 'Geometric Mean',
               'Inv-MAE Weighted', 'Rank Fusion']

DATA_DIR = Path('data/processed')
RES_DIR  = Path('results')
RES_DIR.mkdir(exist_ok=True)

np.random.seed(42)

# ── Data loading (same as ensemble notebook) ────────────────────────────
print("Loading data...")
panel  = pd.read_csv(DATA_DIR / 'panel_long.csv', parse_dates=['date'])
static = pd.read_csv(DATA_DIR / 'well_static.csv',
                      parse_dates=['first_prod_date', 'last_prod_date'])

EXCLUDED = {'M-51-HRL'}
vert_wells = static.loc[~static['is_horizontal'] &
                         ~static['well'].isin(EXCLUDED), 'well'].tolist()
panel_v  = panel[panel['well'].isin(vert_wells)].copy()
static_v = static[static['well'].isin(vert_wells)].copy().reset_index(drop=True)


def detect_breakthrough(pdf, threshold=WGR_THRESHOLD, sustained=BT_SUSTAINED):
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
                if rl >= sustained:
                    bt_found, bt_idx = True, rs; break
            else:
                rs, rl = None, 0
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


def _mi(dates, ref):
    return (dates.dt.year * 12 + dates.dt.month) - (ref.year * 12 + ref.month)


def build_features(coh, pan):
    df = coh.copy()
    df['log_perm'] = np.log10(df['permeability_md'])
    df['perf_midpoint_md'] = (df['top_perf_md'] + df['bottom_perf_md']) / 2
    df['perf_thickness_md'] = df['bottom_perf_md'] - df['top_perf_md']
    df['dist_to_gwc_m'] = GWC_RKB_M - df['perf_midpoint_md']
    df['spud_year'] = (df['first_prod_date'].dt.year +
                       df['first_prod_date'].dt.month / 12)
    for c in ['early_gas_rate', 'peak_gas_rate', 'cum_gas_year1',
              'cum_gas_year3', 'initial_whfp', 'whfp_decline_rate',
              'gas_rate_cv_yr12']:
        df[c] = np.nan
    all_w = df['well'].tolist()
    for idx, row in df.iterrows():
        wp = pan[pan['well'] == row['well']].sort_values('date')
        fg = wp.loc[wp['gas_mmcf'] > 0]
        if len(fg) == 0: continue
        fp = fg['date'].iloc[0]
        mi = _mi(wp['date'], fp)
        e6 = wp[(mi >= 0) & (mi < 6) & (wp['gas_mmcf'] > 0)]
        if len(e6):
            df.at[idx, 'early_gas_rate'] = (e6['gas_mmcf'] /
                                             e6['prod_days'].fillna(30)).mean()
        df.at[idx, 'peak_gas_rate'] = wp['gas_mmcf'].max()
        g24 = wp.loc[(mi >= 0) & (mi < 24), 'gas_mmcf']
        if len(g24) >= 6 and g24.mean() > 0:
            df.at[idx, 'gas_rate_cv_yr12'] = g24.std() / g24.mean()
    df['field_cum_gas_at_spud'] = np.nan
    df['n_wells_producing_at_spud'] = np.nan
    for idx, row in df.iterrows():
        spud = pd.Timestamp(row['spud_date'])
        oth = pan[(pan['well'] != row['well']) & pan['well'].isin(all_w) &
                  (pan['date'] < spud)]
        df.at[idx, 'field_cum_gas_at_spud'] = oth['gas_mmcf'].sum() / 1000
        near = pan[(pan['well'] != row['well']) & pan['well'].isin(all_w) &
                   (pan['date'] >= spud - pd.DateOffset(months=1)) &
                   (pan['date'] <= spud) & (pan['gas_mmcf'] > 0)]
        df.at[idx, 'n_wells_producing_at_spud'] = near['well'].nunique()
    return df


bt_df  = detect_breakthrough(panel_v)
cohort = static_v.merge(bt_df, on='well')
cohort = build_features(cohort, panel_v)
print(f"Cohort: {len(cohort)} wells, {int(cohort['event'].sum())} events")


# ── Model helpers ───────────────────────────────────────────────────────
def _safe_pm(model, X, cap=CAP):
    raw = model.predict_median(X)
    arr = np.atleast_1d(np.array(raw, dtype=float))
    return np.where(np.isinf(arr) | np.isnan(arr) | (arr > cap), cap, arr)


def run_all_models(train, test_X, feats):
    preds = {}

    # 1. Cox PH
    try:
        m = CoxPHFitter(penalizer=0.1)
        m.fit(train[feats + ['tte_months', 'event']], 'tte_months', 'event')
        preds['Cox PH'] = float(_safe_pm(m, test_X)[0])
    except Exception:
        preds['Cox PH'] = float(train['tte_months'].median())

    # 2. Weibull AFT
    try:
        m = WeibullAFTFitter(penalizer=0.05)
        m.fit(train[feats + ['tte_months', 'event']], 'tte_months', 'event')
        preds['Weibull AFT'] = float(_safe_pm(m, test_X)[0])
    except Exception:
        preds['Weibull AFT'] = float(train['tte_months'].median())

    # 3. XGBoost Cox
    try:
        y = (train['tte_months'].values.astype(float) *
             np.where(train['event'].values, 1, -1))
        dt = xgb.DMatrix(train[feats].values, label=y)
        ds = xgb.DMatrix(test_X.values)
        bst = xgb.train({'objective': 'survival:cox', 'tree_method': 'hist',
                          'max_depth': 2, 'learning_rate': 0.1,
                          'min_child_weight': 3, 'verbosity': 0},
                         dt, num_boost_round=20)
        hr = bst.predict(ds)[0]
        preds['XGB Cox'] = float(np.clip(
            train['tte_months'].median() / np.exp(hr), 1, CAP))
    except Exception:
        preds['XGB Cox'] = float(train['tte_months'].median())

    # 4. XGBoost AFT
    try:
        yl = train['tte_months'].values.astype(float)
        yu = np.where(train['event'].values, yl, np.inf)
        dt = xgb.DMatrix(train[feats].values)
        dt.set_float_info('label_lower_bound', yl)
        dt.set_float_info('label_upper_bound', yu)
        ds = xgb.DMatrix(test_X.values)
        bst = xgb.train({'objective': 'survival:aft',
                          'aft_loss_distribution': 'normal',
                          'tree_method': 'hist', 'max_depth': 2,
                          'learning_rate': 0.1, 'verbosity': 0},
                         dt, num_boost_round=20)
        preds['XGB AFT'] = float(np.clip(bst.predict(ds)[0], 1, CAP))
    except Exception:
        preds['XGB AFT'] = float(train['tte_months'].median())

    # 5. sksurv GBSA
    try:
        yt = Surv.from_arrays(train['event'].astype(bool),
                              train['tte_months'].astype(float))
        m = GradientBoostingSurvivalAnalysis(
            n_estimators=50, max_depth=2, learning_rate=0.05,
            subsample=0.7, random_state=42)
        m.fit(train[feats].values, yt)
        risk = m.predict(test_X.values)[0]
        tr_r = m.predict(train[feats].values)
        std_r = max(np.std(tr_r), 1e-6)
        preds['sksurv GBSA'] = float(np.clip(
            train['tte_months'].median() * np.exp(-risk / std_r), 1, CAP))
    except Exception:
        preds['sksurv GBSA'] = float(train['tte_months'].median())

    # 6. sksurv CWGB
    try:
        yt = Surv.from_arrays(train['event'].astype(bool),
                              train['tte_months'].astype(float))
        m = ComponentwiseGradientBoostingSurvivalAnalysis(
            n_estimators=100, learning_rate=0.05, random_state=42)
        m.fit(train[feats].values, yt)
        risk = m.predict(test_X.values)[0]
        tr_r = m.predict(train[feats].values)
        std_r = max(np.std(tr_r), 1e-6)
        preds['sksurv CWGB'] = float(np.clip(
            train['tte_months'].median() * np.exp(-risk / std_r), 1, CAP))
    except Exception:
        preds['sksurv CWGB'] = float(train['tte_months'].median())

    # 7. Stacked (Cox + XGB residual)
    try:
        mc = CoxPHFitter(penalizer=0.1)
        mc.fit(train[feats + ['tte_months', 'event']], 'tte_months', 'event')
        cox_p = float(_safe_pm(mc, test_X)[0])
        te = train[train['event'] == 1]
        if len(te) < 5:
            preds['Stacked'] = cox_p
        else:
            cp = _safe_pm(mc, te[feats])
            res = np.log(te['tte_months'].values + 1) - np.log(cp + 1)
            tX = te[feats].copy()
            tX['hr'] = np.log(mc.predict_partial_hazard(
                te[feats]).values.flatten())
            sX = test_X.copy()
            sX['hr'] = np.log(float(
                mc.predict_partial_hazard(test_X).values[0]))
            dt = xgb.DMatrix(tX.values, label=res)
            ds = xgb.DMatrix(sX.values)
            bst = xgb.train({'max_depth': 1, 'learning_rate': 0.05,
                              'verbosity': 0, 'objective': 'reg:squarederror'},
                             dt, num_boost_round=10)
            rp = bst.predict(ds)[0]
            preds['Stacked'] = float(np.clip(cox_p * np.exp(rp), 1, CAP))
    except Exception:
        preds['Stacked'] = float(train['tte_months'].median())

    return preds


def compute_ensembles(model_pred_vec, model_mae_weights=None):
    """Given dict {model_name: predicted_tte}, return dict of ensemble predictions."""
    vals = np.array([model_pred_vec[mn] for mn in MODEL_NAMES])
    ens = {}

    # Simple mean
    ens['Simple Mean'] = float(np.mean(vals))

    # Trimmed mean (drop min & max)
    sorted_v = np.sort(vals)
    ens['Trimmed Mean'] = float(np.mean(sorted_v[1:-1]))

    # Median
    ens['Median'] = float(np.median(vals))

    # Geometric mean
    ens['Geometric Mean'] = float(gmean(np.clip(vals, 1, CAP)))

    # Inv-MAE weighted (use provided weights or equal)
    if model_mae_weights is not None:
        w = np.array([model_mae_weights.get(mn, 1.0) for mn in MODEL_NAMES])
        w = w / w.sum()
        ens['Inv-MAE Weighted'] = float(np.dot(w, vals))
    else:
        ens['Inv-MAE Weighted'] = float(np.mean(vals))

    # Rank Fusion: convert to ranks using training TTE distribution
    # For a single prediction, we just store the raw value; rank fusion needs
    # all wells at once, so we return NaN here and compute it after the loop
    ens['Rank Fusion'] = np.nan

    return ens


# ── Compute feature-level noise stds ──────────────────────────────────
feat_stds = cohort[FEATURES].std().values  # shape (5,)

# ── Main bootstrap LOOCV loop ──────────────────────────────────────────
# Storage: boot_preds[model_name][well] = list of 200 predictions
boot_preds = {mn: {w: [] for w in cohort['well']} for mn in MODEL_NAMES}
boot_ens   = {en: {w: [] for w in cohort['well']} for en in ENS_NAMES if en != 'Rank Fusion'}
# Rank Fusion needs all wells per rep, handle separately
boot_rank_fusion = {w: [] for w in cohort['well']}

# First compute model-level MAE weights from unperturbed LOOCV
# (already done in ensemble notebook — re-derive here)
print("\nPhase 1: Unperturbed LOOCV for MAE weights...")
unperturbed = {mn: {} for mn in MODEL_NAMES}
for i in range(len(cohort)):
    w = cohort.iloc[i]['well']
    train = cohort.drop(cohort.index[i]).reset_index(drop=True)
    test_X = cohort.iloc[[i]][FEATURES]
    fp = run_all_models(train, test_X, FEATURES)
    for mn in MODEL_NAMES:
        unperturbed[mn][w] = fp[mn]

# Compute MAE per model (events only)
model_maes = {}
evt_wells = cohort.loc[cohort['event'] == 1, 'well'].tolist()
for mn in MODEL_NAMES:
    errs = [abs(unperturbed[mn][w] -
                float(cohort.loc[cohort['well'] == w, 'tte_months'].values[0]))
            for w in evt_wells]
    model_maes[mn] = np.mean(errs)
inv_mae_w = {mn: 1.0 / max(model_maes[mn], 1) for mn in MODEL_NAMES}
total_inv = sum(inv_mae_w.values())
inv_mae_w = {mn: v / total_inv for mn, v in inv_mae_w.items()}

print("MAE weights:", {mn: f"{v:.3f}" for mn, v in inv_mae_w.items()})

# Phase 2: Bootstrap LOOCV
print(f"\nPhase 2: Bootstrap LOOCV ({N_BOOTSTRAP} reps x {len(cohort)} folds)...")
for i in range(len(cohort)):
    w = cohort.iloc[i]['well']
    train_base = cohort.drop(cohort.index[i]).reset_index(drop=True)
    test_X = cohort.iloc[[i]][FEATURES]

    for b in range(N_BOOTSTRAP):
        # Perturb training features with Gaussian noise
        train = train_base.copy()
        noise = np.random.randn(len(train), len(FEATURES)) * feat_stds * NOISE_FRAC
        train[FEATURES] = train_base[FEATURES].values + noise

        # Run all models
        fp = run_all_models(train, test_X, FEATURES)

        # Store individual predictions
        for mn in MODEL_NAMES:
            boot_preds[mn][w].append(fp[mn])

        # Compute ensemble predictions (except rank fusion)
        ens = compute_ensembles(fp, inv_mae_w)
        for en in ENS_NAMES:
            if en != 'Rank Fusion':
                boot_ens[en][w].append(ens[en])

    if (i + 1) % 4 == 0:
        print(f"  Well {i+1}/{len(cohort)} done ({w})")

# Phase 3: Rank Fusion — needs all wells per bootstrap rep
print("\nPhase 3: Computing Rank Fusion percentiles...")
wells_list = cohort['well'].tolist()
actual_tte = {w: float(cohort.loc[cohort['well'] == w, 'tte_months'].values[0])
              for w in wells_list}
sorted_actuals = sorted(actual_tte.values())

for b in range(N_BOOTSTRAP):
    # Collect predictions for this bootstrap rep across all wells
    rep_preds = {}
    for mn in MODEL_NAMES:
        rep_preds[mn] = {w: boot_preds[mn][w][b] for w in wells_list}

    # For each model, rank wells (1=shortest, N=longest)
    model_ranks = {}
    for mn in MODEL_NAMES:
        sorted_wells = sorted(wells_list, key=lambda w: rep_preds[mn][w])
        model_ranks[mn] = {w: r + 1 for r, w in enumerate(sorted_wells)}

    # Average ranks across models
    avg_ranks = {}
    for w in wells_list:
        avg_ranks[w] = np.mean([model_ranks[mn][w] for mn in MODEL_NAMES])

    # Map avg rank back to TTE using linear interpolation on sorted actuals
    rank_order = sorted(wells_list, key=lambda w: avg_ranks[w])
    n = len(rank_order)
    for idx, w in enumerate(rank_order):
        # Map position to TTE: position 0 → min actual, position n-1 → max actual
        frac = idx / max(n - 1, 1)
        mapped_tte = sorted_actuals[0] + frac * (sorted_actuals[-1] - sorted_actuals[0])
        boot_rank_fusion[w].append(mapped_tte)

# ── Compute percentiles ─────────────────────────────────────────────────
PERCENTILES = [10, 25, 50, 75, 90]

print("\nComputing percentiles...")

# Individual models
indiv_pctiles = {}  # {(model, well, pctile): value}
for mn in MODEL_NAMES:
    for w in wells_list:
        arr = np.array(boot_preds[mn][w])
        for p in PERCENTILES:
            indiv_pctiles[(mn, w, p)] = np.percentile(arr, p)

# Ensemble strategies
ens_pctiles = {}
for en in ENS_NAMES:
    for w in wells_list:
        if en == 'Rank Fusion':
            arr = np.array(boot_rank_fusion[w])
        else:
            arr = np.array(boot_ens[en][w])
        for p in PERCENTILES:
            ens_pctiles[(en, w, p)] = np.percentile(arr, p)

# ── Build Excel workbook ────────────────────────────────────────────────
print("\nWriting Excel workbook...")

# Sort wells by actual TTE
wells_sorted = sorted(wells_list,
                       key=lambda w: float(cohort.loc[cohort['well'] == w,
                                                       'tte_months'].values[0]))
events_map = {w: int(cohort.loc[cohort['well'] == w, 'event'].values[0])
              for w in wells_sorted}

outpath = RES_DIR / 'percentile_predictions.xlsx'
with pd.ExcelWriter(outpath, engine='openpyxl') as writer:

    # ── Sheet 1: Individual models, one sheet per percentile ──
    for p in PERCENTILES:
        rows = []
        for w in wells_sorted:
            row = {
                'Well': w,
                'Event': 'Y' if events_map[w] else 'N',
                'Actual (months)': actual_tte[w],
            }
            for mn in MODEL_NAMES:
                row[f'{mn}'] = round(indiv_pctiles[(mn, w, p)])
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name=f'Individual_P{p}', index=False)

    # ── Sheet 2: Ensemble strategies, one sheet per percentile ──
    for p in PERCENTILES:
        rows = []
        for w in wells_sorted:
            row = {
                'Well': w,
                'Event': 'Y' if events_map[w] else 'N',
                'Actual (months)': actual_tte[w],
            }
            for en in ENS_NAMES:
                row[f'{en}'] = round(ens_pctiles[(en, w, p)])
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name=f'Ensemble_P{p}', index=False)

    # ── Sheet 3: Summary — all models, all percentiles, one row per well ──
    all_methods = MODEL_NAMES + ENS_NAMES
    rows = []
    for w in wells_sorted:
        row = {
            'Well': w,
            'Event': 'Y' if events_map[w] else 'N',
            'Actual (months)': actual_tte[w],
        }
        for method in all_methods:
            pctile_dict = indiv_pctiles if method in MODEL_NAMES else ens_pctiles
            for p in PERCENTILES:
                row[f'{method}_P{p}'] = round(pctile_dict[(method, w, p)])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_excel(writer, sheet_name='Summary', index=False)

    # ── Sheet 4: Coverage analysis ──
    # How often does actual fall within P10-P90, P25-P75?
    rows = []
    for method in all_methods:
        pctile_dict = indiv_pctiles if method in MODEL_NAMES else ens_pctiles
        in_10_90 = 0
        in_25_75 = 0
        n_events = 0
        for w in wells_sorted:
            if not events_map[w]:
                continue
            n_events += 1
            a = actual_tte[w]
            p10 = pctile_dict[(method, w, 10)]
            p90 = pctile_dict[(method, w, 90)]
            p25 = pctile_dict[(method, w, 25)]
            p75 = pctile_dict[(method, w, 75)]
            if p10 <= a <= p90:
                in_10_90 += 1
            if p25 <= a <= p75:
                in_25_75 += 1
        rows.append({
            'Method': method,
            'P10-P90 coverage (events)': f'{in_10_90}/{n_events}',
            'P10-P90 coverage %': round(100 * in_10_90 / n_events, 1),
            'P25-P75 coverage (events)': f'{in_25_75}/{n_events}',
            'P25-P75 coverage %': round(100 * in_25_75 / n_events, 1),
        })
    df = pd.DataFrame(rows)
    df.to_excel(writer, sheet_name='Coverage', index=False)

    # ── Sheet 5: Interval widths ──
    rows = []
    for method in all_methods:
        pctile_dict = indiv_pctiles if method in MODEL_NAMES else ens_pctiles
        widths_80 = []
        widths_50 = []
        for w in wells_sorted:
            if not events_map[w]:
                continue
            widths_80.append(pctile_dict[(method, w, 90)] -
                             pctile_dict[(method, w, 10)])
            widths_50.append(pctile_dict[(method, w, 75)] -
                             pctile_dict[(method, w, 25)])
        rows.append({
            'Method': method,
            'Mean P10-P90 width (months)': round(np.mean(widths_80)),
            'Median P10-P90 width (months)': round(np.median(widths_80)),
            'Mean P25-P75 width (months)': round(np.mean(widths_50)),
            'Median P25-P75 width (months)': round(np.median(widths_50)),
        })
    df = pd.DataFrame(rows)
    df.to_excel(writer, sheet_name='Interval_Widths', index=False)

print(f"\nSaved: {outpath}")
print(f"Sheets: Individual_P10..P90, Ensemble_P10..P90, Summary, Coverage, Interval_Widths")
print("Done!")
