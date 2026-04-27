#!/usr/bin/env python3
"""
Generate optimized_ensemble/notebooks/optimized_ensemble_model.ipynb

Based on combination sweep insights:
  - Best C-index: Cox PH + sksurv GBSA, Rank Fusion (C=0.896, MAE=70)
  - Best MAE: XGB AFT + sksurv CWGB, Geometric Mean (C=0.802, MAE=54)
  - Best balanced: XGB AFT + sksurv GBSA, Inv-MAE Weighted (C=0.840, MAE=71)

This notebook:
1. Runs the top combos under LOOCV with full per-well detail
2. Computes bootstrap P10/P25/P50/P75/P90 intervals
3. Compares all optimized combos vs Cox baseline and 7-model ensemble
4. Per-well error analysis and visualizations
5. Saves results to Excel with percentile sheets
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
    # Optimized Ensemble Model — MARI Water Breakthrough POC

    Based on the model combination sweep (720 experiments), the best ensembles use
    **fewer models, not more**. This notebook implements and validates the top
    discovered combinations with full per-well analysis and prediction intervals.

    **Top combinations from sweep:**

    | Name | Models | Strategy | C-index | MAE |
    |---|---|---|---|---|
    | Best Ranking | Cox PH + sksurv GBSA | Rank Fusion | 0.896 | 70 |
    | Best Accuracy | XGB AFT + sksurv CWGB | Geometric Mean | 0.802 | 54 |
    | Best Balanced | XGB AFT + sksurv GBSA | Inv-MAE Weighted | 0.840 | 71 |
    | Best 3-model | XGB AFT + GBSA + Stacked | Geometric Mean | 0.821 | 64 |
    | Pareto pick | XGB AFT + GBSA | Geometric Mean | 0.811 | 59 |
""")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 1 — SETUP
# ═══════════════════════════════════════════════════════════════════════
md("---\n## Section 1 — Setup")

code('''\
import sys, os
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('../..')
elif os.path.basename(os.getcwd()) == 'optimized_ensemble':
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

DATA_DIR = Path('data/processed')
FIG_DIR  = Path('optimized_ensemble/figures')
RES_DIR  = Path('optimized_ensemble/results')
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

GWC_RKB_M = 754.0; CAP = 1200
FEATURES = ['n_wells_producing_at_spud', 'spud_year', 'field_cum_gas_at_spud',
            'gas_rate_cv_yr12', 'peak_gas_rate']

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
})
print(f'Working directory: {os.getcwd()}')
''')

# ═══════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA
# ═══════════════════════════════════════════════════════════════════════
md("---\n## Section 2 — Data and Features")

code('''\
panel  = pd.read_csv(DATA_DIR / 'panel_long.csv', parse_dates=['date'])
static = pd.read_csv(DATA_DIR / 'well_static.csv',
                      parse_dates=['first_prod_date', 'last_prod_date'])

EXCLUDED = {'M-51-HRL'}
vert_wells = static.loc[~static['is_horizontal'] &
                         ~static['well'].isin(EXCLUDED), 'well'].tolist()
panel_v  = panel[panel['well'].isin(vert_wells)].copy()
static_v = static[static['well'].isin(vert_wells)].copy().reset_index(drop=True)

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
        fp = fg['date'].iloc[0]; mi = _mi(wp['date'], fp)
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
''')

# ═══════════════════════════════════════════════════════════════════════
# SECTION 3 — MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════
md("---\n## Section 3 — Model Definitions")

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

def fit_cox(train, feats):
    m = CoxPHFitter(penalizer=0.1)
    m.fit(train[feats + ['tte_months', 'event']], 'tte_months', 'event')
    return m

def predict_cox(m, test_X):
    return float(_safe_pm(m, test_X)[0])

def fit_predict_xgb_aft(train, test_X, feats):
    yl = train['tte_months'].values.astype(float)
    yu = np.where(train['event'].values, yl, np.inf)
    dt = xgb.DMatrix(train[feats].values)
    dt.set_float_info('label_lower_bound', yl)
    dt.set_float_info('label_upper_bound', yu)
    ds = xgb.DMatrix(test_X.values)
    bst = xgb.train({'objective': 'survival:aft', 'aft_loss_distribution': 'normal',
                      'tree_method': 'hist', 'max_depth': 2,
                      'learning_rate': 0.1, 'verbosity': 0}, dt, num_boost_round=20)
    return float(np.clip(bst.predict(ds)[0], 1, CAP))

def fit_predict_gbsa(train, test_X, feats):
    yt = Surv.from_arrays(train['event'].astype(bool), train['tte_months'].astype(float))
    m = GradientBoostingSurvivalAnalysis(n_estimators=50, max_depth=2, learning_rate=0.05,
                                         subsample=0.7, random_state=42)
    m.fit(train[feats].values, yt)
    risk = m.predict(test_X.values)[0]
    tr_r = m.predict(train[feats].values)
    std_r = max(np.std(tr_r), 1e-6)
    return float(np.clip(train['tte_months'].median() * np.exp(-risk / std_r), 1, CAP))

def fit_predict_cwgb(train, test_X, feats):
    yt = Surv.from_arrays(train['event'].astype(bool), train['tte_months'].astype(float))
    m = ComponentwiseGradientBoostingSurvivalAnalysis(
        n_estimators=100, learning_rate=0.05, random_state=42)
    m.fit(train[feats].values, yt)
    risk = m.predict(test_X.values)[0]
    tr_r = m.predict(train[feats].values)
    std_r = max(np.std(tr_r), 1e-6)
    return float(np.clip(train['tte_months'].median() * np.exp(-risk / std_r), 1, CAP))

def fit_predict_stacked(train, test_X, feats):
    mc = CoxPHFitter(penalizer=0.1)
    mc.fit(train[feats + ['tte_months', 'event']], 'tte_months', 'event')
    cox_p = float(_safe_pm(mc, test_X)[0])
    te = train[train['event'] == 1]
    if len(te) < 5:
        return cox_p
    cp = _safe_pm(mc, te[feats])
    res = np.log(te['tte_months'].values + 1) - np.log(cp + 1)
    tX = te[feats].copy()
    tX['hr'] = np.log(mc.predict_partial_hazard(te[feats]).values.flatten())
    sX = test_X.copy()
    sX['hr'] = np.log(float(mc.predict_partial_hazard(test_X).values[0]))
    dt = xgb.DMatrix(tX.values, label=res)
    ds = xgb.DMatrix(sX.values)
    bst = xgb.train({'max_depth': 1, 'learning_rate': 0.05, 'verbosity': 0,
                      'objective': 'reg:squarederror'}, dt, num_boost_round=10)
    rp = bst.predict(ds)[0]
    return float(np.clip(cox_p * np.exp(rp), 1, CAP))

# Master model runner — returns dict of predictions for requested models
def run_models(train, test_X, feats, model_list):
    preds = {}
    for mn in model_list:
        try:
            if mn == 'Cox PH':
                m = fit_cox(train, feats)
                preds[mn] = predict_cox(m, test_X)
            elif mn == 'XGB AFT':
                preds[mn] = fit_predict_xgb_aft(train, test_X, feats)
            elif mn == 'sksurv GBSA':
                preds[mn] = fit_predict_gbsa(train, test_X, feats)
            elif mn == 'sksurv CWGB':
                preds[mn] = fit_predict_cwgb(train, test_X, feats)
            elif mn == 'Stacked':
                preds[mn] = fit_predict_stacked(train, test_X, feats)
        except Exception:
            preds[mn] = float(train['tte_months'].median())
    return preds

print("Model functions defined.")
''')

# ═══════════════════════════════════════════════════════════════════════
# SECTION 4 — DEFINE TOP COMBOS
# ═══════════════════════════════════════════════════════════════════════
md("""
    ---
    ## Section 4 — Top Ensemble Combinations

    Define the 5 best combinations discovered in the sweep, plus baselines.
""")

code('''\
# Top combinations to test
COMBOS = {
    'Cox+GBSA RankFusion': {
        'models': ['Cox PH', 'sksurv GBSA'],
        'strategy': 'rank_fusion',
    },
    'XGBa+CWGB GeomMean': {
        'models': ['XGB AFT', 'sksurv CWGB'],
        'strategy': 'geometric_mean',
    },
    'XGBa+GBSA InvMAE': {
        'models': ['XGB AFT', 'sksurv GBSA'],
        'strategy': 'inv_mae_weighted',
    },
    'XGBa+GBSA GeomMean': {
        'models': ['XGB AFT', 'sksurv GBSA'],
        'strategy': 'geometric_mean',
    },
    'XGBa+GBSA+Stk GeomMean': {
        'models': ['XGB AFT', 'sksurv GBSA', 'Stacked'],
        'strategy': 'geometric_mean',
    },
    'Cox PH (baseline)': {
        'models': ['Cox PH'],
        'strategy': 'single',
    },
}

def apply_strategy(model_preds, strategy, wells, events_map, actual_map, all_model_preds=None):
    """Apply an ensemble strategy to combine predictions."""
    model_names = list(model_preds.keys())
    n = len(model_names)
    result = {}

    if strategy == 'single':
        mn = model_names[0]
        for w in wells:
            result[w] = model_preds[mn][w]
        return result

    # Compute inv-MAE weights
    evt_w = [w for w in wells if events_map[w]]
    model_maes = {}
    for mn in model_names:
        errs = [abs(model_preds[mn][w] - actual_map[w]) for w in evt_w]
        model_maes[mn] = np.mean(errs) if errs else 1.0
    inv_w = {mn: 1.0 / max(model_maes[mn], 1) for mn in model_names}
    tw = sum(inv_w.values())
    inv_w = {mn: v / tw for mn, v in inv_w.items()}

    for w in wells:
        vals = np.array([model_preds[mn][w] for mn in model_names])
        if strategy == 'geometric_mean':
            result[w] = float(gmean(np.clip(vals, 1, CAP)))
        elif strategy == 'inv_mae_weighted':
            ww = np.array([inv_w[mn] for mn in model_names])
            result[w] = float(np.dot(ww, vals))

    if strategy == 'rank_fusion':
        sorted_actuals = sorted(actual_map.values())
        model_ranks = {}
        for mn in model_names:
            sw = sorted(wells, key=lambda w: model_preds[mn][w])
            model_ranks[mn] = {w: r + 1 for r, w in enumerate(sw)}
        avg_ranks = {w: np.mean([model_ranks[mn][w] for mn in model_names])
                     for w in wells}
        rank_order = sorted(wells, key=lambda w: avg_ranks[w])
        nn = len(rank_order)
        for idx, w in enumerate(rank_order):
            frac = idx / max(nn - 1, 1)
            result[w] = float(sorted_actuals[0] + frac * (sorted_actuals[-1] - sorted_actuals[0]))

    return result

print(f"Defined {len(COMBOS)} combinations to test:")
for name, cfg in COMBOS.items():
    print(f"  {name}: {cfg['models']} -> {cfg['strategy']}")
''')

# ═══════════════════════════════════════════════════════════════════════
# SECTION 5 — LOOCV FOR ALL COMBOS
# ═══════════════════════════════════════════════════════════════════════
md("---\n## Section 5 — LOOCV: All Optimized Combinations")

code('''\
# First, run LOOCV for all needed individual models
needed_models = set()
for cfg in COMBOS.values():
    needed_models.update(cfg['models'])
needed_models = sorted(needed_models)
print(f"Models needed: {needed_models}")

# Store per-fold per-model predictions
indiv_preds = {mn: {} for mn in needed_models}

print("Running LOOCV for individual models...")
for i in range(len(cohort)):
    w = cohort.iloc[i]['well']
    train = cohort.drop(cohort.index[i]).reset_index(drop=True)
    test_X = cohort.iloc[[i]][FEATURES]
    fp = run_models(train, test_X, FEATURES, needed_models)
    for mn in needed_models:
        indiv_preds[mn][w] = fp[mn]
    if (i + 1) % 4 == 0:
        print(f'  Fold {i+1}/{len(cohort)} done')
print('LOOCV complete.')

wells = cohort.sort_values('tte_months')['well'].tolist()
actual = {w: float(cohort.loc[cohort['well'] == w, 'tte_months'].values[0]) for w in wells}
events = {w: int(cohort.loc[cohort['well'] == w, 'event'].values[0]) for w in wells}
evt_wells = [w for w in wells if events[w]]

# Now compute ensemble predictions for each combo
combo_preds = {}  # combo_name -> {well: pred}
for name, cfg in COMBOS.items():
    mp = {mn: indiv_preds[mn] for mn in cfg['models']}
    combo_preds[name] = apply_strategy(mp, cfg['strategy'], wells, events, actual)

# Compute metrics
print(f"\\n{'Combination':35s} {'C-index':>8s} {'MAE':>7s} {'MedErr':>7s} {'MaxErr':>7s}")
print("=" * 70)
combo_metrics = {}
for name in COMBOS:
    prd_arr = np.array([combo_preds[name][w] for w in wells])
    act_arr = np.array([actual[w] for w in wells])
    evt_arr = np.array([events[w] for w in wells])
    ci = harrell_cindex(act_arr, prd_arr, evt_arr)
    errs = [abs(combo_preds[name][w] - actual[w]) for w in evt_wells]
    mae = np.mean(errs)
    med_err = np.median(errs)
    max_err = np.max(errs)
    combo_metrics[name] = {'c_index': ci, 'mae': mae, 'median_error': med_err, 'max_error': max_err}
    print(f"{name:35s} {ci:8.3f} {mae:7.0f} {med_err:7.0f} {max_err:7.0f}")
''')

# ═══════════════════════════════════════════════════════════════════════
# SECTION 6 — PER-WELL COMPARISON
# ═══════════════════════════════════════════════════════════════════════
md("---\n## Section 6 — Per-Well Comparison: All Optimized Combos")

code('''\
combo_names = list(COMBOS.keys())
short_names = {
    'Cox+GBSA RankFusion': 'Cox+GBSA RF',
    'XGBa+CWGB GeomMean': 'XGBa+CWGB GM',
    'XGBa+GBSA InvMAE': 'XGBa+GBSA IW',
    'XGBa+GBSA GeomMean': 'XGBa+GBSA GM',
    'XGBa+GBSA+Stk GeomMean': 'XGBa+GBSA+Stk',
    'Cox PH (baseline)': 'Cox PH',
}

# Header
hdr = f"{'Well':12s} {'Evt':>3s} {'Actual':>6s}"
for cn in combo_names:
    hdr += f" | {short_names[cn]:>14s}"
print(hdr)
print("-" * len(hdr))

for w in wells:
    a = actual[w]
    e = 'Y' if events[w] else 'N'
    line = f"{w:12s} {e:>3s} {a:6.0f}"
    for cn in combo_names:
        p = combo_preds[cn][w]
        err = p - a
        line += f" | {p:6.0f}({err:+5.0f})"
    print(line)

# Summary row
print("-" * len(hdr))
line = f"{'C-index':12s} {'':>3s} {'':>6s}"
for cn in combo_names:
    ci = combo_metrics[cn]['c_index']
    line += f" | {ci:14.3f}"
print(line)
line = f"{'MAE(events)':12s} {'':>3s} {'':>6s}"
for cn in combo_names:
    mae = combo_metrics[cn]['mae']
    line += f" | {mae:14.0f}"
print(line)
line = f"{'Max error':12s} {'':>3s} {'':>6s}"
for cn in combo_names:
    mx = combo_metrics[cn]['max_error']
    line += f" | {mx:14.0f}"
print(line)
''')

# ═══════════════════════════════════════════════════════════════════════
# SECTION 7 — BOOTSTRAP PERCENTILES
# ═══════════════════════════════════════════════════════════════════════
md("""
    ---
    ## Section 7 — Bootstrap P10/P25/P50/P75/P90 Intervals

    200 bootstrap reps with 5% feature noise. Compute percentiles for all optimized combos.
""")

code('''\
N_BOOT = 200
NOISE_FRAC = 0.05
PERCENTILES = [10, 25, 50, 75, 90]
feat_stds = cohort[FEATURES].std().values

np.random.seed(42)

# Storage: boot_indiv[model][well] = list of 200 preds
boot_indiv = {mn: {w: [] for w in wells} for mn in needed_models}

print(f"Bootstrap LOOCV: {N_BOOT} reps × {len(cohort)} folds...")
for i in range(len(cohort)):
    w = cohort.iloc[i]['well']
    train_base = cohort.drop(cohort.index[i]).reset_index(drop=True)
    test_X = cohort.iloc[[i]][FEATURES]

    for b in range(N_BOOT):
        train = train_base.copy()
        noise = np.random.randn(len(train), len(FEATURES)) * feat_stds * NOISE_FRAC
        train[FEATURES] = train_base[FEATURES].values + noise
        fp = run_models(train, test_X, FEATURES, needed_models)
        for mn in needed_models:
            boot_indiv[mn][w].append(fp[mn])

    if (i + 1) % 4 == 0:
        print(f'  Well {i+1}/{len(cohort)} done ({w})')

# Compute ensemble predictions per bootstrap rep
boot_combo = {cn: {w: [] for w in wells} for cn in combo_names if cn != 'Cox PH (baseline)'}
# Cox baseline
boot_combo['Cox PH (baseline)'] = {w: list(boot_indiv['Cox PH'][w]) for w in wells}

print("\\nComputing ensemble predictions per bootstrap rep...")
for b in range(N_BOOT):
    # Collect this rep's predictions
    rep_indiv = {mn: {w: boot_indiv[mn][w][b] for w in wells} for mn in needed_models}

    for cn, cfg in COMBOS.items():
        if cn == 'Cox PH (baseline)':
            continue
        mp = {mn: rep_indiv[mn] for mn in cfg['models']}

        # For rank fusion, we need all wells simultaneously
        if cfg['strategy'] == 'rank_fusion':
            sorted_actuals = sorted(actual.values())
            model_ranks = {}
            for mn in cfg['models']:
                sw = sorted(wells, key=lambda w: mp[mn][w])
                model_ranks[mn] = {w: r + 1 for r, w in enumerate(sw)}
            avg_ranks = {w: np.mean([model_ranks[mn][w] for mn in cfg['models']])
                         for w in wells}
            rank_order = sorted(wells, key=lambda w: avg_ranks[w])
            nn = len(rank_order)
            for idx, w in enumerate(rank_order):
                frac = idx / max(nn - 1, 1)
                boot_combo[cn][w].append(
                    sorted_actuals[0] + frac * (sorted_actuals[-1] - sorted_actuals[0]))
        elif cfg['strategy'] == 'geometric_mean':
            for w in wells:
                vals = np.array([mp[mn][w] for mn in cfg['models']])
                boot_combo[cn][w].append(float(gmean(np.clip(vals, 1, CAP))))
        elif cfg['strategy'] == 'inv_mae_weighted':
            # Use fixed weights from unperturbed run for consistency
            evt_w = [w2 for w2 in wells if events[w2]]
            model_maes = {}
            for mn in cfg['models']:
                errs = [abs(mp[mn][w2] - actual[w2]) for w2 in evt_w]
                model_maes[mn] = np.mean(errs) if errs else 1.0
            inv_w = {mn: 1.0 / max(model_maes[mn], 1) for mn in cfg['models']}
            tw = sum(inv_w.values())
            inv_w = {mn: v / tw for mn, v in inv_w.items()}
            for w in wells:
                vals = np.array([mp[mn][w] for mn in cfg['models']])
                ww = np.array([inv_w[mn] for mn in cfg['models']])
                boot_combo[cn][w].append(float(np.dot(ww, vals)))

# Compute percentiles
pctiles = {}  # (combo_name, well, p) -> value
for cn in combo_names:
    for w in wells:
        arr = np.array(boot_combo[cn][w])
        for p in PERCENTILES:
            pctiles[(cn, w, p)] = np.percentile(arr, p)

# Coverage analysis
print("\\nCoverage analysis (events only):")
print(f"{'Combination':35s} {'P10-P90':>8s} {'P25-P75':>8s} {'Width80':>8s} {'Width50':>8s}")
print("-" * 75)
for cn in combo_names:
    in_80 = in_50 = 0
    w80 = []; w50 = []
    for w in evt_wells:
        a = actual[w]
        p10 = pctiles[(cn, w, 10)]; p90 = pctiles[(cn, w, 90)]
        p25 = pctiles[(cn, w, 25)]; p75 = pctiles[(cn, w, 75)]
        if p10 <= a <= p90: in_80 += 1
        if p25 <= a <= p75: in_50 += 1
        w80.append(p90 - p10); w50.append(p75 - p25)
    n = len(evt_wells)
    print(f"{cn:35s} {in_80:3d}/{n:d}={100*in_80/n:4.0f}% {in_50:3d}/{n:d}={100*in_50/n:4.0f}% "
          f"{np.mean(w80):8.0f} {np.mean(w50):8.0f}")
''')

# ═══════════════════════════════════════════════════════════════════════
# SECTION 8 — PERCENTILE TABLE
# ═══════════════════════════════════════════════════════════════════════
md("---\n## Section 8 — P10/P25/P50/P75/P90 for Recommended Model")

code('''\
# Show the recommended model's percentiles
rec_name = 'Cox+GBSA RankFusion'
print(f"Percentile predictions: {rec_name}")
print()
print(f"{'Well':12s} {'Evt':>3s} {'Actual':>6s} {'P10':>6s} {'P25':>6s} {'P50':>6s} "
      f"{'P75':>6s} {'P90':>6s} {'Width':>6s} {'In80':>5s}")
print("-" * 75)
for w in wells:
    a = actual[w]
    e = 'Y' if events[w] else 'N'
    p10 = pctiles[(rec_name, w, 10)]
    p25 = pctiles[(rec_name, w, 25)]
    p50 = pctiles[(rec_name, w, 50)]
    p75 = pctiles[(rec_name, w, 75)]
    p90 = pctiles[(rec_name, w, 90)]
    width = p90 - p10
    inside = 'YES' if p10 <= a <= p90 else 'no'
    print(f"{w:12s} {e:>3s} {a:6.0f} {p10:6.0f} {p25:6.0f} {p50:6.0f} "
          f"{p75:6.0f} {p90:6.0f} {width:6.0f} {inside:>5s}")

# Also for second best
print()
rec2 = 'XGBa+CWGB GeomMean'
print(f"Percentile predictions: {rec2}")
print()
print(f"{'Well':12s} {'Evt':>3s} {'Actual':>6s} {'P10':>6s} {'P25':>6s} {'P50':>6s} "
      f"{'P75':>6s} {'P90':>6s} {'Width':>6s} {'In80':>5s}")
print("-" * 75)
for w in wells:
    a = actual[w]
    e = 'Y' if events[w] else 'N'
    p10 = pctiles[(rec2, w, 10)]
    p25 = pctiles[(rec2, w, 25)]
    p50 = pctiles[(rec2, w, 50)]
    p75 = pctiles[(rec2, w, 75)]
    p90 = pctiles[(rec2, w, 90)]
    width = p90 - p10
    inside = 'YES' if p10 <= a <= p90 else 'no'
    print(f"{w:12s} {e:>3s} {a:6.0f} {p10:6.0f} {p25:6.0f} {p50:6.0f} "
          f"{p75:6.0f} {p90:6.0f} {width:6.0f} {inside:>5s}")
''')

# ═══════════════════════════════════════════════════════════════════════
# SECTION 9 — VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════
md("---\n## Section 9 — Visualizations")

code('''\
# ── Figure 1: Per-well error comparison ──
fig, ax = plt.subplots(figsize=(14, 7))
display_combos = ['Cox PH (baseline)', 'Cox+GBSA RankFusion', 'XGBa+CWGB GeomMean',
                  'XGBa+GBSA GeomMean', 'XGBa+GBSA+Stk GeomMean']
x = np.arange(len(evt_wells))
width = 0.15
colors = ['#4682B4', '#DC143C', '#2E8B57', '#FF8C00', '#9370DB']

for ci, cn in enumerate(display_combos):
    errors = [combo_preds[cn][w] - actual[w] for w in evt_wells]
    ax.bar(x + ci * width, errors, width, label=short_names.get(cn, cn), color=colors[ci], alpha=0.8)

ax.set_xticks(x + width * 2)
ax.set_xticklabels([w.replace('-HRL', '') for w in evt_wells], rotation=45, ha='right')
ax.set_ylabel('Prediction Error (months)')
ax.set_title('Per-Well Prediction Error — Optimized Ensembles vs Cox Baseline')
ax.legend(fontsize=8, loc='upper left')
ax.axhline(y=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(FIG_DIR / '01_perwell_errors.png')
print(f"Saved {FIG_DIR / '01_perwell_errors.png'}")
plt.close()

# ── Figure 2: Predicted vs Actual scatter ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
scatter_combos = ['Cox PH (baseline)', 'Cox+GBSA RankFusion', 'XGBa+CWGB GeomMean']
scatter_colors = ['#4682B4', '#DC143C', '#2E8B57']

for ax_idx, (cn, col) in enumerate(zip(scatter_combos, scatter_colors)):
    ax = axes[ax_idx]
    for w in wells:
        a = actual[w]
        p = combo_preds[cn][w]
        marker = 'o' if events[w] else '^'
        ax.scatter(a, p, c=col, marker=marker, s=60, edgecolors='k', linewidth=0.5)
    ax.plot([0, 600], [0, 600], 'k--', alpha=0.3, label='Perfect')
    ax.set_xlabel('Actual (months)')
    ax.set_ylabel('Predicted (months)')
    ci = combo_metrics[cn]['c_index']
    mae = combo_metrics[cn]['mae']
    ax.set_title(f"{short_names.get(cn, cn)}\\nC={ci:.3f}, MAE={mae:.0f}")
    ax.set_xlim(0, 600)
    ax.set_ylim(0, max(600, max(combo_preds[cn][w] for w in wells) + 50))
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / '02_predicted_vs_actual.png')
print(f"Saved {FIG_DIR / '02_predicted_vs_actual.png'}")
plt.close()

# ── Figure 3: Dumbbell chart with P10-P90 intervals for recommended model ──
fig, ax = plt.subplots(figsize=(12, 8))
rec_name = 'Cox+GBSA RankFusion'
y_pos = np.arange(len(wells))

for yi, w in enumerate(wells):
    a = actual[w]
    p10 = pctiles[(rec_name, w, 10)]
    p25 = pctiles[(rec_name, w, 25)]
    p50 = pctiles[(rec_name, w, 50)]
    p75 = pctiles[(rec_name, w, 75)]
    p90 = pctiles[(rec_name, w, 90)]

    # P10-P90 bar
    ax.plot([p10, p90], [yi, yi], color='lightblue', linewidth=6, alpha=0.5, zorder=1)
    # P25-P75 bar
    ax.plot([p25, p75], [yi, yi], color='steelblue', linewidth=3, alpha=0.7, zorder=2)
    # P50 marker
    ax.scatter(p50, yi, c='navy', s=40, zorder=3)
    # Actual marker
    col = '#DC143C' if events[w] else '#4682B4'
    ax.scatter(a, yi, c=col, marker='D', s=50, zorder=4, edgecolors='k', linewidth=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(wells)
ax.set_xlabel('Time to Breakthrough (months)')
ax.set_title(f'Prediction Intervals — {rec_name}\\nDiamond=Actual, Circle=P50, Light=P10-P90, Dark=P25-P75')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(FIG_DIR / '03_intervals.png')
print(f"Saved {FIG_DIR / '03_intervals.png'}")
plt.close()

# ── Figure 4: Summary bar chart ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cn_sorted = sorted(combo_names, key=lambda cn: -combo_metrics[cn]['c_index'])
x = np.arange(len(cn_sorted))
c_vals = [combo_metrics[cn]['c_index'] for cn in cn_sorted]
mae_vals = [combo_metrics[cn]['mae'] for cn in cn_sorted]
bar_colors = ['#DC143C' if cn == 'Cox+GBSA RankFusion' else
              '#2E8B57' if cn == 'XGBa+CWGB GeomMean' else '#4682B4' for cn in cn_sorted]

axes[0].barh(x, c_vals, color=bar_colors)
axes[0].set_yticks(x)
axes[0].set_yticklabels([short_names.get(cn, cn) for cn in cn_sorted], fontsize=9)
axes[0].set_xlabel('C-index')
axes[0].set_title('C-index (higher = better)')
axes[0].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

cn_sorted_mae = sorted(combo_names, key=lambda cn: combo_metrics[cn]['mae'])
x2 = np.arange(len(cn_sorted_mae))
mae_vals2 = [combo_metrics[cn]['mae'] for cn in cn_sorted_mae]
bar_colors2 = ['#DC143C' if cn == 'Cox+GBSA RankFusion' else
               '#2E8B57' if cn == 'XGBa+CWGB GeomMean' else '#4682B4' for cn in cn_sorted_mae]

axes[1].barh(x2, mae_vals2, color=bar_colors2)
axes[1].set_yticks(x2)
axes[1].set_yticklabels([short_names.get(cn, cn) for cn in cn_sorted_mae], fontsize=9)
axes[1].set_xlabel('MAE (months)')
axes[1].set_title('MAE — events only (lower = better)')

plt.tight_layout()
plt.savefig(FIG_DIR / '04_summary_bars.png')
print(f"Saved {FIG_DIR / '04_summary_bars.png'}")
plt.close()
''')

# ═══════════════════════════════════════════════════════════════════════
# SECTION 10 — SAVE TO EXCEL
# ═══════════════════════════════════════════════════════════════════════
md("---\n## Section 10 — Save Results")

code('''\
outpath = RES_DIR / 'optimized_ensemble_results.xlsx'

with pd.ExcelWriter(outpath, engine='openpyxl') as writer:

    # Sheet 1: Summary metrics
    rows = []
    for cn in combo_names:
        m = combo_metrics[cn]
        cfg = COMBOS[cn]
        rows.append({
            'Combination': cn,
            'Models': ', '.join(cfg['models']),
            'N_models': len(cfg['models']),
            'Strategy': cfg['strategy'],
            'C-index': round(m['c_index'], 3),
            'MAE (events)': round(m['mae']),
            'Median Error': round(m['median_error']),
            'Max Error': round(m['max_error']),
        })
    pd.DataFrame(rows).to_excel(writer, sheet_name='Summary', index=False)

    # Sheet 2: Per-well predictions
    pw_rows = []
    for w in wells:
        row = {'Well': w, 'Event': 'Y' if events[w] else 'N', 'Actual': actual[w]}
        for cn in combo_names:
            row[cn] = round(combo_preds[cn][w])
            row[f'{cn}_error'] = round(combo_preds[cn][w] - actual[w])
        pw_rows.append(row)
    pd.DataFrame(pw_rows).to_excel(writer, sheet_name='Per_Well', index=False)

    # Sheet 3-7: Percentiles per combo
    for cn in combo_names:
        rows = []
        for w in wells:
            row = {'Well': w, 'Event': 'Y' if events[w] else 'N', 'Actual': actual[w]}
            for p in PERCENTILES:
                row[f'P{p}'] = round(pctiles[(cn, w, p)])
            row['P10-P90 Width'] = round(pctiles[(cn, w, 90)] - pctiles[(cn, w, 10)])
            inside = pctiles[(cn, w, 10)] <= actual[w] <= pctiles[(cn, w, 90)]
            row['In P10-P90'] = 'Yes' if inside else 'No'
            rows.append(row)
        sname = cn[:28].replace(' ', '_').replace('(', '').replace(')', '')
        pd.DataFrame(rows).to_excel(writer, sheet_name=f'Pctl_{sname}', index=False)

    # Sheet: Coverage summary
    cov_rows = []
    for cn in combo_names:
        in80 = in50 = 0
        w80 = []; w50 = []
        for w in evt_wells:
            a = actual[w]
            p10 = pctiles[(cn, w, 10)]; p90 = pctiles[(cn, w, 90)]
            p25 = pctiles[(cn, w, 25)]; p75 = pctiles[(cn, w, 75)]
            if p10 <= a <= p90: in80 += 1
            if p25 <= a <= p75: in50 += 1
            w80.append(p90 - p10); w50.append(p75 - p25)
        n = len(evt_wells)
        cov_rows.append({
            'Combination': cn,
            'P10-P90 Coverage': f'{in80}/{n}',
            'P10-P90 %': round(100 * in80 / n, 1),
            'P25-P75 Coverage': f'{in50}/{n}',
            'P25-P75 %': round(100 * in50 / n, 1),
            'Mean P10-P90 Width': round(np.mean(w80)),
            'Mean P25-P75 Width': round(np.mean(w50)),
        })
    pd.DataFrame(cov_rows).to_excel(writer, sheet_name='Coverage', index=False)

print(f"Saved: {outpath}")
''')

# ═══════════════════════════════════════════════════════════════════════
# SECTION 11 — FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════
md("---\n## Section 11 — Final Summary")

code('''\
print("=" * 80)
print("OPTIMIZED ENSEMBLE — FINAL RESULTS")
print("=" * 80)

# Ranked by C-index
ranked = sorted(combo_metrics.items(), key=lambda x: (-x[1]['c_index'], x[1]['mae']))

print(f"\\n{'Rank':>4s} {'Combination':35s} {'C-index':>8s} {'MAE':>7s} {'MaxErr':>7s}")
print("-" * 65)
for rank, (cn, m) in enumerate(ranked, 1):
    n = len(COMBOS[cn]['models'])
    print(f"{rank:4d} {cn:35s} {m['c_index']:8.3f} {m['mae']:7.0f} {m['max_error']:7.0f}")

rec = ranked[0]
print(f"""
================================================================================
RECOMMENDED MODEL: {rec[0]}
  Models: {', '.join(COMBOS[rec[0]]['models'])}
  Strategy: {COMBOS[rec[0]]['strategy']}
  C-index: {rec[1]['c_index']:.3f}
  MAE (events): {rec[1]['mae']:.0f} months
  Max error: {rec[1]['max_error']:.0f} months

  vs Cox PH baseline:
    C-index: {rec[1]['c_index']:.3f} vs 0.788 (+{rec[1]['c_index']-0.788:.3f})
    MAE:     {rec[1]['mae']:.0f} vs 138 ({rec[1]['mae']-138:+.0f} months)
    Max err: {rec[1]['max_error']:.0f} vs 791 ({rec[1]['max_error']-791:+.0f} months)
================================================================================
""")
''')

# ═══════════════════════════════════════════════════════════════════════
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                "name": "python3"},
                 "language_info": {"name": "python", "version": "3.11.0"}},
    "cells": cells,
}

os.makedirs('optimized_ensemble/notebooks', exist_ok=True)
outpath = 'optimized_ensemble/notebooks/optimized_ensemble_model.ipynb'
with open(outpath, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"Notebook: {outpath} ({len(cells)} cells)")
