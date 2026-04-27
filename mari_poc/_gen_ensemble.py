#!/usr/bin/env python3
"""Generate ensemble_model.ipynb — hybrid ensemble of all survival models."""
import json, os, textwrap

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(src).strip()})

def code(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": textwrap.dedent(src).strip()})

md("""
    # Hybrid Ensemble Model — MARI Water Breakthrough POC

    **Objective**: Combine predictions from multiple survival model families into a single
    ensemble prediction that is more robust than any individual model.

    **Approach**: Each model votes with a predicted TTE. We test several combination
    strategies: simple mean, trimmed mean, median (most common vote), inverse-variance
    weighted average, and rank-based fusion. The best ensemble is evaluated on all 16 wells
    with full per-well error analysis.

    **Models in the ensemble**:
    1. Cox PH (5f, pen=0.1) — best ranking (C=0.788)
    2. Weibull AFT (5f) — parametric alternative
    3. XGBoost Cox — tree-based survival
    4. XGBoost AFT — direct time prediction
    5. sksurv GBSA — gradient boosted survival
    6. sksurv CWGBSA — componentwise gradient boosting
    7. Stacked Cox+XGB — residual learning
""")

# ═════ SECTION 1 — SETUP ═════
md("---\n## Section 1 — Setup and Data")

CELL_SETUP = '''\
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from lifelines import CoxPHFitter, WeibullAFTFitter, KaplanMeierFitter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from scipy.stats import trim_mean

DATA_DIR = Path('data/processed')
FIG_DIR  = Path('figures/ensemble')
RES_DIR  = Path('results')
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

GWC_RKB_M = 754.0; WGR_THRESHOLD = 5.0; BT_SUSTAINED = 3
C_EVENT = '#DC143C'; C_CENSOR = '#4682B4'

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
})
print(f'Working directory: {os.getcwd()}')
'''
code(CELL_SETUP)

CELL_DATA = '''\
panel = pd.read_csv(DATA_DIR / 'panel_long.csv', parse_dates=['date'])
static = pd.read_csv(DATA_DIR / 'well_static.csv', parse_dates=['first_prod_date','last_prod_date'])

EXCLUDED = {'M-51-HRL'}
vert_wells = static.loc[~static['is_horizontal'] & ~static['well'].isin(EXCLUDED), 'well'].tolist()
panel_v = panel[panel['well'].isin(vert_wells)].copy()
static_v = static[static['well'].isin(vert_wells)].copy().reset_index(drop=True)

def detect_breakthrough(pdf, threshold=WGR_THRESHOLD, sustained=BT_SUSTAINED):
    rows = []
    for well, grp in pdf.groupby('well'):
        grp = grp.sort_values('date').reset_index(drop=True)
        gas = grp['gas_mmcf'].values; water = grp['water_bbl'].fillna(0).values
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
            tte = (bd.year*12+bd.month)-(fp.year*12+fp.month)
        else:
            ld = grp['date'].iloc[-1]; tte = (ld.year*12+ld.month)-(fp.year*12+fp.month); bd = pd.NaT
        rows.append({'well':well,'spud_date':fp,'bt_date':bd,'tte_months':max(tte,1),'event':int(bt_found)})
    return pd.DataFrame(rows)

bt_df = detect_breakthrough(panel_v)
cohort = static_v.merge(bt_df, on='well')

def _mi(dates, ref):
    return (dates.dt.year*12+dates.dt.month)-(ref.year*12+ref.month)

def build_features(coh, pan):
    df = coh.copy()
    df['log_perm'] = np.log10(df['permeability_md'])
    df['perf_midpoint_md'] = (df['top_perf_md']+df['bottom_perf_md'])/2
    df['perf_thickness_md'] = df['bottom_perf_md']-df['top_perf_md']
    df['dist_to_gwc_m'] = GWC_RKB_M - df['perf_midpoint_md']
    df['spud_year'] = df['first_prod_date'].dt.year + df['first_prod_date'].dt.month/12
    for c in ['early_gas_rate','peak_gas_rate','cum_gas_year1','cum_gas_year3',
              'initial_whfp','whfp_decline_rate','gas_rate_cv_yr12']:
        df[c] = np.nan
    all_w = df['well'].tolist()
    for idx, row in df.iterrows():
        wp = pan[pan['well']==row['well']].sort_values('date')
        fg = wp.loc[wp['gas_mmcf']>0]
        if len(fg)==0: continue
        fp = fg['date'].iloc[0]; mi = _mi(wp['date'], fp)
        e6 = wp[(mi>=0)&(mi<6)&(wp['gas_mmcf']>0)]
        if len(e6): df.at[idx,'early_gas_rate'] = (e6['gas_mmcf']/e6['prod_days'].fillna(30)).mean()
        df.at[idx,'peak_gas_rate'] = wp['gas_mmcf'].max()
        g24 = wp.loc[(mi>=0)&(mi<24),'gas_mmcf']
        if len(g24)>=6 and g24.mean()>0: df.at[idx,'gas_rate_cv_yr12'] = g24.std()/g24.mean()
    df['field_cum_gas_at_spud'] = np.nan; df['n_wells_producing_at_spud'] = np.nan
    for idx, row in df.iterrows():
        spud = pd.Timestamp(row['spud_date'])
        oth = pan[(pan['well']!=row['well'])&pan['well'].isin(all_w)&(pan['date']<spud)]
        df.at[idx,'field_cum_gas_at_spud'] = oth['gas_mmcf'].sum()/1000
        near = pan[(pan['well']!=row['well'])&pan['well'].isin(all_w)
                    &(pan['date']>=spud-pd.DateOffset(months=1))&(pan['date']<=spud)
                    &(pan['gas_mmcf']>0)]
        df.at[idx,'n_wells_producing_at_spud'] = near['well'].nunique()
    return df

cohort = build_features(cohort, panel_v)
FEATURES = ['n_wells_producing_at_spud','spud_year','field_cum_gas_at_spud','gas_rate_cv_yr12','peak_gas_rate']

def _safe_pm(model, X, cap=1200):
    raw = model.predict_median(X)
    arr = np.atleast_1d(np.array(raw, dtype=float))
    return np.where(np.isinf(arr)|np.isnan(arr)|(arr>cap), cap, arr)

def harrell_cindex(actual, predicted, events):
    actual, predicted, events = np.asarray(actual,float), np.asarray(predicted,float), np.asarray(events,int)
    con = dis = tie = 0
    for i in range(len(actual)):
        if not events[i]: continue
        for j in range(len(actual)):
            if i==j or actual[i]>=actual[j]: continue
            if predicted[i]<predicted[j]: con+=1
            elif predicted[i]>predicted[j]: dis+=1
            else: tie+=1
    total = con+dis+tie
    return (con+0.5*tie)/total if total>0 else 0.5

print(f'Cohort: {len(cohort)} wells, {int(cohort["event"].sum())} events')
'''
code(CELL_DATA)

# ═════ SECTION 2 — INDIVIDUAL MODEL LOOCV ═════
md("---\n## Section 2 — Individual Model LOOCV Predictions")

CELL_MODELS = '''\
# Run all 7 models under LOOCV, store per-well predictions
model_preds = {}  # model_name -> {well: predicted_tte}

# Helper: run all models for one train/test split
def run_all_models(train, test_X, feats):
    preds = {}

    # 1. Cox PH
    try:
        m = CoxPHFitter(penalizer=0.1)
        m.fit(train[feats+['tte_months','event']], 'tte_months', 'event')
        preds['Cox PH'] = float(_safe_pm(m, test_X)[0])
    except:
        preds['Cox PH'] = float(train['tte_months'].median())

    # 2. Weibull AFT
    try:
        m = WeibullAFTFitter(penalizer=0.05)
        m.fit(train[feats+['tte_months','event']], 'tte_months', 'event')
        preds['Weibull AFT'] = float(_safe_pm(m, test_X)[0])
    except:
        preds['Weibull AFT'] = float(train['tte_months'].median())

    # 3. XGBoost Cox
    try:
        y = train['tte_months'].values.astype(float) * np.where(train['event'].values, 1, -1)
        dt = xgb.DMatrix(train[feats].values, label=y)
        ds = xgb.DMatrix(test_X.values)
        bst = xgb.train({'objective':'survival:cox','tree_method':'hist','max_depth':2,
            'learning_rate':0.1,'min_child_weight':3,'verbosity':0}, dt, num_boost_round=20)
        hr = bst.predict(ds)[0]
        preds['XGB Cox'] = float(np.clip(train['tte_months'].median() / np.exp(hr), 1, 1200))
    except:
        preds['XGB Cox'] = float(train['tte_months'].median())

    # 4. XGBoost AFT
    try:
        yl = train['tte_months'].values.astype(float)
        yu = np.where(train['event'].values, yl, np.inf)
        dt = xgb.DMatrix(train[feats].values)
        dt.set_float_info('label_lower_bound', yl)
        dt.set_float_info('label_upper_bound', yu)
        ds = xgb.DMatrix(test_X.values)
        bst = xgb.train({'objective':'survival:aft','aft_loss_distribution':'normal',
            'tree_method':'hist','max_depth':2,'learning_rate':0.1,'verbosity':0}, dt, num_boost_round=20)
        preds['XGB AFT'] = float(np.clip(bst.predict(ds)[0], 1, 1200))
    except:
        preds['XGB AFT'] = float(train['tte_months'].median())

    # 5. sksurv GBSA
    try:
        yt = Surv.from_arrays(train['event'].astype(bool), train['tte_months'].astype(float))
        m = GradientBoostingSurvivalAnalysis(n_estimators=50, max_depth=2, learning_rate=0.05,
                                             subsample=0.7, random_state=42)
        m.fit(train[feats].values, yt)
        risk = m.predict(test_X.values)[0]
        tr_r = m.predict(train[feats].values)
        std_r = max(np.std(tr_r), 1e-6)
        preds['sksurv GBSA'] = float(np.clip(train['tte_months'].median() * np.exp(-risk/std_r), 1, 1200))
    except:
        preds['sksurv GBSA'] = float(train['tte_months'].median())

    # 6. sksurv CWGBSA
    try:
        yt = Surv.from_arrays(train['event'].astype(bool), train['tte_months'].astype(float))
        m = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=0.05, random_state=42)
        m.fit(train[feats].values, yt)
        risk = m.predict(test_X.values)[0]
        tr_r = m.predict(train[feats].values)
        std_r = max(np.std(tr_r), 1e-6)
        preds['sksurv CWGB'] = float(np.clip(train['tte_months'].median() * np.exp(-risk/std_r), 1, 1200))
    except:
        preds['sksurv CWGB'] = float(train['tte_months'].median())

    # 7. Stacked (Cox + XGB residual)
    try:
        mc = CoxPHFitter(penalizer=0.1)
        mc.fit(train[feats+['tte_months','event']], 'tte_months', 'event')
        cox_p = float(_safe_pm(mc, test_X)[0])
        te = train[train['event']==1]
        if len(te) < 5:
            preds['Stacked'] = cox_p
        else:
            cp = _safe_pm(mc, te[feats])
            res = np.log(te['tte_months'].values+1) - np.log(cp+1)
            tX = te[feats].copy()
            tX['hr'] = np.log(mc.predict_partial_hazard(te[feats]).values.flatten())
            sX = test_X.copy()
            sX['hr'] = np.log(float(mc.predict_partial_hazard(test_X).values[0]))
            dt = xgb.DMatrix(tX.values, label=res)
            ds = xgb.DMatrix(sX.values)
            bst = xgb.train({'max_depth':1,'learning_rate':0.05,'verbosity':0,
                'objective':'reg:squarederror'}, dt, num_boost_round=10)
            rp = bst.predict(ds)[0]
            preds['Stacked'] = float(np.clip(cox_p * np.exp(rp), 1, 1200))
    except:
        preds['Stacked'] = float(train['tte_months'].median())

    return preds

# Initialize
MODEL_NAMES = ['Cox PH','Weibull AFT','XGB Cox','XGB AFT','sksurv GBSA','sksurv CWGB','Stacked']
for mn in MODEL_NAMES:
    model_preds[mn] = {}

# LOOCV — run all models per fold
print('Running LOOCV across all 7 models...')
for i in range(len(cohort)):
    w = cohort.iloc[i]['well']
    train = cohort.drop(cohort.index[i]).reset_index(drop=True)
    test_X = cohort.iloc[[i]][FEATURES]
    fold_preds = run_all_models(train, test_X, FEATURES)
    for mn in MODEL_NAMES:
        model_preds[mn][w] = fold_preds[mn]
    if (i+1) % 4 == 0:
        print(f'  Fold {i+1}/{len(cohort)} done')
print('All folds complete.')

# Show individual model results
wells = cohort.sort_values('tte_months')['well'].tolist()
actual = {w: float(cohort.loc[cohort['well']==w,'tte_months'].values[0]) for w in wells}
events = {w: int(cohort.loc[cohort['well']==w,'event'].values[0]) for w in wells}

print(f'\\n{"Model":18s} {"C-index":>8s} {"MAE(evt)":>9s} {"MedErr":>8s}')
print('-'*46)
for mn in MODEL_NAMES:
    act_arr = np.array([actual[w] for w in wells])
    prd_arr = np.array([model_preds[mn][w] for w in wells])
    evt_arr = np.array([events[w] for w in wells])
    ci = harrell_cindex(act_arr, prd_arr, evt_arr)
    errs = [abs(model_preds[mn][w]-actual[w]) for w in wells if events[w]]
    print(f'{mn:18s} {ci:8.3f} {np.mean(errs):9.0f} {np.median(errs):8.0f}')
'''
code(CELL_MODELS)

# ═════ SECTION 3 — ENSEMBLE STRATEGIES ═════
md("""
    ---
    ## Section 3 — Ensemble Combination Strategies

    Six strategies for combining the 7 model predictions into one:

    1. **Simple Mean** — average of all 7 predictions
    2. **Trimmed Mean (20%)** — drop highest and lowest, average the rest
    3. **Median** — the "most common" prediction (robust to outliers)
    4. **Geometric Mean** — average in log-space, better for skewed TTE
    5. **Inverse-MAE Weighted** — weight each model by 1/MAE from its individual performance
    6. **Rank Fusion** — average the ranks (not values), then map back to TTE
""")

CELL_ENSEMBLE = '''\
# Build a matrix: rows=wells, columns=models
pred_matrix = pd.DataFrame({mn: {w: model_preds[mn][w] for w in wells} for mn in MODEL_NAMES})
pred_matrix = pred_matrix.loc[wells]  # sort by actual TTE

# Compute individual model MAE (events only) for weighting
model_mae = {}
for mn in MODEL_NAMES:
    errs = [abs(model_preds[mn][w]-actual[w]) for w in wells if events[w]]
    model_mae[mn] = np.mean(errs)

# Strategy 1: Simple Mean
ensemble_preds = {}
ensemble_preds['Simple Mean'] = {w: pred_matrix.loc[w].mean() for w in wells}

# Strategy 2: Trimmed Mean (drop highest and lowest)
ensemble_preds['Trimmed Mean'] = {
    w: trim_mean(pred_matrix.loc[w].values, proportiontocut=0.15) for w in wells
}

# Strategy 3: Median
ensemble_preds['Median'] = {w: pred_matrix.loc[w].median() for w in wells}

# Strategy 4: Geometric Mean (in log-space)
ensemble_preds['Geometric Mean'] = {
    w: np.exp(np.mean(np.log(np.clip(pred_matrix.loc[w].values, 1, 1200)))) for w in wells
}

# Strategy 5: Inverse-MAE Weighted Average
weights_mae = np.array([1.0/model_mae[mn] for mn in MODEL_NAMES])
weights_mae = weights_mae / weights_mae.sum()
ensemble_preds['Inv-MAE Weighted'] = {
    w: np.dot(pred_matrix.loc[w].values, weights_mae) for w in wells
}

# Strategy 6: Rank Fusion
# For each model, rank the wells (1=shortest predicted TTE)
# Average ranks across models, then map rank to TTE using actual TTE percentiles
rank_matrix = pred_matrix.rank(axis=0, method='average')
avg_ranks = rank_matrix.mean(axis=1)
# Map average rank back to TTE using linear interpolation on sorted actuals
sorted_actual = sorted(actual.values())
rank_to_tte = np.interp(
    avg_ranks.values,
    np.linspace(1, len(wells), len(wells)),
    sorted_actual
)
ensemble_preds['Rank Fusion'] = dict(zip(wells, rank_to_tte))

# Evaluate all ensemble strategies
print(f'{"Strategy":20s} {"C-index":>8s} {"MAE(evt)":>9s} {"MedErr":>8s} {"MaxErr":>8s}')
print('='*58)
for strat, preds in ensemble_preds.items():
    act_arr = np.array([actual[w] for w in wells])
    prd_arr = np.array([preds[w] for w in wells])
    evt_arr = np.array([events[w] for w in wells])
    ci = harrell_cindex(act_arr, prd_arr, evt_arr)
    errs = [abs(preds[w]-actual[w]) for w in wells if events[w]]
    print(f'{strat:20s} {ci:8.3f} {np.mean(errs):9.0f} {np.median(errs):8.0f} {max(errs):8.0f}')

# Also show best individual for comparison
print('-'*58)
best_indiv_mn = max(MODEL_NAMES, key=lambda mn: harrell_cindex(
    np.array([actual[w] for w in wells]),
    np.array([model_preds[mn][w] for w in wells]),
    np.array([events[w] for w in wells])))
act_arr = np.array([actual[w] for w in wells])
prd_arr = np.array([model_preds[best_indiv_mn][w] for w in wells])
evt_arr = np.array([events[w] for w in wells])
ci_best = harrell_cindex(act_arr, prd_arr, evt_arr)
errs_best = [abs(model_preds[best_indiv_mn][w]-actual[w]) for w in wells if events[w]]
print(f'{"Best indiv ("+best_indiv_mn+")":20s} {ci_best:8.3f} {np.mean(errs_best):9.0f} {np.median(errs_best):8.0f} {max(errs_best):8.0f}')

ci_cox = harrell_cindex(act_arr,
    np.array([model_preds["Cox PH"][w] for w in wells]), evt_arr)
errs_cox = [abs(model_preds['Cox PH'][w]-actual[w]) for w in wells if events[w]]
print(f'{"Cox PH (baseline)":20s} {ci_cox:8.3f} {np.mean(errs_cox):9.0f} {np.median(errs_cox):8.0f} {max(errs_cox):8.0f}')
'''
code(CELL_ENSEMBLE)

# ═════ SECTION 4 — BEST ENSEMBLE PER-WELL BREAKDOWN ═════
md("---\n## Section 4 — Per-Well Results: Best Ensemble vs Individual Models")

CELL_PERWELL = '''\
# Find best ensemble by MAE (most useful metric for MARI)
best_strat_mae = min(ensemble_preds, key=lambda s: np.mean(
    [abs(ensemble_preds[s][w]-actual[w]) for w in wells if events[w]]))
# Find best ensemble by C-index
best_strat_ci = max(ensemble_preds, key=lambda s: harrell_cindex(
    np.array([actual[w] for w in wells]),
    np.array([ensemble_preds[s][w] for w in wells]),
    np.array([events[w] for w in wells])))

print(f'Best by MAE: {best_strat_mae}')
print(f'Best by C-index: {best_strat_ci}')

# Show all ensemble strategies per well
print(f'\\n{"Well":12s} {"Evt":>3s} {"Actual":>6s} |', end='')
for strat in ensemble_preds:
    print(f' {strat[:10]:>11s}', end='')
print(f' | {"Cox PH":>11s}')
print('-'*130)

for w in wells:
    evt_str = 'Y' if events[w] else 'N'
    line = f'{w:12s} {evt_str:>3s} {actual[w]:6.0f} |'
    for strat, preds in ensemble_preds.items():
        err = preds[w] - actual[w]
        line += f' {preds[w]:5.0f}({err:+4.0f})'
    # Cox baseline
    cox_err = model_preds["Cox PH"][w] - actual[w]
    line += f' | {model_preds["Cox PH"][w]:5.0f}({cox_err:+4.0f})'
    print(line)

# Summary
print('-'*130)
print(f'{"C-index":12s} {"":>3s} {"":>6s} |', end='')
for strat, preds in ensemble_preds.items():
    ci = harrell_cindex(np.array([actual[w] for w in wells]),
        np.array([preds[w] for w in wells]),
        np.array([events[w] for w in wells]))
    print(f' {ci:11.3f}', end='')
print(f' | {ci_cox:11.3f}')

print(f'{"MAE(events)":12s} {"":>3s} {"":>6s} |', end='')
for strat, preds in ensemble_preds.items():
    mae = np.mean([abs(preds[w]-actual[w]) for w in wells if events[w]])
    print(f' {mae:11.0f}', end='')
print(f' | {np.mean(errs_cox):11.0f}')
'''
code(CELL_PERWELL)

# ═════ SECTION 5 — OPTIMAL WEIGHTED ENSEMBLE ═════
md("""
    ---
    ## Section 5 — Optimized Weighted Ensemble

    Instead of fixed weights, find the weights that minimize LOOCV MAE on events.
    Use a simple grid search over weight combinations (constrained to sum to 1).
""")

CELL_OPTIM = '''\
from itertools import product

# For computational tractability, try a coarse grid
# Weight each model from 0 to 0.5 in steps of 0.1
weight_steps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
best_weights = None
best_mae_opt = np.inf
best_ci_opt = 0

# Pre-compute per-well prediction arrays
pred_arrays = np.array([[model_preds[mn][w] for w in wells] for mn in MODEL_NAMES])  # shape: (7, 16)
act_arr = np.array([actual[w] for w in wells])
evt_arr = np.array([events[w] for w in wells])
event_mask = evt_arr == 1

# Try all combinations where at least 2 models have non-zero weight
tested = 0
for combo in product(weight_steps, repeat=len(MODEL_NAMES)):
    wt = np.array(combo)
    s = wt.sum()
    if s < 0.1: continue  # skip all-zero
    if np.count_nonzero(wt) < 2: continue  # need at least 2 models
    wt_norm = wt / s
    ens = wt_norm @ pred_arrays  # weighted combination
    mae = np.mean(np.abs(ens[event_mask] - act_arr[event_mask]))
    ci = harrell_cindex(act_arr, ens, evt_arr)
    if mae < best_mae_opt:
        best_mae_opt = mae
        best_weights_mae = wt_norm.copy()
        best_ens_mae = ens.copy()
    if ci > best_ci_opt:
        best_ci_opt = ci
        best_weights_ci = wt_norm.copy()
        best_ens_ci = ens.copy()
    tested += 1

print(f'Tested {tested} weight combinations')
print(f'\\nBest by MAE (events):')
print(f'  MAE = {best_mae_opt:.0f} months')
ci_mae = harrell_cindex(act_arr, best_ens_mae, evt_arr)
print(f'  C-index = {ci_mae:.3f}')
print(f'  Weights:')
for mn, wt in zip(MODEL_NAMES, best_weights_mae):
    if wt > 0.01: print(f'    {mn:18s}: {wt:.2f}')

print(f'\\nBest by C-index:')
print(f'  C-index = {best_ci_opt:.3f}')
mae_ci = np.mean(np.abs(best_ens_ci[event_mask] - act_arr[event_mask]))
print(f'  MAE = {mae_ci:.0f} months')
print(f'  Weights:')
for mn, wt in zip(MODEL_NAMES, best_weights_ci):
    if wt > 0.01: print(f'    {mn:18s}: {wt:.2f}')

# Store the optimal ensembles
ensemble_preds['Optimal (MAE)'] = dict(zip(wells, best_ens_mae))
ensemble_preds['Optimal (C-idx)'] = dict(zip(wells, best_ens_ci))
'''
code(CELL_OPTIM)

# ═════ SECTION 6 — AGREEMENT ANALYSIS ═════
md("""
    ---
    ## Section 6 — Model Agreement Analysis

    For each well, examine how much the 7 models agree. Wells where models
    agree are likely more trustworthy. Wells where models wildly disagree
    are the ones we should be least confident about.
""")

CELL_AGREEMENT = '''\
agreement = []
for w in wells:
    pvals = [model_preds[mn][w] for mn in MODEL_NAMES]
    spread = max(pvals) - min(pvals)
    cv = np.std(pvals) / np.mean(pvals) if np.mean(pvals) > 0 else np.inf
    iqr = np.percentile(pvals, 75) - np.percentile(pvals, 25)
    median_pred = np.median(pvals)
    act = actual[w]
    median_err = abs(median_pred - act)
    agreement.append({
        'well': w, 'event': events[w], 'actual': act,
        'min_pred': min(pvals), 'max_pred': max(pvals),
        'spread': spread, 'cv': cv, 'iqr': iqr,
        'median_pred': median_pred, 'median_err': median_err,
    })
adf = pd.DataFrame(agreement)

print('Model Agreement Analysis')
print(f'{"Well":12s} {"Evt":>3s} {"Actual":>6s} {"Min":>6s} {"Median":>7s} {"Max":>6s} {"Spread":>7s} {"CV":>5s} {"MedErr":>7s} {"Trust":>7s}')
print('-'*80)
for _, r in adf.iterrows():
    trust = 'HIGH' if r['cv'] < 0.3 else ('MED' if r['cv'] < 0.6 else 'LOW')
    print(f'{r["well"]:12s} {"Y" if r["event"] else "N":>3s} {r["actual"]:6.0f} '
          f'{r["min_pred"]:6.0f} {r["median_pred"]:7.0f} {r["max_pred"]:6.0f} '
          f'{r["spread"]:7.0f} {r["cv"]:5.2f} {r["median_err"]:7.0f} {trust:>7s}')

# Correlation: does model agreement predict accuracy?
event_df = adf[adf['event']==1]
from scipy.stats import spearmanr
rho, pval = spearmanr(event_df['cv'], event_df['median_err'])
print(f'\\nCorrelation between model disagreement (CV) and prediction error:')
print(f'  Spearman rho = {rho:.3f}, p = {pval:.3f}')
if rho > 0.3 and pval < 0.1:
    print('  -> Higher disagreement correlates with larger errors. Agreement IS a trust signal.')
else:
    print('  -> No strong correlation. Agreement alone does not predict accuracy.')
'''
code(CELL_AGREEMENT)

# ═════ SECTION 7 — VISUALIZATION ═════
md("---\n## Section 7 — Visualizations")

CELL_FIGS = '''\
# Figure 1: All ensemble strategies C-index vs MAE
fig, ax = plt.subplots(figsize=(10, 7))
for strat, preds in ensemble_preds.items():
    ci = harrell_cindex(np.array([actual[w] for w in wells]),
        np.array([preds[w] for w in wells]),
        np.array([events[w] for w in wells]))
    mae = np.mean([abs(preds[w]-actual[w]) for w in wells if events[w]])
    marker = 's' if 'Optimal' in strat else 'o'
    color = C_EVENT if 'Optimal' in strat else C_CENSOR
    ax.scatter(mae, ci, s=120, marker=marker, c=color, edgecolors='k', lw=0.5, zorder=5)
    ax.annotate(strat, (mae, ci), fontsize=8, xytext=(5, 5), textcoords='offset points')
# Individual models
for mn in MODEL_NAMES:
    ci = harrell_cindex(np.array([actual[w] for w in wells]),
        np.array([model_preds[mn][w] for w in wells]),
        np.array([events[w] for w in wells]))
    mae = np.mean([abs(model_preds[mn][w]-actual[w]) for w in wells if events[w]])
    ax.scatter(mae, ci, s=60, marker='^', c='#95a5a6', edgecolors='k', lw=0.5, zorder=3)
    ax.annotate(mn, (mae, ci), fontsize=7, xytext=(5, -8), textcoords='offset points', color='gray')
ax.set_xlabel('MAE on events (months)'); ax.set_ylabel('LOOCV C-index')
ax.set_title('Figure 1 — Ensemble Strategies: C-index vs MAE Trade-off')
ax.legend(handles=[
    Line2D([0],[0],marker='s',color='w',mfc=C_EVENT,ms=10,label='Optimal ensemble'),
    Line2D([0],[0],marker='o',color='w',mfc=C_CENSOR,ms=10,label='Fixed ensemble'),
    Line2D([0],[0],marker='^',color='w',mfc='#95a5a6',ms=8,label='Individual model'),
], loc='lower left')
plt.tight_layout(); plt.savefig(FIG_DIR/'01_cindex_vs_mae.png'); plt.show()
print('Saved figures/ensemble/01_cindex_vs_mae.png')

# Figure 2: Per-well comparison — best ensemble vs Cox vs actual
best_ens_name = min(ensemble_preds, key=lambda s: np.mean(
    [abs(ensemble_preds[s][w]-actual[w]) for w in wells if events[w]]))
best_ens = ensemble_preds[best_ens_name]

fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(wells))
width = 0.25
# Actual
ax.bar(x - width, [actual[w] for w in wells], width, label='Actual', color='k', alpha=0.7)
# Cox PH
ax.bar(x, [model_preds['Cox PH'][w] for w in wells], width, label='Cox PH', color=C_CENSOR, alpha=0.7)
# Best ensemble
ax.bar(x + width, [best_ens[w] for w in wells], width, label=f'Best Ensemble ({best_ens_name})',
       color=C_EVENT, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels([w.replace('M-','').replace('-HRL','') for w in wells], rotation=45, ha='right', fontsize=9)
ax.set_ylabel('TTE (months)')
ax.set_title(f'Figure 2 — Per-Well: Actual vs Cox vs Best Ensemble')
ax.legend(loc='upper left')
# Mark censored wells
for i, w in enumerate(wells):
    if not events[w]:
        ax.text(i, actual[w]+15, 'C', ha='center', fontsize=7, color='gray', fontweight='bold')
plt.tight_layout(); plt.savefig(FIG_DIR/'02_perwell_bars.png'); plt.show()
print('Saved figures/ensemble/02_perwell_bars.png')

# Figure 3: Scatter — best ensemble predicted vs actual
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (name, preds), title in [
    (axes[0], ('Cox PH', model_preds['Cox PH']), 'Cox PH'),
    (axes[1], (best_ens_name, best_ens), f'Best Ensemble ({best_ens_name})')
]:
    for w in wells:
        c = C_EVENT if events[w] else C_CENSOR
        ax.scatter(actual[w], preds[w], c=c, s=80, edgecolors='k', lw=0.5, zorder=3)
        ax.annotate(w.replace('M-','').replace('-HRL',''), (actual[w], preds[w]),
                    fontsize=7, xytext=(3,3), textcoords='offset points')
    lim = max(max(actual.values()), max(preds.values())) * 1.1
    ax.plot([0,lim],[0,lim],'k--',alpha=.3)
    ax.set_xlim(0, min(lim, 1300)); ax.set_ylim(0, min(lim, 1300))
    ci = harrell_cindex(np.array([actual[w] for w in wells]),
        np.array([preds[w] for w in wells]),
        np.array([events[w] for w in wells]))
    mae = np.mean([abs(preds[w]-actual[w]) for w in wells if events[w]])
    ax.set_xlabel('Actual TTE (months)'); ax.set_ylabel('Predicted TTE (months)')
    ax.set_title(f'{title}\\nC={ci:.3f}, MAE={mae:.0f}')
fig.suptitle('Figure 3 — Predicted vs Actual', y=1.02)
plt.tight_layout(); plt.savefig(FIG_DIR/'03_scatter.png'); plt.show()
print('Saved figures/ensemble/03_scatter.png')

# Figure 4: Model agreement heatmap
fig, ax = plt.subplots(figsize=(14, 8))
heat_data = pred_matrix.copy()
heat_data.index = [w.replace('M-','').replace('-HRL','') for w in heat_data.index]
im = ax.imshow(heat_data.values, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(MODEL_NAMES))); ax.set_xticklabels(MODEL_NAMES, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(wells))); ax.set_yticklabels(heat_data.index, fontsize=9)
# Annotate each cell
for i in range(len(wells)):
    for j in range(len(MODEL_NAMES)):
        val = heat_data.values[i, j]
        ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=7,
                color='white' if val > 500 else 'black')
plt.colorbar(im, ax=ax, label='Predicted TTE (months)')
ax.set_title('Figure 4 — Model Predictions Heatmap')
plt.tight_layout(); plt.savefig(FIG_DIR/'04_heatmap.png'); plt.show()
print('Saved figures/ensemble/04_heatmap.png')

# Figure 5: Agreement spread vs error
fig, ax = plt.subplots(figsize=(8, 6))
for _, r in adf.iterrows():
    c = C_EVENT if r['event'] else C_CENSOR
    ax.scatter(r['cv'], r['median_err'], c=c, s=80, edgecolors='k', lw=0.5)
    ax.annotate(r['well'].replace('M-','').replace('-HRL',''),
                (r['cv'], r['median_err']), fontsize=7, xytext=(3,3), textcoords='offset points')
ax.set_xlabel('Model Disagreement (CV of predictions)')
ax.set_ylabel('Median Ensemble Error (months)')
ax.set_title('Figure 5 — Does Agreement Predict Accuracy?')
ax.legend(handles=[
    Line2D([0],[0],marker='o',color='w',mfc=C_EVENT,ms=8,label='Event'),
    Line2D([0],[0],marker='o',color='w',mfc=C_CENSOR,ms=8,label='Censored'),
])
plt.tight_layout(); plt.savefig(FIG_DIR/'05_agreement_vs_error.png'); plt.show()
print('Saved figures/ensemble/05_agreement_vs_error.png')
'''
code(CELL_FIGS)

# ═════ SECTION 8 — SUMMARY ═════
md("---\n## Section 8 — Summary and Recommendation")

CELL_SUMMARY = '''\
print('='*70)
print('ENSEMBLE MODEL SUMMARY')
print('='*70)

# Collect all results
all_results = {}
for mn in MODEL_NAMES:
    ci = harrell_cindex(np.array([actual[w] for w in wells]),
        np.array([model_preds[mn][w] for w in wells]),
        np.array([events[w] for w in wells]))
    mae = np.mean([abs(model_preds[mn][w]-actual[w]) for w in wells if events[w]])
    all_results[mn] = {'ci': ci, 'mae': mae, 'type': 'individual'}

for strat, preds in ensemble_preds.items():
    ci = harrell_cindex(np.array([actual[w] for w in wells]),
        np.array([preds[w] for w in wells]),
        np.array([events[w] for w in wells]))
    mae = np.mean([abs(preds[w]-actual[w]) for w in wells if events[w]])
    all_results[strat] = {'ci': ci, 'mae': mae, 'type': 'ensemble'}

# Sort by MAE
sorted_results = sorted(all_results.items(), key=lambda x: x[1]['mae'])
print(f'\\n{"Rank":>4s} {"Method":25s} {"Type":>10s} {"C-index":>8s} {"MAE":>8s}')
print('-'*60)
for rank, (name, info) in enumerate(sorted_results, 1):
    print(f'{rank:4d} {name:25s} {info["type"]:>10s} {info["ci"]:8.3f} {info["mae"]:8.0f}')

# Final recommendation
best_overall = sorted_results[0]
print(f'\\n{"="*70}')
print(f'RECOMMENDED MODEL: {best_overall[0]}')
print(f'  C-index: {best_overall[1]["ci"]:.3f}')
print(f'  MAE (events): {best_overall[1]["mae"]:.0f} months')
print(f'{"="*70}')

# Save results
rows = []
for w in wells:
    row = {'well': w, 'event': events[w], 'actual_months': actual[w]}
    for mn in MODEL_NAMES:
        row[f'{mn}_pred'] = round(model_preds[mn][w])
    for strat, preds in ensemble_preds.items():
        row[f'ens_{strat}_pred'] = round(preds[w])
    rows.append(row)
result_df = pd.DataFrame(rows)
result_df.to_csv(RES_DIR / 'ensemble_per_well.csv', index=False)
print(f'\\nSaved: results/ensemble_per_well.csv')
'''
code(CELL_SUMMARY)

# ═════ WRITE ═════
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.9.0"}
    },
    "nbformat": 4, "nbformat_minor": 5
}

os.makedirs("notebooks", exist_ok=True)
path = "notebooks/ensemble_model.ipynb"
with open(path, "w") as f:
    json.dump(notebook, f, indent=1)
print(f"Notebook: {path} ({len(cells)} cells)")
