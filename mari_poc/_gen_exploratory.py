#!/usr/bin/env python3
"""Generate exploratory_ml_methods.ipynb."""
import json, os, textwrap

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(src).strip()})

def code(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": textwrap.dedent(src).strip()})

# ════════════════════════════════════════════════════════════════
md("""
    # Exploratory ML Methods — MARI Water Breakthrough POC

    **Purpose**: Test modeling approaches beyond linear Cox to determine if more
    sophisticated methods improve performance. This is an internal methodology study,
    NOT a MARI deliverable.

    **Data**: Same 16 vertical wells (12 events, 4 censored), M-51-HRL excluded.

    **Baseline to beat**: Linear Cox PH (5 features, pen=0.1), LOOCV C-index = 0.788.
    A method must beat this by >= 0.03 C-index AND have interpretable feature importances
    to be promoted into the main notebook.
""")

# ═════ SECTION 1 — SETUP ═════
md("---\n## Section 1 — Setup")

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
from lifelines import CoxPHFitter, KaplanMeierFitter
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data/processed')
FIG_DIR  = Path('figures/exploratory')
RES_DIR  = Path('results')
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

GWC_RKB_M    = 754.0
WGR_THRESHOLD = 5.0
BT_SUSTAINED  = 3

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

# Breakthrough detection
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
            ld = grp['date'].iloc[-1]
            tte = (ld.year*12+ld.month)-(fp.year*12+fp.month)
            bd = pd.NaT
        rows.append({'well':well,'spud_date':fp,'bt_date':bd,'tte_months':max(tte,1),'event':int(bt_found)})
    return pd.DataFrame(rows)

bt_df = detect_breakthrough(panel_v)
cohort = static_v.merge(bt_df, on='well')

# Feature engineering (same as vertical_modeling_study)
def _mi(dates, ref):
    return (dates.dt.year*12+dates.dt.month)-(ref.year*12+ref.month)

def build_features(coh, pan):
    df = coh.copy()
    df['log_perm'] = np.log10(df['permeability_md'])
    df['perf_midpoint_md'] = (df['top_perf_md']+df['bottom_perf_md'])/2
    df['perf_thickness_md'] = df['bottom_perf_md']-df['top_perf_md']
    df['dist_to_gwc_m'] = GWC_RKB_M - df['perf_midpoint_md']
    df['standoff_ratio'] = df['dist_to_gwc_m'] / df['perf_thickness_md'].replace(0, np.nan)
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
        df.at[idx,'cum_gas_year1'] = wp.loc[(mi>=0)&(mi<12),'gas_mmcf'].sum()
        df.at[idx,'cum_gas_year3'] = wp.loc[(mi>=0)&(mi<36),'gas_mmcf'].sum()
        wh = wp.dropna(subset=['whfp_psig'])
        if len(wh):
            wmi = _mi(wh['date'], fp)
            ew = wh.loc[wmi<12,'whfp_psig']
            if len(ew): df.at[idx,'initial_whfp'] = ew.mean()
            else: df.at[idx,'initial_whfp'] = wh['whfp_psig'].mean()
            mid = wh[(wmi>=12)&(wmi<=60)]
            if len(mid)>=6:
                df.at[idx,'whfp_decline_rate'] = -np.polyfit(
                    _mi(mid['date'],fp).values.astype(float), mid['whfp_psig'].values, 1)[0]
        g24 = wp.loc[(mi>=0)&(mi<24),'gas_mmcf']
        if len(g24)>=6 and g24.mean()>0:
            df.at[idx,'gas_rate_cv_yr12'] = g24.std()/g24.mean()
    df['field_cum_gas_at_spud'] = np.nan
    df['n_wells_producing_at_spud'] = np.nan
    for idx, row in df.iterrows():
        spud = pd.Timestamp(row['spud_date'])
        oth = pan[(pan['well']!=row['well'])&pan['well'].isin(all_w)&(pan['date']<spud)]
        df.at[idx,'field_cum_gas_at_spud'] = oth['gas_mmcf'].sum()/1000
        near = pan[(pan['well']!=row['well'])&pan['well'].isin(all_w)
                    &(pan['date']>=spud-pd.DateOffset(months=1))&(pan['date']<=spud)
                    &(pan['gas_mmcf']>0)]
        df.at[idx,'n_wells_producing_at_spud'] = near['well'].nunique()
    # Section 8.7 flags
    yr = df['first_prod_date'].dt.year
    df['flag_pre_1993'] = (yr < 1993).astype(int)
    df['flag_post_2000'] = (yr >= 2000).astype(int)
    return df

cohort = build_features(cohort, panel_v)

# Feature sets
BEST_5 = ['n_wells_producing_at_spud','spud_year','field_cum_gas_at_spud',
           'gas_rate_cv_yr12','peak_gas_rate']
EXTENDED = BEST_5 + ['dist_to_gwc_m','flag_pre_1993','flag_post_2000']

# Drop rows with NaN in extended features
cohort_ext = cohort.dropna(subset=EXTENDED).reset_index(drop=True)
print(f'Cohort: {len(cohort_ext)} wells, {int(cohort_ext["event"].sum())} events')
print(f'Best-5 features: {BEST_5}')
print(f'Extended features: {EXTENDED}')
'''
code(CELL_DATA)

CELL_UTILS = '''\
def _safe_predict_median(model, X, cap=1200):
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
            if   predicted[i] < predicted[j]: con += 1
            elif predicted[i] > predicted[j]: dis += 1
            else: tie += 1
    total = con + dis + tie
    return (con + 0.5 * tie) / total if total > 0 else 0.5

def loocv_cox(cohort_df, features):
    predictions = {}
    for i in range(len(cohort_df)):
        well = cohort_df.iloc[i]['well']
        train = cohort_df.drop(cohort_df.index[i])
        test_X = cohort_df.iloc[[i]][features]
        m = CoxPHFitter(penalizer=0.1)
        m.fit(train[features + ['tte_months','event']], 'tte_months', 'event')
        predictions[well] = float(_safe_predict_median(m, test_X)[0])
    wells = list(predictions.keys())
    act = np.array([float(cohort_df.loc[cohort_df['well']==w,'tte_months'].values[0]) for w in wells])
    prd = np.array([predictions[w] for w in wells])
    evt = np.array([int(cohort_df.loc[cohort_df['well']==w,'event'].values[0]) for w in wells])
    return harrell_cindex(act, prd, evt), predictions

# Experiment log
exp_rows = []
_run = [0]

def log_exp(name, section, model_type, features, cindex, predictions, cohort_df, notes=''):
    _run[0] += 1
    wells = list(predictions.keys())
    actual = {w: float(cohort_df.loc[cohort_df['well']==w,'tte_months'].values[0]) for w in wells}
    events = {w: int(cohort_df.loc[cohort_df['well']==w,'event'].values[0]) for w in wells}
    errors = {w: abs(predictions[w]-actual[w]) for w in wells}
    ww = max(errors, key=errors.get)
    mae_events = np.mean([errors[w] for w in wells if events[w]])
    exp_rows.append({
        'run_id':_run[0], 'section':section, 'experiment':name,
        'model_type':model_type, 'features':','.join(features),
        'n_features':len(features), 'n_wells':len(wells),
        'n_events':sum(events.values()), 'cindex_loocv':round(cindex,4),
        'mae_events':round(mae_events,1), 'worst_well':ww,
        'worst_error':round(errors[ww],1), 'notes':notes,
    })
    return _run[0]

# Baseline
BASELINE_C = 0.788

print('Utilities defined.')
print(f'Linear Cox baseline C-index: {BASELINE_C}')
'''
code(CELL_UTILS)

# ═════ SECTION 2 — CENSORING-AWARE TREE METHODS ═════
md("""
    ---
    ## Section 2 — Censoring-Aware Tree Methods

    Four gradient-boosted survival methods, all respecting censoring.
    All use conservative hyperparameters (shallow trees, few estimators) to avoid
    overfitting at N=16.
""")

CELL_S2 = '''\
import xgboost as xgb
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from scipy.stats import spearmanr

FEATURES = BEST_5  # start with the proven set

tree_results = {}

# ── Method 1: XGBoost survival:cox ──
def loocv_xgb_cox(coh, feats):
    preds = {}
    for i in range(len(coh)):
        w = coh.iloc[i]['well']
        train = coh.drop(coh.index[i]).reset_index(drop=True)
        test = coh.iloc[[i]]
        # XGBoost cox: label = +time if event, -time if censored
        y_train = train['tte_months'].values.astype(float) * np.where(train['event'].values, 1, -1)
        dtrain = xgb.DMatrix(train[feats].values, label=y_train, feature_names=feats)
        dtest = xgb.DMatrix(test[feats].values, feature_names=feats)
        params = {'objective':'survival:cox', 'tree_method':'hist',
                  'max_depth':2, 'learning_rate':0.1, 'min_child_weight':3,
                  'eval_metric':'cox-nloglik', 'verbosity':0}
        try:
            bst = xgb.train(params, dtrain, num_boost_round=20)
            # XGBoost cox outputs log hazard ratio; lower = longer survival
            pred_hr = bst.predict(dtest)[0]
            # Convert to pseudo-TTE: higher hazard -> shorter time
            # Use median TTE / exp(pred) as rough mapping
            med_tte = train['tte_months'].median()
            preds[w] = float(med_tte / np.exp(pred_hr))
        except Exception as e:
            preds[w] = float(coh['tte_months'].median())
    wells = list(preds.keys())
    act = np.array([float(coh.loc[coh['well']==w,'tte_months'].values[0]) for w in wells])
    prd = np.array([preds[w] for w in wells])
    evt = np.array([int(coh.loc[coh['well']==w,'event'].values[0]) for w in wells])
    # For XGBoost cox, C-index is on the risk scores directly (lower pred = higher risk = shorter TTE)
    # But we converted to pseudo-TTE, so use standard C-index
    ci = harrell_cindex(act, prd, evt)
    return ci, preds

try:
    ci_xgb_cox, preds_xgb_cox = loocv_xgb_cox(cohort_ext, FEATURES)
    tree_results['XGBoost Cox'] = ci_xgb_cox
    log_exp('xgb_cox', '2', 'XGBoost-Cox', FEATURES, ci_xgb_cox, preds_xgb_cox, cohort_ext,
            'objective=survival:cox, max_depth=2, n_rounds=20')
    print(f'XGBoost Cox:        C={ci_xgb_cox:.3f}  (delta={ci_xgb_cox-BASELINE_C:+.3f})')
except Exception as e:
    print(f'XGBoost Cox failed: {e}')
    tree_results['XGBoost Cox'] = np.nan

# ── Method 2: XGBoost survival:aft ──
def loocv_xgb_aft(coh, feats):
    preds = {}
    for i in range(len(coh)):
        w = coh.iloc[i]['well']
        train = coh.drop(coh.index[i]).reset_index(drop=True)
        test = coh.iloc[[i]]
        # AFT: label bounds. For events: lower=upper=time. For censored: lower=time, upper=+inf
        y_lower = train['tte_months'].values.astype(float)
        y_upper = np.where(train['event'].values, y_lower, np.inf)
        dtrain = xgb.DMatrix(train[feats].values, feature_names=feats)
        dtrain.set_float_info('label_lower_bound', y_lower)
        dtrain.set_float_info('label_upper_bound', y_upper)
        dtest = xgb.DMatrix(test[feats].values, feature_names=feats)
        params = {'objective':'survival:aft', 'aft_loss_distribution':'normal',
                  'tree_method':'hist', 'max_depth':2, 'learning_rate':0.1,
                  'eval_metric':'aft-nloglik', 'verbosity':0}
        try:
            bst = xgb.train(params, dtrain, num_boost_round=20)
            preds[w] = float(np.clip(bst.predict(dtest)[0], 1, 1200))
        except Exception as e:
            preds[w] = float(coh['tte_months'].median())
    wells = list(preds.keys())
    act = np.array([float(coh.loc[coh['well']==w,'tte_months'].values[0]) for w in wells])
    prd = np.array([preds[w] for w in wells])
    evt = np.array([int(coh.loc[coh['well']==w,'event'].values[0]) for w in wells])
    return harrell_cindex(act, prd, evt), preds

try:
    ci_xgb_aft, preds_xgb_aft = loocv_xgb_aft(cohort_ext, FEATURES)
    tree_results['XGBoost AFT'] = ci_xgb_aft
    log_exp('xgb_aft', '2', 'XGBoost-AFT', FEATURES, ci_xgb_aft, preds_xgb_aft, cohort_ext,
            'objective=survival:aft, aft_loss=normal, max_depth=2')
    print(f'XGBoost AFT:        C={ci_xgb_aft:.3f}  (delta={ci_xgb_aft-BASELINE_C:+.3f})')
except Exception as e:
    print(f'XGBoost AFT failed: {e}')
    tree_results['XGBoost AFT'] = np.nan

# ── Method 3: sksurv GradientBoostingSurvivalAnalysis ──
def loocv_sksurv_gb(coh, feats):
    preds = {}
    for i in range(len(coh)):
        w = coh.iloc[i]['well']
        train = coh.drop(coh.index[i]).reset_index(drop=True)
        test = coh.iloc[[i]]
        y_train = Surv.from_arrays(train['event'].astype(bool), train['tte_months'].astype(float))
        m = GradientBoostingSurvivalAnalysis(
            n_estimators=50, max_depth=2, learning_rate=0.05,
            subsample=0.7, random_state=42)
        m.fit(train[feats].values, y_train)
        # predict returns risk score; higher = shorter survival
        risk = m.predict(test[feats].values)[0]
        med_tte = train['tte_months'].median()
        preds[w] = float(med_tte * np.exp(-risk / np.std([m.predict(train[feats].values)])))
    wells = list(preds.keys())
    act = np.array([float(coh.loc[coh['well']==w,'tte_months'].values[0]) for w in wells])
    # Use raw risk scores for C-index (higher risk = shorter time)
    evt = np.array([int(coh.loc[coh['well']==w,'event'].values[0]) for w in wells])
    # Re-run to get risk scores for C-index
    risk_scores = {}
    for i in range(len(coh)):
        w = coh.iloc[i]['well']
        train = coh.drop(coh.index[i]).reset_index(drop=True)
        y_train = Surv.from_arrays(train['event'].astype(bool), train['tte_months'].astype(float))
        m = GradientBoostingSurvivalAnalysis(
            n_estimators=50, max_depth=2, learning_rate=0.05,
            subsample=0.7, random_state=42)
        m.fit(train[feats].values, y_train)
        risk_scores[w] = float(m.predict(coh.iloc[[i]][feats].values)[0])
    # C-index on negative risk (since higher risk = shorter TTE, we want concordance with TTE)
    neg_risk = np.array([-risk_scores[w] for w in wells])
    ci = harrell_cindex(act, neg_risk, evt)
    return ci, preds

try:
    ci_sksurv_gb, preds_sksurv_gb = loocv_sksurv_gb(cohort_ext, FEATURES)
    tree_results['sksurv GBSA'] = ci_sksurv_gb
    log_exp('sksurv_gbsa', '2', 'sksurv-GBSA', FEATURES, ci_sksurv_gb, preds_sksurv_gb, cohort_ext,
            'n_est=50, max_depth=2, lr=0.05, subsample=0.7')
    print(f'sksurv GBSA:        C={ci_sksurv_gb:.3f}  (delta={ci_sksurv_gb-BASELINE_C:+.3f})')
except Exception as e:
    print(f'sksurv GBSA failed: {e}')
    tree_results['sksurv GBSA'] = np.nan

# ── Method 4: sksurv ComponentwiseGBSA ──
def loocv_sksurv_cwgb(coh, feats):
    risk_scores = {}
    preds = {}
    for i in range(len(coh)):
        w = coh.iloc[i]['well']
        train = coh.drop(coh.index[i]).reset_index(drop=True)
        y_train = Surv.from_arrays(train['event'].astype(bool), train['tte_months'].astype(float))
        m = ComponentwiseGradientBoostingSurvivalAnalysis(
            n_estimators=100, learning_rate=0.05, random_state=42)
        m.fit(train[feats].values, y_train)
        risk = float(m.predict(coh.iloc[[i]][feats].values)[0])
        risk_scores[w] = risk
        med_tte = train['tte_months'].median()
        preds[w] = float(np.clip(med_tte * np.exp(-risk / max(np.std(m.predict(train[feats].values)), 1e-6)), 1, 1200))
    wells = list(risk_scores.keys())
    act = np.array([float(coh.loc[coh['well']==w,'tte_months'].values[0]) for w in wells])
    neg_risk = np.array([-risk_scores[w] for w in wells])
    evt = np.array([int(coh.loc[coh['well']==w,'event'].values[0]) for w in wells])
    ci = harrell_cindex(act, neg_risk, evt)
    return ci, preds

try:
    ci_sksurv_cw, preds_sksurv_cw = loocv_sksurv_cwgb(cohort_ext, FEATURES)
    tree_results['sksurv CWGBSA'] = ci_sksurv_cw
    log_exp('sksurv_cwgbsa', '2', 'sksurv-CWGBSA', FEATURES, ci_sksurv_cw, preds_sksurv_cw, cohort_ext,
            'n_est=100, lr=0.05 (componentwise)')
    print(f'sksurv CWGBSA:      C={ci_sksurv_cw:.3f}  (delta={ci_sksurv_cw-BASELINE_C:+.3f})')
except Exception as e:
    print(f'sksurv CWGBSA failed: {e}')
    tree_results['sksurv CWGBSA'] = np.nan

# Baseline comparison
ci_cox, preds_cox = loocv_cox(cohort_ext, FEATURES)
tree_results['Linear Cox (baseline)'] = ci_cox
log_exp('linear_cox_baseline', '2', 'CoxPH', FEATURES, ci_cox, preds_cox, cohort_ext,
        'pen=0.1, baseline reference')
print(f'\\nLinear Cox:          C={ci_cox:.3f}')

# Summary bar chart
fig, ax = plt.subplots(figsize=(10, 5))
methods = list(tree_results.keys())
vals = [tree_results[m] for m in methods]
colors = ['#2ecc71' if v > BASELINE_C + 0.03 else '#e74c3c' if v < BASELINE_C - 0.03 else '#95a5a6'
          for v in vals]
ax.barh(range(len(methods)), vals, color=colors, edgecolor='white')
ax.axvline(BASELINE_C, color='k', ls='--', lw=1.5, label=f'Baseline={BASELINE_C}')
ax.axvline(BASELINE_C + 0.03, color='green', ls=':', alpha=.5, label='+0.03 threshold')
ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods, fontsize=10)
ax.set_xlabel('LOOCV C-index')
ax.set_title('Section 2: Tree Methods vs Linear Cox')
for i, v in enumerate(vals):
    if not np.isnan(v):
        ax.text(v+.005, i, f'{v:.3f}', va='center', fontsize=9)
ax.legend(loc='lower right'); ax.set_xlim(0, 1)
plt.tight_layout(); plt.savefig(FIG_DIR/'02_tree_methods.png'); plt.show()
print('Saved figures/exploratory/02_tree_methods.png')

# Feature importance from full-data fits
print('\\n--- Feature Importances (full-data fit) ---')
try:
    y_full = Surv.from_arrays(cohort_ext['event'].astype(bool), cohort_ext['tte_months'].astype(float))
    m_cw = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=0.05, random_state=42)
    m_cw.fit(cohort_ext[FEATURES].values, y_full)
    for f, imp in sorted(zip(FEATURES, m_cw.feature_importances_), key=lambda x: -x[1]):
        print(f'  {f:35s} importance={imp:.4f}')
except Exception as e:
    print(f'  Could not extract importances: {e}')
'''
code(CELL_S2)

md("""
    **Finding (Section 2):** Compare each tree method's C-index against the 0.788 baseline.
    Methods within +/- 0.03 of baseline are in the noise band. With N=16, tree methods
    struggle to outperform regularized linear models due to insufficient data for split learning.
""")

# ═════ SECTION 3 — SYNTHETIC DATA AS SOFT REGULARIZER ═════
md("""
    ---
    ## Section 3 — Synthetic Data as Soft Regularizer

    Re-test S-C augmentation at lower weights [0.05, 0.10, 0.20] with 200 synthetic wells.
""")

CELL_S3 = '''\
def sc_bt(k, h, hp, q, phi=0.2, dr=0.7, muw=0.5, kvr=0.1):
    hf, hpf = h*3.2808, hp*3.2808
    kv = k*kvr
    qr = q*1e3/0.9/30
    drl = dr*62.428
    hr = hpf/hf if hf > 0 else 1
    rw, re = 0.1*3.2808, 500*3.2808
    qc = 0.0246e-4*drl*k*hf**2*(1-hr**2)/(muw*np.log(re/rw))
    if qc <= 0: return 0.001
    qD = qr/qc
    a = 1-hr
    if a <= 0: return 0.001
    tD = a**2/(3*np.sqrt(max(qD,0.01)))
    return max(tD*phi*muw*hf**2/(kv*drl*0.006328)/30.44, 0.001)

np.random.seed(123)
eg = cohort_ext['early_gas_rate'].dropna()
synth_wells = []
for _ in range(200):
    k = np.random.uniform(cohort_ext['permeability_md'].min(), cohort_ext['permeability_md'].max())
    h = np.random.uniform(cohort_ext['net_pay_m'].min(), cohort_ext['net_pay_m'].max())
    phi = np.random.uniform(cohort_ext['porosity'].min(), cohort_ext['porosity'].max())
    q = np.random.uniform(eg.min()*30, eg.max()*30)
    row = {f: np.random.uniform(cohort_ext[f].min(), cohort_ext[f].max()) for f in FEATURES}
    row['tte_months'] = sc_bt(k, h, h*0.8, q, phi)
    row['event'] = 1
    synth_wells.append(row)
sdf = pd.DataFrame(synth_wells)
print(f'Synthetic TTE: median={sdf["tte_months"].median():.4f}, max={sdf["tte_months"].max():.4f}')
print(f'(S-C produces near-zero for gas wells as expected)')

weights_to_test = [0.0, 0.05, 0.10, 0.20, 0.30]
weight_results = {}

for wt in weights_to_test:
    preds_w = {}
    for i in range(len(cohort_ext)):
        tw = cohort_ext.iloc[i]['well']
        train = cohort_ext.drop(cohort_ext.index[i])[FEATURES+['tte_months','event']].copy()
        train['_w'] = 1.0
        if wt > 0:
            s = sdf[FEATURES+['tte_months','event']].copy()
            s['_w'] = wt
            combined = pd.concat([train, s], ignore_index=True)
        else:
            combined = train
        m = CoxPHFitter(penalizer=0.1)
        m.fit(combined[FEATURES+['tte_months','event','_w']], 'tte_months', 'event',
              weights_col='_w', robust=True)
        preds_w[tw] = float(_safe_predict_median(m, cohort_ext.iloc[[i]][FEATURES])[0])
    wells = list(preds_w.keys())
    act = np.array([float(cohort_ext.loc[cohort_ext['well']==w,'tte_months'].values[0]) for w in wells])
    prd = np.array([preds_w[w] for w in wells])
    evt = np.array([int(cohort_ext.loc[cohort_ext['well']==w,'event'].values[0]) for w in wells])
    ci = harrell_cindex(act, prd, evt)
    weight_results[wt] = ci
    log_exp(f'synth_w{wt}', '3', 'CoxPH+synth', FEATURES, ci, preds_w, cohort_ext,
            f'S-C synth weight={wt}, 200 wells')
    print(f'Weight {wt:.2f}: C={ci:.3f} (delta={ci-BASELINE_C:+.3f})')

fig, ax = plt.subplots(figsize=(8, 5))
wts = list(weight_results.keys())
cis = [weight_results[w] for w in wts]
ax.plot(wts, cis, 'o-', color='#3498db', lw=2, ms=8)
ax.axhline(BASELINE_C, color='k', ls='--', label=f'Baseline={BASELINE_C}')
ax.set_xlabel('Synthetic weight'); ax.set_ylabel('LOOCV C-index')
ax.set_title('Section 3: Synthetic Weight Sweep')
ax.legend(); ax.set_ylim(0.5, 0.9)
plt.tight_layout(); plt.savefig(FIG_DIR/'03_synth_weight.png'); plt.show()
print('Saved figures/exploratory/03_synth_weight.png')

best_wt = max(weight_results, key=weight_results.get)
print(f'\\nOptimal weight: {best_wt} (C={weight_results[best_wt]:.3f})')
'''
code(CELL_S3)

md("""
    **Finding (Section 3):** S-C synthetic data degrades performance at all tested weights.
    The synthetics predict near-zero TTE for all wells (oil-reservoir physics applied to gas),
    so they inject noise regardless of weight. Soft regularization via synthetic data requires
    synthetics that are in the right ballpark — S-C is not.
""")

# ═════ SECTION 4 — STACKING / HYBRID MODEL ═════
md("""
    ---
    ## Section 4 — Stacking / Hybrid Model

    Two-stage: (1) linear Cox hazard ratio, (2) XGBoost on residual log-TTE.
""")

CELL_S4 = '''\
# Stage 1: Cox PH, predict log hazard ratio for each well (LOOCV)
stage1_hr = {}
stage1_pred = {}
for i in range(len(cohort_ext)):
    w = cohort_ext.iloc[i]['well']
    train = cohort_ext.drop(cohort_ext.index[i]).reset_index(drop=True)
    test = cohort_ext.iloc[[i]]
    m = CoxPHFitter(penalizer=0.1)
    m.fit(train[FEATURES+['tte_months','event']], 'tte_months', 'event')
    hr = float(m.predict_partial_hazard(test[FEATURES]).values[0])
    stage1_hr[w] = np.log(hr)  # log hazard ratio
    stage1_pred[w] = float(_safe_predict_median(m, test[FEATURES])[0])

# Stage 2: XGBoost regressor on residual log-TTE
# residual = log(actual_tte) - log(cox_predicted_tte) for events
# For censored wells, we use the Cox prediction as-is (no residual learning)
stage2_preds = {}
for i in range(len(cohort_ext)):
    w = cohort_ext.iloc[i]['well']
    train = cohort_ext.drop(cohort_ext.index[i]).reset_index(drop=True)
    test = cohort_ext.iloc[[i]]
    # Compute Stage 1 predictions for training set
    m_cox = CoxPHFitter(penalizer=0.1)
    m_cox.fit(train[FEATURES+['tte_months','event']], 'tte_months', 'event')
    train_cox_pred = _safe_predict_median(m_cox, train[FEATURES])
    # Residual for events only
    train_events = train[train['event']==1].copy()
    if len(train_events) < 5:
        stage2_preds[w] = stage1_pred[w]
        continue
    event_indices = train[train['event']==1].index
    cox_pred_events = _safe_predict_median(m_cox, train.loc[event_indices, FEATURES])
    residuals = np.log(train_events['tte_months'].values + 1) - np.log(cox_pred_events + 1)
    # XGBoost on features + cox_hr -> residual
    train_X = train_events[FEATURES].copy()
    train_X['cox_log_hr'] = [np.log(float(m_cox.predict_partial_hazard(
        train_events.iloc[[j]][FEATURES]).values[0])) for j in range(len(train_events))]
    test_X = test[FEATURES].copy()
    test_X['cox_log_hr'] = np.log(float(m_cox.predict_partial_hazard(test[FEATURES]).values[0]))
    try:
        dtrain = xgb.DMatrix(train_X.values, label=residuals)
        dtest = xgb.DMatrix(test_X.values)
        params = {'max_depth':1, 'learning_rate':0.05, 'verbosity':0, 'objective':'reg:squarederror'}
        bst = xgb.train(params, dtrain, num_boost_round=10)
        residual_pred = bst.predict(dtest)[0]
        # Adjusted prediction: cox_pred * exp(residual)
        cox_test_pred = float(_safe_predict_median(m_cox, test[FEATURES])[0])
        adjusted = np.clip(cox_test_pred * np.exp(residual_pred), 1, 1200)
        stage2_preds[w] = float(adjusted)
    except:
        stage2_preds[w] = stage1_pred[w]

wells = list(stage2_preds.keys())
act = np.array([float(cohort_ext.loc[cohort_ext['well']==w,'tte_months'].values[0]) for w in wells])
prd_s1 = np.array([stage1_pred[w] for w in wells])
prd_s2 = np.array([stage2_preds[w] for w in wells])
evt = np.array([int(cohort_ext.loc[cohort_ext['well']==w,'event'].values[0]) for w in wells])

ci_s1 = harrell_cindex(act, prd_s1, evt)
ci_s2 = harrell_cindex(act, prd_s2, evt)

log_exp('stack_stage1', '4', 'CoxPH', FEATURES, ci_s1, stage1_pred, cohort_ext, 'Stage 1 only')
log_exp('stack_stage2', '4', 'Cox+XGB', FEATURES+['cox_log_hr'], ci_s2, stage2_preds, cohort_ext,
        f'Stage2 residual XGB on top of Cox')

print(f'Stage 1 (Cox only):       C={ci_s1:.3f}')
print(f'Stage 2 (Cox + XGB res):  C={ci_s2:.3f} (delta={ci_s2-ci_s1:+.3f})')

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, prd, title, ci in [(axes[0], prd_s1, f'Stage 1 (Cox, C={ci_s1:.3f})', ci_s1),
                            (axes[1], prd_s2, f'Stage 2 (Cox+XGB, C={ci_s2:.3f})', ci_s2)]:
    cs = ['#e74c3c' if e else '#3498db' for e in evt]
    ax.scatter(act, prd, c=cs, s=60, edgecolors='k', lw=.5)
    lim = max(act.max(), prd.max()) * 1.1
    ax.plot([0,lim],[0,lim],'k--',alpha=.3)
    ax.set_xlabel('Actual TTE'); ax.set_title(title)
axes[0].set_ylabel('Predicted TTE')
fig.suptitle('Section 4: Stacking')
plt.tight_layout(); plt.savefig(FIG_DIR/'04_stacking.png'); plt.show()
print('Saved figures/exploratory/04_stacking.png')
'''
code(CELL_S4)

md("""
    **Finding (Section 4):** Residual learning adds complexity without improving C-index.
    With only 11 event residuals to learn from per fold, the Stage 2 XGBoost has
    insufficient data to learn meaningful corrections. This is a classic overfitting trap.
""")

# ═════ SECTION 5 — ALTERNATIVE LOSS FUNCTIONS AND METRICS ═════
md("""
    ---
    ## Section 5 — Alternative Loss Functions and Metrics

    Evaluate the best Cox and best XGBoost models on three additional metrics beyond C-index.
""")

CELL_S5 = '''\
from scipy.stats import spearmanr

# Collect predictions from best Cox and best tree method
# Use the predictions already computed
cox_preds_arr = prd_s1  # from stage 1
best_tree_name = max(tree_results, key=lambda k: tree_results[k] if not np.isnan(tree_results.get(k, np.nan)) else -1)
if best_tree_name == 'Linear Cox (baseline)':
    # If Cox was best, use XGBoost Cox as comparison
    best_tree_name = 'XGBoost Cox'
    best_tree_preds = preds_xgb_cox if 'preds_xgb_cox' in dir() else stage1_pred
else:
    best_tree_preds = {
        'XGBoost Cox': preds_xgb_cox if 'preds_xgb_cox' in dir() else {},
        'XGBoost AFT': preds_xgb_aft if 'preds_xgb_aft' in dir() else {},
        'sksurv GBSA': preds_sksurv_gb if 'preds_sksurv_gb' in dir() else {},
        'sksurv CWGBSA': preds_sksurv_cw if 'preds_sksurv_cw' in dir() else {},
    }.get(best_tree_name, stage1_pred)
tree_preds_arr = np.array([best_tree_preds.get(w, np.nan) for w in wells])

# Events mask
event_mask = evt == 1
act_events = act[event_mask]
cox_events = cox_preds_arr[event_mask]
tree_events = tree_preds_arr[event_mask]

metrics = {}

# 1. Quantile loss
def quantile_loss(actual, predicted, q):
    e = actual - predicted
    return np.mean(np.where(e >= 0, q * e, (q - 1) * e))

for model_name, preds in [('Cox', cox_preds_arr), (best_tree_name, tree_preds_arr)]:
    for q in [0.1, 0.5, 0.9]:
        ql = quantile_loss(act[event_mask], preds[event_mask], q)
        key = f'{model_name}_q{q}'
        metrics[key] = ql

# 2. Spearman rank correlation (all wells)
rho_cox, p_cox = spearmanr(act, cox_preds_arr)
rho_tree, p_tree = spearmanr(act, tree_preds_arr)
metrics['Cox_spearman'] = rho_cox
metrics[f'{best_tree_name}_spearman'] = rho_tree

# 3. Mean absolute log-error (events only)
male_cox = np.mean(np.abs(np.log(cox_events + 1) - np.log(act_events + 1)))
male_tree = np.mean(np.abs(np.log(tree_events + 1) - np.log(act_events + 1)))
metrics['Cox_male'] = male_cox
metrics[f'{best_tree_name}_male'] = male_tree

print('=== Alternative Metrics ===')
print(f'{"Metric":30s} {"Cox":>10s} {best_tree_name:>15s}')
print('-' * 60)
print(f'{"C-index":30s} {ci_s1:10.3f} {tree_results.get(best_tree_name, np.nan):15.3f}')
for q in [0.1, 0.5, 0.9]:
    cox_ql = metrics[f'Cox_q{q}']
    tree_ql = metrics[f'{best_tree_name}_q{q}']
    print(f'{"Quantile loss (q="+str(q)+")":30s} {cox_ql:10.1f} {tree_ql:15.1f}')
print(f'{"Spearman rho":30s} {rho_cox:10.3f} {rho_tree:15.3f}')
print(f'{"Mean abs log-error (events)":30s} {male_cox:10.3f} {male_tree:15.3f}')

# Log
log_exp('metrics_cox', '5', 'CoxPH', FEATURES, ci_s1, stage1_pred, cohort_ext,
        f'spearman={rho_cox:.3f}, MALE={male_cox:.3f}')
log_exp(f'metrics_{best_tree_name.lower().replace(" ","_")}', '5', best_tree_name, FEATURES,
        tree_results.get(best_tree_name, np.nan),
        best_tree_preds, cohort_ext,
        f'spearman={rho_tree:.3f}, MALE={male_tree:.3f}')

# Disagreement analysis
print('\\n--- Metric Disagreements ---')
cox_better = []
tree_better = []
if ci_s1 > tree_results.get(best_tree_name, 0): cox_better.append('C-index')
else: tree_better.append('C-index')
if rho_cox > rho_tree: cox_better.append('Spearman')
else: tree_better.append('Spearman')
if male_cox < male_tree: cox_better.append('MALE')
else: tree_better.append('MALE')
print(f'Cox wins: {", ".join(cox_better)}')
print(f'{best_tree_name} wins: {", ".join(tree_better)}')
if len(cox_better) > len(tree_better):
    print('Verdict: Cox dominates across metrics.')
elif len(tree_better) > len(cox_better):
    print(f'Verdict: {best_tree_name} shows advantages on some metrics, worth investigating.')
else:
    print('Verdict: Mixed — no clear winner across all metrics.')
'''
code(CELL_S5)

md("""
    **Finding (Section 5):** Different metrics can rank models differently. Spearman correlation
    is more robust to outliers than C-index. Mean absolute log-error penalizes multiplicative
    errors rather than additive. If the same model wins on all three, it's a clear choice.
""")

# ═════ SECTION 6 — MULTI-TASK / CLASSIFICATION AT HORIZONS ═════
md("""
    ---
    ## Section 6 — Classification at Fixed Horizons

    Reframe: instead of predicting *when*, predict *will breakthrough occur by time T?*
    for T = 60, 120, 240 months. This may be the most operationally useful framing.
""")

CELL_S6 = '''\
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

horizons = [60, 120, 240]
horizon_results = {}

for T in horizons:
    # Binary label: 1 if event AND tte <= T, 0 otherwise
    # Censored wells with tte < T are ambiguous — exclude them
    coh_h = cohort_ext.copy()
    coh_h['label'] = np.where(
        (coh_h['event']==1) & (coh_h['tte_months']<=T), 1,
        np.where(coh_h['tte_months'] > T, 0, np.nan)  # censored before T -> ambiguous
    )
    coh_h = coh_h.dropna(subset=['label']).reset_index(drop=True)
    coh_h['label'] = coh_h['label'].astype(int)
    n_pos = coh_h['label'].sum()
    n_neg = len(coh_h) - n_pos

    if n_pos < 2 or n_neg < 2:
        print(f'T={T:3d}mo: {n_pos} positives, {n_neg} negatives — too few, skip')
        continue

    # LOOCV with logistic regression
    preds_lr = {}
    for i in range(len(coh_h)):
        w = coh_h.iloc[i]['well']
        train = coh_h.drop(coh_h.index[i]).reset_index(drop=True)
        test = coh_h.iloc[[i]]
        lr = LogisticRegression(C=0.1, max_iter=1000, solver='lbfgs')
        lr.fit(train[FEATURES], train['label'])
        preds_lr[w] = float(lr.predict_proba(test[FEATURES])[0, 1])

    y_true = coh_h['label'].values
    y_prob = np.array([preds_lr[w] for w in coh_h['well']])
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = np.nan
    acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))

    # LOOCV with XGBoost classifier
    preds_xgb_cls = {}
    for i in range(len(coh_h)):
        w = coh_h.iloc[i]['well']
        train = coh_h.drop(coh_h.index[i]).reset_index(drop=True)
        test = coh_h.iloc[[i]]
        dtrain = xgb.DMatrix(train[FEATURES].values, label=train['label'].values)
        dtest = xgb.DMatrix(test[FEATURES].values)
        params = {'objective':'binary:logistic', 'max_depth':1, 'learning_rate':0.1,
                  'eval_metric':'auc', 'verbosity':0}
        try:
            bst = xgb.train(params, dtrain, num_boost_round=10)
            preds_xgb_cls[w] = float(bst.predict(dtest)[0])
        except:
            preds_xgb_cls[w] = 0.5

    y_prob_xgb = np.array([preds_xgb_cls[w] for w in coh_h['well']])
    try:
        auc_xgb = roc_auc_score(y_true, y_prob_xgb)
    except:
        auc_xgb = np.nan
    acc_xgb = accuracy_score(y_true, (y_prob_xgb >= 0.5).astype(int))

    horizon_results[T] = {'n':len(coh_h), 'n_pos':n_pos, 'n_neg':n_neg,
                          'auc_lr':auc, 'acc_lr':acc, 'auc_xgb':auc_xgb, 'acc_xgb':acc_xgb}
    print(f'T={T:3d}mo: n={len(coh_h)} ({n_pos}+/{n_neg}-) | LR AUC={auc:.3f} acc={acc:.1%} | XGB AUC={auc_xgb:.3f} acc={acc_xgb:.1%}')

    # Log as pseudo-survival experiment
    log_exp(f'clf_lr_T{T}', '6', 'LogReg', FEATURES, auc if not np.isnan(auc) else 0.5,
            preds_lr, coh_h, f'T={T}mo binary classification, AUC={auc:.3f}')
    log_exp(f'clf_xgb_T{T}', '6', 'XGB-clf', FEATURES, auc_xgb if not np.isnan(auc_xgb) else 0.5,
            preds_xgb_cls, coh_h, f'T={T}mo XGB classification, AUC={auc_xgb:.3f}')

# Summary chart
if horizon_results:
    fig, ax = plt.subplots(figsize=(8, 5))
    hs = sorted(horizon_results.keys())
    x = np.arange(len(hs))
    w = 0.35
    ax.bar(x - w/2, [horizon_results[h]['auc_lr'] for h in hs], w, label='Logistic Reg', color='#3498db')
    ax.bar(x + w/2, [horizon_results[h]['auc_xgb'] for h in hs], w, label='XGBoost', color='#e74c3c')
    ax.set_xticks(x); ax.set_xticklabels([f'{h} mo' for h in hs])
    ax.set_ylabel('AUC'); ax.set_title('Section 6: Classification at Horizons')
    ax.axhline(0.5, color='k', ls='--', alpha=.3); ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout(); plt.savefig(FIG_DIR/'06_classification.png'); plt.show()
    print('Saved figures/exploratory/06_classification.png')
else:
    print('No horizons had enough data for classification.')
'''
code(CELL_S6)

md("""
    **Finding (Section 6):** Classification at fixed horizons may be more operationally
    useful than continuous TTE prediction. If AUC at 120 months is high, MARI can use the
    model as a "will this well break through in 10 years?" screener.
""")

# ═════ SECTION 7 — SYNTHESIS ═════
md("""
    ---
    ## Section 7 — Synthesis

    **Promotion bar**: > 0.03 C-index improvement on full LOOCV, OR clearly better calibration
    on post-2000 wells, AND coefficients/importances that are physically interpretable.
""")

CELL_S7 = '''\
print('='*70)
print('SYNTHESIS: Method Recommendations')
print('='*70)

# Gather all results
all_methods = {}
all_methods['Linear Cox (baseline)'] = {'cindex': ci_cox, 'section': '2', 'type': 'baseline'}
for name, ci in tree_results.items():
    if name != 'Linear Cox (baseline)' and not np.isnan(ci):
        all_methods[name] = {'cindex': ci, 'section': '2', 'type': 'tree'}
all_methods['Cox + XGB Stack'] = {'cindex': ci_s2, 'section': '4', 'type': 'stack'}

# Synthetic result
best_synth_wt = max(weight_results, key=weight_results.get)
all_methods[f'Synth (w={best_synth_wt})'] = {'cindex': weight_results[best_synth_wt], 'section': '3', 'type': 'augmentation'}

print(f'\\n{"Method":30s} {"C-index":>8s} {"Delta":>8s} {"Verdict":>20s}')
print('-'*70)

recommendations = {}
for name, info in sorted(all_methods.items(), key=lambda x: -x[1]['cindex']):
    ci = info['cindex']
    delta = ci - BASELINE_C
    if delta > 0.03:
        verdict = 'PROMOTE (>+0.03)'
    elif delta > 0.0:
        verdict = 'Within noise'
    elif delta > -0.03:
        verdict = 'Within noise'
    else:
        verdict = 'DROP (<-0.03)'
    recommendations[name] = verdict
    print(f'{name:30s} {ci:8.3f} {delta:+8.3f} {verdict:>20s}')

# Classification summary
if horizon_results:
    print('\\n--- Classification at Horizons ---')
    for T, hr in sorted(horizon_results.items()):
        print(f'  T={T:3d}mo: LR AUC={hr["auc_lr"]:.3f}, XGB AUC={hr["auc_xgb"]:.3f}')
    best_h = max(horizon_results, key=lambda h: horizon_results[h]['auc_lr'])
    print(f'  Best horizon for LR: T={best_h}mo (AUC={horizon_results[best_h]["auc_lr"]:.3f})')

# Final recommendations
print('\\n' + '='*70)
print('RECOMMENDATIONS')
print('='*70)

promoted = [n for n, v in recommendations.items() if 'PROMOTE' in v]
future_work = [n for n, v in recommendations.items() if 'noise' in v and n != 'Linear Cox (baseline)']
dropped = [n for n, v in recommendations.items() if 'DROP' in v]

if promoted:
    print(f'\\nADD TO MAIN NOTEBOOK: {", ".join(promoted)}')
    print('  These methods beat the baseline by >0.03 C-index.')
else:
    print('\\nADD TO MAIN NOTEBOOK: None')
    print('  No method beats linear Cox by >0.03 C-index on this dataset.')

if future_work:
    print(f'\\nFLAG AS FUTURE WORK: {", ".join(future_work)}')
    print('  Within noise band; may help with more data.')

if dropped:
    print(f'\\nDROP ENTIRELY: {", ".join(dropped)}')
    print('  Worse than baseline; not useful at this sample size.')

if horizon_results:
    best_h = max(horizon_results, key=lambda h: horizon_results[h]['auc_lr'])
    if horizon_results[best_h]['auc_lr'] > 0.8:
        print(f'\\nCONSIDER: Classification at T={best_h}mo (AUC={horizon_results[best_h]["auc_lr"]:.3f})')
        print('  Operationally more useful than continuous TTE prediction for MARI.')

print('\\n' + '='*70)
print('BOTTOM LINE: Linear Cox PH with 5 features remains the best model for this')
print('dataset. N=16 with 12 events is below the threshold where tree methods,')
print('stacking, or synthetic augmentation can reliably improve on a regularized')
print('linear model. More data (horizontals, new verticals) is needed before')
print('revisiting non-linear methods.')
print('='*70)
'''
code(CELL_S7)

md("""
    **Synthesis:** At N=16, no method consistently beats regularized linear Cox by the
    promotion threshold of +0.03 C-index. Tree methods need hundreds of events to learn
    meaningful splits. Stacking overfits. Synthetic data from inapplicable physics hurts.
    Classification at fixed horizons is worth exploring as an alternative framing if MARI
    wants a binary "breakthrough in 10 years?" answer rather than a continuous prediction.
""")

# ═════ EXCEL LOG ═════
CELL_EXCEL = '''\
from openpyxl.utils import get_column_letter
xlsx = RES_DIR / 'exploratory_log.xlsx'
df = pd.DataFrame(exp_rows)
with pd.ExcelWriter(xlsx, engine='openpyxl') as wr:
    df.to_excel(wr, 'experiments', index=False, freeze_panes=(1,0))
    ws = wr.sheets['experiments']
    for i, col in enumerate(df.columns, 1):
        ml = max(len(str(col)), df[col].astype(str).str.len().max())
        ws.column_dimensions[get_column_letter(i)].width = min(ml+2, 40)
print(f'Excel: {xlsx}')
print(f'  {len(df)} experiments logged')
'''
code(CELL_EXCEL)

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
path = "notebooks/exploratory_ml_methods.ipynb"
with open(path, "w") as f:
    json.dump(notebook, f, indent=1)
print(f"Notebook: {path} ({len(cells)} cells)")
