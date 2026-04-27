#!/usr/bin/env python3
"""
Generate label_analysis/notebooks/label_sensitivity.ipynb

Sweeps WGR thresholds (1,2,3,4,5 bbl/MMcf) × sustained durations (1,2,3,4,6 months)
= 25 label definitions. For each, runs all 7 individual models + 6 ensemble strategies
under LOOCV. Saves comprehensive results to label_analysis/results/.
"""
import json, os, textwrap

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(src).strip()})

def code(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": textwrap.dedent(src).strip()})

# ═══════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════
md("""
    # Label Sensitivity Analysis — WGR Threshold & Sustained Duration Sweep

    **Objective**: Test whether alternative breakthrough definitions improve model
    performance beyond the current standard (WGR > 5 bbl/MMcf, 3 consecutive months).

    **Sweep grid**: 5 WGR thresholds × 5 sustained durations = 25 definitions

    | Parameter | Values |
    |---|---|
    | WGR threshold (bbl/MMcf) | 1, 2, 3, 4, 5 |
    | Sustained months | 1, 2, 3, 4, 6 |

    **Models tested per definition**: 7 individual + 6 ensemble strategies = 13 methods

    **Validation**: Leave-One-Out Cross-Validation (LOOCV) throughout
""")

# ═══════════════════════════════════════════════════════════════════════
# SECTION 1 — SETUP
# ═══════════════════════════════════════════════════════════════════════
md("---\n## Section 1 — Setup and Data Loading")

code('''\
import sys, os
# Navigate to mari_poc root
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('../..')
elif os.path.basename(os.getcwd()) == 'label_analysis':
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
import matplotlib.colors as mcolors
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
FIG_DIR  = Path('label_analysis/figures')
RES_DIR  = Path('label_analysis/results')
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

GWC_RKB_M = 754.0
CAP = 1200

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
})
print(f'Working directory: {os.getcwd()}')
''')

# ═══════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA & FEATURE BUILDING
# ═══════════════════════════════════════════════════════════════════════
md("---\n## Section 2 — Data Loading and Feature Engineering")

code('''\
panel  = pd.read_csv(DATA_DIR / 'panel_long.csv', parse_dates=['date'])
static = pd.read_csv(DATA_DIR / 'well_static.csv',
                      parse_dates=['first_prod_date', 'last_prod_date'])

EXCLUDED = {'M-51-HRL'}
vert_wells = static.loc[~static['is_horizontal'] &
                         ~static['well'].isin(EXCLUDED), 'well'].tolist()
panel_v  = panel[panel['well'].isin(vert_wells)].copy()
static_v = static[static['well'].isin(vert_wells)].copy().reset_index(drop=True)

def detect_breakthrough(pdf, threshold, sustained):
    """Detect breakthrough with configurable WGR threshold and sustained duration."""
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

# Build static features once (these don't change with label definition)
bt_base = detect_breakthrough(panel_v, threshold=5.0, sustained=3)
cohort_base = static_v.merge(bt_base, on='well')
cohort_base = build_features(cohort_base, panel_v)

# Features are the same regardless of label — only tte_months and event change
FEATURES = ['n_wells_producing_at_spud', 'spud_year', 'field_cum_gas_at_spud',
            'gas_rate_cv_yr12', 'peak_gas_rate']

print(f"Base cohort: {len(cohort_base)} wells")
print(f"Features: {FEATURES}")
''')

# ═══════════════════════════════════════════════════════════════════════
# SECTION 3 — LABEL DEFINITIONS SWEEP
# ═══════════════════════════════════════════════════════════════════════
md("""
    ---
    ## Section 3 — Label Definitions: Event Counts Across the Grid

    First, let's see how many events (confirmed breakthroughs) each definition produces.
    More events = more statistical power but potentially noisier labels.
    Fewer events = cleaner signal but higher EPV constraint risk.
""")

code('''\
WGR_THRESHOLDS = [1, 2, 3, 4, 5]
SUSTAINED_MONTHS = [1, 2, 3, 4, 6]

# Build label grid: for each (threshold, sustained), detect BT and count events
label_grid = {}  # (threshold, sustained) -> bt_df
event_counts = pd.DataFrame(index=WGR_THRESHOLDS, columns=SUSTAINED_MONTHS, dtype=int)

for thr in WGR_THRESHOLDS:
    for sus in SUSTAINED_MONTHS:
        bt = detect_breakthrough(panel_v, threshold=thr, sustained=sus)
        label_grid[(thr, sus)] = bt
        event_counts.loc[thr, sus] = int(bt['event'].sum())

event_counts.index.name = 'WGR threshold'
event_counts.columns.name = 'Sustained months'

print("Event counts (rows=WGR threshold, cols=sustained months):")
print(event_counts.to_string())
print(f"\\nBaseline (WGR>5, 3mo): {event_counts.loc[5, 3]} events")

# Heatmap
fig, ax = plt.subplots(figsize=(8, 5))
data = event_counts.values.astype(float)
im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=16)
ax.set_xticks(range(len(SUSTAINED_MONTHS)))
ax.set_xticklabels(SUSTAINED_MONTHS)
ax.set_yticks(range(len(WGR_THRESHOLDS)))
ax.set_yticklabels(WGR_THRESHOLDS)
ax.set_xlabel('Sustained Duration (months)')
ax.set_ylabel('WGR Threshold (bbl/MMcf)')
ax.set_title('Number of Breakthrough Events by Label Definition')
for i in range(len(WGR_THRESHOLDS)):
    for j in range(len(SUSTAINED_MONTHS)):
        val = int(data[i, j])
        color = 'white' if val > 10 else 'black'
        ax.text(j, i, str(val), ha='center', va='center', fontsize=12,
                fontweight='bold', color=color)
fig.colorbar(im, ax=ax, label='Number of events (out of 16 wells)')
plt.tight_layout()
plt.savefig(FIG_DIR / '01_event_count_grid.png')
print(f"\\nSaved {FIG_DIR / '01_event_count_grid.png'}")
plt.close()
''')

# ═══════════════════════════════════════════════════════════════════════
# SECTION 4 — MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════
md("---\n## Section 4 — Model Definitions and Helpers")

code('''\
MODEL_NAMES = ['Cox PH', 'Weibull AFT', 'XGB Cox', 'XGB AFT',
               'sksurv GBSA', 'sksurv CWGB', 'Stacked']
ENS_NAMES = ['Simple Mean', 'Trimmed Mean', 'Median', 'Geometric Mean',
             'Inv-MAE Weighted', 'Rank Fusion']

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

    # 7. Stacked
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
            tX['hr'] = np.log(mc.predict_partial_hazard(te[feats]).values.flatten())
            sX = test_X.copy()
            sX['hr'] = np.log(float(mc.predict_partial_hazard(test_X).values[0]))
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

def compute_ensembles_for_all_wells(model_preds, wells, events_map, actual_map):
    """Compute all ensemble strategies given per-well model predictions."""
    # model_preds: {model_name: {well: pred}}
    ens_preds = {en: {} for en in ENS_NAMES}

    # Compute inv-MAE weights from events
    evt_wells = [w for w in wells if events_map[w]]
    model_maes = {}
    for mn in MODEL_NAMES:
        errs = [abs(model_preds[mn][w] - actual_map[w]) for w in evt_wells]
        model_maes[mn] = np.mean(errs) if errs else 1.0
    inv_w = {mn: 1.0 / max(model_maes[mn], 1) for mn in MODEL_NAMES}
    tw = sum(inv_w.values())
    inv_w = {mn: v / tw for mn, v in inv_w.items()}

    for w in wells:
        vals = np.array([model_preds[mn][w] for mn in MODEL_NAMES])

        ens_preds['Simple Mean'][w] = float(np.mean(vals))

        sv = np.sort(vals)
        ens_preds['Trimmed Mean'][w] = float(np.mean(sv[1:-1]))

        ens_preds['Median'][w] = float(np.median(vals))

        ens_preds['Geometric Mean'][w] = float(gmean(np.clip(vals, 1, CAP)))

        ww = np.array([inv_w[mn] for mn in MODEL_NAMES])
        ens_preds['Inv-MAE Weighted'][w] = float(np.dot(ww, vals))

    # Rank Fusion
    sorted_actuals = sorted(actual_map.values())
    n = len(wells)
    model_ranks = {}
    for mn in MODEL_NAMES:
        sw = sorted(wells, key=lambda w: model_preds[mn][w])
        model_ranks[mn] = {w: r + 1 for r, w in enumerate(sw)}
    avg_ranks = {w: np.mean([model_ranks[mn][w] for mn in MODEL_NAMES]) for w in wells}
    rank_order = sorted(wells, key=lambda w: avg_ranks[w])
    for idx, w in enumerate(rank_order):
        frac = idx / max(n - 1, 1)
        ens_preds['Rank Fusion'][w] = float(
            sorted_actuals[0] + frac * (sorted_actuals[-1] - sorted_actuals[0]))

    return ens_preds

print("Model and ensemble functions defined.")
print(f"Individual models: {len(MODEL_NAMES)}")
print(f"Ensemble strategies: {len(ENS_NAMES)}")
print(f"Total methods per label: {len(MODEL_NAMES) + len(ENS_NAMES)}")
''')

# ═══════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN SWEEP
# ═══════════════════════════════════════════════════════════════════════
md("""
    ---
    ## Section 5 — Full Sweep: All Label Definitions × All Models

    For each of the 25 label definitions:
    1. Re-label wells (detect BT with new threshold/sustained)
    2. Skip if < 4 events (EPV too low for any model)
    3. Run all 7 models under LOOCV
    4. Compute 6 ensemble strategies
    5. Record C-index, MAE, median error for each method
""")

code('''\
ALL_METHODS = MODEL_NAMES + ENS_NAMES
MIN_EVENTS = 4  # minimum events to run models

# Master results storage
results = []  # list of dicts

total_combos = len(WGR_THRESHOLDS) * len(SUSTAINED_MONTHS)
combo_num = 0

for thr in WGR_THRESHOLDS:
    for sus in SUSTAINED_MONTHS:
        combo_num += 1
        bt = label_grid[(thr, sus)]
        n_events = int(bt['event'].sum())

        if n_events < MIN_EVENTS:
            print(f"[{combo_num}/{total_combos}] WGR>{thr}, {sus}mo: "
                  f"{n_events} events — SKIPPED (< {MIN_EVENTS})")
            for method in ALL_METHODS:
                results.append({
                    'wgr_threshold': thr, 'sustained_months': sus,
                    'n_events': n_events, 'method': method,
                    'c_index': np.nan, 'mae_events': np.nan,
                    'median_error': np.nan, 'max_error': np.nan,
                    'skipped': True
                })
            continue

        # Build cohort with new labels (features stay the same)
        cohort = cohort_base.copy()
        # Replace tte_months and event from new label
        bt_map = bt.set_index('well')
        for idx, row in cohort.iterrows():
            w = row['well']
            cohort.at[idx, 'tte_months'] = bt_map.loc[w, 'tte_months']
            cohort.at[idx, 'event'] = bt_map.loc[w, 'event']

        wells = cohort['well'].tolist()
        actual_map = {w: float(cohort.loc[cohort['well'] == w, 'tte_months'].values[0])
                      for w in wells}
        events_map = {w: int(cohort.loc[cohort['well'] == w, 'event'].values[0])
                      for w in wells}

        # LOOCV for all individual models
        model_preds = {mn: {} for mn in MODEL_NAMES}
        for i in range(len(cohort)):
            w = cohort.iloc[i]['well']
            train = cohort.drop(cohort.index[i]).reset_index(drop=True)
            test_X = cohort.iloc[[i]][FEATURES]
            fp = run_all_models(train, test_X, FEATURES)
            for mn in MODEL_NAMES:
                model_preds[mn][w] = fp[mn]

        # Compute ensembles
        ens_preds = compute_ensembles_for_all_wells(
            model_preds, wells, events_map, actual_map)

        # Merge all predictions
        all_preds = {}
        all_preds.update(model_preds)
        all_preds.update(ens_preds)

        # Compute metrics
        evt_wells = [w for w in wells if events_map[w]]
        act_arr = np.array([actual_map[w] for w in wells])
        evt_arr = np.array([events_map[w] for w in wells])

        for method in ALL_METHODS:
            prd_arr = np.array([all_preds[method][w] for w in wells])
            ci = harrell_cindex(act_arr, prd_arr, evt_arr)
            errs = [abs(all_preds[method][w] - actual_map[w]) for w in evt_wells]
            mae = np.mean(errs) if errs else np.nan
            med_err = np.median(errs) if errs else np.nan
            max_err = np.max(errs) if errs else np.nan
            results.append({
                'wgr_threshold': thr, 'sustained_months': sus,
                'n_events': n_events, 'method': method,
                'c_index': round(ci, 3), 'mae_events': round(mae, 1),
                'median_error': round(med_err, 1), 'max_error': round(max_err, 1),
                'skipped': False
            })

        print(f"[{combo_num}/{total_combos}] WGR>{thr}, {sus}mo: "
              f"{n_events} events — Best C={max(r['c_index'] for r in results[-13:]):.3f}, "
              f"Best MAE={min(r['mae_events'] for r in results[-13:]):.0f}")

results_df = pd.DataFrame(results)
print(f"\\nTotal experiment rows: {len(results_df)}")
print(f"Skipped combos: {results_df['skipped'].sum() // len(ALL_METHODS)}")
''')

# ═══════════════════════════════════════════════════════════════════════
# SECTION 6 — C-INDEX HEATMAPS
# ═══════════════════════════════════════════════════════════════════════
md("""
    ---
    ## Section 6 — C-index Heatmaps by Model

    One heatmap per model showing C-index across all (WGR threshold × sustained duration) combinations.
""")

code('''\
active = results_df[~results_df['skipped']].copy()

# Select key methods for visualization
key_methods = ['Cox PH', 'sksurv GBSA', 'XGB AFT', 'Stacked',
               'Rank Fusion', 'Inv-MAE Weighted']

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for ax_idx, method in enumerate(key_methods):
    ax = axes[ax_idx]
    sub = active[active['method'] == method]
    pivot = sub.pivot_table(index='wgr_threshold', columns='sustained_months',
                            values='c_index')
    # Reindex to ensure all values present
    pivot = pivot.reindex(index=WGR_THRESHOLDS, columns=SUSTAINED_MONTHS)
    data = pivot.values

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0.4, vmax=1.0)
    ax.set_xticks(range(len(SUSTAINED_MONTHS)))
    ax.set_xticklabels(SUSTAINED_MONTHS)
    ax.set_yticks(range(len(WGR_THRESHOLDS)))
    ax.set_yticklabels(WGR_THRESHOLDS)
    ax.set_xlabel('Sustained (months)')
    ax.set_ylabel('WGR threshold')
    ax.set_title(f'{method}')

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                ax.text(j, i, 'N/A', ha='center', va='center', fontsize=8, color='gray')
            else:
                color = 'white' if val < 0.6 or val > 0.85 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=9, fontweight='bold', color=color)

fig.colorbar(im, ax=axes, shrink=0.6, label='C-index (LOOCV)')
fig.suptitle('C-index by Label Definition — Key Models', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / '02_cindex_heatmaps.png')
print(f"Saved {FIG_DIR / '02_cindex_heatmaps.png'}")
plt.close()
''')

# ═══════════════════════════════════════════════════════════════════════
# SECTION 7 — MAE HEATMAPS
# ═══════════════════════════════════════════════════════════════════════
md("""
    ---
    ## Section 7 — MAE Heatmaps by Model
""")

code('''\
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for ax_idx, method in enumerate(key_methods):
    ax = axes[ax_idx]
    sub = active[active['method'] == method]
    pivot = sub.pivot_table(index='wgr_threshold', columns='sustained_months',
                            values='mae_events')
    pivot = pivot.reindex(index=WGR_THRESHOLDS, columns=SUSTAINED_MONTHS)
    data = pivot.values

    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=50, vmax=250)
    ax.set_xticks(range(len(SUSTAINED_MONTHS)))
    ax.set_xticklabels(SUSTAINED_MONTHS)
    ax.set_yticks(range(len(WGR_THRESHOLDS)))
    ax.set_yticklabels(WGR_THRESHOLDS)
    ax.set_xlabel('Sustained (months)')
    ax.set_ylabel('WGR threshold')
    ax.set_title(f'{method}')

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                ax.text(j, i, 'N/A', ha='center', va='center', fontsize=8, color='gray')
            else:
                color = 'white' if val > 200 or val < 70 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                        fontsize=9, fontweight='bold', color=color)

fig.colorbar(im, ax=axes, shrink=0.6, label='MAE (months, events only)')
fig.suptitle('MAE by Label Definition — Key Models', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / '03_mae_heatmaps.png')
print(f"Saved {FIG_DIR / '03_mae_heatmaps.png'}")
plt.close()
''')

# ═══════════════════════════════════════════════════════════════════════
# SECTION 8 — BEST LABEL PER MODEL
# ═══════════════════════════════════════════════════════════════════════
md("""
    ---
    ## Section 8 — Best Label Definition per Model

    For each model/ensemble, which label definition gives the best C-index? Best MAE?
""")

code('''\
active = results_df[~results_df['skipped']].copy()

print("=" * 90)
print("BEST LABEL DEFINITION PER METHOD — by C-index")
print("=" * 90)
print(f"{'Method':20s} {'WGR':>5s} {'Sust':>5s} {'Events':>7s} {'C-index':>8s} "
      f"{'MAE':>8s} {'MedErr':>8s}")
print("-" * 90)

best_by_c = []
for method in ALL_METHODS:
    sub = active[active['method'] == method]
    if len(sub) == 0: continue
    best = sub.loc[sub['c_index'].idxmax()]
    best_by_c.append(best)
    print(f"{method:20s} {best['wgr_threshold']:5.0f} {best['sustained_months']:5.0f} "
          f"{best['n_events']:7.0f} {best['c_index']:8.3f} "
          f"{best['mae_events']:8.1f} {best['median_error']:8.1f}")

print()
print("=" * 90)
print("BEST LABEL DEFINITION PER METHOD — by MAE")
print("=" * 90)
print(f"{'Method':20s} {'WGR':>5s} {'Sust':>5s} {'Events':>7s} {'C-index':>8s} "
      f"{'MAE':>8s} {'MedErr':>8s}")
print("-" * 90)

best_by_mae = []
for method in ALL_METHODS:
    sub = active[active['method'] == method]
    if len(sub) == 0: continue
    best = sub.loc[sub['mae_events'].idxmin()]
    best_by_mae.append(best)
    print(f"{method:20s} {best['wgr_threshold']:5.0f} {best['sustained_months']:5.0f} "
          f"{best['n_events']:7.0f} {best['c_index']:8.3f} "
          f"{best['mae_events']:8.1f} {best['median_error']:8.1f}")
''')

# ═══════════════════════════════════════════════════════════════════════
# SECTION 9 — COMPARISON WITH BASELINE
# ═══════════════════════════════════════════════════════════════════════
md("""
    ---
    ## Section 9 — Comparison: Best Labels vs Baseline (WGR>5, 3mo)

    How much does the best label definition improve over our current standard?
""")

code('''\
# Baseline: WGR>5, sustained=3
baseline = active[(active['wgr_threshold'] == 5) & (active['sustained_months'] == 3)]
baseline_dict = {row['method']: row for _, row in baseline.iterrows()}

print("=" * 100)
print("IMPROVEMENT OVER BASELINE (WGR>5, 3mo)")
print("=" * 100)
print(f"{'Method':20s} | {'Baseline C':>10s} {'Best C':>8s} {'Delta C':>8s} | "
      f"{'Base MAE':>9s} {'Best MAE':>9s} {'Delta MAE':>10s} | {'Best Label (C)':>15s} {'Best Label (MAE)':>18s}")
print("-" * 100)

improvement_rows = []
for method in ALL_METHODS:
    sub = active[active['method'] == method]
    if len(sub) == 0 or method not in baseline_dict: continue
    bl = baseline_dict[method]

    best_c_row = sub.loc[sub['c_index'].idxmax()]
    best_mae_row = sub.loc[sub['mae_events'].idxmin()]

    dc = best_c_row['c_index'] - bl['c_index']
    dm = best_mae_row['mae_events'] - bl['mae_events']

    label_c = f"WGR>{int(best_c_row['wgr_threshold'])},{int(best_c_row['sustained_months'])}mo"
    label_mae = f"WGR>{int(best_mae_row['wgr_threshold'])},{int(best_mae_row['sustained_months'])}mo"

    print(f"{method:20s} | {bl['c_index']:10.3f} {best_c_row['c_index']:8.3f} "
          f"{dc:+8.3f} | {bl['mae_events']:9.1f} {best_mae_row['mae_events']:9.1f} "
          f"{dm:+10.1f} | {label_c:>15s} {label_mae:>18s}")

    improvement_rows.append({
        'Method': method,
        'Baseline C-index': bl['c_index'],
        'Best C-index': best_c_row['c_index'],
        'Delta C': dc,
        'Best C Label': label_c,
        'Baseline MAE': bl['mae_events'],
        'Best MAE': best_mae_row['mae_events'],
        'Delta MAE': dm,
        'Best MAE Label': label_mae,
    })

improvement_df = pd.DataFrame(improvement_rows)
''')

# ═══════════════════════════════════════════════════════════════════════
# SECTION 10 — PER-WELL ANALYSIS FOR TOP LABELS
# ═══════════════════════════════════════════════════════════════════════
md("""
    ---
    ## Section 10 — Per-Well Predictions: Top 3 Label Definitions

    Show per-well errors for the top 3 label definitions (by overall best C-index across
    all methods) compared to the baseline.
""")

code('''\
# Find top 3 label definitions by max C-index achieved by any method
label_best = active.groupby(['wgr_threshold', 'sustained_months'])['c_index'].max().reset_index()
label_best = label_best.sort_values('c_index', ascending=False).head(3)

top_labels = [(int(row['wgr_threshold']), int(row['sustained_months']))
              for _, row in label_best.iterrows()]

# Always include baseline if not in top 3
baseline_label = (5, 3)
if baseline_label not in top_labels:
    top_labels.append(baseline_label)

print("Top label definitions to compare:")
for thr, sus in top_labels:
    sub = active[(active['wgr_threshold'] == thr) & (active['sustained_months'] == sus)]
    best_c = sub['c_index'].max()
    best_mae = sub['mae_events'].min()
    n_evt = int(sub['n_events'].iloc[0])
    tag = " (BASELINE)" if (thr, sus) == baseline_label else ""
    print(f"  WGR>{thr}, {sus}mo: {n_evt} events, best C={best_c:.3f}, best MAE={best_mae:.0f}{tag}")

# For the best overall label, show per-well Cox PH and Rank Fusion predictions
print()
best_thr, best_sus = top_labels[0]
print(f"\\nPer-well detail for BEST label: WGR>{best_thr}, {best_sus}mo")
print("Using Cox PH and Rank Fusion:\\n")

bt_best = label_grid[(best_thr, best_sus)]
cohort_best = cohort_base.copy()
bt_m = bt_best.set_index('well')
for idx, row in cohort_best.iterrows():
    w = row['well']
    cohort_best.at[idx, 'tte_months'] = bt_m.loc[w, 'tte_months']
    cohort_best.at[idx, 'event'] = bt_m.loc[w, 'event']

# Re-run LOOCV for best label
mp = {mn: {} for mn in MODEL_NAMES}
for i in range(len(cohort_best)):
    w = cohort_best.iloc[i]['well']
    train = cohort_best.drop(cohort_best.index[i]).reset_index(drop=True)
    test_X = cohort_best.iloc[[i]][FEATURES]
    fp = run_all_models(train, test_X, FEATURES)
    for mn in MODEL_NAMES:
        mp[mn][w] = fp[mn]

wells = cohort_best.sort_values('tte_months')['well'].tolist()
actual_map = {w: float(cohort_best.loc[cohort_best['well'] == w, 'tte_months'].values[0])
              for w in wells}
events_map = {w: int(cohort_best.loc[cohort_best['well'] == w, 'event'].values[0])
              for w in wells}
ep = compute_ensembles_for_all_wells(mp, wells, events_map, actual_map)

print(f"{'Well':12s} {'Evt':>4s} {'Actual':>7s} | {'Cox PH':>10s} {'Rank Fusion':>12s} | "
      f"{'Cox Err':>8s} {'RF Err':>8s}")
print("-" * 75)
for w in wells:
    a = actual_map[w]
    cp = mp['Cox PH'][w]
    rf = ep['Rank Fusion'][w]
    e = 'Y' if events_map[w] else 'N'
    print(f"{w:12s} {e:>4s} {a:7.0f} | {cp:10.0f} {rf:12.0f} | "
          f"{cp-a:+8.0f} {rf-a:+8.0f}")
''')

# ═══════════════════════════════════════════════════════════════════════
# SECTION 11 — SUMMARY FIGURE
# ═══════════════════════════════════════════════════════════════════════
md("---\n## Section 11 — Summary Visualization")

code('''\
# Scatter: C-index vs MAE for ALL label × method combinations
fig, ax = plt.subplots(figsize=(12, 8))

# Color by method type
colors_indiv = plt.cm.Set2(np.linspace(0, 1, len(MODEL_NAMES)))
colors_ens = plt.cm.Dark2(np.linspace(0, 1, len(ENS_NAMES)))

method_colors = {}
for i, mn in enumerate(MODEL_NAMES):
    method_colors[mn] = colors_indiv[i]
for i, en in enumerate(ENS_NAMES):
    method_colors[en] = colors_ens[i]

for method in ALL_METHODS:
    sub = active[active['method'] == method]
    if len(sub) == 0: continue
    c = method_colors[method]
    marker = 'o' if method in MODEL_NAMES else 's'
    ax.scatter(sub['mae_events'], sub['c_index'], c=[c], marker=marker,
               s=50, alpha=0.6, label=method, edgecolors='k', linewidth=0.3)

# Highlight baseline (WGR>5, 3mo) for Cox PH
bl = active[(active['wgr_threshold'] == 5) & (active['sustained_months'] == 3) &
            (active['method'] == 'Cox PH')]
if len(bl):
    ax.scatter(bl['mae_events'].values, bl['c_index'].values, c='red',
               marker='*', s=300, zorder=10, label='Baseline (Cox, WGR>5,3mo)')

ax.set_xlabel('MAE (months, events only)')
ax.set_ylabel('C-index (LOOCV)')
ax.set_title('All Label Definitions × All Models — C-index vs MAE')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='_random')
plt.tight_layout()
plt.savefig(FIG_DIR / '04_all_labels_scatter.png')
print(f"Saved {FIG_DIR / '04_all_labels_scatter.png'}")
plt.close()

# Bar chart: best C-index per method across all labels vs baseline
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: C-index
methods_ordered = ALL_METHODS
x = np.arange(len(methods_ordered))
base_c = [baseline_dict[m]['c_index'] if m in baseline_dict else np.nan for m in methods_ordered]
best_c = []
for m in methods_ordered:
    sub = active[active['method'] == m]
    best_c.append(sub['c_index'].max() if len(sub) else np.nan)

axes[0].bar(x - 0.2, base_c, 0.35, label='Baseline (WGR>5, 3mo)', color='steelblue')
axes[0].bar(x + 0.2, best_c, 0.35, label='Best label', color='coral')
axes[0].set_xticks(x)
axes[0].set_xticklabels(methods_ordered, rotation=45, ha='right', fontsize=8)
axes[0].set_ylabel('C-index')
axes[0].set_title('C-index: Baseline vs Best Label')
axes[0].legend()
axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

# Panel 2: MAE
base_mae = [baseline_dict[m]['mae_events'] if m in baseline_dict else np.nan for m in methods_ordered]
best_mae = []
for m in methods_ordered:
    sub = active[active['method'] == m]
    best_mae.append(sub['mae_events'].min() if len(sub) else np.nan)

axes[1].bar(x - 0.2, base_mae, 0.35, label='Baseline (WGR>5, 3mo)', color='steelblue')
axes[1].bar(x + 0.2, best_mae, 0.35, label='Best label', color='coral')
axes[1].set_xticks(x)
axes[1].set_xticklabels(methods_ordered, rotation=45, ha='right', fontsize=8)
axes[1].set_ylabel('MAE (months)')
axes[1].set_title('MAE: Baseline vs Best Label')
axes[1].legend()

plt.tight_layout()
plt.savefig(FIG_DIR / '05_baseline_vs_best.png')
print(f"Saved {FIG_DIR / '05_baseline_vs_best.png'}")
plt.close()
''')

# ═══════════════════════════════════════════════════════════════════════
# SECTION 12 — SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════
md("---\n## Section 12 — Save All Results")

code('''\
# Save master results to Excel
outpath = RES_DIR / 'label_sensitivity_results.xlsx'
with pd.ExcelWriter(outpath, engine='openpyxl') as writer:

    # Sheet 1: Full results grid
    results_df.to_excel(writer, sheet_name='All_Results', index=False)

    # Sheet 2: Event counts
    event_counts_out = event_counts.reset_index()
    event_counts_out.to_excel(writer, sheet_name='Event_Counts', index=False)

    # Sheet 3: Best label per method (by C-index)
    rows = []
    for method in ALL_METHODS:
        sub = active[active['method'] == method]
        if len(sub) == 0: continue
        best_c = sub.loc[sub['c_index'].idxmax()]
        best_m = sub.loc[sub['mae_events'].idxmin()]
        rows.append({
            'Method': method,
            'Best_C_WGR': int(best_c['wgr_threshold']),
            'Best_C_Sustained': int(best_c['sustained_months']),
            'Best_C_Events': int(best_c['n_events']),
            'Best_C_Cindex': best_c['c_index'],
            'Best_C_MAE': best_c['mae_events'],
            'Best_MAE_WGR': int(best_m['wgr_threshold']),
            'Best_MAE_Sustained': int(best_m['sustained_months']),
            'Best_MAE_Events': int(best_m['n_events']),
            'Best_MAE_Cindex': best_m['c_index'],
            'Best_MAE_MAE': best_m['mae_events'],
        })
    pd.DataFrame(rows).to_excel(writer, sheet_name='Best_Per_Method', index=False)

    # Sheet 4: Improvement over baseline
    if len(improvement_rows) > 0:
        improvement_df.to_excel(writer, sheet_name='Improvement_vs_Baseline', index=False)

    # Sheet 5: Pivot — C-index for each method × label (wide format)
    for method in ['Cox PH', 'Rank Fusion', 'sksurv GBSA']:
        sub = active[active['method'] == method]
        pivot = sub.pivot_table(index='wgr_threshold', columns='sustained_months',
                                values='c_index')
        pivot = pivot.reindex(index=WGR_THRESHOLDS, columns=SUSTAINED_MONTHS)
        pivot.to_excel(writer, sheet_name=f'Cindex_{method.replace(" ","_")}')

    # Sheet 6: Pivot — MAE for each method × label
    for method in ['Cox PH', 'Rank Fusion', 'sksurv GBSA']:
        sub = active[active['method'] == method]
        pivot = sub.pivot_table(index='wgr_threshold', columns='sustained_months',
                                values='mae_events')
        pivot = pivot.reindex(index=WGR_THRESHOLDS, columns=SUSTAINED_MONTHS)
        pivot.to_excel(writer, sheet_name=f'MAE_{method.replace(" ","_")}')

print(f"Saved: {outpath}")

# Also save as CSV for quick access
results_df.to_csv(RES_DIR / 'label_sensitivity_all.csv', index=False)
print(f"Saved: {RES_DIR / 'label_sensitivity_all.csv'}")
''')

# ═══════════════════════════════════════════════════════════════════════
# SECTION 13 — SUMMARY & RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════
md("---\n## Section 13 — Summary and Recommendations")

code('''\
print("=" * 80)
print("LABEL SENSITIVITY ANALYSIS — SUMMARY")
print("=" * 80)

# Overall best across all methods and labels
overall_best_c = active.loc[active['c_index'].idxmax()]
overall_best_mae = active.loc[active['mae_events'].idxmin()]

print(f"""
OVERALL BEST C-INDEX:
  Method: {overall_best_c['method']}
  Label:  WGR > {int(overall_best_c['wgr_threshold'])} bbl/MMcf, {int(overall_best_c['sustained_months'])} consecutive months
  Events: {int(overall_best_c['n_events'])}
  C-index: {overall_best_c['c_index']:.3f}
  MAE: {overall_best_c['mae_events']:.0f} months

OVERALL BEST MAE:
  Method: {overall_best_mae['method']}
  Label:  WGR > {int(overall_best_mae['wgr_threshold'])} bbl/MMcf, {int(overall_best_mae['sustained_months'])} consecutive months
  Events: {int(overall_best_mae['n_events'])}
  C-index: {overall_best_mae['c_index']:.3f}
  MAE: {overall_best_mae['mae_events']:.0f} months

BASELINE (WGR>5, 3mo):
  Cox PH C-index: {baseline_dict['Cox PH']['c_index']:.3f}
  Cox PH MAE: {baseline_dict['Cox PH']['mae_events']:.0f} months
  Rank Fusion MAE: {baseline_dict['Rank Fusion']['mae_events']:.0f} months
""")

# Count how many labels beat baseline for Cox PH
cox_base_c = baseline_dict['Cox PH']['c_index']
cox_base_mae = baseline_dict['Cox PH']['mae_events']
cox_results = active[active['method'] == 'Cox PH']
n_better_c = (cox_results['c_index'] > cox_base_c).sum()
n_better_mae = (cox_results['mae_events'] < cox_base_mae).sum()
n_total = len(cox_results)

print(f"Cox PH: {n_better_c}/{n_total} labels have better C-index than baseline")
print(f"Cox PH: {n_better_mae}/{n_total} labels have better MAE than baseline")

# Recommendation
print()
print("RECOMMENDATIONS:")
print("-" * 80)
print("1. The current baseline (WGR>5, 3mo) is a sensible default for gas wells.")
print("2. If any alternative label consistently improves both C-index AND MAE across")
print("   multiple models, it may be worth considering as a primary definition.")
print("3. Labels with very low WGR thresholds (1-2) may capture noise/transients.")
print("4. Labels with long sustained periods (6mo) are more conservative but may")
print("   miss legitimate early breakthroughs.")
print("5. Present alternative results as sensitivity analysis, not primary findings.")
''')

# ═══════════════════════════════════════════════════════════════════════
# WRITE NOTEBOOK
# ═══════════════════════════════════════════════════════════════════════
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                "name": "python3"},
                 "language_info": {"name": "python", "version": "3.11.0"}},
    "cells": cells,
}

os.makedirs('label_analysis/notebooks', exist_ok=True)
outpath = 'label_analysis/notebooks/label_sensitivity.ipynb'
with open(outpath, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"Notebook: {outpath} ({len(cells)} cells)")
