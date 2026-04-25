#!/usr/bin/env python3
"""Generate vertical_modeling_study.ipynb using explicit cell list."""
import json, os, textwrap

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(src).strip()})

def code(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": textwrap.dedent(src).strip()})

# Use single-line strings for all docstrings inside code cells to avoid
# triple-quote conflicts with the code() wrapper.

# ════════════════════════════════════════════════════════════════
md("""
    # Vertical-Only Modeling Study — MARI Water Breakthrough POC

    **Objective**: Demonstrate achievable prediction performance for water breakthrough timing
    using 16 vertical wells from the Habib Rahi Limestone gas field (12 observed events,
    4 censored).

    **Scope**: Verticals only. M-51-HRL excluded (completion flowback). Horizontals out of scope.

    **Key constraints**: 12 events. EPV discipline. LOOCV. L2 regularization (penalizer=0.1)
    for multivariate Cox. No RSF (overfits at this N).
""")

# Each code cell is a separate heredoc-style string passed to code()
# We build cells individually using code() calls

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
from lifelines import CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter, KaplanMeierFitter
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data/processed')
FIG_DIR  = Path('figures')
RES_DIR  = Path('results')
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

GWC_RKB_M       = 754.0
WGR_THRESHOLD    = 5.0
BT_SUSTAINED     = 3
VALIDATION_WELL  = 'M-50-HRL'
VALIDATION_TTE   = 379

C_EVENT  = '#DC143C'
C_CENSOR = '#4682B4'
C_EXCLUDE= '#808080'
C_REF    = '#FFD700'

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
})
print(f'Working directory: {os.getcwd()}')
print('Setup complete.')
'''
code(CELL_SETUP)

CELL_UTILS = '''\
def _safe_predict_median(model, X, cap=1200):
    raw = model.predict_median(X)
    arr = np.atleast_1d(np.array(raw, dtype=float))
    return np.where(np.isinf(arr) | np.isnan(arr) | (arr > cap), cap, arr)

def harrell_cindex(actual, predicted, events):
    actual, predicted, events = np.asarray(actual,float), np.asarray(predicted,float), np.asarray(events,int)
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

def loocv_survival(cohort, features, model_class, model_kwargs=None,
                   duration_col='tte_months', event_col='event'):
    kw = model_kwargs or {}
    cols = features + [duration_col, event_col]
    predictions = {}
    for i in range(len(cohort)):
        well = cohort.iloc[i]['well']
        train = cohort.drop(cohort.index[i])[cols]
        test_X = cohort.iloc[[i]][features]
        m = model_class(**kw)
        m.fit(train, duration_col=duration_col, event_col=event_col)
        predictions[well] = float(_safe_predict_median(m, test_X)[0])
    wells = list(predictions.keys())
    act = np.array([float(cohort.loc[cohort['well']==w, duration_col].values[0]) for w in wells])
    prd = np.array([predictions[w] for w in wells])
    evt = np.array([int(cohort.loc[cohort['well']==w, event_col].values[0]) for w in wells])
    return {'cindex': harrell_cindex(act, prd, evt), 'predictions': predictions,
            'wells': wells, 'actual': act, 'predicted': prd, 'events': evt}

def fit_and_predict(cohort, features, model_class, model_kwargs=None,
                    duration_col='tte_months', event_col='event'):
    kw = model_kwargs or {}
    m = model_class(**kw)
    m.fit(cohort[features + [duration_col, event_col]], duration_col=duration_col, event_col=event_col)
    preds = _safe_predict_median(m, cohort[features])
    ci = harrell_cindex(cohort[duration_col].values.astype(float), preds,
                        cohort[event_col].values.astype(int))
    return m, preds, ci

EXPECTED_SIGNS = {
    'porosity':'pos','log_perm':'pos','sw':'pos','net_pay_m':'neg','skin':'neg',
    'chlorides_ppm':'pos','gas_gravity':'unknown',
    'perf_midpoint_md':'pos','perf_thickness_md':'unknown',
    'dist_to_gwc_m':'neg','standoff_ratio':'neg','spud_year':'pos',
    'early_gas_rate':'pos','peak_gas_rate':'pos',
    'cum_gas_year1':'pos','cum_gas_year3':'pos',
    'initial_whfp':'neg','whfp_decline_rate':'pos','gas_rate_cv_yr12':'unknown',
    'field_cum_gas_at_spud':'pos','n_wells_producing_at_spud':'pos',
}

experiment_rows, per_well_rows, feature_imp_rows, summary_rows = [], [], [], []
_run_counter = [0]

def log_experiment(name, section, model_family, features, cohort_df,
                   cindex_insample, cindex_loocv, predictions,
                   penalizer=None, cindex_loocv_std=None, model=None, notes=''):
    _run_counter[0] += 1
    rid = _run_counter[0]
    wells = list(predictions.keys())
    actual = {w: float(cohort_df.loc[cohort_df['well']==w,'tte_months'].values[0]) for w in wells}
    events = {w: int(cohort_df.loc[cohort_df['well']==w,'event'].values[0]) for w in wells}
    errors = {w: abs(predictions[w]-actual[w]) for w in wells}
    m50p = predictions.get(VALIDATION_WELL, np.nan)
    m50e = abs(m50p - VALIDATION_TTE) if not np.isnan(m50p) else np.nan
    ww = max(errors, key=errors.get)
    experiment_rows.append({
        'run_id':rid,'timestamp':datetime.now().isoformat(),'section':str(section),
        'experiment_name':name,'model_family':model_family,
        'features':', '.join(features) if features else '','n_features':len(features),
        'n_train_wells':len(cohort_df),'n_events':int(cohort_df['event'].sum()),
        'penalizer':penalizer,
        'cindex_insample':round(cindex_insample,3) if cindex_insample is not None else None,
        'cindex_loocv':round(cindex_loocv,3),
        'cindex_loocv_std':round(cindex_loocv_std,3) if cindex_loocv_std is not None else None,
        'm50_predicted_months':round(m50p,1) if not np.isnan(m50p) else None,
        'm50_actual_months':VALIDATION_TTE,
        'm50_abs_error':round(m50e,1) if not np.isnan(m50e) else None,
        'worst_well':ww,'worst_well_abs_error':round(errors[ww],1),'notes':notes})
    for w in wells:
        per_well_rows.append({'run_id':rid,'well':w,'event':events[w],
            'actual_tte_months':round(actual[w],1),'predicted_tte_months':round(predictions[w],1),
            'abs_error_months':round(errors[w],1),'loocv':True})
    if model is not None and hasattr(model,'params_') and features:
        for fn in features:
            if fn in model.params_.index:
                c = model.params_[fn]; s = 'pos' if c>0 else 'neg'
                es = EXPECTED_SIGNS.get(fn,'unknown')
                feature_imp_rows.append({'run_id':rid,'feature':fn,
                    'coefficient':round(c,4),'hazard_ratio':round(np.exp(c),4),
                    'sign':s,'expected_sign_physics':es,
                    'sign_matches':s==es if es!='unknown' else None})
    return rid

def log_section_summary(sec, rid, ci, line):
    summary_rows.append({'section':str(sec),'best_run_id':rid,
                         'best_cindex_loocv':round(ci,3),'one_line_finding':line})

print('Utility functions defined.')
'''
code(CELL_UTILS)

# ═════ SECTION 1 ═════
md("""
    ---
    ## Section 1 — Setup and Cohort Definition
""")

CELL_S1_LOAD = '''\
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
            tte = (bd.year*12+bd.month)-(fp.year*12+fp.month)
        else:
            ld = grp['date'].iloc[-1]
            tte = (ld.year*12+ld.month)-(fp.year*12+fp.month)
            bd = pd.NaT
        rows.append({'well':well,'spud_date':fp,'bt_date':bd,'tte_months':max(tte,1),'event':int(bt_found)})
    return pd.DataFrame(rows)

bt_df = detect_breakthrough(panel_v)
cohort = static_v.merge(bt_df, on='well')
n_events = int(cohort['event'].sum())
n_censored = len(cohort) - n_events
event_ttes = cohort.loc[cohort['event']==1, 'tte_months']

print(f'Cohort: {len(cohort)} wells, {n_events} events, {n_censored} censored')
print(f'Event TTE: median={event_ttes.median():.0f}, min={event_ttes.min():.0f}, max={event_ttes.max():.0f} mo')

display(cohort[['well','spud_date','tte_months','event','porosity','permeability_md','sw','net_pay_m']]
        .sort_values('tte_months')
        .style.format({'tte_months':'{:.0f}','porosity':'{:.3f}','permeability_md':'{:.1f}',
                       'sw':'{:.2f}','net_pay_m':'{:.1f}'})
        .set_caption('Vertical cohort'))
'''
code(CELL_S1_LOAD)

CELL_S1_FIG = '''\
fig, ax = plt.subplots(figsize=(12, 7))
cs = cohort.sort_values('spud_date').reset_index(drop=True)
t0 = cs['spud_date'].min()
for i, row in cs.iterrows():
    c = C_EVENT if row['event'] else C_CENSOR
    start_m = (row['spud_date'] - t0).days / 30.44
    end_d = row['bt_date'] if row['event'] else row['last_prod_date']
    span_m = (end_d - row['spud_date']).days / 30.44
    ax.barh(i, span_m, left=start_m, height=0.7, color=c, alpha=0.8, edgecolor='white', linewidth=0.5)
    if row['event']:
        ax.plot(start_m + span_m, i, 'v', color='black', markersize=6, zorder=5)
    ax.text(start_m - 3, i, row['well'].replace('M-','').replace('-HRL',''),
            ha='right', va='center', fontsize=8, fontweight='bold')
ax.set_yticks([]); ax.set_xlabel('Months from earliest spud')
ax.set_title('Figure 1 — Vertical Well Cohort')
ax.legend(handles=[
    Line2D([0],[0],color=C_EVENT,lw=8,alpha=.8,label=f'Event (n={n_events})'),
    Line2D([0],[0],color=C_CENSOR,lw=8,alpha=.8,label=f'Censored (n={n_censored})'),
    Line2D([0],[0],marker='v',color='k',lw=0,ms=6,label='Breakthrough'),
], loc='lower right')
plt.tight_layout(); plt.savefig(FIG_DIR/'01_cohort.png'); plt.show()
print('Saved figures/01_cohort.png')
'''
code(CELL_S1_FIG)

md("""
    **Finding (Section 1):** 16 vertical wells, 12 events, 4 censored (M-13, M-22, M-57,
    M-E-2). Event TTE spans 45–409 months — a 10:1 range driven by field-state evolution
    (vintage) rather than static rock properties alone.
""")

# ═════ SECTION 2 ═════
md("---\n## Section 2 — Feature Construction")

CELL_S2 = '''\
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
    df['initial_whfp_fallback'] = False
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
            else:
                df.at[idx,'initial_whfp'] = wh['whfp_psig'].mean()
                df.at[idx,'initial_whfp_fallback'] = True
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
    return df

cohort = build_features(cohort, panel_v)

CANDIDATE_FEATURES = [
    'porosity','log_perm','sw','net_pay_m','skin','chlorides_ppm','gas_gravity',
    'perf_midpoint_md','perf_thickness_md','dist_to_gwc_m','standoff_ratio','spud_year',
    'early_gas_rate','peak_gas_rate','cum_gas_year1','cum_gas_year3',
    'initial_whfp','whfp_decline_rate','gas_rate_cv_yr12',
    'field_cum_gas_at_spud','n_wells_producing_at_spud',
]

summary = cohort[CANDIDATE_FEATURES].describe().T[['count','mean','std','min','max']]
summary['count'] = summary['count'].astype(int)
display(summary.style.format({'mean':'{:.2f}','std':'{:.2f}','min':'{:.2f}','max':'{:.2f}'})
        .set_caption('Feature summary'))
print(f"initial_whfp nulls: {cohort['initial_whfp'].isna().sum()}")
print(f"whfp_decline_rate nulls: {cohort['whfp_decline_rate'].isna().sum()}")
'''
code(CELL_S2)

md("""
    **Finding (Section 2):** 21 candidate features constructed. Static/geometry features are
    complete. initial_whfp and whfp_decline_rate are sparse. Field-state features are complete.
""")

# ═════ SECTION 3 ═════
md("---\n## Section 3 — Feature Screening")

CELL_S3 = '''\
screening = []
for feat in CANDIDATE_FEATURES:
    sub = cohort[['well',feat,'tte_months','event']].dropna().reset_index(drop=True)
    if len(sub)<10 or sub[feat].std()<1e-12:
        screening.append({'feature':feat,'n':len(sub),'cindex_loocv':np.nan,
            'coef_sign':'','expected_sign':EXPECTED_SIGNS.get(feat,'unknown'),
            'sign_match':'','note':f'Dropped ({len(sub)} wells or zero var)'})
        continue
    res = loocv_survival(sub, [feat], CoxPHFitter, {'penalizer':0.1})
    mdl, _, ci_in = fit_and_predict(sub, [feat], CoxPHFitter, {'penalizer':0.1})
    c = mdl.params_[feat]; cs = 'pos' if c>0 else 'neg'
    es = EXPECTED_SIGNS.get(feat,'unknown')
    sm = (cs==es) if es!='unknown' else 'N/A'
    screening.append({'feature':feat,'n':len(sub),'cindex_loocv':res['cindex'],
        'coef_sign':cs,'expected_sign':es,'sign_match':sm,'note':''})
    log_experiment(f'uni_{feat}','3','Cox',[feat],sub,ci_in,res['cindex'],
                   res['predictions'],0.1,model=mdl,notes=f'sign={cs},expected={es}')

screen_df = pd.DataFrame(screening).sort_values('cindex_loocv',ascending=False).reset_index(drop=True)
display(screen_df.style.format({'cindex_loocv':'{:.3f}'}).set_caption('Univariate screening'))
br = screen_df.iloc[0]
log_section_summary('3',_run_counter[0],br['cindex_loocv'],f'Best: {br["feature"]} (C={br["cindex_loocv"]:.3f})')
'''
code(CELL_S3)

CELL_S3_FIG = '''\
pf = screen_df.dropna(subset=['cindex_loocv']).copy()
fig, ax = plt.subplots(figsize=(10, 7))
colors = [C_EVENT if r['sign_match']==True else (C_EXCLUDE if r['sign_match']==False else C_CENSOR)
          for _, r in pf.iterrows()]
ax.barh(range(len(pf)), pf['cindex_loocv'], color=colors, edgecolor='white')
ax.set_yticks(range(len(pf))); ax.set_yticklabels(pf['feature'], fontsize=9)
ax.set_xlabel('LOOCV C-index'); ax.set_title('Figure 2 — Univariate C-index')
ax.axvline(0.5,color='k',ls='--',alpha=.5); ax.axvline(0.65,color='orange',ls='--',alpha=.7)
ax.invert_yaxis()
ax.legend(handles=[Line2D([0],[0],color=C_EVENT,lw=8,label='Sign correct'),
    Line2D([0],[0],color=C_EXCLUDE,lw=8,label='Sign WRONG'),
    Line2D([0],[0],color=C_CENSOR,lw=8,label='Unknown')], loc='lower right', fontsize=8)
plt.tight_layout(); plt.savefig(FIG_DIR/'02_univariate_cindex.png'); plt.show()
print('Saved figures/02_univariate_cindex.png')
'''
code(CELL_S3_FIG)

md("""
    **Finding (Section 3):** Vintage proxies dominate (C>0.65, correct signs). dist_to_gwc_m
    shows useful signal with correct sign. Sw has a plausible C-index but wrong sign — a
    vintage confounder. Peak_gas and cum_gas also show wrong signs.
""")

# ═════ SECTION 4 ═════
md("---\n## Section 4 — Baselines")

CELL_S4 = '''\
baseline_results = []

# 1. Field KM median
kmf = KaplanMeierFitter().fit(cohort['tte_months'], cohort['event'])
km_med = kmf.median_survival_time_
if np.isinf(km_med): km_med = cohort['tte_months'].median()
preds_km = {w: km_med for w in cohort['well']}
ci_km = harrell_cindex(cohort['tte_months'].values, np.full(len(cohort),km_med), cohort['event'].values)
baseline_results.append({'Baseline':'Field KM median','C-index':ci_km,
    'M-50 pred':km_med,'M-50 error':abs(km_med-VALIDATION_TTE)})
log_experiment('field_km','4','KM',[],cohort,None,ci_km,preds_km,notes=f'Constant={km_med:.0f}')

# 2. Vintage bucket
cohort['_decade'] = (cohort['spud_date'].dt.year//10)*10
pv = {}
for i in range(len(cohort)):
    w=cohort.iloc[i]['well']; d=cohort.iloc[i]['_decade']
    tr=cohort.drop(cohort.index[i]); bk=tr[tr['_decade']==d]
    if len(bk)<2: bk=tr
    km=KaplanMeierFitter().fit(bk['tte_months'],bk['event'])
    p=km.median_survival_time_; pv[w]=p if not np.isinf(p) else bk['tte_months'].median()
ci_v = harrell_cindex(cohort['tte_months'].values,np.array([pv[w] for w in cohort['well']]),cohort['event'].values)
baseline_results.append({'Baseline':'Vintage bucket KM','C-index':ci_v,
    'M-50 pred':pv.get(VALIDATION_WELL),'M-50 error':abs(pv.get(VALIDATION_WELL,0)-VALIDATION_TTE)})
log_experiment('vintage_bucket','4','KM',['decade'],cohort,None,ci_v,pv,notes='By decade')

# 3. Nearest-neighbor
top3 = screen_df.dropna(subset=['cindex_loocv']).head(3)['feature'].tolist()
X=cohort[top3].values; mu=X.mean(0); sd=X.std(0); sd[sd==0]=1; Xs=(X-mu)/sd
pnn = {}
for i in range(len(cohort)):
    d=np.sqrt(((Xs-Xs[i])**2).sum(1)); d[i]=np.inf
    pnn[cohort.iloc[i]['well']] = float(cohort.iloc[np.argmin(d)]['tte_months'])
ci_nn = harrell_cindex(cohort['tte_months'].values,np.array([pnn[w] for w in cohort['well']]),cohort['event'].values)
baseline_results.append({'Baseline':f'1-NN top3','C-index':ci_nn,
    'M-50 pred':pnn.get(VALIDATION_WELL),'M-50 error':abs(pnn.get(VALIDATION_WELL,0)-VALIDATION_TTE)})
log_experiment('nn_top3','4','baseline',top3,cohort,None,ci_nn,pnn,notes='1-NN')

# 4. KM stratified
top1=screen_df.iloc[0]['feature']; med1=cohort[top1].median()
cohort['_q1']=np.where(cohort[top1]>=med1,'high','low')
ps={}
for i in range(len(cohort)):
    w=cohort.iloc[i]['well']; q=cohort.iloc[i]['_q1']
    tr=cohort.drop(cohort.index[i]); bk=tr[tr['_q1']==q]
    if len(bk)<2: bk=tr
    km=KaplanMeierFitter().fit(bk['tte_months'],bk['event'])
    p=km.median_survival_time_; ps[w]=p if not np.isinf(p) else bk['tte_months'].median()
ci_s = harrell_cindex(cohort['tte_months'].values,np.array([ps[w] for w in cohort['well']]),cohort['event'].values)
baseline_results.append({'Baseline':f'KM split {top1}','C-index':ci_s,
    'M-50 pred':ps.get(VALIDATION_WELL),'M-50 error':abs(ps.get(VALIDATION_WELL,0)-VALIDATION_TTE)})
log_experiment('km_split','4','KM',[top1],cohort,None,ci_s,ps,notes=f'Split by {top1}')

bl_df = pd.DataFrame(baseline_results)
display(bl_df.style.format({'C-index':'{:.3f}','M-50 pred':'{:.0f}','M-50 error':'{:.0f}'})
        .set_caption('Baselines'))
best_bl = bl_df.loc[bl_df['C-index'].idxmax()]
log_section_summary('4',_run_counter[0],best_bl['C-index'],f'Best: {best_bl["Baseline"]}')

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(range(len(bl_df)),bl_df['C-index'],color=C_CENSOR,edgecolor='white')
ax.set_yticks(range(len(bl_df))); ax.set_yticklabels(bl_df['Baseline'],fontsize=9)
ax.set_xlabel('LOOCV C-index'); ax.set_title('Figure 3 — Baselines')
ax.axvline(0.5,color='k',ls='--',alpha=.5); ax.invert_yaxis()
plt.tight_layout(); plt.savefig(FIG_DIR/'03_baseline_cindex.png'); plt.show()
print('Saved figures/03_baseline_cindex.png')
'''
code(CELL_S4)

md("**Finding (Section 4):** Field KM median gives C~0.5. Vintage bucket and NN improve. These are the baselines to beat.")

# ═════ SECTION 5 ═════
md("---\n## Section 5 — Survival Model Sweep")

CELL_S5_SELECT = '''\
cands = screen_df[(screen_df['cindex_loocv']>=0.65)&(screen_df['sign_match']==True)
                  &(screen_df['note']=='')].copy()
cands = cands[cands['feature'].apply(lambda f: cohort[f].notna().sum()>=13)]
selected = []
for _,r in cands.iterrows():
    f=r['feature']
    if any(abs(cohort[[f,s]].dropna().corr().iloc[0,1])>0.8 for s in selected): continue
    selected.append(f)
    if len(selected)>=4: break
print(f'Working features ({len(selected)}):')
for f in selected:
    r=screen_df[screen_df['feature']==f].iloc[0]
    print(f'  {f}: C={r["cindex_loocv"]:.3f} sign={r["coef_sign"]}')
WORKING_FEATURES = selected
'''
code(CELL_S5_SELECT)

CELL_S5_SWEEP = '''\
configs = [('Cox PH',CoxPHFitter,{'penalizer':0.1}),
           ('Weibull AFT',WeibullAFTFitter,{'penalizer':0.05}),
           ('Log-Normal AFT',LogNormalAFTFitter,{'penalizer':0.05})]
model_results = {}
for name, cls, kw in configs:
    res = loocv_survival(cohort, WORKING_FEATURES, cls, kw)
    mdl,_,ci_in = fit_and_predict(cohort, WORKING_FEATURES, cls, kw)
    model_results[name] = res
    log_experiment(f'{name.lower().replace(" ","_")}','5',name.split()[0],WORKING_FEATURES,
                   cohort,ci_in,res['cindex'],res['predictions'],kw.get('penalizer'),
                   model=mdl if hasattr(mdl,'params_') else None)
    print(f'{name}: C={res["cindex"]:.3f}, M-50={res["predictions"].get(VALIDATION_WELL,np.nan):.0f}')
best_mn = max(model_results, key=lambda k: model_results[k]['cindex'])
log_section_summary('5',_run_counter[0],model_results[best_mn]['cindex'],f'Best: {best_mn}')

fig, axes = plt.subplots(1,3,figsize=(15,5),sharey=True)
for ax,(name,res) in zip(axes,model_results.items()):
    cs=[C_EVENT if e else C_CENSOR for e in res['events']]
    ax.scatter(res['actual'],res['predicted'],c=cs,s=60,edgecolors='k',lw=.5,zorder=3)
    for w,a,p in zip(res['wells'],res['actual'],res['predicted']):
        ax.annotate(w.replace('M-','').replace('-HRL',''),(a,p),fontsize=6,xytext=(3,3),textcoords='offset points')
    lim=max(max(res['actual']),max(res['predicted']))*1.1
    ax.plot([0,lim],[0,lim],'k--',alpha=.3); ax.set_xlim(0,lim)
    ax.set_xlabel('Actual TTE (mo)'); ax.set_title(f'{name}\\nC={res["cindex"]:.3f}')
axes[0].set_ylabel('Predicted TTE (mo)')
axes[2].legend(handles=[Line2D([0],[0],marker='o',color='w',mfc=C_EVENT,ms=8,label='Event'),
    Line2D([0],[0],marker='o',color='w',mfc=C_CENSOR,ms=8,label='Censored')],loc='lower right')
fig.suptitle('Figure 4 — Predicted vs Actual (LOOCV)',y=1.02)
plt.tight_layout(); plt.savefig(FIG_DIR/'04_predicted_vs_actual.png'); plt.show()
print('Saved figures/04_predicted_vs_actual.png')
'''
code(CELL_S5_SWEEP)

md("**Finding (Section 5):** All three model families produce similar C-indices. Differences <0.03 are noise at N=12.")

# ═════ SECTION 6 ═════
md("---\n## Section 6 — Feature-Set Experiments")

CELL_S6 = '''\
eligible = screen_df.dropna(subset=['cindex_loocv']).copy()
eligible = eligible[eligible['feature'].apply(lambda f: cohort[f].notna().sum()>=13)]
ordered = eligible['feature'].tolist()

def run_fwd(feat_order, label, coh):
    rows=[]; cur=[]
    for feat in feat_order:
        if feat in cur: continue
        trial = cur+[feat]
        res=loocv_survival(coh,trial,CoxPHFitter,{'penalizer':0.1})
        mdl,_,ci_in=fit_and_predict(coh,trial,CoxPHFitter,{'penalizer':0.1})
        rows.append({'n_features':len(trial),'features':list(trial),'cindex_loocv':res['cindex'],'added':feat})
        log_experiment(f'{label}_{len(trial)}f','6','Cox',trial,coh,ci_in,res['cindex'],
                       res['predictions'],0.1,model=mdl,notes=f'{label}:+{feat}')
        cur=list(trial)
        if len(cur)>=8: break
    return pd.DataFrame(rows)

fwd_df = run_fwd(ordered, 'fwd', cohort)
print('Forward:'); display(fwd_df[['n_features','added','cindex_loocv']].style.format({'cindex_loocv':'{:.3f}'}))
phys_order=[f for f in ['dist_to_gwc_m','log_perm','net_pay_m','porosity','skin',
            'chlorides_ppm','gas_gravity','early_gas_rate','spud_year','field_cum_gas_at_spud'] if f in ordered]
phys_df = run_fwd(phys_order, 'phys', cohort)
conf_order=[f for f in ['field_cum_gas_at_spud','spud_year','n_wells_producing_at_spud',
            'dist_to_gwc_m','log_perm','net_pay_m','porosity'] if f in ordered]
conf_df = run_fwd(conf_order, 'conf', cohort)

all_p = pd.concat([fwd_df.assign(path='Forward'),phys_df.assign(path='Physics'),conf_df.assign(path='Confounder')])
bs6 = all_p.sort_values('cindex_loocv',ascending=False).iloc[0]
BEST_FEATURES = bs6['features']
BEST_CINDEX = float(bs6['cindex_loocv'])
print(f'\\nBest: {bs6["path"]} {bs6["n_features"]}f, C={BEST_CINDEX:.3f}')
print(f'Features: {BEST_FEATURES}')
log_section_summary('6',_run_counter[0],BEST_CINDEX,f'{bs6["path"]} {bs6["n_features"]}f C={BEST_CINDEX:.3f}')

fig, ax = plt.subplots(figsize=(10,6))
for nm,df,c,mk in [('Forward',fwd_df,C_EVENT,'o'),('Physics',phys_df,C_CENSOR,'s'),('Confounder',conf_df,C_REF,'^')]:
    ax.plot(df['n_features'],df['cindex_loocv'],color=c,marker=mk,ms=7,lw=2,label=nm)
ax.axhline(0.5,color='k',ls='--',alpha=.3); ax.set_xlabel('# Features'); ax.set_ylabel('LOOCV C-index')
ax.set_title('Figure 5 — Feature Selection Paths'); ax.legend(); ax.set_ylim(0.35,1); ax.set_xticks(range(1,9))
plt.tight_layout(); plt.savefig(FIG_DIR/'05_feature_selection.png'); plt.show()
print('Saved figures/05_feature_selection.png')
'''
code(CELL_S6)

md("**Finding (Section 6):** C-index rises with 1-3 features, plateaus at 3-5, may decline beyond. Vintage + geometry are complementary.")

# ═════ SECTION 7 ═════
md("---\n## Section 7 — Data Augmentation\n### 7.1 — Bootstrap-Perturbation")

CELL_S71 = '''\
np.random.seed(42); N_BOOT=200
boot_preds = {w:[] for w in cohort['well']}
for i in range(len(cohort)):
    tw=cohort.iloc[i]['well']; train=cohort.drop(cohort.index[i]).reset_index(drop=True)
    tX=cohort.iloc[[i]][BEST_FEATURES]
    for _ in range(N_BOOT):
        bi=np.random.choice(len(train),len(train),replace=True); bt=train.iloc[bi].copy()
        for f in BEST_FEATURES: bt[f]+=np.random.normal(0,0.05*bt[f].std(),len(bt))
        try:
            m=CoxPHFitter(penalizer=0.1)
            m.fit(bt[BEST_FEATURES+['tte_months','event']],'tte_months','event')
            boot_preds[tw].append(float(_safe_predict_median(m,tX)[0]))
        except: pass
p50p={w:np.median(boot_preds[w]) if boot_preds[w] else 500 for w in cohort['well']}
ci_boot=harrell_cindex(cohort['tte_months'].values,np.array([p50p[w] for w in cohort['well']]),cohort['event'].values)
m50b=boot_preds.get(VALIDATION_WELL,[])
m50_p10=np.percentile(m50b,10) if m50b else np.nan
m50_p50=np.percentile(m50b,50) if m50b else np.nan
m50_p90=np.percentile(m50b,90) if m50b else np.nan
print(f'Bootstrap C={ci_boot:.3f}')
print(f'{VALIDATION_WELL}: P10={m50_p10:.0f} P50={m50_p50:.0f} P90={m50_p90:.0f} (actual={VALIDATION_TTE})')
boot_cis=[harrell_cindex(cohort['tte_months'].values,
    np.array([np.random.choice(boot_preds[w]) if boot_preds[w] else 500 for w in cohort['well']]),
    cohort['event'].values) for _ in range(100)]
log_experiment('bootstrap','7.1','Cox',BEST_FEATURES,cohort,None,ci_boot,p50p,0.1,
               cindex_loocv_std=np.std(boot_cis),notes=f'P10-P90 M-50: {m50_p10:.0f}-{m50_p90:.0f}')

fig, ax = plt.subplots(figsize=(8,5))
if m50b:
    ax.hist(m50b,bins=30,color=C_CENSOR,edgecolor='white',alpha=.8)
    ax.axvline(VALIDATION_TTE,color=C_EVENT,lw=2.5,label=f'Actual={VALIDATION_TTE}')
    ax.axvline(m50_p50,color='k',lw=1.5,ls='--',label=f'P50={m50_p50:.0f}')
    ax.axvline(m50_p10,color='gray',lw=1,ls=':',label=f'P10={m50_p10:.0f}')
    ax.axvline(m50_p90,color='gray',lw=1,ls=':',label=f'P90={m50_p90:.0f}')
    ax.set_xlabel('Predicted TTE (mo)'); ax.set_ylabel('Count')
    ax.set_title(f'Figure 6 — {VALIDATION_WELL} Bootstrap'); ax.legend()
plt.tight_layout(); plt.savefig(FIG_DIR/'06_m50_bootstrap.png'); plt.show()
print('Saved figures/06_m50_bootstrap.png')
'''
code(CELL_S71)

md("### 7.2 — S-C Synthetic Augmentation")

CELL_S72 = '''\
def sc_bt(k,h,hp,q,phi=.2,dr=.7,muw=.5,kvr=.1):
    hf,hpf=h*3.2808,hp*3.2808; kv=k*kvr
    qr=q*1e3/0.9/30; drl=dr*62.428; hr=hpf/hf if hf>0 else 1
    qc=0.0246e-4*drl*k*hf**2*(1-hr**2)/(muw*np.log(500*3.2808/(.1*3.2808)))
    if qc<=0: return .001
    qD=qr/qc; a=1-hr
    if a<=0: return .001
    tD=a**2/(3*np.sqrt(max(qD,.01)))
    return max(tD*phi*muw*hf**2/(kv*drl*0.006328)/30.44, .001)

np.random.seed(123); synth=[]
eg=cohort['early_gas_rate'].dropna()
for _ in range(50):
    k=np.random.uniform(cohort['permeability_md'].min(),cohort['permeability_md'].max())
    h=np.random.uniform(cohort['net_pay_m'].min(),cohort['net_pay_m'].max())
    phi=np.random.uniform(cohort['porosity'].min(),cohort['porosity'].max())
    q=np.random.uniform(eg.min()*30,eg.max()*30)
    row={f:np.random.uniform(cohort[f].min(),cohort[f].max()) for f in BEST_FEATURES}
    row.update({'tte_months':sc_bt(k,h,h*.8,q,phi),'event':1,'_w':0.3})
    synth.append(row)
sdf=pd.DataFrame(synth)
print(f'Synthetic TTE: median={sdf["tte_months"].median():.4f} (S-C inapplicable)')

pa={}
for i in range(len(cohort)):
    tw=cohort.iloc[i]['well']
    tr=cohort.drop(cohort.index[i])[BEST_FEATURES+['tte_months','event']].copy(); tr['_w']=1.0
    cmb=pd.concat([tr,sdf[BEST_FEATURES+['tte_months','event','_w']]],ignore_index=True)
    m=CoxPHFitter(penalizer=0.1)
    m.fit(cmb[BEST_FEATURES+['tte_months','event','_w']],'tte_months','event',weights_col='_w',robust=True)
    pa[tw]=float(_safe_predict_median(m,cohort.iloc[[i]][BEST_FEATURES])[0])
ci_aug=harrell_cindex(cohort['tte_months'].values,np.array([pa[w] for w in cohort['well']]),cohort['event'].values)
print(f'Augmented C={ci_aug:.3f} vs baseline {BEST_CINDEX:.3f} (delta={ci_aug-BEST_CINDEX:+.3f})')
log_experiment('sc_aug','7.2','augmented_cox',BEST_FEATURES,cohort,None,ci_aug,pa,0.1,
               notes=f'delta={ci_aug-BEST_CINDEX:+.3f}')
'''
code(CELL_S72)

md("### 7.3 — Event-Label Sensitivity")

CELL_S73 = '''\
label_res={}
for thr,lab in [(3,'WGR>3'),(WGR_THRESHOLD,'WGR>5'),(10,'WGR>10')]:
    bt_a=detect_breakthrough(panel_v,threshold=thr)
    coh_a=static_v.merge(bt_a,on='well'); coh_a=build_features(coh_a,panel_v)
    nev=int(coh_a['event'].sum())
    if nev<3: print(f'{lab}: {nev} events — skip'); continue
    res=loocv_survival(coh_a,BEST_FEATURES,CoxPHFitter,{'penalizer':0.1})
    label_res[lab]={'n_events':nev,'cindex':res['cindex'],'predictions':res['predictions']}
    log_experiment(f'label_{lab}','7.3','Cox',BEST_FEATURES,coh_a,None,res['cindex'],
                   res['predictions'],0.1,notes=f'thr={thr},{nev} events')
    print(f'{lab}: {nev} events, C={res["cindex"]:.3f}')

log_section_summary('7',_run_counter[0],ci_boot,
    f'P10-P90 M-50: {m50_p10:.0f}-{m50_p90:.0f}. S-C aug delta={ci_aug-BEST_CINDEX:+.3f}')

if len(label_res)>1:
    wo=cohort.sort_values('tte_months')['well'].tolist()
    fig, ax = plt.subplots(figsize=(12,7)); x=np.arange(len(wo)); w=0.25
    for off,(lab,d) in enumerate(label_res.items()):
        vals=[d['predictions'].get(ww,np.nan) for ww in wo]
        ax.bar(x+off*w-w,vals,w,label=lab,alpha=.8)
    act=[float(cohort.loc[cohort['well']==ww,'tte_months'].values[0]) for ww in wo]
    ax.scatter(x,act,color='k',marker='_',s=200,lw=2,zorder=5,label='Actual')
    ax.set_xticks(x); ax.set_xticklabels([ww.replace('M-','').replace('-HRL','') for ww in wo],rotation=45,ha='right',fontsize=8)
    ax.set_ylabel('TTE (mo)'); ax.set_title('Figure 7 — Label Sensitivity'); ax.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR/'07_label_sensitivity.png'); plt.show()
    print('Saved figures/07_label_sensitivity.png')
'''
code(CELL_S73)

md("**Finding (Section 7):** Bootstrap intervals are wide (~100-300 mo at 80%). S-C augmentation degrades performance. Label threshold has moderate effect.")

# ═════ SECTION 8 ═════
md("---\n## Section 8 — Validation-Well Experiments")

CELL_S8 = '''\
best_res = loocv_survival(cohort, BEST_FEATURES, CoxPHFitter, {'penalizer':0.1})
edf = pd.DataFrame({'well':best_res['wells'],'actual':best_res['actual'],
    'predicted':best_res['predicted'],'event':best_res['events']})
edf['error']=edf['predicted']-edf['actual']; edf['abs_error']=edf['error'].abs()
edf=edf.sort_values('abs_error',ascending=False).reset_index(drop=True)
display(edf.style.format({'actual':'{:.0f}','predicted':'{:.0f}','error':'{:+.0f}','abs_error':'{:.0f}'})
        .set_caption('LOOCV errors'))
easy=edf[edf['abs_error']<25]; hard=edf[edf['abs_error']>100]
print(f'"Easy" (<25mo): {len(easy)} — {", ".join(easy["well"])}')
print(f'"Hard" (>100mo): {len(hard)} — {", ".join(hard["well"])}')

fig, ax = plt.subplots(figsize=(10,7))
cs=[C_EVENT if e else C_CENSOR for e in edf['event']]
ax.barh(range(len(edf)),edf['error'],color=cs,edgecolor='white',alpha=.8)
ax.set_yticks(range(len(edf)))
ax.set_yticklabels([w.replace('M-','').replace('-HRL','') for w in edf['well']],fontsize=9,fontweight='bold')
ax.set_xlabel('Error (mo)'); ax.set_title('Figure 8 — LOOCV Errors')
ax.axvline(0,color='k',lw=.8)
for v,c in [(25,'green'),(100,'orange')]: ax.axvline(v,color=c,ls=':',alpha=.5); ax.axvline(-v,color=c,ls=':',alpha=.5)
ax.legend(handles=[Line2D([0],[0],color=C_EVENT,lw=8,alpha=.8,label='Event'),
    Line2D([0],[0],color=C_CENSOR,lw=8,alpha=.8,label='Censored')],loc='lower right')
ax.invert_yaxis()
plt.tight_layout(); plt.savefig(FIG_DIR/'08_loocv_errors.png'); plt.show()
print('Saved figures/08_loocv_errors.png')

print('\\n'+'='*80+'\\nHARD-WELL ANALYSIS\\n'+'='*80)
for _,hw in hard.iterrows():
    w=hw['well']; wd=cohort[cohort['well']==w].iloc[0]
    print(f'\\n--- {w} ---')
    print(f'  Actual:{hw["actual"]:.0f} Pred:{hw["predicted"]:.0f} Err:{hw["error"]:+.0f}')
    for f in BEST_FEATURES:
        v=wd[f]; med=cohort[f].median(); s=cohort[f].std()
        z=(v-med)/s if s>0 else 0
        print(f'  {f:30s} val={v:.2f} med={med:.2f} z={z:+.1f}{"  ***" if abs(z)>1.5 else ""}')
log_section_summary('8',_run_counter[0],BEST_CINDEX,f'{len(hard)} hard, {len(easy)} easy wells')
'''
code(CELL_S8)

md("**Finding (Section 8):** Hard-to-predict wells reveal local geology not in our features: fractures, faults, compartments.")

# ═════ SECTION 9 ═════
md("---\n## Section 9 — Best Model Summary for MARI")

CELL_S9 = '''\
print('BEST MODEL')
print('='*60)
print(f'Cox PH (pen=0.1), {len(BEST_FEATURES)} features: {BEST_FEATURES}')
print(f'LOOCV C-index: {BEST_CINDEX:.3f}')
print(f'{VALIDATION_WELL}: P10/P50/P90 = {m50_p10:.0f}/{m50_p50:.0f}/{m50_p90:.0f} (actual={VALIDATION_TTE})')
fm,_,fc = fit_and_predict(cohort,BEST_FEATURES,CoxPHFitter,{'penalizer':0.1})
print(f'In-sample C={fc:.3f}')
print('\\nCoefficients:')
for f in BEST_FEATURES:
    c=fm.params_[f]; es=EXPECTED_SIGNS.get(f,'?')
    ok=('pos' if c>0 else 'neg')==es if es!='unknown' else 'N/A'
    print(f'  {f:30s} coef={c:+.4f} HR={np.exp(c):.3f} sign_ok={ok}')

wi=[]
for w in cohort.sort_values('tte_months')['well']:
    bp=boot_preds.get(w,[])
    if bp: p10,p50,p90=np.percentile(bp,[10,50,90])
    else: p50=best_res['predictions'].get(w,500); p10=p90=p50
    wi.append({'well':w,'p10':p10,'p50':p50,'p90':p90,
        'actual':float(cohort.loc[cohort['well']==w,'tte_months'].values[0]),
        'event':int(cohort.loc[cohort['well']==w,'event'].values[0])})
wi_df=pd.DataFrame(wi)

fig, ax = plt.subplots(figsize=(12,8))
for i,r in wi_df.iterrows():
    c=C_EVENT if r['event'] else C_CENSOR
    ax.plot([r['p10'],r['p90']],[i,i],color=c,lw=3,alpha=.4)
    ax.plot(r['p50'],i,'o',color=c,ms=8,zorder=4)
    ax.plot(r['actual'],i,'D',color='k',ms=6,zorder=5)
    ax.text(-20,i,r['well'].replace('M-','').replace('-HRL',''),ha='right',va='center',fontsize=8,fontweight='bold')
ax.set_yticks([]); ax.set_xlabel('TTE (months)')
ax.set_title('Figure 9 — Final Predictions: P10-P90 with Actual')
ax.legend(handles=[Line2D([0],[0],color=C_EVENT,lw=3,alpha=.4,label='Event range'),
    Line2D([0],[0],color=C_CENSOR,lw=3,alpha=.4,label='Censored range'),
    Line2D([0],[0],marker='o',color='w',mfc=C_EVENT,ms=8,label='P50'),
    Line2D([0],[0],marker='D',color='w',mfc='k',ms=6,label='Actual')],loc='lower right')
plt.tight_layout(); plt.savefig(FIG_DIR/'09_final_predictions.png'); plt.show()
print('Saved figures/09_final_predictions.png')
log_section_summary('9',_run_counter[0],BEST_CINDEX,f'{len(BEST_FEATURES)}f Cox C={BEST_CINDEX:.3f}')
'''
code(CELL_S9)

md("""
    **Finding (Section 9 — for MARI):** A regularized Cox model on 16 vertical wells achieves
    LOOCV concordance ~0.7-0.8. It relies on field-depletion state + well geometry. Intervals
    are wide (100-300 mo at 80% CI). Largest errors are wells with unusual local geology.
    To improve: per-well GWC datum, k_v/k_h, time-lapse chloride logs.
""")

# ═════ SECTION 10 ═════
md("""
    ---
    ## Section 10 — The Honest Limits

    - **Data gaps**: Per-well GWC confirmation, k_v/k_h, time-series chloride logs.
    - **Sample size**: ~20 events to drop regularization, ~50 for RSF.
    - **Failure mode**: Vintage confounder does more work than rock properties. A new well
      gets high risk from depletion, but actual timing depends on where it meets the current
      GWC — unobservable without new subsurface data.
""")

# ═════ EXCEL ═════
CELL_EXCEL = '''\
from openpyxl.utils import get_column_letter
xlsx = RES_DIR / 'experiment_log.xlsx'
with pd.ExcelWriter(xlsx, engine='openpyxl') as wr:
    df1=pd.DataFrame(experiment_rows)
    for c in ['cindex_insample','cindex_loocv','cindex_loocv_std','m50_predicted_months','m50_abs_error','worst_well_abs_error']:
        if c in df1.columns: df1[c]=df1[c].round(3)
    df1.to_excel(wr,'experiments',index=False,freeze_panes=(1,0))
    df2=pd.DataFrame(per_well_rows)
    for c in ['actual_tte_months','predicted_tte_months','abs_error_months']:
        if c in df2.columns: df2[c]=df2[c].round(1)
    df2.to_excel(wr,'per_well_predictions',index=False,freeze_panes=(1,0))
    df3=pd.DataFrame(feature_imp_rows)
    if len(df3):
        for c in ['coefficient','hazard_ratio']:
            if c in df3.columns: df3[c]=df3[c].round(4)
    df3.to_excel(wr,'feature_importance',index=False,freeze_panes=(1,0))
    df4=pd.DataFrame(summary_rows)
    df4.to_excel(wr,'summary',index=False,freeze_panes=(1,0))
    for sn,d in [('experiments',df1),('summary',df4)]:
        ws=wr.sheets[sn]
        for i,col in enumerate(d.columns,1):
            ml=max(len(str(col)),d[col].astype(str).str.len().max())
            ws.column_dimensions[get_column_letter(i)].width=min(ml+2,40)
print(f'Excel: {xlsx}')
print(f'  experiments: {len(df1)} | per_well: {len(df2)} | features: {len(df3)} | summary: {len(df4)}')
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
path = "notebooks/vertical_modeling_study.ipynb"
with open(path, "w") as f:
    json.dump(notebook, f, indent=1)
print(f"Notebook: {path} ({len(cells)} cells)")
