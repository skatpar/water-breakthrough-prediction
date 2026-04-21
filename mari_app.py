"""
MARI Water Breakthrough Prediction System
==========================================
Gas-Water Breakthrough Dashboard — Habib Rahi Limestone (HRL)
Adapted from Volve oil-field POC for Mari Energies gas reservoir.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import base64
from pathlib import Path
import pickle

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="MARI Water Breakthrough Prediction",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONSTANTS
# ============================================================
GWC_TVD = 684  # m TVD SS
WGR_THRESHOLD = 5  # bbl/MMcf
MIN_CONSECUTIVE_MONTHS = 3
MU_GAS = 0.02   # cp
MU_WATER = 0.5   # cp

# ============================================================
# LOAD TRAINED MODEL
# ============================================================
import os

@st.cache_resource
def load_trained_model():
    """Load the trained Weibull AFT model from pickle file"""
    try:
        SCRIPT_DIR = Path(__file__).parent.resolve()
    except Exception:
        SCRIPT_DIR = Path.cwd()

    possible_paths = [
        SCRIPT_DIR / "mari_water_breakthrough_model.pkl",
        Path("mari_water_breakthrough_model.pkl"),
        Path.cwd() / "mari_water_breakthrough_model.pkl",
    ]

    for model_path in possible_paths:
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    model_artifacts = pickle.load(f)
                return model_artifacts, str(model_path)
            except Exception:
                continue

    return None, "Model file not found"

MODEL_ARTIFACTS, MODEL_STATUS = load_trained_model()

if MODEL_ARTIFACTS:
    TRAINED_MODEL = MODEL_ARTIFACTS['model']
    MODEL_FEATURES = MODEL_ARTIFACTS['features']
    SCALER_PARAMS = MODEL_ARTIFACTS.get('scaler_params', {})
    MODEL_NAME = MODEL_ARTIFACTS.get('model_name', 'Weibull AFT').split('(')[0].strip()
    KM_MEDIAN = MODEL_ARTIFACTS.get('km_median_months', None)
    STORED_FORECASTS = MODEL_ARTIFACTS.get('forecasts', [])
    IMPUTATION_LOG = MODEL_ARTIFACTS.get('imputation_log', [])
else:
    TRAINED_MODEL = None
    MODEL_FEATURES = []
    SCALER_PARAMS = {}
    MODEL_NAME = "Fallback Heuristic"
    KM_MEDIAN = None
    STORED_FORECASTS = []
    IMPUTATION_LOG = []

# ============================================================
# STIXOR COLOR SCHEME
# ============================================================
COLORS = {
    'bg_dark': '#1a1a1a',
    'bg_gradient_start': '#2d2520',
    'surface_dark': '#2a2a2a',
    'surface_darker': '#222222',
    'orange_primary': '#e8734a',
    'orange_light': '#f0956d',
    'orange_dark': '#c55a35',
    'text_white': '#ffffff',
    'text_gray': '#9a9a9a',
    'text_muted': '#6a6a6a',
    'success': '#4a9f6e',
    'danger': '#d64545',
    'border': '#3a3a3a',
    'p90': '#4a9f6e',
    'p50': '#e8734a',
    'p10': '#d64545',
}

# ============================================================
# LOAD LOGOS
# ============================================================
def get_logo_base64():
    try:
        logo_path = Path("stixor_logo.png")
        if logo_path.exists():
            with open(logo_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except Exception:
        pass
    return None

def get_logo_base642():
    try:
        logo_path = Path("logo_sm.png")
        if logo_path.exists():
            with open(logo_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except Exception:
        pass
    return None

LOGO_BASE64 = get_logo_base64()
LOGO_BASE642 = get_logo_base642()

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {{
        background: linear-gradient(135deg, {COLORS['bg_gradient_start']} 0%, {COLORS['bg_dark']} 50%, {COLORS['bg_dark']} 100%);
        font-family: 'Inter', sans-serif;
    }}

    .stApp, .stApp p, .stApp span, .stApp li, .stApp label {{
        color: {COLORS['text_white']} !important;
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: {COLORS['text_white']} !important;
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {COLORS['bg_dark']} 0%, {COLORS['surface_darker']} 100%);
        border-right: 1px solid {COLORS['border']};
    }}

    [data-testid="stSidebar"] .stRadio > div > label {{
        background-color: {COLORS['surface_dark']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        cursor: pointer;
    }}

    [data-testid="stSidebar"] .stRadio > div > label:hover {{
        border-color: {COLORS['orange_primary']};
    }}

    .section-header {{
        color: {COLORS['orange_primary']} !important;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid {COLORS['orange_primary']};
        margin-bottom: 1.25rem;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        background-color: {COLORS['surface_dark']};
        border-radius: 10px;
        padding: 0.5rem;
    }}

    .stTabs [data-baseweb="tab"] {{
        color: {COLORS['text_gray']} !important;
        border-radius: 8px;
    }}

    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {COLORS['orange_primary']} 0%, {COLORS['orange_dark']} 100%) !important;
        color: white !important;
    }}

    [data-testid="stExpander"] {{
        background-color: {COLORS['surface_dark']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
    }}

    .stDownloadButton > button {{
        background: linear-gradient(135deg, {COLORS['orange_primary']} 0%, {COLORS['orange_dark']} 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px;
    }}

    [data-testid="stFileUploader"] {{
        background-color: {COLORS['surface_dark']};
        border: 2px dashed {COLORS['border']};
        border-radius: 10px;
    }}

    [data-testid="stMetric"] {{
        background-color: {COLORS['surface_dark']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        padding: 1rem;
    }}

    [data-testid="stMetric"] label {{
        color: {COLORS['text_muted']} !important;
    }}

    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {COLORS['text_white']} !important;
    }}

    div[data-testid="stAlert"] {{
        background-color: {COLORS['surface_dark']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
    }}

    .model-badge {{
        background: linear-gradient(135deg, {COLORS['orange_primary']} 0%, {COLORS['orange_dark']} 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: 600;
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    ::-webkit-scrollbar {{ width: 8px; }}
    ::-webkit-scrollbar-track {{ background: {COLORS['bg_dark']}; }}
    ::-webkit-scrollbar-thumb {{ background: {COLORS['border']}; border-radius: 4px; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA PROCESSING FUNCTIONS
# ============================================================

def clean_numeric(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        return float(x.replace(',', '').replace('"', '').strip())
    return float(x)


def load_production_data(uploaded_file):
    """Load production data from Excel or CSV."""
    try:
        fname = uploaded_file.name.lower()
        if fname.endswith('.xlsx') or fname.endswith('.xls'):
            # Try reading the PRESSURES sheet (production data)
            try:
                df = pd.read_excel(uploaded_file, sheet_name='PRESSURES', header=2)
            except Exception:
                df = pd.read_excel(uploaded_file, header=2)

            df.columns = df.columns.str.strip()

            # Auto-detect and rename columns
            col_map = {}
            for c in df.columns:
                cl = c.lower()
                if 'date' in cl:
                    col_map[c] = 'date'
                elif 'well' in cl:
                    col_map[c] = 'well'
                elif 'choke' in cl:
                    col_map[c] = 'choke_64ths'
                elif 'days' in cl:
                    col_map[c] = 'days_on_prod'
                elif 'gas' in cl:
                    col_map[c] = 'gas_mmcf'
                elif 'water' in cl:
                    col_map[c] = 'water_bbl'
            df = df.rename(columns=col_map)
        else:
            df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            df.columns = df.columns.str.strip()

        # Parse dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        elif 'DATEPRD' in df.columns:
            df['date'] = pd.to_datetime(df['DATEPRD'], errors='coerce')

        df = df.dropna(subset=['date'])

        # Ensure numeric
        for c in ['gas_mmcf', 'water_bbl', 'choke_64ths', 'days_on_prod']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        if 'well' in df.columns:
            df['well'] = df['well'].astype(str).str.strip()

        # Fill NaN water with 0
        if 'water_bbl' in df.columns:
            df['water_bbl'] = df['water_bbl'].fillna(0)

        # Compute WGR
        if 'gas_mmcf' in df.columns and 'water_bbl' in df.columns:
            df['wgr'] = df['water_bbl'] / df['gas_mmcf'].replace(0, np.nan)
            df['wgr'] = df['wgr'].fillna(0).clip(lower=0)

        # Gas rate
        if 'gas_mmcf' in df.columns:
            days = df.get('days_on_prod', df['date'].dt.days_in_month)
            df['gas_rate_mmcfd'] = df['gas_mmcf'] / days.replace(0, np.nan)

        return df, None
    except Exception as e:
        return None, str(e)


def detect_breakthrough_wgr(df, well_col='well', threshold=WGR_THRESHOLD,
                             min_consecutive=MIN_CONSECUTIVE_MONTHS):
    """Detect water breakthrough per well using WGR threshold."""
    results = {}
    wells = df[well_col].unique() if well_col in df.columns else ['Unknown']

    for well in wells:
        wdata = df[df[well_col] == well].sort_values('date').copy() if well_col in df.columns else df.sort_values('date').copy()
        wdata = wdata[wdata.get('gas_mmcf', pd.Series(dtype=float)) > 0]

        if len(wdata) == 0:
            continue

        first_prod = wdata['date'].min()
        wdata['months_from_start'] = ((wdata['date'] - first_prod).dt.days / 30.44).round(1)

        # Detect sustained WGR above threshold
        wdata['above_thresh'] = (wdata['wgr'] >= threshold).astype(int)
        wdata['consec'] = wdata['above_thresh'].groupby(
            (wdata['above_thresh'] != wdata['above_thresh'].shift()).cumsum()
        ).cumcount() + 1

        sustained = wdata[(wdata['above_thresh'] == 1) & (wdata['consec'] >= min_consecutive)]

        if len(sustained) > 0:
            bt_month = sustained['months_from_start'].min()
            bt_date = sustained['date'].min()
            event = 1
        else:
            bt_month = wdata['months_from_start'].max()
            bt_date = None
            event = 0

        # Early features (first 6 months)
        early = wdata[wdata['months_from_start'] <= 6]

        results[well] = {
            'first_prod_date': first_prod,
            'breakthrough_date': bt_date,
            'time_to_event_months': max(bt_month, 0.5),
            'event_observed': event,
            'production_months': wdata['months_from_start'].max(),
            'early_gas_rate': early['gas_rate_mmcfd'].mean() if len(early) > 0 and 'gas_rate_mmcfd' in early.columns else np.nan,
            'early_wgr': early['wgr'].mean() if len(early) > 0 else np.nan,
            'early_whfp': early['whfp_psig'].mean() if 'whfp_psig' in early.columns and len(early) > 0 else np.nan,
            'total_gas_mmcf': wdata['gas_mmcf'].sum() if 'gas_mmcf' in wdata.columns else 0,
            'total_water_bbl': wdata['water_bbl'].sum() if 'water_bbl' in wdata.columns else 0,
            'final_wgr': wdata['wgr'].iloc[-6:].mean() if len(wdata) >= 6 else wdata['wgr'].mean(),
        }

    return results

# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_breakthrough(params):
    """Predict water breakthrough using trained model or heuristic fallback."""
    if TRAINED_MODEL is not None:
        return predict_with_trained_model(params)
    else:
        return predict_with_heuristic(params)


def predict_with_trained_model(params):
    """Use the trained Weibull AFT model to predict breakthrough."""
    # Extract raw parameters
    early_wgr = params.get('early_wgr', 0)
    early_gas_rate = params.get('early_gas_rate', 5)
    early_whfp = params.get('early_whfp', 500)
    porosity = params.get('porosity', 0.22)
    permeability = params.get('permeability_md', 20)
    sw = params.get('sw', 0.35)
    dist_to_gwc = params.get('dist_to_gwc_m', 0)

    # Handle NaN
    for var_name in ['early_wgr', 'early_gas_rate', 'early_whfp']:
        val = locals().get(var_name, 0)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            if var_name == 'early_gas_rate':
                early_gas_rate = 5
            elif var_name == 'early_whfp':
                early_whfp = 500
            elif var_name == 'early_wgr':
                early_wgr = 0

    # Build feature dict
    feature_values = {}
    for feat in MODEL_FEATURES:
        if feat == 'sw':
            feature_values[feat] = sw
        elif feat == 'log_perm':
            feature_values[feat] = np.log(max(permeability, 0.1))
        elif feat == 'dist_to_gwc_m':
            feature_values[feat] = dist_to_gwc
        elif feat == 'log_gas_rate':
            feature_values[feat] = np.log(max(early_gas_rate, 0.001))
        elif feat == 'porosity':
            feature_values[feat] = porosity
        elif feat == 'log_mobility':
            krw = sw ** 2
            krg = (1 - sw) ** 2
            mob = (krw / MU_WATER) / max(krg / MU_GAS, 1e-6)
            feature_values[feat] = np.log(max(mob, 0.001))
        elif feat == 'early_wgr':
            feature_values[feat] = early_wgr
        elif feat == 'early_whfp':
            feature_values[feat] = early_whfp
        elif feat == 'permeability_md':
            feature_values[feat] = permeability
        elif feat == 'chlorides_ppm':
            feature_values[feat] = params.get('chlorides_ppm', 5000)
        elif feat == 'net_pay_m':
            feature_values[feat] = params.get('net_pay_m', 100)
        elif feat == 'is_horizontal':
            feature_values[feat] = params.get('is_horizontal', 0)
        else:
            feature_values[feat] = 0

    # Standardize
    scaled = {}
    for feat in MODEL_FEATURES:
        if feat in SCALER_PARAMS.get('mean', {}) and feat in SCALER_PARAMS.get('std', {}):
            mean = SCALER_PARAMS['mean'][feat]
            std = SCALER_PARAMS['std'][feat]
            scaled[feat] = (feature_values[feat] - mean) / std if std > 0 else 0
        else:
            scaled[feat] = feature_values[feat]

    X = pd.DataFrame([scaled])[MODEL_FEATURES]

    try:
        surv_func = TRAINED_MODEL.predict_survival_function(X)
        times = surv_func.index.values
        probs = surv_func.values.flatten()

        p90 = times[np.argmin(np.abs(probs - 0.90))]
        p50 = times[np.argmin(np.abs(probs - 0.50))]
        p10 = times[np.argmin(np.abs(probs - 0.10))]

        # Ensure ordering and bounds
        p90 = max(1, min(p90, 600))
        p50 = max(2, min(p50, 800))
        p10 = max(3, min(p10, 1000))

        if p90 >= p50:
            p90 = p50 * 0.7
        if p50 >= p10:
            p10 = p50 * 1.45

        return {
            'P90_months': round(p90, 1),
            'P50_months': round(p50, 1),
            'P10_months': round(p10, 1),
            'model_used': MODEL_NAME,
            'survival_function': (times, probs)
        }
    except Exception as e:
        return predict_with_heuristic(params)


def predict_with_heuristic(params):
    """Fallback heuristic prediction for gas-water breakthrough."""
    base_time = 200  # months

    wgr = params.get('early_wgr', 0)
    if wgr is None or (isinstance(wgr, float) and np.isnan(wgr)):
        wgr = 0
    wgr_factor = np.exp(-0.3 * min(wgr, 20))

    sw = params.get('sw', 0.35)
    if sw is None or (isinstance(sw, float) and np.isnan(sw)):
        sw = 0.35
    sw_factor = np.exp(-3 * sw)

    gas_rate = params.get('early_gas_rate', 5)
    if gas_rate is None or (isinstance(gas_rate, float) and np.isnan(gas_rate)):
        gas_rate = 5
    rate_factor = (5 / max(gas_rate, 0.1)) ** 0.25

    dist = params.get('dist_to_gwc_m', 0)
    if dist is None or (isinstance(dist, float) and np.isnan(dist)):
        dist = 0
    dist_factor = max(0.3, dist / 50) if dist > 0 else 0.3

    p50 = base_time * wgr_factor * sw_factor * rate_factor * dist_factor
    p50 = max(6, min(p50, 600))

    uncertainty = 0.35
    p90 = p50 * (1 - uncertainty)
    p10 = p50 * (1 + uncertainty * 1.5)

    return {
        'P90_months': round(max(3, p90), 1),
        'P50_months': round(p50, 1),
        'P10_months': round(max(p50 + 1, p10), 1),
        'model_used': 'Heuristic (Fallback)',
        'survival_function': None
    }

# ============================================================
# CHART FUNCTIONS
# ============================================================

def create_production_chart(df, well=None):
    """Gas production profile with WGR."""
    wdata = df[df['well'] == well].sort_values('date') if well and 'well' in df.columns else df.sort_values('date')

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.6, 0.4],
                        subplot_titles=('Gas Production', 'Water-Gas Ratio'))

    if 'gas_mmcf' in wdata.columns:
        fig.add_trace(go.Scatter(
            x=wdata['date'], y=wdata['gas_mmcf'], name='Monthly Gas (MMcf)',
            fill='tozeroy', fillcolor='rgba(74, 159, 110, 0.6)',
            line=dict(color=COLORS['success'], width=1)), row=1, col=1)

    if 'water_bbl' in wdata.columns:
        fig.add_trace(go.Scatter(
            x=wdata['date'], y=wdata['water_bbl'], name='Monthly Water (bbl)',
            fill='tozeroy', fillcolor='rgba(80, 144, 192, 0.4)',
            line=dict(color='#5090c0', width=1)), row=1, col=1)

    if 'wgr' in wdata.columns:
        fig.add_trace(go.Scatter(
            x=wdata['date'], y=wdata['wgr'], name='WGR (bbl/MMcf)',
            line=dict(color=COLORS['orange_primary'], width=2)), row=2, col=1)
        # 3-month smoothed
        wdata_copy = wdata.copy()
        wdata_copy['wgr_smooth'] = wdata_copy['wgr'].rolling(3, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=wdata_copy['date'], y=wdata_copy['wgr_smooth'], name='3-mo Avg WGR',
            line=dict(color='#f0956d', width=2, dash='dot')), row=2, col=1)

    fig.add_hline(y=WGR_THRESHOLD, line_dash='dash', line_color=COLORS['danger'],
                  annotation_text=f'WGR={WGR_THRESHOLD}', row=2, col=1)

    fig.update_layout(
        height=450, showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=1, xanchor='right',
                    font=dict(color=COLORS['text_white'])),
        margin=dict(l=60, r=20, t=40, b=40),
        plot_bgcolor=COLORS['surface_dark'], paper_bgcolor=COLORS['surface_dark'],
        font=dict(family='Inter', size=11, color=COLORS['text_white']))
    fig.update_xaxes(showgrid=True, gridcolor=COLORS['border'], tickfont=dict(color=COLORS['text_gray']))
    fig.update_yaxes(showgrid=True, gridcolor=COLORS['border'], tickfont=dict(color=COLORS['text_gray']))
    fig.update_yaxes(title_text='Volume', row=1, col=1, title_font=dict(color=COLORS['text_gray']))
    fig.update_yaxes(title_text='WGR (bbl/MMcf)', row=2, col=1, title_font=dict(color=COLORS['text_gray']))

    return fig


def create_survival_chart(predictions, actual_months=None, well_name='Well'):
    """Survival curve with P10/P50/P90 markers."""
    fig = go.Figure()

    if predictions.get('survival_function') is not None:
        times, probs = predictions['survival_function']
        fig.add_trace(go.Scatter(
            x=list(times) + list(times[::-1]),
            y=[0.9]*len(times) + [0.1]*len(times),
            fill='toself', fillcolor='rgba(232, 115, 74, 0.1)',
            line=dict(color='rgba(0,0,0,0)'), name='P90-P10 Range'))
        fig.add_trace(go.Scatter(
            x=times, y=probs, mode='lines',
            name='Survival Probability (AFT)',
            line=dict(color=COLORS['orange_primary'], width=3)))
    else:
        max_time = predictions['P10_months'] * 1.5
        times = np.linspace(0, max_time, 200)
        scale = predictions['P50_months'] / np.log(2) ** 0.5
        probs = np.exp(-(times / scale) ** 2)

        fig.add_trace(go.Scatter(
            x=list(times) + list(times[::-1]),
            y=[0.9]*len(times) + [0.1]*len(times),
            fill='toself', fillcolor='rgba(232, 115, 74, 0.1)',
            line=dict(color='rgba(0,0,0,0)'), name='P90-P10 Range'))
        fig.add_trace(go.Scatter(
            x=times, y=probs, mode='lines',
            name='Survival Probability',
            line=dict(color=COLORS['orange_primary'], width=3)))

    markers = [
        (predictions['P90_months'], 0.9, COLORS['p90'], 'P90'),
        (predictions['P50_months'], 0.5, COLORS['p50'], 'P50'),
        (predictions['P10_months'], 0.1, COLORS['p10'], 'P10')
    ]

    for pval, prob, color, label in markers:
        fig.add_trace(go.Scatter(
            x=[pval], y=[prob], mode='markers+text',
            name=f"{label}: {pval} mo",
            marker=dict(size=14, color=color),
            text=[label], textposition='top center',
            textfont=dict(size=11, color=color)))
        fig.add_vline(x=pval, line_dash='dot', line_color=color, opacity=0.5)

    if actual_months:
        fig.add_vline(x=actual_months, line_dash='solid', line_color='#a855f7', line_width=3)
        fig.add_trace(go.Scatter(
            x=[actual_months], y=[0.5], mode='markers',
            name=f'Actual: {actual_months:.1f} mo',
            marker=dict(size=16, color='#a855f7', symbol='diamond')))

    if KM_MEDIAN:
        fig.add_vline(x=KM_MEDIAN, line_dash='dot', line_color='gray', opacity=0.4)
        fig.add_annotation(x=KM_MEDIAN, y=0.95, text=f'KM Baseline: {KM_MEDIAN:.0f} mo',
                          showarrow=False, font=dict(color='gray', size=9))

    model_text = predictions.get('model_used', 'Unknown')
    fig.update_layout(
        title=dict(
            text=f'Survival Probability: {well_name}<br><sup style="color:{COLORS["text_gray"]}">Model: {model_text}</sup>',
            font=dict(size=16, color=COLORS['text_white'])),
        xaxis_title='Time (months)',
        yaxis_title='P(No Breakthrough)',
        height=420, showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=1, xanchor='right',
                    font=dict(size=10, color=COLORS['text_white'])),
        margin=dict(l=60, r=20, t=70, b=50),
        plot_bgcolor=COLORS['surface_dark'],
        paper_bgcolor=COLORS['surface_dark'],
        font=dict(family='Inter', size=11, color=COLORS['text_white']),
        yaxis=dict(range=[0, 1.05]))
    fig.update_xaxes(showgrid=True, gridcolor=COLORS['border'], tickfont=dict(color=COLORS['text_gray']))
    fig.update_yaxes(showgrid=True, gridcolor=COLORS['border'], tickfont=dict(color=COLORS['text_gray']))

    return fig


def create_wgr_chart(df, bt_info, well=None):
    """WGR evolution chart."""
    wdata = df[df['well'] == well].sort_values('date').copy() if well and 'well' in df.columns else df.sort_values('date').copy()
    first_prod = bt_info.get('first_prod_date')
    if first_prod is not None:
        wdata['months'] = ((wdata['date'] - first_prod).dt.days / 30.44).round(1)
        wdata = wdata[wdata['months'] >= 0]
    else:
        wdata['months'] = range(len(wdata))

    wdata['wgr_smooth'] = wdata['wgr'].rolling(3, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wdata['months'], y=wdata['wgr'], name='Monthly WGR',
                             line=dict(color='#5090c0', width=1), opacity=0.4))
    fig.add_trace(go.Scatter(x=wdata['months'], y=wdata['wgr_smooth'], name='3-mo Avg',
                             line=dict(color=COLORS['orange_primary'], width=2.5)))

    for thresh, color in [(2, COLORS['orange_light']), (5, COLORS['danger']), (10, '#a855f7')]:
        fig.add_hline(y=thresh, line_dash='dot', line_color=color,
                      annotation_text=f'{thresh} bbl/MMcf')

    if bt_info.get('event_observed') == 1 and bt_info.get('time_to_event_months'):
        fig.add_vline(x=bt_info['time_to_event_months'], line_dash='solid',
                      line_color=COLORS['danger'], line_width=2)

    fig.update_layout(
        title=dict(text='Water-Gas Ratio Evolution', font=dict(size=16, color=COLORS['text_white'])),
        xaxis_title='Months from First Production', yaxis_title='WGR (bbl/MMcf)',
        height=380, showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=1, xanchor='right'),
        margin=dict(l=60, r=20, t=70, b=50),
        plot_bgcolor=COLORS['surface_dark'], paper_bgcolor=COLORS['surface_dark'],
        font=dict(family='Inter', size=11, color=COLORS['text_white']))
    fig.update_xaxes(showgrid=True, gridcolor=COLORS['border'], tickfont=dict(color=COLORS['text_gray']))
    fig.update_yaxes(showgrid=True, gridcolor=COLORS['border'], tickfont=dict(color=COLORS['text_gray']))

    return fig

# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    # Sidebar
    with st.sidebar:
        if LOGO_BASE64:
            st.markdown(f'<div style="text-align:center;padding:1rem;"><img src="data:image/png;base64,{LOGO_BASE64}" width="140"></div>', unsafe_allow_html=True)
        else:
            st.markdown("### STIXOR")

        st.markdown("---")

        if TRAINED_MODEL is not None:
            st.markdown(f"""
            <div style="background:{COLORS['success']}22; border:1px solid {COLORS['success']}; border-radius:8px; padding:0.5rem; margin-bottom:1rem;">
                <span style="color:{COLORS['success']};">Model Loaded</span><br>
                <span style="color:{COLORS['text_gray']}; font-size:0.8rem;">{MODEL_NAME}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:{COLORS['danger']}22; border:1px solid {COLORS['danger']}; border-radius:8px; padding:0.5rem; margin-bottom:1rem;">
                <span style="color:{COLORS['danger']};">Using Fallback</span><br>
                <span style="color:{COLORS['text_gray']}; font-size:0.8rem;">Heuristic Model</span>
            </div>
            """, unsafe_allow_html=True)

        page = st.radio("Navigation",
                        ["Analysis Dashboard", "Forecast Targets", "Data Upload",
                         "Model Information", "Help"],
                        label_visibility="collapsed")

    # Header
    st.markdown(f'<p class="section-header" style="font-size:1.5rem; border-bottom:none; margin-bottom:0.5rem;">MARI Water Breakthrough Prediction</p>', unsafe_allow_html=True)
    st.markdown(f'<span style="color:{COLORS["text_gray"]};">Habib Rahi Limestone (HRL) Gas Reservoir</span>', unsafe_allow_html=True)
    st.markdown("---")

    # ============================================================
    # ANALYSIS DASHBOARD
    # ============================================================
    if page == "Analysis Dashboard":
        if 'uploaded_data' not in st.session_state:
            st.session_state.uploaded_data = None
            st.session_state.well_results = None

        st.markdown('<p class="section-header">Data Input</p>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_file = st.file_uploader(
                "Upload Well Production Data (Excel or CSV)",
                type=['xlsx', 'xls', 'csv'])
        with col2:
            st.markdown("**Expected Columns:**")
            st.code("Date\nWells\nMonthly Gas Produced (MMcf)\nMonthly Water Produced (bbl)", language=None)
            st.markdown("**Optional:**")
            st.code("WHFP (psig)\nChoke\nDays on production", language=None)

        if uploaded_file:
            df, error = load_production_data(uploaded_file)
            if error:
                st.error(f"Error: {error}")
            else:
                st.session_state.uploaded_data = df
                well_results = detect_breakthrough_wgr(df)
                st.session_state.well_results = well_results
                n_wells = len(well_results)
                n_wet = sum(1 for v in well_results.values() if v['event_observed'] == 1)
                st.success(f"Loaded {len(df):,} records for {n_wells} wells ({n_wet} wet, {n_wells - n_wet} dry)")

        if st.session_state.uploaded_data is not None and st.session_state.well_results:
            df = st.session_state.uploaded_data
            well_results = st.session_state.well_results

            # Well selector
            well_names = sorted(well_results.keys())
            selected_well = st.selectbox("Select Well", well_names)

            if selected_well:
                bt_info = well_results[selected_well]

                st.markdown("---")
                st.markdown('<p class="section-header">Well Summary</p>', unsafe_allow_html=True)

                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Well", selected_well)
                col2.metric("Production Start", bt_info['first_prod_date'].strftime('%Y-%m-%d') if bt_info['first_prod_date'] else 'N/A')
                col3.metric("Total Gas", f"{bt_info['total_gas_mmcf']:,.0f} MMcf")
                col4.metric("Current WGR", f"{bt_info['final_wgr']:.1f} bbl/MMcf")
                col5.metric("Breakthrough", "OBSERVED" if bt_info['event_observed'] else "NOT YET")

                st.markdown("---")
                st.markdown('<p class="section-header">Breakthrough Prediction</p>', unsafe_allow_html=True)

                predictions = predict_breakthrough(bt_info)

                model_used = predictions.get('model_used', 'Unknown')
                st.markdown(f"""
                <div style="text-align:center; margin-bottom:1rem;">
                    <span class="model-badge">Model: {model_used}</span>
                </div>
                """, unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div style="background:{COLORS['surface_dark']}; border-radius:12px; padding:2rem; text-align:center; border-top:4px solid {COLORS['p90']};">
                        <div style="color:{COLORS['p90']}; font-weight:700; letter-spacing:2px; margin-bottom:1rem;">P90 - CONSERVATIVE</div>
                        <div style="font-size:4rem; font-weight:700; color:white;">{predictions['P90_months']}</div>
                        <div style="color:{COLORS['text_gray']}; font-size:1.1rem;">months</div>
                        <div style="color:{COLORS['text_muted']}; font-size:0.75rem; margin-top:1rem;">90% probability BT occurs AFTER this</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div style="background:{COLORS['surface_dark']}; border-radius:12px; padding:2rem; text-align:center; border-top:4px solid {COLORS['p50']};">
                        <div style="color:{COLORS['p50']}; font-weight:700; letter-spacing:2px; margin-bottom:1rem;">P50 - MOST LIKELY</div>
                        <div style="font-size:4rem; font-weight:700; color:white;">{predictions['P50_months']}</div>
                        <div style="color:{COLORS['text_gray']}; font-size:1.1rem;">months</div>
                        <div style="color:{COLORS['text_muted']}; font-size:0.75rem; margin-top:1rem;">Median expected breakthrough time</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div style="background:{COLORS['surface_dark']}; border-radius:12px; padding:2rem; text-align:center; border-top:4px solid {COLORS['p10']};">
                        <div style="color:{COLORS['p10']}; font-weight:700; letter-spacing:2px; margin-bottom:1rem;">P10 - OPTIMISTIC</div>
                        <div style="font-size:4rem; font-weight:700; color:white;">{predictions['P10_months']}</div>
                        <div style="color:{COLORS['text_gray']}; font-size:1.1rem;">months</div>
                        <div style="color:{COLORS['text_muted']}; font-size:0.75rem; margin-top:1rem;">Only 10% probability BT takes longer</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown('<p class="section-header">Analysis Charts</p>', unsafe_allow_html=True)

                tab1, tab2, tab3 = st.tabs(["Survival Curve", "Production Profile", "WGR Evolution"])

                with tab1:
                    actual_bt = bt_info['time_to_event_months'] if bt_info['event_observed'] else None
                    st.plotly_chart(create_survival_chart(predictions, actual_bt, selected_well),
                                   use_container_width=True)
                    if bt_info['event_observed']:
                        actual = bt_info['time_to_event_months']
                        if predictions['P90_months'] <= actual <= predictions['P10_months']:
                            st.success(f"Actual ({actual:.1f} mo) within P90-P10 range")
                        else:
                            st.warning(f"Actual ({actual:.1f} mo) outside P90-P10 range")

                with tab2:
                    st.plotly_chart(create_production_chart(df, selected_well),
                                   use_container_width=True)

                with tab3:
                    st.plotly_chart(create_wgr_chart(df, bt_info, selected_well),
                                   use_container_width=True)

                st.markdown("---")
                st.markdown('<p class="section-header">Model Input Parameters</p>', unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(pd.DataFrame({
                        'Parameter': ['Early Gas Rate', 'Early WGR', 'Avg WHFP',
                                      'Total Gas', 'Production Months'],
                        'Value': [
                            f"{bt_info['early_gas_rate']:.2f} MMcf/d" if not np.isnan(bt_info.get('early_gas_rate', np.nan)) else 'N/A',
                            f"{bt_info['early_wgr']:.2f} bbl/MMcf" if not np.isnan(bt_info.get('early_wgr', np.nan)) else 'N/A',
                            f"{bt_info['early_whfp']:.1f} psig" if bt_info.get('early_whfp') and not np.isnan(bt_info['early_whfp']) else 'N/A',
                            f"{bt_info['total_gas_mmcf']:,.0f} MMcf",
                            f"{bt_info['production_months']:.0f}"
                        ],
                        'Source': ['First 6 months', 'First 6 months', 'First 6 months',
                                   'Cumulative', 'Total']
                    }), hide_index=True, use_container_width=True)
                with col2:
                    st.markdown("**Model Features:**")
                    if TRAINED_MODEL is not None:
                        for feat in MODEL_FEATURES:
                            st.markdown(f"- `{feat}`")
                    else:
                        st.markdown("- WGR (early)\n- Gas Rate (early)\n- WHFP (avg)\n- Sw, Permeability")

    # ============================================================
    # FORECAST TARGETS PAGE
    # ============================================================
    elif page == "Forecast Targets":
        st.markdown('<p class="section-header">Forecast Target Wells</p>', unsafe_allow_html=True)

        st.markdown("""
        The following **horizontal wells** drilled in 2022-2024 are the forecast targets.
        Predictions use the trained survival model anchored on 17 vertical + 1 horizontal training wells.
        """)

        if STORED_FORECASTS:
            fcst_df = pd.DataFrame(STORED_FORECASTS)

            st.dataframe(
                fcst_df.style.format({
                    'P90_months': '{:.1f}',
                    'P50_months': '{:.1f}',
                    'P10_months': '{:.1f}'
                }),
                hide_index=True, use_container_width=True
            )

            st.markdown("---")
            st.markdown('<p class="section-header">Individual Forecasts</p>', unsafe_allow_html=True)

            for _, row in fcst_df.iterrows():
                with st.expander(f"{row['well']} — P50: {row['P50_months']:.1f} months"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("P90 (Conservative)", f"{row['P90_months']:.1f} mo")
                    col2.metric("P50 (Most Likely)", f"{row['P50_months']:.1f} mo")
                    col3.metric("P10 (Optimistic)", f"{row['P10_months']:.1f} mo")

                    if KM_MEDIAN:
                        diff = row['P50_months'] - KM_MEDIAN
                        direction = 'earlier' if diff < 0 else 'later'
                        st.info(f"P50 is {abs(diff):.1f} months {direction} than field-wide KM baseline ({KM_MEDIAN:.0f} mo)")
        else:
            st.warning("No stored forecasts. Run the training notebook first to generate predictions.")

        st.markdown("---")
        st.markdown("**Validation: M-122H**")
        st.markdown("M-122H broke through at **month 12** and serves as the horizontal-well validation anchor.")

    # ============================================================
    # DATA UPLOAD PAGE
    # ============================================================
    elif page == "Data Upload":
        st.markdown('<p class="section-header">Data Upload Guidelines</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Required Data Format")
            st.markdown("Upload **Excel (.xlsx)** or **CSV** with monthly production data.")

            st.markdown("**Mandatory Columns:**")
            st.dataframe(pd.DataFrame({
                'Column': ['Date', 'Wells', 'Monthly Gas Produced', 'Monthly Water Produced'],
                'Description': ['Production date', 'Well identifier', 'Monthly gas volume', 'Monthly water volume'],
                'Format/Units': ['YYYY-MM-DD', 'String', 'MMcf', 'bbl']
            }), hide_index=True, use_container_width=True)

            st.markdown("**Optional Columns:**")
            st.dataframe(pd.DataFrame({
                'Column': ['WHFP', 'Choke', 'Days on production'],
                'Description': ['Wellhead flowing pressure', 'Choke size', 'Producing days in month'],
                'Format/Units': ['psig', '1/64 inches', 'days']
            }), hide_index=True, use_container_width=True)

        with col2:
            st.subheader("Breakthrough Definition")
            st.markdown(f"""
            **Event:** Water-Gas Ratio (WGR) > **{WGR_THRESHOLD} bbl/MMcf** sustained for
            **{MIN_CONSECUTIVE_MONTHS}+ consecutive months**

            **WGR Calculation:**
            ```
            WGR = Monthly Water (bbl) / Monthly Gas (MMcf)
            ```

            **Quality Requirements:**
            - Minimum **6 months** of data per well
            - Monthly cadence (one row per well per month)
            - Consistent units (MMcf for gas, bbl for water)
            """)

    # ============================================================
    # MODEL INFORMATION PAGE
    # ============================================================
    elif page == "Model Information":
        st.markdown('<p class="section-header">Model Technical Information</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Current Model")
            if TRAINED_MODEL is not None:
                st.success(f"**{MODEL_NAME}**")
                st.markdown(f"**Model Type:** Weibull Accelerated Failure Time (AFT)")
                st.markdown("**Features:**")
                for feat in MODEL_FEATURES:
                    st.markdown(f"- `{feat}`")

                metrics = MODEL_ARTIFACTS.get('metrics', {})
                st.markdown(f"""
                **Regularization:** {MODEL_ARTIFACTS.get('regularization', {}).get('type', 'None')}
                (penalty={MODEL_ARTIFACTS.get('regularization', {}).get('penalty', 'N/A')})

                **Metrics:**
                - AIC: {metrics.get('AIC', 'N/A')}
                - BIC: {metrics.get('BIC', 'N/A')}
                - C-Index: {metrics.get('C_Index', 'N/A')}
                """)
            else:
                st.warning("Using heuristic fallback model")

            st.markdown("---")
            st.subheader("Reservoir Context")
            st.markdown(f"""
            - **Field:** Mari Gas Field (HRL)
            - **Formation:** Habib Rahi Limestone
            - **GWC:** {GWC_TVD} m TVD SS
            - **Drive:** Natural depletion
            - **Gas type:** Dry gas with condensate
            """)

        with col2:
            st.subheader("Understanding P10/P50/P90")
            st.dataframe(pd.DataFrame({
                'Percentile': ['P90', 'P50', 'P10'],
                'Meaning': ['90% BT occurs AFTER', 'Median (most likely)', '10% BT takes longer'],
                'Use Case': ['Conservative planning', 'Base case', 'Optimistic scenario']
            }), hide_index=True, use_container_width=True)

            st.subheader("Assumptions & Limitations")
            st.warning("""
            **Assumptions:**
            - WGR > 5 bbl/MMcf sustained 3+ months = breakthrough
            - WHFP used as pressure proxy (no BHP available)
            - Measured depth as TVD proxy (inaccurate for horizontals)
            - Two-zone model: Zone A (verticals), Zone B (horizontals)
            - Corey-type relative permeability for mobility ratio
            - Missing water records = zero water production

            **Limitations:**
            - Only 14 events in training (EPV ~2.8 with 5 features)
            - Only 1 horizontal well with breakthrough data
            - No well coordinates (spatial features unavailable)
            - Monthly resolution (+-1 month accuracy)
            - M-51 excluded (month-0 flowback water)
            """)

            if IMPUTATION_LOG:
                st.subheader("Imputation Log")
                st.dataframe(pd.DataFrame(IMPUTATION_LOG), hide_index=True,
                             use_container_width=True)

    # ============================================================
    # HELP PAGE
    # ============================================================
    elif page == "Help":
        st.markdown('<p class="section-header">User Guide</p>', unsafe_allow_html=True)

        st.subheader("Quick Start")
        col1, col2, col3, col4 = st.columns(4)
        col1.info("**Step 1**\n\nGo to Analysis Dashboard")
        col2.info("**Step 2**\n\nUpload Excel/CSV file")
        col3.info("**Step 3**\n\nSelect a well")
        col4.info("**Step 4**\n\nReview P10/P50/P90")

        st.markdown("---")
        st.subheader("FAQ")

        with st.expander("What data format is required?"):
            st.write("Excel or CSV with monthly gas/water production per well. See Data Upload page.")

        with st.expander("How is breakthrough defined?"):
            st.write(f"WGR (Water-Gas Ratio) exceeding {WGR_THRESHOLD} bbl/MMcf for "
                     f"{MIN_CONSECUTIVE_MONTHS}+ consecutive months.")

        with st.expander("Why WGR instead of water cut?"):
            st.write("This is a gas reservoir. Water cut (water / total liquid) is used for oil fields. "
                     "WGR (water bbl per MMcf gas) is the standard metric for gas wells.")

        with st.expander("What about M-51?"):
            st.write("M-51 shows water from month 0, which is likely completion flowback, "
                     "not reservoir water breakthrough. It is excluded from model training.")

        with st.expander("How accurate are forecasts for horizontal wells?"):
            st.write("The model is trained primarily on vertical wells with only 1 horizontal "
                     "(M-122H) as an anchor. Horizontal well forecasts carry higher uncertainty. "
                     "The P90-P10 range reflects this.")

        with st.expander("What do P10/P50/P90 mean?"):
            st.write("P90: 90% chance BT happens after this time (conservative). "
                     "P50: median/most likely. P10: only 10% chance takes longer (optimistic).")

        with st.expander("What model is being used?"):
            if TRAINED_MODEL:
                st.write(f"Currently using: **{MODEL_NAME}**")
                st.write(f"Features: {', '.join(MODEL_FEATURES)}")
            else:
                st.write("Using heuristic fallback (trained model not found)")


if __name__ == "__main__":
    main()
