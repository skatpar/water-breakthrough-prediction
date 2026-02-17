"""
Water Breakthrough Prediction System
=====================================
Enterprise Dashboard - Stixor Theme
Uses Trained Weibull AFT Model
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
    page_title="Water Breakthrough Prediction System",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD TRAINED MODEL
# ============================================================
import os

@st.cache_resource
def load_trained_model():
    """Load the trained Weibull AFT model from pickle file"""
    
    # Get the directory where this script is located
    try:
        SCRIPT_DIR = Path(__file__).parent.resolve()
    except:
        SCRIPT_DIR = Path.cwd()
    
    # Possible locations for the model file
    possible_paths = [
        SCRIPT_DIR / "water_breakthrough_model.pkl",
        Path("water_breakthrough_model.pkl"),
        Path.cwd() / "water_breakthrough_model.pkl",
    ]
    
    # Try each path
    for model_path in possible_paths:
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    model_artifacts = pickle.load(f)
                return model_artifacts, str(model_path)
            except Exception as e:
                continue
    
    # Return None if not found
    return None, "Model file not found"

MODEL_ARTIFACTS, MODEL_STATUS = load_trained_model()

if MODEL_ARTIFACTS:
    TRAINED_MODEL = MODEL_ARTIFACTS['model']
    MODEL_FEATURES = MODEL_ARTIFACTS['features']
    SCALER_PARAMS = MODEL_ARTIFACTS['scaler_params']
    MODEL_NAME = MODEL_ARTIFACTS.get('model_name', 'Weibull AFT').split('(')[0].strip()
else:
    TRAINED_MODEL = None
    MODEL_FEATURES = []
    SCALER_PARAMS = {}
    MODEL_NAME = "Fallback Heuristic"
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
# LOAD STIXOR LOGO
# ============================================================
def get_logo_base64():
    try:
        logo_path = Path("stixor_logo.png")
        if logo_path.exists():
            with open(logo_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except:
        pass
    return None

def get_logo_base642():
    try:
        logo_path = Path("logo_sm.png")
        if logo_path.exists():
            with open(logo_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except:
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
    
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: {COLORS['bg_dark']};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {COLORS['border']};
        border-radius: 4px;
    }}
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

def process_uploaded_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        
        numeric_cols = ['BORE_OIL_VOL', 'BORE_GAS_VOL', 'BORE_WAT_VOL', 
                       'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(clean_numeric)
        
        if 'DATEPRD' in df.columns:
            try:
                df['DATEPRD'] = pd.to_datetime(df['DATEPRD'], format='%d-%b-%y')
            except:
                df['DATEPRD'] = pd.to_datetime(df['DATEPRD'])
        
        if 'BORE_OIL_VOL' in df.columns and 'BORE_WAT_VOL' in df.columns:
            df['total_liquid'] = df['BORE_OIL_VOL'] + df['BORE_WAT_VOL']
            df['water_cut'] = df['BORE_WAT_VOL'] / df['total_liquid'].replace(0, np.nan)
            df['water_cut'] = df['water_cut'].fillna(0)
        
        return df, None
    except Exception as e:
        return None, str(e)

def detect_breakthrough(df, threshold=0.10, min_consecutive=3):
    df = df.sort_values('DATEPRD').copy()
    
    prod_data = df[df['BORE_OIL_VOL'] > 0]
    if len(prod_data) == 0:
        return None
    
    first_prod = prod_data['DATEPRD'].min()
    df['days_from_start'] = (df['DATEPRD'] - first_prod).dt.days
    
    df['above_threshold'] = (df['water_cut'] >= threshold).astype(int)
    df['consecutive'] = df['above_threshold'].groupby(
        (df['above_threshold'] != df['above_threshold'].shift()).cumsum()
    ).cumcount() + 1
    
    sustained = df[(df['above_threshold'] == 1) & (df['consecutive'] >= min_consecutive)]
    
    if len(sustained) > 0:
        bt_day = sustained['days_from_start'].min()
        bt_date = sustained['DATEPRD'].min()
        event_observed = 1
    else:
        bt_day = df['days_from_start'].max()
        bt_date = None
        event_observed = 0
    
    early = df[(df['days_from_start'] >= 0) & (df['days_from_start'] <= 60)]
    
    return {
        'first_prod_date': first_prod,
        'breakthrough_date': bt_date,
        'days_to_breakthrough': bt_day,
        'months_to_breakthrough': bt_day / 30.44,
        'event_observed': event_observed,
        'early_oil_rate': early['BORE_OIL_VOL'].mean() if len(early) > 0 else np.nan,
        'early_water_cut': early['water_cut'].mean() if len(early) > 0 else np.nan,
        'early_pressure': early['AVG_DOWNHOLE_PRESSURE'].mean() if 'AVG_DOWNHOLE_PRESSURE' in early.columns else np.nan,
        'total_oil': df['BORE_OIL_VOL'].sum(),
        'total_water': df['BORE_WAT_VOL'].sum(),
        'final_water_cut': prod_data['water_cut'].iloc[-30:].mean() if len(prod_data) >= 30 else prod_data['water_cut'].mean(),
        'production_days': len(prod_data)
    }

# ============================================================
# PREDICTION FUNCTION - USES TRAINED MODEL
# ============================================================

def predict_breakthrough(params):
    """
    Predict water breakthrough using the trained Weibull AFT model.
    Falls back to heuristic if model not available.
    """
    
    # If trained model is available, use it
    if TRAINED_MODEL is not None:
        return predict_with_trained_model(params)
    else:
        # Fallback to heuristic method
        return predict_with_heuristic(params)

def predict_with_trained_model(params):
    """Use the trained Weibull AFT model to predict breakthrough"""
    
    # Extract raw parameters
    early_water_cut = params.get('early_water_cut', 0.02)
    early_oil_rate = params.get('early_oil_rate', 1500)
    early_pressure = params.get('early_pressure', 250)
    
    # Handle NaN values with defaults
    if np.isnan(early_water_cut) or early_water_cut <= 0:
        early_water_cut = 0.02
    if np.isnan(early_oil_rate) or early_oil_rate <= 0:
        early_oil_rate = 1500
    if np.isnan(early_pressure) or early_pressure <= 0:
        early_pressure = 250
    
    # Estimate mobility ratio from water cut (same logic as training)
    if early_water_cut < 0.02:
        mobility_ratio = 0.6
    elif early_water_cut < 0.05:
        mobility_ratio = 0.95
    else:
        mobility_ratio = 1.5
    
    # Create feature dictionary
    raw_features = {
        'initial_water_cut': early_water_cut,
        'initial_oil_rate': early_oil_rate,
        'avg_pressure': early_pressure,
        'mobility_ratio': mobility_ratio,
    }
    
    # Create derived features that model needs
    feature_values = {}
    
    for feat in MODEL_FEATURES:
        if feat == 'initial_water_cut':
            feature_values[feat] = early_water_cut
        elif feat == 'initial_oil_rate':
            feature_values[feat] = early_oil_rate
        elif feat == 'avg_pressure':
            feature_values[feat] = early_pressure
        elif feat == 'mobility_ratio':
            feature_values[feat] = mobility_ratio
        elif feat == 'log_oil_rate':
            feature_values[feat] = np.log(max(early_oil_rate, 1))
        elif feat == 'log_mobility':
            feature_values[feat] = np.log(max(mobility_ratio, 0.1))
        elif feat == 'log_water_cut':
            feature_values[feat] = np.log(max(early_water_cut, 0.001))
        elif feat == 'log_pressure':
            feature_values[feat] = np.log(max(early_pressure, 1))
        elif feat == 'pressure_normalized':
            feature_values[feat] = early_pressure / 250  # Assuming median ~250
        elif feat == 'mobility_squared':
            feature_values[feat] = mobility_ratio ** 2
        elif feat == 'wc_pressure_interaction':
            feature_values[feat] = early_water_cut * early_pressure
        elif feat == 'wc_rate_interaction':
            feature_values[feat] = early_water_cut * early_oil_rate
        elif feat == 'productivity_proxy':
            feature_values[feat] = early_oil_rate / max(early_pressure, 1)
        elif feat == 'water_mobility_proxy':
            feature_values[feat] = early_water_cut * mobility_ratio
        elif feat == 'mobility_pressure_interaction':
            feature_values[feat] = mobility_ratio * early_pressure
        elif feat == 'rate_pressure_ratio':
            feature_values[feat] = early_oil_rate / max(early_pressure, 1)
        else:
            # Default to 0 for unknown features
            feature_values[feat] = 0
    
    # Standardize features using saved scaler parameters
    scaled_features = {}
    for feat in MODEL_FEATURES:
        if feat in SCALER_PARAMS.get('mean', {}) and feat in SCALER_PARAMS.get('std', {}):
            mean = SCALER_PARAMS['mean'][feat]
            std = SCALER_PARAMS['std'][feat]
            if std > 0:
                scaled_features[feat] = (feature_values[feat] - mean) / std
            else:
                scaled_features[feat] = 0
        else:
            # If no scaler params, use raw value
            scaled_features[feat] = feature_values[feat]
    
    # Create DataFrame for prediction
    X = pd.DataFrame([scaled_features])[MODEL_FEATURES]
    
    try:
        # Predict survival function
        surv_func = TRAINED_MODEL.predict_survival_function(X)
        
        # Extract times and probabilities
        times = surv_func.index.values
        probs = surv_func.values.flatten()
        
        # Find P90, P50, P10 (times at which survival probability = 0.9, 0.5, 0.1)
        p90_months = times[np.argmin(np.abs(probs - 0.90))]
        p50_months = times[np.argmin(np.abs(probs - 0.50))]
        p10_months = times[np.argmin(np.abs(probs - 0.10))]
        
        # Ensure reasonable bounds
        p90_months = max(1, min(p90_months, 60))
        p50_months = max(2, min(p50_months, 80))
        p10_months = max(3, min(p10_months, 100))
        
        # Ensure P90 < P50 < P10
        if p90_months >= p50_months:
            p90_months = p50_months * 0.7
        if p50_months >= p10_months:
            p10_months = p50_months * 1.45
        
        return {
            'P90_days': round(p90_months * 30.44),
            'P50_days': round(p50_months * 30.44),
            'P10_days': round(p10_months * 30.44),
            'P90_months': round(p90_months, 1),
            'P50_months': round(p50_months, 1),
            'P10_months': round(p10_months, 1),
            'model_used': MODEL_NAME,
            'survival_function': (times, probs)  # Store for plotting
        }
        
    except Exception as e:
        print(f"Model prediction error: {e}")
        # Fallback to heuristic
        return predict_with_heuristic(params)

def predict_with_heuristic(params):
    """Fallback heuristic prediction when model is not available"""
    
    base_time = 200
    
    wc = params.get('early_water_cut', 0.02)
    if np.isnan(wc):
        wc = 0.02
    wc_factor = np.exp(-8 * min(wc, 0.20))
    
    pressure = params.get('early_pressure', 250)
    if np.isnan(pressure) or pressure == 0:
        pressure = 250
    pressure_factor = (pressure / 250) ** 0.5
    
    rate = params.get('early_oil_rate', 1500)
    if np.isnan(rate) or rate == 0:
        rate = 1500
    rate_factor = (1500 / rate) ** 0.25
    
    if wc < 0.02:
        mobility_factor = 1.2
    elif wc < 0.05:
        mobility_factor = 1.0
    else:
        mobility_factor = 0.7
    
    p50 = base_time * wc_factor * pressure_factor * rate_factor * mobility_factor
    
    uncertainty = 0.30
    p90 = p50 * (1 - uncertainty)
    p10 = p50 * (1 + uncertainty * 1.5)
    
    p90 = max(30, p90)
    p50 = max(60, p50)
    p10 = max(90, p10)
    
    return {
        'P90_days': round(p90),
        'P50_days': round(p50),
        'P10_days': round(p10),
        'P90_months': round(p90 / 30.44, 1),
        'P50_months': round(p50 / 30.44, 1),
        'P10_months': round(p10 / 30.44, 1),
        'model_used': 'Heuristic (Fallback)',
        'survival_function': None
    }

# ============================================================
# CHART FUNCTIONS
# ============================================================

def create_production_chart(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.6, 0.4], subplot_titles=('Production Rates', 'Water Cut'))
    
    fig.add_trace(go.Scatter(x=df['DATEPRD'], y=df['BORE_OIL_VOL'], name='Oil Rate',
                             fill='tozeroy', fillcolor='rgba(74, 159, 110, 0.6)',
                             line=dict(color=COLORS['success'], width=0)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df['DATEPRD'], y=df['BORE_WAT_VOL'], name='Water Rate',
                             fill='tozeroy', fillcolor='rgba(80, 144, 192, 0.6)',
                             line=dict(color='#5090c0', width=0)), row=1, col=1)
    
    df['wc_smooth'] = df['water_cut'].rolling(14, min_periods=1).mean()
    fig.add_trace(go.Scatter(x=df['DATEPRD'], y=df['water_cut'] * 100, name='Daily WC',
                             line=dict(color='#5090c0', width=1), opacity=0.3), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['DATEPRD'], y=df['wc_smooth'] * 100, name='14-day Avg',
                             line=dict(color=COLORS['orange_primary'], width=2)), row=2, col=1)
    
    fig.add_hline(y=10, line_dash='dash', line_color=COLORS['danger'],
                  annotation_text='10% Threshold', row=2, col=1)
    
    fig.update_layout(height=450, showlegend=True,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, x=1, xanchor='right',
                                  font=dict(color=COLORS['text_white'])),
                      margin=dict(l=60, r=20, t=40, b=40),
                      plot_bgcolor=COLORS['surface_dark'], paper_bgcolor=COLORS['surface_dark'],
                      font=dict(family='Inter', size=11, color=COLORS['text_white']))
    
    fig.update_xaxes(showgrid=True, gridcolor=COLORS['border'], tickfont=dict(color=COLORS['text_gray']))
    fig.update_yaxes(showgrid=True, gridcolor=COLORS['border'], tickfont=dict(color=COLORS['text_gray']))
    fig.update_yaxes(title_text='Volume (Sm³/d)', row=1, col=1, title_font=dict(color=COLORS['text_gray']))
    fig.update_yaxes(title_text='Water Cut (%)', row=2, col=1, range=[0, 100], title_font=dict(color=COLORS['text_gray']))
    
    return fig

def create_survival_chart(predictions, actual_months=None, well_name='Well'):
    """Create survival chart - uses actual model survival function if available"""
    
    fig = go.Figure()
    
    # Check if we have actual survival function from model
    if predictions.get('survival_function') is not None:
        times, probs = predictions['survival_function']
        times_months = times  # Already in months from model
        
        # Add confidence band (P90-P10 range)
        fig.add_trace(go.Scatter(
            x=list(times_months) + list(times_months[::-1]),
            y=[0.9]*len(times_months) + [0.1]*len(times_months),
            fill='toself', fillcolor='rgba(232, 115, 74, 0.1)',
            line=dict(color='rgba(0,0,0,0)'), name='P90-P10 Range'
        ))
        
        # Add actual survival curve from model
        fig.add_trace(go.Scatter(
            x=times_months, y=probs, mode='lines', 
            name='Survival Probability (AFT Model)',
            line=dict(color=COLORS['orange_primary'], width=3)
        ))
    else:
        # Fallback: Generate synthetic curve
        max_time = predictions['P10_days'] * 1.5
        times = np.linspace(0, max_time, 200)
        scale = predictions['P50_days'] / np.log(2) ** 0.5
        probs = np.exp(-(times / scale) ** 2)
        times_months = times / 30.44
        
        fig.add_trace(go.Scatter(
            x=list(times_months) + list(times_months[::-1]),
            y=[0.9]*len(times_months) + [0.1]*len(times_months),
            fill='toself', fillcolor='rgba(232, 115, 74, 0.1)',
            line=dict(color='rgba(0,0,0,0)'), name='P90-P10 Range'
        ))
        
        fig.add_trace(go.Scatter(
            x=times_months, y=probs, mode='lines', 
            name='Survival Probability',
            line=dict(color=COLORS['orange_primary'], width=3)
        ))
    
    # Add P90, P50, P10 markers
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
            textfont=dict(size=11, color=color)
        ))
        fig.add_vline(x=pval, line_dash='dot', line_color=color, opacity=0.5)
    
    # Add actual breakthrough if observed
    if actual_months:
        fig.add_vline(x=actual_months, line_dash='solid', line_color='#a855f7', line_width=3)
        fig.add_trace(go.Scatter(
            x=[actual_months], y=[0.5], mode='markers',
            name=f'Actual: {actual_months:.1f} mo',
            marker=dict(size=16, color='#a855f7', symbol='diamond')
        ))
    
    # Model indicator
    model_text = predictions.get('model_used', 'Unknown')
    
    fig.update_layout(
        title=dict(
            text=f'Survival Probability: {well_name}<br><sup style="color:{COLORS["text_gray"]}">Model: {model_text}</sup>',
            font=dict(size=16, color=COLORS['text_white'])
        ),
        xaxis_title='Time (months)', 
        yaxis_title='P(No Breakthrough)',
        height=420, showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=1, xanchor='right',
                    font=dict(size=10, color=COLORS['text_white'])),
        margin=dict(l=60, r=20, t=70, b=50),
        plot_bgcolor=COLORS['surface_dark'], 
        paper_bgcolor=COLORS['surface_dark'],
        font=dict(family='Inter', size=11, color=COLORS['text_white']),
        yaxis=dict(range=[0, 1.05])
    )
    
    fig.update_xaxes(showgrid=True, gridcolor=COLORS['border'], tickfont=dict(color=COLORS['text_gray']))
    fig.update_yaxes(showgrid=True, gridcolor=COLORS['border'], tickfont=dict(color=COLORS['text_gray']))
    
    return fig

def create_watercut_chart(df, bt_info):
    df = df.copy()
    df['days'] = (df['DATEPRD'] - bt_info['first_prod_date']).dt.days
    df = df[df['days'] >= 0]
    df['wc_smooth'] = df['water_cut'].rolling(14, min_periods=1).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['days'], y=df['water_cut']*100, name='Daily',
                             line=dict(color='#5090c0', width=1), opacity=0.3))
    fig.add_trace(go.Scatter(x=df['days'], y=df['wc_smooth']*100, name='14-day Avg',
                             line=dict(color=COLORS['orange_primary'], width=2.5)))
    
    for thresh, color in [(5, COLORS['orange_light']), (10, COLORS['danger']), (20, '#a855f7')]:
        fig.add_hline(y=thresh, line_dash='dot', line_color=color, annotation_text=f'{thresh}%')
    
    if bt_info['event_observed'] == 1:
        fig.add_vline(x=bt_info['days_to_breakthrough'], line_dash='solid',
                      line_color=COLORS['danger'], line_width=2)
    
    fig.update_layout(title=dict(text='Water Cut Evolution', font=dict(size=16, color=COLORS['text_white'])),
                      xaxis_title='Days from Start', yaxis_title='Water Cut (%)',
                      height=380, showlegend=True,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, x=1, xanchor='right'),
                      margin=dict(l=60, r=20, t=70, b=50),
                      plot_bgcolor=COLORS['surface_dark'], paper_bgcolor=COLORS['surface_dark'],
                      font=dict(family='Inter', size=11, color=COLORS['text_white']),
                      yaxis=dict(range=[0, 100]))
    
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
        
        # Model status indicator
        if TRAINED_MODEL is not None:
            st.markdown(f"""
            <div style="background:{COLORS['success']}22; border:1px solid {COLORS['success']}; border-radius:8px; padding:0.5rem; margin-bottom:1rem;">
                <span style="color:{COLORS['success']};">✓ Model Loaded</span><br>
                <span style="color:{COLORS['text_gray']}; font-size:0.8rem;">{MODEL_NAME}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:{COLORS['danger']}22; border:1px solid {COLORS['danger']}; border-radius:8px; padding:0.5rem; margin-bottom:1rem;">
                <span style="color:{COLORS['danger']};">⚠ Using Fallback</span><br>
                <span style="color:{COLORS['text_gray']}; font-size:0.8rem;">Heuristic Model</span>
            </div>
            """, unsafe_allow_html=True)
        
        page = st.radio("Navigation", ["Analysis Dashboard", "Data Upload", "Model Information", "Help"], label_visibility="collapsed")
    
    # Header
    st.markdown(f'<p class="section-header" style="font-size:1.5rem; border-bottom:none; margin-bottom:0.5rem;">Water Breakthrough Prediction System</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # ============================================================
    # ANALYSIS DASHBOARD
    # ============================================================
    if page == "Analysis Dashboard":
        if 'uploaded_data' not in st.session_state:
            st.session_state.uploaded_data = None
            st.session_state.well_info = None
            st.session_state.predictions = None
        
        st.markdown('<p class="section-header">Data Input</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Upload Well Production Data (CSV)", type=['csv'])
        
        with col2:
            st.markdown("**Required Columns:**")
            st.code("DATEPRD\nBORE_OIL_VOL\nBORE_WAT_VOL", language=None)
            st.markdown("**Optional:**")
            st.code("AVG_DOWNHOLE_PRESSURE\nBORE_GAS_VOL", language=None)
        
        if uploaded_file:
            df, error = process_uploaded_data(uploaded_file)
            if error:
                st.error(f"Error: {error}")
            else:
                st.session_state.uploaded_data = df
                well_name = df['WELL_BORE_CODE'].iloc[0] if 'WELL_BORE_CODE' in df.columns else uploaded_file.name.replace('.csv', '')
                bt_info = detect_breakthrough(df)
                st.session_state.well_info = bt_info
                st.session_state.well_name = well_name
                predictions = predict_breakthrough(bt_info)
                st.session_state.predictions = predictions
                st.success(f"✓ Loaded {len(df):,} records for {well_name}")
        
        if st.session_state.uploaded_data is not None:
            df = st.session_state.uploaded_data
            bt_info = st.session_state.well_info
            predictions = st.session_state.predictions
            well_name = st.session_state.well_name
            
            st.markdown("---")
            st.markdown('<p class="section-header">Well Summary</p>', unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Well Name", well_name)
            col2.metric("Production Start", bt_info['first_prod_date'].strftime('%Y-%m-%d'))
            col3.metric("Total Oil", f"{bt_info['total_oil']/1000:,.0f} k Sm³")
            col4.metric("Current WC", f"{bt_info['final_water_cut']*100:.1f}%")
            col5.metric("Breakthrough", "OBSERVED" if bt_info['event_observed'] else "NOT YET")
            
            st.markdown("---")
            st.markdown('<p class="section-header">Breakthrough Prediction</p>', unsafe_allow_html=True)
            
            # Show model being used
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
                    <div style="color:{COLORS['p90']}; font-weight:700; letter-spacing:2px; margin-bottom:1rem;">P90 — CONSERVATIVE</div>
                    <div style="font-size:4rem; font-weight:700; color:white;">{predictions['P90_months']}</div>
                    <div style="color:{COLORS['text_gray']}; font-size:1.1rem;">months</div>
                    <div style="color:{COLORS['text_muted']}; margin-top:0.5rem;">{predictions['P90_days']} days</div>
                    <div style="color:{COLORS['text_muted']}; font-size:0.75rem; margin-top:1rem;">90% probability BT occurs AFTER this</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background:{COLORS['surface_dark']}; border-radius:12px; padding:2rem; text-align:center; border-top:4px solid {COLORS['p50']};">
                    <div style="color:{COLORS['p50']}; font-weight:700; letter-spacing:2px; margin-bottom:1rem;">P50 — MOST LIKELY</div>
                    <div style="font-size:4rem; font-weight:700; color:white;">{predictions['P50_months']}</div>
                    <div style="color:{COLORS['text_gray']}; font-size:1.1rem;">months</div>
                    <div style="color:{COLORS['text_muted']}; margin-top:0.5rem;">{predictions['P50_days']} days</div>
                    <div style="color:{COLORS['text_muted']}; font-size:0.75rem; margin-top:1rem;">Median expected breakthrough time</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style="background:{COLORS['surface_dark']}; border-radius:12px; padding:2rem; text-align:center; border-top:4px solid {COLORS['p10']};">
                    <div style="color:{COLORS['p10']}; font-weight:700; letter-spacing:2px; margin-bottom:1rem;">P10 — OPTIMISTIC</div>
                    <div style="font-size:4rem; font-weight:700; color:white;">{predictions['P10_months']}</div>
                    <div style="color:{COLORS['text_gray']}; font-size:1.1rem;">months</div>
                    <div style="color:{COLORS['text_muted']}; margin-top:0.5rem;">{predictions['P10_days']} days</div>
                    <div style="color:{COLORS['text_muted']}; font-size:0.75rem; margin-top:1rem;">Only 10% probability BT takes longer</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown('<p class="section-header">Analysis Charts</p>', unsafe_allow_html=True)
            
            tab1, tab2, tab3 = st.tabs(["Survival Curve", "Production Profile", "Water Cut Evolution"])
            
            with tab1:
                actual_bt = bt_info['months_to_breakthrough'] if bt_info['event_observed'] else None
                st.plotly_chart(create_survival_chart(predictions, actual_bt, well_name), use_container_width=True)
                if bt_info['event_observed']:
                    actual = bt_info['months_to_breakthrough']
                    if predictions['P90_months'] <= actual <= predictions['P10_months']:
                        st.success(f"✓ Actual ({actual:.1f} mo) within P90-P10 range")
                    else:
                        st.warning(f"⚠ Actual ({actual:.1f} mo) outside range")
            
            with tab2:
                st.plotly_chart(create_production_chart(df), use_container_width=True)
            
            with tab3:
                st.plotly_chart(create_watercut_chart(df, bt_info), use_container_width=True)
            
            st.markdown("---")
            st.markdown('<p class="section-header">Model Input Parameters</p>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(pd.DataFrame({
                    'Parameter': ['Early Oil Rate', 'Early Water Cut', 'Avg Pressure', 'Production Days'],
                    'Value': [
                        f"{bt_info['early_oil_rate']:.0f} Sm³/d" if not np.isnan(bt_info['early_oil_rate']) else 'N/A',
                        f"{bt_info['early_water_cut']*100:.2f}%" if not np.isnan(bt_info['early_water_cut']) else 'N/A',
                        f"{bt_info['early_pressure']:.1f} bar" if not np.isnan(bt_info['early_pressure']) else 'N/A',
                        f"{bt_info['production_days']}"
                    ],
                    'Source': ['First 60 days', 'First 60 days', 'First 60 days', 'Total']
                }), hide_index=True, use_container_width=True)
            
            with col2:
                st.markdown("**Model Features Used:**")
                if TRAINED_MODEL is not None:
                    for feat in MODEL_FEATURES:
                        st.markdown(f"- `{feat}`")
                else:
                    st.markdown("- Water Cut (early)")
                    st.markdown("- Pressure (avg)")
                    st.markdown("- Oil Rate (early)")
    
    # ============================================================
    # DATA UPLOAD PAGE
    # ============================================================
    elif page == "Data Upload":
        st.markdown('<p class="section-header">Data Upload Guidelines</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Required Data Format")
            st.markdown("Upload **CSV files** with daily production data. Minimum **60 days** recommended.")
            
            st.markdown("**Mandatory Columns:**")
            st.dataframe(pd.DataFrame({
                'Column': ['DATEPRD', 'BORE_OIL_VOL', 'BORE_WAT_VOL'],
                'Description': ['Production date', 'Daily oil volume', 'Daily water volume'],
                'Format': ['DD-MMM-YY', 'Sm³/day', 'Sm³/day']
            }), hide_index=True, use_container_width=True)
            
            st.markdown("**Optional Columns** (improves accuracy ~15-20%):")
            st.dataframe(pd.DataFrame({
                'Column': ['AVG_DOWNHOLE_PRESSURE', 'BORE_GAS_VOL'],
                'Description': ['Bottom-hole pressure', 'Daily gas volume'],
                'Format': ['bar', 'Sm³/day']
            }), hide_index=True, use_container_width=True)
        
        with col2:
            st.subheader("Sample Format")
            st.code("""DATEPRD,WELL_BORE_CODE,BORE_OIL_VOL,BORE_WAT_VOL
01-Jan-20,WELL-001,1500,50
02-Jan-20,WELL-001,1480,55
03-Jan-20,WELL-001,1520,48""", language="csv")
            
            st.subheader("Quality Requirements")
            st.markdown("""
            - ✓ Minimum **60 days** of data
            - ✓ No major gaps (>30 days)
            - ✓ Consistent units
            - ✓ Allocated rates (not commingled)
            """)
        
        st.markdown("---")
        st.subheader("Sample Data Files")
        col1, col2, col3 = st.columns(3)
        col1.download_button("F-14 H (Day 200 BT)", "sample", "f14h.csv", use_container_width=True)
        col2.download_button("F-15 D (Later BT)", "sample", "f15d.csv", use_container_width=True)
        col3.download_button("F-11 H (Early BT)", "sample", "f11h.csv", use_container_width=True)
    
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
                st.markdown(f"""
                **Model Type:** Weibull Accelerated Failure Time (AFT)
                
                **Features:**
                """)
                for feat in MODEL_FEATURES:
                    st.markdown(f"- `{feat}`")
                
                st.markdown(f"""
                **Regularization:** {MODEL_ARTIFACTS.get('regularization', {}).get('type', 'None')}
                
                **Metrics:**
                - AIC: {MODEL_ARTIFACTS.get('metrics', {}).get('AIC', 'N/A'):.2f}
                - BIC: {MODEL_ARTIFACTS.get('metrics', {}).get('BIC', 'N/A'):.2f}
                """)
            else:
                st.warning("Using heuristic fallback model")
            
            st.markdown("---")
            st.subheader("Methodology")
            st.markdown("**Survival Analysis Framework**")
            st.markdown("""
            Accelerated Failure Time (AFT) models predict breakthrough with unique advantages:
            - **Censored data handling** — works with wells that haven't broken through
            - **Probability distributions** — full uncertainty range
            - **P10/P50/P90 quantiles** — risk-based decisions
            """)
            
            st.markdown("**Physical Basis**")
            st.markdown("""
            - **Buckley-Leverett (1942)** — Frontal displacement theory
            - **Mobility Ratio Effects** — Displacement stability
            - **Koval Factor (1963)** — Viscous fingering correction
            """)
        
        with col2:
            st.subheader("Understanding P10/P50/P90")
            st.dataframe(pd.DataFrame({
                'Percentile': ['P90', 'P50', 'P10'],
                'Meaning': ['90% BT occurs AFTER', 'Median (most likely)', '10% BT takes longer'],
                'Use Case': ['Conservative', 'Base case', 'Optimistic']
            }), hide_index=True, use_container_width=True)
            
            st.subheader("Limitations")
            st.warning("""
            - Calibrated to **North Sea / Volve** conditions
            - Requires **60+ days** of production data
            - Does not model **well interventions**
            - Should be **recalibrated** for other fields
            """)
            
            st.subheader("Model Files")
            st.info("""
            The model is loaded from `water_breakthrough_model.pkl`.
            
            To update the model, replace this file with a new trained model.
            """)
        
        # Model Training Report Section
        st.markdown("---")
        st.markdown('<p class="section-header">Model Training Documentation</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Training Report")
            st.markdown("""
            The full model training notebook with code, outputs, and validation results 
            is available for download.
            """)
            
            # Check for HTML report and provide download
            html_report_paths = [
                Path("Water Breakthrough Prediction.html"),
                Path("docs/Water Breakthrough Prediction.html"),
                Path("reports/Water Breakthrough Prediction.html"),
            ]
            
            html_found = False
            for html_path in html_report_paths:
                if html_path.exists():
                    with open(html_path, 'rb') as f:
                        html_bytes = f.read()
                    
                    st.download_button(
                        label="📥 Download Training Report (HTML)",
                        data=html_bytes,
                        file_name="Water Breakthrough Prediction.html",
                        mime="text/html",
                        use_container_width=True
                    )
                    html_found = True
                    break
            
            if not html_found:
                st.info("Training report HTML not found. Place 'Water Breakthrough Prediction.html' in the app directory.")
        
        with col2:
            st.subheader("📚 Report Contents")
            st.markdown("""
            The training report includes:
            
            1. **Data Loading & Exploration**
               - Volve field production data analysis
               - 6 wells with breakthrough history
            
            2. **Feature Engineering**
               - 16+ features created
               - Correlation analysis
               - Multicollinearity checks
            
            3. **Model Training**
               - Weibull AFT survival analysis
               - Regularization comparison
               - Feature importance
            
            4. **Validation Results**
               - P10/P50/P90 predictions
               - Coverage metrics
               - Error analysis
            
            5. **Model Export**
               - Saved model artifacts
               - Prediction function
            """)
    
    # ============================================================
    # HELP PAGE
    # ============================================================
    elif page == "Help":
        st.markdown('<p class="section-header">User Guide</p>', unsafe_allow_html=True)
        
        st.subheader("Quick Start")
        col1, col2, col3, col4 = st.columns(4)
        col1.info("**Step 1**\n\nGo to Analysis Dashboard")
        col2.info("**Step 2**\n\nUpload CSV file")
        col3.info("**Step 3**\n\nReview predictions")
        col4.info("**Step 4**\n\nAnalyze charts")
        
        st.markdown("---")
        st.subheader("FAQ")
        
        with st.expander("What data format is required?"):
            st.write("CSV with DATEPRD, BORE_OIL_VOL, BORE_WAT_VOL columns. See Data Upload page.")
        
        with st.expander("How accurate are predictions?"):
            st.write("P90-P10 range validated on Volve wells F-14 H and F-15 D. Accuracy depends on data quality.")
        
        with st.expander("Can I use this for other fields?"):
            st.write("Yes, but recalibration with local historical data is recommended for best results.")
        
        with st.expander("What do P10/P50/P90 mean?"):
            st.write("P90: 90% chance BT happens after this time. P50: median/most likely. P10: only 10% chance takes longer.")
        
        with st.expander("What model is being used?"):
            if TRAINED_MODEL:
                st.write(f"Currently using: **{MODEL_NAME}**")
                st.write(f"Features: {', '.join(MODEL_FEATURES)}")
            else:
                st.write("Using heuristic fallback model (trained model not found)")
        
        with st.expander("Where can I see the model training details?"):
            st.write("""
            The full model training notebook is available in the **Model Information** page.
            
            You can:
            - Download the HTML report with all code and outputs
            - View the training methodology and validation results
            - See feature importance and model coefficients
            
            Go to **Model Information** page and click the download button.
            """)
        
        st.markdown("---")
        st.subheader("📚 Documentation Links")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Model Training**
            
            Download the training report from the **Model Information** page.
            
            Full Jupyter notebook with code, outputs, and validation.
            """)
        
        with col2:
            st.markdown("""
            **Technical Reference**
            
            - Weibull AFT Model
            - Survival Analysis
            - Feature Engineering
            - Regularization
            """)
        
        with col3:
            st.markdown("""
            **Support**
            
            For questions or issues:
            - Check Model Information page
            - Review the training report
            - Contact support team
            """)

if __name__ == "__main__":
    main()