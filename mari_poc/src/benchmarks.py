"""
Benchmark models for MARI Water Breakthrough Prediction POC.

Independent sanity-check methods: Kaplan-Meier, Sobocinski-Cornelius,
Papatzacos, and simple OLS.
"""

import pandas as pd
import numpy as np
from typing import Optional


def kaplan_meier_median(
    event_times: np.ndarray,
    observed: np.ndarray,
) -> float:
    """
    Compute Kaplan-Meier median survival time.

    Parameters
    ----------
    event_times : array of float — time to event (or censoring)
    observed : array of bool/int — 1 if event observed, 0 if censored

    Returns
    -------
    Median survival time (float), or np.inf if survival never drops below 0.5.
    """
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    kmf.fit(event_times, event_observed=observed)
    median = kmf.median_survival_time_
    return median


def sobocinski_cornelius_bt_time(
    k_h: float,
    h: float,
    h_p: float,
    q: float,
    delta_rho: float = 0.7,
    mu_w: float = 0.5,
    phi: float = 0.20,
    k_v_ratio: float = 0.1,
    re: float = 500.0,
    rw: float = 0.1,
) -> float:
    """
    Sobocinski-Cornelius (1965) water coning breakthrough time for a vertical well.

    Uses the dimensionless-time approach from Ahmed's Reservoir Engineering
    Handbook (Chapter 9).  All inputs are metric except permeability (mD).

    Workflow
    --------
    1. Compute critical coning rate q_c (Muskat-Wyckoff, Darcy units → bbl/d).
    2. Compute dimensionless rate ratio q_D = q_actual / q_c.
    3. Compute dimensionless breakthrough time t_D from S-C correlation.
    4. Convert t_D to real time.

    Parameters
    ----------
    k_h  : horizontal permeability, mD
    h    : net pay thickness, m
    h_p  : perforated interval, m
    q    : gas production rate, MMcf/month (surface)
    delta_rho : water−gas density difference at reservoir conditions, g/cc
    mu_w : water viscosity, cp
    phi  : porosity, fraction
    k_v_ratio : k_v / k_h (default 0.1 — flagged as assumption)
    re   : drainage radius, m
    rw   : wellbore radius, m

    Returns
    -------
    Estimated breakthrough time in months.
    """
    # Convert metric inputs to oilfield (ft)
    h_ft = h * 3.2808
    h_p_ft = h_p * 3.2808
    re_ft = re * 3.2808
    rw_ft = rw * 3.2808

    k_v = k_h * k_v_ratio  # mD

    # ---- Gas rate: MMcf/month → reservoir bbl/day ----
    # Typical Mari HRL conditions: ~3000 psi initial, ~250°F
    # Bg ≈ 0.9 Mscf/res bbl  →  1 res bbl ≈ 0.9 Mscf (= 900 scf)
    # q [MMcf/mo] * 1e3 [Mscf/MMcf] / 0.9 [Mscf/resbbl] / 30 [d/mo]
    Bg_Mscf_per_resbbl = 0.9
    q_rbpd = q * 1e3 / Bg_Mscf_per_resbbl / 30.0

    # ---- Muskat-Wyckoff critical rate (field units: bbl/day) ----
    # q_c = 0.0246e-4 * Δρ_lbft3 * k_h * h_ft^2 * (1 - (hp/h)^2) / (μ_w * Bo * ln(re/rw))
    # For water: Bo ≈ 1.0.  Δρ in lb/ft³ = Δρ_gcc * 62.428
    delta_rho_lbft3 = delta_rho * 62.428
    h_ratio = h_p_ft / h_ft if h_ft > 0 else 1.0
    ln_re_rw = np.log(re_ft / rw_ft) if rw_ft > 0 else np.log(5000)

    q_c = (0.0246e-4 * delta_rho_lbft3 * k_h * h_ft**2
           * (1 - h_ratio**2) / (mu_w * ln_re_rw))

    if q_c <= 0:
        return np.inf

    # ---- Dimensionless rate ----
    q_D = q_rbpd / q_c

    if q_D <= 0:
        return np.inf

    # ---- Sobocinski-Cornelius dimensionless BT time ----
    # Their Fig-3 correlation (simplified fit):
    #   t_D_BT ≈ (1 − h_ratio)^2 / (3 * sqrt(q_D))  for q_D > ~1
    # For very low q_D (sub-critical), t_D → large; for high q_D, t_D → small.
    alpha = 1.0 - h_ratio
    if alpha <= 0:
        return 0.0  # fully penetrating → instant breakthrough
    t_D = alpha**2 / (3.0 * np.sqrt(max(q_D, 0.01)))

    # ---- Convert to real time (days, then months) ----
    # t = t_D * φ * μ_w * h^2 / (k_v * Δρ * 0.006328)  [days, field units]
    if k_v <= 0:
        return np.inf

    t_days = t_D * phi * mu_w * h_ft**2 / (k_v * delta_rho_lbft3 * 0.006328)
    t_months = t_days / 30.44

    return max(t_months, 0)


def papatzacos_horizontal_bt_time(
    k_h: float,
    h: float,
    L: float,
    q: float,
    delta_rho: float = 0.7,
    mu_w: float = 0.5,
    phi: float = 0.20,
    k_v_ratio: float = 0.1,
    d_below: Optional[float] = None,
) -> float:
    """
    Papatzacos et al. (1991, SPE-19822) water cresting breakthrough time
    for a horizontal well.

    Parameters
    ----------
    k_h  : horizontal permeability, mD
    h    : pay thickness (formation height above GWC), m
    L    : horizontal well completed length, m
    q    : gas rate, MMcf/month
    delta_rho : density difference, g/cc
    mu_w : water viscosity, cp
    phi  : porosity
    k_v_ratio : k_v / k_h
    d_below : distance from well to GWC, m  (default h/2)

    Returns
    -------
    Estimated breakthrough time in months.
    """
    h_ft = h * 3.2808
    L_ft = L * 3.2808
    k_v = k_h * k_v_ratio

    if d_below is None:
        d_below = h / 2.0
    d_ft = d_below * 3.2808

    delta_rho_lbft3 = delta_rho * 62.428

    # Gas rate → reservoir bbl/day
    Bg_Mscf_per_resbbl = 0.9
    q_rbpd = q * 1e3 / Bg_Mscf_per_resbbl / 30.0

    # ---- Papatzacos critical rate for horizontal well (field units) ----
    # q_c = 0.0246e-4 * Δρ * k_h * h * d * L / (μ_w * h)   [bbl/d]
    # Simplified from Joshi/Papatzacos — uses the fact that horizontal wells
    # have a line-source geometry that delays cresting.
    q_c = 0.0246e-4 * delta_rho_lbft3 * k_h * d_ft * L_ft / mu_w

    if q_c <= 0:
        return np.inf

    q_D = q_rbpd / q_c

    if q_D <= 0:
        return np.inf

    # ---- Dimensionless BT time (Papatzacos Fig. 7 fit) ----
    h_D = d_ft / h_ft if h_ft > 0 else 0.5
    t_D = h_D**2 / (4.0 * max(q_D, 0.01))

    # ---- Real time ----
    if k_v <= 0:
        return np.inf

    t_days = t_D * phi * mu_w * h_ft**2 / (k_v * delta_rho_lbft3 * 0.006328)
    t_months = t_days / 30.44

    return max(t_months, 0)


def gwc_rise_bt_time(
    cum_gas_at_bt: float,
    phi: float,
    A: float,
    Sg: float,
    d_to_gwc: float,
) -> float:
    """
    Simple material-balance estimate: how long until cumulative gas withdrawal
    causes enough GWC rise to reach the perforations?

    Model
    -----
    Gas withdrawal → pressure drop → aquifer influx → GWC rises.
    Assume 1 bbl of reservoir void from gas withdrawal is replaced by 1 bbl
    of water influx (strong aquifer, which Mari appears to have).

    GWC rise = cum_water_influx / (φ * Sg * A)
    Breakthrough when rise ≥ d_to_gwc.

    Parameters
    ----------
    cum_gas_at_bt : float — cumulative gas produced by the well at estimated
                    breakthrough, MMcf. Use cum_gas_yr1 * (TTE/12) as proxy.
    phi : float — porosity
    A : float — drainage area, m² (default use pi * 500² ≈ 785,000 m²)
    Sg : float — gas saturation = 1 - Sw
    d_to_gwc : float — distance from bottom perforation to GWC, m

    Returns
    -------
    Implied GWC rise in meters from the given cumulative production.
    """
    # Convert MMcf to reservoir m³
    # 1 MMcf = 28,317 m³ at surface; at reservoir Bg ~0.005 → 141.6 res m³/MMcf
    Bg_resm3_per_MMcf = 141.6
    void_m3 = cum_gas_at_bt * Bg_resm3_per_MMcf

    # GWC rise = void / (φ * (1-Sw) * A)  [m]
    pore_volume_per_m = phi * Sg * A  # m³ per metre of rise
    if pore_volume_per_m <= 0:
        return 0.0

    rise_m = void_m3 / pore_volume_per_m
    return rise_m


def ols_log_tte(X: np.ndarray, y: np.ndarray):
    """
    Simple OLS regression of log(time-to-event) against features.

    Parameters
    ----------
    X : ndarray (n, p) — feature matrix
    y : ndarray (n,) — time-to-event in months

    Returns
    -------
    dict with keys: coefficients, r_squared, predictions
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    log_y = np.log(y)
    model = LinearRegression()
    model.fit(X, log_y)

    preds_log = model.predict(X)
    preds = np.exp(preds_log)

    return {
        "coefficients": model.coef_,
        "intercept": model.intercept_,
        "r_squared": r2_score(log_y, preds_log),
        "predictions": preds,
    }
