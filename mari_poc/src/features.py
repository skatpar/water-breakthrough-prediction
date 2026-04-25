"""
Feature engineering functions for MARI Water Breakthrough Prediction POC.

All derivation logic lives here; notebooks import and call these functions.
"""

import pandas as pd
import numpy as np
from .config import (WGR_THRESHOLD_BBL_MMCF, BT_SUSTAINED_MONTHS, EXCLUDED_WELLS,
                     GWC_RKB_M)


# ── Breakthrough detection ───────────────────────────────────────────────────

def compute_wgr(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add a water-gas ratio column (bbl/MMcf) to the panel.
    WGR is undefined (NaN) when gas_mmcf <= 0 or is null.
    """
    df = panel.copy()
    df["wgr_bbl_mmcf"] = np.where(
        df["gas_mmcf"] > 0,
        df["water_bbl"] / df["gas_mmcf"],
        np.nan,
    )
    return df


def detect_breakthrough(
    panel: pd.DataFrame,
    threshold: float = WGR_THRESHOLD_BBL_MMCF,
    sustained: int = BT_SUSTAINED_MONTHS,
) -> pd.DataFrame:
    """
    For each well, detect water breakthrough using the rule:
        WGR > threshold sustained for >= `sustained` consecutive months.

    Parameters
    ----------
    panel : DataFrame with columns [well, date, gas_mmcf, water_bbl]
        Must be sorted by (well, date).
    threshold : float
        WGR threshold in bbl/MMcf (default 5).
    sustained : int
        Minimum consecutive months above threshold (default 3).

    Returns
    -------
    DataFrame with one row per well:
        well            : str
        bt_detected     : bool — True if breakthrough detected
        bt_date         : datetime or NaT — first month of the sustained run
        bt_month_index  : int or NA — 0-indexed month from first_prod_date
    """
    if "wgr_bbl_mmcf" not in panel.columns:
        panel = compute_wgr(panel)

    results = []
    for well, grp in panel.groupby("well"):
        grp = grp.sort_values("date").reset_index(drop=True)
        above = (grp["wgr_bbl_mmcf"] > threshold).astype(int)

        # Find runs of consecutive True
        bt_detected = False
        bt_date = pd.NaT
        bt_month_index = pd.NA

        run_start = None
        run_len = 0
        for i, val in enumerate(above):
            if val == 1:
                if run_start is None:
                    run_start = i
                run_len += 1
                if run_len >= sustained:
                    bt_detected = True
                    bt_date = grp.loc[run_start, "date"]
                    # Month index from first production
                    first_gas = grp.loc[grp["gas_mmcf"] > 0, "date"]
                    if len(first_gas) > 0:
                        first = first_gas.iloc[0]
                        bt_month_index = (
                            bt_date.to_period("M").ordinal
                            - first.to_period("M").ordinal
                        )
                    break
            else:
                run_start = None
                run_len = 0

        results.append({
            "well": well,
            "bt_detected": bt_detected,
            "bt_date": bt_date,
            "bt_month_index": bt_month_index,
        })

    return pd.DataFrame(results)


# ── Temporal feature derivation ──────────────────────────────────────────────

def _months_from_first(dates: pd.Series, first: pd.Timestamp) -> pd.Series:
    """Return integer months elapsed since `first`."""
    return (dates.dt.to_period("M").astype(int) - first.to_period("M").ordinal)


def compute_rate_features(panel: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    """
    Per-well rate-based summary features from the production panel.

    Returns DataFrame with columns:
        well, mean_gas_yr1, mean_gas_yr2, mean_gas_yr3,
        cum_gas_yr1, cum_gas_yr2, cum_gas_yr5,
        peak_gas, time_to_peak_months, arps_di
    """
    records = []
    for well, grp in panel.groupby("well"):
        grp = grp.sort_values("date")
        gas = grp.dropna(subset=["gas_mmcf"])
        if len(gas) == 0:
            records.append({"well": well})
            continue

        first = gas["date"].iloc[0]
        gas = gas.copy()
        gas["month_idx"] = _months_from_first(gas["date"], first)

        rec = {"well": well}

        # Mean gas in years 1/2/3
        for yr, label in [(1, "mean_gas_yr1"), (2, "mean_gas_yr2"), (3, "mean_gas_yr3")]:
            mask = (gas["month_idx"] >= (yr - 1) * 12) & (gas["month_idx"] < yr * 12)
            vals = gas.loc[mask, "gas_mmcf"]
            rec[label] = vals.mean() if len(vals) > 0 else np.nan

        # Cumulative gas at year 1/2/5
        for yr, label in [(1, "cum_gas_yr1"), (2, "cum_gas_yr2"), (5, "cum_gas_yr5")]:
            mask = gas["month_idx"] < yr * 12
            rec[label] = gas.loc[mask, "gas_mmcf"].sum() if mask.any() else np.nan

        # Peak gas rate and time to peak
        peak_idx = gas["gas_mmcf"].idxmax()
        rec["peak_gas"] = gas.loc[peak_idx, "gas_mmcf"]
        rec["time_to_peak_months"] = gas.loc[peak_idx, "month_idx"]

        # Arps exponential decline rate from mid-life
        # Use months 24–120 (years 2–10) if available
        mid = gas[(gas["month_idx"] >= 24) & (gas["month_idx"] <= 120)]
        mid = mid[mid["gas_mmcf"] > 0]
        if len(mid) >= 6:
            ln_q = np.log(mid["gas_mmcf"].values)
            t = mid["month_idx"].values.astype(float)
            # linear fit: ln(q) = ln(q0) - Di*t
            try:
                coeffs = np.polyfit(t, ln_q, 1)
                rec["arps_di"] = -coeffs[0]  # monthly decline rate
            except:
                rec["arps_di"] = np.nan
        else:
            rec["arps_di"] = np.nan

        records.append(rec)

    return pd.DataFrame(records)


def compute_pressure_features(panel: pd.DataFrame, static: pd.DataFrame, bt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-well pressure-based summary features.

    Returns DataFrame with columns:
        well, initial_whfp, whfp_decline_rate, whfp_at_bt, drawdown_proxy, whfp_std
    """
    records = []
    for well, grp in panel.groupby("well"):
        grp = grp.sort_values("date")
        whfp = grp.dropna(subset=["whfp_psig"])
        if len(whfp) == 0:
            records.append({"well": well})
            continue

        first_gas = grp.loc[grp["gas_mmcf"] > 0, "date"]
        if len(first_gas) == 0:
            records.append({"well": well})
            continue
        first = first_gas.iloc[0]

        whfp = whfp.copy()
        whfp["month_idx"] = _months_from_first(whfp["date"], first)

        rec = {"well": well}

        # Initial WHFP: mean of first 3 months
        early = whfp[whfp["month_idx"].between(0, 2)]
        rec["initial_whfp"] = early["whfp_psig"].mean() if len(early) > 0 else np.nan

        # WHFP decline rate (annualized): linear fit over full range
        if len(whfp) >= 6:
            t = whfp["month_idx"].values.astype(float)
            p = whfp["whfp_psig"].values
            try:
                coeffs = np.polyfit(t, p, 1)
                rec["whfp_decline_rate"] = -coeffs[0] * 12  # psig/year
            except:
                rec["whfp_decline_rate"] = np.nan
        else:
            rec["whfp_decline_rate"] = np.nan

        # WHFP at breakthrough (or last observation for dry wells)
        bt_row = bt_df[bt_df["well"] == well]
        if len(bt_row) > 0 and bt_row.iloc[0]["bt_detected"]:
            bt_date = bt_row.iloc[0]["bt_date"]
            at_bt = whfp[whfp["date"] <= bt_date].tail(1)
            rec["whfp_at_bt"] = at_bt["whfp_psig"].iloc[0] if len(at_bt) > 0 else np.nan
        else:
            rec["whfp_at_bt"] = whfp["whfp_psig"].iloc[-1]

        # Drawdown proxy: initial WHFP - WHFP at month 24
        m24 = whfp[whfp["month_idx"].between(22, 26)]
        if len(m24) > 0 and not np.isnan(rec.get("initial_whfp", np.nan)):
            rec["drawdown_proxy"] = rec["initial_whfp"] - m24["whfp_psig"].mean()
        else:
            rec["drawdown_proxy"] = np.nan

        # WHFP volatility
        if len(whfp) >= 12:
            early_whfp = whfp[whfp["month_idx"] < 24]
            rec["whfp_std"] = early_whfp["whfp_psig"].std() if len(early_whfp) >= 3 else np.nan
        else:
            rec["whfp_std"] = np.nan

        records.append(rec)

    return pd.DataFrame(records)


def compute_field_state_features(panel: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    """
    Per-well field-state features at the time of spud (first production).

    Returns DataFrame with columns:
        well, cum_field_gas_at_spud, active_wells_at_spud, cum_field_water_at_spud
    """
    records = []
    first_dates = static.set_index("well")["first_prod_date"]

    for well in static["well"]:
        spud = first_dates[well]
        rec = {"well": well}

        # All other wells' production before this well's spud
        other = panel[(panel["well"] != well) & (panel["date"] < spud)]
        rec["cum_field_gas_at_spud"] = other["gas_mmcf"].sum()
        rec["cum_field_water_at_spud"] = other["water_bbl"].sum()

        # Number of wells actively producing at spud
        # "active" = produced gas in the month of spud or the month before
        near_spud = panel[
            (panel["well"] != well)
            & (panel["date"].between(spud - pd.DateOffset(months=1), spud))
            & (panel["gas_mmcf"] > 0)
        ]
        rec["active_wells_at_spud"] = near_spud["well"].nunique()

        records.append(rec)

    return pd.DataFrame(records)


def compute_volatility_features(panel: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    """
    Per-well volatility features from early production.

    Returns DataFrame with columns:
        well, gas_cov_2yr
    """
    records = []
    first_dates = static.set_index("well")["first_prod_date"]

    for well, grp in panel.groupby("well"):
        grp = grp.sort_values("date")
        gas = grp.dropna(subset=["gas_mmcf"])
        if len(gas) == 0:
            records.append({"well": well})
            continue

        first = first_dates.get(well)
        if first is None:
            records.append({"well": well})
            continue

        gas = gas.copy()
        gas["month_idx"] = _months_from_first(gas["date"], first)
        early = gas[gas["month_idx"] < 24]

        rec = {"well": well}
        if len(early) >= 6 and early["gas_mmcf"].mean() > 0:
            rec["gas_cov_2yr"] = early["gas_mmcf"].std() / early["gas_mmcf"].mean()
        else:
            rec["gas_cov_2yr"] = np.nan

        records.append(rec)

    return pd.DataFrame(records)


# ── Distance to GWC ─────────────────────────────────────────────────────────

def compute_d_to_gwc(static: pd.DataFrame) -> pd.DataFrame:
    """
    Compute distance from bottom perforation to GWC (metres).

    d_to_gwc = GWC_RKB - bottom_perf_md   (for verticals where MD ≈ TVD)

    For horizontals, bottom_perf_md is measured depth along the lateral,
    NOT TVD — so d_to_gwc is physically meaningless. Returns NaN for
    horizontals.

    Returns DataFrame with columns: well, d_to_gwc_m
    """
    records = []
    for _, row in static.iterrows():
        well = row["well"]
        if row.get("is_horizontal", False):
            records.append({"well": well, "d_to_gwc_m": np.nan})
        elif pd.notna(row.get("bottom_perf_md")):
            d = GWC_RKB_M - row["bottom_perf_md"]
            records.append({"well": well, "d_to_gwc_m": d})
        else:
            records.append({"well": well, "d_to_gwc_m": np.nan})
    return pd.DataFrame(records)
