"""
Data loaders for MARI Water Breakthrough Prediction POC.

Each function reads one raw source file, cleans column names, applies the
well-name map, parses dates, and returns a tidy DataFrame.  No imputation
or derived computations are performed.
"""

import pandas as pd
import numpy as np
from .config import RAW_DIR, canonicalize_well


# ── Helpers ──────────────────────────────────────────────────────────────────

def _strip_all_str_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from every object column."""
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_rates() -> pd.DataFrame:
    """
    Load production rates from STIXOR Sharing Data.xlsx → 'Rates' sheet.

    Returns
    -------
    DataFrame with columns:
        well        : str   — canonical well name
        date        : datetime64 — month start
        choke_64ths : float — choke setting in 1/64ths
        prod_days   : float — days on production that month
        gas_mmcf    : float — monthly gas produced, MMcf
        water_bbl   : float — monthly water produced, bbl

    Quirks
    ------
    - Header is on row 1 (row 0 is blank).
    - Column names have heavy leading whitespace.
    - Date column is '      Date ' with leading spaces.
    - 'Notes' and an unnamed column are present but dropped.
    """
    path = RAW_DIR / "STIXOR Sharing Data.xlsx"
    df = pd.read_excel(path, sheet_name="Rates", header=1)

    # Normalise column names
    df.columns = df.columns.str.strip()

    rename = {
        "Date":                               "date",
        "Wells":                              "well",
        'CHOKE 1/64"':                        "choke_64ths",
        "DAYS on production":                 "prod_days",
        "Monthly Gas Produced MMcf (volume)": "gas_mmcf",
        "Monthly Water Produced bbl (volume)":"water_bbl",
    }
    df = df.rename(columns=rename)

    # Keep only the columns we need
    keep = ["well", "date", "choke_64ths", "prod_days", "gas_mmcf", "water_bbl"]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Clean strings, map wells
    df = _strip_all_str_columns(df)
    df = df.dropna(subset=["well"])
    df["well"] = df["well"].apply(canonicalize_well)

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Numeric coercion
    for col in ["choke_64ths", "prod_days", "gas_mmcf", "water_bbl"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values(["well", "date"]).reset_index(drop=True)


def load_pressures_whfp() -> pd.DataFrame:
    """
    Load wellhead flowing pressure and line pressure from
    STIXOR Sharing Data.xlsx → 'Pressures' sheet.

    Returns
    -------
    DataFrame with columns:
        well               : str
        date               : datetime64
        whfp_psig          : float — wellhead flowing pressure
        line_pressure_psig : float — line pressure

    Quirks
    ------
    - Header on row 1.
    - Column names have leading whitespace and trailing spaces.
    """
    path = RAW_DIR / "STIXOR Sharing Data.xlsx"
    df = pd.read_excel(path, sheet_name="Pressures", header=1)

    df.columns = df.columns.str.strip()

    rename = {
        "Date":                                   "date",
        "Wells":                                  "well",
        "Wellhead Flowing Pressure (WHFP)psig":   "whfp_psig",
        "Line Pressure psig":                     "line_pressure_psig",
    }
    df = df.rename(columns=rename)
    keep = ["well", "date", "whfp_psig", "line_pressure_psig"]
    df = df[[c for c in keep if c in df.columns]].copy()

    df = _strip_all_str_columns(df)
    df = df.dropna(subset=["well"])
    df["well"] = df["well"].apply(canonicalize_well)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in ["whfp_psig", "line_pressure_psig"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values(["well", "date"]).reset_index(drop=True)


def load_bhp() -> pd.DataFrame:
    """
    Load annual bottomhole pressure from BHP.csv.

    Returns
    -------
    DataFrame with columns:
        well     : str
        date     : datetime64
        bhp_psig : float

    Quirks
    ------
    - Well names are bare numbers (11, 122H, E-2) — mapped to canonical form.
    - Date strings are in mixed formats (e.g. '1-Nov-88', '01-Jul-23').
    - 315 records, one null BHP (M-126H-HRL).
    """
    path = RAW_DIR / "BHP.csv"
    df = pd.read_csv(path)
    df = _strip_all_str_columns(df)
    df.columns = df.columns.str.strip()

    df = df.rename(columns={"well": "well", "date": "date", "bhp": "bhp_psig"})
    df["well"] = df["well"].apply(canonicalize_well)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["bhp_psig"] = pd.to_numeric(df["bhp_psig"], errors="coerce")

    return df.sort_values(["well", "date"]).reset_index(drop=True)


def load_bhp_pressure_xlsx() -> pd.DataFrame:
    """
    Load BHP records from Pressure data- Stixors Technologies.xlsx → 'Sheet2'.

    This file has a multi-column layout: three groups of (Well#, Date, Pressure)
    side by side in columns 2-4, 6-8, 10-12 (0-indexed).  Also contains a gas
    gravity section (cols 14-15) and a GWC note (col 17); those are NOT returned
    here — use load_gas_gravity_pressure_xlsx() for that.

    Returns
    -------
    DataFrame with columns:
        well     : str
        date     : datetime64
        bhp_psig : float

    Quirks
    ------
    - Header is on row 2 (rows 0-1 are blank).
    - Well names are bare numbers.
    - 21 wells (M-126H-HRL absent from this file).
    - 314 records (one fewer than BHP.csv).
    """
    path = RAW_DIR / "Pressure data- Stixors Technologies.xlsx"
    raw = pd.read_excel(path, sheet_name="Sheet2", header=None)

    frames = []
    for cols in [(2, 3, 4), (6, 7, 8), (10, 11, 12)]:
        g = raw.iloc[3:, list(cols)].copy()
        g.columns = ["well", "date", "bhp_psig"]
        g = g.dropna(how="all")
        frames.append(g)

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["bhp_psig"])

    # Clean
    df["well"] = df["well"].astype(str).str.strip()
    df["well"] = df["well"].apply(canonicalize_well)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["bhp_psig"] = pd.to_numeric(df["bhp_psig"], errors="coerce")

    return df.sort_values(["well", "date"]).reset_index(drop=True)


def load_gas_gravity() -> pd.DataFrame:
    """
    Load per-well gas specific gravity from gas_gravity.csv.

    Returns
    -------
    DataFrame with columns:
        well        : str
        gas_gravity : float

    Quirks
    ------
    - Values have a tab character before the number — stripped via numeric coercion.
    - Well names are already in M-XX-HRL canonical form.
    - 22 rows, range 0.685–0.803.
    """
    path = RAW_DIR / "gas_gravity.csv"
    df = pd.read_csv(path)
    df = _strip_all_str_columns(df)
    df.columns = df.columns.str.strip()

    df["well"] = df["well"].apply(canonicalize_well)
    df["gas_gravity"] = pd.to_numeric(
        df["gas_gravity"].astype(str).str.strip(), errors="coerce"
    )
    return df.reset_index(drop=True)


def load_gas_gravity_pressure_xlsx() -> pd.DataFrame:
    """
    Load gas gravity from the Pressure xlsx Sheet2 (cols 14-15).

    Returns the same schema as load_gas_gravity() but with slightly more
    precise values (e.g. 0.769 vs 0.77).  22 rows.
    """
    path = RAW_DIR / "Pressure data- Stixors Technologies.xlsx"
    raw = pd.read_excel(path, sheet_name="Sheet2", header=None)
    df = raw.iloc[3:, [14, 15]].dropna(how="all").copy()
    df.columns = ["well", "gas_gravity"]
    df["well"] = df["well"].astype(str).str.strip()
    df["well"] = df["well"].apply(canonicalize_well)
    df["gas_gravity"] = pd.to_numeric(df["gas_gravity"], errors="coerce")
    return df.reset_index(drop=True)


def load_subsurface() -> pd.DataFrame:
    """
    Load static rock/completion properties from
    Subsurface data- Stixors Technologies.xlsx → 'Additional Data shared'.

    Returns
    -------
    DataFrame with columns:
        well            : str
        porosity        : float
        permeability_md : float  — PTA-derived
        skin            : float  — NaN for horizontals
        sw              : float  — water saturation
        net_pay_m       : float  — measured depth net pay
        chlorides_ppm   : float
        top_perf_md     : float
        bottom_perf_md  : float

    Quirks
    ------
    - Header on row 2 (rows 0-1 are blank NaN rows).
    - Net pay column header contains a newline character.
    - 22 rows. Well names already in canonical form.
    - Horizontals have NaN skin and very large net_pay values (475–802 m)
      vs verticals (10–15 m) — these are different physical quantities!
    - M-122H/124H/126H share identical (porosity, perm, Sw) = (0.22, 34, 0.46),
      likely analog-copied.
    """
    path = RAW_DIR / "Subsurface data- Stixors Technologies.xlsx"
    df = pd.read_excel(path, sheet_name="Additional Data shared", header=2)

    # Drop unnamed columns
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    rename = {
        "Wells":                        "well",
        "Porosity":                     "porosity",
        "Permeability (mD) - PTA":      "permeability_md",
        "Skin":                         "skin",
        "Water saturation":             "sw",
        "Net pay (m) \nMeasured Depth": "net_pay_m",
        "Chlorides (ppm)":              "chlorides_ppm",
        "Top Perf (m)":                 "top_perf_md",
        "Bottom Perf (m)":              "bottom_perf_md",
    }
    df = df.rename(columns=rename)
    keep = list(rename.values())
    df = df[[c for c in keep if c in df.columns]].copy()

    df = _strip_all_str_columns(df)
    df = df.dropna(subset=["well"])
    df["well"] = df["well"].apply(canonicalize_well)

    for col in keep[1:]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)
