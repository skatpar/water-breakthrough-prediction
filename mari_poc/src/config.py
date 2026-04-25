"""
Configuration for MARI Water Breakthrough Prediction POC.

Contains well-name reconciliation map, file paths, constants, and color palette.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

# ── Well-name reconciliation ────────────────────────────────────────────────
# Maps every variant observed in any source file to the canonical name.
# Canonical form: M-XX-HRL (matching subsurface / gas_gravity files).
WELL_NAME_MAP = {
    # BHP.csv and Pressure xlsx Sheet2 use bare numbers
    "11":   "M-11-HRL",
    "13":   "M-13-HRL",
    "22":   "M-22-HRL",
    "41":   "M-41-HRL",
    "50":   "M-50-HRL",
    "51":   "M-51-HRL",
    "56":   "M-56-HRL",
    "57":   "M-57-HRL",
    "58":   "M-58-HRL",
    "61":   "M-61-HRL",
    "63":   "M-63-HRL",
    "65":   "M-65-HRL",
    "67":   "M-67-HRL",
    "75":   "M-75-HRL",
    "81":   "M-81-HRL",
    "82":   "M-82-HRL",
    "E-2":  "M-E-2-HRL",
    "122H": "M-122H-HRL",
    "123H": "M-123H-HRL",
    "124H": "M-124H-HRL",
    "125H": "M-125H-HRL",
    "126H": "M-126H-HRL",
    # Identity mappings for files already in canonical form
    "M-11-HRL":   "M-11-HRL",
    "M-13-HRL":   "M-13-HRL",
    "M-22-HRL":   "M-22-HRL",
    "M-41-HRL":   "M-41-HRL",
    "M-50-HRL":   "M-50-HRL",
    "M-51-HRL":   "M-51-HRL",
    "M-56-HRL":   "M-56-HRL",
    "M-57-HRL":   "M-57-HRL",
    "M-58-HRL":   "M-58-HRL",
    "M-61-HRL":   "M-61-HRL",
    "M-63-HRL":   "M-63-HRL",
    "M-65-HRL":   "M-65-HRL",
    "M-67-HRL":   "M-67-HRL",
    "M-75-HRL":   "M-75-HRL",
    "M-81-HRL":   "M-81-HRL",
    "M-82-HRL":   "M-82-HRL",
    "M-E-2-HRL":  "M-E-2-HRL",
    "M-122H-HRL": "M-122H-HRL",
    "M-123H-HRL": "M-123H-HRL",
    "M-124H-HRL": "M-124H-HRL",
    "M-125H-HRL": "M-125H-HRL",
    "M-126H-HRL": "M-126H-HRL",
}

# All 22 canonical well names
ALL_WELLS = sorted(set(WELL_NAME_MAP.values()))

# Well classification
HORIZONTAL_WELLS = {"M-122H-HRL", "M-123H-HRL", "M-124H-HRL", "M-125H-HRL", "M-126H-HRL"}
FORECAST_TARGETS = {"M-123H-HRL", "M-124H-HRL", "M-125H-HRL", "M-126H-HRL"}
EXCLUDED_WELLS = {"M-51-HRL"}  # produces water from month 0 (completion flowback)

# ── Physical constants ──────────────────────────────────────────────────────
GWC_TVD_SS_M = 684.0          # GWC datum from MARI (TVD-SS)
ROTARY_TABLE_ELEV_M = 70.0    # from M-11 completion sketch
GWC_RKB_M = GWC_TVD_SS_M + ROTARY_TABLE_ELEV_M  # 754 m RKB

WGR_THRESHOLD_BBL_MMCF = 5.0  # breakthrough WGR threshold
BT_SUSTAINED_MONTHS = 3       # consecutive months above threshold

# ── Plot palette ────────────────────────────────────────────────────────────
COLOR_WET = "#d62728"          # red
COLOR_DRY = "#1f77b4"          # blue
COLOR_FORECAST = "#e6a817"     # gold
COLOR_EXCLUDED = "#7f7f7f"     # gray
COLOR_HORIZONTAL = "#9467bd"   # purple


def canonicalize_well(name):
    """Map any well-name variant to its canonical form. Raises KeyError if unknown."""
    s = str(name).strip()
    if s in WELL_NAME_MAP:
        return WELL_NAME_MAP[s]
    raise KeyError(f"Unknown well name: {s!r}")
