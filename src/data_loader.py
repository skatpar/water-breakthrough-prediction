"""
Data loading and preprocessing for Volve production data.

Handles loading the Kaggle Volve production dataset, cleaning, feature
engineering, and preparing tensors for the physics-informed model.

Expected dataset: https://www.kaggle.com/datasets/lamyalbert/volve-production-data
File: Volve production data.csv
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


# Column names in the Volve production CSV
VOLVE_COLUMNS = {
    "date": "DATEPRD",
    "well": "NPD_WELL_BORE_CODE",
    "well_bore_name": "NPD_WELL_BORE_NAME",
    "on_stream_hrs": "ON_STREAM_HRS",
    "avg_downhole_press": "AVG_DOWNHOLE_PRESSURE",
    "avg_downhole_temp": "AVG_DOWNHOLE_TEMPERATURE",
    "avg_dp_tubing": "AVG_DP_TUBING",
    "avg_annulus_press": "AVG_ANNULUS_PRESS",
    "avg_choke_size": "AVG_CHOKE_SIZE_P",
    "avg_whp": "AVG_WHP_P",
    "avg_wht": "AVG_WHT_P",
    "dp_choke_size": "DP_CHOKE_SIZE",
    "bore_oil_vol": "BORE_OIL_VOL",
    "bore_gas_vol": "BORE_GAS_VOL",
    "bore_wat_vol": "BORE_WAT_VOL",
    "bore_wi_vol": "BORE_WI_VOL",
    "flow_kind": "FLOW_KIND",
    "well_type": "WELL_TYPE",
}

# Features used for the model
PRODUCTION_FEATURES = [
    "ON_STREAM_HRS",
    "AVG_DOWNHOLE_PRESSURE",
    "AVG_DOWNHOLE_TEMPERATURE",
    "AVG_ANNULUS_PRESS",
    "AVG_CHOKE_SIZE_P",
    "AVG_WHP_P",
    "AVG_WHT_P",
]


def load_volve_data(data_path: str) -> pd.DataFrame:
    """Load Volve production CSV and perform initial cleaning."""
    path = Path(data_path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in (".xls", ".xlsx"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    df[VOLVE_COLUMNS["date"]] = pd.to_datetime(df[VOLVE_COLUMNS["date"]])
    df = df.sort_values([VOLVE_COLUMNS["well"], VOLVE_COLUMNS["date"]])
    return df


def compute_water_cut(df: pd.DataFrame) -> pd.DataFrame:
    """Compute water cut (fractional flow of water at surface) for each record."""
    oil = df[VOLVE_COLUMNS["bore_oil_vol"]].fillna(0).clip(lower=0)
    water = df[VOLVE_COLUMNS["bore_wat_vol"]].fillna(0).clip(lower=0)
    total_liquid = oil + water
    df["WATER_CUT"] = np.where(total_liquid > 0, water / total_liquid, 0.0)
    return df


def compute_cumulative_production(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative oil and water volumes per well."""
    well_col = VOLVE_COLUMNS["well"]
    df["CUM_OIL"] = df.groupby(well_col)[VOLVE_COLUMNS["bore_oil_vol"]].cumsum()
    df["CUM_WATER"] = df.groupby(well_col)[VOLVE_COLUMNS["bore_wat_vol"]].cumsum()
    df["CUM_LIQUID"] = df["CUM_OIL"] + df["CUM_WATER"]
    return df


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized time features per well (days since first production)."""
    well_col = VOLVE_COLUMNS["well"]
    date_col = VOLVE_COLUMNS["date"]
    df["DAYS_ON_PRODUCTION"] = df.groupby(well_col)[date_col].transform(
        lambda x: (x - x.min()).dt.days
    )
    return df


def detect_breakthrough(df: pd.DataFrame, wc_threshold: float = 0.02) -> pd.DataFrame:
    """
    Label water breakthrough events.

    Breakthrough is defined as the first time water cut exceeds a threshold
    and remains above it. Returns binary label: 1 = post-breakthrough.
    """
    well_col = VOLVE_COLUMNS["well"]

    breakthrough_labels = np.zeros(len(df))
    days_to_bt = np.zeros(len(df))

    for _, idx in df.groupby(well_col).groups.items():
        wc = df.loc[idx, "WATER_CUT"].values
        above = wc >= wc_threshold
        breakthrough_idx = len(wc)
        for i in range(len(wc) - 2):
            if above[i] and above[i + 1] and above[i + 2]:
                breakthrough_idx = i
                break
        labels = np.zeros(len(wc))
        labels[breakthrough_idx:] = 1.0
        breakthrough_labels[idx] = labels
        days_to_bt[idx] = breakthrough_idx - np.arange(len(wc))

    df["BREAKTHROUGH"] = breakthrough_labels
    df["DAYS_TO_BREAKTHROUGH"] = days_to_bt
    return df


def filter_production_wells(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only production wells (not injectors)."""
    if VOLVE_COLUMNS["well_type"] in df.columns:
        prod_mask = df[VOLVE_COLUMNS["well_type"]].str.upper().str.contains(
            "OP", na=False
        )
        df = df[prod_mask].copy()
    return df


def prepare_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    Prepare feature matrix X and target y (water cut) with scaling.

    Returns:
        X_scaled: Scaled feature array
        y_scaled: Scaled target array
        scaler_X: Fitted feature scaler
        scaler_y: Fitted target scaler
    """
    if feature_cols is None:
        feature_cols = PRODUCTION_FEATURES.copy()

    # Add derived features
    extra_cols = ["DAYS_ON_PRODUCTION", "CUM_OIL", "CUM_LIQUID"]
    all_features = feature_cols + [c for c in extra_cols if c in df.columns]
    available = [c for c in all_features if c in df.columns]

    X = df[available].fillna(0).values.astype(np.float32)
    y = df["WATER_CUT"].values.astype(np.float32).reshape(-1, 1)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    return X_scaled, y_scaled, scaler_X, scaler_y


def create_sequences(
    X: np.ndarray, y: np.ndarray, seq_length: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """Create temporal sequences for time-series modeling."""
    sequences_X, sequences_y = [], []
    for i in range(len(X) - seq_length):
        sequences_X.append(X[i : i + seq_length])
        sequences_y.append(y[i + seq_length])
    return np.array(sequences_X), np.array(sequences_y)


def prepare_physics_inputs(df: pd.DataFrame) -> dict[str, torch.Tensor]:
    """
    Extract physics-relevant quantities as tensors for PINN constraints.

    Returns dict with tensors for:
        - pressure (downhole)
        - time (days on production, normalized 0-1)
        - water_cut (observed fractional flow)
        - oil_rate, water_rate (daily volumes)
        - cumulative_injection (from injector data if available)
    """
    t = df["DAYS_ON_PRODUCTION"].values.astype(np.float32)
    t_norm = t / (t.max() + 1e-8)

    physics = {
        "time": torch.tensor(t_norm, dtype=torch.float32).unsqueeze(1),
        "water_cut": torch.tensor(
            df["WATER_CUT"].values, dtype=torch.float32
        ).unsqueeze(1),
        "oil_rate": torch.tensor(
            df[VOLVE_COLUMNS["bore_oil_vol"]].fillna(0).values, dtype=torch.float32
        ).unsqueeze(1),
        "water_rate": torch.tensor(
            df[VOLVE_COLUMNS["bore_wat_vol"]].fillna(0).values, dtype=torch.float32
        ).unsqueeze(1),
    }

    if VOLVE_COLUMNS["avg_downhole_press"] in df.columns:
        p = df[VOLVE_COLUMNS["avg_downhole_press"]].fillna(0).values.astype(np.float32)
        p_norm = (p - p.min()) / (p.max() - p.min() + 1e-8)
        physics["pressure"] = torch.tensor(p_norm, dtype=torch.float32).unsqueeze(1)

    return physics


def generate_synthetic_volve_data(
    n_wells: int = 5, n_days: int = 1500, seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic production data mimicking Volve field characteristics.

    Used for testing and demonstration when the actual Kaggle dataset is not
    available. Incorporates realistic decline curves, water breakthrough
    behavior, and pressure dynamics.
    """
    rng = np.random.default_rng(seed)
    records = []

    for w in range(n_wells):
        well_name = f"WELL-{w + 1:02d}"
        # Reservoir parameters per well
        initial_oil_rate = rng.uniform(800, 2500)  # Sm3/day
        decline_rate = rng.uniform(0.0003, 0.001)  # exponential decline
        breakthrough_day = int(rng.uniform(200, 800))
        initial_pressure = rng.uniform(250, 350)  # bar
        pressure_decline = rng.uniform(0.02, 0.08)  # bar/day
        temperature = rng.uniform(90, 110)  # Celsius

        base_date = pd.Timestamp("2008-02-01")

        for d in range(n_days):
            date = base_date + pd.Timedelta(days=d)
            on_stream = rng.uniform(20, 24)

            # Exponential oil decline
            oil_rate = initial_oil_rate * np.exp(-decline_rate * d)
            oil_rate *= rng.uniform(0.92, 1.08)  # noise

            # Water breakthrough modeled as sigmoid
            if d < breakthrough_day:
                water_cut = rng.uniform(0, 0.01)
            else:
                dt = d - breakthrough_day
                # Logistic water cut increase post-breakthrough
                wc_max = rng.uniform(0.85, 0.98)
                steepness = rng.uniform(0.005, 0.015)
                water_cut = wc_max / (1 + np.exp(-steepness * (dt - 300)))
                water_cut += rng.normal(0, 0.01)
                water_cut = np.clip(water_cut, 0, 1)

            water_rate = oil_rate * water_cut / (1 - water_cut + 1e-8)
            gas_rate = oil_rate * rng.uniform(80, 150)  # GOR

            # Pressure
            pressure = initial_pressure - pressure_decline * d
            pressure += rng.normal(0, 2)
            pressure = max(pressure, 50)

            whp = pressure * rng.uniform(0.3, 0.5)
            annulus = pressure * rng.uniform(0.4, 0.6)
            choke = rng.uniform(20, 80)

            records.append({
                VOLVE_COLUMNS["date"]: date,
                VOLVE_COLUMNS["well"]: w + 1,
                VOLVE_COLUMNS["well_bore_name"]: well_name,
                VOLVE_COLUMNS["on_stream_hrs"]: round(on_stream, 1),
                VOLVE_COLUMNS["avg_downhole_press"]: round(pressure, 2),
                VOLVE_COLUMNS["avg_downhole_temp"]: round(
                    temperature + rng.normal(0, 1), 1
                ),
                VOLVE_COLUMNS["avg_dp_tubing"]: round(rng.uniform(5, 30), 2),
                VOLVE_COLUMNS["avg_annulus_press"]: round(annulus, 2),
                VOLVE_COLUMNS["avg_choke_size"]: round(choke, 1),
                VOLVE_COLUMNS["avg_whp"]: round(whp, 2),
                VOLVE_COLUMNS["avg_wht"]: round(
                    temperature * 0.7 + rng.normal(0, 2), 1
                ),
                VOLVE_COLUMNS["bore_oil_vol"]: round(max(oil_rate, 0), 2),
                VOLVE_COLUMNS["bore_gas_vol"]: round(max(gas_rate, 0), 2),
                VOLVE_COLUMNS["bore_wat_vol"]: round(max(water_rate, 0), 2),
                VOLVE_COLUMNS["bore_wi_vol"]: 0.0,
                VOLVE_COLUMNS["flow_kind"]: "production",
                VOLVE_COLUMNS["well_type"]: "OP",
            })

    df = pd.DataFrame(records)
    df[VOLVE_COLUMNS["date"]] = pd.to_datetime(df[VOLVE_COLUMNS["date"]])
    return df


def _prepare_survival_targets(
    df: pd.DataFrame, seq_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare survival analysis targets from the dataframe.

    For each sample (after sequencing), computes:
        - time_to_event: Days until breakthrough if pre-breakthrough,
          or days since production start if post-breakthrough (observed time).
          Normalized by max observed time per well for numerical stability.
        - event: 1 if breakthrough has been observed at this time step, 0 if censored.

    Right-censored samples are those where breakthrough hasn't occurred yet
    (DAYS_TO_BREAKTHROUGH > 0 means we haven't reached BT).

    Returns:
        tte: Time-to-event array, shape (n_samples,)
        event: Event indicator array, shape (n_samples,)
    """
    well_col = VOLVE_COLUMNS["well"]
    days_col = "DAYS_ON_PRODUCTION"
    bt_col = "BREAKTHROUGH"
    days_to_bt_col = "DAYS_TO_BREAKTHROUGH"

    # For each row, compute the time-to-event:
    #   - If pre-breakthrough (BT=0): time = days_to_breakthrough (censored, event=0)
    #   - If post-breakthrough (BT=1): time = days from production start to BT day (event=1)
    tte_values = np.zeros(len(df), dtype=np.float32)
    event_values = np.zeros(len(df), dtype=np.float32)

    for _, group_idx in df.groupby(well_col).groups.items():
        group = df.loc[group_idx]
        days = group[days_col].values.astype(np.float32)
        bt = group[bt_col].values
        days_to_bt = group[days_to_bt_col].values

        # Find the breakthrough day for this well
        bt_indices = np.where(bt == 1.0)[0]
        if len(bt_indices) > 0:
            bt_day = days[bt_indices[0]]
        else:
            bt_day = days[-1]  # Never broke through; use last observed day

        for i, idx in enumerate(group_idx):
            if bt[i] == 1.0:
                # Post-breakthrough: event observed, time = breakthrough day
                tte_values[idx] = max(bt_day, 1.0)
                event_values[idx] = 1.0
            else:
                # Pre-breakthrough: censored at current time
                # Time = days_to_bt (how many days until BT from this point)
                # But for survival model we want the total time-to-event
                remaining = days_to_bt[i]
                if remaining > 0:
                    tte_values[idx] = days[i] + remaining
                    event_values[idx] = 1.0  # We know BT will happen
                else:
                    # Well never broke through; censored at current day
                    tte_values[idx] = max(days[i], 1.0)
                    event_values[idx] = 0.0

    # Apply sequencing offset (same as create_sequences)
    tte_seq = tte_values[seq_length:]
    event_seq = event_values[seq_length:]

    # Normalize time-to-event to reasonable scale (divide by max)
    max_tte = tte_seq.max()
    if max_tte > 0:
        tte_seq = tte_seq / max_tte

    # Ensure minimum time for numerical stability
    tte_seq = np.clip(tte_seq, 1e-4, None)

    return tte_seq, event_seq


def build_dataset(
    data_path: str | None = None,
    seq_length: int = 30,
    test_fraction: float = 0.2,
    device: str = "cpu",
) -> dict:
    """
    Full pipeline: load data, engineer features, create train/test tensors.

    Args:
        data_path: Path to Volve CSV. If None, generates synthetic data.
        seq_length: Temporal sequence length for LSTM/transformer input.
        test_fraction: Fraction of data held out for testing.
        device: Torch device.

    Returns:
        Dictionary with train/test tensors and metadata.
    """
    # Load or generate data
    if data_path and Path(data_path).exists():
        print(f"Loading data from {data_path}")
        df = load_volve_data(data_path)
        df = filter_production_wells(df)
    else:
        if data_path:
            print(f"File not found: {data_path}. Using synthetic data.")
        else:
            print("No data path provided. Using synthetic data.")
        df = generate_synthetic_volve_data()

    # Feature engineering
    df = compute_water_cut(df)
    df = compute_cumulative_production(df)
    df = compute_time_features(df)
    df = detect_breakthrough(df)

    # Remove rows with all-zero production
    oil_col = VOLVE_COLUMNS["bore_oil_vol"]
    wat_col = VOLVE_COLUMNS["bore_wat_vol"]
    df = df[(df[oil_col] > 0) | (df[wat_col] > 0)].copy()

    # Prepare features
    X_scaled, y_scaled, scaler_X, scaler_y = prepare_features(df)

    # Create sequences
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)

    # Train/test split (temporal - no shuffle)
    split_idx = int(len(X_seq) * (1 - test_fraction))
    X_train = torch.tensor(X_seq[:split_idx], dtype=torch.float32).to(device)
    y_train = torch.tensor(y_seq[:split_idx], dtype=torch.float32).to(device)
    X_test = torch.tensor(X_seq[split_idx:], dtype=torch.float32).to(device)
    y_test = torch.tensor(y_seq[split_idx:], dtype=torch.float32).to(device)

    # Physics inputs
    physics_inputs = prepare_physics_inputs(df)
    physics_inputs = {k: v.to(device) for k, v in physics_inputs.items()}

    # Breakthrough labels for classification
    bt_labels = df["BREAKTHROUGH"].values.astype(np.float32)
    bt_seq = []
    for i in range(len(bt_labels) - seq_length):
        bt_seq.append(bt_labels[i + seq_length])
    bt_seq = np.array(bt_seq)
    bt_train = torch.tensor(bt_seq[:split_idx], dtype=torch.float32).to(device)
    bt_test = torch.tensor(bt_seq[split_idx:], dtype=torch.float32).to(device)

    well_names = df[VOLVE_COLUMNS["well"]].unique().tolist()

    # Survival targets: time-to-breakthrough and censoring indicator
    tte, event = _prepare_survival_targets(df, seq_length)
    tte_train = torch.tensor(
        tte[:split_idx], dtype=torch.float32
    ).unsqueeze(1).to(device)
    tte_test = torch.tensor(
        tte[split_idx:], dtype=torch.float32
    ).unsqueeze(1).to(device)
    event_train = torch.tensor(
        event[:split_idx], dtype=torch.float32
    ).unsqueeze(1).to(device)
    event_test = torch.tensor(
        event[split_idx:], dtype=torch.float32
    ).unsqueeze(1).to(device)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "bt_train": bt_train,
        "bt_test": bt_test,
        "tte_train": tte_train,
        "tte_test": tte_test,
        "event_train": event_train,
        "event_test": event_test,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "physics_inputs": physics_inputs,
        "dataframe": df,
        "well_names": well_names,
        "n_features": X_scaled.shape[1],
        "seq_length": seq_length,
    }
