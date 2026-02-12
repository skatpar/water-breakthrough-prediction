# Physics-Informed Water Breakthrough Prediction

A Physics-Informed Neural Network (PINN) for predicting water breakthrough in oil reservoirs, using the [Volve production dataset](https://www.kaggle.com/datasets/lamyalbert/volve-production-data) from Equinor's North Sea oil field.

## Overview

Water breakthrough — the arrival of injected or aquifer water at a production well — is a critical event in reservoir management. Early and accurate prediction enables operators to optimize production strategies, plan workovers, and maximize oil recovery.

This project combines classical reservoir engineering physics (Buckley-Leverett theory) with deep learning in a hybrid PINN architecture that:

1. **Learns reservoir parameters** (relative permeability exponents, endpoint saturations, viscosity ratio, porosity) jointly with the neural network
2. **Enforces physical constraints** during training via PDE residual losses
3. **Predicts both** continuous water cut and binary breakthrough classification
4. **Blends** physics-based and data-driven predictions through a learned gating mechanism

## Physics Model

The physics component is based on the **Buckley-Leverett fractional flow theory**:

### Fractional Flow Equation
```
f_w(S_w) = 1 / (1 + (kr_o(S_w) · μ_w) / (kr_w(S_w) · μ_o))
```

### Corey Relative Permeability
```
kr_w = kr_w_max · S_e^n_w
kr_o = kr_o_max · (1 - S_e)^n_o
```
where `S_e = (S_w - S_wc) / (1 - S_wc - S_or)` is the normalized saturation.

### Buckley-Leverett Conservation Law
```
φ · ∂S_w/∂t + ∂f_w/∂x = 0
```

### Physics Constraints in Loss Function
- **PDE residual**: Buckley-Leverett equation satisfaction
- **Monotonicity**: Water cut must be non-decreasing (absent workovers)
- **Material balance**: Production rates consistent with predicted water cut
- **Saturation bounds**: Water saturation within physical limits [S_wc, 1-S_or]

## Architecture

```
Input (production time series)
    │
    ▼
┌──────────────────┐
│  LSTM Encoder    │  ← Temporal patterns from production data
│  (2 layers)      │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌────────┐
│Physics │ │  Data  │
│Branch  │ │Branch  │
│(BL-fw) │ │ (MLP)  │
└───┬────┘ └───┬────┘
    │          │
    ▼          ▼
┌──────────────────┐
│  Learned Gate    │  ← Blends physics and data-driven predictions
│  α·fw + (1-α)·fd│
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
 Water Cut  Breakthrough
 (0-1)      (binary)
```

## Dataset

**Volve Field Production Data** (Equinor, 2008-2016):
- Daily production from 7 wells over ~9 years
- Features: downhole pressure/temperature, choke size, wellhead conditions
- Targets: water cut (oil/water/gas volumes)

Download from: https://www.kaggle.com/datasets/lamyalbert/volve-production-data

If no data file is provided, the model runs on synthetic data that mimics Volve field characteristics (exponential oil decline, sigmoid water breakthrough, realistic pressure dynamics).

## Installation

```bash
pip install -r requirements.txt
```

Requirements: Python 3.10+, PyTorch 2.0+, NumPy, Pandas, scikit-learn, Matplotlib, SciPy.

## Usage

### Run with synthetic data (demo)
```bash
python main.py
```

### Run with real Volve data
```bash
python main.py --data data/Volve_production_data.csv
```

### Custom training configuration
```bash
python main.py --data data/Volve_production_data.csv \
    --epochs 300 \
    --hidden 128 \
    --batch-size 128 \
    --lr 0.001 \
    --physics-weight 0.2
```

### Evaluate a saved model
```bash
python main.py --evaluate models/best_model.pt --data data/Volve_production_data.csv
```

### All options
```
--data              Path to Volve CSV (default: synthetic)
--seq-length        LSTM sequence length (default: 30)
--test-fraction     Test set fraction (default: 0.2)
--hidden            LSTM hidden size (default: 64)
--lstm-layers       Number of LSTM layers (default: 2)
--dropout           Dropout rate (default: 0.1)
--epochs            Max training epochs (default: 200)
--batch-size        Batch size (default: 64)
--lr                Learning rate (default: 0.001)
--physics-weight    Final physics loss weight (default: 0.1)
--save-dir          Output directory (default: results/)
--model-path        Model save path (default: models/best_model.pt)
--device            cpu/cuda/auto (default: auto)
--seed              Random seed (default: 42)
```

## Output

The model produces:
- `results/training_history.png` — Loss curves and parameter evolution
- `results/predictions.png` — Water cut predictions, scatter plots, gate values
- `results/fractional_flow.png` — Learned Buckley-Leverett curves and relative permeability
- `results/metrics.txt` — Quantitative evaluation metrics
- `models/best_model.pt` — Best model checkpoint with learned physics parameters

## Project Structure

```
├── main.py                  # Entry point and CLI
├── src/
│   ├── data_loader.py       # Data loading, preprocessing, feature engineering
│   ├── physics.py           # Buckley-Leverett, Corey rel-perm, material balance
│   ├── model.py             # PINN architecture (encoder, physics/data branches)
│   ├── train.py             # Training loop with composite physics-informed loss
│   └── evaluate.py          # Metrics, visualization, report generation
├── tests/
├── data/                    # Place Volve CSV here
├── models/                  # Saved model checkpoints
├── results/                 # Output plots and metrics
├── requirements.txt
└── README.md
```

## References

- Buckley, S.E. & Leverett, M.C. (1942). "Mechanism of Fluid Displacement in Sands." *Trans. AIME*, 146(01), 107-116.
- Corey, A.T. (1954). "The interrelation between gas and oil relative permeabilities." *Producers Monthly*, 19(1), 38-41.
- Fraces, C.G. et al. (2020). "Physics Informed Deep Learning for Transport in Porous Media. Buckley Leverett Problem." [arXiv:2001.05172](https://arxiv.org/abs/2001.05172)
- Raissi, M. et al. (2019). "Physics-informed neural networks." *Journal of Computational Physics*, 378, 686-707.
- Equinor (2018). [Volve field data](https://www.equinor.com/energy/volve-data-sharing).
