"""
Physics-Informed Neural Network (PINN) for water breakthrough prediction.

Architecture:
    1. Temporal encoder (LSTM) processes production time-series
    2. Physics branch computes fractional flow from learned saturation
    3. Dual output heads:
       a) Water cut regression (continuous 0-1)
       b) Breakthrough classification (binary)
    4. Physics constraints embedded in loss function

The model learns reservoir parameters (relative permeability exponents,
endpoint values, viscosity ratio) jointly with the neural network weights.
"""

import torch
import torch.nn as nn

from .physics import (
    CoreyRelativePermeability,
    FractionalFlow,
    BuckleyLeverettResidual,
    MonotonicityConstraint,
)


class TemporalEncoder(nn.Module):
    """LSTM-based encoder for production time-series data."""

    def __init__(
        self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features)
        Returns:
            h: (batch, hidden_size) - last hidden state
        """
        output, (h_n, _) = self.lstm(x)
        # Use last time step output
        h = output[:, -1, :]
        return self.layer_norm(h)


class PhysicsBranch(nn.Module):
    """
    Maps encoder output to water saturation, then computes fractional
    flow using Buckley-Leverett physics.

    The saturation is predicted by a small MLP, and the fractional flow
    is computed analytically using Corey relative permeability.
    """

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        # MLP to predict water saturation from encoded features
        self.saturation_net = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # Saturation must be in [0, 1]
        )

        # Learnable physics parameters
        self.rel_perm = CoreyRelativePermeability(learnable=True)
        self.fractional_flow = FractionalFlow(
            rel_perm=self.rel_perm, learnable_viscosity=True
        )

    def forward(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Encoded features (batch, hidden_size)

        Returns:
            s_w: Predicted water saturation (batch, 1)
            f_w: Physics-based fractional flow (batch, 1)
        """
        s_w = self.saturation_net(h)
        # Scale to physical range [S_wc, 1 - S_or]
        s_wc = torch.clamp(self.rel_perm.s_wc, 0.01, 0.45)
        s_or = torch.clamp(self.rel_perm.s_or, 0.01, 0.45)
        s_w = s_wc + s_w * (1.0 - s_wc - s_or)

        f_w = self.fractional_flow(s_w)
        return s_w, f_w


class DataDrivenBranch(nn.Module):
    """
    Purely data-driven MLP branch for water cut prediction.

    This branch captures patterns that the physics model may miss
    (e.g., workover effects, choke changes, complex geology).
    """

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 48),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(48, 24),
            nn.GELU(),
            nn.Linear(24, 1),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class BreakthroughClassifier(nn.Module):
    """Binary classifier head for breakthrough detection."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class PhysicsInformedBreakthroughModel(nn.Module):
    """
    Main PINN model for water breakthrough prediction.

    Combines:
    - Temporal encoder (LSTM) for sequential production data
    - Physics branch (Buckley-Leverett fractional flow)
    - Data-driven branch (MLP for residual patterns)
    - Breakthrough classifier

    The final water cut prediction blends physics and data-driven outputs
    using a learned gating mechanism, ensuring physical consistency while
    capturing data-driven corrections.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        physics_weight: float = 0.5,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size

        # Shared temporal encoder
        self.encoder = TemporalEncoder(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout,
        )

        # Physics branch
        self.physics_branch = PhysicsBranch(hidden_size)

        # Data-driven branch
        self.data_branch = DataDrivenBranch(hidden_size)

        # Gating network to blend physics and data-driven predictions
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Breakthrough classifier
        self.breakthrough_head = BreakthroughClassifier(hidden_size)

        # Physics loss modules
        self.bl_residual = BuckleyLeverettResidual(
            fractional_flow=self.physics_branch.fractional_flow,
        )
        self.monotonicity = MonotonicityConstraint()

        # Initial physics weight
        self._physics_weight = physics_weight

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, n_features)

        Returns:
            Dictionary with:
                - water_cut: Blended water cut prediction (batch, 1)
                - water_cut_physics: Physics-only prediction (batch, 1)
                - water_cut_data: Data-only prediction (batch, 1)
                - saturation: Predicted water saturation (batch, 1)
                - breakthrough_logit: Breakthrough probability logit (batch, 1)
                - gate_value: Physics vs data weight (batch, 1)
        """
        # Encode temporal sequence
        h = self.encoder(x)

        # Physics prediction via Buckley-Leverett
        s_w, f_w_physics = self.physics_branch(h)

        # Data-driven prediction
        f_w_data = self.data_branch(h)

        # Gating: blend physics and data-driven
        gate = self.gate(h)
        water_cut = gate * f_w_physics + (1 - gate) * f_w_data

        # Breakthrough classification
        bt_logit = self.breakthrough_head(h)

        return {
            "water_cut": water_cut,
            "water_cut_physics": f_w_physics,
            "water_cut_data": f_w_data,
            "saturation": s_w,
            "breakthrough_logit": bt_logit,
            "gate_value": gate,
        }

    def get_physics_parameters(self) -> dict[str, float]:
        """Extract learned physics parameters for interpretation."""
        rp = self.physics_branch.rel_perm
        ff = self.physics_branch.fractional_flow
        bl = self.bl_residual
        return {
            "s_wc": torch.clamp(rp.s_wc, 0.01, 0.45).item(),
            "s_or": torch.clamp(rp.s_or, 0.01, 0.45).item(),
            "kr_w_max": torch.clamp(rp.kr_w_max, 0.05, 1.0).item(),
            "kr_o_max": torch.clamp(rp.kr_o_max, 0.05, 1.0).item(),
            "n_w": torch.clamp(rp.n_w, 1.0, 6.0).item(),
            "n_o": torch.clamp(rp.n_o, 1.0, 6.0).item(),
            "mu_w": ff.mu_w.item(),
            "mu_o": ff.mu_o.item(),
            "porosity": bl.porosity.item(),
        }
