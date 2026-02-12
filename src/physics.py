"""
Physics equations for water breakthrough prediction.

Implements reservoir engineering equations used as constraints in the
physics-informed neural network:

1. Buckley-Leverett fractional flow theory
2. Darcy's law for flow in porous media
3. Material balance equations
4. Relative permeability models (Corey correlations)

References:
    - Buckley, S.E. & Leverett, M.C. (1942). "Mechanism of Fluid
      Displacement in Sands." Trans. AIME, 146(01), 107-116.
    - Corey, A.T. (1954). "The interrelation between gas and oil
      relative permeabilities." Producers Monthly, 19(1), 38-41.
"""

import torch
import torch.nn as nn
import numpy as np


class CoreyRelativePermeability(nn.Module):
    """
    Corey-type relative permeability model.

    kr_w = kr_w_max * S_e^n_w
    kr_o = kr_o_max * (1 - S_e)^n_o

    where S_e = (S_w - S_wc) / (1 - S_wc - S_or) is the normalized saturation.
    """

    def __init__(
        self,
        s_wc: float = 0.2,
        s_or: float = 0.2,
        kr_w_max: float = 0.3,
        kr_o_max: float = 1.0,
        n_w: float = 3.0,
        n_o: float = 2.0,
        learnable: bool = True,
    ):
        super().__init__()
        if learnable:
            self.s_wc = nn.Parameter(torch.tensor(s_wc))
            self.s_or = nn.Parameter(torch.tensor(s_or))
            self.kr_w_max = nn.Parameter(torch.tensor(kr_w_max))
            self.kr_o_max = nn.Parameter(torch.tensor(kr_o_max))
            self.n_w = nn.Parameter(torch.tensor(n_w))
            self.n_o = nn.Parameter(torch.tensor(n_o))
        else:
            self.register_buffer("s_wc", torch.tensor(s_wc))
            self.register_buffer("s_or", torch.tensor(s_or))
            self.register_buffer("kr_w_max", torch.tensor(kr_w_max))
            self.register_buffer("kr_o_max", torch.tensor(kr_o_max))
            self.register_buffer("n_w", torch.tensor(n_w))
            self.register_buffer("n_o", torch.tensor(n_o))

    def normalized_saturation(self, s_w: torch.Tensor) -> torch.Tensor:
        """Compute normalized water saturation S_e."""
        s_wc = torch.clamp(self.s_wc, 0.01, 0.45)
        s_or = torch.clamp(self.s_or, 0.01, 0.45)
        s_e = (s_w - s_wc) / (1.0 - s_wc - s_or + 1e-8)
        return torch.clamp(s_e, 0.0, 1.0)

    def kr_water(self, s_w: torch.Tensor) -> torch.Tensor:
        """Water relative permeability."""
        s_e = self.normalized_saturation(s_w)
        n_w = torch.clamp(self.n_w, 1.0, 6.0)
        kr_w_max = torch.clamp(self.kr_w_max, 0.05, 1.0)
        return kr_w_max * torch.pow(s_e + 1e-8, n_w)

    def kr_oil(self, s_w: torch.Tensor) -> torch.Tensor:
        """Oil relative permeability."""
        s_e = self.normalized_saturation(s_w)
        n_o = torch.clamp(self.n_o, 1.0, 6.0)
        kr_o_max = torch.clamp(self.kr_o_max, 0.05, 1.0)
        return kr_o_max * torch.pow(1.0 - s_e + 1e-8, n_o)

    def forward(
        self, s_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (kr_w, kr_o) for given water saturation."""
        return self.kr_water(s_w), self.kr_oil(s_w)


class FractionalFlow(nn.Module):
    """
    Buckley-Leverett fractional flow function.

    f_w = 1 / (1 + (kr_o * mu_w) / (kr_w * mu_o))

    This describes the fraction of water in the total flow at any point
    in the reservoir as a function of water saturation.
    """

    def __init__(
        self,
        mu_w: float = 0.5,
        mu_o: float = 2.0,
        rel_perm: CoreyRelativePermeability | None = None,
        learnable_viscosity: bool = True,
    ):
        super().__init__()
        if learnable_viscosity:
            self.log_mu_w = nn.Parameter(torch.tensor(np.log(mu_w)))
            self.log_mu_o = nn.Parameter(torch.tensor(np.log(mu_o)))
        else:
            self.register_buffer("log_mu_w", torch.tensor(np.log(mu_w)))
            self.register_buffer("log_mu_o", torch.tensor(np.log(mu_o)))

        self.rel_perm = rel_perm or CoreyRelativePermeability()

    @property
    def mu_w(self) -> torch.Tensor:
        return torch.exp(self.log_mu_w)

    @property
    def mu_o(self) -> torch.Tensor:
        return torch.exp(self.log_mu_o)

    def forward(self, s_w: torch.Tensor) -> torch.Tensor:
        """
        Compute fractional flow of water f_w(S_w).

        Args:
            s_w: Water saturation tensor [0, 1]

        Returns:
            f_w: Fractional flow of water [0, 1]
        """
        kr_w, kr_o = self.rel_perm(s_w)
        mobility_ratio = (kr_o * self.mu_w) / (kr_w * self.mu_o + 1e-8)
        f_w = 1.0 / (1.0 + mobility_ratio)
        return f_w

    def derivative(self, s_w: torch.Tensor) -> torch.Tensor:
        """
        Compute df_w/dS_w using autograd.

        This derivative is critical for the Buckley-Leverett shock speed
        and Welge tangent construction.
        """
        s_w = s_w.detach().requires_grad_(True)
        f_w = self.forward(s_w)
        df_ds = torch.autograd.grad(
            f_w.sum(), s_w, create_graph=True
        )[0]
        return df_ds


class BuckleyLeverettResidual(nn.Module):
    """
    Computes the residual of the Buckley-Leverett equation:

        phi * dS_w/dt + (u_t / A) * df_w/dx = 0

    In 1D this reduces to the conservation law for water saturation.
    The residual is used as a physics loss in the PINN.
    """

    def __init__(
        self,
        porosity: float = 0.25,
        fractional_flow: FractionalFlow | None = None,
        learnable_porosity: bool = True,
    ):
        super().__init__()
        if learnable_porosity:
            self.log_porosity = nn.Parameter(torch.tensor(np.log(porosity)))
        else:
            self.register_buffer("log_porosity", torch.tensor(np.log(porosity)))

        self.fractional_flow = fractional_flow or FractionalFlow()

    @property
    def porosity(self) -> torch.Tensor:
        return torch.clamp(torch.exp(self.log_porosity), 0.05, 0.45)

    def forward(
        self,
        s_w: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Buckley-Leverett PDE residual.

        Uses time as a proxy for spatial advancement (1D displacement).
        The residual = phi * dS/dt + df_w/dS * dS/dt ≈ 0.

        Args:
            s_w: Predicted water saturation, shape (N, 1)
            t: Time tensor, shape (N, 1), requires_grad=True

        Returns:
            Residual tensor, shape (N, 1). Should be ≈ 0 if physics is satisfied.
        """
        # dS_w/dt
        ds_dt = torch.autograd.grad(
            s_w, t, grad_outputs=torch.ones_like(s_w),
            create_graph=True, retain_graph=True
        )[0]

        # f_w and df_w/dS_w
        f_w = self.fractional_flow(s_w)
        df_ds = torch.autograd.grad(
            f_w, s_w, grad_outputs=torch.ones_like(f_w),
            create_graph=True, retain_graph=True
        )[0]

        # BL residual: phi * dS/dt + v * df/dS * dS/dt ≈ 0
        # Simplified: phi * dS/dt should be balanced by flux derivative
        residual = self.porosity * ds_dt + df_ds * ds_dt

        return residual


class MaterialBalance(nn.Module):
    """
    Material balance constraint for reservoir production.

    Ensures that cumulative production is consistent:
        N_p(t) + W_p(t) ≤ OOIP + W_inj(t)

    And that the decline in reservoir pressure is consistent with
    fluid withdrawal via compressibility.
    """

    def __init__(self, total_compressibility: float = 1e-5):
        super().__init__()
        self.c_t = total_compressibility

    def production_balance_residual(
        self,
        oil_rate: torch.Tensor,
        water_rate: torch.Tensor,
        water_cut_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Check that predicted water cut is consistent with observed rates.

        f_w_pred should ≈ water_rate / (oil_rate + water_rate)
        """
        total_rate = oil_rate + water_rate + 1e-8
        f_w_observed = water_rate / total_rate
        return water_cut_pred - f_w_observed

    def pressure_decline_residual(
        self,
        pressure: torch.Tensor,
        time: torch.Tensor,
        total_rate: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simplified pressure decline: dP/dt ∝ -q_total * c_t.

        Ensures pressure changes are physically consistent with production.
        """
        dp_dt = torch.autograd.grad(
            pressure.sum(), time,
            create_graph=True, retain_graph=True
        )[0]
        # Pressure should decline with production
        expected_decline = -self.c_t * total_rate
        return dp_dt - expected_decline


class MonotonicityConstraint(nn.Module):
    """
    Enforce physical monotonicity constraints:
    - Water cut should be non-decreasing over time (in the absence of
      workover events)
    - Cumulative production should be non-decreasing
    - Fractional flow f_w(S_w) should be monotonically increasing
    """

    def forward(
        self, predictions: torch.Tensor, dim: int = 0
    ) -> torch.Tensor:
        """
        Penalize violations of monotonicity.

        Returns a loss term that is 0 when predictions are non-decreasing
        along the specified dimension.
        """
        diffs = predictions[1:] - predictions[:-1]
        violations = torch.clamp(-diffs, min=0)
        return violations.mean()


def compute_breakthrough_time_analytical(
    s_wc: float = 0.2,
    s_or: float = 0.2,
    mu_w: float = 0.5,
    mu_o: float = 2.0,
    n_w: float = 3.0,
    n_o: float = 2.0,
    pore_volumes_injected: np.ndarray | None = None,
) -> dict:
    """
    Analytical Buckley-Leverett solution for breakthrough time.

    Uses Welge tangent construction to find:
    - Water saturation at the front (S_wf)
    - Breakthrough time in pore volumes
    - Average saturation behind front

    Returns dict with analytical solution parameters.
    """
    # Evaluate fractional flow curve on a fine grid
    n_points = 1000
    s_w = np.linspace(s_wc, 1 - s_or, n_points)
    s_e = (s_w - s_wc) / (1 - s_wc - s_or)

    kr_w = 0.3 * s_e**n_w
    kr_o = 1.0 * (1 - s_e) ** n_o
    f_w = 1.0 / (1.0 + (kr_o * mu_w) / (kr_w * mu_o + 1e-10))

    # Numerical derivative df/dS
    df_ds = np.gradient(f_w, s_w)

    # Welge tangent: find tangent from (s_wc, 0) to f_w curve
    # slope = f_w(s) / (s - s_wc)
    tangent_slopes = f_w / (s_w - s_wc + 1e-10)

    # Front saturation is where tangent slope = df/dS
    # Find intersection
    diff = tangent_slopes - df_ds
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) > 0:
        idx_front = sign_changes[-1]
    else:
        idx_front = n_points // 2

    s_wf = s_w[idx_front]
    f_wf = f_w[idx_front]
    df_ds_front = df_ds[idx_front]

    # Breakthrough time in pore volumes
    t_bt_pv = 1.0 / (df_ds_front + 1e-10)

    # Average saturation behind front (Welge)
    s_w_avg = s_wc + t_bt_pv * f_wf

    return {
        "s_w_front": float(s_wf),
        "f_w_front": float(f_wf),
        "breakthrough_pv": float(t_bt_pv),
        "s_w_avg_behind_front": float(np.clip(s_w_avg, s_wc, 1 - s_or)),
        "saturation_grid": s_w,
        "fractional_flow_curve": f_w,
        "df_ds": df_ds,
    }
