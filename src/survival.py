"""
Parametric survival model for time-to-breakthrough prediction.

Predicts the distribution of time until water breakthrough using a
log-normal parametric model, providing P10, P50, P90 percentile estimates
for uncertainty quantification.

The survival head attaches to the shared LSTM encoder and outputs
distribution parameters (mu, log_sigma) that define a log-normal
distribution over breakthrough time. This enables:
    - Point estimates (median = P50)
    - Uncertainty bands (P10, P90)
    - Survival function S(t) = P(T > t)
    - Hazard function h(t)

Censored observations (wells that haven't broken through yet) are handled
via the standard survival likelihood.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import stats as sp_stats


_SQRT2 = float(np.sqrt(2.0))


class LogNormalSurvivalHead(nn.Module):
    """
    Predicts log-normal distribution parameters for time-to-breakthrough.

    Outputs:
        mu: Location parameter of log(T) distribution
        log_sigma: Log of scale parameter (ensures sigma > 0)
    """

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(16, 1)
        self.log_sigma_head = nn.Linear(16, 1)

        # Initialize log_sigma bias to log(1.0) = 0 for unit variance start
        nn.init.constant_(self.log_sigma_head.bias, 0.0)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            h: Encoded features from temporal encoder (batch, hidden_size)

        Returns:
            Dictionary with:
                - mu: Location parameter (batch, 1)
                - log_sigma: Log-scale parameter (batch, 1)
                - sigma: Scale parameter (batch, 1)
        """
        features = self.net(h)
        mu = self.mu_head(features)
        log_sigma = self.log_sigma_head(features)
        # Clamp log_sigma for numerical stability
        log_sigma = torch.clamp(log_sigma, min=-3.0, max=3.0)
        sigma = torch.exp(log_sigma)

        return {"mu": mu, "log_sigma": log_sigma, "sigma": sigma}


class SurvivalLoss(nn.Module):
    """
    Negative log-likelihood loss for right-censored survival data
    under a log-normal model.

    For uncensored (event observed):
        L = -log f(t | mu, sigma)
          = -log [ (1 / (t * sigma * sqrt(2*pi))) * exp(-(log(t) - mu)^2 / (2*sigma^2)) ]

    For right-censored (event not yet observed):
        L = -log S(t | mu, sigma)
          = -log [ 1 - Phi((log(t) - mu) / sigma) ]

    where Phi is the standard normal CDF.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute survival negative log-likelihood.

        Args:
            mu: Predicted location parameter (batch, 1)
            sigma: Predicted scale parameter (batch, 1)
            time: Observed time-to-event or censoring time (batch, 1), must be > 0
            event: Event indicator: 1 = breakthrough observed, 0 = censored (batch, 1)

        Returns:
            Scalar loss (mean NLL over batch)
        """
        # Ensure time is positive
        time = torch.clamp(time, min=1e-6)
        sigma = torch.clamp(sigma, min=1e-4)

        log_t = torch.log(time)
        z = (log_t - mu) / sigma

        # Log-PDF of log-normal: -log(t) - log(sigma) - 0.5*log(2*pi) - 0.5*z^2
        log_pdf = (
            -log_t
            - torch.log(sigma)
            - 0.5 * np.log(2 * np.pi)
            - 0.5 * z * z
        )

        # Log-survival: log(1 - Phi(z)) = log(Phi(-z))
        # Use log_ndtr for numerical stability
        log_survival = self._log_ndtr(-z)

        # Combine: event * log_pdf + (1 - event) * log_survival
        nll = -(event * log_pdf + (1.0 - event) * log_survival)

        return nll.mean()

    @staticmethod
    def _log_ndtr(z: torch.Tensor) -> torch.Tensor:
        """
        Numerically stable log of the standard normal CDF.

        Uses the complementary error function for stability in the tails.
        """
        # log(Phi(z)) = log(0.5 * erfc(-z / sqrt(2)))
        return torch.log(0.5 * torch.erfc(-z / _SQRT2) + 1e-12)


def compute_percentiles(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    percentiles: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute percentiles of the predicted log-normal distribution.

    For a log-normal with parameters (mu, sigma):
        Q(p) = exp(mu + sigma * Phi^{-1}(p))

    where Phi^{-1} is the standard normal quantile function.

    Args:
        mu: Location parameter (batch, 1)
        sigma: Scale parameter (batch, 1)
        percentiles: List of percentiles to compute (default: [0.1, 0.5, 0.9])

    Returns:
        Dictionary mapping percentile names to tensors, e.g.:
            {"P10": tensor, "P50": tensor, "P90": tensor}
    """
    if percentiles is None:
        percentiles = [0.10, 0.50, 0.90]

    results = {}
    for p in percentiles:
        z_p = float(sp_stats.norm.ppf(p))
        q = torch.exp(mu + sigma * z_p)
        label = f"P{int(p * 100)}"
        results[label] = q

    return results


def compute_survival_function(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    t_grid: torch.Tensor,
) -> torch.Tensor:
    """
    Compute survival function S(t) = P(T > t) over a time grid.

    Args:
        mu: Location parameter (batch, 1) or scalar
        sigma: Scale parameter (batch, 1) or scalar
        t_grid: Time points to evaluate (n_points,)

    Returns:
        S(t) values, shape (batch, n_points) or (n_points,)
    """
    t_grid = torch.clamp(t_grid, min=1e-6)
    log_t = torch.log(t_grid)

    if mu.dim() > 0 and mu.shape[0] > 1:
        # Batched: mu (batch, 1), t_grid (n_points,) -> (batch, n_points)
        z = (log_t.unsqueeze(0) - mu) / sigma
    else:
        z = (log_t - mu) / sigma

    # S(t) = 1 - Phi(z) = Phi(-z) = 0.5 * erfc(z / sqrt(2))
    survival = 0.5 * torch.erfc(z / _SQRT2)
    return survival


def compute_hazard_function(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    t_grid: torch.Tensor,
) -> torch.Tensor:
    """
    Compute hazard function h(t) = f(t) / S(t).

    For log-normal:
        h(t) = phi(z) / (t * sigma * Phi(-z))

    where z = (log(t) - mu) / sigma, phi is standard normal PDF.

    Args:
        mu: Location parameter
        sigma: Scale parameter
        t_grid: Time points to evaluate

    Returns:
        h(t) values
    """
    t_grid = torch.clamp(t_grid, min=1e-6)
    sigma = torch.clamp(sigma, min=1e-4)
    log_t = torch.log(t_grid)
    z = (log_t - mu) / sigma

    # Standard normal PDF: phi(z)
    phi_z = torch.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)

    # Standard normal survival: Phi(-z)
    survival_z = 0.5 * torch.erfc(z / _SQRT2)

    hazard = phi_z / (t_grid * sigma * (survival_z + 1e-12))
    return hazard
