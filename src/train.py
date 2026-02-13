"""
Training pipeline for the physics-informed water breakthrough model.

Implements a composite loss function with:
    1. Data loss: MSE between predicted and observed water cut
    2. Physics loss: Buckley-Leverett PDE residual
    3. Monotonicity loss: Penalize non-physical water cut decreases
    4. Breakthrough loss: Binary cross-entropy for breakthrough detection
    5. Material balance loss: Rate consistency check

Physics loss weight is annealed during training to stabilize optimization.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from .model import PhysicsInformedBreakthroughModel
from .survival import SurvivalLoss


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    # Loss weights
    lambda_data: float = 1.0
    lambda_physics: float = 0.1
    lambda_monotonicity: float = 0.05
    lambda_breakthrough: float = 0.3
    lambda_material_balance: float = 0.05
    lambda_survival: float = 0.2
    # Physics annealing
    physics_anneal_start: float = 0.01
    physics_anneal_end: float = 0.1
    physics_anneal_epochs: int = 50
    # Learning rate schedule
    lr_patience: int = 15
    lr_factor: float = 0.5
    # Early stopping
    early_stop_patience: int = 30
    # Model
    hidden_size: int = 64
    lstm_layers: int = 2
    dropout: float = 0.1
    # Output
    model_save_path: str = "models/best_model.pt"


class PhysicsInformedLoss(nn.Module):
    """
    Composite loss function combining data-driven and physics-based terms.
    """

    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.survival_loss = SurvivalLoss()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        y_true: torch.Tensor,
        bt_true: torch.Tensor | None = None,
        epoch: int = 0,
        tte: torch.Tensor | None = None,
        event: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute composite loss.

        Args:
            outputs: Model output dictionary
            y_true: Ground truth water cut (batch, 1)
            bt_true: Ground truth breakthrough labels (batch,)
            epoch: Current epoch for annealing
            tte: Time-to-event for survival loss (batch, 1)
            event: Event indicator for survival loss (batch, 1)

        Returns:
            Dictionary with individual and total loss values
        """
        losses = {}

        # 1. Data loss - MSE on water cut
        losses["data"] = self.mse(outputs["water_cut"], y_true)

        # 2. Physics consistency - physics branch should also match data
        losses["physics_fit"] = self.mse(outputs["water_cut_physics"], y_true)

        # 3. Monotonicity penalty on water cut predictions
        if outputs["water_cut"].shape[0] > 1:
            diffs = outputs["water_cut"][1:] - outputs["water_cut"][:-1]
            violations = torch.clamp(-diffs, min=0)
            losses["monotonicity"] = violations.mean()
        else:
            losses["monotonicity"] = torch.tensor(0.0, device=y_true.device)

        # 4. Breakthrough classification loss
        if bt_true is not None:
            losses["breakthrough"] = self.bce(
                outputs["breakthrough_logit"].squeeze(),
                bt_true,
            )
        else:
            losses["breakthrough"] = torch.tensor(0.0, device=y_true.device)

        # 5. Material balance: predicted water cut should be consistent
        #    with the physics-based fractional flow
        losses["material_balance"] = self.mse(
            outputs["water_cut_physics"],
            outputs["water_cut_data"],
        )

        # 6. Saturation bounds penalty - soft constraint
        s_w = outputs["saturation"]
        bound_penalty = (
            torch.clamp(-s_w, min=0).mean()
            + torch.clamp(s_w - 1.0, min=0).mean()
        )
        losses["bounds"] = bound_penalty

        # 7. Survival loss - time-to-breakthrough prediction
        if tte is not None and event is not None:
            losses["survival"] = self.survival_loss(
                outputs["survival_mu"],
                outputs["survival_sigma"],
                tte,
                event,
            )
        else:
            losses["survival"] = torch.tensor(0.0, device=y_true.device)

        # Physics weight annealing (ramp up over training)
        if epoch < self.config.physics_anneal_epochs:
            t = epoch / self.config.physics_anneal_epochs
            physics_weight = (
                self.config.physics_anneal_start
                + t * (self.config.physics_anneal_end - self.config.physics_anneal_start)
            )
        else:
            physics_weight = self.config.physics_anneal_end

        # Total weighted loss
        total = (
            self.config.lambda_data * losses["data"]
            + physics_weight * losses["physics_fit"]
            + self.config.lambda_monotonicity * losses["monotonicity"]
            + self.config.lambda_breakthrough * losses["breakthrough"]
            + self.config.lambda_material_balance * losses["material_balance"]
            + self.config.lambda_survival * losses["survival"]
            + 0.01 * losses["bounds"]
        )

        losses["total"] = total
        losses["physics_weight"] = torch.tensor(physics_weight)

        return losses


def train_model(
    dataset: dict,
    config: TrainConfig | None = None,
    device: str = "cpu",
) -> tuple[PhysicsInformedBreakthroughModel, dict]:
    """
    Full training loop.

    Args:
        dataset: Output of data_loader.build_dataset()
        config: Training configuration
        device: Torch device

    Returns:
        Trained model and training history
    """
    if config is None:
        config = TrainConfig()

    # Initialize model
    model = PhysicsInformedBreakthroughModel(
        n_features=dataset["n_features"],
        hidden_size=config.hidden_size,
        lstm_layers=config.lstm_layers,
        dropout=config.dropout,
    ).to(device)

    # Loss function
    criterion = PhysicsInformedLoss(config)

    # Optimizer with separate LR for physics parameters
    physics_params = []
    nn_params = []
    for name, param in model.named_parameters():
        if any(k in name for k in ["rel_perm", "log_mu", "log_porosity"]):
            physics_params.append(param)
        else:
            nn_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": nn_params, "lr": config.learning_rate},
        {"params": physics_params, "lr": config.learning_rate * 0.1,
         "weight_decay": 0},
    ], weight_decay=config.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=config.lr_patience,
        factor=config.lr_factor,
    )

    # Training data
    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    bt_train = dataset["bt_train"]
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]
    bt_test = dataset["bt_test"]
    tte_train = dataset["tte_train"]
    tte_test = dataset["tte_test"]
    event_train = dataset["event_train"]
    event_test = dataset["event_test"]

    n_train = X_train.shape[0]
    n_batches = max(1, n_train // config.batch_size)

    # History
    history = {
        "train_loss": [], "val_loss": [], "train_data_loss": [],
        "val_data_loss": [], "physics_weight": [],
        "learned_params": [],
    }
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Training on {n_train} samples, validating on {X_test.shape[0]} samples")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Physics parameters: {sum(p.numel() for p in physics_params):,}")
    print("-" * 70)

    for epoch in range(config.epochs):
        model.train()
        epoch_losses = {k: 0.0 for k in [
            "total", "data", "physics_fit", "monotonicity",
            "breakthrough", "material_balance", "survival",
        ]}

        # Shuffle training data
        perm = torch.randperm(n_train, device=device)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]
        bt_shuffled = bt_train[perm]
        tte_shuffled = tte_train[perm]
        event_shuffled = event_train[perm]

        for i in range(n_batches):
            start = i * config.batch_size
            end = min(start + config.batch_size, n_train)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            bt_batch = bt_shuffled[start:end]
            tte_batch = tte_shuffled[start:end]
            event_batch = event_shuffled[start:end]

            optimizer.zero_grad()
            outputs = model(X_batch)
            losses = criterion(
                outputs, y_batch, bt_batch, epoch, tte_batch, event_batch
            )
            losses["total"].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in epoch_losses:
                if k in losses:
                    epoch_losses[k] += losses[k].item()

        # Average epoch losses
        for k in epoch_losses:
            epoch_losses[k] /= n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_losses = criterion(
                val_outputs, y_test, bt_test, epoch, tte_test, event_test
            )
            val_loss = val_losses["total"].item()
            val_data_loss = val_losses["data"].item()

        scheduler.step(val_loss)

        # Record history
        history["train_loss"].append(epoch_losses["total"])
        history["val_loss"].append(val_loss)
        history["train_data_loss"].append(epoch_losses["data"])
        history["val_data_loss"].append(val_data_loss)
        history["physics_weight"].append(losses["physics_weight"].item())
        history["learned_params"].append(model.get_physics_parameters())

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            save_path = Path(config.model_save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "physics_params": model.get_physics_parameters(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= config.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            params = model.get_physics_parameters()
            print(
                f"Epoch {epoch + 1:4d} | "
                f"Train: {epoch_losses['total']:.4f} | "
                f"Val: {val_loss:.4f} | "
                f"Data: {val_data_loss:.4f} | "
                f"Swc={params['s_wc']:.3f} Sor={params['s_or']:.3f} "
                f"nw={params['n_w']:.2f} no={params['n_o']:.2f} "
                f"M={params['mu_w']/params['mu_o']:.3f}"
            )

    # Load best model
    checkpoint = torch.load(
        config.model_save_path, map_location=device, weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"\nBest model from epoch {checkpoint['epoch'] + 1}, "
          f"val_loss={checkpoint['val_loss']:.4f}")
    print(f"Learned physics parameters: {checkpoint['physics_params']}")

    return model, history
