"""
Evaluation and visualization for the water breakthrough prediction model.

Provides:
- Quantitative metrics (MAE, RMSE, R², breakthrough accuracy)
- Training history plots
- Water cut prediction vs actual
- Learned fractional flow curves
- Physics parameter evolution during training
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from .model import PhysicsInformedBreakthroughModel
from .physics import compute_breakthrough_time_analytical
from .survival import compute_survival_function, compute_hazard_function


def evaluate_model(
    model: PhysicsInformedBreakthroughModel,
    dataset: dict,
    device: str = "cpu",
) -> dict:
    """
    Compute evaluation metrics on test set.

    Returns dict with regression and classification metrics.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(dataset["X_test"].to(device))

    y_true = dataset["y_test"].cpu().numpy().flatten()
    y_pred = outputs["water_cut"].cpu().numpy().flatten()

    # Inverse transform if scalers available
    scaler_y = dataset.get("scaler_y")
    if scaler_y is not None:
        y_true_orig = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred

    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mae_original_scale": float(mean_absolute_error(y_true_orig, y_pred_orig)),
        "rmse_original_scale": float(
            np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
        ),
    }

    # Breakthrough classification metrics
    if "bt_test" in dataset:
        bt_true = dataset["bt_test"].cpu().numpy()
        bt_logit = outputs["breakthrough_logit"].cpu().numpy().flatten()
        bt_pred = (bt_logit > 0).astype(float)

        metrics["bt_accuracy"] = float(accuracy_score(bt_true, bt_pred))
        metrics["bt_precision"] = float(
            precision_score(bt_true, bt_pred, zero_division=0)
        )
        metrics["bt_recall"] = float(
            recall_score(bt_true, bt_pred, zero_division=0)
        )
        metrics["bt_f1"] = float(f1_score(bt_true, bt_pred, zero_division=0))

    # Physics parameters
    metrics["physics_params"] = model.get_physics_parameters()

    # Gate statistics (how much physics vs data-driven)
    gate_values = outputs["gate_value"].cpu().numpy().flatten()
    metrics["gate_mean"] = float(gate_values.mean())
    metrics["gate_std"] = float(gate_values.std())

    # Survival model metrics
    if "survival_mu" in outputs:
        percentiles = outputs["survival_percentiles"]
        p10 = percentiles["P10"].cpu().numpy().flatten()
        p50 = percentiles["P50"].cpu().numpy().flatten()
        p90 = percentiles["P90"].cpu().numpy().flatten()

        metrics["survival_p10_mean"] = float(p10.mean())
        metrics["survival_p50_mean"] = float(p50.mean())
        metrics["survival_p90_mean"] = float(p90.mean())
        metrics["survival_p10_std"] = float(p10.std())
        metrics["survival_p50_std"] = float(p50.std())
        metrics["survival_p90_std"] = float(p90.std())

        mu = outputs["survival_mu"].cpu().numpy().flatten()
        sigma = outputs["survival_sigma"].cpu().numpy().flatten()
        metrics["survival_mu_mean"] = float(mu.mean())
        metrics["survival_sigma_mean"] = float(sigma.mean())

        # Calibration: fraction of observed events within P10-P90 interval
        if "tte_test" in dataset and "event_test" in dataset:
            tte_true = dataset["tte_test"].cpu().numpy().flatten()
            event_true = dataset["event_test"].cpu().numpy().flatten()
            observed_mask = event_true == 1.0
            if observed_mask.sum() > 0:
                tte_observed = tte_true[observed_mask]
                p10_obs = p10[observed_mask]
                p90_obs = p90[observed_mask]
                in_interval = (
                    (tte_observed >= p10_obs) & (tte_observed <= p90_obs)
                )
                metrics["survival_calibration_80"] = float(
                    in_interval.mean()
                )

    return metrics


def print_metrics(metrics: dict) -> None:
    """Print evaluation metrics in a formatted table."""
    print("\n" + "=" * 60)
    print("  EVALUATION METRICS")
    print("=" * 60)

    print("\n  Water Cut Prediction (scaled):")
    print(f"    MAE:  {metrics['mae']:.4f}")
    print(f"    RMSE: {metrics['rmse']:.4f}")
    print(f"    R²:   {metrics['r2']:.4f}")

    print(f"\n  Water Cut Prediction (original scale):")
    print(f"    MAE:  {metrics['mae_original_scale']:.4f}")
    print(f"    RMSE: {metrics['rmse_original_scale']:.4f}")

    if "bt_accuracy" in metrics:
        print(f"\n  Breakthrough Detection:")
        print(f"    Accuracy:  {metrics['bt_accuracy']:.4f}")
        print(f"    Precision: {metrics['bt_precision']:.4f}")
        print(f"    Recall:    {metrics['bt_recall']:.4f}")
        print(f"    F1 Score:  {metrics['bt_f1']:.4f}")

    print(f"\n  Physics-Data Gate:")
    print(f"    Mean gate value: {metrics['gate_mean']:.3f} "
          f"(1=physics, 0=data-driven)")
    print(f"    Gate std: {metrics['gate_std']:.3f}")

    if "physics_params" in metrics:
        print(f"\n  Learned Reservoir Parameters:")
        p = metrics["physics_params"]
        print(f"    S_wc (connate water):   {p['s_wc']:.3f}")
        print(f"    S_or (residual oil):    {p['s_or']:.3f}")
        print(f"    kr_w_max:               {p['kr_w_max']:.3f}")
        print(f"    kr_o_max:               {p['kr_o_max']:.3f}")
        print(f"    n_w (Corey water):      {p['n_w']:.2f}")
        print(f"    n_o (Corey oil):        {p['n_o']:.2f}")
        print(f"    mu_w (water viscosity): {p['mu_w']:.3f} cP")
        print(f"    mu_o (oil viscosity):   {p['mu_o']:.3f} cP")
        print(f"    Mobility ratio M:       {p['mu_w']/p['mu_o']:.3f}")
        print(f"    Porosity:               {p['porosity']:.3f}")

    if "survival_p50_mean" in metrics:
        print(f"\n  Survival Model (Time-to-Breakthrough):")
        print(f"    P10 (optimistic):   {metrics['survival_p10_mean']:.4f} "
              f"(+/- {metrics['survival_p10_std']:.4f})")
        print(f"    P50 (median):       {metrics['survival_p50_mean']:.4f} "
              f"(+/- {metrics['survival_p50_std']:.4f})")
        print(f"    P90 (conservative): {metrics['survival_p90_mean']:.4f} "
              f"(+/- {metrics['survival_p90_std']:.4f})")
        print(f"    mu (location):      {metrics['survival_mu_mean']:.4f}")
        print(f"    sigma (scale):      {metrics['survival_sigma_mean']:.4f}")
        if "survival_calibration_80" in metrics:
            cal = metrics["survival_calibration_80"]
            print(f"    80% interval calibration: {cal:.1%} "
                  f"(ideal: 80%)")

    print("=" * 60)


def plot_training_history(
    history: dict, save_dir: str = "results"
) -> None:
    """Plot training and validation loss curves."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Total loss
    ax = axes[0, 0]
    ax.plot(history["train_loss"], label="Train", linewidth=1.5)
    ax.plot(history["val_loss"], label="Validation", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss (Physics + Data)")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Data loss only
    ax = axes[0, 1]
    ax.plot(history["train_data_loss"], label="Train", linewidth=1.5)
    ax.plot(history["val_data_loss"], label="Validation", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Data Loss (Water Cut MSE)")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Physics weight annealing
    ax = axes[1, 0]
    ax.plot(history["physics_weight"], linewidth=1.5, color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight")
    ax.set_title("Physics Loss Weight (Annealing)")
    ax.grid(True, alpha=0.3)

    # Learned parameters evolution
    ax = axes[1, 1]
    params_history = history.get("learned_params", [])
    if params_history:
        epochs = range(len(params_history))
        s_wc = [p["s_wc"] for p in params_history]
        s_or = [p["s_or"] for p in params_history]
        n_w = [p["n_w"] for p in params_history]
        n_o = [p["n_o"] for p in params_history]
        ax.plot(epochs, s_wc, label="S_wc", linewidth=1.5)
        ax.plot(epochs, s_or, label="S_or", linewidth=1.5)
        ax.plot(epochs, [n / 6 for n in n_w], label="n_w/6", linewidth=1.5)
        ax.plot(epochs, [n / 6 for n in n_o], label="n_o/6", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_title("Learned Physics Parameters")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / "training_history.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path / 'training_history.png'}")


def plot_predictions(
    model: PhysicsInformedBreakthroughModel,
    dataset: dict,
    save_dir: str = "results",
    device: str = "cpu",
) -> None:
    """Plot predicted vs actual water cut on test set."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        outputs = model(dataset["X_test"].to(device))

    y_true = dataset["y_test"].cpu().numpy().flatten()
    y_pred = outputs["water_cut"].cpu().numpy().flatten()
    y_physics = outputs["water_cut_physics"].cpu().numpy().flatten()
    y_data = outputs["water_cut_data"].cpu().numpy().flatten()
    gate = outputs["gate_value"].cpu().numpy().flatten()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Time series comparison
    ax = axes[0, 0]
    n_plot = min(500, len(y_true))
    ax.plot(range(n_plot), y_true[:n_plot], label="Actual", linewidth=1.2, alpha=0.8)
    ax.plot(range(n_plot), y_pred[:n_plot], label="Predicted (blended)",
            linewidth=1.2, alpha=0.8)
    ax.plot(range(n_plot), y_physics[:n_plot], label="Physics only",
            linewidth=0.8, alpha=0.5, linestyle="--")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Water Cut (scaled)")
    ax.set_title("Water Cut: Actual vs Predicted")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Scatter plot
    ax = axes[0, 1]
    ax.scatter(y_true, y_pred, alpha=0.3, s=10)
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Water Cut")
    ax.set_ylabel("Predicted Water Cut")
    ax.set_title("Prediction Scatter Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gate values
    ax = axes[1, 0]
    ax.plot(range(n_plot), gate[:n_plot], linewidth=1, color="purple")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Gate Value")
    ax.set_title("Physics vs Data-Driven Gate (1=Physics, 0=Data)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Breakthrough detection
    ax = axes[1, 1]
    if "bt_test" in dataset:
        bt_true = dataset["bt_test"].cpu().numpy()
        bt_logit = outputs["breakthrough_logit"].cpu().numpy().flatten()
        bt_prob = 1 / (1 + np.exp(-bt_logit))  # sigmoid

        ax.plot(range(n_plot), bt_true[:n_plot], label="Actual",
                linewidth=1.2, alpha=0.8)
        ax.plot(range(n_plot), bt_prob[:n_plot], label="Predicted probability",
                linewidth=1.2, alpha=0.8)
        ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.5,
                    label="Decision boundary")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Breakthrough")
        ax.set_title("Breakthrough Detection")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / "predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path / 'predictions.png'}")


def plot_fractional_flow_curve(
    model: PhysicsInformedBreakthroughModel,
    save_dir: str = "results",
) -> None:
    """
    Plot the learned fractional flow curve f_w(S_w) and compare
    with analytical Buckley-Leverett solution.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    params = model.get_physics_parameters()

    # Analytical solution
    analytical = compute_breakthrough_time_analytical(
        s_wc=params["s_wc"],
        s_or=params["s_or"],
        mu_w=params["mu_w"],
        mu_o=params["mu_o"],
        n_w=params["n_w"],
        n_o=params["n_o"],
    )

    # Learned model curve
    model.eval()
    s_w_tensor = torch.linspace(
        params["s_wc"], 1 - params["s_or"], 200
    ).unsqueeze(1)

    with torch.no_grad():
        f_w_learned = model.physics_branch.fractional_flow(s_w_tensor)
        kr_w, kr_o = model.physics_branch.rel_perm(s_w_tensor)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Fractional flow curve
    ax = axes[0]
    ax.plot(
        analytical["saturation_grid"],
        analytical["fractional_flow_curve"],
        "b-", linewidth=2, label="Learned f_w(S_w)",
    )
    ax.axvline(
        x=analytical["s_w_front"],
        color="r", linestyle="--", alpha=0.7,
        label=f"Front S_wf={analytical['s_w_front']:.3f}",
    )
    # Welge tangent
    s_wc = params["s_wc"]
    slope = analytical["f_w_front"] / (analytical["s_w_front"] - s_wc)
    s_tangent = np.linspace(s_wc, analytical["s_w_front"], 50)
    f_tangent = slope * (s_tangent - s_wc)
    ax.plot(s_tangent, f_tangent, "g--", linewidth=1.5, label="Welge tangent")
    ax.set_xlabel("Water Saturation S_w")
    ax.set_ylabel("Fractional Flow f_w")
    ax.set_title("Buckley-Leverett Fractional Flow")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Relative permeability
    ax = axes[1]
    s_np = s_w_tensor.numpy().flatten()
    ax.plot(s_np, kr_w.numpy().flatten(), "b-", linewidth=2, label="kr_w")
    ax.plot(s_np, kr_o.numpy().flatten(), "r-", linewidth=2, label="kr_o")
    ax.set_xlabel("Water Saturation S_w")
    ax.set_ylabel("Relative Permeability")
    ax.set_title(
        f"Corey Relative Permeability\n"
        f"n_w={params['n_w']:.2f}, n_o={params['n_o']:.2f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # df/dS (shock speed)
    ax = axes[2]
    ax.plot(
        analytical["saturation_grid"],
        analytical["df_ds"],
        "b-", linewidth=2,
    )
    ax.axvline(
        x=analytical["s_w_front"],
        color="r", linestyle="--", alpha=0.7,
        label=f"BT at {analytical['breakthrough_pv']:.2f} PV",
    )
    ax.set_xlabel("Water Saturation S_w")
    ax.set_ylabel("df_w/dS_w")
    ax.set_title("Fractional Flow Derivative (Shock Speed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        save_path / "fractional_flow.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"Saved: {save_path / 'fractional_flow.png'}")
    print(f"Analytical breakthrough time: {analytical['breakthrough_pv']:.3f} PV")
    print(f"Front saturation: {analytical['s_w_front']:.3f}")


def plot_survival_analysis(
    model: PhysicsInformedBreakthroughModel,
    dataset: dict,
    save_dir: str = "results",
    device: str = "cpu",
) -> None:
    """
    Plot survival analysis results: percentile predictions, survival curves,
    and hazard function.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        outputs = model(dataset["X_test"].to(device))

    percentiles = outputs["survival_percentiles"]
    p10 = percentiles["P10"].cpu().numpy().flatten()
    p50 = percentiles["P50"].cpu().numpy().flatten()
    p90 = percentiles["P90"].cpu().numpy().flatten()
    mu_vals = outputs["survival_mu"].cpu()
    sigma_vals = outputs["survival_sigma"].cpu()

    has_tte = "tte_test" in dataset and "event_test" in dataset
    if has_tte:
        tte_true = dataset["tte_test"].cpu().numpy().flatten()
        event_true = dataset["event_test"].cpu().numpy().flatten()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Percentile predictions over samples
    ax = axes[0, 0]
    n_plot = min(500, len(p50))
    x_range = range(n_plot)
    ax.fill_between(
        x_range, p10[:n_plot], p90[:n_plot],
        alpha=0.3, color="steelblue", label="P10-P90 interval",
    )
    ax.plot(x_range, p50[:n_plot], color="steelblue", linewidth=1.5,
            label="P50 (median)")
    if has_tte:
        observed = event_true[:n_plot] == 1.0
        ax.scatter(
            np.where(observed)[0], tte_true[:n_plot][observed],
            s=8, color="red", alpha=0.6, label="Observed BT time",
            zorder=5,
        )
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Time to Breakthrough (normalized)")
    ax.set_title("Predicted Time-to-Breakthrough with Uncertainty")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Survival curve (using mean predicted parameters)
    ax = axes[0, 1]
    mu_mean = mu_vals.mean()
    sigma_mean = sigma_vals.mean()
    t_grid = torch.linspace(0.01, 2.0, 200)
    survival = compute_survival_function(
        mu_mean.unsqueeze(0), sigma_mean.unsqueeze(0), t_grid
    ).numpy().flatten()
    ax.plot(t_grid.numpy(), survival, color="steelblue", linewidth=2)

    # Mark P10, P50, P90 on the survival curve
    for pval, label, ls in [
        (0.90, "P10", "--"), (0.50, "P50", "-"), (0.10, "P90", "--")
    ]:
        ax.axhline(y=pval, color="gray", linestyle=ls, alpha=0.4)
        # Find corresponding time
        idx = np.searchsorted(-survival, -pval)
        if idx < len(t_grid):
            t_pct = t_grid[idx].item()
            ax.axvline(x=t_pct, color="gray", linestyle=ls, alpha=0.4)
            ax.annotate(
                label, xy=(t_pct, pval),
                xytext=(t_pct + 0.05, pval + 0.03),
                fontsize=9,
            )

    ax.set_xlabel("Time (normalized)")
    ax.set_ylabel("Survival Probability S(t)")
    ax.set_title("Mean Survival Curve")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # 3. Hazard function
    ax = axes[1, 0]
    hazard = compute_hazard_function(
        mu_mean.unsqueeze(0), sigma_mean.unsqueeze(0), t_grid
    ).numpy().flatten()
    ax.plot(t_grid.numpy(), hazard, color="darkred", linewidth=2)
    ax.set_xlabel("Time (normalized)")
    ax.set_ylabel("Hazard Rate h(t)")
    ax.set_title("Mean Hazard Function")
    ax.grid(True, alpha=0.3)

    # 4. P10/P50/P90 distribution histograms
    ax = axes[1, 1]
    ax.hist(p10, bins=30, alpha=0.5, color="green", label="P10", density=True)
    ax.hist(p50, bins=30, alpha=0.5, color="steelblue", label="P50", density=True)
    ax.hist(p90, bins=30, alpha=0.5, color="orange", label="P90", density=True)
    if has_tte:
        observed_tte = tte_true[event_true == 1.0]
        if len(observed_tte) > 0:
            ax.hist(
                observed_tte, bins=30, alpha=0.4, color="red",
                label="Observed", density=True,
            )
    ax.set_xlabel("Time to Breakthrough (normalized)")
    ax.set_ylabel("Density")
    ax.set_title("Percentile Distributions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        save_path / "survival_analysis.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"Saved: {save_path / 'survival_analysis.png'}")


def generate_report(
    model: PhysicsInformedBreakthroughModel,
    dataset: dict,
    history: dict,
    save_dir: str = "results",
    device: str = "cpu",
) -> dict:
    """
    Generate full evaluation report with metrics and plots.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Compute metrics
    metrics = evaluate_model(model, dataset, device)
    print_metrics(metrics)

    # Generate plots
    plot_training_history(history, save_dir)
    plot_predictions(model, dataset, save_dir, device)
    plot_fractional_flow_curve(model, save_dir)
    plot_survival_analysis(model, dataset, save_dir, device)

    # Save metrics to file
    with open(save_path / "metrics.txt", "w") as f:
        f.write("Water Breakthrough Prediction - Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        for key, value in metrics.items():
            if isinstance(value, dict):
                f.write(f"\n{key}:\n")
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{key}: {value}\n")

    print(f"\nSaved metrics to {save_path / 'metrics.txt'}")
    return metrics
