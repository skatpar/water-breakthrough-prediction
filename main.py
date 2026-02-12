"""
Physics-Informed Water Breakthrough Prediction

Main entry point for training and evaluating the PINN model
on the Volve production dataset.

Usage:
    python main.py                              # Run with synthetic data
    python main.py --data path/to/volve.csv     # Run with real Volve data
    python main.py --epochs 300 --hidden 128    # Custom hyperparameters
    python main.py --evaluate models/best.pt    # Evaluate saved model

Dataset: https://www.kaggle.com/datasets/lamyalbert/volve-production-data
"""

import argparse
import sys
import torch
import numpy as np

from src.data_loader import build_dataset
from src.train import train_model, TrainConfig
from src.evaluate import generate_report
from src.physics import compute_breakthrough_time_analytical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Physics-Informed Neural Network for Water Breakthrough Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to Volve production CSV. If not provided, uses synthetic data.",
    )
    parser.add_argument(
        "--seq-length", type=int, default=30,
        help="Temporal sequence length for LSTM input.",
    )
    parser.add_argument(
        "--test-fraction", type=float, default=0.2,
        help="Fraction of data held out for testing.",
    )

    # Model architecture
    parser.add_argument("--hidden", type=int, default=64, help="LSTM hidden size.")
    parser.add_argument("--lstm-layers", type=int, default=2, help="LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")

    # Training
    parser.add_argument("--epochs", type=int, default=200, help="Max training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--physics-weight", type=float, default=0.1,
        help="Final physics loss weight (after annealing).",
    )

    # Output
    parser.add_argument(
        "--save-dir", type=str, default="results",
        help="Directory for output plots and metrics.",
    )
    parser.add_argument(
        "--model-path", type=str, default="models/best_model.pt",
        help="Path to save/load model checkpoint.",
    )

    # Evaluate only
    parser.add_argument(
        "--evaluate", type=str, default=None,
        help="Path to saved model for evaluation only (skip training).",
    )

    # Device
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'cpu', 'cuda', or 'auto'.",
    )

    # Seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Build dataset
    print("\n--- Building Dataset ---")
    dataset = build_dataset(
        data_path=args.data,
        seq_length=args.seq_length,
        test_fraction=args.test_fraction,
        device=device,
    )
    print(f"Training samples: {dataset['X_train'].shape[0]}")
    print(f"Test samples: {dataset['X_test'].shape[0]}")
    print(f"Features: {dataset['n_features']}")
    print(f"Wells: {dataset['well_names']}")

    if args.evaluate:
        # Evaluation only mode
        print(f"\n--- Loading model from {args.evaluate} ---")
        from src.model import PhysicsInformedBreakthroughModel

        checkpoint = torch.load(args.evaluate, map_location=device, weights_only=False)
        model = PhysicsInformedBreakthroughModel(
            n_features=dataset["n_features"],
            hidden_size=args.hidden,
            lstm_layers=args.lstm_layers,
            dropout=args.dropout,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model loaded successfully.")

        metrics = generate_report(
            model, dataset, history={}, save_dir=args.save_dir, device=device,
        )
    else:
        # Training mode
        config = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            hidden_size=args.hidden,
            lstm_layers=args.lstm_layers,
            dropout=args.dropout,
            physics_anneal_end=args.physics_weight,
            model_save_path=args.model_path,
        )

        print("\n--- Training Model ---")
        model, history = train_model(dataset, config, device)

        print("\n--- Evaluating Model ---")
        metrics = generate_report(
            model, dataset, history, save_dir=args.save_dir, device=device,
        )

    # Analytical comparison
    print("\n--- Analytical Buckley-Leverett Solution ---")
    params = model.get_physics_parameters() if not args.evaluate else checkpoint.get(
        "physics_params", {}
    )
    if params:
        analytical = compute_breakthrough_time_analytical(
            s_wc=params.get("s_wc", 0.2),
            s_or=params.get("s_or", 0.2),
            mu_w=params.get("mu_w", 0.5),
            mu_o=params.get("mu_o", 2.0),
            n_w=params.get("n_w", 3.0),
            n_o=params.get("n_o", 2.0),
        )
        print(f"Breakthrough time: {analytical['breakthrough_pv']:.3f} pore volumes")
        print(f"Front saturation: {analytical['s_w_front']:.3f}")
        print(f"Avg saturation behind front: {analytical['s_w_avg_behind_front']:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
