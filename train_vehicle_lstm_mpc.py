"""Train the LSTM+MPC model."""

import argparse
import os

from torch.utils.data import DataLoader

from vehicle_lstm_mpc_lib import (
    VehicleLSTMMPC,
    VehicleLSTMMPCDataset,
    VehicleLSTMMPCTrainer,
    load_dataset_from_folder,
)


def main():
    parser = argparse.ArgumentParser(description="Train Vehicle LSTM-MPC model")
    parser.add_argument("--data_folder", default="vehicle_datasets", help="Folder containing datasets")
    parser.add_argument("--train_file", default="vehicle_train_dataset.pkl", help="Training dataset filename")
    parser.add_argument("--val_file", default="vehicle_test_dataset.pkl", help="Validation dataset filename")
    parser.add_argument("--save_dir", default="models", help="Directory to save models")
    parser.add_argument("--hidden_dim", type=int, default=64, help="LSTM hidden dimension")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of LSTM layers")
    parser.add_argument("--downsample_factor", type=int, default=10, help="MPC downsample factor")
    parser.add_argument("--num_targets", type=int, default=10, help="Number of target points")
    parser.add_argument("--lqr_iter", type=int, default=10, help="Number of LQR iterations")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("=== Train Vehicle LSTM-MPC ===")
    print(f"Data folder: {args.data_folder}")
    print(f"Train file: {args.train_file}")
    print(f"Validation file: {args.val_file}")
    print(f"Save dir: {args.save_dir}")
    print(f"Downsample factor: {args.downsample_factor}")
    print(f"Number of targets: {args.num_targets}")
    print(f"LQR iterations: {args.lqr_iter}")

    train_dataset_full = load_dataset_from_folder(args.data_folder, args.train_file)
    val_dataset_full = load_dataset_from_folder(args.data_folder, args.val_file)

    train_dataset = VehicleLSTMMPCDataset(train_dataset_full["training_sequences"])
    val_dataset = VehicleLSTMMPCDataset(val_dataset_full["training_sequences"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = VehicleLSTMMPC(
        input_dim=6,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        state_dim=4,
        control_dim=2,
        num_targets=args.num_targets,
    )

    trainer = VehicleLSTMMPCTrainer(
        model,
        downsample_factor=args.downsample_factor,
        num_targets=args.num_targets,
        lqr_iter=args.lqr_iter,
    )
    trainer.train_model(
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
