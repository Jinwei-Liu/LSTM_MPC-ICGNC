"""Train the pure LSTM trajectory predictor."""

import argparse
import os
import pickle

from torch.utils.data import DataLoader

from vehicle_lstm_lib import (
    VehicleDataset,
    VehicleLSTM,
    VehicleLSTMTrainer,
    load_dataset_from_folder,
)


def main():
    parser = argparse.ArgumentParser(description="Train Vehicle LSTM trajectory predictor")
    parser.add_argument("--data_folder", default="vehicle_datasets", help="Folder containing datasets")
    parser.add_argument("--train_file", default="vehicle_train_dataset.pkl", help="Training dataset filename")
    parser.add_argument("--val_file", default="vehicle_test_dataset.pkl", help="Validation dataset filename")
    parser.add_argument("--save_dir", default="models", help="Directory to save models and scalers")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="LSTM hidden dimension")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of LSTM layers")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("=== Train Vehicle LSTM ===")
    print(f"Data folder: {args.data_folder}")
    print(f"Train file: {args.train_file}")
    print(f"Validation file: {args.val_file}")
    print(f"Save dir: {args.save_dir}")

    train_dataset_full = load_dataset_from_folder(args.data_folder, args.train_file)
    val_dataset_full = load_dataset_from_folder(args.data_folder, args.val_file)

    train_dataset = VehicleDataset(train_dataset_full["training_sequences"])
    val_dataset = VehicleDataset(
        val_dataset_full["training_sequences"],
        train_dataset.scaler_states,
        train_dataset.scaler_controls,
        is_train=False,
    )

    scalers_path = os.path.join(args.save_dir, "scalers.pkl")
    with open(scalers_path, "wb") as f:
        pickle.dump(
            {
                "scaler_states": train_dataset.scaler_states,
                "scaler_controls": train_dataset.scaler_controls,
            },
            f,
        )
    print(f"Scalers saved to: {scalers_path}")

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

    model = VehicleLSTM(hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    trainer = VehicleLSTMTrainer(model, save_dir=args.save_dir)
    trainer.train_model(
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
