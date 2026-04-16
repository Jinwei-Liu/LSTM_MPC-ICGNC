"""Visualize or evaluate the pure LSTM trajectory predictor."""

import argparse
import os
import pickle

from vehicle_lstm_lib import (
    VehiclePredictor,
    evaluate_full_dataset,
    load_dataset_from_folder,
    plot_error_trends,
    print_error_statistics,
)


def resolve_model_path(save_dir: str, model_path: str | None) -> str:
    return model_path or os.path.join(save_dir, "best_vehicle_lstm.pth")


def load_scalers(save_dir: str, scalers_path: str | None):
    path = scalers_path or os.path.join(save_dir, "scalers.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scalers not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f), path


def run_visualization(args):
    test_dataset_full = load_dataset_from_folder(args.data_folder, args.test_file)
    scalers, scalers_path = load_scalers(args.save_dir, args.scalers_path)
    model_path = resolve_model_path(args.save_dir, args.model_path)

    predictor = VehiclePredictor(model_path, device=args.device)

    total = len(test_dataset_full["training_sequences"])
    start = min(args.sample_start, max(total - 1, 0))
    shown = 0

    print(f"Loaded scalers from: {scalers_path}")
    print(f"Visualizing up to {args.num_visualizations} samples from index {start}")

    for idx in range(start, total):
        seq = test_dataset_full["training_sequences"][idx]
        pred_states, _ = predictor.predict_trajectory(
            seq["hist_states"],
            seq["hist_controls"],
            scalers["scaler_states"],
            scalers["scaler_controls"],
        )
        print(f"\nVisualizing sample {idx}")
        predictor.visualize_prediction(seq["hist_states"], pred_states, seq["future_states"])
        shown += 1
        if shown >= args.num_visualizations:
            break


def run_evaluation(args):
    print("Loading test dataset and model...")
    test_dataset_full = load_dataset_from_folder(args.data_folder, args.test_file)
    scalers, scalers_path = load_scalers(args.save_dir, args.scalers_path)
    model_path = resolve_model_path(args.save_dir, args.model_path)

    predictor = VehiclePredictor(model_path, device=args.device)
    error_results = evaluate_full_dataset(predictor, test_dataset_full, scalers, device=args.device)
    if not error_results:
        print("Evaluation failed!")
        return

    print(f"Loaded scalers from: {scalers_path}")
    print_error_statistics(error_results)
    plot_error_trends(error_results)

    if args.save_results:
        results_path = os.path.join(args.save_dir, "temporal_error_results.pkl")
        with open(results_path, "wb") as f:
            pickle.dump(error_results, f)
        print(f"Results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Test or evaluate Vehicle LSTM trajectory predictor")
    parser.add_argument("--data_folder", default="vehicle_datasets", help="Folder containing datasets")
    parser.add_argument("--test_file", default="vehicle_test_dataset.pkl", help="Test dataset filename")
    parser.add_argument("--save_dir", default="models", help="Directory containing model artifacts")
    parser.add_argument("--model_path", default=None, help="Optional explicit model path")
    parser.add_argument("--scalers_path", default=None, help="Optional explicit scalers path")
    parser.add_argument(
        "--mode",
        choices=["visualize", "evaluate"],
        default="evaluate",
        help="visualize sample predictions or evaluate the full test set",
    )
    parser.add_argument("--sample_start", type=int, default=700, help="Start index for visualization mode")
    parser.add_argument("--num_visualizations", type=int, default=10, help="Number of samples to visualize")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or cuda")
    parser.add_argument("--save_results", action="store_true", help="Save evaluation results to file")
    args = parser.parse_args()

    print("=== Test Vehicle LSTM ===")
    print(f"Mode: {args.mode}")

    if args.mode == "visualize":
        run_visualization(args)
    else:
        run_evaluation(args)


if __name__ == "__main__":
    main()
