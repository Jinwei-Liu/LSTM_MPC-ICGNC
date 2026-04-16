"""Visualize or evaluate the LSTM+MPC model."""

import argparse
import os
import pickle

import numpy as np

from vehicle_lstm_mpc_lib import (
    VehicleLSTMMPCPredictor,
    evaluate_full_dataset,
    load_dataset_from_folder,
    plot_error_trends,
    print_error_statistics,
)


def resolve_model_path(args) -> str:
    if args.model_path:
        return args.model_path
    return os.path.join(
        args.save_dir,
        f"best_vehicle_lstm_mpc_ds{args.downsample_factor}_targets{args.num_targets}_lqr{args.lqr_iter}.pth",
    )


def build_hist_actions(hist_controls):
    max_accel = 10.0
    max_steer = np.deg2rad(70)
    accelerations = (hist_controls[:, 0] - hist_controls[:, 1]) * max_accel
    steer_angles = hist_controls[:, 2] * max_steer
    return np.column_stack([accelerations, steer_angles])


def run_visualization(args):
    test_dataset_full = load_dataset_from_folder(args.data_folder, args.test_file)
    model_path = resolve_model_path(args)
    predictor = VehicleLSTMMPCPredictor(model_path, device=args.device, lqr_iter=args.lqr_iter)

    total = len(test_dataset_full["training_sequences"])
    if total == 0:
        print("No test samples found.")
        return

    indices = []
    idx = min(args.sample_start, total - 1)
    while idx < total and len(indices) < args.num_visualizations:
        indices.append(idx)
        idx += max(args.sample_step, 1)

    for sample_idx in indices:
        seq = test_dataset_full["training_sequences"][sample_idx]
        hist_states = seq["hist_states"]
        current_state = seq["current_state"]
        future_states = seq["future_states"]
        hist_actions = build_hist_actions(seq["hist_controls"])

        optimal_controls, predicted_states, mpc_params = predictor.predict_control(
            hist_states,
            hist_actions,
            current_state,
        )

        print(f"\nTest sample {sample_idx}")
        print(f"  Current state: {current_state}")
        print(f"  State weights: {mpc_params['state_weights']}")
        print(f"  Control weights: {mpc_params['control_weights']}")
        print(f"  MPC dt: {mpc_params['mpc_dt']:.2f}s")
        print(f"  MPC horizon: {mpc_params['mpc_horizon']} steps")

        ds_indices = np.arange(
            predictor.downsample_factor - 1,
            len(future_states),
            predictor.downsample_factor,
        )
        downsampled_future_states = future_states[ds_indices[: len(predicted_states)]]

        predictor.visualize_prediction_with_heatmap(
            hist_states,
            predicted_states,
            downsampled_future_states,
            mpc_params,
        )

        if args.show_simple_viz:
            predictor.visualize_prediction(
                hist_states,
                predicted_states,
                downsampled_future_states,
                mpc_params,
            )


def run_evaluation(args):
    print("Loading test dataset and model...")
    test_dataset_full = load_dataset_from_folder(args.data_folder, args.test_file)
    model_path = resolve_model_path(args)
    predictor = VehicleLSTMMPCPredictor(model_path, device=args.device, lqr_iter=args.lqr_iter)

    error_results = evaluate_full_dataset(
        predictor,
        test_dataset_full,
        device=args.device,
        batch_size=args.eval_batch_size,
    )
    if not error_results:
        print("Evaluation failed!")
        return

    print_error_statistics(
        error_results,
        downsample_factor=predictor.downsample_factor,
        num_targets=predictor.num_targets,
    )
    plot_error_trends(
        error_results,
        downsample_factor=predictor.downsample_factor,
        num_targets=predictor.num_targets,
    )

    if args.save_results:
        results_path = os.path.join(
            args.save_dir,
            f"lstm_mpc_temporal_error_results_ds{predictor.downsample_factor}_targets{predictor.num_targets}.pkl",
        )
        with open(results_path, "wb") as f:
            pickle.dump(
                {
                    "error_results": error_results,
                    "downsample_factor": predictor.downsample_factor,
                    "num_targets": predictor.num_targets,
                    "mpc_dt": predictor.mpc_solver.mpc_dt,
                    "mpc_horizon": predictor.mpc_solver.mpc_horizon,
                },
                f,
            )
        print(f"Results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Test or evaluate Vehicle LSTM-MPC model")
    parser.add_argument("--data_folder", default="vehicle_datasets", help="Folder containing datasets")
    parser.add_argument("--test_file", default="vehicle_test_dataset.pkl", help="Test dataset filename")
    parser.add_argument("--save_dir", default="models", help="Directory containing model artifacts")
    parser.add_argument("--model_path", default=None, help="Optional explicit model path")
    parser.add_argument("--downsample_factor", type=int, default=10, help="MPC downsample factor")
    parser.add_argument("--num_targets", type=int, default=10, help="Number of target points")
    parser.add_argument("--lqr_iter", type=int, default=10, help="Number of LQR iterations")
    parser.add_argument(
        "--mode",
        choices=["visualize", "evaluate"],
        default="evaluate",
        help="visualize sample predictions or evaluate the full test set",
    )
    parser.add_argument("--sample_start", type=int, default=0, help="Start index for visualization mode")
    parser.add_argument("--sample_step", type=int, default=100, help="Step size between visualization samples")
    parser.add_argument("--num_visualizations", type=int, default=10, help="Number of samples to visualize")
    parser.add_argument("--show_simple_viz", action="store_true", help="Also show the simple visualization")
    parser.add_argument("--eval_batch_size", type=int, default=128, help="Evaluation batch size")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or cuda")
    parser.add_argument("--save_results", action="store_true", help="Save evaluation results to file")
    args = parser.parse_args()

    print("=== Test Vehicle LSTM-MPC ===")
    print(f"Mode: {args.mode}")

    if args.mode == "visualize":
        run_visualization(args)
    else:
        run_evaluation(args)


if __name__ == "__main__":
    main()
