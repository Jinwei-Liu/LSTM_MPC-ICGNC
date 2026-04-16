"""Visualize or evaluate the LSTM+Dynamics baseline model."""

import argparse
import os
import pickle

import numpy as np

from vehicle_lstm_dynamics_lib import (
    VehicleLSTMDynamicsPredictor,
    evaluate_full_dataset,
    load_dataset_from_folder,
    plot_error_trends,
    print_error_statistics,
)


def resolve_model_path(args):
    if args.model_path:
        return args.model_path
    return os.path.join(args.save_dir, 'best_vehicle_lstm_dynamics.pth')


def build_hist_actions(hist_controls):
    max_accel = 10.0
    max_steer = np.deg2rad(70)
    accelerations = (hist_controls[:, 0] - hist_controls[:, 1]) * max_accel
    steer_angles = hist_controls[:, 2] * max_steer
    return np.column_stack([accelerations, steer_angles])


def build_future_actions(future_controls):
    max_accel = 10.0
    max_steer = np.deg2rad(70)
    accelerations = (future_controls[:, 0] - future_controls[:, 1]) * max_accel
    steer_angles = future_controls[:, 2] * max_steer
    return np.column_stack([accelerations, steer_angles])


def run_visualization(args):
    test_dataset_full = load_dataset_from_folder(args.data_folder, args.test_file)
    model_path = resolve_model_path(args)
    predictor = VehicleLSTMDynamicsPredictor(model_path, device=args.device)

    total = len(test_dataset_full['training_sequences'])
    if total == 0:
        print('No test samples found.')
        return

    indices = []
    idx = min(args.sample_start, total - 1)
    while idx < total and len(indices) < args.num_visualizations:
        indices.append(idx)
        idx += max(args.sample_step, 1)

    for sample_idx in indices:
        seq = test_dataset_full['training_sequences'][sample_idx]
        hist_states = seq['hist_states']
        current_state = seq['current_state']
        future_states = seq['future_states']
        hist_actions = build_hist_actions(seq['hist_controls'])
        future_actions = build_future_actions(seq['future_controls'])

        predicted_states, predicted_actions = predictor.predict_trajectory(
            hist_states,
            hist_actions,
            current_state,
        )

        print(f'\nTest sample {sample_idx}')
        print(f'  Current state: {current_state}')
        predictor.visualize_prediction(
            hist_states,
            predicted_states,
            future_states,
            predicted_actions,
            future_actions,
        )


def run_evaluation(args):
    print('Loading test dataset and model...')
    test_dataset_full = load_dataset_from_folder(args.data_folder, args.test_file)
    model_path = resolve_model_path(args)
    predictor = VehicleLSTMDynamicsPredictor(model_path, device=args.device)

    error_results = evaluate_full_dataset(
        predictor,
        test_dataset_full,
        device=args.device,
        batch_size=args.eval_batch_size,
    )
    if not error_results:
        print('Evaluation failed!')
        return

    print_error_statistics(error_results)
    plot_error_trends(error_results)

    if args.save_results:
        results_path = os.path.join(args.save_dir, 'lstm_dynamics_temporal_error_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(
                {
                    'error_results': error_results,
                    'dt': predictor.dt,
                },
                f,
            )
        print(f'Results saved to: {results_path}')


def main():
    parser = argparse.ArgumentParser(description='Test or evaluate Vehicle LSTM-Dynamics model')
    parser.add_argument('--data_folder', default='vehicle_datasets', help='Folder containing datasets')
    parser.add_argument('--test_file', default='vehicle_test_dataset.pkl', help='Test dataset filename')
    parser.add_argument('--save_dir', default='models', help='Directory containing model artifacts')
    parser.add_argument('--model_path', default=None, help='Optional explicit model path')
    parser.add_argument(
        '--mode',
        choices=['visualize', 'evaluate'],
        default='evaluate',
        help='visualize sample predictions or evaluate the full test set',
    )
    parser.add_argument('--sample_start', type=int, default=0, help='Start index for visualization mode')
    parser.add_argument('--sample_step', type=int, default=100, help='Step size between visualization samples')
    parser.add_argument('--num_visualizations', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--eval_batch_size', type=int, default=128, help='Evaluation batch size')
    parser.add_argument('--device', default='cpu', help='Inference device, e.g. cpu or cuda')
    parser.add_argument('--save_results', action='store_true', help='Save evaluation results to file')
    args = parser.parse_args()

    print('=== Test Vehicle LSTM-Dynamics ===')
    print(f'Mode: {args.mode}')

    if args.mode == 'visualize':
        run_visualization(args)
    else:
        run_evaluation(args)


if __name__ == '__main__':
    main()
