"""Evaluate all trained runs with unified ADE/FDE metrics and export summary tables."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vehicle_lstm_lib import VehicleDataset, VehiclePredictor
from vehicle_lstm_dynamics_lib import VehicleLSTMDynamicsDataset, VehicleLSTMDynamicsPredictor


MODEL_LABELS = {
    'lstm': 'LSTM',
    'lstm_mpc': 'LSTM+MPC',
    'lstm_dynamics': 'LSTM+Dynamics',
}


def resolve_device(device_arg: str) -> str:
    if device_arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_arg


def load_pickle_dataset(data_folder: str, file_name: str):
    dataset_path = Path(data_folder) / file_name
    if not dataset_path.exists():
        raise FileNotFoundError(f'Dataset not found: {dataset_path}')
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    print(f'Dataset loaded: {dataset_path}')
    print(f"  Sequences: {len(dataset['training_sequences'])}")
    return dataset


def maybe_downsample_np(array: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return array
    return array[:, factor - 1::factor, :]


def compute_ade_fde(pred_states: np.ndarray, true_states: np.ndarray) -> dict:
    position_errors = np.linalg.norm(pred_states[:, :, :2] - true_states[:, :, :2], axis=2)
    ade_per_sample = position_errors.mean(axis=1)
    fde_per_sample = position_errors[:, -1]
    return {
        'ade_mean': float(np.mean(ade_per_sample)),
        'ade_sample_std': float(np.std(ade_per_sample, ddof=1)) if len(ade_per_sample) > 1 else 0.0,
        'fde_mean': float(np.mean(fde_per_sample)),
        'fde_sample_std': float(np.std(fde_per_sample, ddof=1)) if len(fde_per_sample) > 1 else 0.0,
        'num_samples': int(len(ade_per_sample)),
    }


def make_loader(dataset, batch_size: int, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def resolve_mpc_model_path(run_dir: Path) -> Path:
    candidates = sorted(glob.glob(str(run_dir / 'best_vehicle_lstm_mpc*.pth')))
    if not candidates:
        raise FileNotFoundError(f'No LSTM-MPC checkpoint found in {run_dir}')
    return Path(candidates[0])


def evaluate_lstm_run(run_dir: Path, test_full, batch_size: int, num_workers: int, device: str, common_ds: int):
    model_path = run_dir / 'best_vehicle_lstm.pth'
    scalers_path = run_dir / 'scalers.pkl'
    if not model_path.exists():
        raise FileNotFoundError(f'Model not found: {model_path}')
    if not scalers_path.exists():
        raise FileNotFoundError(f'Scalers not found: {scalers_path}')

    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)

    predictor = VehiclePredictor(str(model_path), device=device)
    dataset = VehicleDataset(
        test_full['training_sequences'],
        scalers['scaler_states'],
        scalers['scaler_controls'],
        is_train=False,
    )
    loader = make_loader(dataset, batch_size, num_workers)

    all_pred_states = []
    all_true_states = []

    predictor.model.eval()
    with torch.no_grad():
        for batch in loader:
            input_features = batch['input_features'].to(device)
            pred_states, _ = predictor.model(input_features)

            batch_size_now, horizon, state_dim = pred_states.shape
            pred_np = pred_states.cpu().numpy().reshape(-1, state_dim)
            true_np = batch['future_states'].cpu().numpy().reshape(-1, state_dim)

            pred_np = scalers['scaler_states'].inverse_transform(pred_np).reshape(batch_size_now, horizon, state_dim)
            true_np = scalers['scaler_states'].inverse_transform(true_np).reshape(batch_size_now, horizon, state_dim)

            all_pred_states.append(maybe_downsample_np(pred_np, common_ds))
            all_true_states.append(maybe_downsample_np(true_np, common_ds))

    return compute_ade_fde(np.concatenate(all_pred_states, axis=0), np.concatenate(all_true_states, axis=0))


def evaluate_lstm_dynamics_run(run_dir: Path, test_full, batch_size: int, num_workers: int, device: str, common_ds: int):
    model_path = run_dir / 'best_vehicle_lstm_dynamics.pth'
    if not model_path.exists():
        raise FileNotFoundError(f'Model not found: {model_path}')

    predictor = VehicleLSTMDynamicsPredictor(str(model_path), device=device)
    dataset = VehicleLSTMDynamicsDataset(test_full['training_sequences'])
    loader = make_loader(dataset, batch_size, num_workers)

    all_pred_states = []
    all_true_states = []

    predictor.model.eval()
    with torch.no_grad():
        for batch in loader:
            input_seq = batch['input_seq'].to(device)
            current_state = batch['current_state'].to(device)
            future_states = batch['future_states'].cpu().numpy()

            pred_states, _ = predictor.model(input_seq, current_state)
            pred_np = pred_states.cpu().numpy()

            all_pred_states.append(maybe_downsample_np(pred_np, common_ds))
            all_true_states.append(maybe_downsample_np(future_states, common_ds))

    return compute_ade_fde(np.concatenate(all_pred_states, axis=0), np.concatenate(all_true_states, axis=0))


def evaluate_lstm_mpc_run(run_dir: Path, test_full, batch_size: int, num_workers: int, device: str, common_ds: int | None):
    from vehicle_lstm_mpc_lib import VehicleLSTMMPCDataset, VehicleLSTMMPCPredictor

    model_path = resolve_mpc_model_path(run_dir)
    predictor = VehicleLSTMMPCPredictor(str(model_path), device=device)

    if common_ds is not None and predictor.downsample_factor != common_ds:
        raise ValueError(
            f'Run {run_dir} uses MPC downsample_factor={predictor.downsample_factor}, '
            f'but common_downsample_factor={common_ds}. Please keep them consistent for fair comparison.'
        )

    dataset = VehicleLSTMMPCDataset(test_full['training_sequences'])
    loader = make_loader(dataset, batch_size, num_workers)

    all_pred_states = []
    all_true_states = []

    predictor.model.eval()
    with torch.no_grad():
        for batch in loader:
            input_seq = batch['input_seq'].to(device)
            current_state = batch['current_state'].to(device)
            future_states = batch['future_states'].to(device)

            state_weights, control_weights, target = predictor.model(input_seq)
            pred_states, _ = predictor.mpc_solver.solve(current_state, state_weights, control_weights, target)

            ds = predictor.downsample_factor
            indices = torch.arange(ds - 1, future_states.size(1), ds, device=device)
            indices = indices[:pred_states.size(1)]
            true_states = future_states[:, indices, :]
            pred_states = pred_states[:, :len(indices), :]

            all_pred_states.append(pred_states.cpu().numpy())
            all_true_states.append(true_states.cpu().numpy())

    return compute_ade_fde(np.concatenate(all_pred_states, axis=0), np.concatenate(all_true_states, axis=0))


def summarize_results(per_run_rows: list[dict]) -> list[dict]:
    summary = []
    for model_key in ['lstm', 'lstm_mpc', 'lstm_dynamics']:
        rows = [row for row in per_run_rows if row['model'] == model_key]
        if not rows:
            continue
        ade_vals = [row['ade_mean'] for row in rows]
        fde_vals = [row['fde_mean'] for row in rows]
        summary.append({
            'model': model_key,
            'label': MODEL_LABELS[model_key],
            'runs': len(rows),
            'ade_mean': float(np.mean(ade_vals)),
            'ade_std_over_runs': float(np.std(ade_vals, ddof=1)) if len(ade_vals) > 1 else 0.0,
            'fde_mean': float(np.mean(fde_vals)),
            'fde_std_over_runs': float(np.std(fde_vals, ddof=1)) if len(fde_vals) > 1 else 0.0,
        })
    return summary


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_markdown_table(summary_rows: list[dict]) -> str:
    lines = [
        '| Method | ADE (m) | FDE (m) |',
        '|---|---:|---:|',
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['label']} | {row['ade_mean']:.4f} ± {row['ade_std_over_runs']:.4f} | "
            f"{row['fde_mean']:.4f} ± {row['fde_std_over_runs']:.4f} |"
        )
    return '\n'.join(lines) + '\n'


def format_latex_table(summary_rows: list[dict]) -> str:
    lines = [
        '\\begin{tabular}{lcc}',
        '\\toprule',
        'Method & ADE (m) & FDE (m) \\\\',
        '\\midrule',
    ]
    for row in summary_rows:
        lines.append(
            f"{row['label']} & {row['ade_mean']:.4f} $\\pm$ {row['ade_std_over_runs']:.4f} & "
            f"{row['fde_mean']:.4f} $\\pm$ {row['fde_std_over_runs']:.4f} \\\\"
        )
    lines.extend(['\\bottomrule', '\\end{tabular}', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Evaluate all trained runs with unified ADE/FDE metrics')
    parser.add_argument('--experiment_root', default='experiments_multirun', help='Root directory produced by train_all_models.py')
    parser.add_argument('--data_folder', default='vehicle_datasets', help='Folder containing datasets')
    parser.add_argument('--test_file', default='vehicle_test_dataset.pkl', help='Test dataset filename')
    parser.add_argument('--batch_size', type=int, default=128, help='Evaluation batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--device', default='auto', help='auto, cpu, or cuda')
    parser.add_argument(
        '--common_downsample_factor',
        type=int,
        default=10,
        help='Downsample factor used to compare all models fairly; should match the LSTM-MPC setting',
    )
    parser.add_argument('--repeats', type=int, default=3, help='Expected number of runs per model')
    args = parser.parse_args()

    device = resolve_device(args.device)
    experiment_root = Path(args.experiment_root)
    summary_dir = experiment_root / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)

    test_full = load_pickle_dataset(args.data_folder, args.test_file)

    evaluators = {
        'lstm': lambda run_dir: evaluate_lstm_run(run_dir, test_full, args.batch_size, args.num_workers, device, args.common_downsample_factor),
        'lstm_mpc': lambda run_dir: evaluate_lstm_mpc_run(run_dir, test_full, args.batch_size, args.num_workers, device, args.common_downsample_factor),
        'lstm_dynamics': lambda run_dir: evaluate_lstm_dynamics_run(run_dir, test_full, args.batch_size, args.num_workers, device, args.common_downsample_factor),
    }

    per_run_rows = []
    for model_key in ['lstm', 'lstm_mpc', 'lstm_dynamics']:
        print('\n' + '=' * 90)
        print(f"Evaluating model: {MODEL_LABELS[model_key]}")
        print('=' * 90)
        for run_idx in range(1, args.repeats + 1):
            run_dir = experiment_root / model_key / f'run_{run_idx:02d}'
            if not run_dir.exists():
                print(f'Skip missing run directory: {run_dir}')
                continue
            print(f'Running evaluation for {run_dir} ...')
            metrics = evaluators[model_key](run_dir)
            row = {
                'model': model_key,
                'label': MODEL_LABELS[model_key],
                'run': run_idx,
                **metrics,
            }
            per_run_rows.append(row)
            print(
                f"  ADE={metrics['ade_mean']:.4f} m, FDE={metrics['fde_mean']:.4f} m, "
                f"samples={metrics['num_samples']}"
            )

    summary_rows = summarize_results(per_run_rows)
    save_csv(per_run_rows, summary_dir / 'ade_fde_per_run.csv')
    save_csv(summary_rows, summary_dir / 'ade_fde_summary.csv')

    markdown_table = format_markdown_table(summary_rows)
    latex_table = format_latex_table(summary_rows)

    with open(summary_dir / 'ade_fde_summary.md', 'w', encoding='utf-8') as f:
        f.write(markdown_table)
    with open(summary_dir / 'ade_fde_summary.tex', 'w', encoding='utf-8') as f:
        f.write(latex_table)

    print('\nUnified ADE/FDE summary table:')
    print(markdown_table)
    print(f'Per-run CSV saved to: {summary_dir / "ade_fde_per_run.csv"}')
    print(f'Summary CSV saved to: {summary_dir / "ade_fde_summary.csv"}')
    print(f'Markdown table saved to: {summary_dir / "ade_fde_summary.md"}')
    print(f'LaTeX table saved to: {summary_dir / "ade_fde_summary.tex"}')


if __name__ == '__main__':
    main()
