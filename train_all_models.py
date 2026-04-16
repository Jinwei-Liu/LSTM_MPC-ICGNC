"""Train all trajectory-prediction baselines multiple times and plot mean±std loss curves."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from plot_training_comparison import NATURE_COLORS, apply_nature_style
from vehicle_lstm_lib import VehicleDataset, VehicleLSTM, VehicleLSTMTrainer
from vehicle_lstm_dynamics_lib import (
    VehicleLSTMDynamics,
    VehicleLSTMDynamicsDataset,
    VehicleLSTMDynamicsTrainer,
)


MODEL_SPECS = {
    'lstm': {
        'label': 'LSTM',
        'history_file': 'vehicle_lstm_epoch_history.csv',
        'color': NATURE_COLORS[0],
    },
    'lstm_mpc': {
        'label': 'LSTM+MPC',
        'history_file': 'vehicle_lstm_mpc_epoch_history.csv',
        'color': NATURE_COLORS[1],
    },
    'lstm_dynamics': {
        'label': 'LSTM+Dynamics',
        'history_file': 'vehicle_lstm_dynamics_epoch_history.csv',
        'color': NATURE_COLORS[2],
    },
}


def resolve_device(device_arg: str) -> str:
    if device_arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_arg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pickle_dataset(data_folder: str, file_name: str):
    dataset_path = Path(data_folder) / file_name
    if not dataset_path.exists():
        raise FileNotFoundError(f'Dataset not found: {dataset_path}')
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    print(f'Dataset loaded: {dataset_path}')
    print(f"  Sequences: {len(dataset['training_sequences'])}")
    return dataset


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_csv_rows(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def checkpoint_exists(model_key: str, run_dir: Path, args) -> bool:
    if model_key == 'lstm':
        return (run_dir / 'best_vehicle_lstm.pth').exists() and (run_dir / 'scalers.pkl').exists()
    if model_key == 'lstm_mpc':
        name = f'best_vehicle_lstm_mpc_ds{args.downsample_factor}_targets{args.num_targets}_lqr{args.lqr_iter}.pth'
        return (run_dir / name).exists()
    if model_key == 'lstm_dynamics':
        return (run_dir / 'best_vehicle_lstm_dynamics.pth').exists()
    return False


def aggregate_histories(experiment_root: Path, model_key: str, repeats: int):
    rows_by_run = []
    max_epoch = 0
    history_name = MODEL_SPECS[model_key]['history_file']

    for run_idx in range(1, repeats + 1):
        history_path = experiment_root / model_key / f'run_{run_idx:02d}' / history_name
        if not history_path.exists():
            continue
        rows = read_csv_rows(history_path)
        if not rows:
            continue
        processed = []
        for row in rows:
            processed.append({
                'epoch': int(row['epoch']),
                'train_loss': float(row['train_loss']),
                'val_loss': float(row['val_loss']),
            })
        max_epoch = max(max_epoch, max(r['epoch'] for r in processed))
        rows_by_run.append(processed)

    if not rows_by_run:
        return []

    summary_rows = []
    for epoch in range(1, max_epoch + 1):
        train_vals = []
        val_vals = []
        for run_rows in rows_by_run:
            matching = [r for r in run_rows if r['epoch'] == epoch]
            if matching:
                train_vals.append(matching[0]['train_loss'])
                val_vals.append(matching[0]['val_loss'])
        if not train_vals:
            continue
        summary_rows.append({
            'model': model_key,
            'epoch': epoch,
            'num_runs': len(train_vals),
            'train_loss_mean': float(np.mean(train_vals)),
            'train_loss_std': float(np.std(train_vals, ddof=1)) if len(train_vals) > 1 else 0.0,
            'val_loss_mean': float(np.mean(val_vals)),
            'val_loss_std': float(np.std(val_vals, ddof=1)) if len(val_vals) > 1 else 0.0,
        })

    return summary_rows


def save_loss_summary_csv(summary_rows, output_path: Path) -> None:
    if not summary_rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def plot_loss_with_error_bands(all_summaries: dict[str, list[dict]], output_stem: Path) -> None:
    apply_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    for model_key, rows in all_summaries.items():
        if not rows:
            continue
        label = MODEL_SPECS[model_key]['label']
        color = MODEL_SPECS[model_key]['color']
        epochs = np.array([r['epoch'] for r in rows], dtype=float)
        train_mean = np.array([r['train_loss_mean'] for r in rows], dtype=float)
        train_std = np.array([r['train_loss_std'] for r in rows], dtype=float)
        val_mean = np.array([r['val_loss_mean'] for r in rows], dtype=float)
        val_std = np.array([r['val_loss_std'] for r in rows], dtype=float)

        axes[0].plot(epochs, train_mean, color=color, label=label)
        axes[0].fill_between(epochs, train_mean - train_std, train_mean + train_std, color=color, alpha=0.18)

        axes[1].plot(epochs, val_mean, color=color, label=label)
        axes[1].fill_between(epochs, val_mean - val_std, val_mean + val_std, color=color, alpha=0.18)

    axes[0].set_title('Training Loss')
    axes[1].set_title('Validation Loss')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(axis='y', color='#D9D9D9', linewidth=0.8, alpha=0.8)
        ax.grid(axis='x', visible=False)
        ax.legend(loc='best')

    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix('.png'), bbox_inches='tight', dpi=300)
    fig.savefig(output_stem.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def train_lstm_run(run_dir: Path, seed: int, train_full, val_full, args, device: str) -> None:
    set_seed(seed)
    train_dataset = VehicleDataset(train_full['training_sequences'])
    val_dataset = VehicleDataset(
        val_full['training_sequences'],
        train_dataset.scaler_states,
        train_dataset.scaler_controls,
        is_train=False,
    )

    with open(run_dir / 'scalers.pkl', 'wb') as f:
        pickle.dump(
            {
                'scaler_states': train_dataset.scaler_states,
                'scaler_controls': train_dataset.scaler_controls,
            },
            f,
        )

    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, seed)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers, seed)

    model = VehicleLSTM(hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    trainer = VehicleLSTMTrainer(model, device=device, save_dir=str(run_dir))
    trainer.train_model(
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )


def train_lstm_mpc_run(run_dir: Path, seed: int, train_full, val_full, args, device: str) -> None:
    from vehicle_lstm_mpc_lib import VehicleLSTMMPC, VehicleLSTMMPCDataset, VehicleLSTMMPCTrainer

    set_seed(seed)
    train_dataset = VehicleLSTMMPCDataset(train_full['training_sequences'])
    val_dataset = VehicleLSTMMPCDataset(val_full['training_sequences'])

    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, seed)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers, seed)

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
        device=device,
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
        save_dir=str(run_dir),
    )


def train_lstm_dynamics_run(run_dir: Path, seed: int, train_full, val_full, args, device: str) -> None:
    set_seed(seed)
    train_dataset = VehicleLSTMDynamicsDataset(train_full['training_sequences'])
    val_dataset = VehicleLSTMDynamicsDataset(val_full['training_sequences'])

    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, seed)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers, seed)

    model = VehicleLSTMDynamics(
        input_dim=6,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        action_dim=2,
        predict_steps=100,
        dt=0.05,
    )
    trainer = VehicleLSTMDynamicsTrainer(model, device=device)
    trainer.train_model(
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        save_dir=str(run_dir),
    )


def main():
    parser = argparse.ArgumentParser(description='Train all three models multiple times and summarize loss curves')
    parser.add_argument('--data_folder', default='vehicle_datasets', help='Folder containing datasets')
    parser.add_argument('--train_file', default='vehicle_train_dataset.pkl', help='Training dataset filename')
    parser.add_argument('--val_file', default='vehicle_test_dataset.pkl', help='Validation dataset filename')
    parser.add_argument('--experiment_root', default='experiments_multirun', help='Root directory for all runs')
    parser.add_argument('--repeats', type=int, default=3, help='Number of runs per model')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension for all models')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--device', default='auto', help='auto, cpu, or cuda')
    parser.add_argument('--seed_base', type=int, default=42, help='Base random seed')
    parser.add_argument('--downsample_factor', type=int, default=10, help='MPC downsample factor')
    parser.add_argument('--num_targets', type=int, default=10, help='Number of MPC targets')
    parser.add_argument('--lqr_iter', type=int, default=10, help='Number of LQR iterations for MPC')
    parser.add_argument('--skip_existing', action='store_true', help='Skip runs whose checkpoints already exist')
    args = parser.parse_args()

    device = resolve_device(args.device)
    experiment_root = Path(args.experiment_root)
    summary_dir = experiment_root / 'summary'
    experiment_root.mkdir(parents=True, exist_ok=True)

    train_full = load_pickle_dataset(args.data_folder, args.train_file)
    val_full = load_pickle_dataset(args.data_folder, args.val_file)

    write_json(
        experiment_root / 'experiment_config.json',
        {
            'device': device,
            'repeats': args.repeats,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'patience': args.patience,
            'downsample_factor': args.downsample_factor,
            'num_targets': args.num_targets,
            'lqr_iter': args.lqr_iter,
            'seed_base': args.seed_base,
            'data_folder': args.data_folder,
            'train_file': args.train_file,
            'val_file': args.val_file,
        },
    )

    trainers = {
        'lstm': train_lstm_run,
        'lstm_mpc': train_lstm_mpc_run,
        'lstm_dynamics': train_lstm_dynamics_run,
    }

    for model_idx, model_key in enumerate(['lstm', 'lstm_mpc', 'lstm_dynamics']):
        print('\n' + '=' * 90)
        print(f"Training model: {MODEL_SPECS[model_key]['label']}")
        print('=' * 90)

        for run_idx in range(1, args.repeats + 1):
            run_dir = experiment_root / model_key / f'run_{run_idx:02d}'
            run_dir.mkdir(parents=True, exist_ok=True)
            seed = args.seed_base + model_idx * 100 + run_idx

            if args.skip_existing and checkpoint_exists(model_key, run_dir, args):
                print(f'Skipping existing run: {run_dir}')
                continue

            print('\n' + '-' * 90)
            print(f"{MODEL_SPECS[model_key]['label']} | Run {run_idx}/{args.repeats} | Seed {seed} | Device {device}")
            print('-' * 90)

            write_json(
                run_dir / 'run_config.json',
                {
                    'model': model_key,
                    'label': MODEL_SPECS[model_key]['label'],
                    'run_index': run_idx,
                    'seed': seed,
                    'device': device,
                },
            )
            trainers[model_key](run_dir, seed, train_full, val_full, args, device)

    all_summary_rows = []
    all_summaries = {}
    for model_key in ['lstm', 'lstm_mpc', 'lstm_dynamics']:
        summary_rows = aggregate_histories(experiment_root, model_key, args.repeats)
        all_summaries[model_key] = summary_rows
        all_summary_rows.extend(summary_rows)
        save_loss_summary_csv(summary_rows, summary_dir / f'{model_key}_loss_summary.csv')

    save_loss_summary_csv(all_summary_rows, summary_dir / 'all_models_loss_summary.csv')
    plot_loss_with_error_bands(all_summaries, summary_dir / 'loss_mean_std_comparison')

    print('\nTraining finished.')
    print(f'Summary CSV saved to: {summary_dir / "all_models_loss_summary.csv"}')
    print(f'Summary plot saved to: {summary_dir / "loss_mean_std_comparison.png"}')


if __name__ == '__main__':
    main()
