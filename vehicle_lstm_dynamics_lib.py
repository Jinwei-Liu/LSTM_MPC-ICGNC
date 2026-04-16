import csv
import os
import pickle
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from kinematic_bicycle import Kinematic_Bicycle_MPC

warnings.filterwarnings('ignore')


class VehicleLSTMDynamicsDataset(Dataset):
    """Dataset for the LSTM+Dynamics baseline using relative-coordinate sequences."""

    def __init__(self, sequences):
        self.sequences = sequences
        print(f"Dataset initialized with {len(sequences)} sequences")
        print("Using raw relative coordinate data (no normalization)")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        hist_states = torch.FloatTensor(seq['hist_states'])
        hist_controls = torch.FloatTensor(seq['hist_controls'])
        current_state = torch.FloatTensor(seq['current_state'])
        future_states = torch.FloatTensor(seq['future_states'])
        future_controls = torch.FloatTensor(seq['future_controls'])

        max_accel = 10.0
        max_steer = np.deg2rad(70)

        hist_acc = (hist_controls[:, 0] - hist_controls[:, 1]) * max_accel
        hist_delta = hist_controls[:, 2] * max_steer
        hist_actions = torch.stack([hist_acc, hist_delta], dim=1)

        future_acc = (future_controls[:, 0] - future_controls[:, 1]) * max_accel
        future_delta = future_controls[:, 2] * max_steer
        future_actions = torch.stack([future_acc, future_delta], dim=1)

        input_seq = torch.cat([hist_states, hist_actions], dim=1)

        return {
            'input_seq': input_seq,
            'current_state': current_state,
            'future_states': future_states,
            'future_actions': future_actions,
        }


class VehicleLSTMDynamics(nn.Module):
    """Encoder-decoder LSTM that predicts future actions and rolls them out with dynamics."""

    def __init__(
        self,
        input_dim=6,
        hidden_dim=128,
        num_layers=1,
        action_dim=2,
        predict_steps=100,
        dt=0.05,
    ):
        super(VehicleLSTMDynamics, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.action_dim = action_dim
        self.predict_steps = predict_steps
        self.dt = dt

        self.a_bound = 10.0
        self.delta_f_bound = np.deg2rad(70)

        self.encoder_lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
        )
        self.decoder_lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

        self.dynamics = Kinematic_Bicycle_MPC(dt=dt)

    def apply_action_bounds(self, action_raw):
        a = torch.tanh(action_raw[..., 0:1]) * self.a_bound
        delta_f = torch.tanh(action_raw[..., 1:2]) * self.delta_f_bound
        return torch.cat([a, delta_f], dim=-1)

    def decode_actions(self, context):
        decoder_input = context.unsqueeze(1).repeat(1, self.predict_steps, 1)
        decoder_out, _ = self.decoder_lstm(decoder_input)
        action_raw = self.action_head(decoder_out)
        return self.apply_action_bounds(action_raw)

    def rollout(self, current_state, actions):
        states = []
        state = current_state

        for step in range(actions.size(1)):
            state = self.dynamics(state, actions[:, step, :])
            state = torch.cat([state[:, :3], torch.clamp(state[:, 3:4], min=0.0)], dim=1)
            states.append(state.unsqueeze(1))

        return torch.cat(states, dim=1)

    def forward(self, input_seq, current_state=None):
        encoder_out, _ = self.encoder_lstm(input_seq)
        context = encoder_out[:, -1, :]
        predicted_actions = self.decode_actions(context)

        if current_state is None:
            return predicted_actions

        predicted_states = self.rollout(current_state, predicted_actions)
        return predicted_states, predicted_actions


class DynamicsLoss(nn.Module):
    """Loss for the LSTM+Dynamics baseline."""

    def __init__(self, control_weight=0.1, reg_weight=0.01):
        super(DynamicsLoss, self).__init__()
        self.control_weight = control_weight
        self.reg_weight = reg_weight
        self.mse = nn.MSELoss()

    def forward(self, pred_states, true_states, pred_actions, true_actions):
        position_loss = self.mse(pred_states[:, :, :2], true_states[:, :, :2])
        heading_loss = self.mse(pred_states[:, :, 2], true_states[:, :, 2])
        speed_loss = self.mse(pred_states[:, :, 3], true_states[:, :, 3])
        trajectory_loss = position_loss + 0.1 * heading_loss + 0.1 * speed_loss

        control_loss = self.mse(pred_actions, true_actions)
        regularization = pred_actions.pow(2).mean()

        total_loss = trajectory_loss + self.control_weight * control_loss + self.reg_weight * regularization

        return {
            'total': total_loss,
            'trajectory': trajectory_loss,
            'control': control_loss,
            'regularization': regularization,
        }


class VehicleLSTMDynamicsTrainer:
    """Trainer for the LSTM+Dynamics baseline."""

    def __init__(self, model, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(device)
        self.device = device
        self.criterion = DynamicsLoss()
        self.train_losses = []
        self.val_losses = []
        self.step_history = []
        self.epoch_history = []
        self.save_dir = None

    def _write_csv(self, path, rows):
        if not rows:
            return
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _save_training_artifacts(self):
        if not self.save_dir:
            return

        self._write_csv(
            os.path.join(self.save_dir, 'vehicle_lstm_dynamics_step_history.csv'),
            self.step_history,
        )
        self._write_csv(
            os.path.join(self.save_dir, 'vehicle_lstm_dynamics_epoch_history.csv'),
            self.epoch_history,
        )

        if not self.epoch_history:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            [row['epoch'] for row in self.epoch_history],
            [row['train_loss'] for row in self.epoch_history],
            label='Train Loss',
            linewidth=2,
        )
        ax.plot(
            [row['epoch'] for row in self.epoch_history],
            [row['val_loss'] for row in self.epoch_history],
            label='Val Loss',
            linewidth=2,
        )
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Vehicle LSTM-Dynamics Training Curve')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, 'vehicle_lstm_dynamics_loss_curve.png'), dpi=200)
        plt.close(fig)

    def train_model(self, train_loader, val_loader, epochs=100, lr=1e-3, patience=15, save_dir='models'):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_val_loss = float('inf')
        patience_counter = 0
        global_step = 0
        samples_seen = 0
        train_start_time = time.time()

        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Model will be saved to: {self.save_dir}")

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_trajectory_loss = 0.0
            train_control_loss = 0.0
            train_reg_loss = 0.0

            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"), start=1):
                optimizer.zero_grad()

                input_seq = batch['input_seq'].to(self.device)
                current_state = batch['current_state'].to(self.device)
                future_states = batch['future_states'].to(self.device)
                future_actions = batch['future_actions'].to(self.device)

                pred_states, pred_actions = self.model(input_seq, current_state)
                loss_dict = self.criterion(pred_states, future_states, pred_actions, future_actions)
                loss = loss_dict['total']

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

                batch_size = input_seq.size(0)
                samples_seen += batch_size
                global_step += 1

                self.step_history.append({
                    'global_step': global_step,
                    'epoch': epoch + 1,
                    'batch': batch_idx,
                    'samples_seen': samples_seen,
                    'batch_size': batch_size,
                    'train_loss': float(loss.item()),
                    'trajectory_loss': float(loss_dict['trajectory'].item()),
                    'control_loss': float(loss_dict['control'].item()),
                    'regularization': float(loss_dict['regularization'].item()),
                    'lr': float(optimizer.param_groups[0]['lr']),
                    'elapsed_sec': float(time.time() - train_start_time),
                })

                train_loss += loss.item()
                train_trajectory_loss += loss_dict['trajectory'].item()
                train_control_loss += loss_dict['control'].item()
                train_reg_loss += loss_dict['regularization'].item()

            train_loss /= len(train_loader)
            train_trajectory_loss /= len(train_loader)
            train_control_loss /= len(train_loader)
            train_reg_loss /= len(train_loader)

            val_loss, val_trajectory_loss, val_control_loss, val_reg_loss = self.evaluate(val_loader)
            scheduler.step(val_loss)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.epoch_history.append({
                'epoch': epoch + 1,
                'samples_seen': samples_seen,
                'train_loss': float(train_loss),
                'train_trajectory_loss': float(train_trajectory_loss),
                'train_control_loss': float(train_control_loss),
                'train_reg_loss': float(train_reg_loss),
                'val_loss': float(val_loss),
                'val_trajectory_loss': float(val_trajectory_loss),
                'val_control_loss': float(val_control_loss),
                'val_reg_loss': float(val_reg_loss),
                'lr': float(optimizer.param_groups[0]['lr']),
                'elapsed_sec': float(time.time() - train_start_time),
            })
            self._save_training_artifacts()

            print(f"Epoch {epoch + 1}/{epochs}:")
            print(
                f"  Train Loss: {train_loss:.6f} "
                f"(Traj: {train_trajectory_loss:.6f}, Ctrl: {train_control_loss:.6f}, Reg: {train_reg_loss:.6f})"
            )
            print(
                f"  Val Loss: {val_loss:.6f} "
                f"(Traj: {val_trajectory_loss:.6f}, Ctrl: {val_control_loss:.6f}, Reg: {val_reg_loss:.6f})"
            )
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                model_path = os.path.join(save_dir, 'best_vehicle_lstm_dynamics.pth')
                torch.save(
                    {
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'model_config': {
                            'input_dim': self.model.input_dim,
                            'hidden_dim': self.model.hidden_dim,
                            'num_layers': self.model.num_layers,
                            'action_dim': self.model.action_dim,
                            'predict_steps': self.model.predict_steps,
                            'dt': self.model.dt,
                        },
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'train_losses': self.train_losses,
                        'val_losses': self.val_losses,
                    },
                    model_path,
                )
                print(f"  Best model saved to {model_path}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")

    def evaluate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        val_trajectory_loss = 0.0
        val_control_loss = 0.0
        val_reg_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_seq = batch['input_seq'].to(self.device)
                current_state = batch['current_state'].to(self.device)
                future_states = batch['future_states'].to(self.device)
                future_actions = batch['future_actions'].to(self.device)

                pred_states, pred_actions = self.model(input_seq, current_state)
                loss_dict = self.criterion(pred_states, future_states, pred_actions, future_actions)

                val_loss += loss_dict['total'].item()
                val_trajectory_loss += loss_dict['trajectory'].item()
                val_control_loss += loss_dict['control'].item()
                val_reg_loss += loss_dict['regularization'].item()

        return (
            val_loss / len(val_loader),
            val_trajectory_loss / len(val_loader),
            val_control_loss / len(val_loader),
            val_reg_loss / len(val_loader),
        )


class VehicleLSTMDynamicsPredictor:
    """Inference and visualization helper for the LSTM+Dynamics baseline."""

    def __init__(self, model_path, device='cpu'):
        self.device = device

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint['model_config']

        self.model = VehicleLSTMDynamics(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            action_dim=config['action_dim'],
            predict_steps=config['predict_steps'],
            dt=config.get('dt', 0.05),
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        self.dt = self.model.dt

        print(f"Model loaded from {model_path}")
        print(f"Validation loss: {checkpoint['val_loss']:.6f}")

    def predict_trajectory(self, hist_states, hist_actions, current_state):
        input_seq = np.concatenate([hist_states, hist_actions], axis=1)
        input_seq = torch.FloatTensor(input_seq).unsqueeze(0).to(self.device)
        current_state = torch.FloatTensor(current_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predicted_states, predicted_actions = self.model(input_seq, current_state)

        return (
            predicted_states.squeeze(0).cpu().numpy(),
            predicted_actions.squeeze(0).cpu().numpy(),
        )

    def visualize_prediction(
        self,
        hist_states,
        pred_states,
        true_states=None,
        pred_actions=None,
        true_actions=None,
    ):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        ax = axes[0, 0]
        ax.plot(hist_states[:, 0], hist_states[:, 1], 'b-', label='History', linewidth=2)
        ax.plot(pred_states[:, 0], pred_states[:, 1], 'r--', label='Predicted', linewidth=2)
        if true_states is not None:
            ax.plot(true_states[:, 0], true_states[:, 1], 'g-', label='Ground Truth', linewidth=2)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Trajectory')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend()

        ax = axes[0, 1]
        time_hist = np.arange(len(hist_states)) * self.dt
        time_pred = np.arange(len(pred_states)) * self.dt
        ax.plot(time_hist, hist_states[:, 3] * 3.6, 'b-', label='History', linewidth=2)
        ax.plot(time_pred, pred_states[:, 3] * 3.6, 'r--', label='Predicted', linewidth=2)
        if true_states is not None:
            ax.plot(time_pred, true_states[:, 3] * 3.6, 'g-', label='Ground Truth', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (km/h)')
        ax.set_title('Speed Profile')
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1, 0]
        if pred_actions is not None:
            ax.plot(time_pred, pred_actions[:, 0], 'r--', label='Predicted', linewidth=2)
        if true_actions is not None:
            ax.plot(time_pred, true_actions[:, 0], 'g-', label='Ground Truth', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (m/s²)')
        ax.set_title('Acceleration Profile')
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1, 1]
        if pred_actions is not None:
            ax.plot(time_pred, np.rad2deg(pred_actions[:, 1]), 'r--', label='Predicted', linewidth=2)
        if true_actions is not None:
            ax.plot(time_pred, np.rad2deg(true_actions[:, 1]), 'g-', label='Ground Truth', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Steering (deg)')
        ax.set_title('Steering Profile')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()


def calculate_temporal_errors(pred_states, true_states, time_step=0.05):
    time_points = [1, 2, 3, 4, 5]
    time_indices = [int(t / time_step) - 1 for t in time_points]
    error_results = {}

    for t, idx in zip(time_points, time_indices):
        if idx >= pred_states.shape[1]:
            continue

        position_error = np.sqrt(
            (pred_states[:, idx, 0] - true_states[:, idx, 0]) ** 2
            + (pred_states[:, idx, 1] - true_states[:, idx, 1]) ** 2
        )

        yaw_error = np.abs(pred_states[:, idx, 2] - true_states[:, idx, 2])
        yaw_error = np.minimum(yaw_error, 2 * np.pi - yaw_error)
        yaw_error = np.rad2deg(yaw_error)

        speed_error = np.abs(pred_states[:, idx, 3] - true_states[:, idx, 3]) * 3.6

        error_results[f'{t}s'] = {
            'position_error_mean': np.mean(position_error),
            'position_error_std': np.std(position_error),
            'position_error_max': np.max(position_error),
            'speed_error_mean': np.mean(speed_error),
            'speed_error_std': np.std(speed_error),
            'speed_error_max': np.max(speed_error),
            'yaw_error_mean': np.mean(yaw_error),
            'yaw_error_std': np.std(yaw_error),
            'yaw_error_max': np.max(yaw_error),
            'sample_count': len(position_error),
        }

    return error_results


def evaluate_full_dataset(predictor, test_dataset_full, device='cpu', batch_size=128):
    print("Starting full dataset evaluation with batch processing...")
    print(f"Total test sequences: {len(test_dataset_full['training_sequences'])}")
    print(f"Batch size: {batch_size}")

    test_dataset = VehicleLSTMDynamicsDataset(test_dataset_full['training_sequences'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_pred_states = []
    all_true_states = []

    predictor.model.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Processing batches'):
            input_seq = batch['input_seq'].to(device)
            current_state = batch['current_state'].to(device)
            future_states = batch['future_states'].to(device)

            pred_states, _ = predictor.model(input_seq, current_state)
            all_pred_states.append(pred_states.cpu().numpy())
            all_true_states.append(future_states.cpu().numpy())

    if not all_pred_states:
        print('No valid predictions generated!')
        return None

    all_pred_states = np.concatenate(all_pred_states, axis=0)
    all_true_states = np.concatenate(all_true_states, axis=0)

    print(f"Successfully processed {len(all_pred_states)} sequences")
    print(f"Prediction horizon: {all_pred_states.shape[1]} steps (dt = {predictor.dt}s)")

    return calculate_temporal_errors(all_pred_states, all_true_states, time_step=predictor.dt)


def print_error_statistics(error_results):
    print('\n' + '=' * 80)
    print('TEMPORAL ERROR ANALYSIS RESULTS (LSTM-Dynamics)')
    print('=' * 80)
    print(f"{'Time':<6} {'Pos.Error(m)':<22} {'Speed Error(km/h)':<24} {'Yaw Error(deg)':<22} {'Samples':<8}")
    print('-' * 80)

    for time_point, errors in error_results.items():
        print(
            f"{time_point:<6} "
            f"{errors['position_error_mean']:.3f}±{errors['position_error_std']:.3f} ({errors['position_error_max']:.3f})   "
            f"{errors['speed_error_mean']:.2f}±{errors['speed_error_std']:.2f} ({errors['speed_error_max']:.2f})     "
            f"{errors['yaw_error_mean']:.2f}±{errors['yaw_error_std']:.2f} ({errors['yaw_error_max']:.2f})   "
            f"{errors['sample_count']:<8}"
        )

    print('=' * 80)


def plot_error_trends(error_results):
    time_points = [1, 2, 3, 4, 5]
    pos_means = [error_results[f'{t}s']['position_error_mean'] for t in time_points if f'{t}s' in error_results]
    speed_means = [error_results[f'{t}s']['speed_error_mean'] for t in time_points if f'{t}s' in error_results]
    yaw_means = [error_results[f'{t}s']['yaw_error_mean'] for t in time_points if f'{t}s' in error_results]

    pos_stds = [error_results[f'{t}s']['position_error_std'] for t in time_points if f'{t}s' in error_results]
    speed_stds = [error_results[f'{t}s']['speed_error_std'] for t in time_points if f'{t}s' in error_results]
    yaw_stds = [error_results[f'{t}s']['yaw_error_std'] for t in time_points if f'{t}s' in error_results]

    valid_times = [t for t in time_points if f'{t}s' in error_results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].errorbar(valid_times, pos_means, yerr=pos_stds, marker='o', capsize=5, capthick=2)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Position Error (m)')
    axes[0].set_title('Position Error vs Time (LSTM-Dynamics)')
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(valid_times, speed_means, yerr=speed_stds, marker='s', capsize=5, capthick=2, color='orange')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Speed Error (km/h)')
    axes[1].set_title('Speed Error vs Time (LSTM-Dynamics)')
    axes[1].grid(True, alpha=0.3)

    axes[2].errorbar(valid_times, yaw_means, yerr=yaw_stds, marker='^', capsize=5, capthick=2, color='green')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Yaw Error (deg)')
    axes[2].set_title('Yaw Error vs Time (LSTM-Dynamics)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def load_dataset_from_folder(data_folder, dataset_name):
    dataset_path = os.path.join(data_folder, dataset_name)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    print(f"Dataset loaded: {dataset_path}")
    print(f"  Sequences: {len(dataset['training_sequences'])}")
    if 'config' in dataset:
        config = dataset['config']
        print('  Configuration:')
        print(f"    - Coordinate system: {config.get('coordinate_system', 'unknown')}")
        print(f"    - History steps: {config.get('history_steps', 'unknown')}")
        print(f"    - Predict steps: {config.get('predict_steps', 'unknown')}")

    return dataset
