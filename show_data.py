"""
Vehicle Data Visualization Tool
Visualizes raw CARLA data vs processed relative coordinate data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow
import os
import glob
import argparse
from pathlib import Path

class VehicleDataVisualizer:
    """Visualize vehicle trajectory data before and after processing"""
    
    def __init__(self, sampling_freq=20.0, history_time=3.0, predict_time=5.0):
        """
        Args:
            sampling_freq: Sampling frequency (Hz)
            history_time: History time window (seconds)
            predict_time: Prediction time window (seconds)
        """
        self.sampling_freq = sampling_freq
        self.history_time = history_time
        self.predict_time = predict_time
        
        # Calculate corresponding steps
        self.history_steps = int(history_time * sampling_freq) + 1  # 3s * 20Hz + 1 = 61 steps
        self.predict_steps = int(predict_time * sampling_freq)       # 5s * 20Hz = 100 steps
        
        print(f"Visualization Config:")
        print(f"  Sampling freq: {sampling_freq} Hz")
        print(f"  History window: {history_time}s + current ({self.history_steps} steps total)")
        print(f"  Prediction window: {predict_time}s ({self.predict_steps} steps)")
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def transform_to_relative_coordinates(self, states, current_state):
        """
        Transform states to relative coordinate system with current_state as origin
        Args:
            states: (N, 4) array [x, y, yaw, speed]  
            current_state: (4,) array [x_current, y_current, yaw_current, speed_current]
        Returns:
            relative_states: (N, 4) array in relative coordinate system
        """
        if states.ndim == 1:
            states = states.reshape(1, -1)
            
        relative_states = states.copy()
        
        # Extract current state
        x_current, y_current, yaw_current, speed_current = current_state
        print(f"Current state: x={x_current:.2f}, y={y_current:.2f}, yaw={yaw_current:.2f}°, speed={speed_current:.2f} m/s")
        
        # 1. Translation: set current position as origin
        relative_states[:, 0] = states[:, 0] - x_current  # x' = x - x_current
        relative_states[:, 1] = states[:, 1] - y_current  # y' = y - y_current
        
        # 2. Rotation: set current heading as 0 degree direction
        cos_yaw = np.cos(-yaw_current)
        sin_yaw = np.sin(-yaw_current)
        print(f"Rotation: cos(yaw)={cos_yaw:.4f}, sin(yaw)={sin_yaw:.4f}")
        
        x_translated = relative_states[:, 0].copy()  # 创建副本
        y_translated = relative_states[:, 1].copy()  # 创建副本
        
        # Rotation transformation
        relative_states[:, 0] = x_translated * cos_yaw - y_translated * sin_yaw
        relative_states[:, 1] = x_translated * sin_yaw + y_translated * cos_yaw
        
        # 3. Angle transformation: relative to current heading
        relative_states[:, 2] = self.normalize_angle(states[:, 2] - yaw_current)
        
        # 4. Speed remains unchanged
        relative_states[:, 3] = states[:, 3]
        
        return relative_states
    
    def load_carla_episode(self, episode_path):
        """Load single CARLA episode data"""
        trajectory_file = os.path.join(episode_path, 'trajectory.csv')
        
        if not os.path.exists(trajectory_file):
            print(f"Warning: trajectory file not found {trajectory_file}")
            return None
        
        try:
            df = pd.read_csv(trajectory_file)
            return df
        except Exception as e:
            print(f"Failed to load trajectory file {trajectory_file}: {e}")
            return None
    
    def extract_vehicle_states(self, df):
        """Extract vehicle states from CARLA data"""
        # Vehicle states: [x, y, yaw, speed]
        states = np.column_stack([
            df['x'].values,      # x position
            df['y'].values,      # y position
            np.deg2rad(df['yaw'].values),  # yaw angle (deg to rad)
            df['speed'].values / 3.6       # speed (km/h to m/s)
        ])
        
        return states
    
    def create_processing_example(self, states):
        """Create an example of how data gets processed"""
        total_steps = len(states)
        min_length = self.history_steps + self.predict_steps  # 61 + 100 = 161
        
        if total_steps < min_length:
            print(f"Trajectory too short: {total_steps} < {min_length}")
            return None
        
        # Pick a sequence from the middle of trajectory
        start_idx = total_steps // 3
        current_idx = start_idx + self.history_steps - 1  # Current state index
        
        # Extract raw sequences
        hist_states_raw = states[start_idx:start_idx + self.history_steps]
        current_state_raw = states[current_idx]
        future_states_raw = states[start_idx + self.history_steps:start_idx + self.history_steps + self.predict_steps]
        
        # Transform to relative coordinates
        hist_states_relative = self.transform_to_relative_coordinates(hist_states_raw, current_state_raw)
        future_states_relative = self.transform_to_relative_coordinates(future_states_raw, current_state_raw)
        current_state_relative = np.array([0.0, 0.0, 0.0, current_state_raw[3]])
        
        return {
            'raw_data': {
                'hist_states': hist_states_raw,
                'current_state': current_state_raw,
                'future_states': future_states_raw,
                'start_idx': start_idx,
                'current_idx': current_idx
            },
            'processed_data': {
                'hist_states': hist_states_relative,
                'current_state': current_state_relative,
                'future_states': future_states_relative
            }
        }
    
    def visualize_transformation(self, episode_path, save_plots=False):
        """Visualize data transformation for a single episode"""
        # Load episode data
        df = self.load_carla_episode(episode_path)
        if df is None:
            return
        
        # Clean data
        df = df[df['speed'] > 0.1]  # Remove stationary points
        df = df.reset_index(drop=True)
        
        if len(df) < self.history_steps + self.predict_steps:
            print(f"Episode too short: {len(df)} < {self.history_steps + self.predict_steps}")
            return
        
        # Extract states
        states = self.extract_vehicle_states(df)
        
        # Create processing example

        example = self.create_processing_example(states)
        if example is None:
            return
        
        # Create visualization
        self._plot_transformation_comparison(example, episode_path, save_plots)
    
    def _plot_transformation_comparison(self, example, episode_path, save_plots=False):
        """Plot comparison between raw and processed data"""
        raw = example['raw_data']
        processed = example['processed_data']
        
        fig = plt.figure(figsize=(20, 12))
        episode_name = os.path.basename(episode_path)
        fig.suptitle(f'Vehicle Data Transformation Visualization: {episode_name}', fontsize=16)
        
        # === Raw Data Visualization ===

        # 1. Raw Trajectory in Global Coordinates
        ax1 = plt.subplot(3, 4, 1)
        self._plot_trajectory(ax1, raw['hist_states'], raw['current_state'], raw['future_states'], 
                            'Raw Data: Global Coordinates', is_relative=False)
        
        # 2. Raw Speed Profile
        ax2 = plt.subplot(3, 4, 2)
        self._plot_speed_profile(ax2, raw['hist_states'], raw['future_states'], 'Raw Data: Speed Profile')
        
        # 3. Raw Heading Profile
        ax3 = plt.subplot(3, 4, 3)
        self._plot_heading_profile(ax3, raw['hist_states'], raw['future_states'], 'Raw Data: Heading Profile')
        
        # 4. Raw Position Distribution
        ax4 = plt.subplot(3, 4, 4)
        self._plot_position_distribution(ax4, raw['hist_states'], raw['future_states'], 'Raw Data: Position Distribution')
        
        # === Processed Data Visualization ===
        
        # 5. Processed Trajectory in Relative Coordinates
        ax5 = plt.subplot(3, 4, 5)
        self._plot_trajectory(ax5, processed['hist_states'], processed['current_state'], processed['future_states'], 
                            'Processed Data: Relative Coordinates', is_relative=True)
        
        # 6. Processed Speed Profile
        ax6 = plt.subplot(3, 4, 6)
        self._plot_speed_profile(ax6, processed['hist_states'], processed['future_states'], 'Processed Data: Speed Profile')
        
        # 7. Processed Heading Profile
        ax7 = plt.subplot(3, 4, 7)
        self._plot_heading_profile(ax7, processed['hist_states'], processed['future_states'], 'Processed Data: Heading Profile')
        
        # 8. Processed Position Distribution
        ax8 = plt.subplot(3, 4, 8)
        self._plot_position_distribution(ax8, processed['hist_states'], processed['future_states'], 'Processed Data: Position Distribution')
        
        # === Comparison Plots ===
        
        # 9. Coordinate Transformation Illustration
        ax9 = plt.subplot(3, 4, 9)
        self._plot_coordinate_transformation(ax9, raw, processed)
        
        # 10. Data Statistics Comparison
        ax10 = plt.subplot(3, 4, 10)
        self._plot_statistics_comparison(ax10, raw, processed)
        
        # 11. Heading Transformation
        ax11 = plt.subplot(3, 4, 11)
        self._plot_heading_transformation(ax11, raw, processed)
        
        # 12. Processing Summary
        ax12 = plt.subplot(3, 4, 12)
        self._plot_processing_summary(ax12, raw, processed)
        
        plt.tight_layout()
        
        if save_plots:
            output_dir = 'visualization_output'
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f'data_transformation_{episode_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def _plot_trajectory(self, ax, hist_states, current_state, future_states, title, is_relative=False):
        """Plot trajectory"""
        ax.plot(hist_states[:, 0], hist_states[:, 1], 'b-', linewidth=2, label='History', alpha=0.8)
        ax.plot(current_state[0], current_state[1], 'ro', markersize=10, label='Current')
        ax.plot(future_states[:, 0], future_states[:, 1], 'g--', linewidth=2, label='Future', alpha=0.8)
        
        # Add vehicle orientation at current state
        if is_relative:
            # In relative coordinates, current heading is always 0
            dx, dy = 5, 0  # Arrow pointing in positive x direction
        else:
            dx = 5 * np.cos(current_state[2])
            dy = 5 * np.sin(current_state[2])
        
        ax.arrow(current_state[0], current_state[1], dx, dy, 
                head_width=2, head_length=1.5, fc='red', ec='red', alpha=0.7)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        if is_relative:
            # Add coordinate system indicators
            ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)
            ax.axvline(x=0, color='k', linestyle=':', alpha=0.5)
            ax.text(0.05, 0.95, 'Origin: Current Position', transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    def _plot_speed_profile(self, ax, hist_states, future_states, title):
        """Plot speed profile"""
        hist_time = np.arange(len(hist_states)) * (1.0 / self.sampling_freq)
        future_time = np.arange(len(hist_states), len(hist_states) + len(future_states)) * (1.0 / self.sampling_freq)
        
        ax.plot(hist_time, hist_states[:, 3] * 3.6, 'b-', linewidth=2, label='History')
        ax.plot(future_time, future_states[:, 3] * 3.6, 'g--', linewidth=2, label='Future')
        ax.axvline(x=hist_time[-1], color='r', linestyle=':', alpha=0.7, label='Current Time')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (km/h)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_heading_profile(self, ax, hist_states, future_states, title):
        """Plot heading profile"""
        hist_time = np.arange(len(hist_states)) * (1.0 / self.sampling_freq)
        future_time = np.arange(len(hist_states), len(hist_states) + len(future_states)) * (1.0 / self.sampling_freq)
        
        ax.plot(hist_time, np.rad2deg(hist_states[:, 2]), 'b-', linewidth=2, label='History')
        ax.plot(future_time, np.rad2deg(future_states[:, 2]), 'g--', linewidth=2, label='Future')
        ax.axvline(x=hist_time[-1], color='r', linestyle=':', alpha=0.7, label='Current Time')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Heading (degrees)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_position_distribution(self, ax, hist_states, future_states, title):
        """Plot position distribution"""
        all_states = np.vstack([hist_states, future_states])
        
        ax.hist2d(all_states[:, 0], all_states[:, 1], bins=20, alpha=0.7)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        x_range = np.max(all_states[:, 0]) - np.min(all_states[:, 0])
        y_range = np.max(all_states[:, 1]) - np.min(all_states[:, 1])
        ax.text(0.05, 0.95, f'X range: {x_range:.1f}m\nY range: {y_range:.1f}m', 
               transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def _plot_coordinate_transformation(self, ax, raw, processed):
        """Illustrate coordinate transformation with detailed debugging"""
        current_raw = raw['current_state']
        
        # Show the actual transformation step by step
        # Step 1: Translation (move current position to origin)
        hist_translated = raw['hist_states'].copy()
        hist_translated[:, 0] -= current_raw[0]  # Remove current x
        hist_translated[:, 1] -= current_raw[1]  # Remove current y
        
        # Step 2: Rotation (align current heading with x-axis)
        cos_yaw = np.cos(-current_raw[2])
        sin_yaw = np.sin(-current_raw[2])
        
        x_trans = hist_translated[:, 0]
        y_trans = hist_translated[:, 1]
        
        hist_rotated = hist_translated.copy()
        hist_rotated[:, 0] = x_trans * cos_yaw - y_trans * sin_yaw
        hist_rotated[:, 1] = x_trans * sin_yaw + y_trans * cos_yaw
        
        # Plot transformation steps
        ax.plot(hist_translated[:, 0], hist_translated[:, 1], 'b-', linewidth=2, 
               label='After Translation', alpha=0.7)
        ax.plot(hist_rotated[:, 0], hist_rotated[:, 1], 'r--', linewidth=2, 
               label='After Rotation', alpha=0.7)
        ax.plot(processed['hist_states'][:, 0], processed['hist_states'][:, 1], 'g:', linewidth=3, 
               label='Final Processed', alpha=0.9)
        
        # Mark key points
        ax.plot(0, 0, 'ko', markersize=10, label='Origin (Current Pos)')
        ax.plot(hist_translated[-1, 0], hist_translated[-1, 1], 'bs', markersize=8, label='Translated Current')
        ax.plot(hist_rotated[-1, 0], hist_rotated[-1, 1], 'rs', markersize=8, label='Rotated Current')
        ax.plot(processed['hist_states'][-1, 0], processed['hist_states'][-1, 1], 'gs', markersize=8, label='Final Current')
        
        # Add coordinate system indicators
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Add transformation info
        ax.text(0.02, 0.98, f'Raw Current: ({current_raw[0]:.1f}, {current_raw[1]:.1f}, {np.rad2deg(current_raw[2]):.1f}°)', 
               transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        ax.text(0.02, 0.85, f'Translation: -({current_raw[0]:.1f}, {current_raw[1]:.1f})', 
               transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        ax.text(0.02, 0.72, f'Rotation: -{np.rad2deg(current_raw[2]):.1f}°', 
               transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Step-by-Step Coordinate Transformation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    def _plot_statistics_comparison(self, ax, raw, processed):
        """Compare statistics between raw and processed data"""
        # Calculate statistics
        raw_all = np.vstack([raw['hist_states'], raw['future_states']])
        proc_all = np.vstack([processed['hist_states'], processed['future_states']])
        
        stats_labels = ['X Range', 'Y Range', 'Speed Range', 'Heading Range']
        raw_stats = [
            np.ptp(raw_all[:, 0]),  # X range
            np.ptp(raw_all[:, 1]),  # Y range
            np.ptp(raw_all[:, 3] * 3.6),  # Speed range in km/h
            np.ptp(np.rad2deg(raw_all[:, 2]))  # Heading range in degrees
        ]
        proc_stats = [
            np.ptp(proc_all[:, 0]),  # X range
            np.ptp(proc_all[:, 1]),  # Y range
            np.ptp(proc_all[:, 3] * 3.6),  # Speed range in km/h
            np.ptp(np.rad2deg(proc_all[:, 2]))  # Heading range in degrees
        ]
        
        x = np.arange(len(stats_labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, raw_stats, width, label='Raw Data', alpha=0.8)
        bars2 = ax.bar(x + width/2, proc_stats, width, label='Processed Data', alpha=0.8)
        
        ax.set_xlabel('Statistics')
        ax.set_ylabel('Range Values')
        ax.set_title('Data Statistics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(stats_labels, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_heading_transformation(self, ax, raw, processed):
        """Show heading transformation"""
        time = np.arange(len(raw['hist_states'])) * (1.0 / self.sampling_freq)
        
        ax.plot(time, np.rad2deg(raw['hist_states'][:, 2]), 'b-', linewidth=2, label='Raw Heading')
        ax.plot(time, np.rad2deg(processed['hist_states'][:, 2]), 'r--', linewidth=2, label='Relative Heading')
        
        # Mark current time
        current_time = time[-1]
        ax.axvline(x=current_time, color='g', linestyle=':', alpha=0.7, label='Current Time')
        ax.axhline(y=0, color='k', linestyle=':', alpha=0.5, label='Zero Reference')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Heading (degrees)')
        ax.set_title('Heading Transformation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add text explanation
        ax.text(0.05, 0.95, 'Relative heading = Raw heading - Current heading', 
               transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    def _plot_processing_summary(self, ax, raw, processed):
        """Show processing summary"""
        ax.axis('off')
        
        # Processing steps
        summary_text = f"""
        Data Processing Summary:
        
        Original Sequence Length: {len(raw['hist_states']) + len(raw['future_states'])} steps
        History Window: {len(raw['hist_states'])} steps ({self.history_time}s + current)
        Prediction Window: {len(raw['future_states'])} steps ({self.predict_time}s)
        
        Transformations Applied:
        1. Translation: Current position → Origin [0, 0]
        2. Rotation: Current heading → 0° direction
        3. Relative heading: All angles relative to current
        4. Speed: Unchanged
        
        Current State Verification:
        Raw: [{raw['current_state'][0]:.1f}, {raw['current_state'][1]:.1f}, {np.rad2deg(raw['current_state'][2]):.1f}°, {raw['current_state'][3]*3.6:.1f} km/h]
        Processed: [{processed['current_state'][0]:.1f}, {processed['current_state'][1]:.1f}, {np.rad2deg(processed['current_state'][2]):.1f}°, {processed['current_state'][3]*3.6:.1f} km/h]
        
        Benefits:
        • Consistent origin across all sequences
        • Improved model convergence
        • Reduced coordinate system variance
        • Better generalization capability
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    def visualize_multiple_episodes(self, data_root, max_episodes=5, save_plots=False):
        """Visualize multiple episodes"""
        episode_dirs = []
        for session_dir in glob.glob(os.path.join(data_root, '*')):
            if os.path.isdir(session_dir):
                for episode_dir in glob.glob(os.path.join(session_dir, 'episode_*')):
                    if os.path.isdir(episode_dir):
                        episode_dirs.append(episode_dir)
        
        if not episode_dirs:
            print(f"No episode data found in {data_root}")
            return
        
        print(f"Found {len(episode_dirs)} episodes, visualizing first {max_episodes}")
        
        for i, episode_dir in enumerate(episode_dirs[:max_episodes]):
            print(f"\nProcessing episode {i+1}/{max_episodes}: {os.path.basename(episode_dir)}")
            self.visualize_transformation(episode_dir, save_plots)

def main():
    """Main function for vehicle data visualization"""
    parser = argparse.ArgumentParser(description='Vehicle Data Visualization Tool')
    parser.add_argument('--data_root', default='collected_data', help='CARLA data root directory')
    parser.add_argument('--episode_path', default=None, help='Specific episode path to visualize')
    parser.add_argument('--max_episodes', type=int, default=3, help='Maximum episodes to visualize')
    parser.add_argument('--save_plots', action='store_true', help='Save plots to files')
    parser.add_argument('--sampling_freq', type=float, default=20.0, help='Sampling frequency (Hz)')
    parser.add_argument('--history_time', type=float, default=3.0, help='History time window (seconds)')
    parser.add_argument('--predict_time', type=float, default=5.0, help='Prediction time window (seconds)')
    
    args = parser.parse_args()
    
    print("=== Vehicle Data Transformation Visualization Tool ===")
    print(f"This tool visualizes how CARLA data is transformed from global to relative coordinates")
    print("-" * 70)
    
    # Create visualizer
    visualizer = VehicleDataVisualizer(
        sampling_freq=args.sampling_freq,
        history_time=args.history_time,
        predict_time=args.predict_time
    )
    
    if args.episode_path:
        # Visualize specific episode
        print(f"Visualizing specific episode: {args.episode_path}")
        visualizer.visualize_transformation(args.episode_path, args.save_plots)
    else:
        # Visualize multiple episodes
        print(f"Visualizing episodes from: {args.data_root}")
        visualizer.visualize_multiple_episodes(args.data_root, args.max_episodes, args.save_plots)
    
    print("\nVisualization completed!")
    print("Key observations:")
    print("• Raw data shows vehicle trajectory in global CARLA coordinates")
    print("• Processed data shows same trajectory relative to current vehicle state")
    print("• Current state becomes origin [0, 0, 0°] in processed data")
    print("• This transformation helps ML models learn relative motion patterns")

if __name__ == "__main__":
    main()