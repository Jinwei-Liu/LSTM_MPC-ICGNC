"""
Vehicle Kinematic Bicycle Model and MPC Testing Suite
Tests the mathematical model validity and MPC solver performance
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation
import os
import sys

# Import the required modules (assuming they're in the same directory)
from kinematic_bicycle import Kinematic_Bicycle_MPC
from vehicle_lstm_mpc import VehicleMPCSolver

class VehicleModelTester:
    """Test suite for vehicle kinematic model and MPC solver"""
    
    def __init__(self, dt=0.05, device='cpu'):
        """
        Initialize the tester
        
        Args:
            dt: Time step for simulation
            device: 'cpu' or 'cuda'
        """
        self.dt = dt
        self.device = device
        
        # Create kinematic bicycle model
        self.model = Kinematic_Bicycle_MPC(dt=dt)
        
        # Create MPC solver
        self.mpc_solver = VehicleMPCSolver(dt=dt, horizon=50, device=device)
        
        print(f"Vehicle Model Tester initialized")
        print(f"  Time step: {dt}s")
        print(f"  Device: {device}")
        print(f"  Wheelbase: {self.model._L}m")
        print(f"  lr: {self.model._lr}m, lf: {self.model._lf}m")
    
    def test_kinematic_model(self, test_scenarios=None):
        """
        Test kinematic bicycle model with various control inputs
        
        Args:
            test_scenarios: List of test scenarios, each containing:
                - name: scenario name
                - initial_state: [x, y, yaw, speed]
                - controls: list of [acceleration, steering_angle] pairs
                - duration: simulation time in seconds
        """
        if test_scenarios is None:
            test_scenarios = self._get_default_scenarios()
        
        print(f"\n{'='*60}")
        print("KINEMATIC BICYCLE MODEL TESTS")
        print(f"{'='*60}")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for i, scenario in enumerate(test_scenarios[:4]):  # Test up to 4 scenarios
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            print(f"\nTesting Scenario: {scenario['name']}")
            print(f"  Initial state: {scenario['initial_state']}")
            print(f"  Control: a={scenario['controls'][0]:.2f} m/sÂ², Î´={np.rad2deg(scenario['controls'][1]):.1f}Â°")
            print(f"  Duration: {scenario['duration']}s")
            
            # Run simulation
            states, controls, side_slip_angles = self._simulate_scenario(scenario)
            
            # Plot results
            self._plot_scenario_results(ax, states, controls, side_slip_angles, scenario)
            
            # Print final state
            final_state = states[-1]
            print(f"  Final state: x={final_state[0]:.2f}m, y={final_state[1]:.2f}m, " +
                  f"yaw={np.rad2deg(final_state[2]):.1f}Â°, v={final_state[3]*3.6:.1f}km/h")
        
        plt.tight_layout()
        plt.suptitle('Kinematic Bicycle Model Test Results', fontsize=16, y=0.98)
        plt.show()
        
        return True
    
    def test_mpc_solver(self, test_cases=None):
        """
        Test MPC solver with various target scenarios
        
        Args:
            test_cases: List of MPC test cases, each containing:
                - name: test case name
                - initial_state: [x, y, yaw, speed]
                - target_state: [x, y, yaw, speed]
                - state_weights: [w_x, w_y, w_yaw, w_speed]
                - control_weights: [w_a, w_delta]
        """
        if test_cases is None:
            test_cases = self._get_default_mpc_cases()
        
        print(f"\n{'='*60}")
        print("MPC SOLVER TESTS")
        print(f"{'='*60}")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for i, case in enumerate(test_cases[:4]):  # Test up to 4 cases
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            print(f"\nTesting MPC Case: {case['name']}")
            print(f"  Initial: {case['initial_state']}")
            print(f"  Target: {case['target_state']}")
            print(f"  State weights: {case['state_weights']}")
            print(f"  Control weights: {case['control_weights']}")
            
            # Solve MPC problem
            success, predicted_states, optimal_controls, cost = self._solve_mpc_case(case)
            
            if success:
                # Plot results
                self._plot_mpc_results(ax, predicted_states, optimal_controls, case)
                
                # Calculate final error
                final_state = predicted_states[-1]
                target_state = case['target_state']
                position_error = np.sqrt((final_state[0] - target_state[0])**2 + 
                                       (final_state[1] - target_state[1])**2)
                speed_error = abs(final_state[3] - target_state[3]) * 3.6
                yaw_error = abs(final_state[2] - target_state[2]) * 180/np.pi
                
                print(f"  MPC Results:")
                print(f"    Final cost: {cost:.4f}")
                print(f"    Position error: {position_error:.3f}m")
                print(f"    Speed error: {speed_error:.2f}km/h")
                print(f"    Yaw error: {yaw_error:.2f}Â°")
            else:
                print("  MPC solve failed!")
                ax.text(0.5, 0.5, f'MPC Solve Failed\n{case["name"]}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        plt.tight_layout()
        plt.suptitle('MPC Solver Test Results', fontsize=16, y=0.98)
        plt.show()
        
        return True
    
    def test_interactive_scenario(self, scenario_type='lane_change'):
        """
        Create an interactive test scenario for detailed analysis
        
        Args:
            scenario_type: 'lane_change', 'parking', 'curve_following', 'emergency_brake'
        """
        print(f"\n{'='*60}")
        print(f"INTERACTIVE SCENARIO TEST: {scenario_type.upper()}")
        print(f"{'='*60}")
        
        if scenario_type == 'lane_change':
            initial_state = [0.0, 0.0, 0.0, 15.0]  # 15 m/s (54 km/h)
            target_state = [30.0, 30.0, np.pi/2, 15.0]  # Lane change to left lane
            state_weights = [0.1, 0.1, 10, 0.0]
            control_weights = [0.0, 0.0]
            
        elif scenario_type == 'parking':
            initial_state = [0.0, 0.0, 0.0, 5.0]  # 5 m/s (18 km/h)
            target_state = [10.0, 10.0, np.pi/2, 0.0]  # Park with 90Â° turn
            state_weights = [0.1, 0.1, 0.0, 0.0]
            control_weights = [0.00, 0.00]
            
        elif scenario_type == 'curve_following':
            initial_state = [0.0, 0.0, 0.0, 12.0]  # 12 m/s (43 km/h)
            target_state = [50.0, 25.0, np.pi/4, 12.0]  # Follow curve
            state_weights = [2.0, 2.0, 3.0, 1.0]
            control_weights = [0.2, 1.5]
            
        elif scenario_type == 'emergency_brake':
            initial_state = [0.0, 0.0, 0.0, 20.0]  # 20 m/s (72 km/h)
            target_state = [50.0, 0.0, 0.0, 0.0]  # Emergency stop
            state_weights = [1.0, 2.0, 2.0, 5.0]
            control_weights = [2.0, 0.5]
        
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
        
        # Create test case
        test_case = {
            'name': scenario_type,
            'initial_state': initial_state,
            'target_state': target_state,
            'state_weights': state_weights,
            'control_weights': control_weights
        }
        
        # Solve MPC
        success, predicted_states, optimal_controls, cost = self._solve_mpc_case(test_case)
        
        if success:
            # Create detailed visualization
            self._create_detailed_visualization(predicted_states, optimal_controls, test_case)
            return predicted_states, optimal_controls
        else:
            print("Failed to solve MPC problem!")
            return None, None
    
    def _get_default_scenarios(self):
        """Get default test scenarios for kinematic model"""
        return [
            {
                'name': 'Straight Line Acceleration',
                'initial_state': [0.0, 0.0, 0.0, 5.0],
                'controls': [2.0, 0.0],  # 2 m/sÂ² acceleration, no steering
                'duration': 5.0
            },
            {
                'name': 'Constant Speed Turn',
                'initial_state': [0.0, 0.0, 0.0, 10.0],
                'controls': [0.0, np.deg2rad(15)],  # No acceleration, 15Â° steering
                'duration': 8.0
            },
            {
                'name': 'Acceleration with Turn',
                'initial_state': [0.0, 0.0, 0.0, 5.0],
                'controls': [1.5, np.deg2rad(10)],  # 1.5 m/sÂ² accel, 10Â° steering
                'duration': 6.0
            },
            {
                'name': 'Deceleration with Sharp Turn',
                'initial_state': [0.0, 0.0, 0.0, 15.0],
                'controls': [-3.0, np.deg2rad(25)],  # -3 m/sÂ² decel, 25Â° steering
                'duration': 4.0
            }
        ]
    
    def _get_default_mpc_cases(self):
        """Get default MPC test cases"""
        return [
            {
                'name': 'Simple Forward Motion',
                'initial_state': [0.0, 0.0, 0.0, 0.0],
                'target_state': [30.0, 0.0, 0.0, 10.0],
                'state_weights': [1.0, 1.0, 1.0, 1.0],
                'control_weights': [0.1, 0.1]
            },
            {
                'name': 'Lane Change Maneuver',
                'initial_state': [0.0, 0.0, 0.0, 12.0],
                'target_state': [50.0, 13.5, 0.0, 12.0],
                'state_weights': [0.5, 2.0, 0.0, 0.0],
                'control_weights': [0.0, 0.0]
            },
            {
                'name': 'U-Turn Maneuver',
                'initial_state': [0.0, 0.0, 0.0, 8.0],
                'target_state': [0.0, 10.0, np.pi, 8.0],
                'state_weights': [2.0, 2.0, 3.0, 1.0],
                'control_weights': [0.2, 1.5]
            },
            {
                'name': 'Emergency Stop',
                'initial_state': [0.0, 0.0, 0.0, 15.0],
                'target_state': [40.0, 0.0, 0.0, 0.0],
                'state_weights': [1.0, 1.0, 1.0, 3.0],
                'control_weights': [1.0, 0.5]
            }
        ]
    
    def _simulate_scenario(self, scenario):
        """Simulate a single kinematic model scenario"""
        initial_state = torch.tensor(scenario['initial_state'], dtype=torch.float32, device=self.device)
        controls = torch.tensor(scenario['controls'], dtype=torch.float32, device=self.device)
        duration = scenario['duration']
        
        # Number of simulation steps
        num_steps = int(duration / self.dt)
        
        # Initialize arrays
        states = torch.zeros(num_steps + 1, 4, device=self.device)
        states[0] = initial_state
        
        control_history = torch.zeros(num_steps, 2, device=self.device)
        side_slip_angles = torch.zeros(num_steps, device=self.device)
        
        # Simulate forward
        current_state = initial_state
        for i in range(num_steps):
            # Apply constant control
            control_history[i] = controls
            
            # Calculate side slip angle
            delta_f = controls[1]
            side_slip_angles[i] = self.model.get_beta(delta_f)
            
            # Step forward
            next_state = self.model.forward(current_state.unsqueeze(0), controls.unsqueeze(0))
            print(controls)
            current_state = next_state.squeeze(0)
            states[i + 1] = current_state
        
        return states.cpu().numpy(), control_history.cpu().numpy(), side_slip_angles.cpu().numpy()
    
    def _solve_mpc_case(self, case):
        """Solve a single MPC test case"""
        try:
            # Prepare tensors
            initial_state = torch.tensor([case['initial_state']], dtype=torch.float32, device=self.device)
            state_weights = torch.tensor([case['state_weights']], dtype=torch.float32, device=self.device)
            control_weights = torch.tensor([case['control_weights']], dtype=torch.float32, device=self.device)
            
            # Create target tensor [state, control]
            target_state = torch.tensor([case['target_state']], dtype=torch.float32, device=self.device)
            target_control = torch.zeros(1, 2, dtype=torch.float32, device=self.device)  # Zero target control
            target = torch.cat([target_state, target_control], dim=1)
            
            # Solve MPC
            predicted_states, optimal_controls = self.mpc_solver.solve(
                initial_state, state_weights, control_weights, target
            )
            
            # Calculate cost (simplified) - detach tensors first
            with torch.no_grad():
                final_state = predicted_states[0, -1, :].detach()
                target_final = target_state[0].detach()
                position_error = torch.norm(final_state[:2] - target_final[:2])
                cost = position_error.item()
            
            return True, predicted_states[0].detach().cpu().numpy(), optimal_controls[0].detach().cpu().numpy(), cost
            
        except Exception as e:
            print(f"MPC solve error: {e}")
            return False, None, None, float('inf')
    
    def _plot_scenario_results(self, ax, states, controls, side_slip_angles, scenario):
        """Plot results for a kinematic model scenario"""
        # Trajectory plot
        ax.plot(states[:, 0], states[:, 1], 'b-', linewidth=2, label='Trajectory')
        ax.plot(states[0, 0], states[0, 1], 'go', markersize=8, label='Start')
        ax.plot(states[-1, 0], states[-1, 1], 'ro', markersize=8, label='End')
        
        # Draw vehicle at a few time points
        time_points = [0, len(states)//3, 2*len(states)//3, len(states)-1]
        for i, t in enumerate(time_points):
            if t < len(states):
                self._draw_vehicle(ax, states[t], alpha=0.3 + 0.2*i)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{scenario["name"]}\nControl: a={controls[0,0]:.1f} m/sÂ², Î´={np.rad2deg(controls[0,1]):.1f}Â°')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    def _plot_mpc_results(self, ax, predicted_states, optimal_controls, case):
        """Plot results for an MPC test case"""
        # Trajectory plot
        ax.plot(predicted_states[:, 0], predicted_states[:, 1], 'b-', linewidth=2, label='MPC Trajectory')
        ax.plot(predicted_states[0, 0], predicted_states[0, 1], 'go', markersize=8, label='Start')
        ax.plot(case['target_state'][0], case['target_state'][1], 'r*', markersize=12, label='Target')
        ax.plot(predicted_states[-1, 0], predicted_states[-1, 1], 'ro', markersize=8, label='Final')
        
        # Draw vehicle orientation at key points
        time_points = [0, len(predicted_states)//2, len(predicted_states)-1]
        for i, t in enumerate(time_points):
            if t < len(predicted_states):
                self._draw_vehicle(ax, predicted_states[t], alpha=0.4 + 0.3*i)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(case['name'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    def _draw_vehicle(self, ax, state, alpha=1.0):
        """Draw vehicle representation at given state"""
        x, y, yaw, speed = state
        
        # Vehicle dimensions (simplified)
        length = 4.0
        width = 2.0
        
        # Vehicle corners in local coordinates
        corners = np.array([
            [-length/2, -width/2],
            [length/2, -width/2],
            [length/2, width/2],
            [-length/2, width/2]
        ])
        
        # Rotation matrix
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        R = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        
        # Transform corners to global coordinates
        corners_global = corners @ R.T + np.array([x, y])
        
        # Draw vehicle body
        vehicle = plt.Polygon(corners_global, fill=False, edgecolor='black', linewidth=1, alpha=alpha)
        ax.add_patch(vehicle)
        
        # Draw direction arrow
        arrow_length = length * 0.6
        arrow_end = np.array([x, y]) + arrow_length * np.array([cos_yaw, sin_yaw])
        ax.arrow(x, y, arrow_end[0]-x, arrow_end[1]-y, head_width=0.5, head_length=0.8, 
                fc='red', ec='red', alpha=alpha)
    
    def _create_detailed_visualization(self, predicted_states, optimal_controls, test_case):
        """Create detailed visualization with multiple subplots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        time = np.arange(len(predicted_states)) * self.dt
        
        # Trajectory plot
        ax = axes[0, 0]
        ax.plot(predicted_states[:, 0], predicted_states[:, 1], 'b-', linewidth=2, label='MPC Trajectory')
        ax.plot(predicted_states[0, 0], predicted_states[0, 1], 'go', markersize=10, label='Start')
        ax.plot(test_case['target_state'][0], test_case['target_state'][1], 'r*', markersize=15, label='Target')
        ax.plot(predicted_states[-1, 0], predicted_states[-1, 1], 'ro', markersize=10, label='Final')
        
        # Draw vehicle at multiple time points
        for i in range(0, len(predicted_states), max(1, len(predicted_states)//10)):
            self._draw_vehicle(ax, predicted_states[i], alpha=0.3)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Trajectory: {test_case["name"]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Speed profile
        ax = axes[0, 1]
        ax.plot(time, predicted_states[:, 3] * 3.6, 'b-', linewidth=2, label='Speed')
        ax.axhline(y=test_case['target_state'][3] * 3.6, color='r', linestyle='--', label='Target Speed')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (km/h)')
        ax.set_title('Speed Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Yaw angle
        ax = axes[0, 2]
        ax.plot(time, np.rad2deg(predicted_states[:, 2]), 'b-', linewidth=2, label='Yaw Angle')
        ax.axhline(y=np.rad2deg(test_case['target_state'][2]), color='r', linestyle='--', label='Target Yaw')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Yaw Angle (degrees)')
        ax.set_title('Yaw Angle')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Control inputs - acceleration
        ax = axes[1, 0]
        control_time = np.arange(len(optimal_controls)) * self.dt
        ax.plot(control_time, optimal_controls[:, 0], 'g-', linewidth=2, label='Acceleration')
        ax.axhline(y=10.0, color='r', linestyle=':', alpha=0.5, label='Max Accel')
        ax.axhline(y=-10.0, color='r', linestyle=':', alpha=0.5, label='Max Decel')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (m/sÂ²)')
        ax.set_title('Acceleration Control')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Control inputs - steering
        ax = axes[1, 1]
        ax.plot(control_time, np.rad2deg(optimal_controls[:, 1]), 'orange', linewidth=2, label='Steering Angle')
        ax.axhline(y=70, color='r', linestyle=':', alpha=0.5, label='Max Steer')
        ax.axhline(y=-70, color='r', linestyle=':', alpha=0.5, label='Max Steer')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Steering Angle (degrees)')
        ax.set_title('Steering Control')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Side slip angles
        ax = axes[1, 2]
        side_slip_angles = []
        for i in range(len(optimal_controls)):
            beta = self.model.get_beta(torch.tensor(optimal_controls[i, 1]))
            side_slip_angles.append(beta.item())
        
        ax.plot(control_time, np.rad2deg(side_slip_angles), 'purple', linewidth=2, label='Side Slip Angle')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Side Slip Angle (degrees)')
        ax.set_title('Vehicle Side Slip Angle')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f'Detailed Analysis: {test_case["name"]}', fontsize=16, y=0.98)
        plt.show()

def run_complete_test_suite():
    """Run the complete test suite for vehicle model and MPC"""
    print("="*80)
    print("VEHICLE MODEL AND MPC COMPLETE TEST SUITE")
    print("="*80)
    
    # Initialize tester
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tester = VehicleModelTester(dt=0.1, device=device)
    
    print(f"\nRunning tests on device: {device}")
    
    # Test 1: Kinematic Model
    print(f"\n{'='*40}")
    print("TEST 1: KINEMATIC BICYCLE MODEL")
    print(f"{'='*40}")
    tester.test_kinematic_model()
    
    # Test 2: MPC Solver
    print(f"\n{'='*40}")
    print("TEST 2: MPC SOLVER")
    print(f"{'='*40}")
    tester.test_mpc_solver()
    
    # Test 3: Interactive Scenarios
    print(f"\n{'='*40}")
    print("TEST 3: INTERACTIVE SCENARIOS")
    print(f"{'='*40}")
    
    scenarios = ['lane_change', 'parking', 'curve_following', 'emergency_brake']
    for scenario in scenarios:
        print(f"\n--- Testing {scenario} scenario ---")
        try:
            states, controls = tester.test_interactive_scenario(scenario)
            if states is not None:
                print(f"Successfully solved {scenario} scenario!")
            else:
                print(f"Failed to solve {scenario} scenario!")
        except Exception as e:
            print(f"Error in {scenario} scenario: {e}")
    
    print(f"\n{'='*80}")
    print("TEST SUITE COMPLETED")
    print(f"{'='*80}")

if __name__ == "__main__":
    # Run the complete test suite
    run_complete_test_suite()