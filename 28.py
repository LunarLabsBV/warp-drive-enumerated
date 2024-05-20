import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Define function to compute velocity based on the given equation
def compute_velocity(E, m, dt):
    return np.sqrt(2 * E * (1 - (m * E * dt)**6) / m)

# Define function to generate qubits with velocity
def generate_qubits_with_velocity(trajectory_x, trajectory_y, trajectory_z, velocity, color):
    num_points = len(trajectory_x)
    qubit_coordinates_x = trajectory_x[::10]  # Sample every 10th point for qubits
    qubit_coordinates_y = trajectory_y[::10]
    qubit_coordinates_z = trajectory_z[::10]
    ax.scatter(qubit_coordinates_x, qubit_coordinates_y, qubit_coordinates_z, color=color, marker='o', label=f'Velocity: {velocity:.2f}')

# Generate initial parameters
num_points = 100
num_trajectories = 10
E_values = np.linspace(0.1, 2, num_trajectories)
m = 1  # Mass (arbitrary units)
dt = 0.1  # Time interval (arbitrary units)

# Create figure and subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each trajectory and place qubits with velocity
for i, E in enumerate(E_values):
    velocity = compute_velocity(E, m, dt)
    time_step = np.linspace(0, 2*np.pi, num_points)  # Different initial time steps for each trajectory
    trajectory_x = np.sin(time_step) * np.cos(time_step)
    trajectory_y = np.sin(time_step) * np.sin(time_step)
    trajectory_z = np.cos(time_step)
    ax.plot(trajectory_x, trajectory_y, trajectory_z, label=f'Trajectory {i+1}')
    generate_qubits_with_velocity(trajectory_x, trajectory_y, trajectory_z, velocity, 'goldenrod')

# Add labels and legend
ax.set_xlabel('X (Distance)')
ax.set_ylabel('Y (Distance)')
ax.set_zlabel('Z (Distance)')
ax.set_title('10 Trajectories with Qubits and Velocity')
ax.legend()

plt.show()
