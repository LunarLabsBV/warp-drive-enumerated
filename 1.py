import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Define function to compute trajectory based on Gott time machine metric equation
def compute_trajectory(num_points, time_step):
    # Define parameters for trajectory generation
    r_max = 50  # Maximum radial distance
    theta_max = np.pi  # Maximum theta angle
    phi_max = 2 * np.pi  # Maximum phi angle

    # Generate random values for theta and phi
    theta_values = np.linspace(0, theta_max, num_points)
    phi_values = np.linspace(0, phi_max, num_points)

    # Compute trajectory coordinates based on Gott time machine metric equation
    trajectory_x = r_max * np.sin(theta_values + time_step) * np.cos(phi_values)
    trajectory_y = r_max * np.sin(theta_values + time_step) * np.sin(phi_values)
    trajectory_z = r_max * np.cos(theta_values + time_step)
    
    return trajectory_x, trajectory_y, trajectory_z

# Define function to generate qubits
def generate_qubits(trajectory_x, trajectory_y, trajectory_z, time_step):
    qubit_coordinates_x = trajectory_x[::10]  # Sample every 10th point for qubits
    qubit_coordinates_y = trajectory_y[::10]
    qubit_coordinates_z = trajectory_z[::10]
    c = time_step * np.ones_like(qubit_coordinates_x)  # Assign a color based on time_step
    ax.scatter(qubit_coordinates_x, qubit_coordinates_y, qubit_coordinates_z, c=c, cmap='hsv', marker='o')

# Generate initial trajectory
num_points = 100
num_trajectories = 100
time_steps = np.linspace(0, 2*np.pi, num_trajectories)  # Different initial time steps for each trajectory

# Create figure and subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each trajectory and place qubits
for i, time_step in enumerate(time_steps):
    trajectory_x, trajectory_y, trajectory_z = compute_trajectory(num_points, time_step)
    ax.plot(trajectory_x, trajectory_y, trajectory_z, label=f'Trajectory {i+1}', alpha=0.7)
    generate_qubits(trajectory_x, trajectory_y, trajectory_z, time_step)  # Pass time_step to generate_qubits

# Add labels and legend
ax.set_xlabel('X (Distance)')
ax.set_ylabel('Y (Distance)')
ax.set_zlabel('Z (Distance)')
ax.set_title('10 Trajectories with Qubits')

# Add color bar for wall clock representation
norm = Normalize(vmin=0, vmax=2*np.pi)
sm = ScalarMappable(norm=norm, cmap='hsv')
sm.set_array([])
cbar = fig.colorbar(sm)
cbar.set_label('Wall Clock Time')

plt.show()
