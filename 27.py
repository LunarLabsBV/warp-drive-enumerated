import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np

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
def generate_qubits(trajectory_x, trajectory_y, trajectory_z, color):
    qubit_coordinates_x = trajectory_x[::10]  # Sample every 10th point for qubits
    qubit_coordinates_y = trajectory_y[::10]
    qubit_coordinates_z = trajectory_z[::10]
    ax.scatter(qubit_coordinates_x, qubit_coordinates_y, qubit_coordinates_z, color=color, marker='o')

# Generate initial trajectory
num_points = 100
num_trajectories = 10  # Reduced number of trajectories for better visualization
time_steps = np.linspace(0, 2*np.pi, num_trajectories)  # Different initial time steps for each trajectory

# Create figure and subplot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot initial trajectories and qubits
lines = []
for i, time_step in enumerate(time_steps):
    trajectory_x, trajectory_y, trajectory_z = compute_trajectory(num_points, time_step)
    line, = ax.plot([], [], [], label=f'Trajectory {i+1}')
    lines.append(line)
    generate_qubits(trajectory_x, trajectory_y, trajectory_z, 'goldenrod')

# Add labels and legend
ax.set_xlabel('X (Distance)')
ax.set_ylabel('Y (Distance)')
ax.set_zlabel('Z (Distance)')
ax.set_title('Animated Trajectories with Qubits')
ax.legend()

# Function to update plot at each frame
def update(frame):
    for i, time_step in enumerate(time_steps):
        trajectory_x, trajectory_y, trajectory_z = compute_trajectory(num_points, time_step)
        lines[i].set_data(trajectory_x[:frame], trajectory_y[:frame])
        lines[i].set_3d_properties(trajectory_z[:frame])
    return lines

# Animate plot
ani = FuncAnimation(fig, update, frames=num_points, interval=100, blit=False, repeat=True)

plt.show()
