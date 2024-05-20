import matplotlib.pyplot as plt
import numpy as np

# Define function to compute trajectory based on Gott time machine metric equation
def compute_bulb_trajectory(num_points, time_step, bulb_radius):
    # Define parameters for trajectory generation
    theta_max = np.pi  # Maximum theta angle
    phi_max = 2 * np.pi  # Maximum phi angle

    # Generate random values for theta and phi
    theta_values = np.linspace(0, theta_max, num_points)
    phi_values = np.linspace(0, phi_max, num_points)

    # Compute trajectory coordinates for a bulb shape
    trajectory_x = bulb_radius * np.sin(theta_values + time_step) * np.cos(phi_values)
    trajectory_y = bulb_radius * np.sin(theta_values + time_step) * np.sin(phi_values)
    trajectory_z = bulb_radius * np.cos(theta_values + time_step)
    
    return trajectory_x, trajectory_y, trajectory_z

# Define function to generate bulbs
def generate_bulb(ax, num_bulbs, bulb_radius):
    for i in range(num_bulbs):
        time_step = 2 * np.pi / num_bulbs * i
        trajectory_x, trajectory_y, trajectory_z = compute_bulb_trajectory(num_points, time_step, bulb_radius)
        ax.plot(trajectory_x, trajectory_y, trajectory_z, label=f'Bulb {i+1}')
        generate_qubits(ax, trajectory_x, trajectory_y, trajectory_z, 'goldenrod')

# Define function to generate qubits
def generate_qubits(ax, trajectory_x, trajectory_y, trajectory_z, color):
    qubit_coordinates_x = trajectory_x[::10]  # Sample every 10th point for qubits
    qubit_coordinates_y = trajectory_y[::10]
    qubit_coordinates_z = trajectory_z[::10]
    ax.scatter(qubit_coordinates_x, qubit_coordinates_y, qubit_coordinates_z, color=color, marker='o')

# Parameters
num_points = 100
num_bulbs_per_row = 5
bulb_radius = 50

# Create figure and subplot
fig = plt.figure(figsize=(15, 10))
for i in range(num_bulbs_per_row):
    for j in range(num_bulbs_per_row):
        ax = fig.add_subplot(num_bulbs_per_row, num_bulbs_per_row, i * num_bulbs_per_row + j + 1, projection='3d')
        generate_bulb(ax, num_bulbs_per_row * num_bulbs_per_row, bulb_radius)
        ax.set_title(f'Bulb {i * num_bulbs_per_row + j + 1}')

# Add labels and legend
fig.suptitle('Power Grid of Bulbs', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()
