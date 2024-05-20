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
def generate_bulb(ax, bulb_radius):
    time_step = 0
    trajectory_x, trajectory_y, trajectory_z = compute_bulb_trajectory(num_points, time_step, bulb_radius)
    ax.plot(trajectory_x, trajectory_y, trajectory_z, label='Bulb')
    power_generator = (np.mean(trajectory_x), np.mean(trajectory_y), np.mean(trajectory_z))
    ax.scatter(power_generator[0], power_generator[1], power_generator[2], color='red', s=100, marker='o', label='Power Generator')
    generate_qubits(ax, trajectory_x, trajectory_y, trajectory_z, 'goldenrod')
    ax.set_title('Single Bulb with Power Generator')

# Define function to generate qubits
def generate_qubits(ax, trajectory_x, trajectory_y, trajectory_z, color):
    qubit_coordinates_x = trajectory_x[::10]  # Sample every 10th point for qubits
    qubit_coordinates_y = trajectory_y[::10]
    qubit_coordinates_z = trajectory_z[::10]
    ax.scatter(qubit_coordinates_x, qubit_coordinates_y, qubit_coordinates_z, color=color, marker='o')

# Parameters
num_points = 100
bulb_radius = 50

# Create figure and subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
generate_bulb(ax, bulb_radius)

# Add labels and legend
ax.set_xlabel('X (Distance)')
ax.set_ylabel('Y (Distance)')
ax.set_zlabel('Z (Distance)')
ax.legend()

plt.show()