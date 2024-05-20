import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define Earth and Neptune coordinates (arbitrary for visualization)
earth_coordinates = [0, 0, 0]
neptune_coordinates = [300, 0, 0]  # Assuming Neptune is 300 units away from Earth in the x-direction

# Define function to compute trajectory based on Gott time machine metric equation
def compute_trajectory(num_points):
    # Define parameters for trajectory generation
    r_max = 300  # Maximum radial distance
    theta_max = np.pi  # Maximum theta angle
    phi_max = 2 * np.pi  # Maximum phi angle

    # Generate random values for r, theta, and phi
    r_values = np.linspace(0, r_max, num_points)
    theta_values = np.random.uniform(0, theta_max, num_points)
    phi_values = np.random.uniform(0, phi_max, num_points)

    # Compute trajectory coordinates based on Gott time machine metric equation
    trajectory_x = r_values * np.tan(theta_values) * np.tan(phi_values)
    trajectory_y = r_values * np.tan(theta_values) * np.tan(phi_values)
    trajectory_z = r_values * np.tan(theta_values)
    
    return trajectory_x, trajectory_y, trajectory_z

# Generate trajectory based on Gott time machine metric equation
num_points = 500
trajectory_x, trajectory_y, trajectory_z = compute_trajectory(num_points)

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot Earth and Neptune
ax.scatter(*earth_coordinates, color='blue', label='Earth')
ax.scatter(*neptune_coordinates, color='red', label='Neptune')

# Plot Trajectory based on Gott time machine metric equation
trajectory_line, = ax.plot([], [], [], color='green', linestyle='--', label='Trajectory (Gott Equation)')

# Label Axes and Add Title
ax.set_xlabel('X (Distance)')
ax.set_ylabel('Y (Distance)')
ax.set_zlabel('Z (Distance)')
ax.set_title('Quantum Communication Between Earth and Neptune with Gott Time Machine Metric')

# Add Legend
ax.legend()

# Function to update animation
def update(frame):
    perturbation = np.random.uniform(-10, 10, size=num_points)  # Generate random perturbations
    perturbed_trajectory_z = trajectory_z + perturbation  # Add perturbations to trajectory_z
    trajectory_line.set_data(trajectory_x, trajectory_y)
    trajectory_line.set_3d_properties(perturbed_trajectory_z)
    return trajectory_line,

# Create animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 100), interval=50, blit=True)

plt.show()
