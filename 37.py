import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Generate sample data for directional changes (random for demonstration)
directional_changes = np.random.rand(10, 2)
original_direction_changes = np.random.rand(10, 2)

# Compute average displacement vector
avg_displacement_vector = np.mean(np.diff(directional_changes, axis=0), axis=0)

# Define Earth and Mars coordinates (arbitrary for visualization)
earth_coordinates = [0, 0, 0]
mars_coordinates = [50, 0, 0]  # Assuming Mars is 50 units away from Earth in the x-direction

# Define function to compute trajectory based on Gott time machine metric equation
def compute_trajectory(num_points):
    # Define parameters for trajectory generation
    r_max = 50  # Maximum radial distance
    theta_max = np.pi  # Maximum theta angle
    phi_max = 2 * np.pi  # Maximum phi angle

    # Generate random values for r, theta, and phi
    r_values = np.linspace(0, r_max, num_points)
    theta_values = np.random.uniform(0, theta_max, num_points)
    phi_values = np.random.uniform(0, phi_max, num_points)

    # Compute trajectory coordinates based on Gott time machine metric equation
    trajectory_x = r_values * np.sin(theta_values) * np.cos(phi_values)
    trajectory_y = r_values * np.sin(theta_values) * np.sin(phi_values)
    trajectory_z = r_values * np.cos(theta_values)
    
    return trajectory_x, trajectory_y, trajectory_z

# Generate trajectory based on Gott time machine metric equation
num_points = 100
trajectory_x, trajectory_y, trajectory_z = compute_trajectory(num_points)

# Create figure and subplots
fig = plt.figure(figsize=(15, 6))

# 3D Plot
ax1 = fig.add_subplot(121, projection='3d')

# Plot Earth and Mars
ax1.scatter(*earth_coordinates, color='blue', label='Earth')
ax1.scatter(*mars_coordinates, color='red', label='Mars')

# Plot Trajectory based on Gott time machine metric equation
trajectory_line, = ax1.plot(trajectory_x, trajectory_y, trajectory_z, color='green', linestyle='--', label='Trajectory (Gott Equation)')

# Label Axes and Add Title
ax1.set_xlabel('X (Distance)')
ax1.set_ylabel('Y (Distance)')
ax1.set_zlabel('Z (Distance)')
ax1.set_title('3D Trajectory - Earth to Mars')

# Add Legend
ax1.legend()

# 2D Plot
ax2 = fig.add_subplot(122)

# Plot Trajectory on xy-plane (z=0)
ax2.plot(trajectory_x, trajectory_y, color='green', linestyle='--', label='Trajectory (xy-plane)')
ax2.scatter(*earth_coordinates[:2], color='blue', label='Earth')
ax2.scatter(*mars_coordinates[:2], color='red', label='Mars')

# Label Axes and Add Title
ax2.set_xlabel('X (Distance)')
ax2.set_ylabel('Y (Distance)')
ax2.set_title('2D Trajectory - Earth to Mars (xy-plane)')

# Add Legend
ax2.legend()

# Generate qubits along the trajectory
num_qubits = 10  # Number of qubits
qubit_coordinates = np.linspace(0, 50, num_qubits)  # Linearly spaced qubit coordinates from Earth to Mars

# Plot qubits
qubit_scatter = ax1.scatter(qubit_coordinates, np.zeros(num_qubits), np.zeros(num_qubits), color='orange', marker='o')
qubit_scatter2D = ax2.scatter(qubit_coordinates, np.zeros(num_qubits), color='orange', marker='o')

# Function to update animation
def update(frame):
    perturbation = np.random.uniform(-10, 10, size=num_qubits)  # Generate random perturbations
    perturbed_qubit_coordinates = qubit_coordinates + perturbation  # Add perturbations to qubit coordinates
    
    qubit_scatter._offsets3d = (perturbed_qubit_coordinates, np.zeros(num_qubits), np.zeros(num_qubits))
    qubit_scatter2D.set_offsets(np.column_stack((perturbed_qubit_coordinates, np.zeros(num_qubits))))
    
    return qubit_scatter, qubit_scatter2D

# Create animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 100), interval=50, blit=True)

plt.tight_layout()
plt.show()
