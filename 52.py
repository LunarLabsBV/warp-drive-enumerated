import matplotlib.pyplot as plt
import numpy as np

# Define function to compute trajectory resembling a pottery pot
def compute_pot_trajectory(num_points, time_step):
    # Define parameters for trajectory generation
    r_max = 20  # Maximum radial distance for the pot
    pot_height = 40  # Height of the pot
    pot_radius = 10   # Radius of the pot

    # Generate random values for theta
    theta_values = np.linspace(0, 2 * np.pi, num_points)

    # Compute trajectory coordinates resembling a pottery pot
    pot_x = r_max * np.cos(theta_values)
    pot_y = r_max * np.sin(theta_values)
    pot_z = pot_height * (1 - np.cos(theta_values + time_step))

    # Apply skewing transformation to compress the trajectory along the length of the pot
    pot_z_skewed = pot_z * np.cos(theta_values + time_step)
    pot_y_skewed = pot_y * np.cos(theta_values + time_step)
    pot_x_skewed = pot_x * np.cos(theta_values + time_step)

    return pot_x, pot_y_skewed, pot_z_skewed

# Define function to generate qubits
def generate_qubits(trajectory_x, trajectory_y, trajectory_z, color):
    qubit_coordinates_x = trajectory_x[::5]  # Sample every 5th point for qubits
    qubit_coordinates_y = trajectory_y[::5]
    qubit_coordinates_z = trajectory_z[::5]
    ax.scatter(qubit_coordinates_x, qubit_coordinates_y, qubit_coordinates_z, color=color, marker='o', alpha=0.5)

# Generate initial trajectory
num_points = 200
num_trajectories = 100
time_steps = np.linspace(0, 2*np.pi, num_trajectories)  # Different initial time steps for each trajectory

# Create figure and subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each trajectory and place qubits
for i, time_step in enumerate(time_steps):
    pot_x, pot_y, pot_z = compute_pot_trajectory(num_points, time_step)
    ax.plot(pot_x, pot_y, pot_z, color='brown')  # Plot pot body
    generate_qubits(pot_x, pot_y, pot_z, 'goldenrod')  # Place qubits inside the pot body

# Add labels and title
ax.set_xlabel('X (Distance)')
ax.set_ylabel('Y (Distance)')
ax.set_zlabel('Z (Distance)')
ax.set_title('Pottery Pot Trajectories filled with Qubits')

plt.show()
