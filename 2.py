import matplotlib.pyplot as plt
import numpy as np

# Define function to compute trajectory resembling an airplane with a fuselage and wings
def compute_airplane_trajectory(num_points, time_step):
    # Define parameters for trajectory generation
    r_max = 20  # Maximum radial distance for the fuselage
    fuselage_length = 40  # Length of the fuselage
    fuselage_radius = 5   # Radius of the fuselage
    wing_span = 30        # Span of the wings

    # Generate random values for theta
    theta_values = np.linspace(0, 2 * np.pi, num_points)

    # Compute trajectory coordinates resembling an airplane with a fuselage and wings
    fuselage_x = r_max * np.cos(theta_values)
    fuselage_y = np.zeros_like(theta_values)
    fuselage_z = np.linspace(-fuselage_length / 2, fuselage_length / 2, num_points)

    # Compute trajectory coordinates for the wings
    wing_x = np.linspace(-wing_span / 2, wing_span / 2, num_points)
    wing_y = np.zeros_like(theta_values)
    wing_z = np.zeros_like(theta_values)

    return fuselage_x, fuselage_y, fuselage_z, wing_x, wing_y, wing_z

# Define function to generate qubits
def generate_qubits(trajectory_x, trajectory_y, trajectory_z, color):
    qubit_coordinates_x = trajectory_x[::10]  # Sample every 10th point for qubits
    qubit_coordinates_y = trajectory_y[::10]
    qubit_coordinates_z = trajectory_z[::10]
    ax.scatter(qubit_coordinates_x, qubit_coordinates_y, qubit_coordinates_z, color=color, marker='o')

# Generate initial trajectory
num_points = 10000
num_trajectories = 100
time_steps = np.linspace(0, 2*np.pi, num_trajectories)  # Different initial time steps for each trajectory

# Create figure and subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each trajectory and place qubits
for i, time_step in enumerate(time_steps):
    fuselage_x, fuselage_y, fuselage_z, wing_x, wing_y, wing_z = compute_airplane_trajectory(num_points, time_step)
    ax.plot(fuselage_x, fuselage_y, fuselage_z, label=f'Trajectory {i+1}')  # Plot fuselage
    ax.plot(wing_x, wing_y, wing_z, color='blue')  # Plot wings
    generate_qubits(fuselage_x, fuselage_y, fuselage_z, 'goldenrod')  # Place qubits on fuselage

# Add labels and legend
ax.set_xlabel('X (Distance)')
ax.set_ylabel('Y (Distance)')
ax.set_zlabel('Z (Distance)')
ax.set_title('Airplane Trajectories with Fuselage, Wings, and Qubits')
ax.legend()

plt.show()