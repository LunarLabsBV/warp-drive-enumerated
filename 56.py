import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

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
    return ax.scatter(qubit_coordinates_x, qubit_coordinates_y, qubit_coordinates_z, color=color, marker='o')

# Generate initial trajectory for the airplane
num_points_airplane = 10000
num_trajectories_airplane = 100
time_steps_airplane = np.linspace(0, 2*np.pi, num_trajectories_airplane)  # Different initial time steps for each trajectory

# Generate initial trajectory for the collapsing effect
num_points_collapsing = 100
num_trajectories_collapsing = 10
time_steps_collapsing = np.linspace(0, 2*np.pi, num_trajectories_collapsing)  # Different initial time steps for each trajectory

# Create figure and subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot airplane trajectories
for i, time_step in enumerate(time_steps_airplane):
    fuselage_x, fuselage_y, fuselage_z, wing_x, wing_y, wing_z = compute_airplane_trajectory(num_points_airplane, time_step)
    ax.plot(fuselage_x, fuselage_y, fuselage_z, label=f'Trajectory {i+1}')  # Plot fuselage
    ax.plot(wing_x, wing_y, wing_z, color='blue')  # Plot wings
    generate_qubits(fuselage_x, fuselage_y, fuselage_z, 'goldenrod')  # Place qubits on fuselage

# Plot collapsing effect trajectories
scatters = []
for i, time_step in enumerate(time_steps_collapsing):
    trajectory_x, trajectory_y, trajectory_z = compute_trajectory(num_points_collapsing, time_step)
    line, = ax.plot(trajectory_x, trajectory_y, trajectory_z, label=f'Trajectory {i+1}')
    scatters.append(generate_qubits(trajectory_x, trajectory_y, trajectory_z, 'goldenrod'))

# Add labels and legend
ax.set_xlabel('X (Distance)')
ax.set_ylabel('Y (Distance)')
ax.set_zlabel('Z (Distance)')
ax.set_title('Airplane Trajectories with Fuselage, Wings, and Qubits')
ax.legend()

def update(frame):
    # Calculate scaling factor for collapsing effect
    scale_factor = 1.0 - frame / 100.0  # Adjust the denominator for the desired duration of the animation

    for scatter in scatters:
        scatter.remove()
    for i, time_step in enumerate(time_steps_collapsing):
        new_time_step = time_step + frame / 10  # Adjust speed here
        trajectory_x, trajectory_y, trajectory_z = compute_trajectory(num_points_collapsing, new_time_step)
        scaled_trajectory_x = scale_factor * trajectory_x
        scaled_trajectory_y = scale_factor * trajectory_y
        scaled_trajectory_z = scale_factor * trajectory_z
        scatters[i] = generate_qubits(scaled_trajectory_x, scaled_trajectory_y, scaled_trajectory_z, 'goldenrod')

ani = FuncAnimation(fig, update, frames=range(100), interval=50)  # Control speed with interval parameter
plt.show()
