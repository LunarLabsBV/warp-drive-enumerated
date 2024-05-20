import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

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
def generate_qubits(ax, trajectory_x, trajectory_y, trajectory_z, color):
    qubit_coordinates_x = trajectory_x[::10]  # Sample every 10th point for qubits
    qubit_coordinates_y = trajectory_y[::10]
    qubit_coordinates_z = trajectory_z[::10]
    ax.scatter(qubit_coordinates_x, qubit_coordinates_y, qubit_coordinates_z, color=color, marker='o')

# Generate initial trajectory
num_points = 100
num_trajectories = 10
time_steps = np.linspace(0, 2*np.pi, num_trajectories)  # Different initial time steps for each trajectory

# Create figure and subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each trajectory and place qubits
for i, time_step in enumerate(time_steps):
    trajectory_x, trajectory_y, trajectory_z = compute_trajectory(num_points, time_step)
    ax.plot(trajectory_x, trajectory_y, trajectory_z, label=f'Trajectory {i+1}')
    generate_qubits(ax, trajectory_x, trajectory_y, trajectory_z, 'goldenrod')

# Define the base of the crane
base_width = 4
base_height = 2
base_depth = 2
base_x = -base_width / 2
base_y = -base_height / 2
base_z = 0

# Define the tower of the crane
tower_height = 20
tower_width = 1
tower_depth = 1
tower_x = -tower_width / 2
tower_y = -tower_depth / 2
tower_z = base_z + base_height

# Define the lifting arm
arm_length = 15
arm_width = 0.5
arm_depth = 0.5
arm_x = -arm_width / 2
arm_y = -arm_depth / 2
arm_z = tower_z + tower_height

# Define the light bulb
bulb_radius = 1
bulb_x = 0
bulb_y = 0
bulb_z = arm_z - arm_length

# Plot the first crane
crane_vertices = np.array([
    [base_x, base_y, base_z],
    [base_x + base_width, base_y, base_z],
    [base_x + base_width, base_y + base_height, base_z],
    [base_x, base_y + base_height, base_z],
    [tower_x, tower_y, tower_z],
    [tower_x + tower_width, tower_y, tower_z],
    [tower_x + tower_width, tower_y + tower_depth, tower_z],
    [tower_x, tower_y + tower_depth, tower_z],
    [arm_x, arm_y, arm_z],
    [arm_x + arm_width, arm_y, arm_z],
    [arm_x + arm_width, arm_y + arm_depth, arm_z],
    [arm_x, arm_y + arm_depth, arm_z],
    [bulb_x - bulb_radius, bulb_y, bulb_z],
    [bulb_x + bulb_radius, bulb_y, bulb_z],
    [bulb_x, bulb_y - bulb_radius, bulb_z],
    [bulb_x, bulb_y + bulb_radius, bulb_z]
])

crane_faces = [
    [crane_vertices[0], crane_vertices[1], crane_vertices[2], crane_vertices[3]],
    [crane_vertices[4], crane_vertices[5], crane_vertices[6], crane_vertices[7]],
    [crane_vertices[8], crane_vertices[9], crane_vertices[10], crane_vertices[11]],
    [crane_vertices[12], crane_vertices[13], crane_vertices[14], crane_vertices[15]],
    [crane_vertices[0], crane_vertices[1], crane_vertices[5], crane_vertices[4]],
    [crane_vertices[1], crane_vertices[2], crane_vertices[6], crane_vertices[5]],
    [crane_vertices[2], crane_vertices[3], crane_vertices[7], crane_vertices[6]],
    [crane_vertices[0], crane_vertices[3], crane_vertices[7], crane_vertices[4]],
    [crane_vertices[8], crane_vertices[9], crane_vertices[13], crane_vertices[12]],
    [crane_vertices[9], crane_vertices[10], crane_vertices[14], crane_vertices[13]],
    [crane_vertices[10], crane_vertices[11], crane_vertices[15], crane_vertices[14]],
    [crane_vertices[8], crane_vertices[11], crane_vertices[15], crane_vertices[12]],
    [crane_vertices[4], crane_vertices[5], crane_vertices[9], crane_vertices[8]],
    [crane_vertices[5], crane_vertices[6], crane_vertices[10], crane_vertices[9]],
    [crane_vertices[6], crane_vertices[7], crane_vertices[11], crane_vertices[10]],
    [crane_vertices[4], crane_vertices[7], crane_vertices[11], crane_vertices[8]],
    [crane_vertices[12], crane_vertices[13], crane_vertices[1], crane_vertices[0]],
    [crane_vertices[13], crane_vertices[14], crane_vertices[2], crane_vertices[1]],
    [crane_vertices[14], crane_vertices[15], crane_vertices[3], crane_vertices[2]],
    [crane_vertices[15], crane_vertices[12], crane_vertices[0], crane_vertices[3]]
]

crane_collection = Poly3DCollection(crane_faces, color='gray')
ax.add_collection3d(crane_collection)

# Define the translation for the second crane
translation = np.array([50, -50, 50])

# Plot the second crane
crane_vertices_translated = crane_vertices + translation
crane_collection_translated = Poly3DCollection(crane_faces, color='gray')
crane_collection_translated.set_verts(crane_vertices_translated)
ax.add_collection3d(crane_collection_translated)

# Create the time tunnel
for i in range(len(crane_vertices)):
    x = [crane_vertices[i][0], crane_vertices_translated[i][0]]
    y = [crane_vertices[i][1], crane_vertices_translated[i][1]]
    z = [crane_vertices[i][2], crane_vertices_translated[i][2]]
    ax.plot(x, y, z, color='blue', alpha=0.5)

# Add labels and legend
ax.set_xlabel('X (Distance)')
ax.set_ylabel('Y (Distance)')
ax.set_zlabel('Z (Distance)')
ax.set_title('Compacted Tower Crane (CTC) with Trajectories and Qubits')
ax.legend()

plt.show()