import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define function to compute rocket-shaped trajectory
def compute_rocket_trajectory(num_points, time_step):
    # Define parameters for trajectory generation
    r_max = 50  # Maximum radial distance
    theta_max = np.pi  # Maximum theta angle
    phi_max = 2 * np.pi  # Maximum phi angle

    # Generate values for theta and phi
    theta_values = np.linspace(0, theta_max, num_points)
    phi_values = np.linspace(0, phi_max, num_points)

    # Compute rocket-shaped trajectory coordinates
    trajectory_x = r_max * np.sin(theta_values + time_step) * np.cos(phi_values)
    trajectory_y = r_max * np.sin(theta_values + time_step) * np.sin(phi_values)
    trajectory_z = r_max * np.cos(theta_values + time_step)

    # Modify z-coordinate to create rocket shape
    trajectory_z += 0.3 * r_max * np.cos(phi_values)

    return trajectory_x, trajectory_y, trajectory_z

# Define function to compute fuselage trajectory
def compute_fuselage_trajectory(num_points, time_step):
    # Define parameters for trajectory generation
    r_max = 50  # Maximum radial distance
    theta_max = np.pi  # Maximum theta angle

    # Generate random values for theta
    theta_values = np.linspace(0, theta_max, num_points)

    # Compute fuselage trajectory coordinates
    fuselage_trajectory_x = r_max * np.sin(theta_values + time_step)
    fuselage_trajectory_y = r_max * np.cos(theta_values + time_step)
    
    return fuselage_trajectory_x, fuselage_trajectory_y

# Define function to generate qubits
def generate_qubits(trajectory_x, trajectory_y, trajectory_z, color):
    qubit_coordinates_x = trajectory_x[::10]  # Sample every 10th point for qubits
    qubit_coordinates_y = trajectory_y[::10]
    qubit_coordinates_z = trajectory_z[::10]
    ax.scatter(qubit_coordinates_x, qubit_coordinates_y, qubit_coordinates_z, color=color, marker='o')
    return qubit_coordinates_x, qubit_coordinates_y, qubit_coordinates_z

# Define function to draw transport lines from one qubit to all other qubits
def draw_transport_lines(qubit_coordinates_x, qubit_coordinates_y, qubit_coordinates_z):
    for i in range(len(qubit_coordinates_x)):
        for j in range(len(qubit_coordinates_x)):
            if i != j:
                ax.plot([qubit_coordinates_x[i], qubit_coordinates_x[j]], 
                        [qubit_coordinates_y[i], qubit_coordinates_y[j]], 
                        [qubit_coordinates_z[i], qubit_coordinates_z[j]], 
                        color='silver', alpha=0.5)

# Define function to draw ribs connecting fuselage to qubits
def draw_ribs(fuselage_trajectory_x, fuselage_trajectory_y, qubit_coordinates_x, qubit_coordinates_y):
    for i in range(len(qubit_coordinates_x)):
        ax.plot([fuselage_trajectory_x[i], qubit_coordinates_x[i]], 
                [fuselage_trajectory_y[i], qubit_coordinates_y[i]], 
                color='blue', alpha=0.5)

# Function to update the plot for each frame of the animation
def update(frame):
    ax.cla()  # Clear the current axes
    time_step = frame * animation_speed
    fuselage_trajectory_x, fuselage_trajectory_y = compute_fuselage_trajectory(num_points, time_step)
    ax.plot(fuselage_trajectory_x, fuselage_trajectory_y, np.zeros_like(fuselage_trajectory_x), color='black')
    trajectory_x, trajectory_y, trajectory_z = compute_rocket_trajectory(num_points, time_step)
    qubit_coordinates_x, qubit_coordinates_y, qubit_coordinates_z = generate_qubits(trajectory_x, trajectory_y, trajectory_z, 'silver')
    draw_transport_lines(qubit_coordinates_x, qubit_coordinates_y, qubit_coordinates_z)
    draw_ribs(fuselage_trajectory_x, fuselage_trajectory_y, qubit_coordinates_x, qubit_coordinates_y)

# Generate initial trajectory
num_points = 100
num_trajectories = 10
time_steps = np.linspace(0, 2*np.pi, num_trajectories)  # Different initial time steps for each trajectory

# Create figure and subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot initial trajectory and qubits
fuselage_trajectory_x, fuselage_trajectory_y = compute_fuselage_trajectory(num_points, 0)
ax.plot(fuselage_trajectory_x, fuselage_trajectory_y, np.zeros_like(fuselage_trajectory_x), color='black')
trajectory_x, trajectory_y, trajectory_z = compute_rocket_trajectory(num_points, 0)
generate_qubits(trajectory_x, trajectory_y, trajectory_z, 'silver')
draw_transport_lines(trajectory_x, trajectory_y, trajectory_z)
draw_ribs(fuselage_trajectory_x, fuselage_trajectory_y, trajectory_x, trajectory_y)

# Set animation parameters
num_frames = 100
animation_speed = 1.0  # Control the speed of the animation

# Create animation
ani = FuncAnimation(fig, update, frames=num_frames, interval=50)

plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Define function to compute rocket-shaped trajectory
def compute_rocket_trajectory(num_points, time_step):
    # Define parameters for trajectory generation
    r_max = 50  # Maximum radial distance
    theta_max = np.pi  # Maximum theta angle
    phi_max = 2 * np.pi  # Maximum phi angle

    # Generate values for theta and phi
    theta_values = np.linspace(0, theta_max, num_points)
    phi_values = np.linspace(0, phi_max, num_points)

    # Compute rocket-shaped trajectory coordinates
    trajectory_x = r_max * np.sin(theta_values + time_step) * np.cos(phi_values)
    trajectory_y = r_max * np.sin(theta_values + time_step) * np.sin(phi_values)
    trajectory_z = r_max * np.cos(theta_values + time_step)

    # Modify z-coordinate to create rocket shape
    trajectory_z += 0.3 * r_max * np.cos(phi_values)

    return trajectory_x, trajectory_y, trajectory_z

# Define function to compute fuselage trajectory
def compute_fuselage_trajectory(num_points, time_step):
    # Define parameters for trajectory generation
    r_max = 50  # Maximum radial distance
    theta_max = np.pi  # Maximum theta angle

    # Generate random values for theta
    theta_values = np.linspace(0, theta_max, num_points)

    # Compute fuselage trajectory coordinates
    fuselage_trajectory_x = r_max * np.sin(theta_values + time_step)
    fuselage_trajectory_y = r_max * np.cos(theta_values + time_step)
    
    return fuselage_trajectory_x, fuselage_trajectory_y

# Define function to generate qubits
def generate_qubits(trajectory_x, trajectory_y, trajectory_z, color):
    qubit_coordinates_x = trajectory_x[::10]  # Sample every 10th point for qubits
    qubit_coordinates_y = trajectory_y[::10]
    qubit_coordinates_z = trajectory_z[::10]
    ax.scatter(qubit_coordinates_x, qubit_coordinates_y, qubit_coordinates_z, color=color, marker='o')
    return qubit_coordinates_x, qubit_coordinates_y, qubit_coordinates_z

# Define function to draw transport lines from one qubit to all other qubits
def draw_transport_lines(qubit_coordinates_x, qubit_coordinates_y, qubit_coordinates_z):
    for i in range(len(qubit_coordinates_x)):
        for j in range(len(qubit_coordinates_x)):
            if i != j:
                ax.plot([qubit_coordinates_x[i], qubit_coordinates_x[j]], 
                        [qubit_coordinates_y[i], qubit_coordinates_y[j]], 
                        [qubit_coordinates_z[i], qubit_coordinates_z[j]], 
                        color='silver', alpha=0.5)

# Define function to draw fuselage
def draw_fuselage(fuselage_trajectory_x, fuselage_trajectory_y):
    ax.plot(fuselage_trajectory_x, fuselage_trajectory_y, color='black')

# Define function to draw ribs connecting fuselage to qubits
def draw_ribs(fuselage_trajectory_x, fuselage_trajectory_y, qubit_coordinates_x, qubit_coordinates_y):
    for i in range(len(qubit_coordinates_x)):
        ax.plot([fuselage_trajectory_x[i], qubit_coordinates_x[i]], 
                [fuselage_trajectory_y[i], qubit_coordinates_y[i]], 
                color='blue', alpha=0.5)

# Generate initial trajectory
num_points = 100
num_trajectories = 10
time_steps = np.linspace(0, 2*np.pi, num_trajectories)  # Different initial time steps for each trajectory

# Create figure and subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each trajectory and place qubits
for i, time_step in enumerate(time_steps):
    fuselage_trajectory_x, fuselage_trajectory_y = compute_fuselage_trajectory(num_points, time_step)
    ax.plot(fuselage_trajectory_x, fuselage_trajectory_y, np.zeros_like(fuselage_trajectory_x), color='black')
    trajectory_x, trajectory_y, trajectory_z = compute_rocket_trajectory(num_points, time_step)
    qubit_coordinates_x, qubit_coordinates_y, qubit_coordinates_z = generate_qubits(trajectory_x, trajectory_y, trajectory_z, 'silver')
    draw_transport_lines(qubit_coordinates_x, qubit_coordinates_y, qubit_coordinates_z)
    draw_ribs(fuselage_trajectory_x, fuselage_trajectory_y, qubit_coordinates_x, qubit_coordinates_y)

# Add labels and title
ax.set_xlabel('X (Distance)')
ax.set_ylabel('Y (Distance)')
ax.set_zlabel('Z (Distance)')
ax.set_title('Fuselage with Rocket Trajectory and Ribs')

plt.show()
