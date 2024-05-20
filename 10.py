import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Define function to compute perturbed fuselage trajectory
def compute_perturbed_fuselage_trajectory(num_points, time_step, perturbation_factor):
    # Define parameters for trajectory generation
    r_max = 50  # Maximum radial distance
    theta_max = np.pi  # Maximum theta angle

    # Generate random values for theta
    theta_values = np.linspace(0, theta_max, num_points)

    # Compute fuselage trajectory coordinates with perturbation
    fuselage_trajectory_x = r_max * np.cos(theta_values + time_step)
    fuselage_trajectory_y = r_max * np.tan(theta_values + time_step)
    
    # Apply perturbation to the fuselage trajectory along the y-axis
    fuselage_trajectory_y_perturbed = fuselage_trajectory_y + perturbation_factor * np.sin(theta_values + time_step)

    return fuselage_trajectory_x, fuselage_trajectory_y_perturbed

# Define function to generate qubits
def generate_qubits(trajectory_x, trajectory_y, color):
    qubit_coordinates_x = trajectory_x[::10]  # Sample every 10th point for qubits
    qubit_coordinates_y = trajectory_y[::10]
    return qubit_coordinates_x, qubit_coordinates_y

# Define function to update the plot
def update(frame):
    ax.clear()
    ax.set_xlabel('X (Distance)')
    ax.set_ylabel('Y (Distance)')
    ax.set_title('Perturbed Fuselage Trajectory with Ribs')

    # Plot fuselage and draw ribs
    fuselage_trajectory_x, fuselage_trajectory_y_perturbed = compute_perturbed_fuselage_trajectory(num_points, time_steps[frame], perturbation_factor)
    ax.plot(fuselage_trajectory_x, fuselage_trajectory_y_perturbed, color='black')

    qubit_coordinates_x, qubit_coordinates_y = generate_qubits(fuselage_trajectory_x, fuselage_trajectory_y_perturbed, 'goldenrod')
    for i in range(len(qubit_coordinates_x)):
        ax.plot([fuselage_trajectory_x[i], qubit_coordinates_x[i]], 
                [fuselage_trajectory_y_perturbed[i], qubit_coordinates_y[i]], 
                color='blue', alpha=0.5)

# Generate initial fuselage trajectory
num_points = 100
num_trajectories = 10
time_steps = np.linspace(0, 2*np.pi, num_trajectories)

# Create figure and subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

# Set perturbation factor
perturbation_factor = 0.5

# Create animation
ani = FuncAnimation(fig, update, frames=len(time_steps), interval=200, repeat=True)

plt.show()
