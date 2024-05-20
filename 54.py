import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define function to compute trajectory resembling a pottery pot with perturbation
def compute_pot_trajectory_with_perturbation(num_points, time_step, perturbation_factor):
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

    # Apply perturbation to the trajectory along the z-axis
    pot_z_perturbed = pot_z + perturbation_factor * np.sin(theta_values + time_step)

    return pot_x, pot_y, pot_z_perturbed

# Generate initial trajectory with perturbation
num_points = 200
num_trajectories = 100
time_steps = np.linspace(0, 2*np.pi, num_trajectories)  # Different initial time steps for each trajectory

# Create figure and subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Define perturbation factor
perturbation_factor = 2

# Plot each trajectory and place qubits
lines = [ax.plot([], [], [], color='brown')[0] for _ in range(num_trajectories)]
ax.set_xlabel('X (Distance)')
ax.set_ylabel('Y (Distance)')
ax.set_zlabel('Z (Distance)')
ax.set_title('Pottery Pot Trajectories with Perturbation')

def init():
    for line in lines:
        line.set_data([], [])
        line.set_3d_properties([])
    return lines

def update(frame):
    pot_x, pot_y, pot_z_perturbed = compute_pot_trajectory_with_perturbation(num_points, time_steps[frame], perturbation_factor)
    for i, line in enumerate(lines):
        line.set_data(pot_x, pot_y)
        line.set_3d_properties(pot_z_perturbed)
    return lines

ani = FuncAnimation(fig, update, frames=len(time_steps), init_func=init, interval=50, blit=True)

plt.show()
