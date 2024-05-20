import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Create figure and subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Set axis limits
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

# Define function for Bose Factor
def bose_factor(energy, mu, T):
    k = 1  # Boltzmann constant for simplicity
    return 1 / (np.exp((energy - mu) / (k * T)) - 1)

# Define parameters
T = 1.0  # Temperature
mu = 0.5  # Chemical potential

# Define energy levels
time_step = np.linspace(0, 2 * np.pi, 100)  # Time steps
energies = np.sin(time_step)

# Define function to animate trajectory with Bose Factor
def update(frame):
    #trajectory_x = np.cos(time_step + frame / 10) * np.cos(time_step + frame / 10) # For a slimmed plasma drive
    trajectory_x = np.sin(time_step + frame / 10) * np.cos(time_step + frame / 10) # For a circular warp drive
    trajectory_y = np.sin(time_step + frame / 10) * np.sin(time_step + frame / 10)
    
    # Calculate z-coordinate based on Bose Factor
    bose_factors = bose_factor(energies, mu, T)
    trajectory_z = -np.clip(bose_factors * 0.5, 0, np.inf)
    
    ax.clear()
    ax.plot(trajectory_x, trajectory_y, trajectory_z)
    ax.set_xlabel('X (Distance)')
    ax.set_ylabel('Y (Distance)')
    ax.set_zlabel('Z (Distance)')
    ax.set_title(f'Trajectory with Animated Bose Factor (Frame {frame})')

# Create animation
ani = FuncAnimation(fig, update, frames=np.linspace(0.1, 200, 75), interval=50)

plt.show()
