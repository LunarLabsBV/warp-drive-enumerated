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

# Define function for Reubens Factor
def reubens_factor(energy, mu, T):
    # Reubens factor - just a playful term
    return np.sin(energy * T)

# Define parameters
T = 1.0  # Temperature
mu = 0.5  # Chemical potential

# Define energy levels
time_step = np.linspace(0, 2 * np.pi, 100)  # Time steps
energies = np.tan(time_step) # Any one of sin, cos, tan

# Initialize sandwich
sandwich_x = np.zeros_like(time_step)
sandwich_y = np.zeros_like(time_step)
sandwich_z = np.zeros_like(time_step)

# Define function to animate sandwich with Reubens Factor
def update(frame):
    global sandwich_x, sandwich_y, sandwich_z
    
    # Update sandwich
    sandwich_x = np.cos(time_step + frame / 10) * np.cos(time_step + frame / 10) # For a slimmed signature like oscillation
    # sandwich_x = np.sin(time_step + frame / 10) * np.cos(time_step + frame / 10) # For a circular motion
    sandwich_y = np.sin(time_step + frame / 10) * np.sin(time_step + frame / 10)
    
    # Calculate z-coordinate based on Reubens Factor
    reubens_factors = reubens_factor(energies, mu, T)
    sandwich_z = reubens_factors * 0.5
    
    # Plot sandwich
    ax.clear()
    ax.plot(sandwich_x, sandwich_y, sandwich_z)
    ax.set_xlabel('X (Width)')
    ax.set_ylabel('Y (Length)')
    ax.set_zlabel('Z (Grilledness)')
    ax.set_title(f'Sandwich Grilling Machine (Frame {frame})')

# Create animation
ani = FuncAnimation(fig, update, frames=np.linspace(0.1, 200, 75), interval=50)

plt.show()
