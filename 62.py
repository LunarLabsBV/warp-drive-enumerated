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

# Define function for gradient descent
def gradient_descent(tangent_factor, learning_rate):
    return tangent_factor - learning_rate * np.tan(tangent_factor)

# Define function to animate tangent function with gradient descent
def update(frame):
    time_step = np.linspace(0, 2*np.pi, 100)  # Time steps
    trajectory_x = np.sin(time_step) * np.cos(time_step)
    trajectory_y = np.sin(time_step) * np.sin(time_step)
    
    # Initialize tangent factor and learning rate
    tangent_factor = frame
    learning_rate = 0.1
    
    # Perform gradient descent for the tangent factor
    tangent_factor = gradient_descent(tangent_factor, learning_rate)
    
    # Bound the z-coordinate to be non-negative (perturbation at bottom half of curve)
    trajectory_z = -np.clip(np.tan(tangent_factor * time_step), 0, np.inf)
    
    ax.clear()
    ax.plot(trajectory_x, trajectory_y, trajectory_z)
    ax.set_xlabel('X (Distance)')
    ax.set_ylabel('Y (Distance)')
    ax.set_zlabel('Z (Distance)')
    ax.set_title(f'Trajectory with Animated Tangent Function (Frame {frame})')

# Create animation
ani = FuncAnimation(fig, update, frames=np.linspace(0.1, 200, 75), interval=50)

plt.show()
