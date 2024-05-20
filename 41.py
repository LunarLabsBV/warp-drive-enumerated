import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Create figure and subplots
fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Set axis limits for subplot 2
ax2.set_xlim(-1, 1)
ax2.set_ylim(-1, 1)
ax2.set_zlim(-1, 1)

# Define PID control gains
Kp = 1.0  # Proportional gain
Ki = 0.1  # Integral gain
Kd = 0.2  # Derivative gain

# Initialize integral term for PID control
integral = 0

# Define data for subplot 1
x_values = np.linspace(0, 10, 100)
channel_curve = np.sin(x_values)
np.random.seed(0)
noise = np.random.normal(0, 0.1, len(x_values))
perturbed_channel_curve = channel_curve + noise
lower_bound = channel_curve - 0.2
upper_bound = channel_curve + 0.2
perturbed_channel_curve = np.clip(perturbed_channel_curve, lower_bound, upper_bound)

# Plot initial data for subplot 1
line1, = ax1.plot(x_values, np.zeros_like(x_values), channel_curve, label='Original Channel Curve', color='blue')
line2, = ax1.plot(x_values, np.zeros_like(x_values), perturbed_channel_curve, label='Perturbed Channel Curve', color='red', alpha=0.7)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Communication Channel')
ax1.set_title('Perturbation of Radial Communication Channel (Hawking)')
ax1.legend()
ax1.grid(True)

# Define update function for subplot 1
def update_subplot1(frame):
    line2.set_ydata(np.random.normal(0, 0.1, len(x_values)))
    return line2,

# Define function to animate tangent function with PID control
def update_subplot2(frame):
    time_step = np.linspace(0, 2*np.pi, 100)  # Time steps
    trajectory_x = np.sin(time_step) * np.cos(time_step)
    trajectory_y = np.sin(time_step) * np.sin(time_step)
    trajectory_z = np.zeros_like(time_step)
    color = np.cos(time_step)  # Using cosine function to represent color
    size = np.abs(np.sin(frame * time_step)) * 50  # Using sine function to represent size
    
    # Spacecraft position (arbitrary)
    spacecraft_x = np.sin(frame) * 0.5
    spacecraft_y = np.cos(frame) * 0.5
    spacecraft_z = np.sin(frame * 2) * 0.2
    
    ax2.clear()
    ax2.scatter(trajectory_x, trajectory_y, trajectory_z, c=color, s=size)
    ax2.scatter(spacecraft_x, spacecraft_y, spacecraft_z, color='green', label='Quantum Spacecraft')  # Adding spacecraft
    ax2.set_xlabel('X (Distance)')
    ax2.set_ylabel('Y (Distance)')
    ax2.set_zlabel('Z (Distance)')
    ax2.set_title(f'Trajectory with Animated Tangent Function (Frame {frame})')
    ax2.legend()

# Create animations
ani1 = FuncAnimation(fig, update_subplot1, frames=np.linspace(0, 10, 100), interval=50, blit=True)
ani2 = FuncAnimation(fig, update_subplot2, frames=np.linspace(0.1, 200, 75), interval=50)

plt.show()
