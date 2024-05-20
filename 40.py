import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Create figure and subplots
fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

# Set axis limits for subplot 2
ax2.set_xlim(-1, 1)
ax2.set_ylim(-1, 1)
ax2.set_zlim(-1, 1)

# Define function for gradient descent
def gradient_descent(tangent_factor, learning_rate):
    return tangent_factor - learning_rate * np.tan(tangent_factor)

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
line1, = ax1.plot(x_values, channel_curve, label='Original Channel Curve', color='blue')
line2, = ax1.plot(x_values, perturbed_channel_curve, label='Perturbed Channel Curve', color='red', alpha=0.7)
ax1.fill_between(x_values, lower_bound, upper_bound, color='gray', alpha=0.2, label='Bounds')
ax1.set_xlabel('Radial Distance')
ax1.set_ylabel('Communication Channel')
ax1.set_title('Perturbation of Radial Communication Channel (Hawking)')
ax1.legend()
ax1.grid(True)

# Define update function for subplot 1
def update_subplot1(frame):
    line2.set_ydata(channel_curve + np.random.normal(0, 0.1, len(x_values)))
    return line2,

# Define function to animate tangent function with PID control
def update_subplot2(frame):
    time_step = np.linspace(0, 2*np.pi, 100)  # Time steps
    trajectory_x = np.sin(time_step) * np.cos(time_step)
    trajectory_y = np.sin(time_step) * np.sin(time_step)
    
    global integral  # Access the integral term
    
    # Calculate error signal
    error = np.sin(time_step) - np.sin(time_step + frame / 10)
    
    # Calculate integral of error
    integral += np.sum(error)
    
    # Calculate derivative of error
    derivative = np.diff(error) / np.diff(time_step)
    derivative = np.insert(derivative, 0, derivative[0])  # Pad derivative array
    
    # Calculate control signal using PID control law
    control_signal = Kp * error + Ki * integral + Kd * derivative
    
    # Perform gradient descent for the tangent factor
    tangent_factor = frame
    learning_rate = 0.1
    
    # Update tangent factor using PID control
    tangent_factor += learning_rate * np.mean(control_signal)  # Adjusted by the mean control signal
    
    # Bound the z-coordinate to be non-negative (perturbation at bottom half of curve)
    trajectory_z = -np.clip(np.tan(tangent_factor * time_step), 0, np.inf)
    
    ax2.clear()
    ax2.plot(trajectory_x, trajectory_y, trajectory_z)
    ax2.set_xlabel('X (Distance)')
    ax2.set_ylabel('Y (Distance)')
    ax2.set_zlabel('Z (Distance)')
    ax2.set_title(f'Trajectory with Animated Tangent Function (Frame {frame})')

# Create animations
ani1 = FuncAnimation(fig, update_subplot1, frames=np.linspace(0, 10, 100), interval=50, blit=True)
ani2 = FuncAnimation(fig, update_subplot2, frames=np.linspace(0.1, 200, 75), interval=50)

plt.show()
