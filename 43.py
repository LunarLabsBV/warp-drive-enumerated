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

# Define PID control gains
Kp = 1.0  # Proportional gain
Ki = 0.1  # Integral gain
Kd = 0.2  # Derivative gain

# Initialize integral term for PID control
integral = 0

# Define function to animate tangent function with PID control
def update(frame):
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
    
    ax.clear()
    # Plot the warp thrust (animated tangent function)
    ax.plot(trajectory_x, trajectory_y, trajectory_z, color='blue', alpha=0.5)  
    
    # Plot the fuselage (3D model of the spacecraft)
    fuselage_x = np.linspace(-0.2, 0.2, 10)
    fuselage_y = np.linspace(-0.1, 0.1, 10)
    fuselage_X, fuselage_Y = np.meshgrid(fuselage_x, fuselage_y)
    fuselage_Z = np.zeros_like(fuselage_X)
    ax.plot_surface(fuselage_X, fuselage_Y, fuselage_Z, color='gray', alpha=0.8)  # Fuselage
    ax.scatter(0, 0, 0, color='green', s=100)  # Cockpit
    ax.set_xlabel('X (Distance)')
    ax.set_ylabel('Y (Distance)')
    ax.set_zlabel('Z (Distance)')
    ax.set_title(f'Roman Warp Drive Spaceship Trajectory (Frame {frame})')

# Create animation
ani = FuncAnimation(fig, update, frames=np.linspace(0.1, 200, 75), interval=50)

plt.show()
