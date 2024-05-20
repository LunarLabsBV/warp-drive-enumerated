import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create figure and subplot
fig, ax = plt.subplots(figsize=(10, 8))

# Set axis limits
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

# Define function for gradient descent
def gradient_descent(tangent_factor, learning_rate):
    return tangent_factor - learning_rate * np.tan(tangent_factor)

# Define PID control gains
Kp = 1.0  # Proportional gain
Ki = 0.1  # Integral gain
Kd = 0.2  # Derivative gain

# Initialize integral term for PID control
integral = 0

# Initialize variables for user interaction
derivative_point = None

# Define function to handle mouse click event
def onclick(event):
    global derivative_point
    if event.inaxes == ax:
        derivative_point = (event.xdata, event.ydata)
        update(0)

# Connect the mouse click event to the onclick function
fig.canvas.mpl_connect('button_press_event', onclick)

# Define function to animate tangent function with PID control
def update(frame):
    time_step = np.linspace(0, 2*np.pi, 100)  # Time steps
    trajectory_x = np.sin(time_step) * np.cos(time_step)
    trajectory_y = np.sin(time_step) * np.sin(time_step)
    
    global integral  # Access the integral term
    global derivative_point  # Access the derivative point
    
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
    ax.plot(trajectory_x, trajectory_y, color='blue', alpha=0.5, label='Warp Thrust')  
    
    # Plot the fuselage (2D representation of the derivative of the CTC)
    fuselage_x = np.linspace(-0.2, 0.2, 10)
    fuselage_y = np.zeros_like(fuselage_x)
    ax.plot(fuselage_x, fuselage_y, color='gray', alpha=0.8, label='Derivative of CTC')  # Fuselage
    
    # Draw the derivative line if a point on the CTC is selected
    if derivative_point is not None:
        x, y = derivative_point
        ax.plot([x, 0], [y, 0], color='red', linestyle='--', label='Derivative Line')  # Derivative Line
        ax.scatter(x, y, color='red', label='Derivative Point')  # Derivative Point
        
        # Find the index where the derivative line intersects with the CTC
        intersection_idx = np.argmin(np.abs(trajectory_x - x))
        
        # Calculate the integral under the line up to the intersection point with the CTC
        integral_value = np.trapz(np.abs(y - trajectory_y[:intersection_idx]), x=trajectory_x[:intersection_idx])
        ax.text(0, -0.9, f'Integral: {integral_value:.3f}', ha='center', va='center')
        
    ax.scatter(0, 0, color='green', s=100, label='Cockpit')  # Cockpit
    ax.set_xlabel('X (Distance)')
    ax.set_ylabel('Y (Distance)')
    ax.set_title(f'Roman Warp Drive Spaceship Trajectory (Frame {frame})')
    ax.legend()

# Create animation
ani = FuncAnimation(fig, update, frames=np.linspace(0.1, 200, 75), interval=50)

plt.show()
