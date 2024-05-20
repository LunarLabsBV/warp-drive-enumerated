import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Create figure and subplots
fig = plt.figure(figsize=(15, 8))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)

# Set axis limits for subplot 1
ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 1)
ax1.set_zlim(-1, 1)

# Set axis limits for subplot 2
ax2.set_xlim(0, 2)
ax2.set_ylim(-2, 2)

# Define function for gradient descent
def gradient_descent(tangent_factor, learning_rate):
    return tangent_factor - learning_rate * np.tan(tangent_factor)

# Define function to animate tangent function with gradient descent for subplot 1
def update_3d(frame):
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
    
    ax1.clear()
    ax1.plot(trajectory_x, trajectory_y, trajectory_z)
    ax1.set_xlabel('X (Distance)')
    ax1.set_ylabel('Y (Distance)')
    ax1.set_zlabel('Z (Distance)')
    ax1.set_title(f'Trajectory with Animated Tangent Function (Frame {frame})')

# Define the functions and their derivatives for subplot 2
def f1(rho):
    return 1 - rho**2

def df1(rho):
    return -2 * rho

def f2(rho):
    return rho**2 - 1

def df2(rho):
    return 2 * rho

# Apply Newton-Raphson method for subplot 2
def newton_raphson(f, df, initial_guess, tolerance=1e-6, max_iterations=100):
    rho = initial_guess
    iterations = 0
    while abs(f(rho)) > tolerance and iterations < max_iterations:
        rho = rho - f(rho) / df(rho)
        iterations += 1
    return rho, iterations

# Initial guess and tolerance
initial_guess = 1.777
tolerance = 1e-6

# Apply Newton-Raphson method to both equations for subplot 2
root1, iterations1 = newton_raphson(f1, df1, initial_guess, tolerance)
root2, iterations2 = newton_raphson(f2, df2, initial_guess, tolerance)

print("Equation 1: Root =", root1, "Iterations =", iterations1)
print("Equation 2: Root =", root2, "Iterations =", iterations2)

# Plot the functions for subplot 2
rho_values = np.linspace(0, 2, 100)
f1_values = f1(rho_values)
f2_values = f2(rho_values)

# Define function for gradient descent for subplot 2
def gradient_descent(tangent_factor, learning_rate):
    return tangent_factor - learning_rate * np.tan(tangent_factor)

# Define function to animate tangent function with gradient descent for subplot 2
def update_2d(frame):
    tangent_factor = frame
    learning_rate = 0.1
    
    # Perform gradient descent for the tangent factor
    tangent_factor = gradient_descent(tangent_factor, learning_rate)
    
    # Find indices before and after the root for both functions
    index_before_root_f1 = np.where(rho_values < root1)[0][-1]  # Last index before the root of f1
    index_after_root_f1 = np.where(rho_values >= root1)[0][0]  # First index at or after the root of f1
    
    index_before_root_f2 = np.where(rho_values < root2)[0][-1]  # Last index before the root of f2
    index_after_root_f2 = np.where(rho_values >= root2)[0][0]  # First index at or after the root of f2
    
    # Apply perturbation separately before and after the root for both functions
    f1_values_skewed = np.copy(f1_values)
    f2_values_skewed = np.copy(f2_values)
    
    # Perturbation for f1
    f1_values_skewed[:index_after_root_f1] *= np.clip(-np.tan(tangent_factor * (rho_values[:index_after_root_f1] - root1)), -np.inf, 0)
    
    # Perturbation for f2
    f2_values_skewed[:index_after_root_f2] *= np.clip(-np.tan(tangent_factor * (rho_values[:index_after_root_f2] - root2)), -np.inf, 0)
    
    ax2.clear()
    ax2.plot(rho_values, f1_values_skewed, label=r'$1 - \rho^2$')
    ax2.plot(rho_values, f2_values_skewed, label=r'$\rho^2 - |1\rangle$', color='orange')
    ax2.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax2.scatter(root1, 0, color='red', label='Root of Equation 1')
    ax2.scatter(root2, 0, color='blue', label='Root of Equation 2')
    
    # Draw circular loop for f1
    theta_f1 = np.linspace(0, np.pi, 100)  # Angle values from 0 to pi
    rho_circle_f1 = root1 - 0.1 * np.cos(theta_f1)  # Radius 0.1 for circular loop
    f1_circle = 0.1 * np.sin(theta_f1)  # Y-values of the circle
    
    ax2.plot(rho_circle_f1, f1_circle, color='green', linestyle='-', linewidth=2)
    
    # Mark "X" within the loop for f1
    x_pos_f1 = root1 - 0.1  # Adjust x-position of "X"
    y_pos_f1 = 0.1  # Adjust y-position of "X"
    ax2.text(x_pos_f1, y_pos_f1, "X", color='green', fontsize=12, ha='center', va='center')
    
    ax2.set_xlabel(r'$\rho$')
    ax2.set_ylabel('Function Value')
    ax2.set_title('Convergence of Newton-Raphson Method with Skewed Curve and Closed Loop')
    ax2.legend()
    ax2.grid(True)

# Create animations
ani_3d = FuncAnimation(fig, update_3d, frames=np.linspace(0.1, 200, 75), interval=50)
ani_2d = FuncAnimation(fig, update_2d, frames=np.linspace(0.1, 200, 75), interval=50)

plt.show()
