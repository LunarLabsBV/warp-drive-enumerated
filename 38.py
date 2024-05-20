import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

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
    residuals = []
    while abs(f(rho)) > tolerance and iterations < max_iterations:
        residual = f(rho)
        residuals.append(residual)
        rho = rho - residual / df(rho)
        iterations += 1
    return rho, iterations, residuals

# Initial guess and tolerance
initial_guess = 1.777
tolerance = 1e-6

# Apply Newton-Raphson method to both equations for subplot 2
root1, iterations1, residuals1 = newton_raphson(f1, df1, initial_guess, tolerance)
root2, iterations2, residuals2 = newton_raphson(f2, df2, initial_guess, tolerance)

# Plot the residual versus the root estimate
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [2, 1]})

# Define function for gradient descent for subplot 2
def gradient_descent(tangent_factor, learning_rate):
    return tangent_factor - learning_rate * np.tan(tangent_factor)

# Define function to animate tangent function with gradient descent for subplot 2
def update_2d(frame, rho_values, f1_values_skewed, f2_values_skewed):
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
    
    ax1.clear()
    ax1.plot(rho_values, f1_values_skewed, label=r'$1 - \rho^2$')
    ax1.plot(rho_values, f2_values_skewed, label=r'$\rho^2 - |1\rangle$', color='orange')
    ax1.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax1.scatter(root1, 0, color='red', label='Root of Equation 1')
    ax1.scatter(root2, 0, color='blue', label='Root of Equation 2')
    
    # Draw circular loop for f1
    theta_f1 = np.linspace(0, np.pi, 100)  # Angle values from 0 to pi
    rho_circle_f1 = root1 - 0.1 * np.cos(theta_f1)  # Radius 0.1 for circular loop
    f1_circle = 0.1 * np.sin(theta_f1)  # Y-values of the circle
    
    ax1.plot(rho_circle_f1, f1_circle, color='green', linestyle='-', linewidth=2)
    
    # Mark "X" within the loop for f1
    x_pos_f1 = root1 - 0.1  # Adjust x-position of "X"
    y_pos_f1 = 0.1  # Adjust y-position of "X"
    ax1.text(x_pos_f1, y_pos_f1, "X", color='green', fontsize=12, ha='center', va='center')
    
    ax1.set_xlabel(r'$\rho$')
    ax1.set_ylabel('Function Value')
    ax1.set_title('Convergence of Newton-Raphson Method with Skewed Curve and Closed Loop')
    ax1.legend()
    ax1.grid(True)
    
    # Plot the residual versus the root estimate
    ax2.clear()
    ax2.plot(range(len(residuals1)), residuals1, label='Residuals for Equation 1')
    ax2.plot(range(len(residuals2)), residuals2, label='Residuals for Equation 2')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Residual')
    ax2.set_title('Residual Plot for Newton-Raphson Method')
    ax2.legend()

# Generate rho values
rho_values = np.linspace(0, 2, 100)
f1_values = f1(rho_values)
f2_values = f2(rho_values)

# Create animation
ani_2d = FuncAnimation(fig, update_2d, fargs=(rho_values, f1_values, f2_values), frames=np.linspace(0.1, 200, 75), interval=50)

plt.show()
