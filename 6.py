import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define the functions and their derivatives
def f1(rho):
    return 1 - rho**2

def df1(rho):
    return -2 * rho

def f2(rho):
    return rho**2 - 1

def df2(rho):
    return 2 * rho

# Apply Newton-Raphson method
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

# Apply Newton-Raphson method to both equations
root1, iterations1, residuals1 = newton_raphson(f1, df1, initial_guess, tolerance)
root2, iterations2, residuals2 = newton_raphson(f2, df2, initial_guess, tolerance)

# Generate rho values
rho_values = np.linspace(0, 2, 100)
f1_values = f1(rho_values)
f2_values = f2(rho_values)

# Plot the functions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [2, 1]})
ax1.plot(rho_values, f1_values, label=r'$1 - \rho^2$')
ax1.plot(rho_values, f2_values, label=r'$\rho^2 - 1$', color='orange')
ax1.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax1.scatter(root1, 0, color='red', label='Root of Equation 1')
ax1.scatter(root2, 0, color='blue', label='Root of Equation 2')
ax1.set_xlabel(r'$\rho$')
ax1.set_ylabel('Function Value')
ax1.set_title('Convergence of Newton-Raphson Method')
ax1.legend()
ax1.grid(True)

# Define function for gradient descent
def gradient_descent(tangent_factor, learning_rate):
    return tangent_factor - learning_rate * np.tan(tangent_factor)

# Define function to animate tangent function with gradient descent
def update(frame):
    tangent_factor = frame
    learning_rate = 0.1
    
    # Perform gradient descent for the tangent factor
    tangent_factor = gradient_descent(tangent_factor, learning_rate)
    
    # Apply perturbation to f1 and f2
    perturbation1 = -np.tan(tangent_factor * (rho_values - root1))
    perturbation2 = -np.tan(tangent_factor * (rho_values - root2))
    f1_values_perturbed = np.clip(f1_values * perturbation1, -np.inf, 0)
    f2_values_perturbed = np.clip(f2_values * perturbation2, -np.inf, 0)
    
    # Plot the perturbed functions
    ax1.clear()
    ax1.plot(rho_values, f1_values_perturbed, label=r'$1 - \rho^2$')
    ax1.plot(rho_values, f2_values_perturbed, label=r'$\rho^2 - 1$', color='orange')
    ax1.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax1.scatter(root1, 0, color='red', label='Root of Equation 1')
    ax1.scatter(root2, 0, color='blue', label='Root of Equation 2')
    ax1.set_xlabel(r'$\rho$')
    ax1.set_ylabel('Function Value')
    ax1.set_title('Convergence of Newton-Raphson Method')
    ax1.legend()
    ax1.grid(True)
    
    # Update the error convergence subplot
    ax2.plot(range(len(residuals1)), residuals1, label='Residuals for Equation 1')
    ax2.plot(range(len(residuals2)), residuals2, label='Residuals for Equation 2')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Residual')
    ax2.set_title('Error Convergence')
    ax2.legend()

# Create animation
ani = FuncAnimation(fig, update, frames=np.linspace(0.1, 200, 75), interval=50)

plt.show()
