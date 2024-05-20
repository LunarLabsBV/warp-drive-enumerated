import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the range for each variable
t_values = np.linspace(-10, 10, 100)
theta_values = np.linspace(0, 2*np.pi, 100)
r_values = np.linspace(0, 10, 100)

# Define the equations for each part
equations = [
    lambda t, tangent_factor: -t**2 + tangent_factor * np.sin(t),    # Part 1 with perturbation
    lambda theta, tangent_factor: theta**2,                          # Part 4
    lambda theta, tangent_factor: np.sin(theta)**2,                  # Part 5
    lambda phi, tangent_factor: phi**2,                              # Part 6
    lambda r, tangent_factor: r**2,                                  # Part 2 with perturbation
    lambda r, tangent_factor: r**2,                                  # Part 3 with perturbation
    lambda t, tangent_factor: -t**2,                                 # Part 7
    lambda r, tangent_factor: r**2,                                  # Part 8
    lambda r, tangent_factor: r**2                                   # Part 9
]

# Initialize the figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the equations
lines = [ax.plot([], [], label=f'Part {i+1}')[0] for i in range(len(equations))]

# Function to update the plot for each frame
def update(frame):
    tangent_factor = frame / 100  # Adjust the range of tangent_factor as needed
    for i, equation in enumerate(equations):
        if i != 2:  # Exclude Part 5 since it involves sine function
            lines[i].set_data(t_values, equation(t_values, tangent_factor))
    
    ax.set_title(f'CTC-Like Equations with Perturbation (Tangent Factor: {tangent_factor})')

# Create animation
ani = FuncAnimation(fig, update, frames=range(100), interval=100)

# Show the plot
plt.xlabel('Variable Value')
plt.ylabel('Equation Value')
plt.title('CTC-Like Equations with Perturbation')
plt.legend()
plt.grid(True)
plt.show()
