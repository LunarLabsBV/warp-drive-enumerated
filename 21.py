import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the range of time values
t_values = np.linspace(0, 10, 100)

# Define some placeholder values for the metric components (replace these with actual values)
g_00_values = np.sin(t_values)
g_11_values = np.cos(t_values)
g_22_values = np.tan(t_values)

# Define the radial communication channel function or points (replace this with actual data)
radial_channel_values = np.sqrt(t_values)  # Placeholder function

# Function to apply perturbation to g_22_values
def perturb_g_22(t_values, tangent_factor):
    return np.tan(t_values) + tangent_factor * np.sin(t_values)

# Initialize the figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot g_00, g_11, and radial communication channel
line_g_00, = ax.plot(t_values, g_00_values, label=r'$g_{00}$', color='blue')
line_g_11, = ax.plot(t_values, g_11_values, label=r'$g_{11}$', color='green')
line_radial_channel, = ax.plot(t_values, radial_channel_values, label='Radial Communication Channel', color='orange', linestyle='--')

# Define the line for g_22
line_g_22, = ax.plot(t_values, g_22_values, label=r'$g_{22}$', color='red')

# Function to update the plot for each frame
def update(frame):
    tangent_factor = frame / 100  # Adjust the range of tangent_factor as needed
    
    # Apply perturbation to g_22_values
    g_22_perturbed_values = perturb_g_22(t_values, tangent_factor)
    
    # Update the plot for g_22
    line_g_22.set_ydata(g_22_perturbed_values)
    
    ax.set_title(f'Metric Components with Radial Communication Channel and Perturbation (Tangent Factor: {tangent_factor})')

# Create animation
ani = FuncAnimation(fig, update, frames=range(100), interval=100)

# Show the plot
plt.xlabel('Time (t)')
plt.ylabel('Metric Components')
plt.title('Metric Components with Radial Communication Channel and Perturbation')
plt.grid(True)
plt.legend()
plt.show()
