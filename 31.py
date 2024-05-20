import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the radial communication channel curve (example data)
x_values = np.linspace(0, 10, 100)
channel_curve = np.sin(x_values)

# Generate random noise for perturbation
np.random.seed(0)  # for reproducibility
noise = np.random.normal(0, 0.1, len(x_values))  # adjust the parameters for desired noise level

# Add noise to the channel curve
perturbed_channel_curve = channel_curve + noise

# Define the bounds for the perturbed channel curve relative to the original curve
lower_bound = channel_curve - 0.2  # Adjust the lower bound as needed
upper_bound = channel_curve + 0.2  # Adjust the upper bound as needed

# Ensure the perturbed channel curve stays within the bounds
perturbed_channel_curve = np.clip(perturbed_channel_curve, lower_bound, upper_bound)

# Create subplots within a single figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot the original and perturbed communication channel
ax1.plot(x_values, channel_curve, label='Original Channel Curve', color='blue')
ax1.plot(x_values, perturbed_channel_curve, label='Perturbed Channel Curve', color='red', alpha=0.7)
ax1.fill_between(x_values, lower_bound, upper_bound, color='gray', alpha=0.2, label='Bounds')
ax1.set_xlabel('Radial Distance')
ax1.set_ylabel('Communication Channel')
ax1.set_title('Perturbation of Radial Communication Channel (Hawking)')
ax1.legend()
ax1.grid(True)

# Define data for the animation
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 - Y**2  # Placeholder function for curvature, adjust as needed
quantum_points_x = np.random.uniform(-5, 5, 50)
quantum_points_y = np.random.uniform(-5, 5, 50)
quantum_superconductors_x = np.random.uniform(-5, 5, 20)
quantum_superconductors_y = np.random.uniform(-5, 5, 20)
turing_machine = np.zeros((10, 10))  # Placeholder for Turing Machine grid

# Initialize plot elements for the animation
contour = ax2.contour(X, Y, Z, levels=[0], colors='blue', linestyles='solid')
quantum_scatter = ax2.scatter(quantum_points_x, quantum_points_y, color='green', alpha=0.5)
quantum_superconductors_scatter = ax2.scatter(quantum_superconductors_x, quantum_superconductors_y, color='blue', alpha=0.8)
turing_machine_grid = ax2.imshow(turing_machine, cmap='gray', extent=(-5, 5, -5, 5), alpha=0.5)

# Arrows indicating the flow of influence
kerr_gravity_arrow = ax2.annotate("Kerr Gravity", xy=(-4, 4), xytext=(-4, 4.5),
                                 arrowprops=dict(facecolor='blue', arrowstyle='->'))
quantum_mechanics_arrow = ax2.annotate("Quantum Mechanics", xy=(0, 4), xytext=(0, 4.5),
                                       arrowprops=dict(facecolor='green', arrowstyle='->'))
quantum_superconductors_arrow = ax2.annotate("Quantum Superconductors", xy=(-4, 0), xytext=(-4, -0.5),
                                             arrowprops=dict(facecolor='blue', arrowstyle='->'))
turing_machine_arrow = ax2.annotate("Turing Machine", xy=(4, 4), xytext=(4, 4.5),
                                    arrowprops=dict(facecolor='gray', arrowstyle='->'))

# Title and axis labels
ax2.set_title('Flattened Turing Machine from Kerr Gravity, Quantum Mechanics, and Quantum Superconductors')
ax2.set_xlabel('X-axis')
ax2.set_ylabel('Y-axis')

# Set equal aspect ratio
ax2.set_aspect('equal')

# Hide axes
ax2.axis('off')

# Define update function for animation
def update(frame):
    # Rotate arrows
    kerr_gravity_arrow.set_position((-4 * np.cos(np.radians(frame)), 4 * np.sin(np.radians(frame))))
    quantum_mechanics_arrow.set_position((4 * np.sin(np.radians(frame)), 4 * np.cos(np.radians(frame))))
    quantum_superconductors_arrow.set_position((-4 * np.cos(np.radians(frame)), -4 * np.sin(np.radians(frame))))
    turing_machine_arrow.set_position((4 * np.cos(np.radians(frame)), 4 * np.sin(np.radians(frame))))
    
    # Rotate scatter points
    quantum_scatter.set_offsets(np.column_stack((quantum_points_x * np.cos(np.radians(frame)) - quantum_points_y * np.sin(np.radians(frame)),
                                                 quantum_points_x * np.sin(np.radians(frame)) + quantum_points_y * np.cos(np.radians(frame)))))
    
    quantum_superconductors_scatter.set_offsets(np.column_stack((quantum_superconductors_x * np.cos(np.radians(frame)) - quantum_superconductors_y * np.sin(np.radians(frame)),
                                                                 quantum_superconductors_x * np.sin(np.radians(frame)) + quantum_superconductors_y * np.cos(np.radians(frame)))))

# Create animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 1), interval=50)

plt.show()
