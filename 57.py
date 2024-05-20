import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# Define data for the Turing Machine
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 - Y**2  # Placeholder function for curvature, adjust as needed
quantum_superconductors_x = np.random.uniform(-5, 5, 284)  # Increased to 284 superconductors
quantum_superconductors_y = np.random.uniform(-5, 5, 284)  # Increased to 284 superconductors
turing_machine = np.zeros((10, 10))  # Placeholder for Turing Machine grid

# Generate hypothetical fourth-dimensional data
fourth_dimension_data = np.random.rand(len(quantum_superconductors_x))

# Create the plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the perturbed communication channels with stock fluctuation style and singularity wormholes
for qx1, qy1, fourth_dim in zip(quantum_superconductors_x, quantum_superconductors_y, fourth_dimension_data):
    min_dist = float('inf')
    nearest_x, nearest_y = None, None
    for qx2, qy2 in zip(quantum_superconductors_x, quantum_superconductors_y):
        if qx1 != qx2 or qy1 != qy2:  # Skip if the same point
            dist = np.sqrt((qx1 - qx2)**2 + (qy1 - qy2)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_x, nearest_y = qx2, qy2
    # Perturb the channel by adding noise to coordinates with stock fluctuation style
    perturbed_qx = [qx1, nearest_x]
    perturbed_qy = [qy1, nearest_y]
    # Add random fluctuations to the lines
    num_segments = 20  # Adjust the number of line segments
    for _ in range(num_segments - 1):
        new_x = perturbed_qx[-1] + np.random.normal(0, 0.35)
        new_y = perturbed_qy[-1] + np.random.normal(0, 0.35)
        perturbed_qx.append(new_x)
        perturbed_qy.append(new_y)
    # Add singularity wormhole at the middle of each connecting line segment
    for i in range(len(perturbed_qx) - 1):
        mid_x = (perturbed_qx[i] + perturbed_qx[i + 1]) / 2
        mid_y = (perturbed_qy[i] + perturbed_qy[i + 1]) / 2
        mid_z = np.mean(perturbed_channel_curve)  # Use mean of perturbed channel curve for z-coordinate
        ax.scatter(mid_x, mid_y, mid_z, c='red', s=2)  # Singularity wormhole point
    ax.plot(perturbed_qx, perturbed_qy, np.mean(perturbed_channel_curve), color=plt.cm.viridis(fourth_dim), linewidth=2, alpha=0.5)

# Plot the quantum superconductors
ax.scatter(quantum_superconductors_x, quantum_superconductors_y, np.mean(perturbed_channel_curve), c=fourth_dimension_data, cmap='viridis', label='Quantum Superconductors')

# Plot the Turing Machine grid
ax.imshow(turing_machine, cmap='gray', extent=(-5, 5, -5, 5), alpha=0.5)

# Add arrows indicating the flow of influence
ax.text(-4, 4, 0, "Kerr Gravity", color='blue')
ax.text(0, 4, 0, "Quantum Mechanics", color='green')
ax.text(4, 4, 0, "Turing Machine", color='gray')

# Add labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('Interaction of Quantum Superconductors through Perturbed Hawking Channels')

# Normalize the range of each axis to set the aspect ratio manually
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(np.min(perturbed_channel_curve), np.max(perturbed_channel_curve))

# Create a ScalarMappable for the colorbar
sm = plt.cm.ScalarMappable(cmap='viridis')
sm.set_array(fourth_dimension_data)

# Add colorbar with a specific Axes for placement
fig.colorbar(sm, ax=ax, label='Fourth Dimension')

# Show the plot
plt.show()
