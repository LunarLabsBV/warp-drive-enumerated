import numpy as np
import matplotlib.pyplot as plt

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
quantum_superconductors_x = np.random.uniform(-5, 5, 284)  # Increased to 20 superconductors
quantum_superconductors_y = np.random.uniform(-5, 5, 284)  # Increased to 20 superconductors
turing_machine = np.zeros((10, 10))  # Placeholder for Turing Machine grid

# Generate hypothetical fourth-dimensional data
fourth_dimension_data = np.random.rand(len(quantum_superconductors_x))

# Create the plot
plt.figure(figsize=(10, 6))

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
    plt.plot(perturbed_qx, perturbed_qy, color=plt.cm.viridis(fourth_dim), linewidth=2, alpha=0.5)
    # Add singularity wormhole at the middle of each connecting line segment
    for i in range(len(perturbed_qx) - 1):
        mid_x = (perturbed_qx[i] + perturbed_qx[i + 1]) / 2
        mid_y = (perturbed_qy[i] + perturbed_qy[i + 1]) / 2
        plt.plot(mid_x, mid_y, 'ro', markersize=1)  # Singularity wormhole point

# Plot the quantum superconductors
plt.scatter(quantum_superconductors_x, quantum_superconductors_y, c=fourth_dimension_data, cmap='viridis', label='Quantum Superconductors')

# Plot the Turing Machine grid
plt.imshow(turing_machine, cmap='gray', extent=(-5, 5, -5, 5), alpha=0.5)

# Add arrows indicating the flow of influence
plt.annotate("Kerr Gravity", xy=(-4, 4), xytext=(-4, 4.5),
             arrowprops=dict(facecolor='blue', arrowstyle='->'))
plt.annotate("Quantum Mechanics", xy=(0, 4), xytext=(0, 4.5),
             arrowprops=dict(facecolor='green', arrowstyle='->'))
plt.annotate("Turing Machine", xy=(4, 4), xytext=(4, 4.5),
             arrowprops=dict(facecolor='gray', arrowstyle='->'))

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Interaction of Quantum Superconductors through Perturbed Hawking Channels')

# Set equal aspect ratio
plt.gca().set_aspect('equal', adjustable='box')

# Show legend
plt.colorbar(label='Fourth Dimension')

# Show the plot
plt.show()
