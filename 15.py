import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Define historical epochs and their significance (y-values)
epochs = [
    "Prehistoric",
    "Ancient",
    "Medieval",
    "Exploration",
    "Enlightenment",
    "Industrial Revolution",
    "World Wars",
    "Space Age",
    "Globalization",
    "Future"
]
significance = [5, 8, 6, 7, 8, 9, 9, 8, 7, 6]  # Arbitrary significance values

# Define time points for the z-axis
time_points = np.arange(len(epochs))

# Create a meshgrid
X, Y = np.meshgrid(np.arange(len(epochs)), significance)
Z = np.ones_like(X) * time_points[:, np.newaxis]

# Create 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the mesh
ax.plot_surface(X, Y, Z, alpha=0.7)

# Label axes and set title
ax.set_xlabel('Time')
ax.set_ylabel('Significance')
ax.set_zlabel('Historical Epochs')
ax.set_title('3D Timescape Mesh')

# Set ticks and labels
ax.set_xticks(np.arange(len(epochs)))
ax.set_xticklabels(epochs, rotation=45)

# Add straight vertical transport lines for quarks
for i in range(len(epochs)):
    ax.plot([i, i+100000000000000], [0, max(significance)], [0, len(epochs)], color='goldenrod', linestyle='-', alpha=1.0)

# Add the speed of light line
c = 1  # Speed of light
ax.plot([0, len(epochs)], [0, max(significance)], [0, c * len(epochs)], color='blue', linestyle='-', alpha=0.005)

# Display the plot
plt.tight_layout()
plt.show()
