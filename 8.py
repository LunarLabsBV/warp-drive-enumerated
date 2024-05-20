import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Values from the initial energy tensor matrix
initial_values = [10, 0, 0, 0, 9.8, 9.6, 9.4, 1, 0.5, -1, -2, -3, -4]

# Corresponding values from the constructed energy-momentum tensor matrix
constructed_values = [10, 0, 0, 0, 1, 0.1, 0.1, 2, 0.2, 0, 0.1, 0.3, 3]

# Initialize the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Create an empty scatter plot
sc = ax.scatter([], [], color='blue')

# Set plot properties
ax.set_title('Correlation between Initial and Constructed Energy-Momentum Tensor Elements')
ax.set_xlabel('Initial Matrix Elements')
ax.set_ylabel('Constructed Matrix Elements')
ax.grid(True)

# Function to update the plot
def update(frame):
    # Update the scatter plot with the same data
    sc.set_offsets([[x, y] for x, y in zip(initial_values, constructed_values)])
    # Change the color of the scatter plot points in each frame
    sc.set_color(['blue' if i % (frame + 1) == 0 else 'red' for i in range(len(initial_values))])
    return sc,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(initial_values), interval=500)

# Show the animation
plt.show()
