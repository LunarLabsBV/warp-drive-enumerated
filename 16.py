import matplotlib.pyplot as plt

# Values from the initial energy tensor matrix
initial_values = [10, 0, 0, 0, 9.8, 9.6, 9.4, 1, 0.5, -1, -2, -3, -4]

# Corresponding values from the constructed energy-momentum tensor matrix
constructed_values = [10, 0, 0, 0, 1, 0.1, 0.1, 2, 0.2, 0, 0.1, 0.3, 3]

# Plotting the values
plt.figure(figsize=(8, 6))
plt.scatter(initial_values, constructed_values, color='blue')
plt.plot(initial_values, initial_values, color='red', linestyle='--')  # Identity line
plt.title('Correlation between Initial and Constructed Energy-Momentum Tensor Elements')
plt.xlabel('Initial Matrix Elements')
plt.ylabel('Constructed Matrix Elements')
plt.grid(True)
plt.show()
