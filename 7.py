import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

# Define constants
h_bar = 1  # Planck's constant (arbitrary units)
L = 10     # Length scale (arbitrary units)
N = 1000   # Number of points

# Define potential function (more complex)
def potential(x):
    return 10 * np.sin(x) + 5 * np.cos(x) + 15 * np.sin(2 * x) + 7  # Example complex potential

# Define the grid
x_values = np.linspace(0, L, N)
dx = x_values[1] - x_values[0]

# Initialize arrays
V = np.array([potential(x) for x in x_values])

# Solve Schrödinger equation using finite difference method
# (simplified for demonstration)
psi = np.zeros_like(x_values)
psi[0] = 1   # Initial condition: psi(0) = 1 (incident wave)

for i in range(1, N-1):
    psi[i+1] = ((2 - 5*dx**2 * V[i]/h_bar**2) * psi[i] - (1 + dx**2 * V[i-1]/h_bar**2) * psi[i-1]) / (1 + dx**2 * V[i+1]/h_bar**2)

# Calculate the CTC curve (potential for creating a closed timelike curve)
k = 2  # Arbitrary value for k
CTC = k * x_values**2

# Calculate the light curve (energy of photons)
frequency = np.sqrt(V)  # Arbitrary scaling for frequency
light_curve = h_bar * frequency

# Define function to integrate using trapezoidal rule
def integrate_curve(x, y):
    return simps(y, x)

# Calculate the area under each curve
area_wavefunction = integrate_curve(x_values, psi)
area_potential = integrate_curve(x_values, V)
area_ctc = integrate_curve(x_values, CTC)
area_light_curve = integrate_curve(x_values, light_curve)

# Create 5-cross plot
fig, axs = plt.subplots(5, 5, figsize=(15, 15))

# Plot wavefunction
axs[0, 2].plot(x_values, psi, label='Wavefunction')
axs[0, 2].set_xlabel('Position')
axs[0, 2].set_ylabel('Wavefunction')
axs[0, 2].set_title('Wavefunction of the Particle\nArea: {:.2f}'.format(area_wavefunction))

# Plot potential barrier
axs[1, 2].plot(x_values, V, 'r--', label='Potential Barrier')
axs[1, 2].set_xlabel('Position')
axs[1, 2].set_ylabel('Potential')
axs[1, 2].set_title('Potential Barrier\nArea: {:.2f}'.format(area_potential))

# Plot CTC curve
axs[2, 2].plot(x_values, CTC, 'g--', label='CTC Curve')
axs[2, 2].set_xlabel('Position')
axs[2, 2].set_ylabel('CTC Potential')
axs[2, 2].set_title('Closed Timelike Curve Potential\nArea: {:.2f}'.format(area_ctc))

# Plot light curve
axs[3, 2].plot(x_values, light_curve, 'm--', label='Light Curve')
axs[3, 2].set_xlabel('Position')
axs[3, 2].set_ylabel('Energy')
axs[3, 2].set_title('Light Curve\nArea: {:.2f}'.format(area_light_curve))

# Hide axes for the other subplots
for i in range(5):
    for j in range(5):
        if (i != 0 or j != 2) and (i != 1 or j != 2) and (i != 2 or j != 2) and (i != 3 or j != 2):
            axs[i, j].axis('off')

plt.tight_layout()
plt.show()
