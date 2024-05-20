import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# Define system parameters
K = 1
zeta = 0.5
omega_n = 1

# Define transfer function numerator and denominator coefficients
numerator = [K]
denominator = [1, 2*zeta*omega_n, omega_n**2]

# Define transfer function
system = signal.TransferFunction(numerator, denominator)

# Convert transfer function to state-space representation
A, B, C, D = signal.tf2ss(numerator, denominator)

# Compute eigenvalues of the state-space representation
eigenvalues = np.linalg.eigvals(A)

# Check the stability using eigenvalues
stable = np.all(np.real(eigenvalues) < 0)

# Plot Routh-Hurwitz array
plt.figure(figsize=(10, 6))
plt.plot(np.real(eigenvalues), np.imag(eigenvalues), 'bo', markersize=10)
plt.title('Eigenvalues of the State-Space Representation')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.axhline(0, color='k', linestyle='--')
plt.grid(True)
plt.show()

# Plot Bode plot
plt.figure(figsize=(10, 6))
_ = system.bode()
plt.title('Bode Plot')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Magnitude (dB), Phase (deg)')
plt.grid(True)
plt.show()

# Output stability information
if stable:
    print("The system is stable.")
else:
    print("The system is unstable.")
