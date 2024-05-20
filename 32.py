import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def draw_gear(x, y, num_teeth, radius, tangent_factor, ax):
    theta = np.linspace(0, 2*np.pi, num_teeth*2, endpoint=True)
    for i in range(num_teeth):
        angle = theta[i*2]
        perturbation_angle = tangent_factor * (angle - np.pi)  # Apply perturbation
        ax.plot([x + np.cos(angle)*radius, x + np.cos(angle+perturbation_angle)*radius], 
                [y + np.sin(angle)*radius, y + np.sin(angle+perturbation_angle)*radius], color='black')

def draw_q_thruster(ax, tangent_factor):
    # Draw Equation 1 Gear
    draw_gear(0, 0, 10, 1, tangent_factor, ax)
    ax.text(0, 0, r'$\rho$', fontsize=12, ha='center', va='center')

    # Draw Equation 2 Gear
    draw_gear(4, 0, 10, 1, tangent_factor, ax)
    ax.text(4, 0, r'$\rho$', fontsize=12, ha='center', va='center')

    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

def update(frame):
    ax.clear()
    tangent_factor = frame / 100  # Adjust the range of tangent_factor as needed
    draw_q_thruster(ax, tangent_factor)
    ax.set_title(f'Q-Thruster Gear System with Perturbation (Tangent Factor: {tangent_factor})')

fig, ax = plt.subplots(figsize=(8, 4))
ani = FuncAnimation(fig, update, frames=range(100), interval=100)
plt.title('Q-Thruster Gear System with Perturbation')
plt.show()
