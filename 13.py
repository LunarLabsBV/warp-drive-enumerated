import matplotlib.pyplot as plt
import numpy as np

# Define function to compute trajectory based on Gott time machine metric equation
def compute_trajectory(num_points, time_step, is_cap=False, is_fuse=False, is_stem=False, is_exhaust=False, is_dumet=False, is_lead=False, is_support=False, is_filament=False, is_gas=False):
    # Define parameters for trajectory generation
    r_max = 50  # Maximum radial distance
    theta_max = np.pi  # Maximum theta angle
    phi_max = 2 * np.pi  # Maximum phi angle
    
    if is_cap:
        r_max *= 0.5  # Reduce radius for the cap trajectories

    # Generate random values for theta and phi
    theta_values = np.linspace(0, theta_max, num_points)
    phi_values = np.linspace(0, phi_max, num_points)

    # Straight line stem trajectory
    stem_length = 30
    # Compute trajectory coordinates based on Gott time machine metric equation
    if is_fuse:
        # Perturb the radius to create a curved shape resembling a fuse
        r_perturbed = r_max * (1 + np.sin(theta_values + time_step))
        trajectory_x = r_perturbed * np.sin(theta_values) * np.cos(phi_values)
        trajectory_y = r_perturbed * np.sin(theta_values) * np.sin(phi_values)
        trajectory_z = r_perturbed * np.cos(theta_values)
    elif is_stem:
        trajectory_x = np.zeros(num_points)
        trajectory_y = np.zeros(num_points)
        trajectory_z = np.linspace(-r_max, -r_max - stem_length, num_points)
    elif is_exhaust:
        # Curved exhaust tube trajectory
        exhaust_radius = 10
        exhaust_length = 20
        trajectory_x = exhaust_radius * np.cos(phi_values)
        trajectory_y = exhaust_radius * np.sin(phi_values)
        trajectory_z = -r_max - stem_length - exhaust_length + (exhaust_length * np.sin(theta_values))
    elif is_dumet:
        # Dumet wire trajectory
        wire_length = 50
        trajectory_x = np.zeros(num_points)
        trajectory_y = np.zeros(num_points)
        trajectory_z = np.linspace(r_max, -r_max, num_points)
    elif is_lead:
        # Lead wire trajectory
        lead_length = 30
        trajectory_x = r_max * np.sin(theta_max / 4) * np.cos(phi_values)
        trajectory_y = r_max * np.sin(theta_max / 4) * np.sin(phi_values)
        trajectory_z = np.linspace(-r_max, -r_max - lead_length, num_points)
    elif is_support:
        # Support wire trajectory
        support_length = 30
        trajectory_x = r_max * np.sin(theta_values) * np.cos(np.pi / 4)
        trajectory_y = r_max * np.sin(theta_values) * np.sin(np.pi / 4)
        trajectory_z = np.linspace(-r_max - stem_length, -r_max, num_points)
    elif is_filament:
        # Tungsten filament trajectory (coiled inside the bulb)
        coil_radius = 20
        filament_radius = 10
        filament_length = 20
        theta_values = np.linspace(0, 2 * np.pi, num_points)
        phi_values = np.linspace(0, 2 * np.pi, num_points)
        trajectory_x = coil_radius * np.sin(theta_values)
        trajectory_y = coil_radius * np.cos(theta_values)
        trajectory_z = np.linspace(-filament_length / 2, filament_length / 2, num_points)
    elif is_gas:
        # Gas filling trajectory (random cloud of points within the bulb)
        num_gas_points = 500
        gas_radius = 25
        gas_theta = np.random.uniform(0, np.pi, num_gas_points)
        gas_phi = np.random.uniform(0, 2 * np.pi, num_gas_points)
        gas_r = np.random.uniform(0, gas_radius, num_gas_points)
        trajectory_x = gas_r * np.sin(gas_theta) * np.cos(gas_phi)
        trajectory_y = gas_r * np.sin(gas_theta) * np.sin(gas_phi)
        trajectory_z = gas_r * np.cos(gas_theta)
    else:
        trajectory_x = r_max * np.sin(theta_values + time_step) * np.cos(phi_values)
        trajectory_y = r_max * np.sin(theta_values + time_step) * np.sin(phi_values)
        trajectory_z = r_max * np.cos(theta_values + time_step)
    
    return trajectory_x, trajectory_y, trajectory_z

# Define function to generate qubits
def generate_qubits(ax, trajectory_x, trajectory_y, trajectory_z, color):
    qubit_coordinates_x = trajectory_x[::10]  # Sample every 10th point for qubits
    qubit_coordinates_y = trajectory_y[::10]
    qubit_coordinates_z = trajectory_z[::10]
    ax.scatter(qubit_coordinates_x, qubit_coordinates_y, qubit_coordinates_z, color=color, marker='o', s=50)

# Generate initial trajectory
num_points = 100
num_trajectories = 10
time_steps = np.linspace(0, 2*np.pi, num_trajectories)  # Different initial time steps for each trajectory

# Create figure and subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each trajectory and place qubits
for i, time_step in enumerate(time_steps):
    trajectory_x, trajectory_y, trajectory_z = compute_trajectory(num_points, time_step)
    ax.plot(trajectory_x, trajectory_y, trajectory_z, label=f'Trajectory {i+1}')
    generate_qubits(ax, trajectory_x, trajectory_y, trajectory_z, 'goldenrod')

# Append CTC trajectories to the bottom of the shape
for i, time_step in enumerate(time_steps):
    trajectory_x, trajectory_y, trajectory_z = compute_trajectory(num_points, time_step, is_cap=True)
    ax.plot(trajectory_x, trajectory_y, trajectory_z, color='blue', linestyle='dashed')

# Append fuse trajectory
fuse_time_steps = np.linspace(0, np.pi, num_trajectories)  # Fuse time steps
for i, time_step in enumerate(fuse_time_steps):
    trajectory_x, trajectory_y, trajectory_z = compute_trajectory(num_points, time_step, is_fuse=True)
    ax.plot(trajectory_x, trajectory_y, trajectory_z, color='red')

# Append stem trajectory
stem_time_steps = np.linspace(0, 0, num_trajectories)  # Stem time steps
for i, time_step in enumerate(stem_time_steps):
    trajectory_x, trajectory_y, trajectory_z = compute_trajectory(num_points, time_step, is_stem=True)
    ax.plot(trajectory_x, trajectory_y, trajectory_z, color='green')

# Append exhaust tube trajectory
exhaust_time_steps = np.linspace(0, np.pi, num_trajectories)  # Exhaust tube time steps
for i, time_step in enumerate(exhaust_time_steps):
    trajectory_x, trajectory_y, trajectory_z = compute_trajectory(num_points, time_step, is_exhaust=True)
    ax.plot(trajectory_x, trajectory_y, trajectory_z, color='purple')

# Append Dumet wire trajectory
dumet_time_steps = np.linspace(0, 0, num_trajectories)  # Dumet wire time steps
for i, time_step in enumerate(dumet_time_steps):
    trajectory_x, trajectory_y, trajectory_z = compute_trajectory(num_points, time_step, is_dumet=True)
    ax.plot(trajectory_x, trajectory_y, trajectory_z, color='orange')

# Append lead wire trajectories
lead_time_steps = np.linspace(0, 0, num_trajectories)  # Lead wire time steps
for i, time_step in enumerate(lead_time_steps):
    trajectory_x, trajectory_y, trajectory_z = compute_trajectory(num_points, time_step, is_lead=True)
    ax.plot(trajectory_x, trajectory_y, trajectory_z, color='cyan')

# Append support wire trajectories
support_time_steps = np.linspace(0, 0, num_trajectories)  # Support wire time steps
for i, time_step in enumerate(support_time_steps):
    trajectory_x, trajectory_y, trajectory_z = compute_trajectory(num_points, time_step, is_support=True)
    ax.plot(trajectory_x, trajectory_y, trajectory_z, color='magenta')

# Append glass bulb trajectory (spherical surface)
bulb_radius = 50
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = bulb_radius * np.outer(np.cos(u), np.sin(v))
y = bulb_radius * np.outer(np.sin(u), np.sin(v))
z = bulb_radius * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)

# Append light source (cloud of points)
light_source_radius = 40
num_light_points = 500
light_theta = np.random.uniform(0, np.pi, num_light_points)
light_phi = np.random.uniform(0, 2 * np.pi, num_light_points)
light_r = np.random.uniform(0, light_source_radius, num_light_points)
light_x = light_r * np.sin(light_theta) * np.cos(light_phi)
light_y = light_r * np.sin(light_theta) * np.sin(light_phi)
light_z = light_r * np.cos(light_theta)
ax.scatter(light_x, light_y, light_z, color='yellow', alpha=0.5)

# Add labels and legend
ax.set_xlabel('X (Distance)')
ax.set_ylabel('Y (Distance)')
ax.set_zlabel('Z (Distance)')
ax.set_title('Light Bulb Model (On)')
ax.legend()

plt.show()





