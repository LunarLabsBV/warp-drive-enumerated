import matplotlib.pyplot as plt
import yaml

# YAML data
yaml_data = """
- time: 0
  value: 0
- time: 1
  value: 1
- time: 2
  value: 4
- time: 3
  value: 9
- time: 4
  value: 16
- time: 5
  value: 25
- time: 6
  value: 36
"""

# Load YAML data
data = yaml.load(yaml_data, Loader=yaml.FullLoader)

# Extract time and value
times = [entry['time'] for entry in data]
values = [entry['value'] for entry in data]

# Create the plot
plt.plot(times, values, color='blue')  # Blue line for the curve

# Apply Wall Clock Relativity Color Index
for i in range(len(times)):
    relativity_color = i / len(times)  # Adjusting color index based on time
    plt.scatter(times[i], values[i], c=[[relativity_color]], cmap='coolwarm')

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Wall Clock Relativity Color Index')

# Display the plot
plt.colorbar(label='Relativity Color Index')
plt.grid(True)
plt.show()
