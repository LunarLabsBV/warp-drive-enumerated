import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate example data in 12 dimensions
np.random.seed(0)
data_12d = np.random.randn(100, 12)  # Example 100 data points in 12 dimensions

# Convert to DataFrame for easy visualization
import pandas as pd
df = pd.DataFrame(data_12d, columns=[f'Dimension_{i}' for i in range(1, 13)])

# Create a scatter plot matrix
sns.pairplot(df)
plt.show()
