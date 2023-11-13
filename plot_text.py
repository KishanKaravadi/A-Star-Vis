import numpy as np
import matplotlib.pyplot as plt

# Your dictionary
data = {0.005: 15, 0.01: 43, 0.015: 54}

# Extract keys and values
alphas, total_actions = zip(*sorted(data.items()))

# Convert to NumPy arrays
alphas = np.array(alphas)
total_actions = np.array(total_actions)

# Create the plot
plt.scatter(alphas, total_actions, marker='o', linestyle='-', color='b')
plt.title('Alpha vs Total Actions')
plt.xlabel('Alpha')
plt.ylabel('Total Actions')
plt.grid(True)
plt.savefig('scatter_plot.png')
plt.show()
