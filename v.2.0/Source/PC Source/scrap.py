########################  Scrap file ###################################################

import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt(r"C:\Althamish\Project\Posture Monitor\Git_PostureMonitor\PostureMonitor_V.2.0\v.2.0\data\v.3.0 ML-Ready-with_rows\krill")  # Read from a text file

# Get the number of columns
num_columns = data.shape[1]
# %%
# Create separate subplots for each column
'''
fig, axes = plt.subplots(num_columns, 1, figsize=(8, num_columns * 2), sharex=True)

for i in range(num_columns):
    axes[i].plot(data[:, i], label=f"Column {i+1}", color="b")
    axes[i].set_ylabel(f"Column {i+1}")
    axes[i].legend()
    axes[i].grid()


axes[-1].set_xlabel("Sample Index")  # Set x-axis label for the last subplot
plt.suptitle("Separate Plots for Each Data Column")
plt.tight_layout()
plt.show()
'''


column_pairs = [(i, i + 6) for i in range(6)]  # [(0,6), (1,7), (2,8), ..., (5,11)]

fig, axes = plt.subplots(len(column_pairs), 1, figsize=(8, len(column_pairs) * 3), sharex=True)

for i, (col1, col2) in enumerate(column_pairs):
    axes[i].plot(data[:, col1], label=f"Column {col1+1}", color="b")
    axes[i].plot(data[:, col2], label=f"Column {col2+1}", color="r")
    axes[i].set_ylabel(f"Columns {col1+1} & {col2+1}")
    axes[i].legend()
    axes[i].grid()

axes[-1].set_xlabel("Sample Index")  # Set x-axis label for the last subplot
plt.suptitle("Paired Plots for Selected Data Columns")
plt.tight_layout()
plt.show()