'''
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt(r"C:\Althamish\Project\Posture Monitor\Git_PostureMonitor\PostureMonitor_V.2.0\v.2.0\data\v.3.0 ML-Ready-with_rows\pil")  # Read from a text file

# Get the number of columns
num_columns = data.shape[1]
# %%
# Create separate subplots for each column

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
'''
#%% ############################################ specific columns analysis ################################

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

file_path = (
    Path(r"C:\Althamish\Project\Posture Monitor\Git_PostureMonitor")
    / "PostureMonitor_V.2.0"
    / "v.2.0"
    / "data"
    / "v.3.0 ML-Ready-with_rows"
    / "calcheck"
)

# Load dataset with headers
#file_path = r"C:\Althamish\Project\Posture Monitor\Git_PostureMonitor\PostureMonitor_V.2.0\v.2.0\data\v.3.0 ML-Ready-with_rows\pil"
df = pd.read_csv(file_path)

# Define specific columns to plot
columns_to_plot = ["xgb_S1", "ybg_S1", "zbg_S2", "tilt_z_S1", "tilt_accy_S1", "tilt_z_S2"]

# Define the number of rows to plot
N = 6000  # Change this to the number of rows you want

# Filter DataFrame to include only selected columns and first N rows
#df_filtered = df[columns_to_plot].head(N)

#columns_to_plot = df.columns.tolist()
#df_filtered = df.head(N)

# Get number of selected columns
num_columns = len(columns_to_plot)

# Create subplots
fig, axes = plt.subplots(num_columns, 1, figsize=(8, num_columns * 2), sharex=True)

# Plot each selected column
for i, col in enumerate(columns_to_plot):
    axes[i].plot(df_filtered[col], label=col, color="b")
    axes[i].set_ylabel(col)
    axes[i].legend()
    axes[i].grid()

# X-axis label for the last subplot
axes[-1].set_xlabel("Sample Index")  

plt.suptitle(f"Plots for Selected Data Columns (First {N} Rows)")
plt.tight_layout()
plt.show()

#%% ############################################ all columns analysis ################################
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Define the file path
file_path = (
    Path(r"C:\Althamish\Project\Posture Monitor\Git_PostureMonitor")
    / "PostureMonitor_V.2.0"
    / "v.2.0"
    / "data"
    / "v.3.0 ML-Ready-with_rows"
    / "calcheck"
)

# Load dataset
df = pd.read_csv(file_path)

# Convert all columns to numeric (force conversion, replace errors with NaN)
df = df.apply(pd.to_numeric, errors='coerce')

# Define the number of rows to plot
N = 6000  # Change this as needed

# Select only the first N rows and all columns
df_filtered = df.head(N)

# Get all column names
columns_to_plot = df_filtered.columns.tolist()
num_columns = len(columns_to_plot)

# Create subplots
fig, axes = plt.subplots(num_columns, 1, figsize=(8, num_columns * 2), sharex=True)

# Plot each column
for i, col in enumerate(columns_to_plot):
    axes[i].plot(df_filtered[col], label=col, color="b")
    axes[i].set_ylabel(col)
    axes[i].legend()
    axes[i].grid()

# X-axis label for the last subplot
axes[-1].set_xlabel("Sample Index")  

plt.suptitle(f"Plots for All Numeric Columns (First {N} Rows)")
plt.tight_layout()
plt.show()
