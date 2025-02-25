## the bottom file is now used for analysis of actually acquired posture data's validity - whether if routine takes in 
## actual data or not.

'''{
    File Description:
     This file contains cells that each perform specific functions that help analyse performance of Posture_data after the GYRO spike fix, 
     and other fixes including tilt drifts, and more, they also checked calibration data
    }'''

#%% ###################################### Comparative Plots: Fixed vs Unfixed #######################################

import matplotlib.pyplot as plt
import numpy as np

# Load numerical data from a text file
# Ensure the file contains numeric values and follows a structured format
# Update the file path as needed
data = np.loadtxt(r"C:\Althamish\Project\Posture Monitor\Git_PostureMonitor\PostureMonitor_V.2.0\v.2.0\data\v.3.0 ML-Ready-with_rows\pil")

# Get the number of columns in the dataset
num_columns = data.shape[1]

# Define pairs of columns to compare (e.g., fixed vs unfixed values)
column_pairs = [(i, i + 6) for i in range(6)]  # Pairs: (0,6), (1,7), (2,8), ..., (5,11)

# Create subplots for each column pair
fig, axes = plt.subplots(len(column_pairs), 1, figsize=(8, len(column_pairs) * 3), sharex=True)

# Iterate through each column pair and plot their values
for i, (col1, col2) in enumerate(column_pairs):
    axes[i].plot(data[:, col1], label=f"Column {col1+1}", color="b")  # Plot first column in blue
    axes[i].plot(data[:, col2], label=f"Column {col2+1}", color="r")  # Plot second column in red
    axes[i].set_ylabel(f"Columns {col1+1} & {col2+1}")  # Label y-axis
    axes[i].legend()  # Display legend
    axes[i].grid()  # Enable grid for readability

# Label x-axis only for the last subplot
axes[-1].set_xlabel("Sample Index")  

# Set overall title and adjust layout
plt.suptitle("Paired Plots for Selected Data Columns")
plt.tight_layout()
plt.show()

#%% ############################################ Name-Specific Columns Analysis ################################

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Define the file path using pathlib for better cross-platform compatibility
file_path = (
    Path(r"C:\Althamish\Project\Posture Monitor\Git_PostureMonitor")
    / "PostureMonitor_V.2.0"
    / "v.2.0"
    / "data"
    / "v.3.0 ML-Ready-with_rows"
    / "calcheck"
)

# Load dataset with headers
# Ensure the CSV file contains valid data
# Update the file path if necessary
df = pd.read_csv(file_path)

# Define specific columns to plot based on their names
columns_to_plot = ["xgb_S1", "ybg_S1", "zbg_S2", "tilt_z_S1", "tilt_accy_S1", "tilt_z_S2"]

# Define the number of rows to plot (adjust based on data size)
N = 6000  # Select first 6000 rows

# Get number of selected columns
num_columns = len(columns_to_plot)

# Create subplots for each selected column
fig, axes = plt.subplots(num_columns, 1, figsize=(8, num_columns * 2), sharex=True)

# Plot each selected column individually
for i, col in enumerate(columns_to_plot):
    axes[i].plot(df[col].head(N), label=col, color="b")  # Plot first N rows
    axes[i].set_ylabel(col)  # Label y-axis with column name
    axes[i].legend()  # Display legend
    axes[i].grid()  # Enable grid for readability

# Label x-axis for the last subplot
axes[-1].set_xlabel("Sample Index")  

# Set overall title and adjust layout
plt.suptitle(f"Plots for Selected Data Columns (First {N} Rows)")
plt.tight_layout()
plt.show()

#%% ############################################ All Columns Analysis ################################

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Define the file path using pathlib for flexibility
file_path = (
    Path(r"C:\Althamish\Project")
    / "PostureMonitor_V.2.0"
    / "v.2.0"
    / "data"
    / "V.3.0"
    / "v.3.0 ML-Ready-with_rows"
    / "abd"
)

# Load dataset with headers
df = pd.read_csv(file_path)

# Convert the last column(postures) to categorical codes
df.iloc[:, -1] = pd.Categorical(df.iloc[:, -1]).codes

# Convert all columns to numeric format (coerce errors to NaN to avoid issues)
df = df.apply(pd.to_numeric, errors='coerce')

# Define number of rows to plot (adjust based on data size)
N = 8000  # Select first 6000 rows

# Select first N rows for all columns
df_filtered = df.head(N)

# Get a list of all column names
columns_to_plot = df_filtered.columns.tolist()
num_columns = len(columns_to_plot)

# Create subplots for each column
fig, axes = plt.subplots(num_columns, 1, figsize=(8, num_columns * 2), sharex=True)

# Plot each column individually
for i, col in enumerate(columns_to_plot):
    axes[i].plot(df_filtered[col], label=col, color="b")  # Plot data
    axes[i].set_ylabel(col)  # Label y-axis with column name
    axes[i].legend()  # Display legend
    axes[i].grid()  # Enable grid for readability

# Label x-axis for the last subplot
axes[-1].set_xlabel("Sample Index")  

# Set overall title and adjust layout
plt.suptitle(f"Plots for All Numeric Columns (First {N} Rows)")
plt.tight_layout()
plt.show()

# %%
