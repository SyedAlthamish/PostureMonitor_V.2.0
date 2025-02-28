
'''{
    File Description:
     This file contains cells that each perform specific functions that help analyse & fix performance of Posture_data after a sample acquisition
     from a patient particularly fix of Nan values and observance Posture Change for each Posture
    }'''
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

# %%################################ Color graded posture-change recognition of Data #############################
''' Takes 1.5 minutes to plot, insane'''
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns

# Define the file path
file_path = (
    Path(r"C:\Althamish\Project")
    / "PostureMonitor_V.2.0"
    / "v.2.0"
    / "data"
    / "V.3.0"
    / "v.3.0 ML-Ready-with_rows"
    / "syd2"
)

# Load dataset
df = pd.read_csv(file_path)

# Convert all columns except the last one to numeric
df.iloc[:, :-1] = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')

# Select first N rows
N = 8000
df_filtered = df.head(N)

# Get column names
columns_to_plot = df_filtered.columns[:-1]  # Exclude last column from numerical plots
final_col = df_filtered.columns[-1]  # Categorical column

# Fill NaN values in the final column with a placeholder (e.g., "Unknown")
df_filtered[final_col] = df_filtered[final_col].fillna("Unknown")

# Convert final column to categorical
df_filtered[final_col] = df_filtered[final_col].astype("category")
categories = df_filtered[final_col].cat.categories  # Unique values

# Assign a color to each category
color_map = dict(zip(categories, sns.color_palette("hsv", len(categories))))

# Create subplots
num_columns = len(columns_to_plot)
fig, axes = plt.subplots(num_columns, 1, figsize=(8, num_columns * 2), sharex=True)

# Plot each numeric column
for i, col in enumerate(columns_to_plot):
    axes[i].plot(df_filtered[col], label=col, color="b")
    axes[i].set_ylabel(col)
    axes[i].legend()
    axes[i].grid()

    # Highlight different regions based on final column's values
    prev_category = df_filtered.iloc[0][final_col]  # Initialize with first value
    start_index = 0

    for idx in range(1, len(df_filtered)):
        current_category = df_filtered.iloc[idx][final_col]

        # If the category changes or it's the last index, draw a highlight
        if current_category != prev_category or idx == len(df_filtered) - 1:
            end_index = idx
            color = color_map.get(prev_category, "gray")  # Default color for unknown
            axes[i].axvspan(start_index, end_index, color=color, alpha=0.2)
            prev_category = current_category
            start_index = idx

# Label x-axis
axes[-1].set_xlabel("Sample Index")

# Set title and adjust layout
plt.suptitle(f"Plots with Highlighted Regions (First {N} Rows)")
plt.tight_layout()
plt.show()

#%% ####################################### Displaying Nan Rows ############################################
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns

# Define the file path
file_path = (
    Path(r"C:\Althamish\Project")
    / "PostureMonitor_V.2.0"
    / "v.2.0"
    / "data"
    / "V.3.0"
    / "v.3.0 ML-Ready-with_rows"
    / "syd"
)

df = pd.read_csv(file_path)

# Find rows with NaN values
nan_rows = df[df.isna().any(axis=1)]  # Select rows where any column has NaN

# Print rows with NaN values
print("Rows with NaN values:")
print(nan_rows)


#Find rows where the second column has a value of -0.325162
second_col_name = df.columns[1]  # Get the name of the second column
filtered_rows = df[df[second_col_name] == -0.325162]

# Print the matching rows
print("Rows where the second column equals -0.325162:")
print(filtered_rows)

# %%#################################### Raw - Displaying NAN rows and corresponding row display################################
import pandas as pd
from pathlib import Path

# Define the file path
file_path = (
    Path(r"C:\Althamish\Project")
    / "PostureMonitor_V.2.0"
    / "v.2.0"
    / "data"
    / "V.3.0"
    / "v.3.0 Raw"
    / "syd.txt"
)

# Load dataset with space as the separator, handling multiple spaces
df = pd.read_csv(file_path, delimiter=r"\s+", engine="python")  

# Find rows with NaN values
nan_rows = df[df.isna().any(axis=1)]  

# Print rows with NaN values
print("Rows with NaN values:")
print(nan_rows)

# %% ################################# Removing Nan from raw .txt, save it to same repo, create og's copy ##########################