'''
this file is used to take an input raw text document dataset from the 
Data_Acq_Protocol.py file and convert into a ML ready csv file
i.e. to produce structured and labelled dataset with columns from 'xa_S1' to 'tilt_accy_S2'&'State_S2' from raw text output of  pico
'''
#%% ################################## Importing data and Init #####################
import pandas as pd
from pathlib import Path

# Load your data
file_name = input("Enter the FileName: ")

file_path = (
    Path(r"C:\Althamish\Project")
    / "PostureMonitor_V.2.0"
    / "v.2.0"
    / "data"
    / "V.3.0"
    / "v.3.0 Raw"
    / str(file_name + ".txt")
)
# Load dataset with space as the separator, handling multiple spaces
df = pd.read_csv(file_path, delimiter=r"\s+", engine="python")

no_of_sensors = 2


#%% ############################### Combining rows & assign col.names #################################
# Initialize an empty list to store the combined rows
combined_rows = []
no_of_rows_to_combine = no_of_sensors

# Loop through the DataFrame in steps of 3 to get rows with ID 1, 2, 3
for row_no in range(0, len(df), no_of_rows_to_combine):
    # Check if there are at least three rows remaining to combine
    if row_no + (no_of_sensors - 1) < len(df):
        row1 = df.iloc[row_no]       # Row with Val13 == 1
        row2 = df.iloc[row_no + 1]   # Row with Val13 == 2
        
        # Combine the values of row1, row2, and row3 into a single row
        combined_row = pd.concat([row1, row2]).reset_index(drop=True)
        combined_rows.append(combined_row)

# Convert combined rows list into a DataFrame
combined_df = pd.DataFrame(combined_rows)

### Creating Column names for each sensor
df_column_names = [
    "time_stamp", "xa", "ya", "za", "xgb", "ybg", "zbg",
    "tilt_x", "tilt_y", "tilt_z", "tilt_accx", "tilt_accy",
    "tilt_accz","dt", "sensor_no","State"
]
# Generate column names for two sensors
new_column_names = []
for sensor_id in range(1, no_of_sensors + 1):  # For sensor 1 and 2
    sensor_specific_columns = [f"{col}_S{sensor_id}" for col in df_column_names]
    new_column_names.extend(sensor_specific_columns)

print(new_column_names)

combined_df.columns = new_column_names  # Assign new column names

#%% ##################################### Remove NaN values ########################
# Find and remove rows with NaN values
combined_df = combined_df.dropna()

#%% ############################################# Removing rows and columns
# Removing Irrelevant columns
cleaned_df = combined_df


# Define column names to drop
columns_to_drop = [
    'time_stamp_S1', 'dt_S1', "State_S1",
    'sensor_no_S1', 'time_stamp_S2', 'dt_S2', 'sensor_no_S2'
]
# Drop the specified columns
cleaned_df.drop(columns=columns_to_drop, axis=1, inplace=True)
cleaned_with_rows_df = cleaned_df

# Removing irrelevant rows
cleaned_df = cleaned_df.loc[~cleaned_df["State_S2"].isin(["Unknown", "Transition"])]

# Rename the last column
cleaned_df.rename(columns={cleaned_df.columns[-1]: "Posture"}, inplace=True)
# Remove the last 4 characters from the "Posture" column - "ss.png" -> "ss"
cleaned_df["Posture"] = cleaned_df["Posture"].astype(str).str.slice(stop=-4)

# Print first few rows to check
print(cleaned_df.head())


#%%##############################################creating the ML ready CSV file#############################################

###Pure ML
ML_path = r"C:\Althamish\Project\PostureMonitor_V.2.0\v.2.0\data\V.3.0\v.3.0 ML-Ready\\" + file_name +".csv"
finaldf=cleaned_df    
finaldf.to_csv(ML_path, index=False)
print(f"Combined rows saved to {ML_path}")

###Irrelevant Rows ML
ML_path = r"C:\Althamish\Project\PostureMonitor_V.2.0\v.2.0\data\V.3.0\v.3.0 ML-Ready-with_rows\\" + file_name +".csv"
finaldf=cleaned_with_rows_df    
finaldf.to_csv(ML_path, index=False)
print(f"Combined rows saved to {ML_path}")

#%%############################################## 5 Label File Creation and Saving #############################################

import pandas as pd

# Assume 'df' is the main DataFrame containing your data.
# Also assume that the label column is named 'final'. Adjust if needed.
df = cleaned_df.copy()  # or use the appropriate DataFrame variable
label_col = 'Posture'

# --- Process for S_UP & UP -> H_UP ---
# Filter the rows for each class from the original DataFrame
s_up_df = df[df[label_col] == 'S_UP']
up_df = df[df[label_col] == 'UP']

# Sample 50% of each class (use a fixed random_state for reproducibility)
sample_s_up = s_up_df.sample(frac=0.5, random_state=42)
sample_up = up_df.sample(frac=0.5, random_state=42)

# Combine the samples and set the new label
merged_up = pd.concat([sample_s_up, sample_up], ignore_index=True)
merged_up[label_col] = 'H_UP'

# --- Process for S_LF & LF -> H_LF ---
# Filter the rows for each class from the original DataFrame
s_lf_df = df[df[label_col] == 'S_LF']
lf_df = df[df[label_col] == 'LF']

# Sample 50% of each class
sample_s_lf = s_lf_df.sample(frac=0.5, random_state=42)
sample_lf = lf_df.sample(frac=0.5, random_state=42)

# Combine the samples and set the new label
merged_lf = pd.concat([sample_s_lf, sample_lf], ignore_index=True)
merged_lf[label_col] = 'H_LF'

# --- Remove the original rows for the merged classes ---
# Keep only rows that are not in the original S_UP, UP, S_LF, LF groups
df_remaining = df[~df[label_col].isin(['S_UP', 'UP', 'S_LF', 'LF'])]

# --- Append the new merged rows ---
df_5Lab = pd.concat([df_remaining, merged_up, merged_lf], ignore_index=True)

# Optional: Check the unique classes to verify that there are only 5
print("Unique classes in final DataFrame:", df_5Lab[label_col].unique())

# Save the final DataFrame to CSV (adjust the path as needed)
output_path = r"C:\Althamish\Project\PostureMonitor_V.2.0\v.2.0\data\V.3.0\v.3.0 ML-Ready 5Lab\\" + file_name +".csv"
df_5Lab.to_csv(output_path, index=False)
print(f"Final DataFrame with 5 classes saved to {output_path}")


'''{
    
    do not forget to run the normalization code for additional data, after all 
    patient's data has been collected
    
    }'''
