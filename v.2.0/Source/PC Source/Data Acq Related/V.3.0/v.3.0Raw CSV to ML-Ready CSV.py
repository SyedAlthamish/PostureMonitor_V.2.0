'''
this file is used to take an input raw text document dataset from the 
Data_Acq_Protocol.py file and convert into a ML ready csv file
i.e. to produce structured and labelled dataset with columns from 'xa_S1' to 'tilt_accy_S2'&'State_S2' from raw text output of  pico
'''

import pandas as pd

# Load your data
file_name = input("Enter the FileName: ")
file_path = r"C:\Althamish\Project\Posture Monitor\Git_PostureMonitor\PostureMonitor_V.2.0\v.2.0\data\v.3.0 Raw\\" + file_name +".txt"
df = pd.read_csv(file_path, header=None, delim_whitespace=True)
no_of_sensors = 2

#---------------------------------Combining rows and dropping unnecessary columns--------------------------------------


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


#----------------------------------------creating the ML ready CSV file------------------------------------------------

ML_path = r"C:\Althamish\Project\Posture Monitor\Git_PostureMonitor\PostureMonitor_V.2.0\v.2.0\data\v.3.0 ML-Ready\\" + file_name
finaldf=cleaned_df    
finaldf.to_csv(ML_path, index=False)
print(f"Combined rows saved to {ML_path}")

ML_path = r"C:\Althamish\Project\Posture Monitor\Git_PostureMonitor\PostureMonitor_V.2.0\v.2.0\data\v.3.0 ML-Ready-with_rows\\" + file_name
finaldf=cleaned_with_rows_df    
finaldf.to_csv(ML_path, index=False)
print(f"Combined rows saved to {ML_path}")