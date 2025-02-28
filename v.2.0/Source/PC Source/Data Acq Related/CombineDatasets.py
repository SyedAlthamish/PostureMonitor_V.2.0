'''{
    File is used to combine multiple ML ready dataset files into one big file in CSV
    }'''

import pandas as pd
import os

# Specify the folder containing CSV files
folder_path = r'C:\Althamish\Project\PostureMonitor_V.2.0\v.2.0\data\V.3.0\v.3.0 ML-Ready'  # Change this to your directory path

# Get a list of all CSV files in the specified folder
file_names = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Initialize an empty list to hold the dataframes
dataframes = []

# Loop through each CSV file in the folder and read it
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)  # Construct full file path
    df = pd.read_csv(file_path)  # Read the CSV file
    dataframes.append(df)  # Append the dataframe to the list

# Concatenate all the dataframes into one
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined dataframe to the same folder
output_file_path = os.path.join(folder_path, 'NET_combined_filtered_labeled_output.csv')
combined_df.to_csv(output_file_path, index=False)
