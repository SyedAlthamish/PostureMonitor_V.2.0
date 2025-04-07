'''{
    This file converts all csv files into one big file
    }'''

import pandas as pd
import glob
import os

# Define the path to the folder containing the CSV files
folder_path = r'C:\Althamish\Project\PostureMonitor_V.2.0\v.2.0\data\Trash\_'

# Create a list of all CSV file paths in the folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Initialize an empty list to hold the DataFrames
dataframes = []

# Loop through the list of CSV files and read each one into a DataFrame
for file in csv_files:
    df = pd.read_csv(file)
    dataframes.append(df)

# Concatenate all the DataFrames into one
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv(folder_path + '\combined_output.csv', index=False)

print("Combined CSV file created successfully.")
