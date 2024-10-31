import pandas as pd

# List of specific CSV files to combine
file_names = [f"combined_filtered_labeled_output_{i}.csv" for i in range(1, 13)]  # Generates ['1.csv', '2.csv', ..., '12.csv']
path = 'datasets/'  # Change this to your directory path

# Initialize an empty list to hold the dataframes
dataframes = []

# Loop through the specified file names and read each one
for file_name in file_names:
    file_path = f"{path}{file_name}"  # Construct full file path
    df = pd.read_csv(file_path)  # Read the CSV file
    dataframes.append(df)  # Append the dataframe to the list

# Concatenate all the dataframes into one
combined_df = pd.concat(dataframes, ignore_index=True)

# Optionally, save the combined dataframe to a new CSV file
combined_df.to_csv('NET_combined_filtered_labeled_output.csv', index=False)
