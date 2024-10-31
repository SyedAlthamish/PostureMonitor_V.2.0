'''{
    Just a basic undocumented file on how to acquire a fresh dataset and convert it to a ML ready csv file.
    }'''

import pandas as pd

# Load your data
file_path = "C:\\Althamish\\Project\\Posture Monitor\\Git_PostureMonitor\\PostureMonitor_V.2.0\\v.2.0\\data\\datasets\\syed_finaldemo - Copy.txt"
df = pd.read_csv(file_path, header=None, delim_whitespace=True)

# Give appropriate column names for clarity
df.columns = ['Time', 'Val1', 'Val2', 'Val3', 'Val4', 'Val5', 'Val6', 'Val7', 
              'Val8', 'Val9', 'Val10', 'Val11', 'Val12', 'Val13']

# Initialize an empty list to store the combined rows
combined_rows = []

# Loop through the DataFrame in steps of 3 to get rows with ID 1, 2, 3
for i in range(0, len(df), 3):
    # Check if there are at least three rows remaining to combine
    if i + 2 < len(df):
        row1 = df.iloc[i]       # Row with Val13 == 1
        row2 = df.iloc[i + 1]   # Row with Val13 == 2
        row3 = df.iloc[i + 2]   # Row with Val13 == 3
        
        # Combine the values of row1, row2, and row3 into a single row
        combined_row = pd.concat([row1, row2, row3]).reset_index(drop=True)
        combined_rows.append(combined_row)

# Convert combined rows list into a DataFrame
combined_df = pd.DataFrame(combined_rows)

columns_to_drop = [12, 13, 14, 26, 27, 28, 40, 41]
combined_df.drop(combined_df.columns[columns_to_drop], axis=1, inplace=True)

print(combined_df)
'''
# Save the combined DataFrame to a new CSV file
combined_df.to_csv("combined_output.csv", index=False)
print("Combined rows saved to 'combined_output.csv'")
'''