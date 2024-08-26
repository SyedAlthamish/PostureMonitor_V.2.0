
import pandas as pd

# Load the CSV file
file_path = 'C:/Users/DELL/Desktop/syed_1.csv'
df = pd.read_csv(file_path)

# Define the time ranges to remove
ranges_to_remove = [
    (0.34, 0.39),  # Example range 1
    (0.42, 0.47),
    (0.52,0.56),
    (0.59,1.06),
    (1.1,1.17),
    (1.21,1.3),
    (1.33,1.38),
    (1.41,1.46)# Example range 2
]

# Convert timestamp column to datetime if it's not already


# Remove rows that fall within the specified time ranges

# Function to label the posture based on the timestamp
def get_label(t):
    # Replace the following logic with your actual labeling criteria
    if 0.3 <= t <= 0.33:
        return 'NEUTRAL'
    elif 0.4 <= t <= 0.41:
        return 'SLOUCH_MOD'
    elif 0.48 <= t <= 0.51:
        return 'SLOUCH_MILD'
    elif 0.57 <= t <= 0.58:
        return 'SLOUCH_EXT'
    elif 1.07 <= t <= 1.09:
        return 'HUNCH_MOD'
    elif 1.18<= t <= 1.20:
        return 'HUNCH_EXT'
    elif 1.31 <= t <= 1.32:
        return 'HUNCH_RIGHT'
    elif 1.39 <= t <= 1.40:
        return 'HUNCH_LEFT'
    else:
        return 'UNKNOWN'

def is_in_remove_range(t, ranges):
    return any(start <= t <= end for start, end in ranges)

# Remove rows that fall within the specified ranges to remove
for start_value, end_value in ranges_to_remove:
    mask = (df['t'] >= start_value) & (df['t'] <= end_value)
    df = df[~mask]

# Apply the labeling function to the remaining rows
df['posture'] = df['t'].apply(get_label)

# Remove any rows labeled as 'UNKNOWN'
df = df[df['posture'] != 'UNKNOWN']

# Combine every three rows into a single row, ensuring that all three rows are within the same range
combined_rows = []
labels = []

i = 0
while i < len(df) - 2:
    # Extract the next three rows
    t_values = df.iloc[i:i+3]['t'].values
    
    # Check if all three rows' t values are not in the removal ranges
    if all(not is_in_remove_range(t, ranges_to_remove) for t in t_values):
        # Combine the next three rows
        combined_row = pd.concat([df.iloc[i, 1:10], df.iloc[i+1, 1:10], df.iloc[i+2, 1:10]])
        combined_rows.append(combined_row.values)
        
        # Use the 't' value of the first row in the triplet for labeling
        labels.append(df.iloc[i]['posture'])
        
        # Move to the next set of three rows
        i += 3
    else:
        # Skip rows that do not form a valid triplet
        i += 1

# Create a new DataFrame with the combined rows and labels
combined_df = pd.DataFrame(combined_rows, columns=[f'sensor-{sensor}-{col+1}' for sensor in range(1, 4) for col in range(9)])
combined_df['posture'] = labels

# Save the updated DataFrame back to a CSV file
output_file_path = 'C:/Users/DELL/Desktop/combined_filtered_labeled_output_4.csv'
combined_df.to_csv(output_file_path, index=False)

print(f"Combined rows and labels have been added and saved to {output_file_path}")