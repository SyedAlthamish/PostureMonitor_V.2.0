'''
this file is used to take an input raw text document dataset from the 
DAcq_run.py file and convert into a ML ready csv file
i.e. to produce structured and labelled dataset with columns from 'sensor-1-1' to 'sensor-3-11'&'postures' from raw text output of  pico
'''

import pandas as pd

# Load your data
file_path = "C:\\Althamish\\Project\\Posture Monitor\\Git_PostureMonitor\\PostureMonitor_V.2.0\\v.2.0\\data\\datasets\\syed_finaldemo - Copy.txt"
df = pd.read_csv(file_path, header=None, delim_whitespace=True)


#---------------------------------Combining rows and dropping unnecessary columns--------------------------------------


# Initialize an empty list to store the combined rows
combined_rows = []
no_of_rows_to_combine=3
# Loop through the DataFrame in steps of 3 to get rows with ID 1, 2, 3
for row_no in range(0, len(df), no_of_rows_to_combine):
    # Check if there are at least three rows remaining to combine
    if row_no + 2 < len(df):
        row1 = df.iloc[row_no]       # Row with Val13 == 1
        row2 = df.iloc[row_no + 1]   # Row with Val13 == 2
        row3 = df.iloc[row_no + 2]   # Row with Val13 == 3
        
        # Combine the values of row1, row2, and row3 into a single row
        combined_row = pd.concat([row1, row2, row3]).reset_index(drop=True)
        combined_rows.append(combined_row)

# Convert combined rows list into a DataFrame
combined_df = pd.DataFrame(combined_rows)

columns_to_drop = [12, 13, 14, 26, 27, 28, 40, 41]          #these column indexes represent: sensor_no,timestamps,and dt of all 3 sensors except for the timestamp of 1st sensor
combined_df.drop(combined_df.columns[columns_to_drop], axis=1, inplace=True)


#-----------------------------------------------------------------------------------------------------------------------
#-------------------------------Opening Timestamp folder and extracting time val from it--------------------------------


# Specify the path to your input text file, with the basic format of txt file named: timestamp_format.txt in datasets directory
file_path = r"C:\Althamish\Project\Posture Monitor\Git_PostureMonitor\PostureMonitor_V.2.0\v.2.0\data\datasets\syed_finaldemo_timestamps.txt"  


# Read the text file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Initialize collections
p_collection = []  # For pairs surrounding labels other than 'change'; i.e. postures
c_collection = []  # For pairs surrounding 'change'

# Process the lines
for i in range(0, len(lines) - 1, 2):  # Step through pairs of lines
    value = float(lines[i].strip())
    label = lines[i + 1].strip()
    previous_value = float(lines[i + 2].strip())
    
    if label == "change":
        # If the label is 'change', create a pair with the previous value
        c_collection.append((value, previous_value))
    else:
        # If the label is not 'change', create a pair with the next value
        p_collection.append((value, previous_value))

# Print the collections
print("p_collection:", p_collection)
print("c_collection:", c_collection)


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------Labelling the dataset and cleaning up rows, col, and naming them----------------------


p_collection_bounds = p_collection

# Corresponding labels for each range
labels = [
    'NEUTRAL',
    'SLOUCH_MILD',
    'SLOUCH_MOD',
    'SLOUCH_EXT',
    'HUNCH_MOD',
    'HUNCH_EXT',
    'HUNCH_RIGHT',
    'HUNCH_LEFT',
    'LEAN_RIGHT',
    'LEAN_LEFT'
]

# Combine bounds with labels
p_collection = [(lower, upper, label) for (lower, upper), label in zip(p_collection_bounds, labels)]

def get_label(t):
    # Iterate through each pair in p_collection
    for lower_bound, upper_bound, label in p_collection:
        if lower_bound <= t <= upper_bound:
            return label
    
    # If no conditions were met, return UNKNOWN
    return 'UNKNOWN'

labeledC_df=combined_df
labeledC_df['posture'] = labeledC_df[0].apply(get_label)                #applying posture column into dataset

cleanedLC_df = labeledC_df[labeledC_df['posture'] != 'UNKNOWN']         #removing 'unknown' postures from dataset

cleanedLC_df.drop(cleanedLC_df.columns[0], axis=1, inplace=True)        #removing timestamp column from dataset

colnames=['sensor-1-1', 'sensor-1-2', 'sensor-1-3', 'sensor-1-4', 'sensor-1-5', 'sensor-1-6', 'sensor-1-7', 'sensor-1-8', 'sensor-1-9', 'sensor-1-10', 'sensor-1-11', 
'sensor-2-1', 'sensor-2-2', 'sensor-2-3', 'sensor-2-4', 'sensor-2-5', 'sensor-2-6', 'sensor-2-7', 'sensor-2-8', 'sensor-2-9', 'sensor-2-10', 'sensor-2-11', 
'sensor-3-1', 'sensor-3-2', 'sensor-3-3', 'sensor-3-4', 'sensor-3-5', 'sensor-3-6', 'sensor-3-7', 'sensor-3-8', 'sensor-3-9', 'sensor-3-10', 'sensor-3-11','postures']
cleanedLC_df.columns = colnames                                         #adding column names to the dataset


#----------------------------------------creating the ML ready CSV file------------------------------------------------


finaldf=cleanedLC_df    
finaldf.to_csv("syed_finaldemo_MLReady.csv", index=False)
print("Combined rows saved to 'combined_output.csv'")