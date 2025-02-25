'''{
    File Description:
        This file's function is to take the pre-calibrated data from sensor and estimate the performance of the threshold
        used to pre-calibrate by looking at the post-calibrated data's stability in various situations.
    }'''

#%% ######################################### Basic details about Calibrated data #####################################
import statistics                                                                                          # Import statistics module for calculations
from tabulate import tabulate                                                                             # Import tabulate for pretty table formatting

# Data Structure
data = {                                                                                                   # Dictionary to hold sensor data for different states
    "Idle": {                                                                                              # State: Idle
        "Sensor 1": {                                                                                     # Data for Sensor 1
            "X": [-3.621863, -3.639724, -3.617922, -3.619632, -3.611907, -3.618043, -3.619632, -3.611908, -3.619418, -3.611876],  # X-axis values
            "Y": [0.899389, 0.8889156, 0.9021679, 0.900977, 0.8973742, 0.9113582, 0.9069615, 0.9081218, 0.9039999, 0.9010987],  # Y-axis values
            "Z": [-3.58794, -3.566075, -3.590048, -3.563508, -3.577311, -3.567115, -3.575482, -3.567878, -3.567571, -3.56284]   # Z-axis values
        },
        "Sensor 2": {                                                                                     # Data for Sensor 2
            "X": [-1.09371, -1.101648, -1.089069, -1.098015, -1.097222, -1.104457, -1.099847, -1.101343, -1.10568, -1.100214],  # X-axis values
            "Y": [0.6949006, 0.701588, 0.6959084, 0.7018014, 0.7038473, 0.6796336, 0.7021373, 0.7035113, 0.702809, 0.7039698],  # Y-axis values
            "Z": [-0.1783512, -0.1738015, -0.1703511, -0.178687, -0.1557252, -0.1904732, -0.2279694, -0.176, -0.1791449, -0.2040304]  # Z-axis values
        }
    },
    "Lean Back": {                                                                                        # State: Lean Back
        "Sensor 1": {                                                                                     # Data for Sensor 1
            "X": [-3.689922, -3.706382, -3.641189, -3.676886, -3.574565, -3.619511, -3.683726, -3.652946, -3.640764, -3.61426],   # X-axis values
            "Y": [0.9582293, 0.9964887, 0.9608246, 0.9042138, 1.220794, 1.113222, 1.04177, 1.000031, 1.050595, 1.153771],         # Y-axis values
            "Z": [-3.568456, -3.614135, -3.568244, -3.534747, -3.627724, -3.635999, -3.565465, -3.542565, -3.551266, -3.591816]   # Z-axis values
        },
        "Sensor 2": {                                                                                     # Data for Sensor 2
            "X": [-1.326931, -1.229252, -1.355451, -1.289557, -1.193343, -1.251817, -1.34516, -1.298993, -1.270107, -1.302657],    # X-axis values
            "Y": [0.9753284, 0.7792669, 0.9428399, 0.8726107, 0.772458, 0.8319698, 0.920916, 0.8967023, 0.8242748, 0.8899537],  # Y-axis values
            "Z": [-0.2690686, -0.3362745, -0.272061, -0.2950228, -0.2861982, -0.2998166, -0.2808853, -0.3055877, -0.2964887, -0.3069618]  # Z-axis values
        }
    },
    "Lean Forward Slouch": {                                                                              # State: Lean Forward Slouch
        "Sensor 1": {                                                                                     # Data for Sensor 1
            "X": [-3.70638, -3.60919, -3.659021, -3.67948, -3.649496, -3.634928, -3.633313, -3.699172, -3.676518, -3.643145],     # X-axis values
            "Y": [1.06113, 1.000244, 1.043969, 1.02226, 0.9921829, 1.052917, 1.014657, 1.179939, 1.031053, 1.032153],             # Y-axis values
            "Z": [-3.784122, -3.51545, -3.555968, -3.521283, -3.526535, -3.551206, -3.588395, -3.638378, -3.592089, -3.549832]   # Z-axis values
        },
        "Sensor 2": {                                                                                     # Data for Sensor 2
            "X": [-1.308458, -1.298444, -1.212092, -1.229894, -1.227817, -1.232854, -1.17716, -1.184977, -1.222289, -1.267267],    # X-axis values
            "Y": [0.8029612, 0.8012825, 0.8036339, 0.7974962, 0.812702, 0.7952976, 0.8224121, 0.7632057, 0.7981378, 0.8211293], # Y-axis values
            "Z": [-0.2842442, -0.289374, -0.3199695, -0.2614351, -0.2821067, -0.3014961, -0.2440306, -0.278931, -0.3135266, -0.2548092]  # Z-axis values
        }
    }
}

# Prepare table data
table_data = []                                                                                           # Initialize list for table data
for state, sensors in data.items():                                                                       # Iterate through states
    for sensor, axes in sensors.items():                                                                  # Iterate through sensors
        for axis, values in axes.items():                                                                  # Iterate through axes
            mean_val = statistics.mean(values)                                                             # Calculate mean
            std_dev = statistics.stdev(values)                                                             # Calculate standard deviation
            var_val = statistics.variance(values)                                                          # Calculate variance
            median_val = statistics.median(values)                                                         # Calculate median
            min_val = min(values)                                                                          # Calculate minimum value
            max_val = max(values)                                                                          # Calculate maximum value
            table_data.append([state, sensor, axis, mean_val, std_dev, var_val, median_val, min_val, max_val])  # Append data to table

# Define headers
headers = ["State", "Sensor", "Axis", "Mean", "Std Dev", "Variance", "Median", "Min", "Max"]              # Define table headers

# Print the table
try:                                                                                                      # Try to print the table
    print(tabulate(table_data, headers=headers, tablefmt="grid"))                                        # Print formatted table
except ImportError:                                                                                       # Handle ImportError for tabulate
    print("\nTabulate module not found. Showing data in plain format:\n")                                 # Inform user of plain format
    print(f"{'State':<12}{'Sensor':<10}{'Axis':<6}{'Mean':<10}{'Std Dev':<10}{'Variance':<10}{'Median':<10}{'Min':<6}{'Max':<6}")  # Print header
    print("=" * 80)                                                                                      # Print separator
    for row in table_data:                                                                               # Iterate through table data
        print(f"{row[0]:<12}{row[1]:<10}{row[2]:<6}{row[3]:<10.2f}{row[4]:<10.2f}{row[5]:<10.2f}{row[6]:<10.2f}{row[7]:<6}{row[8]:<6}")  # Print each row


#%% ################################################## Difference Between one calib_state vs other calib_state ##########################

import itertools                                                                                          # Import itertools for combinations
import statistics                                                                                          # Import statistics module for calculations
from tabulate import tabulate                                                                             # Import tabulate for pretty table formatting

# Prepare comparison table data
comparison_data = []                                                                                      # Initialize list for comparison data
mean_differences = {}                                                                                     # Initialize dictionary for mean differences
states = list(data.keys())                                                                                # List of states
for (state1, state2) in itertools.combinations(states, 2):                                              # Compare each state pair
    for sensor in data[state1]:                                                                          # Iterate through sensors
        for axis in data[state1][sensor]:                                                                # Iterate through axes
            mean1 = statistics.mean(data[state1][sensor][axis])                                         # Calculate mean for state1
            mean2 = statistics.mean(data[state2][sensor][axis])                                         # Calculate mean for state2
            mean_diff = mean1 - mean2                                                                     # Compute mean difference
            comparison_data.append([state1, state2, sensor, axis, mean1, mean2, mean_diff])              # Append comparison data
            # Store mean difference for extraction
            key = (state1, state2, sensor, axis)                                                         # Create key for mean differences
            mean_differences[key] = mean_diff                                                             # Store mean difference

# Define headers for comparison table
comparison_headers = ["State 1", "State 2", "Sensor", "Axis", "Mean 1", "Mean 2", "Mean Diff"]            # Define headers for comparison table

# Print the comparison table
print(tabulate(comparison_data, headers=comparison_headers, tablefmt="grid"))                            # Print formatted comparison table

sensor_names_list = []                                                                                   # Initialize list for sensor names
for i in data["Idle"].keys():                                                                            # Iterate through sensors in Idle state
    sensor_names_list.append(i)                                                                          # Append sensor names to list

axes_names_list = []                                                                                     # Initialize list for axes names
for i in data["Idle"][sensor_names_list[0]].keys():                                                   # Iterate through axes of the first sensor
    axes_names_list.append(i)                                                                            # Append axes names to list

def get_meanof_mean_difference(state1, state2):                                                                 # Function to calculate mean of mean_ difference between two states
    mean_difference_list = []                                                                             # Initialize list for mean differences
    for sensor in sensor_names_list:                                                                      # Iterate through sensors
        for axis in axes_names_list:                                                                      # Iterate through axes
            key = (state1, state2, sensor, axis)                                                         # Create key for mean differences
            mean_difference_list.append(abs(mean_differences[key]))                                     # Append absolute mean differences
    return(statistics.mean(mean_difference_list))                                                         # Return mean of mean differences

# Test the get_mean_difference function
print("get_mean_difference('Idle','Lean Forward Slouch')", get_meanof_mean_difference("Idle", "Lean Forward Slouch"))  # Print mean difference
print("get_mean_difference('Idle','Lean Back')", get_meanof_mean_difference("Idle", "Lean Back"))                # Print mean difference