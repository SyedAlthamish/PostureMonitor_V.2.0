"this is a file to show a representative basic display of classified posture from the posture monitoring system to run on PC"

import time

# List of strings to print
strings = ["Calibrating","You are Neutral", "You are Slouching extensively", "You are Leaning right","You are Hunching right", "You are Neutral"]
string_index=-1
# Loop through each string

for string_index in range(0,len(strings)):
    start_time = time.time()
    if string_index==3: int_time=3
    else: int_time=7
    # Keep printing the current string until 7 seconds have passed
    while time.time() - start_time < int_time:
        print(strings[string_index])
        time.sleep(0.01)  # Delay of 1 second between each print within the 7 seconds
