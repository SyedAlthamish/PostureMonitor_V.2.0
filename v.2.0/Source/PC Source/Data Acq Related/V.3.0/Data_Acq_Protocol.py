'''{
    This file is to be run on PC while connected to the pico to perform v.3.0's postural data acquisiton protocol
    }'''
# %% ############################ Connection Initialization #################################

import serial  # Import serial module for communication with Pico

# Define Pico's serial communication parameters
pico_port = "COM16"  # Update with your correct port
baud_rate = 115200   # Ensure this matches the Pico's configured baud rate

# Establish a serial connection with the Pico
ser = serial.Serial(pico_port, baud_rate, timeout=1)

# Ask Permission loop before proceeding
while(1):
    permission = input("Shall We Begin(y/n)?")      # asking permission to user
    if permission == 'y':
        ser.write(b"y")  # Send confirmation signal to Pico
        break  # Exit loop once permission is granted

# %% ########################## Randomized Image Display ###############################
import os
import random
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import threading
import matplotlib.image as mpimg
import serial
from pathlib import Path

# Define the path to the folder containing posture images
image_folder = (
    Path(r"C:\Althamish\Project\Posture Monitor\Git_PostureMonitor")
    / "PostureMonitor_V.2.0"
    / "v.2.0"
    / "Documents & Others"
    / "Others"
    / "Work_Pictures"
    / "Data_Acquisiton v.3.0 postures"
    / "Sample"
)

# List all image files in the directory
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
random.shuffle(image_files)  # Randomize the order of images

# Initialize shared variables for serial data collection
main_data = ""  # Stores received serial data
running = False  # Flag to control serial data reading
posture = ''  # Tracks current posture

def read_serial_data():
    """ Continuously reads serial data while 'running' is True """
    global running, posture, ser, main_data
    while running:
        current_sample = ser.readline().decode().strip()
        if current_sample:  # Append only if valid data is received
            amended_sample = "\n" + current_sample + "    " + posture
            main_data += amended_sample  # Store formatted data

def countdown(duration, message):
    """ Displays a countdown timer of 'duration' in the terminal with 'message' message """
    for sec in range(duration, 0, -1):
        print(f"\r{message} {sec} seconds", end="", flush=True)
        time.sleep(1)
    print("\r" + " " * 40, end="\r")  # Clears countdown text

def add_text_overlay(image_path, text="Next"):
    """ Adds a 'Next' label to the image """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)  # Use Arial if available
    except:
        font = ImageFont.load_default()  # Use default font as fallback

    text_position = (img.width // 2 - 60, 3)  # Position text near the top center
    draw.text(text_position, text, fill="red", font=font)
    
    return img

# Lock to manage shared resource (posture variable)
lock = threading.Lock()

def change_posture(data):
    """ Safely updates the shared posture variable i.e. Posture_State """
    global posture, lock
    with lock:
        posture = data
    print("\nMain thread changed the shared variable!")

### Main Function
def display_images_with_timer(image_files):
    """ Displays images from ' image_files' sequentially while simultaneously logging sensor data """
    
    change_posture("Unknown")
    
    # Begin Serial Data collection in a seperate thread
    global running
    running = True  
    serial_thread = threading.Thread(target=read_serial_data)
    serial_thread.start()
    
    # Display the first image as an introduction
    current_img_path = os.path.join(image_folder, image_files[0])
    current_img = Image.open(current_img_path)
    plt.figure(figsize=(6, 6)) # Adjust size as needed
    mng = plt.get_current_fig_manager()
    mng.window.geometry("+600+100")  # Move to the right (X=600, Y=100)
    plt.imshow(current_img)
    plt.axis("off")
    plt.title(f"Starting with: {image_files[0]}")
    plt.show(block=False) # Show without blocking execution
    plt.pause(0.1)  # Allow image to render

    countdown(10, f"With {image_files[0]} Starting in") # Wait before proceedin
    plt.close()

    # Set up Matplotlib for displaying images side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    current_ax, next_ax = axes
    mng = plt.get_current_fig_manager()
    mng.window.geometry("+600+100")

    for i in range(len(image_files)):
        current_img_path = os.path.join(image_folder, image_files[i]) #current image acq
        next_img_path = os.path.join(image_folder, image_files[(i+1) % len(image_files)])  #next image acq without overflowing index
        
        current_img = Image.open(current_img_path)
        next_img = add_text_overlay(next_img_path, "Next") #placing next on the next image

        # Display current image
        current_ax.clear()
        current_ax.imshow(current_img)
        current_ax.axis("off")
        current_ax.set_title(f"Current Image: {image_files[i]}")

        # Display next image
        next_ax.clear()
        next_ax.imshow(next_img)
        next_ax.axis("off")
        next_ax.set_title(f"Upcoming Image: {image_files[(i+1) % len(image_files)]}")

        # Time to render
        fig.canvas.draw()
        plt.pause(1)
        
        # Change the Posture variable and begin countdown for current and next images
        change_posture(image_files[i])
        countdown(7, f"Hold this {i+1}th posture:")
        change_posture("Transition")
        countdown(5, f"Get ready for {i+2}th:")
    
    # Stopping serial thread after routine
    running = False  # Stop serial thread
    serial_thread.join()
    plt.close(fig)  # Close visualization


## Calling of main function
display_images_with_timer(image_files)
print("Process end")

# %% ########################## Storage and Conversion of data ###############################

# Save collected serial data to a user_defined file
file_name = (
    r"C:\Althamish\Project\Posture Monitor\Git_PostureMonitor\PostureMonitor_V.2.0\v.2.0\data\V.3.0\v.3.0 Raw\\"
    + input("Enter the Data_File Name: ") + ".txt"
)

# opening and storing main_data
with open(file_name, "w") as file:
    file.write(main_data)


# Convert raw data to ML-Ready dataset
import subprocess

# Occasionally, errors may occur at line 98 of the following script.
# In such cases, manually correct the error and run the script independently.
subprocess.run(["python", r"C:\Althamish\Project\Posture Monitor\Git_PostureMonitor\PostureMonitor_V.2.0\v.2.0\Source\PC Source\Data Acq Related\V.3.0\v.3.0Raw CSV to ML-Ready CSV.py"])
