'''{
    This file is to be run on PC while connected to the pico to perform v.3.0's postural data acquisiton protocol
    }'''
# %% ############################ Connection Initialization #################################

import serial

pico_port = "COM16"  # Update with your correct port
baud_rate = 115200   # Same as Pico's baud rate

ser = serial.Serial(pico_port, baud_rate, timeout=1)

#Ask Permission loop
while(1):
    permission = 'y'#input("Shall We Begin?(y/n)")
    if permission == 'y':
        ser.write(b"y")
        break

# %% ########################## Randomized Image Display ###############################
import os
import random
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import threading
import matplotlib.image as mpimg
import time
import serial
import os

# Folder containing images
image_folder = r"C:\Althamish\Project\Posture Monitor\Git_PostureMonitor\PostureMonitor_V.2.0\v.2.0\Documents & Others\Others\Work_Pictures\Data_Acquisiton v.3.0 postures copy"

# List of image filenames
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
random.shuffle(image_files)  # Shuffle images initially


main_data = ""  # List to store collected serial data
running = False  # Control flag for serial reading
posture=''
check_data =0

def read_serial_data():
    """ Continuously reads serial data while 'running' is True """
    global running,posture,ser,main_data,check_data
    while running:
        #check_data = check_data + 1
        current_sample = ser.readline().decode().strip()
        if current_sample:  # Avoid appending empty reads
            amended_sample = "\n" + current_sample + "    " + posture  # Fast string concatenation
            main_data += amended_sample  # Append to the existing string
        else:
            print("empty")

def countdown(duration, message):
    """ Function to run a countdown timer """
    for sec in range(duration, 0, -1):
        print(f"\r{message} {sec} seconds", end="", flush=True)
        time.sleep(1)
    print("\r" + " " * 40, end="\r")  # Clear the text

def add_text_overlay(image_path, text="Next"):
    """Adds a 'Next' label to the image."""
    img = Image.open(image_path).convert("RGB")  # Ensure RGB mode
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf",20)  # Use standard font
    except:
        font = ImageFont.load_default()  # Fallback if arial.ttf isn't available

    text_position = (img.width // 2 - 60, 3)  # Center near the top
    draw.text(text_position, text, fill="red", font=font)
    
    return img

lock = threading.Lock()  # Create a lock
def change_posture(data):
    global posture,lock
    with lock:
        posture = data
    print("\nMain thread changed the shared variable!")


def display_images_with_timer(image_files):
    """Displays current and next image while running a countdown in the terminal."""
    

    
    change_posture("Unknown")
    
    global running
    running = True  # Activate flag
    serial_thread = threading.Thread(target=read_serial_data)
    serial_thread.start()
    print("Before for")
    
    # Display first image using Matplotlib
    current_img_path = os.path.join(image_folder, image_files[0])
    current_img = Image.open(current_img_path)

    plt.figure(figsize=(6, 6))  # Adjust size as needed
    plt.imshow(current_img)
    plt.axis("off")
    plt.title(f"Starting with: {image_files[0]}")
    plt.show(block=False)  # Show without blocking execution
    plt.pause(0.1)  # Allow the image to render properly

    countdown(10, f"With {image_files[0]} Starting in")  # Wait before proceeding

    plt.close() 

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Two images side by side
    current_ax, next_ax = axes
    
    # Get figure manager
    mng = plt.get_current_fig_manager()

    # Set window position (adjust values based on your screen resolution)
    mng.window.geometry("+600+100")  # Move to the right (X=1200, Y=100)
    

    for i in range(len(image_files)):
       
        current_img_path = os.path.join(image_folder, image_files[i])
        next_img_path = os.path.join(image_folder, image_files[(i+1)% len(image_files)])  # Wrap around

        current_img = Image.open(current_img_path)
        next_img = add_text_overlay(next_img_path, "Next")  # Add "Next" text

        current_ax.clear()
        current_ax.imshow(current_img)
        current_ax.axis("off")
        current_ax.set_title(f"Current Image: {image_files[i]}")

        next_ax.clear()
        next_ax.imshow(next_img)
        next_ax.axis("off")
        next_ax.set_title(f"Upcoming Image: {image_files[(i+1)% len(image_files)]}")
        
        print("befoere plt")
        fig.canvas.draw()  # Ensures Matplotlib refreshes
        plt.pause(1)  # Give time to render
        print("after plt")
        
        
        # Start serial reading in a separate thread
        
        change_posture(image_files[i])
        
        # Run countdown for posture hold (7 seconds)
        countdown(7, f"Hold this {i+1}th posture:")

        change_posture("Transition")
        # Transition to next posture (5 seconds)
        
        countdown(5, f"Get ready for {i+2}th:")

    running = False  # Stop the thread
    serial_thread.join()    
    
    plt.close(fig)  # Close the figure after the full cycle


display_images_with_timer(image_files)
print("process end")

# %% ########################## Raw 2 ML-Ready Dataset ###############################y
print(check_data)
data = main_data

with open("output.txt", "w") as file:  # 'w' mode overwrites existing content
    file.write(data)

