# -*- coding: utf-8 -*-
"""
File to input a Pickle classifier file, continuous data from a serial port and output a classification based on o/p
"""

import serial
import pickle  # Example for generating dummy data (remove if using real model)
import sklearn
from sklearn.preprocessing import StandardScaler
#from serial import SerialException
sc = StandardScaler()

# Replace with the actual COM port connected to the HC-05
port = "COM16"  # Example for Windows, adjust for your OS
baudrate = 9600  # Standard baud rate for HC-05 communication

# Path to your pickled model file
model_path = "model3.pkl"

# Function to receive and process data
def receive_and_convert_data(ser):
    try:
        # Read data from serial buffer (modify based on data format)
        data = ser.readline().decode().strip()
        #print(data)
        # Split data based on three tab spaces
        values = data.split("    ")
        #print(values)
        # Check if we received exactly 6 values
        if len(values) != 6:
            print(f"Received unexpected number of values: {len(values)}")
            return None
        # Convert values to float
        float_values = [float(val) for val in values]
        # Create a double enclosed list with the converted values
        data_list = [[val for val in float_values]]
        return data_list
    except serial.SerialException as e:  # This is the correct exception handling
        print(f"Error receiving data: {e}")
        return None

# Load the machine learning model from pickle file
try:
    with open(model_path, "rb") as f:
        model1 = pickle.load(f)
    print(f"Loaded model from: {model_path}")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Open serial connection
try:
    ser = serial.Serial(port, baudrate)
    print(f"Connected to serial port: {port}")
    print("waiting for PICO to send something")
    for i in range(0,9):
        print(ser.readline())
except serial.SerialException as e:  # This is the correct exception handling
    print(f"Error opening serial port: {e}")
    exit()
    
'''
    
# Tkinter GUI setup
import tkinter as tk
root = tk.Tk()
root.title("Posture Monitor")

posture_label = tk.Label(root, text="Current posture: Unknown")
posture_label.pack()

# Function to update the GUI and display alerts
def update_gui(prediction):
    posture_text = "Neutral"
    alert_message = ""
    if prediction == 0:
        posture_text = "Hunched"
        alert_message = "Alert: Slouching detected!"
    elif prediction == 2:
        posture_text = "Slouching"
        alert_message = "Alert: Slouching detected!"

    posture_label.config(text=f"Current posture: {posture_text}")

    if alert_message:
        tk.messagebox.showwarning("Posture Alert", alert_message)
'''
# Main program loop with GUI updates
while True:
    # Receive and process data
    data_list = receive_and_convert_data(ser)
    if data_list is None:
        continue  # Retry on error

    # Make prediction using the loaded model
    try:

        #print(data_list)
        prediction = model1.predict((data_list))  # Assuming your model has a predict method
        # Extract the predicted value (might need adjustment based on your model output)
        # predicted_value = prediction[0]  # Assuming single output value
        print(prediction)
        if(prediction[0] == 0):
            print("You are Hunching extensively")
        if (prediction[0] == 1):
            print(" You are Hunching left")
        if (prediction[0] == 2):
            print("You are Hunching Moderately")
        if (prediction[0] == 3):
              print("You are Hunching right")
        if (prediction[0] == 4):
              print("You are Leeaning left")
        if (prediction[0] == 5):
              print("You are Leaning right")
        if (prediction[0] == 6):
              print("You are Neutral")
        if (prediction[0] == 7):
              print("You are Slouching extensively")
        if (prediction[0] == 8):
               print("You are Slouching mild")
        if (prediction[0] == 9):
               print("You are Slouching moderately")
    except Exception as e:
        print(f"Error making prediction: {e}")
        prediction[0] = None

    # Print the received data, prediction (or error message)
    #print(f"Received data: {data_list}")
