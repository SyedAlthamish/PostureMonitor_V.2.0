'''{
    File Description:
    
    Short: 
        this file is a basic implementation of real-time inputting of sensor data and outputting of lstm-classified
        labels from an imported trained lstm classifier.
    
    Long:
        This script implements an LSTM-based posture classification system for live data processing on a 
        Raspberry Pi 5. It begins by loading the necessary preprocessing tools, including a StandardScaler for 
        feature scaling and a LabelEncoder for mapping predictions back to posture labels. The TensorFlow Lite 
        (.tflite) model is then loaded and allocated with input and output tensors for inference. A serial connection 
        is established to receive real-time sensor data from an external device, with adjustable port and baud rate 
        settings. Incoming data is read, decoded, and converted into a numerical array, ensuring it matches the expected 
        input format. The data is then standardized using the preloaded scaler and reshaped to fit the LSTM model’s 
        input structure. The model makes predictions, which are mapped back to posture labels using the label encoder. 
        The script continuously processes live sensor data and prints the predicted posture, along with specific messages
        for different classifications, such as "Neutral posture," "Slouching," "Hunching," or "Leaning" in various 
        directions.
    }'''


# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:20:06 2025

@author: saisr
"""

# -*- coding: utf-8 -*-
"""
    LSTM-based posture classification for live data on Raspberry Pi 5
"""

import serial
import pickle
import numpy as np
import tensorflow.lite as tflite
from sklearn.preprocessing import StandardScaler

# Load the scaler and label encoder
scaler_path = "scaler.pkl"
label_encoder_path = "label_encoder.pkl"
model_path = "lstm_model.tflite"  # Path to your converted TFLite model

try:
    scaler = pickle.load(open(scaler_path, 'rb'))
    label_encoder = pickle.load(open(label_encoder_path, 'rb'))
    print("Scaler and label encoder loaded successfully.")
except Exception as e:
    print(f"Error loading scaler or label encoder: {e}")
    exit()

# Load the TensorFlow Lite model
try:
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print("TFLite model loaded successfully.")
except Exception as e:
    print(f"Error loading TensorFlow Lite model: {e}")
    exit()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Serial port configuration (adjust as per Raspberry Pi settings)
port = "/dev/serial0"  # Adjust based on your setup (e.g., "/dev/ttyUSB0")
baudrate = 9600  # Standard baud rate

try:
    ser = serial.Serial(port, baudrate, timeout=1)
    print(f"Connected to serial port: {port}")
    print("Waiting for data...")
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    exit()

# Function to read and process data from the serial port
def receive_and_convert_data(ser):
    try:
        data = ser.readline().decode().strip()
        values = data.split("    ")  # Adjust separator if necessary

        if len(values) != 6:
            print(f"Unexpected number of values received: {len(values)}")
            return None

        float_values = np.array([float(val) for val in values]).reshape(1, -1)
        return float_values
    except Exception as e:
        print(f"Error receiving data: {e}")
        return None

# Function to make predictions using the TensorFlow Lite model
def predict_posture(data):
    try:
        # Scale input data
        scaled_data = scaler.transform(data)
        scaled_data = np.array(scaled_data, dtype=np.float32).reshape(1, 1, -1)  # Reshape for LSTM

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], scaled_data)

        # Run inference
        interpreter.invoke()

        # Get output tensor and predict class
        output = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output)

        # Decode predicted class to label
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        return predicted_label
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

# Main loop for live detection
while True:
    data_list = receive_and_convert_data(ser)
    if data_list is None:
        continue

    predicted_posture = predict_posture(data_list)
    
    if predicted_posture:
        print(f"Predicted Posture: {predicted_posture}")

        # Custom messages based on detected posture
        if predicted_posture == "NEUTRAL":
            print("You are in a Neutral posture.")
        elif predicted_posture == "SLOUCH_EXT":
            print("You are Slouching extensively.")
        elif predicted_posture == "SLOUCH_MILD":
            print("You are Slouching mildly.")
        elif predicted_posture == "SLOUCH_MOD":
            print("You are Slouching moderately.")
        elif predicted_posture == "HUNCH_LEFT":
            print("You are Hunching to the left.")
        elif predicted_posture == "HUNCH_MOD":
            print("You are Hunching moderately.")
        elif predicted_posture == "HUNCH_EXT":
            print("You are Hunching extensively.")
        elif predicted_posture == "HUNCH_RIGHT":
            print("You are Hunching to the right.")
        elif predicted_posture == "LEAN_LEFT":
            print("You are Leaning to the left.")
        elif predicted_posture == "LEAN_RIGHT":
            print("You are Leaning to the right.")

    print(f"Received data: {data_list}")
