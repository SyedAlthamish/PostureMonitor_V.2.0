from machine import Pin, I2C, UART
import utime
import math
#from mpu6050 import init_mpu6050, get_mpu6050_data, calibrate_gyroonlyZ, calci_tilt_accangles
'''
i2c = I2C(0, scl=Pin(21), sda=Pin(20), freq=400000)
init_mpu6050(i2c, 0x68)
init_mpu6050(i2c, 0x69)  # Initialize second MPU6050 sensor with address 0x69

calibrate_gyroonlyZ(i2c, 0x68,0,100)
calibrate_gyroonlyZ(i2c, 0x69,1,100)  # Calibrate the second sensor as well
'''
#uart = machine.UART(0, baudrate=9600, tx=machine.Pin(16), rx=machine.Pin(17))
'''
pitch = 0
roll = 0
prev_time = utime.ticks_ms()
'''
'''
while True:

    data_68 = get_mpu6050_data(i2c, 0x68,0)
    data_69 = get_mpu6050_data(i2c, 0x69,1)  # Get data from the second sensor
    
    curr_time = utime.ticks_ms()
    dt = (curr_time - prev_time) / 1000
    prev_time = curr_time
    
    tilt_x_68, tilt_y_68, tilt_z_68 = calci_tilt_accangles(data_68, dt,0, 0.98)
    tilt_x_69, tilt_y_69, tilt_z_69 = calci_tilt_accangles(data_69, dt,1, 0.98)  # Calculate tilt angles for the second sensor
    
    # Assuming you want to print the tilt angles for both sensors
    print(tilt_x_68, tilt_y_68, tilt_z_68,tilt_x_69, tilt_y_69, tilt_z_69)
    output_string = str(tilt_x_68) + "\t\t\t" + str(tilt_y_68) + "\t\t\t" + str(tilt_z_68) + "\t\t\t" + str(tilt_x_69) + "\t\t\t" + str(tilt_y_69) + "\t\t\t" + str(tilt_z_69) + "\n"
    uart.write("hi".encode())
    
    if uart.any():
        data = uart.read()  # Read available data
        uart.write("received:"+data)
        
    utime.sleep(0.01)
'''
'''# Main loop
while True:
    # Send a message every 2 seconds
    print("Hello from Pico via USB!")

    # Check if data is received from the laptop (via PuTTY)
    received_data = input()  # Waits for input from the serial connection
    
    if received_data:
        print("Received: " + received_data)  # Echo the received data back

    # Short delay
    utime.sleep(2)
'''
'''
# Main loop
import utime
import sys

# Main loop
while True:
    # Send a message every 2 seconds
    print("Hello from Pico via USB!")
    
    # Check if there's data in the input buffer (non-blocking)
    print("out if")
    if sys.stdin in sys.stdin.read():
        print("in if")
        received_data = sys.stdin.read(1)  # Read one character (or adjust as needed)
        
        # Process the received data
        print("Received: " + received_data)
    print("follow if")
    # Short delay
    utime.sleep(2)

'''
import utime
import sys
'''
# Function to attempt non-blocking read from USB (sys.stdin)
def non_blocking_read():
    try:
        # Attempt to read a single character
        return sys.stdin.read(1)
    except Exception as e:
        # No data was available, or other error
        return None
'''
def get_timestamp():
    # Get the current time
    current_time = utime.localtime()
    # Format the time as [HH:MM:SS]
    return "{:02}.{:02} ".format(current_time[4], current_time[5])

# Main loop
while True:
    # Send a message every 2 seconds
    print(get_timestamp()+ "Hello from Pico via USB!")
    utime.sleep(2)
    # Attempt to read data without blocking
'''
    received_data = non_blocking_read()
    
    if received_data:
        print("Received: " + received_data)
''' 
    # Short delay
    
