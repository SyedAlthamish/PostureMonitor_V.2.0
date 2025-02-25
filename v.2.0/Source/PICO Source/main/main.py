#Scrap file
'''{
    File Description(main.py):
        this file is expected to run during the data acquisition protocol for v.3.0 when pico is connected to pc via usb
        wire. This file involves pre-calibration and gyro_spike fixes with post-permission automated data collection from
        pc side.
    }'''

###importion of libraries
from machine import Pin, I2C																				#for Pin Manipulation, i2c communication
import utime																								#for delay between each loop
import math																									#for tilt calculation from each sensor
from lib_mpu6050 import *
from lib_misc import *

###The setting of MPU6050 sensor's addresses
pin1 = Pin(0, Pin.OUT)			#making pin_no_0 or GPIO_0 as output mode
pin1.value(1)					#making GPIO_0 as HIGH to represent one of the sensors in i2c0 channel as addresses 0x69
utime.sleep(3)                  # some time for the sensor to power up

###Initialization of I2C, MPU, and Data-Processing
i2c = I2C(0, scl=Pin(21), sda=Pin(20), freq=400000)		#initializing i2c0 channel
init_mpu6050(i2c, 0x68)									#initializing first sensor in i2c0		
init_mpu6050(i2c, 0x69)     							#initializing second sensor in i2c0

### Waiting for Permission from the PC to begin sending data
while True:
    read_data = sys.stdin.read(1)  # hangs here till a byte is read
    if read_data == 'y':
        break
    print("Waiting for Permission(y/n)")

prev_time = utime.ticks_ms()							#initializing time parameteric for integrating gyro data

### Obtains data from MPU6050 sensors and sends it to the PC in a structured format
while True:
    # Retrieve comprehensive sensor data with gyro spike fix applied
    data_68 = get_mpu6050_comprehensive_data_Gyro_Spike_Fix(i2c, 0x68, 1)  # Data from first sensor on i2c0
    data_69 = get_mpu6050_comprehensive_data_Gyro_Spike_Fix(i2c, 0x69, 2)  # Data from second sensor on i2c0
    # data_681 = get_mpu6050_comprehensive_data(i2c1, 0x68, 3)  # Uncomment for third sensor on i2c1
    
    # Record current time and compute time difference (dt) for processing
    curr_time = utime.ticks_ms()  # Get current timestamp in milliseconds
    dt = (curr_time - prev_time) / 1000  # Compute time difference in seconds
    prev_time = curr_time  # Update previous timestamp
    
    # Calculate tilt angles for each sensor using complementary filter
    tilt_x_68, tilt_y_68, tilt_z_68, tilt_xacc_68, tilt_yacc_68, tilt_zacc_68 = calci_tilt_angles(data_68, 1, dt, 0.98)  # Sensor 1
    tilt_x_69, tilt_y_69, tilt_z_69, tilt_xacc_69, tilt_yacc_69, tilt_zacc_69 = calci_tilt_angles(data_69, 2, dt, 0.98)  # Sensor 2
    
    # Group tilt angle values into a list
    tiltlist = [tilt_x_68, tilt_y_68, tilt_z_68, tilt_xacc_68, tilt_yacc_68, tilt_zacc_68,
                tilt_x_69, tilt_y_69, tilt_z_69, tilt_xacc_69, tilt_yacc_69, tilt_zacc_69]
    
    # Group all sensor data into a list, corresponding to their sensor numbers
    datalist = [data_68, data_69]  
    
    # Print all sensor data in CSV format
    # Format: time_stamp, xa, ya, za, xgb, ybg, zbg, tilt_x, y, z, accx, tilt_accy,z, dt, sensor_no
    # Function located in lib_misc.py
    printall(get_timestamp(), datalist, tiltlist, dt)
    
    # Estimated sampling time is approximately 0.22 sec (~50Hz) based on dt observation



