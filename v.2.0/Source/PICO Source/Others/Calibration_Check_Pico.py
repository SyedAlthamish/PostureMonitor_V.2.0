'''{
    File Description: 
        The purpose of this file is to estimate the performance of pre-calibrated system IDLE system performance check, where
        the system is calibrated before hand and the data necessary for the performance estimation was extracted for 
        further analysis
    }'''

#importion of libraries
from machine import Pin, I2C																					# for Pin Manipulation, i2c communication
import utime																									# for delay between each loop
import math																									# for tilt calculation from each sensor
from lib_mpu6050 import init_mpu6050, get_mpu6050_data, calibrate_gyro, calci_tilt_angles, get_mpu6050_comprehensive_data, calibrate_checkgyro		# for individual user-defined functions
from lib_misc import *																							# Import miscellaneous functions

# The setting of MPU6050 sensor's addresses
pin1 = Pin(0, Pin.OUT)			# making pin_no_0 or GPIO_0 as output mode
pin1.value(1)					# making GPIO_0 as HIGH to represent one of the sensors in i2c0 channel as address 0x69
utime.sleep(3)					# Wait for 3 seconds for sensor's to power-up

### Initialization of I2C, MPU, and Data-Processing
i2c = I2C(0, scl=Pin(21), sda=Pin(20), freq=400000)		# initializing i2c0 channel
init_mpu6050(i2c, 0x68)									# initializing first sensor in i2c0		
init_mpu6050(i2c, 0x69)     								# initializing second sensor in i2c0

### Variable Initialization
no_of_calibrations = 20										# Set number of calibration routines 
avggx1_list, avggy1_list, avggz1_list = [], [], []		# sensor 1 calibration data - they hold avg of x,y,z value for all iterations
avggx2_list, avggy2_list, avggz2_list = [], [], []		# sensor 2 calibration data - they hold avg of x,y,z value for all iterations

### Performs calibration for no_of_calibration times to record data
for i in range(no_of_calibrations):  										
    # Calibration routine for gyros
    avgg_x1, avgg_y1, avgg_z1 = calibrate_gyro(i2c, 0x68, 1)	# Calibrate gyro for sensor 1
    avgg_x2, avgg_y2, avgg_z2 = calibrate_gyro(i2c, 0x69, 2)	# Calibrate gyro for sensor 2
    print(f"{i} th iteration done")						# Print iteration completion status
    # Append values to respective lists
    avggx1_list.append(avgg_x1)							# Append X-axis calibration data for sensor 1
    avggy1_list.append(avgg_y1)							# Append Y-axis calibration data for sensor 1
    avggz1_list.append(avgg_z1)							# Append Z-axis calibration data for sensor 1
    avggx2_list.append(avgg_x2)							# Append X-axis calibration data for sensor 2
    avggy2_list.append(avgg_y2)							# Append Y-axis calibration data for sensor 2
    avggz2_list.append(avgg_z2)							# Append Z-axis calibration data for sensor 2

### Displaying the Calibration data for the user
print("\nSensor 1 Calibration Data:")						# Print sensor 1 calibration results
print("X:", avggx1_list)									# Print X-axis data for sensor 1
print("Y:", avggy1_list)									# Print Y-axis data for sensor 1
print("Z:", avggz1_list)									# Print Z-axis data for sensor 1
print("\nSensor 2 Calibration Data:")						# Print sensor 2 calibration results
print("X:", avggx2_list)									# Print X-axis data for sensor 2
print("Y:", avggy2_list)									# Print Y-axis data for sensor 2
print("Z:", avggz2_list)									# Print Z-axis data for sensor 2

while(1):													# Infinite loop to keep the program running
    continue												# Do nothing, just keep running