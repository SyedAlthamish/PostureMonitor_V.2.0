'''
File Name: calibration_final.py

Purpose:This file runs the calibration for 3 MPUs and checks its calibration performance
The following comment is invalid

Functional Description:-
    1. once initialized, the mpu sensor's gyro data is taken as input for a set no of samples.
    2. These values are averaged and subtracted from the raw gyro outputs from mpu when ever reading from it
    3. After calibration another algorithm checks for variation and mean in the stable biased output of the MPUs to measure efficacy of calibration routine
    
Note:
    - The gbz values or gyro_biased_z values have an abnormality where sudden unpredictable spikes from ~0 to ~ +-3.8 occurs. The matter hasn't been resolved.
'''

#importion of libraries
from machine import Pin, I2C																				#for Pin Manipulation, i2c communication
import utime																								#for delay between each loop
import math																									#for tilt calculation from each sensor
from lib_mpu6050 import init_mpu6050, get_mpu6050_data, calibrate_gyro, calci_tilt_accangles,get_mpu6050_comprehensive_data,calibrate_checkgyro		#for individual user-defined functions
from lib_misc import *

#The setting of MPU6050 sensor's addresses
pin1 = Pin(0, Pin.OUT)			#making pin_no_0 or GPIO_0 as output mode
pin1.value(1)					#making GPIO_0 as HIGH to represent one of the sensors in i2c0 channel as addresses 0x69


#Initialization of I2C, MPU, and Data-Processing
i2c = I2C(0, scl=Pin(21), sda=Pin(20), freq=400000)		#initializing i2c0 channel
i2c1 = I2C(1, scl=Pin(19), sda=Pin(18), freq=400000)	#initializing i2c1 channel
init_mpu6050(i2c, 0x68)									#initializing first sensor in i2c0		
init_mpu6050(i2c, 0x69)     							#initializing second sensor in i2c0
init_mpu6050(i2c1,0x68)									#initializing first sensor in i2c1		

#calibration routine for gyros
calibrate_gyro(i2c,0x68,1)
calibrate_gyro(i2c,0x69,2)
calibrate_gyro(i2c1,0x68,3)

#calibration performance check for gyros
calibrate_checkgyro(i2c,0x68,1)
calibrate_checkgyro(i2c,0x69,2)
calibrate_checkgyro(i2c1,0x68,3)


prev_time = utime.ticks_ms()							#initializing time parameteric for integrating gyro data
#main
while True:
    data_68 = get_mpu6050_comprehensive_data(i2c, 0x68,1)				#a dict with all mpu related data from 1sti2c0 sensor
    data_69 = get_mpu6050_comprehensive_data(i2c, 0x69,2)  			#a dict with all mpu related data from 2ndi2c0 sensor
    data_681 = get_mpu6050_comprehensive_data(i2c1, 0x68,3)			#a dict with all mpu related data from 1sti2c1 sensor
    datalist=[data_68,data_69,data_681]								#all 3 sensor's data are grouped together into a list correspondent to their sensor no.
    
    printall(datalist)							#func call to print all the data in a csv format, #prints as xa,ya,za,xg,yg,zg,xgb,ybg,zbg ; func located in lib_misc.py
    utime.sleep(0.01)							#to sample data at 0.01 sec per sample or 100Hz

