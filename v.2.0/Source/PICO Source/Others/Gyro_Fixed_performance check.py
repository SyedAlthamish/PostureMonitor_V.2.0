#Scrap file
'''{
    File Description:(main.py file)
        The purpose of this file is to check the performance of the gyro_spike fix that was implemented.
        It operates in the same order as data acquisition, except it outputs data only relevant to the gyro_values.
    }'''


#importion of libraries
from machine import Pin, I2C																				#for Pin Manipulation, i2c communication
import utime																								#for delay between each loop
import math																									#for tilt calculation from each sensor
from lib_mpu6050 import *
from lib_misc import *

#The setting of MPU6050 sensor's addresses
pin1 = Pin(0, Pin.OUT)			#making pin_no_0 or GPIO_0 as output mode
pin1.value(1)					#making GPIO_0 as HIGH to represent one of the sensors in i2c0 channel as addresses 0x69
utime.sleep(3)                  # some time for the sensors to power up

#Initialization of I2C, MPU, and Data-Processing
i2c = I2C(0, scl=Pin(21), sda=Pin(20), freq=400000)		#initializing i2c0 channel
init_mpu6050(i2c, 0x68)									#initializing first sensor in i2c0		
init_mpu6050(i2c, 0x69)     							#initializing second sensor in i2c0


prev_time = utime.ticks_ms()							#initializing time parameteric for integrating gyro data
#main
while True:
    data_68_fix = get_mpu6050_comprehensive_data_Gyro_Spike_Fix(i2c, 0x68,1)				#a dict with all mpu related data from 1sti2c0 sensor
    data_69_fix = get_mpu6050_comprehensive_data_Gyro_Spike_Fix(i2c, 0x69,2)  			#a dict with all mpu related data from 2ndi2c0 sensor
    #data_681 = get_mpu6050_comprehensive_data(i2c1, 0x68,3)			#a dict with all mpu related data from 1sti2c1 sensor
    
    curr_time = utime.ticks_ms()
    dt = (curr_time - prev_time) / 1000
    prev_time = curr_time
    
        
    tilt_x_68, tilt_y_68, tilt_z_68, tilt_xacc_68, tilt_yacc_68 = calci_tilt_angles(data_68_fix,1, dt, 0.98)
    tilt_x_69, tilt_y_69, tilt_z_69, tilt_xacc_69, tilt_yacc_69 = calci_tilt_angles(data_69_fix,2, dt, 0.98)  # Calculate tilt angles for the second sensor
    #tilt_x_681, tilt_y_681, tilt_z_681, tilt_xacc_681, tilt_yacc_681 = calci_tilt_angles(data_681, 3, dt, 0.98)
    
    tiltlist=[tilt_x_68, tilt_y_68, tilt_z_68, tilt_xacc_68, tilt_yacc_68,tilt_x_69, tilt_y_69, tilt_z_69, tilt_xacc_69, tilt_yacc_69]
    datalist_fixed=[data_68_fix,data_69_fix]								#all 3 sensor's data are grouped together into a list correspondent to their sensor no.	
    
    printall_Gyro(get_timestamp(),datalist_fixed,tiltlist,dt)							#func call to print all the data in a csv format, #prints as time_stamp,xa,ya,za,xgb,ybg,zbg,tilt_x,y,z,accx,tilt_accy,dt,sensor_no ; func located in lib_misc.py
    #estimated sampling time is 0.22 sec or ~50Hz upon observing dt




