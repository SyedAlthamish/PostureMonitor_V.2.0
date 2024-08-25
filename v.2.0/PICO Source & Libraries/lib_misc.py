'''
{
    This Library Contains the following functions:
        1. printall(datalist): to csv-print all necessary mpu data
        2. calcianglealone(data): to return tilt from Accelerometer data
        3. get_timestamp(): returns current local time in MM.SS

}
'''


import utime
import sys


'''
Function: to print data in the following format of output string and input argument
    Input: get_timestamp() output[string: MM.SS] , datalist={'accel'{'x','y','z'},'gyro'{'x','y','z'},'gyro_biased'{'x','y','z'}} ; this is the format of datalist returned from function [get_mpu6050_comprehensive_data(...),get_mpu6050_comprehensive_data(...),...]
    Output: MM.SS,ax,ay,az,gx,gy,gz,gbx,gby,gbz,1
            MM.SS,ax,ay,az,gx,gy,gz,gbx,gby,gbz,2
            MM.SS,ax,ay,az,gx,gy,gz,gbx,gby,gbz,3				; where gbx is gyro_biased data of x axis, and 1/2/3 is the sensor_no who's data is being printed
'''
def printall(time_stamp,datalist):
    datatypes=['accel','gyro','gyro_biased']			#the different indexes within datalist
    axes=['x','y','z']
    for i in range(0,len(datalist)):
        for j in datatypes:
            for k in axes:
                print(time_stamp+str(datalist[i][j][k]),end=" ")
        print(i+1)
        utime.sleep(0.01)            



'''
Function: to acquire tilt data only with the ACCeleromenter data 
'''
def calcianglealone(data):
    x, y, z = data['accel']['x'], data['accel']['y'], data['accel']['z']
    angleAccX = math.atan2(y, math.sqrt(x * x + z * z)) * 180 / math.pi
    angleAccY = - math.atan2(x, math.sqrt(y * y + z * z)) * 180 / math.pi
    angleAccZ = math.atan2(math.sqrt(x * x + y * y), z) * 180 / math.pi
    return angleAccX,angleAccY,angleAccZ



'''
Function: to get the current local time from the MC
Output: MM.SS in string format
'''
def get_timestamp():
    current_time = utime.localtime()								# Get the current time
    return "{:02}.{:02} ".format(current_time[4], current_time[5])	# Format the time as [MM.SS]


