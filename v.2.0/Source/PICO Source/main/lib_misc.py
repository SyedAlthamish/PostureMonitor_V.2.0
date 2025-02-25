'''{
    File Description:
        This file contains functions that support the operation of the PICO to handle the data acquired from the sensors,
        including its transmitting to the PC
    
    This Library Contains the following functions:
        1. printall(datalist): to csv-print all necessary mpu data
        2. calcianglealone(data): to return tilt from Accelerometer data
        3. get_timestamp(): returns current local time in MM.SS
        4. and etc...

    }'''


import utime
import sys

'''
{
Function: to print data in the following format of output string and input argument for real time integration
    Input:
        datalist={'accel'{'x','y','z'},'gyro'{'x','y','z'},'gyro_biased'{'x','y','z'}} ; this is the format of datalist returned from function [get_mpu6050_comprehensive_data(...),get_mpu6050_comprehensive_data(...),...]
    Output: prints as ['accx1','accy1','accz1','accx2','accy2','accz2',accx3','accy3','accz3','gyrox1','gyroy1','gyroz1','gyrox2','gyroy2','gyroz2','gyrox3','gyroy3','gyroz3', 'biased_gyrox1','biased_gyroy1','biased_gyroz1','biased_gyrox2','biased_gyroy2','biased_gyroz2','biased_gyrox3','biased_gyroy3',
                        'biased_gyroz3']
}                
'''
def printall_Rtime(time_stamp,datalist,tiltlist,dt):
    
    
    datatypes=['accel','gyro','gyro_biased']			#the different indexes within datalist
    axes=['x','y','z']
    printlist=[]
    for j in datatypes:
        for i in range(0,len(datalist)):
            for k in axes:
                    datapoint=datalist[i][j][k]
                    #print(str(datapoint),end=" ")
                    printlist.append(str(datapoint)+str(i)+str(j)+str(k)+"            ")
    #print(printlist)
    for f in (printlist):
        print(f,end="")
    print("\n",end="n")
    
'''
{
Function: to print data in the following format of output string and input argument for real time integration with ML model running in PC
    Input: time_stamp,datalist={'accel'{'x','y','z'},'gyro'{'x','y','z'},'gyro_biased'{'x','y','z'}},tiltlist,dt ; this is the format of datalist returned from function [get_mpu6050_comprehensive_data(...),get_mpu6050_comprehensive_data(...),...]
    Output: prints as ['accx1','accy1','accz1','accx2','accy2','accz2',accx3','accy3','accz3','gyrox1','gyroy1','gyroz1','gyrox2','gyroy2','gyroz2','gyrox3','gyroy3','gyroz3', 'biased_gyrox1','biased_gyroy1','biased_gyroz1','biased_gyrox2','biased_gyroy2','biased_gyroz2','biased_gyrox3','biased_gyroy3',
                        'biased_gyroz3']
}                
'''
def printall_Rtime6Comp(time_stamp,datalist,tiltlist,dt):
    '''{
        File
        }'''
    
    printlist=tiltlist[0:3]
    printlist.extend(tiltlist[5:8])
    for i in printlist:
        print(i,end='    ')
    print()
    #print('Postural State: Uncalibrated')
    #print(tiltlist)



'''
{
Function: to print data in the following format of output string and input argument for real time integration with ML model running in PC
    Input: time_stamp,datalist={'accel'{'x','y','z'},'gyro'{'x','y','z'},'gyro_biased'{'x','y','z'}},tiltlist,dt ; this is the format of datalist returned from function [get_mpu6050_comprehensive_data(...),get_mpu6050_comprehensive_data(...),...]
    Output: prints as ['accx1','accy1','accz1','accx2','accy2','accz2',accx3','accy3','accz3','gyrox1','gyroy1','gyroz1','gyrox2','gyroy2','gyroz2','gyrox3','gyroy3','gyroz3', 'biased_gyrox1','biased_gyroy1','biased_gyroz1','biased_gyrox2','biased_gyroy2','biased_gyroz2','biased_gyrox3','biased_gyroy3',
                        'biased_gyroz3']
}                
'''
def printall_Gyro(time_stamp,datalist_fixed,tiltlist,dt):
    
    
    datatypes=['gyro_biased','gyro_biased_fixed']			#the different indexes within datalist
    axes=['x','y','z']
    printlist=[]
    for j in datatypes:
        for i in range(0,len(datalist_fixed)):
            for k in axes:
                    datapoint=datalist_fixed[i][j][k]
                    #print(str(datapoint),end=" ")
                    printlist.append(str(datapoint)+"    ")
    #print(printlist)
    for f in (printlist):
        print(f,end="")
    print("\n",end="")
    '''
    gyro_beg_index = 18
    printlist=tiltlist[0:3]
    printlist.extend(tiltlist[5:8])
    for i in printlist[gyro_beg_index:]:
        print(i,end='    ')
    print()
    #print('Postural State: Uncalibrated')
    #print(tiltlist)
    '''


'''
Function: to print data in the following format of output string and input argument
    Input:
            1. time_stamp= {get_timestamp() output[string: MM.SS]}
            2. datalist= {'accel'{'x','y','z'},'gyro'{'x','y','z'},'gyro_biased'{'x','y','z'}}
            3. Tiltlist=[tilt_x_68, tilt_y_68, tilt_z_68, tilt_xacc_68, tilt_yacc_68,tilt_x_69, tilt_y_69, tilt_z_69, tilt_xacc_69, tilt_yacc_69]
            4. dt = integer/float
    Output:
            Prints as
                   "time_stamp,xa,ya,za,xgb,ybg,zbg,tilt_x,y,z,tilt_accx,tilt_accy,dt,sensor_no1
                    time_stamp,xa,ya,za,xgb,ybg,zbg,tilt_x,y,z,tilt_accx,tilt_accy,dt,sensor_no2
                    time_stamp,xa,ya,za,xgb,ybg,zbg,tilt_x,y,z,tilt_accx,tilt_accy,dt,sensor_no3"
            for 3 sensors.
'''
def printall(time_stamp,datalist,tiltlist,dt):
    n_of_sen = len(datalist)  # the number of sensors used
    tilts_p_sen = int(len(tiltlist) / n_of_sen) #the total no of tilt variables per sensor
    datatypes=['accel','gyro_biased_fixed']			#the different indexes within datalist
    axes=['x','y','z']
    for i in range(0,n_of_sen):
        print(time_stamp,end=" ")
        for j in datatypes:
            for k in axes:
                print(str(datalist[i][j][k]),end=" ")
        for index in range((0+i*tilts_p_sen),(tilts_p_sen+i*tilts_p_sen)):
            print(tiltlist[index],end=" ")
        print(dt,end=" ")            
        print(i+1)            



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




