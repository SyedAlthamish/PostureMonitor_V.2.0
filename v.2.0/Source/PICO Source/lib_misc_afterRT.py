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
Function: to print data in the following format of output string and input argument for real time integration
    Input: datalist={'accel'{'x','y','z'},'gyro'{'x','y','z'},'gyro_biased'{'x','y','z'}} ; this is the format of datalist returned from function [get_mpu6050_comprehensive_data(...),get_mpu6050_comprehensive_data(...),...]
    Output: prints as ['accx1','accy1','accz1','accx2','accy2','accz2',accx3','accy3','accz3','gyrox1','gyroy1','gyroz1','gyrox2','gyroy2','gyroz2','gyrox3','gyroy3','gyroz3', 'biased_gyrox1','biased_gyroy1','biased_gyroz1','biased_gyrox2','biased_gyroy2','biased_gyroz2','biased_gyrox3','biased_gyroy3',
                        'biased_gyroz3']
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
    Output: prints as 68comp,68compy,68compz,69compx,69compy,69comp
}                
'''    
def printall_Rtime6Comp(time_stamp,datalist,tiltlist,dt):
    
    '''
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
    printlist=tiltlist[0:3]
    printlist.extend(tiltlist[5:8])
    for i in printlist:
        print(i,end='    ')
    print('')
    #print(tiltlist)

'''
Function: to print data in the following format of output string and input argument
    Input: get_timestamp() output[string: MM.SS] , datalist={'accel'{'x','y','z'},'gyro'{'x','y','z'},'gyro_biased'{'x','y','z'}} ; this is the format of datalist returned from function [get_mpu6050_comprehensive_data(...),get_mpu6050_comprehensive_data(...),...]
    Output: prints as time_stamp,xa,ya,za,xgb,ybg,zbg,tilt_x,y,z,accx,tilt_accy,dt,sensor_no for 3 sensors
'''
def printall(time_stamp,datalist,tiltlist,dt):
    datatypes=['accel','gyro_biased']			#the different indexes within datalist
    axes=['x','y','z']
    for i in range(0,len(datalist)):
        print(time_stamp,end=" ")
        for j in datatypes:
            for k in axes:
                print(str(datalist[i][j][k]),end=" ")
        for index in range((0+i*5),((4+i*5)+1)):
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



