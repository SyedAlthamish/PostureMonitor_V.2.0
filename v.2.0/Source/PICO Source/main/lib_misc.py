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


def printall_Rtime(time_stamp, datalist, tiltlist, dt):
    '''
    {
    Function: to print data in the following format of output string and input argument for real-time integration
        Input:
            datalist={'accel'{'x','y','z'},'gyro'{'x','y','z'},'gyro_biased'{'x','y','z'}} 
            This is the format of datalist returned from function 
            [get_mpu6050_comprehensive_data(...), get_mpu6050_comprehensive_data(...),...]
        Output: 
            Prints data as:
            ['accx1','accy1','accz1','accx2','accy2','accz2','accx3','accy3','accz3',
             'gyrox1','gyroy1','gyroz1','gyrox2','gyroy2','gyroz2','gyrox3','gyroy3','gyroz3',
             'biased_gyrox1','biased_gyroy1','biased_gyroz1','biased_gyrox2','biased_gyroy2',
             'biased_gyroz2','biased_gyrox3','biased_gyroy3','biased_gyroz3']
    }                
    '''
    
    datatypes = ['accel', 'gyro', 'gyro_biased']  # List of data categories in datalist
    axes = ['x', 'y', 'z']  # The three measurement axes
    printlist = []  # List to store formatted strings for printing

    # Iterate over each data type ('accel', 'gyro', 'gyro_biased')
    for j in datatypes:
        # Iterate over the number of sensors (length of datalist)
        for i in range(0, len(datalist)):
            # Iterate over the three axes
            for k in axes:
                datapoint = datalist[i][j][k]  # Extract the specific data value
                # Append the formatted string to printlist
                printlist.append(str(datapoint) + str(i) + str(j) + str(k) + "            ")

    # Print all values from printlist in a single line
    for f in printlist:
        print(f, end="")

    print("\n", end="n")  # Print newline at the end

    

def printall_Rtime6Comp(time_stamp, datalist, tiltlist, dt):
    '''
    {
    Function: to print data in the following format of output string and input argument 
              for real-time integration with ML model running on a PC.
        Input:
            - time_stamp
            - datalist={'accel'{'x','y','z'},'gyro'{'x','y','z'},'gyro_biased'{'x','y','z'}}
            - tiltlist
            - dt
        Output:
            Prints only complimentary tilt values from 2 sensors_discarding acc tilt values.
    }
    '''
    printlist = tiltlist[0:3]                  # Extracts first 3 tilt values
    printlist.extend(tiltlist[5:8])            # Extracts tilt values from indices 5 to 7
    for i in printlist:                        
        print(i, end='    ')                   # Prints extracted values with spacing
    print()                                    # Prints newline


def printall_Gyro(time_stamp, datalist_fixed, tiltlist, dt):
    '''
    {
    Function: to print gyro data in a formatted output for checking its performance, currently after the spike fix.
        Input:
            - time_stamp
            - datalist_fixed={'gyro_biased', 'gyro_biased_fixed'}
            - tiltlist
            - dt
        Output:
            Prints formatted gyro data.
    }
    '''
    datatypes = ['gyro_biased', 'gyro_biased_fixed']   # Different gyro data types
    axes = ['x', 'y', 'z']                             # Measurement axes
    printlist = []                                     # List to store formatted values

    for j in datatypes:                                # Loop through data types
        for i in range(len(datalist_fixed)):          # Loop through each sensor
            for k in axes:                             # Loop through each axis
                datapoint = datalist_fixed[i][j][k]    # Extract data point
                printlist.append(str(datapoint) + "    ") # Append formatted value

    for f in printlist:                               # Loop through formatted values
        print(f, end="")                              # Print without newline each of the values
    print("\n", end="")                               # Print newline for the next sampple of values


def printall(time_stamp, datalist, tiltlist, dt):
    '''{
    Function: to print sensor data along with tilt angles in a structured format.
        Input:
            1. time_stamp= {get_timestamp() output[string: MM.SS]}
            2. datalist= {'accel'{'x','y','z'},'gyro'{'x','y','z'},'gyro_biased'{'x','y','z'}}
            3. tiltlist=[tilt_x_68, tilt_y_68, tilt_z_68, tilt_xacc_68, tilt_yacc_68, 
                        tilt_x_69, tilt_y_69, tilt_z_69, tilt_xacc_69, tilt_yacc_69]
            4. dt = integer/float
        Output:
            Prints formatted sensor and tilt data per sensor.
    }'''
    
    n_of_sen = len(datalist)                          # Number of sensors used
    tilts_p_sen = int(len(tiltlist) / n_of_sen)       # Tilt variables per sensor
    datatypes = ['accel', 'gyro_biased_fixed']        # Data types to be extracted from container var in datalist
    axes = ['x', 'y', 'z']                            # Measurement axes

    for i in range(n_of_sen):                         # Loop through sensors
        print(time_stamp, end=" ")                    # Print timestamp
        
        for j in datatypes:                           # Loop through data types
            for k in axes:                            # Loop through each axis
                print(str(datalist[i][j][k]), end=" ") # Print sensor data
        
        for index in range((0 + i * tilts_p_sen), (tilts_p_sen + i * tilts_p_sen)):  
            print(tiltlist[index], end=" ")           # Print corresponding tilt data
        
        print(dt, end=" ")                            # Print time difference
        print(i + 1)                                  # Print sensor number while moving to the next sample of data


def calcianglealone(data):
    '''{
    Function: to compute tilt angles using only accelerometer data.
    }'''
    x, y, z = data['accel']['x'], data['accel']['y'], data['accel']['z']  # Extract acceleration data
    # Formulas in accordance with r.papers
    angleAccX = math.atan2(y, math.sqrt(x * x + z * z)) * 180 / math.pi    # Compute tilt angle in X
    angleAccY = - math.atan2(x, math.sqrt(y * y + z * z)) * 180 / math.pi  # Compute tilt angle in Y
    angleAccZ = math.atan2(math.sqrt(x * x + y * y), z) * 180 / math.pi    # Compute tilt angle in Z
    return angleAccX, angleAccY, angleAccZ                                 # Return computed angles


def get_timestamp():
    '''
    Function: to get the current local time from the microcontroller.
    Output: MM.SS in string format.
    '''
    current_time = utime.localtime()                              # Get the current local time
    return "{:02}.{:02} ".format(current_time[4], current_time[5])  # Format as MM.SS string





