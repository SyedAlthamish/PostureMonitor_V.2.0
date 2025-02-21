from machine import Pin, I2C
import utime
import math 
 
PWR_MGMT_1 = 0x6B
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
ACCEL_CONFIG = 0x1C
TEMP_OUT_H = 0x41
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43
angleX=[0,0,0]
angleY=[0,0,0]
angleZ=[0,0,0]

avgg_z=[-3.572,-0.183,0]
avgg_x=[-3.619,-1.090,0]
avgg_y=[0.902,0.699,0]
avga_x=[0,0,0]
avga_y=[0,0,0]
avga_z=[0,0,0]
def init_mpu6050(i2c, address=0x68):
    i2c.writeto_mem(address, PWR_MGMT_1, b'\x01')#0x00
    utime.sleep_ms(100)
    i2c.writeto_mem(address, SMPLRT_DIV, b'\x00')#x07
    i2c.writeto_mem(address, CONFIG, b'\x00')
    i2c.writeto_mem(address, GYRO_CONFIG, b'\x08')#0x00
    i2c.writeto_mem(address, ACCEL_CONFIG, b'\x00')
    
def read_raw_data(i2c, addr, address=0x68):
    high = i2c.readfrom_mem(address, addr, 1)[0]
    low = i2c.readfrom_mem(address, addr + 1, 1)[0]
    value = high << 8 | low
    if value > 32768:
        value = value - 65536
    return value

def variance(data):
    n = len(data)
    if n == 0:
        return 0
    mean = sum(data) / n
    squared_diffs = [(x - mean) ** 2 for x in data]
    return sum(squared_diffs) / (n - 1)  # Sample variance (ddof=1)

def calibrate_gyro(i2c,address,sensor_no,num_samples=500):#1000
    print("Calibrating gyroscope...")
    sumg_x = 0
    sumg_y = 0
    sumg_z = 0
    suma_x = 0
    suma_y = 0
    suma_z = 0
    
    for _ in range(num_samples):
        gyro_x = read_raw_data(i2c, GYRO_XOUT_H, address) / 65.5 #131.0
        gyro_y = read_raw_data(i2c, GYRO_XOUT_H + 2, address) / 65.5
        gyro_z = read_raw_data(i2c, GYRO_XOUT_H + 4, address) / 65.5
        accel_x = read_raw_data(i2c, ACCEL_XOUT_H, address) / 16384.0
        accel_y = read_raw_data(i2c, ACCEL_XOUT_H + 2, address) / 16384.0
        accel_z = read_raw_data(i2c, ACCEL_XOUT_H + 4, address) / 16384.0
        
        sumg_x += gyro_x
        sumg_y += gyro_y
        sumg_z += gyro_z
        suma_x += accel_x
        suma_y += accel_y
        suma_z += accel_z
        
        utime.sleep_ms(10)  #10

    sensor=sensor_no-1
    # Compute average
    global avgg_x, avgg_y, avgg_z, avga_x, avga_y, avga_z
    
    avgg_x[sensor] = sumg_x / num_samples
    avgg_y[sensor] = sumg_y / num_samples
    avgg_z[sensor] = sumg_z / num_samples
    avga_x[sensor] = suma_x / num_samples
    avga_y[sensor] = suma_y / num_samples
    avga_z[sensor] = suma_z / num_samples

    print("Gyroscope calibration complete.",avgg_x[sensor],avgg_y[sensor],avgg_z[sensor])
    #print("Average values: X: {:.2f}, Y: {:.2f}, Z: {:.2f}".format(avgg_x, avgg_y, avg_z))
    return avgg_x[sensor],avgg_y[sensor],avgg_z[sensor]

def calibrate_gyroonlyZ(i2c, address,sensor, num_samples=1000):
    print("Calibrating gyroscope...")
    sumg_z = 0
    
    for _ in range(num_samples):
        gyro_z = read_raw_data(i2c, GYRO_XOUT_H + 4, address) / 65.5
        sumg_z += gyro_z
        utime.sleep_ms(10)  # Adjust sleep time if needed

    # Compute average
    global avgg_z
    avgg_z[sensor] = sumg_z / num_samples

    print("Gyroscope Z-axis calibration complete.")

def calci_tilt_angles(data,sensor_no,dtime,alpha=0.98):
    x, y, z = data['accel']['x'], data['accel']['y'], data['accel']['z']
 
    angleAccX = math.atan2(y, math.sqrt(x * x + z * z)) * 180 / math.pi
    angleAccY = - math.atan2(x, math.sqrt(y * y + z * z)) * 180 / math.pi# minus inside bracker
    global angleX
    global angleY
    global angleZ
    
    sensor_index=sensor_no-1
    
    angleX[sensor_index]= wrap(alpha*(angleAccX + wrap(angleX[sensor_index] + data['gyro_biased_fixed']['x']*dtime - angleAccX,180)) + (1.0-alpha)*angleAccX,180)				#the explanation is given below
    angleY[sensor_index] = wrap(alpha * (angleAccY + wrap(angleY[sensor_index] + data['gyro_biased_fixed']['y'] * dtime - angleAccY, 180)) + (1.0 - alpha) * angleAccY, 180)
    angleZ[sensor_index] = wrap(angleZ[sensor_index] + data['gyro_biased_fixed']['z'] * dtime, 180)
 
    return angleX[sensor_index], angleY[sensor_index], angleZ[sensor_index],angleAccX,angleAccY

def calci_tilt_accangles(data, dtime, sensor, alpha=0.98):
    x, y, z = data['accel']['x'], data['accel']['y'], data['accel']['z']
    
    angleAccX = math.atan2(y, math.sqrt(x * x + z * z)) * 180 / math.pi
    angleAccY = - math.atan2(x, math.sqrt(y * y + z * z)) * 180 / math.pi
    angleAccZ = math.atan2(math.sqrt(x * x + y * y), z) * 180 / math.pi
    
    global angleZ
    angleZ[sensor] += data['gyro']['z']*dtime;
    
    
    angleX = wrap(alpha * angleAccX + (1.0 - alpha) * angleAccX, 180)
    angleY = wrap(alpha * angleAccY + (1.0 - alpha) * angleAccY, 180)
    #angleZ = wrap(alpha * angleAccZ + (1.0 - alpha) * angleAccZ, 180)
    
    return angleAccX, angleAccY, angleZ[sensor]

def wrap(angle, limit):
    while angle > limit:
        angle -= 2 * limit
    while angle < -limit:
        angle += 2 * limit
    return angle

def get_mpu6050_data(i2c, address,sensor):
    #temp = read_raw_data(i2c, TEMP_OUT_H, address) / 340.0 + 36.53
    accel_x = read_raw_data(i2c, ACCEL_XOUT_H, address) / 16384.0
    accel_y = read_raw_data(i2c, ACCEL_XOUT_H + 2, address) / 16384.0
    accel_z = read_raw_data(i2c, ACCEL_XOUT_H + 4, address) / 16384.0
    gyro_x = read_raw_data(i2c, GYRO_XOUT_H, address) / 65.5#131.0
    gyro_y = read_raw_data(i2c, GYRO_XOUT_H + 2, address) / 65.5
    gyro_z = read_raw_data(i2c, GYRO_XOUT_H + 4, address) / 65.5
    
    
    global avgg_z
    #print(avgg_z[sensor])
    return {
        'accel': {
            'x': accel_x, #w/out offsets changes
            'y': accel_y,
            'z': accel_z,
        },
        'gyro': {
            'x': gyro_x,
            'y': gyro_y,
            'z': gyro_z,
        }
    }

def get_mpu6050_comprehensive_data(i2c, address,sensor):
    #temp = read_raw_data(i2c, TEMP_OUT_H, address) / 340.0 + 36.53
    accel_x = read_raw_data(i2c, ACCEL_XOUT_H, address) / 16384.0
    accel_y = read_raw_data(i2c, ACCEL_XOUT_H + 2, address) / 16384.0
    accel_z = read_raw_data(i2c, ACCEL_XOUT_H + 4, address) / 16384.0
    gyro_x = read_raw_data(i2c, GYRO_XOUT_H, address) / 65.5#131.0
    gyro_y = read_raw_data(i2c, GYRO_XOUT_H + 2, address) / 65.5
    gyro_z = read_raw_data(i2c, GYRO_XOUT_H + 4, address) / 65.5
    
    
    global avgg_z,avgg_x,avgg_y
    
    gyro_bx = gyro_x-avgg_x[sensor-1]
    gyro_by = gyro_y-avgg_y[sensor-1]
    gyro_bz = gyro_z-avgg_z[sensor-1]
    
     
    #print(avgg_z[sensor])
    return {
        'accel': {
            'x': accel_x, #w/out offsets changes
            'y': accel_y,
            'z': accel_z,
        },
        'gyro': {
            'x': gyro_x,
            'y': gyro_y,
            'z': gyro_z,
        },
        'gyro_biased':{
            'x': gyro_bx,
            'y': gyro_by,
            'z': gyro_bz,
        }   
    }

tri_sample_data = [     # [ax, ay, az, gx, gy, gz, gbx, gby, gbz] - prev
                        # [ax, ay, az, gx, gy, gz, gbx, gby, gbz] - current  
                        # [ax, ay, az, gx, gy, gz, gbx, gby, gbz] - new
                        # for all 3 sensors
    # First sensor samples
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    # Second sensors samples
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    # Third sensor samples
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
]
prev_index = 0
curr_index = 1
new_index = 2
gyrob_index_start = 6
gyrob_index_end = 8
def get_mpu6050_comprehensive_data_Gyro_Spike_Fix(i2c, address,sensor):
    #temp = read_raw_data(i2c, TEMP_OUT_H, address) / 340.0 + 36.53
    sensor_index = sensor - 1 
    accel_x = read_raw_data(i2c, ACCEL_XOUT_H, address) / 16384.0
    accel_y = read_raw_data(i2c, ACCEL_XOUT_H + 2, address) / 16384.0
    accel_z = read_raw_data(i2c, ACCEL_XOUT_H + 4, address) / 16384.0
    gyro_x = read_raw_data(i2c, GYRO_XOUT_H, address) / 65.5#131.0
    gyro_y = read_raw_data(i2c, GYRO_XOUT_H + 2, address) / 65.5
    gyro_z = read_raw_data(i2c, GYRO_XOUT_H + 4, address) / 65.5
  
    global avgg_z,avgg_x,avgg_y
    
    gyro_bx = gyro_x-avgg_x[sensor_index]
    gyro_by = gyro_y-avgg_y[sensor_index]
    gyro_bz = gyro_z-avgg_z[sensor_index]
    
    tri_sample_data[sensor_index][new_index] = [accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,gyro_bx,gyro_by,gyro_bz]
    
    for i in range (gyrob_index_start,gyrob_index_end + 1): #cycling through all axes
        if math.floor(abs(tri_sample_data[sensor_index][new_index][i])) == 0 and math.floor(abs(tri_sample_data[sensor_index][prev_index][i])) == 0 and math.floor(abs(tri_sample_data[sensor_index][curr_index][i])) != 0:
            tri_sample_data[sensor_index][curr_index][i] = 0
    
    gyro_bx_f = tri_sample_data[sensor_index][curr_index][gyrob_index_start]
    gyro_by_f = tri_sample_data[sensor_index][curr_index][gyrob_index_start+1]
    gyro_bz_f = tri_sample_data[sensor_index][curr_index][gyrob_index_end]    
    
    tri_sample_data[sensor_index][prev_index] =  tri_sample_data[sensor_index][curr_index]
    tri_sample_data[sensor_index][curr_index] =  tri_sample_data[sensor_index][new_index]
    
    
    
    
    #print(avgg_z[sensor])
    return {
        'accel': {
            'x': accel_x, #w/out offsets changes
            'y': accel_y,
            'z': accel_z,
        },
        'gyro': {
            'x': gyro_x,
            'y': gyro_y,
            'z': gyro_z,
        },
        'gyro_biased':{
            'x': gyro_bx,
            'y': gyro_by,
            'z': gyro_bz,
        },
        'gyro_biased_fixed':{
            'x': gyro_bx_f,
            'y': gyro_by_f,
            'z': gyro_bz_f,
        }   
    }


def calibrate_checkgyro(i2c,address,sensor_no,num_samples=200):#1000
    gyroxlist=[]
    gyroylist=[]
    gyrozlist=[]
    for i in range(num_samples):
        data=get_mpu6050_comprehensive_data(i2c,address,sensor_no)
        gyro_x = data['gyro_biased']['x']
        gyro_y = data['gyro_biased']['y']
        gyro_z = data['gyro_biased']['z']
        gyroxlist.append(gyro_x)
        gyroylist.append(gyro_y)
        gyrozlist.append(gyro_z)
    
    sensor_index=sensor_no-1
    sensor_var_threshold=[[1.2,1.2,1.2],[1.2,1.2,1.2],[1.2,1.2,1.2]]
    
    if (variance(gyroxlist) < sensor_var_threshold[sensor_index][0] and variance(gyroylist) < sensor_var_threshold[sensor_index][1] and variance(gyrozlist) < sensor_var_threshold[sensor_index][2]):
        print("variance checks out", variance(gyroxlist),variance(gyroylist),variance(gyrozlist))
    else:
        print("variance doesn't check out")
        print(variance(gyroxlist),variance(gyroylist),variance(gyrozlist) )



'''
The Line in Question:
python
Copy code
angleX = wrap(alpha * (angleAccX + wrap(angleX + data['gyro']['x'] * dtime - angleAccX, 180)) + (1.0 - alpha) * angleAccX, 180)
Purpose:
This line updates the angleX, which represents the pitch angle, by combining data from the accelerometer and gyroscope using a complementary filter. However, it does so with additional complexity to handle potential drift and error correction.

Breakdown:
Basic Structure:

python
Copy code
angleX = wrap(
    alpha * (gyro-based term + correction) + (1.0 - alpha) * accel-based term, 
    180
)
The formula combines two main components:

A gyro-based term: The angular rate data from the gyroscope integrated over time to estimate the angle.
An accel-based term: The direct angle estimate from the accelerometer.
alpha: Controls the balance between the gyro and accelerometer data.

wrap(): Ensures the angle remains within a specific range, typically [-180, 180] degrees.

Gyro-Based Term:

python
Copy code
gyro-based term = angleAccX + wrap(angleX + data['gyro']['x'] * dtime - angleAccX, 180)
data['gyro']['x'] * dtime: This part integrates the gyroscope's rate of rotation (in degrees per second) over the time interval dtime to estimate the angle change.
angleX + data['gyro']['x'] * dtime: Adds the new estimated angle change to the previous angle (angleX).
- angleAccX: Subtracts the accelerometer's angle to correct the drift that accumulates when integrating gyroscope data over time.
wrap(..., 180): Ensures the corrected gyro-based angle remains within [-180, 180] degrees.
angleAccX + ...: Adds the accelerometer's direct angle to the corrected gyro-based angle. This suggests that the gyro-based term is being continuously corrected by the accelerometer's estimate.
Weighted Combination with Accel-Based Term:

python
Copy code
alpha * (gyro-based term) + (1.0 - alpha) * angleAccX
alpha * (gyro-based term): Multiplies the gyro-based term (now corrected by the accelerometer) by alpha. This gives more weight to the gyroscope's data for rapid, short-term changes.
(1.0 - alpha) * angleAccX: Multiplies the accelerometer's angle by (1.0 - alpha), giving it more influence for long-term stability.
Final Wrapping:

python
Copy code
wrap(..., 180)
The entire expression is passed through the wrap function again to ensure the final angleX remains within [-180, 180] degrees.
What Makes This More Than a Standard Complementary Filter?
Error Correction Mechanism:

The line includes an additional correction term: wrap(angleX + data['gyro']['x'] * dtime - angleAccX, 180). This corrects the gyro-based estimate by subtracting the accelerometer's direct angle estimate. It helps mitigate the drift that typically occurs when integrating gyroscope data over time.
Nested Wrapping:

The use of wrap within the gyro-based term suggests that the angle is being carefully controlled to avoid issues like overflow or excessive drift. This ensures that the angle doesn't exceed the expected range, which could lead to errors in angle calculation.
Continuous Adjustment:

The formula continuously adjusts the gyroscope's estimate by comparing it to the accelerometer's reading, rather than simply blending the two data sources. This makes it more robust against the drift that can occur in gyroscope data over time.
Summary:
This line is indeed more sophisticated than a typical complementary filter. It incorporates a correction mechanism to adjust the gyroscope's angle estimate using the accelerometer's direct angle, which is then wrapped to stay within a valid range. This approach aims to combine the strengths of both sensors: the stability of the accelerometer and the responsiveness of the gyroscope, while actively correcting for drift and maintaining accuracy over time.
'''



