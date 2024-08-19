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
angleX=0
angleY=0
angleZ=[0,0]
avgg_z=[0,0,0]
avgg_x=[0,0,0]
avgg_y=[0,0,0]
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
    #return avg_x, avg_y, avg_z

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


def calci_tilt_angles(data,dtime,alpha=0.98):
    x, y, z = data['accel']['x'], data['accel']['y'], data['accel']['z']
 
    angleAccX = math.atan2(y, math.sqrt(x * x + z * z)) * 180 / math.pi
    angleAccY = - math.atan2(x, math.sqrt(y * y + z * z)) * 180 / math.pi# minus inside bracker
    global angleX
    global angleY
    global angleZ
    
    angleX= wrap(alpha*(angleAccX + wrap(angleX + data['gyro']['x']*dtime - angleAccX,180)) + (1.0-alpha)*angleAccX,180)
    angleY = wrap(alpha * (angleAccY + wrap(angleY + data['gyro']['y'] * dtime - angleAccY, 180)) + (1.0 - alpha) * angleAccY, 180)
    angleZ += data['gyro']['z']*dtime;
 
    return angleX, angleY, angleZ

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
    #print(avgg_z[sensor])
    return {
        'accel': {
            'x': accel_x, #w/out offsets changes
            'y': accel_y,
            'z': accel_z,
        },
        'gyrooooooooooooooooo': {
            'x': gyro_x,
            'y': gyro_y,
            'z': gyro_z,
        },
        'gyro_biased':{
            'x': gyro_x-avgg_x[sensor-1],
            'y': gyro_y-avgg_y[sensor-1],
            'z': gyro_z-avgg_z[sensor-1],
        }   
    }



