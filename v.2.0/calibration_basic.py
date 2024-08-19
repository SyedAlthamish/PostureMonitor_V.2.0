'''
File Name: calibration_basic.py
Purpose:This file runs the calibration and checks its performance
The following comment is invalid
Functional Description:-
    1. The i2c pins(both i2c0 and i2c1 channels) are first initialized.
    2. The mpu6050's addressess are accordingly set.
    3. Each MPU is sequentially initialized.
    4. Data is Acquired from each sensor infinitely.
    5. Tilt is acquired from each data point and then printed in the console.
'''

#importion of libraries
from machine import Pin, I2C																				#for Pin Manipulation, i2c communication
import utime																								#for delay between each loop
import math																									#for tilt calculation from each sensor
from mpu6050 import init_mpu6050, get_mpu6050_data, calibrate_gyro, calci_tilt_accangles,get_mpu6050_comprehensive_data		#for individual user-defined functions


#The setting of MPU6050 sensor's addresses
pin1 = Pin(0, Pin.OUT)			#making pin_no_0 or GPIO_0 as output mode
pin1.value(1)					#making GPIO_0 as HIGH to represent one of the sensors in i2c0 channel as addresses 0x69


#Initialization of I2C, MPU, and Data-Processing
i2c = I2C(0, scl=Pin(21), sda=Pin(20), freq=400000)		#initializing i2c0 channel
i2c1 = I2C(1, scl=Pin(19), sda=Pin(18), freq=400000)	#initializing i2c1 channel
init_mpu6050(i2c, 0x68)									#initializing first sensor in i2c0		
init_mpu6050(i2c, 0x69)     							#initializing second sensor in i2c0
init_mpu6050(i2c1,0x68)									#initializing first sensor in i2c1		
prev_time = utime.ticks_ms()							#initializing time parameteric for integrating gyro data

#calibration of the sensor

def calcianglealone(data):
    x, y, z = data['accel']['x'], data['accel']['y'], data['accel']['z']
    angleAccX = math.atan2(y, math.sqrt(x * x + z * z)) * 180 / math.pi
    angleAccY = - math.atan2(x, math.sqrt(y * y + z * z)) * 180 / math.pi
    angleAccZ = math.atan2(math.sqrt(x * x + y * y), z) * 180 / math.pi
    return angleAccX,angleAccY,angleAccZ

calibrate_gyro(i2c,0x68,1)
calibrate_gyro(i2c,0x69,2)
calibrate_gyro(i2c1,0x68,3)

while True:
    data_68 = get_mpu6050_comprehensive_data(i2c, 0x68,1)				#a dict with all mpu related data from 1sti2c0 sensor
    data_69 = get_mpu6050_comprehensive_data(i2c, 0x69,2)  			#a dict with all mpu related data from 2ndi2c0 sensor
    data_681 = get_mpu6050_comprehensive_data(i2c1, 0x68,3)			#a dict with all mpu related data from 1sti2c1 sensor
    
    curr_time = utime.ticks_ms()						#to find the current time
    dt = (curr_time - prev_time) / 1000					#to find the difference in time in m.seconds
    prev_time = curr_time								#to initialize variable for next run.
    
    #calculating tilt using time difference, with alpha=0.98 in complementary filter fusion method
    #tilt_x_68, tilt_y_68, tilt_z_68 = calci_tilt_accangles(data_68, dt,0, 0.98)			
    #tilt_x_681, tilt_y_681, tilt_z_681 = calci_tilt_accangles(data_681, dt,0, 0.98)
    #tilt_x_69, tilt_y_69, tilt_z_69 = calci_tilt_accangles(data_69, dt,1, 0.98)  
    
    #printing the respective variables in console
    #print(data_68)								#prints the acc and gyro data independently
    print(data_68)
    
    #print(tilt_x_69, tilt_y_69, tilt_z_69)
    #print(tilt_x_681, tilt_y_681, tilt_z_681)
    
    utime.sleep(0.01)


'''
int16_t accelXOffset, accelYOffset, accelZOffset;
int16_t gyroXOffset, gyroYOffset, gyroZOffset;

void setup() {
```
// Collect raw data for calibration
long gyroXSum = 0, gyroYSum = 0, gyroZSum = 0;
int numReadings = 1000; // Number of readings to average

for (int i = 0; i < numReadings; i++) {
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    accelXSum += ax;
    accelYSum += ay;
    accelZSum += az;
    gyroXSum += gx;
    gyroYSum += gy;
    gyroZSum += gz;
    delay(5);
}

// Calculate offsets
accelXOffset = accelXSum / numReadings;
accelYOffset = accelYSum / numReadings;
accelZOffset = accelZSum / numReadings - 16384; // Assuming ±2g range

gyroXOffset = gyroXSum / numReadings;
gyroYOffset = gyroYSum / numReadings;
gyroZOffset = gyroZSum / numReadings;
```

}

void loop() {
int16_t ax, ay, az, gx, gy, gz;

```
mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

// Apply offsets
float gyroX = (gx - gyroXOffset) 
float gyroY = (gy - gyroYOffset) 
float gyroZ = (gz - gyroZOffset) 

// Use the corrected values
```

}


'''