'''{
    File Description:
        This file contains functions that support the interaction of the pico with the MPU6050 sensors for their data values,
        including the feature extraction process from raw-values.
    }'''
from machine import Pin, I2C  # Import necessary modules for hardware control
import utime  # Import utime for delays
import math  # Import math module for calculations

# Define MPU6050 register addresses
PWR_MGMT_1 = 0x6B     # Power management register
SMPLRT_DIV = 0x19     # Sample rate divider
CONFIG = 0x1A         # Configuration register
GYRO_CONFIG = 0x1B    # Gyroscope configuration register
ACCEL_CONFIG = 0x1C   # Accelerometer configuration register
TEMP_OUT_H = 0x41     # Temperature data register (high byte)
ACCEL_XOUT_H = 0x3B   # Accelerometer X-axis data register (high byte)
GYRO_XOUT_H = 0x43    # Gyroscope X-axis data register (high byte)

# Initialize lists to store angles for different sensors
angleX = [0, 0, 0]  # Stores computed X-axis tilt angles for three sensors
angleY = [0, 0, 0]  # Stores computed Y-axis tilt angles for three sensors
angleZ = [0, 0, 0]  # Stores computed Z-axis tilt angles for three sensors

# Pre-calculated gyro bias values for three sensors
avgg_z = [-3.572, -0.183, 0]  # Bias values for gyroscope Z-axis
avgg_x = [-3.619, -1.090, 0]  # Bias values for gyroscope X-axis
avgg_y = [0.902, 0.699, 0]    # Bias values for gyroscope Y-axis

# Placeholder for accelerometer biases (to be calibrated if necessary)
avga_x = [0, 0, 0]  # Bias values for accelerometer X-axis
avga_y = [0, 0, 0]  # Bias values for accelerometer Y-axis
avga_z = [0, 0, 0]  # Bias values for accelerometer Z-axis

def init_mpu6050(i2c, address=0x68):
    """{
    Initializes the MPU6050 sensor by configuring necessary registers.
    
    Parameters:
        i2c (I2C): I2C instance for communication
        address (int): I2C address of the sensor (default: 0x68)
    }"""
    i2c.writeto_mem(address, PWR_MGMT_1, b'\x01')  # Wake up MPU6050 from sleep mode
    utime.sleep_ms(100)  # Wait for sensor stabilization
    i2c.writeto_mem(address, SMPLRT_DIV, b'\x00')  # Set sample rate divider
    i2c.writeto_mem(address, CONFIG, b'\x00')  # Configure digital low-pass filter
    i2c.writeto_mem(address, GYRO_CONFIG, b'\x08')  # Set gyroscope sensitivity to ±500°/s
    i2c.writeto_mem(address, ACCEL_CONFIG, b'\x00')  # Set accelerometer sensitivity to ±2g

def read_raw_data(i2c, addr, address=0x68):
    """{
    Reads raw 16-bit sensor data from the given register address.

    Parameters:
        i2c (I2C): I2C instance for communication
        addr (int): Register address to read data from
        address (int): I2C address of the sensor (default: 0x68)
    
    Returns:
        int: Processed raw data value (signed 16-bit)
    }"""
    high = i2c.readfrom_mem(address, addr, 1)[0]  # Read high byte
    low = i2c.readfrom_mem(address, addr + 1, 1)[0]  # Read low byte
    value = (high << 8) | low  # Combine high and low bytes

    if value > 32768:  # Convert to signed 16-bit integer
        value -= 65536

    return value  # Return processed data value

def variance(data):
    """{
    Computes the sample variance of a given dataset.

    Parameters:
        data (list of float): List of numerical values.

    Returns:
        float: Sample variance of the dataset (ddof=1).
    }"""
    n = len(data)  # Get the number of elements in the dataset
    if n == 0:  # Handle empty list case
        return 0
    mean = sum(data) / n  # Compute mean of the dataset
    squared_diffs = [(x - mean) ** 2 for x in data]  # Compute squared differences from mean
    return sum(squared_diffs) / (n - 1)  # Return sample variance

def calibrate_gyro(i2c, address, sensor_no, num_samples=500):  # Default to 500 samples
    """{
    Calibrates the gyroscope by averaging multiple sensor readings.

    Parameters:
        i2c (I2C): I2C instance for communication.
        address (int): I2C address of the sensor.
        sensor_no (int): Sensor number (1-based indexing).
        num_samples (int): Number of samples to average for calibration (default: 500).

    Returns:
        tuple: Averaged gyroscope bias values (avgg_x, avgg_y, avgg_z) for the given sensor.
    }"""
    
    print("Calibrating gyroscope...")                   #User-Information

    # Initialize sum variables for gyroscope and accelerometer readings
    sumg_x = 0
    sumg_y = 0
    sumg_z = 0
    suma_x = 0
    suma_y = 0
    suma_z = 0

    # Collect `num_samples` readings from the sensor to find bias from
    for _ in range(num_samples):
        
        # collect and scale data with respect to established sensitivity
        gyro_x = read_raw_data(i2c, GYRO_XOUT_H, address) / 65.5  # Convert raw gyroscope X data
        gyro_y = read_raw_data(i2c, GYRO_XOUT_H + 2, address) / 65.5  # Convert raw gyroscope Y data
        gyro_z = read_raw_data(i2c, GYRO_XOUT_H + 4, address) / 65.5  # Convert raw gyroscope Z data
        accel_x = read_raw_data(i2c, ACCEL_XOUT_H, address) / 16384.0  # Convert raw accelerometer X data
        accel_y = read_raw_data(i2c, ACCEL_XOUT_H + 2, address) / 16384.0  # Convert raw accelerometer Y data
        accel_z = read_raw_data(i2c, ACCEL_XOUT_H + 4, address) / 16384.0  # Convert raw accelerometer Z data

        # Accumulate values for averaging
        sumg_x += gyro_x
        sumg_y += gyro_y
        sumg_z += gyro_z
        suma_x += accel_x
        suma_y += accel_y
        suma_z += accel_z

        utime.sleep_ms(10)  # Delay between readings to stabilize measurements

    sensor = sensor_no - 1  # Convert 1-based sensor number to 0-based index

    # Compute and store average gyroscope and accelerometer biases
    global avgg_x, avgg_y, avgg_z, avga_x, avga_y, avga_z
    avgg_x[sensor] = sumg_x / num_samples
    avgg_y[sensor] = sumg_y / num_samples
    avgg_z[sensor] = sumg_z / num_samples
    avga_x[sensor] = suma_x / num_samples
    avga_y[sensor] = suma_y / num_samples
    avga_z[sensor] = suma_z / num_samples

    print("Gyroscope calibration complete.", avgg_x[sensor], avgg_y[sensor], avgg_z[sensor])
    return avgg_x[sensor], avgg_y[sensor], avgg_z[sensor]

def calibrate_gyroonlyZ(i2c, address, sensor, num_samples=1000):
    """{
        Calibrates the gyroscope for the Z-axis by computing its average offset.

        Parameters:
            i2c (I2C): I2C instance for communication.
            address (int): I2C address of the MPU6050 sensor.
            sensor (int): Index of the sensor in the global array.
            num_samples (int): Number of samples for averaging (default: 1000).
        
        Returns:
            None
    }"""
    print("Calibrating gyroscope...")           # User Info
    sumg_z = 0
    
    # summing the multiple values for num_samples samples.
    for _ in range(num_samples):
        gyro_z = read_raw_data(i2c, GYRO_XOUT_H + 4, address) / 65.5  # Read and convert raw Z-axis gyroscope data.
        sumg_z += gyro_z
        utime.sleep_ms(10)  # Delay to avoid excessive polling.

    # Compute average
    global avgg_z
    avgg_z[sensor] = sumg_z / num_samples       # storing in a global variable.

    print("Gyroscope Z-axis calibration complete.") # User Info

def calci_tilt_angles(data, sensor_no, dtime, alpha=0.98):
    """{
        Computes tilt angles using accelerometer and gyroscope data via a complementary filter.

        Parameters:
            data (dict): Dictionary containing accelerometer and gyroscope data.
            sensor_no (int): Sensor index (1-based).
            dtime (float): Time step between measurements.
            alpha (float): Complementary filter weighting factor (default: 0.98).
        
        Returns:
            tuple: Filtered tilt angles (angleX, angleY, angleZ) and raw accelerometer tilt angles.
    }"""
    x, y, z = data['accel']['x'], data['accel']['y'], data['accel']['z']

    # Compute accelerometer-based angles - using established formulas
    angleAccX = math.atan2(y, math.sqrt(x * x + z * z)) * 180 / math.pi
    angleAccY = -math.atan2(x, math.sqrt(y * y + z * z)) * 180 / math.pi
    angleAccZ = -math.atan2(math.sqrt(y * y + x * x), z) * 180 / math.pi

    global angleX, angleY, angleZ
    sensor_index = sensor_no - 1  # Convert 1-based index to 0-based.

    # Apply complementary filter for tilt angle estimation. - using established formulas - further explained below the library file
    angleX[sensor_index] = wrap(alpha * (angleAccX + wrap(angleX[sensor_index] + data['gyro_biased_fixed']['x'] * dtime - angleAccX, 180)) + (1.0 - alpha) * angleAccX, 180)
    angleY[sensor_index] = wrap(alpha * (angleAccY + wrap(angleY[sensor_index] + data['gyro_biased_fixed']['y'] * dtime - angleAccY, 180)) + (1.0 - alpha) * angleAccY, 180)
    angleZ[sensor_index] = wrap(alpha * (angleAccZ + wrap(angleZ[sensor_index] + data['gyro_biased_fixed']['z'] * dtime - angleAccZ, 180)) + (1.0 - alpha) * angleAccZ, 180)

    return angleX[sensor_index], angleY[sensor_index], angleZ[sensor_index], angleAccX, angleAccY, angleAccZ

def wrap(angle, limit):
    """{
        Wraps an angle within a given limit to ensure it stays within bounds. particularly used for gyro angle estimation

        Parameters:
            angle (float): Input angle in degrees.
            limit (float): Maximum absolute value for the angle.
        
        Returns:
            float: Wrapped angle within the specified limit.
    }"""
    while angle > limit:
        angle -= 2 * limit
    while angle < -limit:
        angle += 2 * limit
    return angle

def get_mpu6050_data(i2c, address, sensor):
    """{
        Reads raw accelerometer and gyroscope data from the MPU6050 sensor.

        Parameters:
            i2c (I2C): I2C instance for communication.
            address (int): I2C address of the MPU6050 sensor.
            sensor (int): Sensor index (for tracking purposes).
        
        Returns:
            dict: Dictionary containing accelerometer and gyroscope readings.
    }"""
    # Read accelerometer data and convert to g-forces.
    accel_x = read_raw_data(i2c, ACCEL_XOUT_H, address) / 16384.0
    accel_y = read_raw_data(i2c, ACCEL_XOUT_H + 2, address) / 16384.0
    accel_z = read_raw_data(i2c, ACCEL_XOUT_H + 4, address) / 16384.0

    # Read gyroscope data and convert to degrees per second.
    gyro_x = read_raw_data(i2c, GYRO_XOUT_H, address) / 65.5
    gyro_y = read_raw_data(i2c, GYRO_XOUT_H + 2, address) / 65.5
    gyro_z = read_raw_data(i2c, GYRO_XOUT_H + 4, address) / 65.5

    return {
        'accel': {'x': accel_x, 'y': accel_y, 'z': accel_z},
        'gyro': {'x': gyro_x, 'y': gyro_y, 'z': gyro_z},
    }

def get_mpu6050_comprehensive_data(i2c, address, sensor):
    """{
        Reads raw accelerometer and gyroscope data and computes bias-corrected gyroscope values.

        Parameters:
            i2c (I2C): I2C instance for communication.
            address (int): I2C address of the MPU6050 sensor.
            sensor (int): Sensor index (1-based).
        
        Returns:
            dict: Dictionary containing raw accelerometer and gyroscope readings, along with bias-corrected gyroscope values.
    }"""
    # Read accelerometer data and convert to g-forces.
    accel_x = read_raw_data(i2c, ACCEL_XOUT_H, address) / 16384.0
    accel_y = read_raw_data(i2c, ACCEL_XOUT_H + 2, address) / 16384.0
    accel_z = read_raw_data(i2c, ACCEL_XOUT_H + 4, address) / 16384.0

    # Read gyroscope data and convert to degrees per second.
    gyro_x = read_raw_data(i2c, GYRO_XOUT_H, address) / 65.5
    gyro_y = read_raw_data(i2c, GYRO_XOUT_H + 2, address) / 65.5
    gyro_z = read_raw_data(i2c, GYRO_XOUT_H + 4, address) / 65.5

    global avgg_x, avgg_y, avgg_z

    # Compute bias-corrected gyroscope values.
    sensor_index = sensor - 1  # Convert 1-based index to 0-based.
    gyro_bx = gyro_x - avgg_x[sensor_index]
    gyro_by = gyro_y - avgg_y[sensor_index]
    gyro_bz = gyro_z - avgg_z[sensor_index]

    return {
        'accel': {'x': accel_x, 'y': accel_y, 'z': accel_z},
        'gyro': {'x': gyro_x, 'y': gyro_y, 'z': gyro_z},
        'gyro_biased': {'x': gyro_bx, 'y': gyro_by, 'z': gyro_bz},
    }


### Gyro_Spike_Fix Function Variable Initialization
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
gyrob_index_start = 6          # the starting index of the biased gyro values(x,y,z) in the tri_sample_data
gyrob_index_end = 8             # the ending index of the biased gyro values(x,y,z) in the tri_sample_data
def get_mpu6050_comprehensive_data_Gyro_Spike_Fix(i2c, address,sensor):
    '''{
        This function operates the same as get_mpu6050_comprehensive_data but also fixes the irrelevant spikes that happen,
        but delays the output by one sample.
        }'''
    
    # Acquiring relevant data
    sensor_index = sensor - 1 
    accel_x = read_raw_data(i2c, ACCEL_XOUT_H, address) / 16384.0       # in g = 9.8m/s^2
    accel_y = read_raw_data(i2c, ACCEL_XOUT_H + 2, address) / 16384.0
    accel_z = read_raw_data(i2c, ACCEL_XOUT_H + 4, address) / 16384.0
    gyro_x = read_raw_data(i2c, GYRO_XOUT_H, address) / 65.5#131.0      #degrees per second
    gyro_y = read_raw_data(i2c, GYRO_XOUT_H + 2, address) / 65.5
    gyro_z = read_raw_data(i2c, GYRO_XOUT_H + 4, address) / 65.5
  
    global avgg_z,avgg_x,avgg_y
    
    # Applying Bias fixes for gyroscopes
    gyro_bx = gyro_x-avgg_x[sensor_index]
    gyro_by = gyro_y-avgg_y[sensor_index]
    gyro_bz = gyro_z-avgg_z[sensor_index]
    
    # initializing the newest sample of the tri_sample_data with the newest data.
    tri_sample_data[sensor_index][new_index] = [accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,gyro_bx,gyro_by,gyro_bz]
    
    # returns true for checking if two values are oppositely signed (+&-)
    def opposite_signs(a, b):
        return (a * b) < 0  # Works for int and float

    #cycling through all values of gyroscope axes, to eliminate the spikes.
    for i in range (gyrob_index_start,gyrob_index_end + 1): 
        
        # Extracting Values
        new_val = tri_sample_data[sensor_index][new_index][i]
        curr_val = tri_sample_data[sensor_index][curr_index][i]
        prev_val = tri_sample_data[sensor_index][prev_index][i]
        
        # The Spike_Fix Algorithm
        if not (math.floor(abs(new_val)) == 0 and math.floor(abs(prev_val)) == 0 and math.floor(abs(curr_val)) == 0): # making sure it isn't a normal 0 based value
            # to 0 the single peak values
            if math.floor(abs(new_val)) == 0 and math.floor(abs(prev_val)) == 0 and math.floor(abs(curr_val)) != 0:   
                tri_sample_data[sensor_index][curr_index][i] = 0
            
            # to 0 the mirror peak
            elif math.floor(abs(prev_val)) == 0 and opposite_signs(curr_val,new_val) and abs((abs(curr_val)) - (abs(new_val))) < 0.4  :
                tri_sample_data[sensor_index][curr_index][i] = 0
            
            # to 0 the double peak sync values
            elif math.floor(abs(prev_val)) == 0 and abs((abs(curr_val)) - (abs(new_val))) < 0.095  :
                tri_sample_data[sensor_index][curr_index][i] = 0
        
    # storing the fixed data for transmitting
    gyro_bx_f = tri_sample_data[sensor_index][curr_index][gyrob_index_start]
    gyro_by_f = tri_sample_data[sensor_index][curr_index][gyrob_index_start+1]
    gyro_bz_f = tri_sample_data[sensor_index][curr_index][gyrob_index_end]    
    
    #preparing the tri_sample_data for the next loop; Left shifting the array for the next new_index
    tri_sample_data[sensor_index][prev_index] =  tri_sample_data[sensor_index][curr_index]
    tri_sample_data[sensor_index][curr_index] =  tri_sample_data[sensor_index][new_index]


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

def calibrate_checkgyro(i2c, address, sensor_no, num_samples=200):  # Collects and checks gyroscope variance
    """{ 
    Calibrates the gyroscope by collecting multiple readings and checking their variance.
    
    Parameters:
    - i2c: I2C interface object.
    - address: I2C address of the MPU6050 sensor.
    - sensor_no: Sensor number (1-based index).
    - num_samples: Number of samples to collect for calibration (default: 200).
    
    The function reads gyroscope data from the sensor, calculates the variance of each axis,
    and checks if the variance is below a predefined threshold. If variance is within limits,
    it prints confirmation; otherwise, it warns of excessive variance.
    }"""
    
    gyroxlist = []  # List to store X-axis gyroscope readings
    gyroylist = []  # List to store Y-axis gyroscope readings
    gyrozlist = []  # List to store Z-axis gyroscope readings

    for i in range(num_samples):  
        data = get_mpu6050_comprehensive_data(i2c, address, sensor_no)  # Retrieve sensor data
        gyro_x = data['gyro_biased']['x']  # Extract bias-corrected X-axis gyro data
        gyro_y = data['gyro_biased']['y']  # Extract bias-corrected Y-axis gyro data
        gyro_z = data['gyro_biased']['z']  # Extract bias-corrected Z-axis gyro data
        gyroxlist.append(gyro_x)  # Store X-axis data
        gyroylist.append(gyro_y)  # Store Y-axis data
        gyrozlist.append(gyro_z)  # Store Z-axis data

    sensor_index = sensor_no - 1  # Convert sensor number to 0-based index
    
    # Predefined variance threshold for each sensor and axis
    sensor_var_threshold = [[1.2, 1.2, 1.2],  # Sensor 1: [X, Y, Z] variance thresholds
                            [1.2, 1.2, 1.2],  # Sensor 2: [X, Y, Z] variance thresholds
                            [1.2, 1.2, 1.2]]  # Sensor 3: [X, Y, Z] variance thresholds

    # Check if variance for each axis is below the threshold
    if (variance(gyroxlist) < sensor_var_threshold[sensor_index][0] and
        variance(gyroylist) < sensor_var_threshold[sensor_index][1] and
        variance(gyrozlist) < sensor_var_threshold[sensor_index][2]):
        print("variance checks out", variance(gyroxlist), variance(gyroylist), variance(gyrozlist))  # Print success message
    else:
        print("variance doesn't check out")  # Print warning if variance exceeds the threshold
        print(variance(gyroxlist), variance(gyroylist), variance(gyrozlist))  # Display the computed variances


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



