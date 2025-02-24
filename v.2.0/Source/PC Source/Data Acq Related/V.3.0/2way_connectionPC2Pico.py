import serial

pico_port = "COM16"  # Update with your correct port
baud_rate = 115200   # Same as Pico's baud rate

ser = serial.Serial(pico_port, baud_rate, timeout=1)

# Send message to Pico

i=0
# Read response from Pico
while True:
    ser.write(b"hi")
    response = ser.readline().decode().strip()  # Read response from Pico 
    if response:
        i=i+1
        print(i)
        print(response)
