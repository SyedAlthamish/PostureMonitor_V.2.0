'''{
    A basic integration of BLE hc05 to computer
    }'''
from machine import Pin, UART
import utime

uart = machine.UART(0, baudrate=9600, tx=machine.Pin(16), rx=machine.Pin(17))

while True:

    uart.write(output_string)
    
    utime.sleep(0.01)