'''{
    File Description: 
        The purpose of this file is to test a two way communication connection between PC and the Microcontroller pico
    }'''

import sys

while True:
    data = sys.stdin.read(1)  # Read 1 byte at a time - the pico will block here until an byte is sent by pc and read here
    if data:                # checks if response is non-empty
        print(f"Received: {data}")  # Send response back to PC
