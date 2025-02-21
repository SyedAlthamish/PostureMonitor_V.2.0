import sys

while True:
    data = sys.stdin.read(1)  # Read 1 byte at a time (prevents blocking)
    if data:
        print(f"Received: {data}")  # Send response back to PC
