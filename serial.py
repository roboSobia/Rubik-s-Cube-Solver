import serial
import time

# Serial port configuration
SERIAL_PORT = 'COM1' 
SERIAL_BAUDRATE = 9600
SERIAL_TIMEOUT = 1

def serial_send(solution):
  try:
    # Open serial port
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT)
    time.sleep(2) # Wait for the serial connection to initialize
    print(f"Connected to {SERIAL_PORT}")

    # Send the solution string
    ser.write(solution.encode()) # encode string to bytes
    print(f"Sent solution: {solution}")

    # Close the communication
    ser.close()
    print(f"Closed connection to {SERIAL_PORT}")

  except serial.SerialException:
    print(f"Failed to connect to {SERIAL_PORT}")
    return
