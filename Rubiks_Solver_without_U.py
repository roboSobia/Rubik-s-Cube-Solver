import cv2
import numpy as np
from collections import Counter
import kociemba
import os
import websocket  # We'll use the websocket-client library instead of socket
import json

# Initial HSV color ranges (will be updated during calibration)
color_ranges = {
    "W": (np.array([0, 0, 150]), np.array([180, 60, 255])),      # White
    "R": (np.array([0, 100, 100]), np.array([10, 255, 255])),    # Red
    "G": (np.array([35, 100, 100]), np.array([85, 255, 255])),   # Green
    "Y": (np.array([25, 100, 100]), np.array([35, 255, 255])),   # Yellow
    "O": (np.array([10, 100, 100]), np.array([25, 255, 255])),   # Orange
    "B": (np.array([85, 100, 100]), np.array([130, 255, 255]))   # Blue
}

# Update these with your ESP32's IP address
ESP32_IP = "192.168.1.x"  # Replace with your ESP32's IP address
ESP32_WEBSOCKET_PORT = 81  # The WebSocket port from your ESP32 code

# Global variable to store the last message received from the WebSocket
last_message = None

# Callback function when a message is received
def on_message(ws, message):
    global last_message
    print(f"Received from ESP32: {message}")
    last_message = message

# Callback for errors
def on_error(ws, error):
    print(f"WebSocket error: {error}")

# Callback for connection close
def on_close(ws, close_status_code, close_msg):
    print("WebSocket connection closed")

# Callback for connection open
def on_open(ws):
    print("WebSocket connection opened")

def send_command(command):
    """Send a command to the ESP32 via WebSocket and wait for a response."""
    global last_message
    last_message = None
    try:
        # Create a WebSocket connection
        ws_url = f"ws://{ESP32_IP}:{ESP32_WEBSOCKET_PORT}/"
        ws = websocket.WebSocketApp(ws_url,
                                   on_message=on_message,
                                   on_error=on_error,
                                   on_close=on_close)
        ws.on_open = on_open
        
        # Start the WebSocket connection in a background thread
        import threading
        wst = threading.Thread(target=ws.run_forever)
        wst.daemon = True
        wst.start()
        
        # Wait for connection to establish
        import time
        timeout = 5
        start_time = time.time()
        while not ws.sock or not ws.sock.connected:
            if time.time() - start_time > timeout:
                raise Exception("Connection timeout")
            time.sleep(0.1)
        
        # Send the command
        print(f"Sending command: {command}")
        ws.send(command)
        
        # Wait for a response
        start_time = time.time()
        while last_message is None:
            if time.time() - start_time > timeout:
                raise Exception("Response timeout")
            time.sleep(0.1)
        
        # Close the connection
        ws.close()
        return last_message
        
    except Exception as e:
        print(f"Error sending command: {str(e)}")
        return None

# The rest of your functions remain the same

def main():
    global color_ranges
    
    # Make sure to install the websocket-client package
    try:
        import websocket
    except ImportError:
        print("The websocket-client package is not installed.")
        print("Please install it using: pip install websocket-client")
        return
    
    temp_dir = "cube_scans"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    stream_url = "http://192.168.1.48:8080/video"  # Adjust if needed
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print("Error: Could not open video stream. Trying default camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open default camera.")
            exit()
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    WINDOW_SIZE = (800, 600)
    
    print("Stream opened successfully.")
    print("Scan the U face 12 times in different orientations.")
    print("Colors: B=Blue, Y=Yellow, R=Red, G=Green, W=White, O=Orange")
    print("Press 'q' to quit, 'c' to capture, 'r' to restart, 'a' to calibrate.")
    
    u_scans = [[] for _ in range(12)]  # 12 scans, each will hold 9 colors
    current_scan_idx = 0
    
    cv2.namedWindow("Rubik's Cube Scanner", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Rubik's Cube Scanner", WINDOW_SIZE[0], WINDOW_SIZE[1])
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame = cv2.resize(frame, WINDOW_SIZE, interpolation=cv2.INTER_AREA)
        height, width = frame.shape[:2]
        
        grid_size = int(min(height, width) * 0.4)
        grid_cell_size = grid_size // 3
        pad_x, pad_y = 20, 50
        
        display = frame.copy()
        for i in range(1, 3):
            x = pad_x + i * grid_cell_size
            y = pad_y + i * grid_cell_size
            cv2.line(display, (x, pad_y), (x, pad_y + grid_size), (0, 255, 0), 2)
            cv2.line(display, (pad_x, y), (pad_x + grid_size, y), (0, 255, 0), 2)
        cv2.rectangle(display, (pad_x, pad_y), (pad_x + grid_size, pad_y + grid_size), (0, 255, 0), 2)
        
        if current_scan_idx < 12:
            instruction = f"Scan U face #{current_scan_idx + 1}/12 (B=Blue, Y=Yellow, R=Red, G=Green, W=White, O=Orange)"
            cv2.putText(display, instruction, (pad_x, pad_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(display, "All scans captured. Press 'r' to restart, 'a' to calibrate, or 'q' to quit.", 
                        (pad_x, pad_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Rubik's Cube Scanner", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if current_scan_idx < 12:
                # We'll use "SCAN" instead of "STOP" to match the ESP32 code
                response = send_command("SCAN")
                if response is None:
                    print("Warning: Command failed or no response received\n")
                image = frame.copy()
                print(f"\nCaptured U face scan #{current_scan_idx + 1}")
                
                face_colors = []
                print(f"What I see on U face scan #{current_scan_idx + 1} (3x3 grid):")
                for i in range(3):
                    row = []
                    for j in range(3):
                        y_start = pad_y + i * grid_cell_size
                        y_end = pad_y + (i + 1) * grid_cell_size
                        x_start = pad_x + j * grid_cell_size
                        x_end = pad_x + (j + 1) * grid_cell_size
                        roi = image[y_start:y_end, x_start:x_end]
                        color = detect_color(roi, color_ranges)
                        face_colors.append(color)
                        row.append(color)
                        cv2.putText(image, color, 
                                   (x_start + grid_cell_size//4, y_start + grid_cell_size//2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    print(f"  Row {i+1}: {' '.join(row)}")
                
                u_scans[current_scan_idx] = face_colors
                image_path = os.path.join(temp_dir, f"u_scan_{current_scan_idx + 1}_processed.jpg")
                cv2.imwrite(image_path, image)
                cv2.imshow(f"U Face Scan #{current_scan_idx + 1} Processed", image)
                
                current_scan_idx += 1
                if current_scan_idx == 12:
                    print("\nAll 12 U face scans captured!")
                    cube_state = construct_cube_from_u_scans(u_scans)
                    print_full_cube_state(cube_state)
                    try:
                        solution = solve_cube_frblud(cube_state)
                        if solution:
                            print(f"\nSolution: {solution}")
                            print("Apply these moves to solve your cube!")
                            response = send_command(f"SOLVE:{solution}")
                            if response is None:
                                print("Warning: Solve command failed or no response received")
                    except Exception as e:
                        print(f"Failed to solve: {e}")
                    print("\nPress 'q' to quit, 'r' to restart, 'a' to calibrate.")
        elif key == ord('r'):
            u_scans = [[] for _ in range(12)]
            current_scan_idx = 0
            print("\nRestarting cube scan...")
        elif key == ord('a'):
            print("\nEntering calibration mode...")
            calibrated_ranges = calibrate_colors(cap, WINDOW_SIZE)
            if calibrated_ranges is not None:
                color_ranges.clear()
                color_ranges.update(calibrated_ranges)
                print("Calibration complete. Using custom color ranges.")
                u_scans = [[] for _ in range(12)]
                current_scan_idx = 0
                print("Restarting cube scan...")
            else:
                print("Calibration aborted. Resuming with previous settings.")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()