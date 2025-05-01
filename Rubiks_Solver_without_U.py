import cv2
import numpy as np
from collections import Counter
import kociemba
import os
import serial
import time
import json
import random
import argparse

# Global variables and constants
WINDOW_SIZE = (640, 480)
SCAN_COOLDOWN = 0.5
MOTOR_STABILIZATION_TIME = 0.5
STABILITY_THRESHOLD = 3
MIN_CONTOUR_AREA = 5000
MAX_CONTOUR_AREA = 50000

# Rotation sequence for scanning
rotation_sequence = [
    "",        # Scan 1: Initial U face
    "R L'",    # Scan 2
    "B F'",    # Scan 3
    "R L'",    # Scan 4
    "B F'",    # Scan 5
    "R L'",    # Scan 6
    "B F'",    # Scan 7
    "R L'",    # Scan 8
    "B F'",    # Scan 9
    "R L'",    # Scan 10
    "B F'",    # Scan 11
    "R L'"     # Scan 12
]

# Initial HSV color ranges (will be updated during calibration or loaded from file)
color_ranges = {
    "W": (np.array([0, 0, 150]), np.array([180, 60, 255])),      # White
    "R": (np.array([0, 100, 100]), np.array([10, 255, 255])),    # Red
    "G": (np.array([35, 100, 100]), np.array([85, 255, 255])),   # Green
    "Y": (np.array([25, 100, 100]), np.array([35, 255, 255])),   # Yellow
    "O": (np.array([10, 100, 100]), np.array([25, 255, 255])),   # Orange
    "B": (np.array([85, 100, 100]), np.array([130, 255, 255]))   # Blue
}

def save_color_ranges(color_ranges, filename="color_ranges.json"):
    """Save color ranges to a JSON file."""
    serializable_ranges = {
        color: (lower.tolist(), upper.tolist()) for color, (lower, upper) in color_ranges.items()
    }
    with open(filename, 'w') as f:
        json.dump(serializable_ranges, f, indent=4)
    print(f"Color ranges saved to {filename}")

def load_color_ranges(filename="color_ranges.json"):
    """Load color ranges from a JSON file."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            serializable_ranges = json.load(f)
        color_ranges = {
            color: (np.array(lower), np.array(upper)) for color, (lower, upper) in serializable_ranges.items()
        }
        print(f"Color ranges loaded from {filename}")
        return color_ranges
    return None

def calibrate_colors(cap, window_size, color_names=["W", "R", "G", "Y", "O", "B"]):
    """Calibrate HSV ranges by sampling the center of each face."""
    print("\nStarting calibration. Show each face's center when prompted.")
    print("Press 'c' to capture, 'q' to quit early.")
    
    calibrated_ranges = {}
    grid_size = int(min(window_size) * 0.4)
    grid_cell_size = grid_size // 3
    pad_x, pad_y = 20, 50
    
    for color in color_names:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame during calibration")
                return None
            
            frame = cv2.resize(frame, window_size, interpolation=cv2.INTER_AREA)
            display = frame.copy()
            
            # Draw a small box around the center cell (1,1)
            center_y_start = pad_y + grid_cell_size
            center_y_end = pad_y + 2 * grid_cell_size
            center_x_start = pad_x + grid_cell_size
            center_x_end = pad_x + 2 * grid_cell_size
            cv2.rectangle(display, (center_x_start, center_y_start), 
                         (center_x_end, center_y_end), (0, 255, 0), 2)
            
            instruction = f"Show {color} center (e.g., White=W, Red=R), then press 'c'"
            cv2.putText(display, instruction, (pad_x, pad_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Calibration", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                roi = frame[center_y_start:center_y_end, center_x_start:center_x_end]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                avg_hsv = np.mean(hsv_roi, axis=(0, 1))
                
                # Define a range around the sampled HSV
                h_range = 10 if color != "W" else 90  # Wider hue for white
                s_range = 50
                v_range = 50
                lower = np.array([max(0, avg_hsv[0] - h_range), 
                                max(0, avg_hsv[1] - s_range), 
                                max(0, avg_hsv[2] - v_range)])
                upper = np.array([min(180, avg_hsv[0] + h_range), 
                                min(255, avg_hsv[1] + s_range), 
                                min(255, avg_hsv[2] + v_range)])
                
                calibrated_ranges[color] = (lower, upper)
                print(f"Calibrated {color}: HSV={avg_hsv}, Range={lower} to {upper}")
                break
            elif key == ord('q'):
                print("Calibration aborted.")
                return None
    
    cv2.destroyWindow("Calibration")
    return calibrated_ranges

def detect_color(roi, color_ranges):
    """
    Detect the dominant color in a region of interest (ROI) using multiple methods.
    Returns the first letter of the color (e.g., 'R' for Red).
    """
    if len(roi.shape) == 3 and roi.shape[2] == 3:
        if roi.dtype != np.uint8:
            roi = np.uint8(roi)
        
        h, w = roi.shape[:2]
        center_roi = roi[h//4:3*h//4, w//4:3*w//4]
        hsv_roi = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)
        
        color_matches = {}
        for color, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv_roi, lower, upper)
            match_percentage = cv2.countNonZero(mask) / (center_roi.shape[0] * center_roi.shape[1])
            color_matches[color] = match_percentage * 100
        
        range_best_color = max(color_matches, key=color_matches.get)
        range_best_match = color_matches[range_best_color]
        
        pixels = hsv_roi.reshape((-1, 3))
        pixel_list = [tuple(p) for p in pixels]
        most_common_hsv = Counter(pixel_list).most_common(1)[0][0]
        
        dominant_color = None
        min_distance = float('inf')
        
        for color, (lower, upper) in color_ranges.items():
            middle_hsv = (lower + upper) / 2
            h_dist = min(abs(most_common_hsv[0] - middle_hsv[0]), 
                        180 - abs(most_common_hsv[0] - middle_hsv[0]))
            s_dist = abs(most_common_hsv[1] - middle_hsv[1])
            v_dist = abs(most_common_hsv[2] - middle_hsv[2])
            
            if color == "W":
                distance = 0.1 * h_dist + 0.3 * s_dist + 0.6 * v_dist
            else:
                distance = 0.6 * h_dist + 0.3 * s_dist + 0.1 * v_dist
                
            if distance < min_distance:
                min_distance = distance
                dominant_color = color
        
        print(f"ROI Analysis:")
        print(f"  Range-based best: {range_best_color} ({range_best_match:.1f}%)")
        print(f"  Dominant HSV: {most_common_hsv}, closest match: {dominant_color}")
        
        if range_best_match > 10:
            return range_best_color
        elif dominant_color:
            return dominant_color
        
        avg_hsv = np.mean(hsv_roi, axis=(0,1))
        closest_color = None
        min_distance = float('inf')
        
        for color, (lower, upper) in color_ranges.items():
            middle_hsv = (lower + upper) / 2
            h_dist = min(abs(avg_hsv[0] - middle_hsv[0]), 
                        180 - abs(avg_hsv[0] - middle_hsv[0]))
            s_dist = abs(avg_hsv[1] - middle_hsv[1])
            v_dist = abs(avg_hsv[2] - middle_hsv[2])
            
            if color == "W":
                distance = 0.1 * h_dist + 0.3 * s_dist + 0.6 * v_dist
            else:
                distance = 0.6 * h_dist + 0.3 * s_dist + 0.1 * v_dist
                
            if distance < min_distance:
                min_distance = distance
                closest_color = color
        
        print(f"  Using closest color match: {closest_color}")
        return closest_color
    
    return "W"

def validate_cube(cube, order_name):
    if len(cube) != 54:
        raise ValueError(f"{order_name} must be 54 characters")
    counts = Counter(cube)
    if len(counts) != 6 or any(count != 9 for count in counts.values()):
        raise ValueError(f"{order_name} invalid: {counts} (need 9 of each of 6 colors)")

def remap_colors_to_kociemba(cube_frblud):
    validate_cube(cube_frblud, "FRBLUD")
    centers = [cube_frblud[i] for i in [4, 13, 22, 31, 40, 49]]  # F, R, B, L, U, D
    color_map = {
        centers[4]: 'U',  # Up
        centers[1]: 'R',  # Right
        centers[0]: 'F',  # Front
        centers[5]: 'D',  # Down
        centers[3]: 'L',  # Left
        centers[2]: 'B'   # Back
    }
    return color_map, ''.join(color_map[c] for c in cube_frblud)

def remap_cube_to_kociemba(cube_frblud_remapped):
    front, right, back, left, up, down = [cube_frblud_remapped[i:i+9] for i in range(0, 54, 9)]
    return up + right + front + down + left + back

def get_solved_state(cube_frblud, color_map):
    centers = [cube_frblud[i] for i in [4, 13, 22, 31, 40, 49]]  # F, R, B, L, U, D
    return ''.join(c * 9 for c in centers)

def is_cube_solved(cube_state):
    """Check if the cube is already solved by verifying each face has the same color."""
    # Check each face (9 stickers per face)
    for i in range(0, 54, 9):
        face = cube_state[i:i+9]
        # If any sticker is different from the center of the face, cube is not solved
        if not all(sticker == face[4] for sticker in face):
            return False
    return True

def simplify_cube_moves(moves_str):
    # Split the string into individual moves
    moves = moves_str.strip().split()
    
    # Function to get the net effect of a single move
    def move_value(move):
        if move.endswith("2"):
            return 2
        elif move.endswith("'"):
            return -1
        return 1
    
    # Function to convert net value back to move notation
    def value_to_move(face, value):
        value = value % 4
        if value == 0:
            return None
        elif value == 1:
            return face
        elif value == 2:
            return face + "2"
        elif value == 3:
            return face + "'"
    
    # Process moves for each face type
    face_groups = [['L', 'R'], ['F', 'B'], ['U', 'D']]
    
    # First pass: Combine consecutive moves of the same face
    i = 0
    simplified = []
    while i < len(moves):
        current_face = moves[i][0]
        current_value = move_value(moves[i])
        
        # Look ahead for same face moves
        j = i + 1
        while j < len(moves) and moves[j][0] == current_face:
            current_value += move_value(moves[j])
            j += 1
            
        # Add the simplified move if needed
        move = value_to_move(current_face, current_value)
        if move:
            simplified.append(move)
            
        i = j
    
    # Second pass: Combine moves by face groups
    final_simplified = []
    i = 0
    while i < len(simplified):
        current_face = simplified[i][0]
        
        # Find which group this face belongs to
        face_group = None
        for group in face_groups:
            if current_face in group:
                face_group = group
                break
        
        if face_group:
            # Count moves for each face in this group
            counts = {face: 0 for face in face_group}
            j = i
            
            # Collect consecutive moves in this group
            while j < len(simplified) and simplified[j][0] in face_group:
                face = simplified[j][0]
                counts[face] += move_value(simplified[j])
                j += 1
                
            # Add simplified moves for this group
            for face in face_group:
                move = value_to_move(face, counts[face])
                if move:
                    final_simplified.append(move)
                    
            i = j
        else:
            # For faces not in any group (which shouldn't happen with standard cube notation)
            final_simplified.append(simplified[i])
            i += 1
    
    return " ".join(final_simplified) if final_simplified else "No moves"

def solve_cube_frblud(cube_frblud):
    try:
        # First check if cube is already solved
        if is_cube_solved(cube_frblud):
            print("\nCube is already solved! No moves needed.")
            return ""
            
        # If not solved, proceed with normal solving process
        color_map, cube_frblud_remapped = remap_colors_to_kociemba(cube_frblud)
        scrambled_kociemba = remap_cube_to_kociemba(cube_frblud_remapped)
        solved_frblud = get_solved_state(cube_frblud, color_map)
        _, solved_frblud_remapped = remap_colors_to_kociemba(solved_frblud)
        solved_kociemba = remap_cube_to_kociemba(solved_frblud_remapped)
        validate_cube(scrambled_kociemba, "Scrambled Kociemba")
        validate_cube(solved_kociemba, "Solved Kociemba")
        solution = kociemba.solve(scrambled_kociemba, solved_kociemba)
        
        u_replacement = "R L F2 B2 R' L' D R L F2 B2 R' L'"
        u_prime_replacement = "R L F2 B2 R' L' D' R L F2 B2 R' L'"
        u2_replacement = "R L F2 B2 R' L' D2 R L F2 B2 R' L'"
        
        moves = solution.split()
        modified_solution = []
        for move in moves:
            if move == "U":
                modified_solution.append(u_replacement)
            elif move == "U'":
                modified_solution.append(u_prime_replacement)
            elif move == "U2":
                modified_solution.append(u2_replacement)
            else:
                modified_solution.append(move)
        
        final_solution = " ".join(modified_solution)
        
        # Optimize the solution using simplify_cube_moves
        optimized_solution = simplify_cube_moves(final_solution)
        print("\nOriginal solution length:", len(final_solution.split()))
        print("Optimized solution length:", len(optimized_solution.split()))
        
        return optimized_solution
    
    except Exception as e:
        print(f"Error solving cube: {str(e)}")
        return None

def print_full_cube_state(cube_state):
    print("\nFull cube state (Front, Right, Back, Left, Up, Down):")
    print("".join(cube_state))
    print("\nVisual representation:")
    idx = [0, 9, 18, 27, 36, 45]  # F, R, B, L, U, D
    for i in range(3):
        start = idx[4] + i*3  # Up face
        print("        " + " ".join(cube_state[start:start+3]))
    for i in range(3):
        line = ""
        for face_start in idx[:4]:  # F, R, B, L
            start = face_start + i*3
            line += " ".join(cube_state[start:start+3]) + " | "
        print(line[:-3])
    for i in range(3):
        start = idx[5] + i*3  # Down face
        print("        " + " ".join(cube_state[start:start+3]))

def construct_cube_from_u_scans(u_scans):
    """Construct the full cube state from 12 U face scans."""
    cube_state = [''] * 54
    cube_state[4] = 'B'   # F center
    cube_state[13] = 'O'  # R center
    cube_state[22] = 'G'  # B center
    cube_state[31] = 'R'  # L center
    cube_state[40] = 'W'  # U center
    cube_state[49] = 'Y'  # D center
    
    cube_state[36:45] = u_scans[0]
    for i in range(54):
        if not cube_state[i]:
            cube_state[i] = '-'
    
    cube_state[0] = u_scans[1][0]
    cube_state[2] = u_scans[1][2]
    cube_state[3] = u_scans[1][3]
    cube_state[5] = u_scans[1][5]
    cube_state[6] = u_scans[1][6]
    cube_state[8] = u_scans[1][8]
    
    cube_state[9] = u_scans[2][0]
    cube_state[10] = u_scans[2][1]
    cube_state[11] = u_scans[2][2]
    cube_state[15] = u_scans[2][6]
    cube_state[16] = u_scans[2][7]
    cube_state[17] = u_scans[2][8]
    
    cube_state[47] = u_scans[3][0]
    cube_state[53] = u_scans[3][2]
    cube_state[1] = u_scans[3][3]
    cube_state[7] = u_scans[3][5]
    cube_state[45] = u_scans[3][6]
    cube_state[51] = u_scans[3][8]
    
    cube_state[24] = u_scans[4][0]
    cube_state[12] = u_scans[4][1]
    cube_state[18] = u_scans[4][2]
    cube_state[26] = u_scans[4][6]
    cube_state[14] = u_scans[4][7]
    cube_state[20] = u_scans[4][8]
    
    cube_state[33] = u_scans[5][0]
    cube_state[27] = u_scans[5][2]
    cube_state[50] = u_scans[5][3]
    cube_state[48] = u_scans[5][5]
    cube_state[35] = u_scans[5][6]
    cube_state[29] = u_scans[5][8]
    
    cube_state[36] = u_scans[6][0]
    cube_state[46] = u_scans[6][1]
    cube_state[38] = u_scans[6][2]
    cube_state[42] = u_scans[6][6]
    cube_state[52] = u_scans[6][7]
    cube_state[44] = u_scans[6][8]
    
    cube_state[21] = u_scans[7][3]
    cube_state[23] = u_scans[7][5]
    
    cube_state[34] = u_scans[8][1]
    cube_state[28] = u_scans[8][7]
    
    cube_state[25] = u_scans[9][3]
    cube_state[19] = u_scans[9][5]
    
    cube_state[30] = u_scans[10][1]
    cube_state[32] = u_scans[10][7]
    
    cube_state[39] = u_scans[11][3]
    cube_state[41] = u_scans[11][5]
    
    return ''.join(cube_state)

def generate_scramble(moves=20):
    """Generate a random scramble sequence."""
    basic_moves = ['F', 'B', 'R', 'L', 'D']  # No U moves as we can't execute them
    modifiers = ['', '\'', '2']
    scramble = []
    last_face = None
    
    for _ in range(moves):
        # Don't allow same face moves consecutively
        available_moves = [move for move in basic_moves if move != last_face]
        face = random.choice(available_moves)
        modifier = random.choice(modifiers)
        scramble.append(face + modifier)
        last_face = face
    
    return ' '.join(scramble)

def scramble_cube():
    """Execute a random scramble sequence."""
    print("\nGenerating scramble sequence...")
    scramble = generate_scramble()
    print(f"Scramble sequence: {scramble}")
    print("Executing scramble...")
    send_arduino_command(scramble)
    print("Scramble completed!")
    time.sleep(1)  # Give time for the last move to settle

def close_all_scan_windows(scan_windows):
    """Close all scan windows."""
    for window in scan_windows:
        cv2.destroyWindow(window)
    scan_windows.clear()

def send_arduino_command(cmd, wait_time=None):
    """Send command to Arduino and wait for acknowledgment."""
    global ser
    ser.write(f"{cmd}\n".encode())
    # Wait for acknowledgment from Arduino
    while True:
        if ser.in_waiting:
            response = ser.readline().decode().strip()
            if "completed" in response.lower() or "executed" in response.lower():
                break
    if wait_time:
        time.sleep(wait_time)

def send_compound_move(move):
    """Send a compound move command to Arduino."""
    if move:  # Only send if there's a move to make
        send_arduino_command(move)

def show_menu():
    """Display the main menu and get user choice."""
    print("\n=== Rubik's Cube Solver Menu ===")
    print("1. Calibrate Colors")
    print("2. Solve Cube")
    print("3. Scramble Cube")
    print("4. Exit")
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ")
            if choice in ['1', '2', '3', '4']:
                return int(choice)
            print("Invalid choice. Please enter a number between 1 and 4.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def calibration_mode(cap):
    """Run the calibration process."""
    print("\n=== Color Calibration Mode ===")
    print("Starting color calibration...")
    new_ranges = calibrate_colors(cap, WINDOW_SIZE)
    if new_ranges:
        color_ranges.update(new_ranges)
        save_color_ranges(color_ranges)
        print("Color calibration completed and saved.")
    else:
        print("Calibration aborted.")
    input("\nPress Enter to return to main menu...")

def solve_mode(cap):
    """Run the cube solving process."""
    print("\n=== Solve Mode ===")
    print("Controls:")
    print("- Press 'r' to reset scan process")
    print("- Press 'q' to return to main menu")
    print("\nHold the cube steady in frame for automatic scanning.")
    
    # Create temp directory for scans if it doesn't exist
    temp_dir = "cube_scans"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # Initialize scanning variables
    u_scans = [[] for _ in range(12)]
    current_scan_idx = 0
    prev_face_colors = None
    scan_windows = []
    last_scan_time = time.time()
    last_motor_move_time = 0
    stability_counter = 0
    prev_contour = None
    last_valid_grid = None
    
    cv2.namedWindow("Rubik's Cube Scanner", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Rubik's Cube Scanner", WINDOW_SIZE[0], WINDOW_SIZE[1])
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        frame = cv2.resize(frame, WINDOW_SIZE, interpolation=cv2.INTER_AREA)
        display = frame.copy()
        
        # Color detection and cube processing code
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
        for color, (lower, upper) in color_ranges.items():
            combined_mask |= cv2.inRange(hsv, lower, upper)
        
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cube_detected = False
        best_contour = None
        best_score = float('inf')
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if MIN_CONTOUR_AREA < area < MAX_CONTOUR_AREA:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                if len(approx) >= 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    squareness_score = abs(1 - aspect_ratio)
                    
                    if 0.7 < aspect_ratio < 1.3 and squareness_score < best_score:
                        best_score = squareness_score
                        best_contour = contour
                        cube_detected = True
                        break
        
        if cube_detected and best_contour is not None:
            x, y, w, h = cv2.boundingRect(best_contour)
            grid_size = min(w, h)
            
            center_x = x + w // 2
            center_y = y + h // 2
            pad_x = center_x - grid_size // 2
            pad_y = center_y - grid_size // 2
            
            last_valid_grid = (pad_x, pad_y, grid_size)
            
            cv2.drawContours(display, [best_contour], -1, (0, 255, 0), 2)
            
            grid_cell_size = grid_size // 3
            for i in range(1, 3):
                cv2.line(display, (pad_x + i * grid_cell_size, pad_y), 
                        (pad_x + i * grid_cell_size, pad_y + grid_size), (0, 255, 0), 2)
                cv2.line(display, (pad_x, pad_y + i * grid_cell_size), 
                        (pad_x + grid_size, pad_y + i * grid_cell_size), (0, 255, 0), 2)
            cv2.rectangle(display, (pad_x, pad_y), 
                        (pad_x + grid_size, pad_y + grid_size), (0, 255, 0), 2)
            
            if prev_contour is not None:
                shape_diff = cv2.matchShapes(best_contour, prev_contour, 1, 0.0)
                position_diff = abs(x - prev_x) + abs(y - prev_y)
                
                if shape_diff < 0.3 and position_diff < 20:
                    stability_counter += 1
                else:
                    stability_counter = 0
            
            prev_contour = best_contour
            prev_x, prev_y = x, y
        else:
            stability_counter = 0
            prev_contour = None
        
        current_time = time.time()
        time_since_last_scan = current_time - last_scan_time
        time_since_last_move = current_time - last_motor_move_time
        
        if current_scan_idx < 12:
            if cube_detected:
                status = f"Scan {current_scan_idx + 1}/12 - "
                if time_since_last_move < MOTOR_STABILIZATION_TIME:
                    status += f"Waiting for cube to stabilize: {MOTOR_STABILIZATION_TIME - time_since_last_move:.1f}s"
                elif time_since_last_scan < SCAN_COOLDOWN:
                    status += f"Cooldown: {SCAN_COOLDOWN - time_since_last_scan:.1f}s"
                else:
                    status += f"Stability: {stability_counter}/{STABILITY_THRESHOLD}"
            else:
                status = f"Position cube for scan {current_scan_idx + 1}/12"
            
            cv2.putText(display, status, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Rubik's Cube Scanner", display)
        
        if (cube_detected and stability_counter >= STABILITY_THRESHOLD and 
            current_scan_idx < 12 and 
            time_since_last_scan >= SCAN_COOLDOWN and
            time_since_last_move >= MOTOR_STABILIZATION_TIME):
            
            if last_valid_grid is not None:
                face_colors = []
                pad_x, pad_y, grid_size = last_valid_grid
                grid_cell_size = grid_size // 3
                
                for i in range(3):
                    for j in range(3):
                        y_start = pad_y + i * grid_cell_size
                        y_end = pad_y + (i + 1) * grid_cell_size
                        x_start = pad_x + j * grid_cell_size
                        x_end = pad_x + (j + 1) * grid_cell_size
                        
                        padding = grid_cell_size // 8
                        y_start += padding
                        y_end -= padding
                        x_start += padding
                        x_end -= padding
                        
                        roi = frame[y_start:y_end, x_start:x_end]
                        color = detect_color(roi, color_ranges)
                        face_colors.append(color)
                
                if prev_face_colors is None or not all(a == b for a, b in zip(face_colors, prev_face_colors)):
                    u_scans[current_scan_idx] = face_colors
                    prev_face_colors = face_colors
                    
                    if current_scan_idx < 11:
                        send_compound_move(rotation_sequence[current_scan_idx + 1])
                        last_motor_move_time = time.time()
                    
                    window_name = f"U Face Scan #{current_scan_idx + 1}"
                    scan_windows.append(window_name)
                    
                    scan_image = frame.copy()
                    for i in range(3):
                        for j in range(3):
                            x_start = pad_x + j * grid_cell_size
                            y_start = pad_y + i * grid_cell_size
                            color = face_colors[i * 3 + j]
                            cv2.putText(scan_image, color, 
                                      (x_start + grid_cell_size//4, y_start + grid_cell_size//2), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    
                    cv2.imshow(window_name, scan_image)
                    cv2.imwrite(os.path.join(temp_dir, f"u_scan_{current_scan_idx + 1}_processed.jpg"), scan_image)
                    
                    current_scan_idx += 1
                    if current_scan_idx == 12:
                        print("\nAll scans completed! Processing solution...")
                        send_compound_move("B F'")  # Final rotation
                        last_motor_move_time = time.time()
                        
                        cube_state = construct_cube_from_u_scans(u_scans)
                        try:
                            solution = solve_cube_frblud(cube_state)
                            if solution:
                                print(f"\nSolution: {solution}")
                                send_arduino_command(f"SOLUTION:{solution}")
                        except Exception as e:
                            print(f"Failed to solve: {e}")
                        print("\nPress 'q' to return to main menu.")
                    
                    last_scan_time = current_time
                    stability_counter = 0
                else:
                    print("Duplicate face detected, skipping")
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            close_all_scan_windows(scan_windows)
            cv2.destroyWindow("Rubik's Cube Scanner")
            return
        elif key == ord('r'):
            print("Resetting scan process...")
            close_all_scan_windows(scan_windows)
            u_scans = [[] for _ in range(12)]
            current_scan_idx = 0
            stability_counter = 0
            prev_contour = None
            prev_face_colors = None
            last_scan_time = time.time()
            print("Scan process reset. Ready for scan 1.")

def scramble_mode():
    """Run the cube scrambling process."""
    print("\n=== Scramble Mode ===")
    try:
        scramble_cube()
        print("Scramble completed successfully!")
    except Exception as e:
        print(f"Error during scramble: {e}")
    input("\nPress Enter to return to main menu...")

def main():
    global color_ranges, ser
    
    # Load color ranges from file if available
    loaded_ranges = load_color_ranges()
    if loaded_ranges:
        color_ranges.update(loaded_ranges)
        print("Using previously calibrated color ranges.")
    else:
        print("No saved color ranges found. Calibration will be required.")
    
    # Initialize serial communication with Arduino
    try:
        ser = serial.Serial('COM7', 9600, timeout=1)
        time.sleep(2)
        print("Serial connection established.")
    except Exception as e:
        print(f"Error opening serial port: {e}")
        exit()

    # Initialize video capture
    stream_url = "http://192.168.1.9:8080/video"
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print("Error: Could not open video stream. Trying default camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open default camera.")
            ser.close()
            exit()
    
    # Optimize video capture settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Clear any buffered frames
    for _ in range(5):
        cap.grab()
    
    while True:
        choice = show_menu()
        
        if choice == 1:  # Calibrate
            calibration_mode(cap)
        elif choice == 2:  # Solve
            solve_mode(cap)
        elif choice == 3:  # Scramble
            scramble_mode()
        elif choice == 4:  # Exit
            print("\nExiting program...")
            break
    
    # Cleanup
    cap.release()
    ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()