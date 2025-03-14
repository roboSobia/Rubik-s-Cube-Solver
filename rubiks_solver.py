import cv2
import numpy as np
from collections import Counter
import kociemba
import os

# Initial HSV color ranges (will be updated during calibration)
color_ranges = {
    "W": (np.array([0, 0, 150]), np.array([180, 60, 255])),      # White
    "R": (np.array([0, 100, 100]), np.array([10, 255, 255])),    # Red
    "G": (np.array([35, 100, 100]), np.array([85, 255, 255])),   # Green
    "Y": (np.array([25, 100, 100]), np.array([35, 255, 255])),   # Yellow
    "O": (np.array([10, 100, 100]), np.array([25, 255, 255])),   # Orange
    "B": (np.array([85, 100, 100]), np.array([130, 255, 255]))   # Blue
}

def calibrate_colors(cap, window_size, color_names=["W", "R", "G", "Y", "O", "B"]):
    """Calibrate HSV ranges by sampling the center of each face."""
    print("\nStarting calibration. Show each face’s center when prompted.")
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

def solve_cube_frblud(cube_frblud):
    try:
        color_map, cube_frblud_remapped = remap_colors_to_kociemba(cube_frblud)
        scrambled_kociemba = remap_cube_to_kociemba(cube_frblud_remapped)
        solved_frblud = get_solved_state(cube_frblud, color_map)
        _, solved_frblud_remapped = remap_colors_to_kociemba(solved_frblud)
        solved_kociemba = remap_cube_to_kociemba(solved_frblud_remapped)
        validate_cube(scrambled_kociemba, "Scrambled Kociemba")
        validate_cube(solved_kociemba, "Solved Kociemba")
        solution = kociemba.solve(scrambled_kociemba, solved_kociemba)
        return solution
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

def main():
    global color_ranges  # Declare color_ranges as global at the start
    
    temp_dir = "temp"
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
    print("Align cube in this order: Front, Right, Back, Left, Up, Down")
    print("Colors: B=Blue, Y=Yellow, R=Red, G=Green, W=White, O=Orange")
    print("Press 'q' to quit, 'c' to capture, 'r' to restart, 'a' to calibrate.")
    
    cube_faces = {}
    cube_state = []
    face_order = ["Front", "Right", "Back", "Left", "Up", "Down"]
    current_face_idx = 0
    
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
        
        if current_face_idx < len(face_order):
            instruction = f"Align {face_order[current_face_idx]} face (B=Blue, Y=Yellow, R=Red, G=Green, W=White, O=Orange)"
            cv2.putText(display, instruction, (pad_x, pad_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(display, "All faces captured. Press 'r' to restart, 'a' to calibrate, or 'q' to quit.", 
                        (pad_x, pad_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Rubik's Cube Scanner", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if current_face_idx < len(face_order):
                face_name = face_order[current_face_idx]
                image = frame.copy()
                cube_faces[face_name] = image
                print(f"\nCaptured {face_name} face")
                
                face_colors = []
                print(f"What I see on the {face_name} face (3x3 grid):")
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
                
                cube_state.extend(face_colors)
                image_path = os.path.join(temp_dir, f"{face_name.lower()}_processed.jpg")
                cv2.imwrite(image_path, image)
                cv2.imshow(f"{face_name} Face Processed", image)
                
                current_face_idx += 1
                if current_face_idx == len(face_order):
                    print("\nAll faces captured!")
                    print_full_cube_state(cube_state)
                    try:
                        solution = solve_cube_frblud(''.join(cube_state))
                        if solution:
                            print(f"\nSolution: {solution}")
                            print("Apply these moves to solve your cube!")
                    except Exception as e:
                        print(f"Failed to solve: {e}")
                    print("\nPress 'q' to quit, 'r' to restart, 'a' to calibrate.")
        elif key == ord('r'):
            cube_faces = {}
            cube_state = []
            current_face_idx = 0
            print("\nRestarting cube scan...")
        elif key == ord('a'):
            print("\nEntering calibration mode...")
            calibrated_ranges = calibrate_colors(cap, WINDOW_SIZE)
            if calibrated_ranges is not None:
                color_ranges.clear()  # Clear old ranges
                color_ranges.update(calibrated_ranges)  # Update with new ranges
                print("Calibration complete. Using custom color ranges.")
                # Reset scanning state
                cube_faces = {}
                cube_state = []
                current_face_idx = 0
                print("Restarting cube scan...")
            else:
                print("Calibration aborted. Resuming with previous settings.")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()