# Rubik's Cube Solver

A simple python script that scans a Rubik's Cube, detects colors, and solves it using the Kociemba algorithm. The solution will be sent to an ESP32 later in the project for execution.

## Requirements
Make sure you have the following installed:
- Python
- OpenCV
- Numpy
- kociemba

You can install the required packages using:
```sh
pip install opencv-python numpy kociemba
```

## Running the Project
1. Connect a camera or use an IP camera stream.
2. Run the script:
   ```sh
   python rubiks_solver.py
   ```
3. Follow the instructions to scan all six faces of the cube.
4. The program will process the scanned colors and output the solution.
5. The solution will be sent to the Arduino, which will execute the moves.

## Controls
- Press `c` to capture the current face.
- Press `r` to restart scanning.
- Press `q` to quit the program.

## Basic Rubik's Cube Notation
The Rubik's Cube is manipulated using a set of standard move notations:
- **U** (Up) - Rotate the top face clockwise.
- **U'** (Up Inverse) - Rotate the top face counterclockwise.
- **U2** - Rotate the top face 180 degrees.
- **D** (Down) - Rotate the bottom face clockwise.
- **D'** (Down Inverse) - Rotate the bottom face counterclockwise.
- **D2** - Rotate the bottom face 180 degrees.
- **L** (Left) - Rotate the left face clockwise.
- **L'** (Left Inverse) - Rotate the left face counterclockwise.
- **L2** - Rotate the left face 180 degrees.
- **R** (Right) - Rotate the right face clockwise.
- **R'** (Right Inverse) - Rotate the right face counterclockwise.
- **R2** - Rotate the right face 180 degrees.
- **F** (Front) - Rotate the front face clockwise.
- **F'** (Front Inverse) - Rotate the front face counterclockwise.
- **F2** - Rotate the front face 180 degrees.
- **B** (Back) - Rotate the back face clockwise.
- **B'** (Back Inverse) - Rotate the back face counterclockwise.
- **B2** - Rotate the back face 180 degrees.

## Notes
- Ensure proper lighting for accurate color detection.
- Position the cube correctly in front of the camera as guided by the interface.
