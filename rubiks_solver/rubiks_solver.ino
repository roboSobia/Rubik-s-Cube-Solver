#include <Stepper.h>

// Shared step and direction pins
#define DIR_PIN 2
#define STEP_PIN 3

// Enable pins for each motor (active LOW)
#define ENABLE_PIN_frontStepper 9
#define ENABLE_PIN_leftStepper 10
#define ENABLE_PIN_rightStepper 11
#define ENABLE_PIN_backStepper 12
#define ENABLE_PIN_downStepper 13

// Stepper settings
#define STEPS_PER_REV 6400  // 6400 steps for 360 degrees
#define MOTOR_RPM 25        // Slow speed for high torque
#define STEPS_PER_90 200   // Steps for 90 degrees (6400 / 4)

// Initialize a single stepper object for shared step/dir pins
Stepper stepper(STEPS_PER_REV, STEP_PIN, DIR_PIN);

// Array to map motors to their enable pins
const int enablePins[5] = {
  ENABLE_PIN_backStepper,
  ENABLE_PIN_rightStepper,
  ENABLE_PIN_leftStepper,
  ENABLE_PIN_downStepper,
  ENABLE_PIN_frontStepper
};

// Enum for motor identification
enum Motor { BACK, RIGHT, LEFT, DOWN, FRONT };

String inputBuffer = ""; // Buffer to store incoming serial data

void setup() {
  // Initialize Serial communication
  Serial.begin(9600);
  while (!Serial) {
    ; // Wait for Serial port to connect
  }
  Serial.println("Rubik's Cube Motor Control Initialized");

  // Set up step and direction pins
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);

  // Set up enable pins and disable all motors initially
  for (int i = 0; i < 5; i++) {
    pinMode(enablePins[i], OUTPUT);
    digitalWrite(enablePins[i], HIGH);  // Disable (active LOW)
  }

  // Set stepper speed
  stepper.setSpeed(MOTOR_RPM);

  delay(3000);  // Initial delay
}

void enableMotor(Motor motor) {
  // Disable all motors
  for (int i = 0; i < 5; i++) {
    digitalWrite(enablePins[i], HIGH);
  }
  // Enable the selected motor (active LOW)
  digitalWrite(enablePins[motor], LOW);
}

void moveStepper(Motor motor, bool clockwise, int steps = 1) {
  // Enable the selected motor
  enableMotor(motor);
  // Move 90 degrees per step (clockwise or counterclockwise)
  for (int i = 0; i < steps; i++) {
    stepper.step(clockwise ? STEPS_PER_90 : -STEPS_PER_90);
    delay(200); // Pause between moves
  }
  // Disable all motors after movement
  for (int i = 0; i < 5; i++) {
    digitalWrite(enablePins[i], HIGH);
  }
}

void executeMove(char move) {
  Motor motor;
  bool clockwise = true;
  int steps = 1;

  // Determine the motor and direction
  switch (move) {
    case 'B': motor = BACK; clockwise = true; break;
    case 'b': motor = BACK; clockwise = false; break;
    case 'R': motor = RIGHT; clockwise = true; break;
    case 'r': motor = RIGHT; clockwise = false; break;
    case 'L': motor = LEFT; clockwise = true; break;
    case 'l': motor = LEFT; clockwise = false; break;
    case 'D': motor = DOWN; clockwise = true; break;
    case 'd': motor = DOWN; clockwise = false; break;
    case 'F': motor = FRONT; clockwise = true; break;
    case 'f': motor = FRONT; clockwise = false; break;
    default: return; // Ignore invalid moves
  }

  moveStepper(motor, clockwise, steps);
}

void executeSolution(String solution) {
  for (int i = 0; i < solution.length(); i++) {
    char c = solution.charAt(i);
    Motor motor;
    bool clockwise = true;
    int steps = 1;

    // Determine the motor based on the face
    if (c == 'B') {
      motor = BACK;
    } else if (c == 'R') {
      motor = RIGHT;
    } else if (c == 'L') {
      motor = LEFT;
    } else if (c == 'D') {
      motor = DOWN;
    } else if (c == 'F') {
      motor = FRONT;
    } else {
      continue; // Skip invalid characters
    }

    // Check for modifiers (' or 2)
    if (i + 1 < solution.length()) {
      char nextChar = solution.charAt(i + 1);
      if (nextChar == '\'') {
        clockwise = false; // Counterclockwise
        i++; // Skip the ' character
      } else if (nextChar == '2') {
        steps = 2; // 180 degrees
        i++; // Skip the 2 character
      }
    }

    // Execute the move
    moveStepper(motor, clockwise, steps);
  }
}

void loop() {
  // Read incoming serial data
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      // Process the complete command
      if (inputBuffer.startsWith("SOLUTION:")) {
        String solution = inputBuffer.substring(9); // Extract solution after "SOLUTION:"
        Serial.println("Received solution: " + solution);
        executeSolution(solution);
        Serial.println("Solution executed.");
      } else {
        // Treat as a rotation command (e.g., "R L'")
        Serial.println("Received rotation command: " + inputBuffer);
        executeSolution(inputBuffer); // Use executeSolution to handle rotation commands
        Serial.println("Rotation completed.");
      }
      inputBuffer = ""; // Clear the buffer
    } else {
      inputBuffer += c; // Add character to buffer
    }
  }
}