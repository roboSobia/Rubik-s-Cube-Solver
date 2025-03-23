#include <AccelStepper.h>

// Step pins
#define STEP_PIN_stepper1 18
#define STEP_PIN_stepper2 21
#define STEP_PIN_stepper3 23
#define STEP_PIN_stepper4 26
#define STEP_PIN_stepper5 32

// Direction pins 
#define DIR_PIN_stepper1 19
#define DIR_PIN_stepper2 22
#define DIR_PIN_stepper3 25
#define DIR_PIN_stepper4 27
#define DIR_PIN_stepper5 33

// Stepper settings
// NEMA17 stepper motors have: 200 steps per revolution
// DRV8825 has inverse of microstepping = 32
#define stepsPerRevolution 200
#define microsteppingInverse 32

#define stepsForHalfCycle (stepsPerRevolution * microsteppingInverse / 4)  // 3200 steps

// Initialize stepper motors
AccelStepper stepper1(AccelStepper::DRIVER, STEP_PIN_stepper1, DIR_PIN_stepper1);
AccelStepper stepper2(AccelStepper::DRIVER, STEP_PIN_stepper2, DIR_PIN_stepper2);
AccelStepper stepper3(AccelStepper::DRIVER, STEP_PIN_stepper3, DIR_PIN_stepper3);
AccelStepper stepper4(AccelStepper::DRIVER, STEP_PIN_stepper4, DIR_PIN_stepper4);
AccelStepper stepper5(AccelStepper::DRIVER, STEP_PIN_stepper5, DIR_PIN_stepper5);

void setup() {
  
  // Start serial communication at 115200 baud rate
  Serial.begin(115200);
  
  stepper1.setMaxSpeed(1000); stepper1.setAcceleration(500);
  stepper2.setMaxSpeed(1000); stepper2.setAcceleration(500);
  stepper3.setMaxSpeed(1000); stepper3.setAcceleration(500);
  stepper4.setMaxSpeed(1000); stepper4.setAcceleration(500);
  stepper5.setMaxSpeed(1000); stepper5.setAcceleration(500);
}

void moveStepper(AccelStepper &stepper, bool left) {
  stepper.move(left ? stepsForHalfCycle : -stepsForHalfCycle);
  while (stepper.distanceToGo() != 0) {
    stepper.run();
  }
}

void executeSolution(String solution) {
  for (int i = 0; i < solution.length(); i++) {
      char c = solution.charAt(i);
      if (c == 'B' && i + 1 < solution.length() && solution.charAt(i + 1) != '\'') {        // green
        moveStepper(stepper1, true);
      } else if (c == 'R' && i + 1 < solution.length() && solution.charAt(i + 1) != '\'') {     // orange
        moveStepper(stepper2, true);
      } else if (c == 'L' && i + 1 < solution.length() && solution.charAt(i + 1) != '\'') {      // red
        moveStepper(stepper3, true);
      } else if (c == 'D' && i + 1 < solution.length() && solution.charAt(i + 1) != '\'') {     //yellow
        moveStepper(stepper4, true);
      } else if (c == 'F' && i + 1 < solution.length() && solution.charAt(i + 1) != '\'') {   // Blue
        moveStepper(stepper5, true);
      } else if (c == '2') {
        if (solution.charAt(i - 1) == 'B') {
          moveStepper(stepper1, true);
        } else if (solution.charAt(i - 1) == 'R') {
          moveStepper(stepper2, true);
        } else if (solution.charAt(i - 1) == 'L') {
          moveStepper(stepper3, true);
        } else if (solution.charAt(i - 1) == 'D') {
          moveStepper(stepper4, true);
        } else if (solution.charAt(i - 1) == 'F') {
          moveStepper(stepper5, true);
        }
      } else if (c == '\'') {
        if (solution.charAt(i - 1) == 'B') {
          moveStepper(stepper1, false);
        } else if (solution.charAt(i - 1) == 'R') {
          moveStepper(stepper2, false);
        } else if (solution.charAt(i - 1) == 'L') {
          moveStepper(stepper3, false);
        } else if (solution.charAt(i - 1) == 'D') {
          moveStepper(stepper4, false);
        } else if (solution.charAt(i - 1) == 'F') {
          moveStepper(stepper5, false);
        }
      }
      delay(1000);
    }
}

void loop() {
  // Check if data is available to read
  if (Serial.available() > 0) {
    // Read the incoming data as a string
    String solution = Serial.readString();
    // Print the received solution to the Serial Monitor 
    Serial.print("Received: ");
    Serial.println(solution);

    // Process the solution
    executeSolution(solution);
  }
  // B2 R' F' R2 B' R F B2 L R L F2 B2 R' L' D R L F2 B2 R' L' R2 B2 L2 D' R2 F2 D2 F2 R L F2 B2 R' L' D R L F2 B2 R' L' 
}
