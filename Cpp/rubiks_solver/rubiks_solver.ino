#include <AccelStepper.h>
#include <WiFi.h>

// Step pins
#define STEP_PIN_backStepper 18
#define STEP_PIN_rightStepper 21
#define STEP_PIN_leftStepper 23
#define STEP_PIN_downStepper 26
#define STEP_PIN_frontStepper 32

// Direction pins 
#define DIR_PIN_backStepper 19
#define DIR_PIN_rightStepper 22
#define DIR_PIN_leftStepper 25
#define DIR_PIN_downStepper 27
#define DIR_PIN_frontStepper 33

// Stepper settings
// NEMA17 stepper motors have: 200 steps per revolution
// DRV8825 has inverse of microstepping = 32
#define stepsPerRevolution 200
#define microsteppingInverse 32

#define stepsForHalfCycle (stepsPerRevolution * microsteppingInverse / 4)  // 3200 steps

const char* ssid = "WE4DD9A3";
const char* password = "aa07002c";
int numOfScans = 0;
bool firstSequence = false; // R L'

// Initialize stepper motors
AccelStepper backStepper(AccelStepper::DRIVER, STEP_PIN_backStepper, DIR_PIN_backStepper);
AccelStepper rightStepper(AccelStepper::DRIVER, STEP_PIN_rightStepper, DIR_PIN_rightStepper);
AccelStepper leftStepper(AccelStepper::DRIVER, STEP_PIN_leftStepper, DIR_PIN_leftStepper);
AccelStepper downStepper(AccelStepper::DRIVER, STEP_PIN_downStepper, DIR_PIN_downStepper);
AccelStepper frontStepper(AccelStepper::DRIVER, STEP_PIN_frontStepper, DIR_PIN_frontStepper);

// TCP server on port 80
WiFiServer server(80);

void setup() {
  // Start serial communication at 115200 baud rate
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  // Connect to the wifi and print the ESP32 IP address
  Serial.println("Connecting to WiFi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected!");
  Serial.print("ESP32 IP Address: ");
  Serial.println(WiFi.localIP());

  server.begin();
  
  backStepper.setMaxSpeed(1000); backStepper.setAcceleration(500);
  rightStepper.setMaxSpeed(1000); rightStepper.setAcceleration(500);
  leftStepper.setMaxSpeed(1000); leftStepper.setAcceleration(500);
  downStepper.setMaxSpeed(1000); downStepper.setAcceleration(500);
  frontStepper.setMaxSpeed(1000); frontStepper.setAcceleration(500);
}

void moveStepper(AccelStepper &stepper, bool left) {
  stepper.move(left ? stepsForHalfCycle : -stepsForHalfCycle);
  while (stepper.distanceToGo() != 0) {
    stepper.run();
  }
}

void stopMotors() {
  backStepper.stop();
  rightStepper.stop();
  leftStepper.stop();
  downStepper.stop();
  frontStepper.stop();
}

void executeSolution(String solution) {
  for (int i = 0; i < solution.length(); i++) {
      char c = solution.charAt(i);
      if (c == 'B' && i + 1 < solution.length() && solution.charAt(i + 1) != '\'') {        // green
        moveStepper(backStepper, true);
      } else if (c == 'R' && i + 1 < solution.length() && solution.charAt(i + 1) != '\'') {     // orange
        moveStepper(rightStepper, true);
      } else if (c == 'L' && i + 1 < solution.length() && solution.charAt(i + 1) != '\'') {      // red
        moveStepper(leftStepper, true);
      } else if (c == 'D' && i + 1 < solution.length() && solution.charAt(i + 1) != '\'') {     //yellow
        moveStepper(downStepper, true);
      } else if (c == 'F' && i + 1 < solution.length() && solution.charAt(i + 1) != '\'') {   // Blue
        moveStepper(frontStepper, true);
      } else if (c == '2') {
        if (solution.charAt(i - 1) == 'B') {
          moveStepper(backStepper, true);
        } else if (solution.charAt(i - 1) == 'R') {
          moveStepper(rightStepper, true);
        } else if (solution.charAt(i - 1) == 'L') {
          moveStepper(leftStepper, true);
        } else if (solution.charAt(i - 1) == 'D') {
          moveStepper(downStepper, true);
        } else if (solution.charAt(i - 1) == 'F') {
          moveStepper(frontStepper, true);
        }
      } else if (c == '\'') {
        if (solution.charAt(i - 1) == 'B') {
          moveStepper(backStepper, false);
        } else if (solution.charAt(i - 1) == 'R') {
          moveStepper(rightStepper, false);
        } else if (solution.charAt(i - 1) == 'L') {
          moveStepper(leftStepper, false);
        } else if (solution.charAt(i - 1) == 'D') {
          moveStepper(downStepper, false);
        } else if (solution.charAt(i - 1) == 'F') {
          moveStepper(frontStepper, false);
        }
      }
      delay(1000);
    }
}

// return 0 if all scans are finished
// return 1 if there are scans left
int scan() {
  // No more scans left
  if (numOfScans > 12) {
    return 0;
  }

  //  Sequence => scan => R L' => scan => B F // Repeat for all 6 faces
  if (firstSequence) {
    executeSolution("R L'");
  } else {
    executeSolution("B F'");
  }
  delay(1000);

  firstSequence = !firstSequence;
  numOfScans++;

  return 1;
}

String getSolution(WiFiClient client) {
  Serial.println("New Client Connected!");
  String solution;
  unsigned long timeout = millis();
  while (client.connected() && millis() - timeout < 10000) { // 10 seconds timeout
    if (client.available()) {
      timeout = millis(); // Reset timeout
      String command = client.readStringUntil('\n');
      command.trim();
      Serial.println("Received: " + command);

      if (command == "STOP") {
        stopMotors();
        client.println("ACK_STOP");
      } else if (command == "MOVE") {
        if(!scan()) {
          // All scans are done => send 
          client.println("SCANS DONE");
        } else {
          client.println("ACK_MOVE");
        }
      } else if (command.startsWith("SOLVE:")) {
        solution = command.substring(6);
        client.println("ACK_SOLVE");
        Serial.println("Solution: " + solution);
      }
    }
  }
  return solution;
}

void loop() {
  WiFiClient client = server.available();
  String solution;
  if (client) {
    solution = getSolution(client);
  }
  client.stop();

  if(solution.length() > 0) {
    executeSolution(solution);
  }
  
  Serial.println("Client Disconnected");
  // B2 R' F' R2 B' R F B2 L R L F2 B2 R' L' D R L F2 B2 R' L' R2 B2 L2 D' R2 F2 D2 F2 R L F2 B2 R' L' D R L F2 B2 R' L' 
}
