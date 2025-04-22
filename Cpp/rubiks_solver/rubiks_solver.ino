#include <AccelStepper.h>
#include <WiFi.h>
#include <WebServer.h>
#include <WebSocketsServer.h>
#include <SPIFFS.h>

// WiFi credentials - replace with your network info
const char* ssid = "YourWiFiName";
const char* password = "YourWiFiPassword";

// Create web server and websocket server
WebServer server(80);
WebSocketsServer webSocket = WebSocketsServer(81);

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
#define stepsPerRevolution 200
#define microsteppingInverse 32

#define stepsForHalfCycle (stepsPerRevolution * microsteppingInverse / 4)  // 3200 steps

// Initialize stepper motors
AccelStepper backStepper(AccelStepper::DRIVER, STEP_PIN_backStepper, DIR_PIN_backStepper);
AccelStepper rightStepper(AccelStepper::DRIVER, STEP_PIN_rightStepper, DIR_PIN_rightStepper);
AccelStepper leftStepper(AccelStepper::DRIVER, STEP_PIN_leftStepper, DIR_PIN_leftStepper);
AccelStepper downStepper(AccelStepper::DRIVER, STEP_PIN_downStepper, DIR_PIN_downStepper);
AccelStepper frontStepper(AccelStepper::DRIVER, STEP_PIN_frontStepper, DIR_PIN_frontStepper);

// HTML page
const char index_html[] PROGMEM = R"rawliteral(
<!DOCTYPE HTML>
<html>
<head>
  <title>ESP32 Rubik's Cube Solver</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: Arial; text-align: center; margin:0px auto; padding-top: 30px; }
    .button { padding: 10px 20px; font-size: 20px; margin: 10px; cursor: pointer; }
    #solution { width: 80%; height: 100px; margin: 20px auto; font-size: 18px; }
  </style>
</head>
<body>
  <h1>ESP32 Rubik's Cube Solver</h1>
  <textarea id="solution" placeholder="Enter solution sequence here..."></textarea><br>
  <button class="button" id="sendBtn">Execute Solution</button>
  <button class="button" id="scanBtn">Scan Cube</button>
  <div id="status">Status: Ready</div>
  
  <script>
    var websocket;
    window.addEventListener('load', onLoad);
    
    function initWebSocket() {
      console.log('Trying to open a WebSocket connection...');
      // Using port 81 for WebSocket
      websocket = new WebSocket('ws://' + window.location.hostname + ':81/');
      websocket.onopen = onOpen;
      websocket.onclose = onClose;
      websocket.onmessage = onMessage;
    }
    
    function onOpen(event) {
      console.log('Connection opened');
      document.getElementById('status').innerHTML = 'Status: Connected';
    }
    
    function onClose(event) {
      console.log('Connection closed');
      document.getElementById('status').innerHTML = 'Status: Disconnected';
      setTimeout(initWebSocket, 2000);
    }
    
    function onMessage(event) {
      document.getElementById('status').innerHTML = 'Status: ' + event.data;
    }
    
    function onLoad(event) {
      initWebSocket();
      document.getElementById('sendBtn').addEventListener('click', function() {
        var solution = document.getElementById('solution').value;
        if(solution.length > 0) {
          websocket.send('SOLVE:' + solution);
        }
      });
      document.getElementById('scanBtn').addEventListener('click', function() {
        websocket.send('SCAN');
      });
    }
  </script>
</body>
</html>
)rawliteral";

void moveStepper(AccelStepper &stepper, bool left) {
  stepper.move(left ? stepsForHalfCycle : -stepsForHalfCycle);
  while (stepper.distanceToGo() != 0) {
    stepper.run();
  }
}

void executeSolution(String solution) {
  webSocket.broadcastTXT("Executing solution: " + solution);
  
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
    
    // Send current move to connected clients
    if (c == 'B' || c == 'R' || c == 'L' || c == 'D' || c == 'F') {
      String moveInfo = String("Executing move: ") + c;
      if (i + 1 < solution.length()) {
        if (solution.charAt(i + 1) == '\'') {
          moveInfo += "'";
        } else if (solution.charAt(i + 1) == '2') {
          moveInfo += "2";
        }
      }
      webSocket.broadcastTXT(moveInfo);
    }
    
    delay(1000);
  }
  
  webSocket.broadcastTXT("Solution completed");
}

void getInitState() {
  webSocket.broadcastTXT("Starting cube scan sequence");
  
  for (int i = 0; i < 6; i++) {
    // Sequence => scan => R L' => scan => B F // Repeat for all 6 faces
    webSocket.broadcastTXT("Scanning face " + String(i+1) + "/6");
    delay(3000);   // scan
    executeSolution("R L'");
    delay(3000);
    executeSolution("B F'");
  }
  
  webSocket.broadcastTXT("Scan completed");
}

void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.printf("[%u] Disconnected!\n", num);
      break;
    case WStype_CONNECTED:
      {
        IPAddress ip = webSocket.remoteIP(num);
        Serial.printf("[%u] Connected from %d.%d.%d.%d\n", num, ip[0], ip[1], ip[2], ip[3]);
        webSocket.sendTXT(num, "Connected to ESP32");
      }
      break;
    case WStype_TEXT:
      Serial.printf("[%u] got text: %s\n", num, payload);
      
      // Handle received message
      String message = String((char*)payload);
      if (message.startsWith("SOLVE:")) {
        String solution = message.substring(6); // Extract the solution part
        executeSolution(solution);
      } 
      else if (message == "SCAN") {
        getInitState();
      }
      break;
  }
}

void setup() {
  // Start serial communication at 115200 baud rate
  Serial.begin(115200);
  
  // Initialize stepper motors
  backStepper.setMaxSpeed(1000); backStepper.setAcceleration(500);
  rightStepper.setMaxSpeed(1000); rightStepper.setAcceleration(500);
  leftStepper.setMaxSpeed(1000); leftStepper.setAcceleration(500);
  downStepper.setMaxSpeed(1000); downStepper.setAcceleration(500);
  frontStepper.setMaxSpeed(1000); frontStepper.setAcceleration(500);
  
  // Connect to WiFi
  WiFi.begin(ssid, password);
  Serial.println("");
  
  // Wait for connection
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("");
  Serial.print("Connected to ");
  Serial.println(ssid);
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
  
  // Initialize WebSocket server
  webSocket.begin();
  webSocket.onEvent(webSocketEvent);
  
  // Handle Web Server requests
  server.on("/", HTTP_GET, []() {
    server.send(200, "text/html", index_html);
  });
  
  // Start web server
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  webSocket.loop();
  server.handleClient();
}