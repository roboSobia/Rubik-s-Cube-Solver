void setup() {
  // Start serial communication at 9600 baud rate
  Serial.begin(9600);
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
    // executeSolution(solution);
  }
}
