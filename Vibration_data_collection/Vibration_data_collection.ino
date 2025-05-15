#include <Arduino_LSM9DS1.h>

#define SAMPLE_RATE 400 // Target 400 Hz
#define WINDOW_SIZE 800 // 2 seconds at 400 Hz
#define INTERVAL_US (1000000 / SAMPLE_RATE) // 2500 µs

float sensorData[WINDOW_SIZE][6]; // 800 samples, 6 axes, 19.2 KB
int sampleCount = 0;
int windowCount = 0;

void setup() {
  Serial.begin(1000000); // 1 Mbps
  while (!Serial); 
  
  if (!IMU.begin()) {
    Serial.println("IMU failed!");
    while (1);
  }
  Serial.println("rel_timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z"); // CSV header
}

void loop() {
  unsigned long windowStart = micros();
  sampleCount = 0;

  // Collect exactly 800 samples (~400 Hz)
  unsigned long nextSampleTime = windowStart;
  while (sampleCount < WINDOW_SIZE) {
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable() && micros() >= nextSampleTime) {
      IMU.readAcceleration(sensorData[sampleCount][0], sensorData[sampleCount][1], sensorData[sampleCount][2]);
      IMU.readGyroscope(sensorData[sampleCount][3], sensorData[sampleCount][4], sensorData[sampleCount][5]);
      sampleCount++;
      nextSampleTime += INTERVAL_US;
    }
  }
  unsigned long collectionEnd = micros();
  
  // Stream window
  for (int i = 0; i < WINDOW_SIZE; i++) {
    unsigned long relTimestamp = i * INTERVAL_US;
    Serial.print(relTimestamp);
    Serial.print(",");
    Serial.print(sensorData[i][0], 6);
    Serial.print(",");
    Serial.print(sensorData[i][1], 6);
    Serial.print(",");
    Serial.print(sensorData[i][2], 6);
    Serial.print(",");
    Serial.print(sensorData[i][3], 6);
    Serial.print(",");
    Serial.print(sensorData[i][4], 6);
    Serial.print(",");
    Serial.println(sensorData[i][5], 6);
  }
  Serial.println("END_WINDOW"); // Marker for end of window
  unsigned long transmissionEnd = micros();

  // Diagnostics every 5th window
  if (windowCount % 5 == 0) {
    Serial.print("Collection time (ms): ");
    Serial.println((collectionEnd - windowStart) / 1000);
    Serial.print("Transmission time (ms): ");
    Serial.println((transmissionEnd - collectionEnd) / 1000);
  }

  sampleCount = 0;
  windowCount++;
  delay(500); // 0.5s delay
}