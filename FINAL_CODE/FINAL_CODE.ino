/* Edge Impulse ingestion SDK
 * Copyright (c) 2022 EdgeImpulse Inc.
 * Licensed under Apache License 2.0
 */

/* Includes ---------------------------------------------------------------- */
#include <V2March_inferencing.h> // Replace with your new model header
#include <Arduino_LSM9DS1.h>     // IMU library
#include <ArduinoBLE.h>          // BLE library

/* Constants --------------------------------------------------------------- */
#define SAMPLE_RATE 400          // 400 Hz, matches training
#define WINDOW_SIZE 800          // 2 seconds at 400 Hz (2000 ms)
#define STRIDE 400               // 1 second stride (1000 ms)
#define INTERVAL_US (1000000 / SAMPLE_RATE) // 2500 µs
#define CONVERT_G_TO_MS2 9.80665f // Convert g to m/s²
#define MAX_ACCEPTED_RANGE 2.0f  // Matches Edge Impulse Nano 33 BLE Sense default range
#define PREDICTION_INTERVAL 2000 // 2 seconds between BLE updates
#define ANOMALY_THRESHOLD 43      

/* BLE Configuration ------------------------------------------------------- */
BLEService predictionService("0000180c-0000-1000-8000-00805f9b34fb");
BLEStringCharacteristic predictionCharacteristic("00002a56-0000-1000-8000-00805f9b34fb", BLERead | BLENotify, 32);

/* Private Variables ------------------------------------------------------- */
static float sensorData[WINDOW_SIZE][6]; // Buffer for accel + gyro (19.2 KB)
static int sampleIndex = 0;
static const bool debug_nn = false;      // Set to true for debug output
static unsigned long lastPredictionTime = 0; // For BLE update timing

/* Function Prototypes ----------------------------------------------------- */
void collect_samples();
void run_inference_and_transmit();
void setupBLE();

/**
 * @brief Arduino setup function
 */
void setup() {
  Serial.begin(115200);
  // while (!Serial);

  // Initialize IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // Setup BLE
  setupBLE();

  Serial.println("Edge Impulse Continuous Inference with BLE and Anomaly Detection");
  Serial.println("Classes: normal, abnormal, idle");
}

/**
 * @brief Setup BLE service and advertising
 */
void setupBLE() {
  if (!BLE.begin()) {
    Serial.println("Failed to start BLE!");
    while (1);
  }

  BLE.setConnectionInterval(10, 200); // Min 10ms, Max 200ms
  BLE.setSupervisionTimeout(10000);   // 10s timeout
  BLE.setLocalName("VibrationPredictor");
  BLE.setAdvertisedService(predictionService);
  predictionService.addCharacteristic(predictionCharacteristic);
  BLE.addService(predictionService);
  predictionCharacteristic.writeValue("Idle: 0.00000");
  BLE.advertise();

  Serial.println("BLE Device Ready & Advertising...");
  Serial.println("Service UUID: 0000180c-0000-1000-8000-00805f9b34fb");
  Serial.println("Characteristic UUID: 00002a56-0000-1000-8000-00805f9b34fb");
}

/**
 * @brief Arduino loop function
 */
void loop() {
  BLEDevice central = BLE.central();

  if (central) {
    Serial.print("Connected to central: ");
    Serial.println(central.address());

    while (central.connected()) {
      // Collect samples if buffer isn’t full
      if (sampleIndex < WINDOW_SIZE) {
        collect_samples();
      }

      // Run inference when buffer is full
      if (sampleIndex >= WINDOW_SIZE) {
        run_inference_and_transmit();

        // Slide window: shift data left by stride
        for (int i = 0; i < WINDOW_SIZE - STRIDE; i++) {
          for (int j = 0; j < 6; j++) {
            sensorData[i][j] = sensorData[i + STRIDE][j];
          }
        }
        sampleIndex = WINDOW_SIZE - STRIDE; // Reset index to fill remaining samples
      }

      // Yield to avoid blocking
      delay(1);
    }

    Serial.print("Disconnected from central: ");
    Serial.println(central.address());
    BLE.end();
    BLE.begin();
    setupBLE(); // Restart advertising after disconnect
  }

  delay(100);
}

/**
 * @brief Collect samples from IMU at 400 Hz
 */
void collect_samples() {
  static unsigned long nextSampleTime = micros();
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable() && micros() >= nextSampleTime) {
    // Read accel (g) and gyro (degrees/s)
    IMU.readAcceleration(sensorData[sampleIndex][0], sensorData[sampleIndex][1], sensorData[sampleIndex][2]);
    IMU.readGyroscope(sensorData[sampleIndex][3], sensorData[sampleIndex][4], sensorData[sampleIndex][5]);

    // Scale accel to m/s² and clip to training range (matches Edge Impulse default)
    for (int i = 0; i < 3; i++) {
      sensorData[sampleIndex][i] *= CONVERT_G_TO_MS2; // Convert g to m/s²
      if (fabs(sensorData[sampleIndex][i]) > MAX_ACCEPTED_RANGE * CONVERT_G_TO_MS2) {
        sensorData[sampleIndex][i] = ei_get_sign(sensorData[sampleIndex][i]) * MAX_ACCEPTED_RANGE * CONVERT_G_TO_MS2;
      }
    }

    // Gyro remains in degrees/s (matches training)
    sampleIndex++;
    nextSampleTime += INTERVAL_US;
  }
}

/**
 * @brief Run inference, check for anomalies, and transmit result via BLE
 */
void run_inference_and_transmit() {
  // Flatten buffer for Edge Impulse
  float features[WINDOW_SIZE * 6];
  for (int i = 0; i < WINDOW_SIZE; i++) {
    for (int j = 0; j < 6; j++) {
      features[i * 6 + j] = sensorData[i][j];
    }
  }

  // Create signal
  signal_t signal;
  int err = numpy::signal_from_buffer(features, WINDOW_SIZE * 6, &signal);
  if (err != 0) {
    ei_printf("ERR: Signal creation failed (%d)\r\n", err);
    return;
  }

  // Run classifier
  ei_impulse_result_t result = {0};
  err = run_classifier(&signal, &result, debug_nn);
  if (err != EI_IMPULSE_OK) {
    ei_printf("ERR: Inference failed (%d)\r\n", err);
    return;
  }

  // Find highest probability class
  float max_value = 0.0;
  int max_index = 0;
  for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
    if (result.classification[ix].value > max_value) {
      max_value = result.classification[ix].value;
      max_index = ix;
    }
  }
  String prediction = result.classification[max_index].label;

  // Check for anomaly
  String final_label = prediction;
  float final_score = max_value;
#if EI_CLASSIFIER_HAS_ANOMALY == 1
  float anomalyScore = result.anomaly;
  if (anomalyScore > ANOMALY_THRESHOLD) {
    final_label = "Anomaly";
    final_score = anomalyScore;
  }
  ei_printf("Anomaly score: %.3f\r\n", anomalyScore);
#endif

  // Print prediction and timing
  ei_printf("Prediction: %s (Confidence: %.5f)\r\n", prediction.c_str(), max_value);
  ei_printf("Final Label: %s (Score: %.5f)\r\n", final_label.c_str(), final_score);
  ei_printf("DSP: %d ms, Classification: %d ms, Anomaly: %d ms\r\n", 
            result.timing.dsp, result.timing.classification, result.timing.anomaly);
  ei_printf("Probabilities - normal: %.5f, abnormal: %.5f, idle: %.5f\r\n",
            result.classification[0].value, result.classification[1].value, result.classification[2].value);

  // Transmit via BLE every PREDICTION_INTERVAL (2 seconds)
  unsigned long currentTime = millis();
  if (currentTime - lastPredictionTime >= PREDICTION_INTERVAL) {
    String predictionData = final_label + ": " + String(final_score, 5);
    if (predictionCharacteristic.subscribed()) {
      predictionCharacteristic.writeValue(predictionData);
      Serial.print("Sent over BLE: ");
      Serial.println(predictionData);
    } else {
      Serial.println("Central not subscribed to notifications");
    }
    lastPredictionTime = currentTime;
  }
}

/**
 * @brief Return the sign of the number
 */
float ei_get_sign(float number) {
  return (number >= 0.0) ? 1.0 : -1.0;
}