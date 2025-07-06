Vibration Analysis Using Machine Learning
Welcome to the Vibration Analysis Using Machine Learning project! This repository delivers a cutting-edge solution for monitoring motor and fan health through vibration data analysis using machine learning (ML). Built with an Arduino Nano BLE 33, Python scripting, and Edge Impulse, this project identifies normal, idle, and abnormal (e.g., imbalance) states, with anomaly detection for unforeseen faults. Ideal for engineers, researchers, and hobbyists in IoT, ML, and mechanical diagnostics.

🎯 Project Overview
This project tackles vibration-based condition monitoring for motors and fans using a dual-sensor setup (accelerometer and gyroscope) on the Arduino Nano BLE 33. Data is sampled at 400 Hz, processed into CSV files, and trained on Edge Impulse with a custom neural network. The trained model runs onboard, transmitting predictions via Bluetooth Low Energy (BLE) to a GUI for real-time monitoring. Key highlights include:

Data Collection: Automated 8-axis vibration sampling (6 accelerometer, 2 gyroscope axes).
ML Pipeline: Spectral preprocessing and a convolutional neural network (CNN) for classification, enhanced with K-Means for anomaly detection.
Deployment: Optimized for Arduino Nano BLE 33 with BLE-based visualization.
Inspiration: Built upon open-source foundations like Single_Fault_Vibration_Analysis_Using_MachineLearning, refined for broader fault detection.
Applications: Predictive maintenance, fault detection in industrial or DIY systems.
🚀 Getting Started
Prerequisites
Hardware: Arduino Nano BLE 33, motor/fan setup (normal and faulty configurations).
Software:
Arduino IDE (for sketches).
Python 3.8+ with serial, csv, tkinter, and os libraries.
Edge Impulse account and CLI (for model training).
Web browser (for GUI via BLE).
Installation
Clone this repository:

git clone https://github.com/yourusername/vibration-analysis-ml.git
cd vibration-analysis-ml
Install Python dependencies:
bash

pip install -r requirements.txt

📊 Data Collection
Step 1: Run Arduino Sketch
File: Vibration_Data_collection.ino
Details:
Samples vibration data at 400 Hz for 2-second intervals (800 readings).
Captures 8 axes: 6 from the LSM9DS1 IMU (3-axis accelerometer, 3-axis gyroscope).
Data is output to the Serial Monitor as CSV-like rows (timestamp, ax, ay, az, gx, gy, gz).
Setup: Upload to Arduino Nano BLE 33, connect to motor/fan, and open Serial Monitor (115200 baud).
Step 2: Run Python Script
File: data_collection.py
Details:
Reads serial data and saves it to CSV files.
Features a Tkinter GUI to:
Display the number of files collected.
Label batches (e.g., "normal", "idle", "imbalance") with dropdown options.
Generates files in subdirectories (e.g., ./normal/normal_1.csv) with 800 rows each.
Execution:
bash
python data_collection.py

Adjust PORT (e.g., "COM4") and LABEL in the script as needed.
Collect multiple files per label for robust training.
Output Structure
Dataset/
├── normal/
│   ├── normal_1.csv
│   ├── normal_2.csv
│   └── ...
├── idle/
│   ├── idle_1.csv
│   ├── idle_2.csv
│   └── ...
├── imbalance/
│   ├── imbalance_1.csv
│   ├── imbalance_2.csv
│   └── ...

🧠 Model Training with Edge Impulse
Step 3: Upload and Train
Process:
Upload CSV files to Edge Impulse via the CSV Wizard.
Preprocessing: Apply spectral filters (e.g., FFT) to extract frequency-domain features.
Model:
Classification Block: Custom neural network.
Anomaly Detection: K-Means clustering for unseen faults.
Architecture:
Reshape Layer: Input (800, 6) reshaped to (800, 1, 6) for 1D convolution (timesteps=800, channels=6 axes).
1D Conv Layer (1): 8 filters, kernel size 3, ReLU activation.
Dropout: 0.5 (prevents overfitting).
1D Conv Layer (2): 16 filters, kernel size 3, ReLU activation.
Dropout: 0.5.
Flatten: Prepares data for dense layer.
Dense Classifier: 3 outputs (normal, idle, imbalance) with Softmax.
Training: Use default hyperparameters, adjust based on validation accuracy.
Download: Export the trained model as an Arduino library (.h file) for Nano BLE 33.
Note: Architecture refined from insights in Single_Fault_Vibration_Analysis_Using_MachineLearning.

🛠️ Deployment
Step 4: Integrate Model
Include the downloaded Edge Impulse library and .h model file in your Arduino project.
Step 5: Run Final Sketch
File: FINAL_CODE.ino
Details:
Loads the trained model.
Runs inference on live vibration data.
Transmits predictions via BLE.
Execution: Upload to Arduino Nano BLE 33 and power on.
Step 6: Monitor via GUI
Open the Tkinter GUI by running data_collection.py (or a modified monitor_gui.py if separate).
Connect to the Arduino via BLE:
Pair the device in the GUI.
View real-time status (e.g., "Normal", "Imbalance", or "Anomaly").

📝 Documentation
Architecture Rationale
Reshape: Converts 800x6 raw data into a 3D tensor for 1D Conv, preserving temporal structure.
1D Conv: Extracts spatial features from vibration sequences, with 8→16 filters to deepen feature learning.
Dropout (0.5): Mitigates overfitting on limited datasets.
Flatten + Dense: Maps features to 3 classes, optimized for Nano BLE 33’s constraints.
Performance
Expected accuracy: ~90%+ on labeled data (varies with dataset quality).
Anomaly detection sensitivity depends on K-Means cluster tuning.
Limitations
BLE range (~10m) and 400 Hz sampling may miss ultra-high-frequency faults.
GUI assumes stable Arduino connection.

🌟 Star This Repo
If you find this project useful, please star it on GitHub!
