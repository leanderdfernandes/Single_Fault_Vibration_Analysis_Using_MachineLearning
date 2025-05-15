import serial
import csv
import time, re
from datetime import datetime, timedelta
import os

# Configure Serial port
# ser = serial.Serial('COM4', 1000000, timeout=1)
time.sleep(2)

# Parameters
WINDOW_SIZE = 800
max_examples = 10  # Number of new files to collect
condition = "Idle"  # Toggle as needed
SAMPLE_INTERVAL = 0.0025

# Find the next available index
directory = f"./Dataset/{condition}/"
os.makedirs(directory, exist_ok=True)
existing_files = [f for f in os.listdir(directory) if f.startswith(f"{condition}_") and f.endswith(".csv")]
if existing_files:
    indices = [int(f.split('_')[1].split('.')[0]) for f in existing_files]
    start_count = max(indices) + 1  # Start at next index
else:
    start_count = 0  # Start at 0 if no files exist

example_count = start_count  # Current file index
files_collected = 0  # Number of new files collected

while files_collected < max_examples:  # Collect exactly 'max_examples' new files
    start_time = time.time()
    window_data = []
    lines_read = 0

    # Wait for start of window
    while True:
        line = ser.readline().decode('utf-8').strip()
        if not line or "rel_timestamp" in line or "time" in line or "END_WINDOW" in line:
            continue
        try:
            rel_timestamp, ax, ay, az, gx, gy, gz = line.split(',')
            window_data.append([int(rel_timestamp), float(ax), float(ay), float(az), 
                              float(gx), float(gy), float(gz)])
            lines_read = 1
            break
        except ValueError:
            continue

    # Collect exactly 800 lines
    while lines_read < WINDOW_SIZE:
        line = ser.readline().decode('utf-8').strip()
        if not line or "time" in line:
            continue
        if "END_WINDOW" in line:
            break
        try:
            rel_timestamp, ax, ay, az, gx, gy, gz = line.split(',')
            window_data.append([int(rel_timestamp), float(ax), float(ay), float(az), 
                              float(gx), float(gy), float(gz)])
            lines_read += 1
        except ValueError:
            continue
    
    # Convert timestamps
    base_time = datetime.utcnow()
    csv_rows = []
    for rel_ts, ax, ay, az, gx, gy, gz in window_data[:WINDOW_SIZE]:
        timestamp = base_time + timedelta(microseconds=rel_ts)
        iso_timestamp = timestamp.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        csv_rows.append([iso_timestamp, ax, ay, az, gx, gy, gz])
    
    # Save to CSV
    filename = f"./Dataset/{condition}/{condition}_{example_count:03d}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"])
        writer.writerows(csv_rows)
    
    end_time = time.time()
    print(f"Saved {filename} in {end_time - start_time:.2f} seconds, {len(csv_rows)} rows")
    example_count += 1
    files_collected += 1

ser.close()