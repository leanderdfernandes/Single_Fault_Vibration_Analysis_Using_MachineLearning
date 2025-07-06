import tkinter as tk
from tkinter import ttk, messagebox
import threading
import serial
import csv
import time
import os

# Configuration
SERIAL_PORT = 'COM4'
BAUD_RATE = 1000000
WINDOW_SIZE = 800
SAMPLE_INTERVAL = 0.0025
MAX_EXAMPLES = 20  # ~10 min

# Fault conditions
conditions = ["Normal Operation", "Idle State", "Imbalance", "Misalignment", "Bearing Fault", "Mechanical Looseness"]

class DataCollectorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Sensor Data Collector")
        master.geometry("400x400")
        self.condition = tk.StringVar(value="Idle State")
        tk.Label(master, text="Select Fault Condition:", font=("Arial", 12)).pack(pady=(10, 5))
        for cond in conditions:
            tk.Radiobutton(master, text=cond, variable=self.condition, value=cond, font=("Arial", 10)).pack(anchor="w", padx=20)
        self.collect_button = tk.Button(master, text="Start Collection", font=("Arial", 11), command=self.start_collection)
        self.collect_button.pack(pady=15)
        self.progress = ttk.Progressbar(master, mode='determinate', length=300, maximum=MAX_EXAMPLES)
        self.progress.pack(pady=10)
        self.progress_label = tk.Label(master, text=f"Progress: 0 / {MAX_EXAMPLES}", font=("Arial", 10))
        self.progress_label.pack()

    def start_collection(self):
        self.collect_button.config(state="disabled")
        self.progress['value'] = 0
        self.progress_label.config(text=f"Progress: 0 / {MAX_EXAMPLES}")
        threading.Thread(target=self.collect_data_thread).start()

    def collect_data_thread(self):
        try:
            self.collect_data(self.condition.get())
            messagebox.showinfo("Done", f"Data collection for '{self.condition.get()}' completed.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.collect_button.config(state="normal")

    def update_progress(self, value):
        self.progress['value'] = value
        self.progress_label.config(text=f"Progress: {value} / {MAX_EXAMPLES}")

    def collect_data(self, condition):
        retries = 3
        for attempt in range(retries):
            try:
                ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
                time.sleep(3)  # Wait for Arduino to stabilize
                break
            except serial.SerialException as e:
                if attempt == retries - 1:
                    raise e
                time.sleep(1)

        directory = f"./Dataset/{condition}/"
        os.makedirs(directory, exist_ok=True)
        existing_files = [f for f in os.listdir(directory) if f.startswith(f"{condition}_") and f.endswith(".csv")]
        start_count = max([int(f.split('_')[1].split('.')[0]) for f in existing_files], default=-1) + 1

        files_collected = 0
        example_count = start_count
        error_log = open(f"{directory}/error_log.txt", 'a')

        while files_collected < MAX_EXAMPLES:
            window_data = []
            lines_read = 0

            while True:
                line = ser.readline().decode('utf-8').strip()
                if not line or "rel_timestamp" in line or "time" in line or "END_WINDOW" in line:
                    continue
                try:
                    rel_timestamp, ax, ay, az, gx, gy, gz = line.split(',')
                    window_data.append([float(rel_timestamp) / 1000, float(ax), float(ay), float(az),
                                       float(gx), float(gy), float(gz)])
                    lines_read = 1
                    break
                except ValueError:
                    error_log.write(f"Malformed line: {line}\n")
                    continue

            while lines_read < WINDOW_SIZE:
                line = ser.readline().decode('utf-8').strip()
                if not line or "time" in line:
                    continue
                if "END_WINDOW" in line:
                    break
                try:
                    rel_timestamp, ax, ay, az, gx, gy, gz = line.split(',')
                    window_data.append([float(rel_timestamp) / 1000, float(ax), float(ay), float(az),
                                       float(gx), float(gy), float(gz)])
                    lines_read += 1
                except ValueError:
                    error_log.write(f"Malformed line: {line}\n")

            if len(window_data) == WINDOW_SIZE:
                filename = f"{directory}/{condition}_{example_count:03d}.csv"
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"])
                    writer.writerows(window_data)
                files_collected += 1
                example_count += 1
                self.master.after(0, self.update_progress, files_collected)
                print(f"Saved {filename}")
            else:
                error_log.write(f"Incomplete window: {len(window_data)} samples\n")

        error_log.close()
        ser.close()

root = tk.Tk()
app = DataCollectorGUI(root)
root.mainloop()