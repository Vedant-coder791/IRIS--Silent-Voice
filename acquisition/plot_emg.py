import pandas as pd
import matplotlib.pyplot as plt

# CSV file
filename = "emg_20260920_154208.csv"

# Load data
data = pd.read_csv(filename)

# -----------------------------
# Basic data information
# -----------------------------

total_samples = len(data)

first_sample = data["sample"].iloc[0]
last_sample = data["sample"].iloc[-1]

sample_difference = last_sample - first_sample

sampling_rate = sample_difference / (total_samples - 1)

estimated_duration = sample_difference / sampling_rate

total_samples = len(data)

first_timestamp = data["sample"].iloc[0]
last_timestamp = data["sample"].iloc[-1]

elapsed_microseconds = last_timestamp - first_timestamp
elapsed_seconds = elapsed_microseconds / 1_000_000

sampling_rate = (total_samples - 1) / elapsed_seconds

print("\n===== IRIS DATA QUALITY CHECK =====")

print(f"Total samples : {total_samples}")
print(f"Duration      : {elapsed_seconds:.3f} seconds")
print(f"Sampling rate : {sampling_rate:.2f} Hz")

print("===================================\n")


# -----------------------------
# Plot the four channels
# -----------------------------

plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(data["CH1"])
plt.ylabel("CH1")
plt.grid()

plt.subplot(4, 1, 2)
plt.plot(data["CH2"])
plt.ylabel("CH2")
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(data["CH3"])
plt.ylabel("CH3")
plt.grid()

plt.subplot(4, 1, 4)
plt.plot(data["CH4"])
plt.ylabel("CH4")
plt.xlabel("Sample")
plt.grid()

plt.suptitle("IRIS — 4-Channel EMG Data")

plt.tight_layout()
plt.show()