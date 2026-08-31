import numpy as np
import matplotlib.pyplot as plt

filename = "/Users/vedantdwivedi/Desktop/IRIS/EMG-UKA-Trial-Corpus/emg/002/001/e07_002_001_0100.adc"
CHANNELS = 7

# Read the ADC file
raw = np.fromfile(filename, dtype=np.int16)
data = raw.reshape(-1, CHANNELS).T

# Time axis
fs = 600  # Sampling frequency
time = np.arange(data.shape[1]) / fs

# Plot all channels
plt.figure(figsize=(14, 10))

for i in range(CHANNELS):
    plt.subplot(CHANNELS, 1, i + 1)
    plt.plot(time, data[i], linewidth=0.8)
    plt.ylabel(f"Ch {i+1}")

plt.xlabel("Time (s)")
plt.suptitle("Raw EMG Signals")
plt.tight_layout()
plt.show()