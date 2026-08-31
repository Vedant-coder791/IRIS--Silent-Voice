import numpy as np
import matplotlib.pyplot as plt

filename = "/Users/vedantdwivedi/Desktop/IRIS/EMG-UKA-Trial-Corpus/emg/002/001/e07_002_001_0100.adc"

CHANNELS = 7
FS = 600      # Sampling frequency

raw = np.fromfile(filename, dtype=np.int16)

data = raw.reshape(-1, CHANNELS).T

# Remove marker channel
emg = data[:6]

signal = emg[0].astype(np.float64)

# Remove DC offset
signal = signal - np.mean(signal)

fft = np.fft.rfft(signal)

freq = np.fft.rfftfreq(
    len(signal),
    d=1/FS
)

magnitude = np.abs(fft)


plt.figure(figsize=(12,5))

plt.plot(freq, magnitude)

plt.title("Frequency Spectrum of EMG")

plt.xlabel("Frequency (Hz)")

plt.ylabel("Magnitude")

plt.grid(True)

plt.xlim(0,100)

plt.show()