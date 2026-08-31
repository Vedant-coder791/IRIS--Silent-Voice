import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math

fs = 1000     #sampling frequency is 1000, Target signal frequency is 80, noise frequency is 60hz (elecrical main)
t = np.linspace(0, 2, fs*2)

signal = (
    np.sin(2*np.pi*80*t) +
    0.4*np.sin(2*np.pi*50*t) +
    0.2*np.random.randn(len(t))
)

plt.plot(t, signal)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter
from scipy.signal import filtfilt

def highpass_filter(signal, cutoff=20, fs=1000, order=4):
  nyquist= 0.5*fs
  normal_cutoff = cutoff/nyquist

  b, a = butter(order, normal_cutoff, btype='high', analog=False)

  filtered_signal = filtfilt(b, a, signal)

  return filtered_signal

filtered = highpass_filter(signal)

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

plt.plot(t, signal, label="Original Signal")
plt.plot(t, filtered, label="High-pass Filtered", linewidth=2)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Butterworth High-pass Filter")
plt.legend()
plt.grid(True)

plt.show()

from scipy.signal import iirnotch, filtfilt

def notch_filter(signal, notch_freq=50, fs=1000, quality_factor=30):

    b, a = iirnotch(notch_freq, quality_factor, fs)

    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal

notched_signal = notch_filter(filtered)

plt.figure(figsize=(12,6))

plt.plot(t, signal, label="Original", alpha=0.6)

plt.plot(t, filtered, label="After High-pass")

plt.plot(t, notched_signal, label="After Notch", linewidth=2)

plt.legend()

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.grid(True)

plt.show()


window_size = 200
hop_size = 100

windows = []

for i in range(0, len(notched_signal) - window_size + 1, hop_size):
    window = notched_signal[i:i + window_size]
    windows.append(window)

import librosa
import numpy as np


#------------------------------------------------------------need fixing

mfcc = librosa.feature.mfcc(
    y=window.astype(np.float32),
    sr=1000,
    n_fft=128,
    hop_length=64,
    n_mfcc=13,
    n_mels=20,
    fmin=20,
    fmax=450
)

feature_vector = np.mean(mfcc, axis= 1)

print(feature_vector.shape)
print(len(windows))

X = np.array(mfcc)

print(X.shape)