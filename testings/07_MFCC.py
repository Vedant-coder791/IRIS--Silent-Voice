
import librosa

#TEMPORARY 
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt

filename = "/Users/vedantdwivedi/Desktop/IRIS/EMG-UKA-Trial-Corpus/emg/002/001/e07_002_001_0100.adc"

CHANNELS = 7
FS = 600

raw = np.fromfile(filename, dtype=np.int16)

data = raw.reshape(-1, CHANNELS).T

# Keep only EMG channels
emg = data[:6].astype(np.float64)

def highpass_filter(signal, cutoff=20, fs=600, order=4):

    b, a = butter(order, cutoff, btype='highpass', fs=fs)

    return filtfilt(b, a, signal)

filtered = highpass_filter(emg[0])

filtered_emg = np.zeros_like(emg)

for ch in range(emg.shape[0]):
    filtered_emg[ch] = highpass_filter(emg[ch])


time = np.arange(emg.shape[1]) / FS

plt.figure(figsize=(14,6))

plt.plot(time,
         emg[0],
         label="Raw",
         alpha=0.7)

plt.plot(time,
         filtered_emg[0],
         label="Filtered",
         linewidth=2)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.title("Channel 1 Before and After High-Pass Filtering")

plt.legend()

plt.grid(True)

plt.show()



plt.figure(figsize=(14,10))

for ch in range(filtered_emg.shape[0]):

    plt.subplot(6,1,ch+1)

    plt.plot(time, filtered_emg[ch])

    plt.ylabel(f"Ch {ch+1}")

plt.xlabel("Time (s)")
plt.suptitle("High-Pass Filtered EMG")

plt.tight_layout()

plt.show()


# FFT before filtering
fft_raw = np.abs(np.fft.rfft(emg[0] - np.mean(emg[0])))

# FFT after filtering
fft_filtered = np.abs(np.fft.rfft(filtered_emg[0]))

freq = np.fft.rfftfreq(len(emg[0]), d=1/FS)

plt.figure(figsize=(12,5))

plt.plot(freq, fft_raw, label="Raw", alpha=0.7)
plt.plot(freq, fft_filtered, label="Filtered", linewidth=2)

plt.xlim(0,100)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT Before vs After High-Pass Filter")

plt.grid(True)
plt.legend()

plt.show()

print(filtered_emg.shape)

print(np.mean(filtered_emg[0]))

print(np.std(filtered_emg[0]))

from scipy.signal import iirnotch, filtfilt

def notch_filter(signal, notch_freq=50, fs=600, quality_factor=30):
   
    b, a = iirnotch(notch_freq, quality_factor, fs)

    if signal.ndim == 1:
        return filtfilt(b, a, signal)

    elif signal.ndim == 2:
        return filtfilt(b, a, signal, axis=1)

    else:
        raise ValueError("Signal must be 1D or 2D")


filtered_emg = highpass_filter(emg)

filtered_emg = notch_filter(filtered_emg)

time = np.arange(emg.shape[1]) / FS

plt.figure(figsize=(14,6))

plt.plot(time, emg[0], label="Raw", alpha=0.6)

plt.plot(time, filtered_emg[0], label="High-pass + Notch", linewidth=2)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Channel 1 After EMG Preprocessing")

plt.legend()
plt.grid(True)

plt.show()

fft_before = np.abs(np.fft.rfft(emg[0] - np.mean(emg[0])))
fft_after = np.abs(np.fft.rfft(filtered_emg[0]))

freq = np.fft.rfftfreq(len(emg[0]), d=1/FS)

plt.figure(figsize=(12,5))

plt.plot(freq, fft_before, label="Raw")
plt.plot(freq, fft_after, label="Filtered")

plt.xlim(0,100)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT Before and After High-pass + Notch")

plt.grid(True)
plt.legend()

plt.show()



from Windowing.windowing import create_windows
windows = create_windows(filtered_emg)

print(windows.shape)

window = windows[0, 0]
print(window.shape)




mfcc = librosa.feature.mfcc(
    y=window.astype(np.float32),
    sr=600,
    n_fft=64,
    hop_length=32,
    n_mfcc=13,
    n_mels=20,
    fmin=20,
    fmax=250
)


print(mfcc.shape)

import matplotlib.pyplot as plt

plt.imshow(
    mfcc,
    aspect="auto",
    origin="lower"
)

plt.xlabel("Frame")
plt.ylabel("MFCC")
plt.title("MFCC Features")
plt.colorbar()
plt.show()

all_features = []

for window in windows:

    channel_features = []

    for ch in window:

        mfcc = librosa.feature.mfcc(
            y=ch.astype(np.float32),
            sr=600,
            n_fft=64,
            hop_length=32,
            n_mfcc=13,
            n_mels=20,
            fmin=20,
            fmax=250
        )

        channel_features.append(mfcc)

    all_features.append(channel_features)

all_features = np.array(all_features)

print(all_features.shape)