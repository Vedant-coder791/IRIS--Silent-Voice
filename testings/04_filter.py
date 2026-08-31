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



import numpy as np

from Windowing.windowing import create_windows

windows = create_windows(filtered_emg)

print("Windows shape:", windows.shape)


# ==========================================
# READ PHONEME ALIGNMENT FILE
# ==========================================

phone_file = "/Users/vedantdwivedi/Desktop/IRIS/EMG-UKA-Trial-Corpus/Alignments/002/001/phones_002_001_0100.txt"

phones = []

with open(phone_file, "r") as f:
    for line in f:
        parts = line.strip().split()

        if len(parts) == 3:
            start = int(parts[0])
            end = int(parts[1])
            phone = parts[2]

            phones.append((start, end, phone))


print("Number of phonemes:", len(phones))


print("\nFirst 10 phonemes:")

for phone in phones[:10]:
    print(phone)


print("\nLast 5 phonemes:")

for phone in phones[-5:]:
    print(phone)


# ==========================================
# ALIGN PHONEMES WITH EMG WINDOWS
# ==========================================

FS = 600              # EMG sampling rate
WINDOW_SIZE = 120     # 200 ms = 120 samples at 600 Hz
STEP_SIZE = 60        # 50% overlap

# Each phoneme frame represents 10 ms
FRAME_DURATION = 0.010


print("\nAligning phonemes with EMG windows...\n")


def get_window_label(window_start, window_end):
    """
    Find the phoneme that occupies the largest
    portion of an EMG window.
    """

    # Convert EMG samples to seconds
    window_start_time = window_start / FS
    window_end_time = window_end / FS

    best_phone = None
    best_overlap = 0

    for phone_start, phone_end, phone in phones:

        # Convert phoneme frames to seconds
        phone_start_time = phone_start * FRAME_DURATION
        phone_end_time = (phone_end + 1) * FRAME_DURATION

        # Find overlap between the EMG window
        # and the phoneme
        overlap_start = max(
            window_start_time,
            phone_start_time
        )

        overlap_end = min(
            window_end_time,
            phone_end_time
        )

        overlap = max(
            0,
            overlap_end - overlap_start
        )

        # Keep the phoneme with the largest overlap
        if overlap > best_overlap:
            best_overlap = overlap
            best_phone = phone

    return best_phone


# ==========================================
# ALIGN EMG WINDOWS WITH PHONEMES
# ==========================================

aligned_windows = []
labels = []

# 'windows' should already contain your
# windowed EMG data with shape (59, 6, 120)

num_windows = len(windows)


for i in range(num_windows):

    start = i * STEP_SIZE
    end = start + WINDOW_SIZE

    label = get_window_label(start, end)

    # Only keep windows that have a valid label
    if label is not None:

        aligned_windows.append(windows[i])
        labels.append(label)


# ==========================================
# CONVERT TO NUMPY ARRAYS
# ==========================================

aligned_windows = np.array(aligned_windows)
labels = np.array(labels)


# ==========================================
# PRINT RESULTS
# ==========================================

print("\nOriginal number of EMG windows:", len(windows))

print(
    "Valid aligned windows:",
    len(aligned_windows)
)

print("\nAligned EMG shape:")
print(aligned_windows.shape)

print("\nLabels shape:")
print(labels.shape)


print("\nWindow labels:")

for i, label in enumerate(labels):

    print(
        f"Window {i}: {label}"
    )
