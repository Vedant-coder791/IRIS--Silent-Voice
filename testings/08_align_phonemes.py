import numpy as np




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
