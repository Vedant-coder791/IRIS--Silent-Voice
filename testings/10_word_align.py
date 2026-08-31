import numpy as np


# ==========================================
# WORD ALIGNMENT FILE
# ==========================================

word_file = "/Users/vedantdwivedi/Desktop/IRIS/EMG-UKA-Trial-Corpus/Alignments/002/001/words_002_001_0100.txt"

words = []

with open(word_file, "r") as f:

    for line in f:

        parts = line.strip().split()

        if len(parts) == 3:

            start = int(parts[0])
            end = int(parts[1])
            word = parts[2]

            words.append((start, end, word))


print("Number of words:", len(words))

print("\nWords:")

for word in words:
    print(word)


# ==========================================
# EMG PARAMETERS
# ==========================================

FS = 600

WINDOW_SIZE = 120
STEP_SIZE = 60

# Alignment frames are 10 ms each
FRAME_DURATION = 0.010


# ==========================================
# FIND WORD FOR EACH EMG WINDOW
# ==========================================

def get_window_word(window_start, window_end):

    window_start_time = window_start / FS
    window_end_time = window_end / FS

    window_duration = window_end_time - window_start_time

    best_word = None
    best_overlap = 0

    for word_start, word_end, word in words:

        # Ignore beginning/end markers
        if word == "$":
            continue

        word_start_time = word_start * FRAME_DURATION
        word_end_time = (word_end + 1) * FRAME_DURATION

        # Find overlap
        overlap_start = max(
            window_start_time,
            word_start_time
        )

        overlap_end = min(
            window_end_time,
            word_end_time
        )

        overlap = max(
            0,
            overlap_end - overlap_start
        )

        if overlap > best_overlap:

            best_overlap = overlap
            best_word = word

    # Calculate how much of the window
    # belongs to the best word
    purity = best_overlap / window_duration

    # Only accept sufficiently clean windows
    if purity >= 0.70:
        return best_word

    return None

    window_start_time = window_start / FS
    window_end_time = window_end / FS

    best_word = None
    best_overlap = 0

    for word_start, word_end, word in words:

        # Ignore beginning/end markers
        if word == "$":
            continue

        word_start_time = word_start * FRAME_DURATION
        word_end_time = (word_end + 1) * FRAME_DURATION

        overlap_start = max(
            window_start_time,
            word_start_time
        )

        overlap_end = min(
            window_end_time,
            word_end_time
        )

        overlap = max(
            0,
            overlap_end - overlap_start
        )

        if overlap > best_overlap:

            best_overlap = overlap
            best_word = word

    return best_word


# ==========================================
# TEST WITH 59 WINDOWS
# ==========================================

num_windows = 59

labels = []

for i in range(num_windows):

    start = i * STEP_SIZE
    end = start + WINDOW_SIZE

    label = get_window_word(start, end)

    labels.append(label)


# ==========================================
# PRINT RESULTS
# ==========================================

print("\nNumber of EMG windows:", len(labels))

print("\nWindow labels:")

for i, label in enumerate(labels):

    print(f"Window {i}: {label}")