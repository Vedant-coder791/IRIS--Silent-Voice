
"""
======================================================================
IRIS - BERKELEY WORD-LEVEL TIME-FREQUENCY DATASET
======================================================================

Purpose:
    Create a CNN-ready dataset from the Berkeley Silent Speech EMG data.

IMPORTANT:
    We DO NOT label every window in a recording with every word.

    Instead:
        EMG recording
              ↓
        word alignment
              ↓
        word-level EMG segment
              ↓
        200-sample windows
              ↓
        STFT / log-power spectrogram
              ↓
        CNN dataset

Expected CNN input:
    X -> (N, 8, 24, 9)
    y -> (N,)

======================================================================
"""

import os
import json
import glob
import numpy as np
from scipy.signal import stft
from collections import Counter


# =====================================================================
# PATHS
# =====================================================================

EMG_ROOT = (
    "/Users/vedantdwivedi/Desktop/IRIS/"
    "emg_data/closed_vocab/silent"
)

ALIGNMENT_CSV = (
    "/Users/vedantdwivedi/Desktop/IRIS/emg_data/"
    "closed_vocab/silent/closed_vocab_silent_alignment_diagnostic.csv"
)

OUTPUT_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/Data/"
    "BERKELEY_WORD_TIME_FREQUENCY_DATA.npz"
)


# =====================================================================
# SIGNAL PARAMETERS
# =====================================================================

FS = 600

WINDOW_SIZE = 200
STEP_SIZE = 100

N_FFT = 64
STFT_HOP = 16

EXPECTED_CHANNELS = 8

FMIN = 20
FMAX = 250


# =====================================================================
# TARGET VOCABULARY
# =====================================================================
#
# Start with reasonably frequent words.
#
# You can expand this later.
#
# =====================================================================

TARGET_WORDS = {
    "AM",
    "PM",

    "JANUARY",
    "FEBRUARY",
    "MARCH",
    "APRIL",
    "MAY",
    "JUNE",
    "JULY",
    "AUGUST",
    "SEPTEMBER",
    "OCTOBER",
    "NOVEMBER",
    "DECEMBER",

    "MONDAY",
    "TUESDAY",
    "WEDNESDAY",
    "THURSDAY",
    "FRIDAY",
    "SATURDAY",
    "SUNDAY",
}


# =====================================================================
# CHECK PATHS
# =====================================================================

print("=" * 70)
print("IRIS - BERKELEY WORD-LEVEL TIME-FREQUENCY EXTRACTION")
print("=" * 70)

print()
print("EMG root:")
print(EMG_ROOT)

print()
print("Alignment CSV:")
print(ALIGNMENT_CSV)

print()

if not os.path.isdir(EMG_ROOT):
    raise FileNotFoundError(
        "EMG directory does not exist."
    )

if not os.path.isfile(ALIGNMENT_CSV):
    raise FileNotFoundError(
        "Alignment CSV does not exist."
    )


# =====================================================================
# READ CSV WITHOUT PANDAS
# =====================================================================
#
# This is intentional because your Python environment currently
# does not have pandas installed.
#
# =====================================================================

import csv

print("=" * 70)
print("READING ALIGNMENT CSV")
print("=" * 70)

with open(
    ALIGNMENT_CSV,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)

    rows = list(reader)

    columns = reader.fieldnames


print()
print("CSV columns:")
print(columns)

print()
print("Rows:", len(rows))


# =====================================================================
# LOOK FOR POSSIBLE ALIGNMENT COLUMNS
# =====================================================================

print()
print("=" * 70)
print("SEARCHING FOR ALIGNMENT COLUMNS")
print("=" * 70)

if columns is None:
    raise RuntimeError("CSV has no header.")

possible_word_columns = [
    c for c in columns
    if any(
        x in c.lower()
        for x in [
            "word",
            "label",
            "text",
            "token"
        ]
    )
]

possible_start_columns = [
    c for c in columns
    if any(
        x in c.lower()
        for x in [
            "start",
            "begin",
            "onset"
        ]
    )
]

possible_end_columns = [
    c for c in columns
    if any(
        x in c.lower()
        for x in [
            "end",
            "stop",
            "offset"
        ]
    )
]

print()
print("Possible word columns:")
for c in possible_word_columns:
    print("  ", c)

print()
print("Possible start columns:")
for c in possible_start_columns:
    print("  ", c)

print()
print("Possible end columns:")
for c in possible_end_columns:
    print("  ", c)


# =====================================================================
# IMPORTANT SAFETY CHECK
# =====================================================================
#
# Your previous diagnostic CSV may NOT contain actual word timings.
#
# If it does not, STOP rather than generating fake labels.
#
# =====================================================================

if not possible_word_columns:
    raise RuntimeError(
        "\n"
        "NO WORD LABEL COLUMN FOUND.\n\n"
        "The diagnostic CSV contains recording-level information, "
        "but does not appear to contain word-level alignment.\n\n"
        "Do NOT train the CNN yet.\n"
        "We need the actual word start/end alignment first."
    )

if not possible_start_columns or not possible_end_columns:
    raise RuntimeError(
        "\n"
        "WORD LABELS WERE FOUND, BUT WORD START/END TIMINGS "
        "WERE NOT FOUND.\n\n"
        "This means we know which words occur in a recording, "
        "but not where those words occur in the EMG signal.\n\n"
        "Do NOT assign the word label to every window."
    )


# =====================================================================
# SELECT COLUMNS
# =====================================================================

WORD_COL = possible_word_columns[0]
START_COL = possible_start_columns[0]
END_COL = possible_end_columns[0]

print()
print("=" * 70)
print("USING ALIGNMENT COLUMNS")
print("=" * 70)

print("Word :", WORD_COL)
print("Start:", START_COL)
print("End  :", END_COL)


# =====================================================================
# EMG FILE INDEX
# =====================================================================

print()
print("=" * 70)
print("SEARCHING FOR EMG FILES")
print("=" * 70)

emg_files = glob.glob(
    os.path.join(
        EMG_ROOT,
        "**",
        "*_emg.npy"
    ),
    recursive=True
)

print()
print("EMG files:", len(emg_files))


emg_lookup = {}

for path in emg_files:

    filename = os.path.basename(path)

    # Example:
    # 101_emg.npy -> 101

    stem = filename.replace(
        "_emg.npy",
        ""
    )

    emg_lookup[stem] = path


# =====================================================================
# FEATURE EXTRACTION
# =====================================================================

def extract_time_frequency(window):
    """
    Input:
        window -> (8, 200)

    Output:
        feature -> (8, 24, 9)
    """

    features = []

    for ch in range(EXPECTED_CHANNELS):

        signal = window[ch]

        frequencies, times, Zxx = stft(
            signal,
            fs=FS,
            nperseg=N_FFT,
            noverlap=N_FFT - STFT_HOP,
            nfft=N_FFT,
            boundary=None
        )

        power = np.abs(Zxx) ** 2

        power = np.log1p(power)

        # Keep only 20-250 Hz
        mask = (
            (frequencies >= FMIN)
            &
            (frequencies <= FMAX)
        )

        power = power[mask]

        # Expected roughly:
        # 24 frequency bins
        #
        # Force fixed dimensions if scipy gives slightly
        # different dimensions.

        if power.shape[0] < 24:

            pad_rows = 24 - power.shape[0]

            power = np.pad(
                power,
                (
                    (0, pad_rows),
                    (0, 0)
                ),
                mode="constant"
            )

        elif power.shape[0] > 24:

            power = power[:24]

        if power.shape[1] < 9:

            pad_cols = 9 - power.shape[1]

            power = np.pad(
                power,
                (
                    (0, 0),
                    (0, pad_cols)
                ),
                mode="constant"
            )

        elif power.shape[1] > 9:

            power = power[:, :9]

        features.append(power)

    return np.asarray(
        features,
        dtype=np.float32
    )


# =====================================================================
# STORAGE
# =====================================================================

X_list = []
y_list = []

metadata = []

word_counter = Counter()

processed_words = 0
processed_windows = 0
failed_segments = 0


# =====================================================================
# PROCESS ALIGNMENT ROWS
# =====================================================================

print()
print("=" * 70)
print("STARTING WORD EXTRACTION")
print("=" * 70)


for index, row in enumerate(rows):

    try:

        # -------------------------------------------------------------
        # FIND RECORDING
        # -------------------------------------------------------------

        filename = (
            row.get("file")
            or row.get("emg_file")
            or row.get("filename")
            or ""
        )

        recording_id = (
            os.path.basename(filename)
            .replace("_emg.npy", "")
            .replace(".npy", "")
        )

        if recording_id not in emg_lookup:
            continue

        emg_path = emg_lookup[recording_id]

        emg = np.load(
            emg_path
        )

        if emg.ndim != 2:
            continue

        if emg.shape[1] != EXPECTED_CHANNELS:
            continue

        # -------------------------------------------------------------
        # WORD
        # -------------------------------------------------------------

        word = str(
            row.get(WORD_COL, "")
        ).strip().upper()

        if word not in TARGET_WORDS:
            continue

        # -------------------------------------------------------------
        # START / END
        # -------------------------------------------------------------

        start_raw = str(
            row.get(START_COL, "")
        ).strip()

        end_raw = str(
            row.get(END_COL, "")
        ).strip()

        if not start_raw or not end_raw:
            continue

        start = int(
            float(start_raw)
        )

        end = int(
            float(end_raw)
        )

        # -------------------------------------------------------------
        # VALIDATE
        # -------------------------------------------------------------

        if start < 0:
            start = 0

        if end > len(emg):
            end = len(emg)

        if end <= start:
            continue

        segment = emg[start:end]

        # -------------------------------------------------------------
        # NEED ENOUGH SAMPLES
        # -------------------------------------------------------------

        if len(segment) < WINDOW_SIZE:
            continue

        # -------------------------------------------------------------
        # WINDOW
        # -------------------------------------------------------------

        local_windows = 0

        for w_start in range(
            0,
            len(segment) - WINDOW_SIZE + 1,
            STEP_SIZE
        ):

            w_end = (
                w_start
                + WINDOW_SIZE
            )

            window = segment[
                w_start:w_end
            ]

            if window.shape != (
                WINDOW_SIZE,
                EXPECTED_CHANNELS
            ):
                continue

            # ---------------------------------------------------------
            # TRANSPOSE
            #
            # (200, 8)
            #       ↓
            # (8, 200)
            # ---------------------------------------------------------

            window = window.T

            # ---------------------------------------------------------
            # TIME-FREQUENCY
            # ---------------------------------------------------------

            feature = extract_time_frequency(
                window
            )

            if feature.shape != (
                8,
                24,
                9
            ):
                continue

            X_list.append(
                feature
            )

            y_list.append(
                word
            )

            metadata.append({
                "recording": recording_id,
                "word": word,
                "start": start + w_start,
                "end": start + w_end
            })

            local_windows += 1

        if local_windows > 0:

            word_counter[word] += local_windows

            processed_words += 1

            processed_windows += local_windows

    except Exception as e:

        failed_segments += 1

        print(
            "WARNING:",
            index,
            str(e)
        )

    if (index + 1) % 50 == 0:

        print(
            f"Processed {index + 1}/{len(rows)} "
            f"| Windows: {processed_windows}"
        )


# =====================================================================
# CHECK RESULT
# =====================================================================

print()
print("=" * 70)
print("EXTRACTION COMPLETE")
print("=" * 70)

print()
print("Word segments processed:", processed_words)
print("Failed segments:", failed_segments)
print("Total windows:", processed_windows)


if len(X_list) == 0:

    raise RuntimeError(
        "\n"
        "ZERO CNN SAMPLES WERE CREATED.\n\n"
        "Most likely the CSV does not contain true word-level "
        "sample timings."
    )


# =====================================================================
# CONVERT TO ARRAYS
# =====================================================================

X = np.asarray(
    X_list,
    dtype=np.float32
)

y = np.asarray(
    y_list
)


# =====================================================================
# LABEL ENCODING
# =====================================================================

classes = sorted(
    np.unique(y)
)

class_to_index = {
    word: i
    for i, word in enumerate(classes)
}

y_encoded = np.asarray(
    [
        class_to_index[word]
        for word in y
    ],
    dtype=np.int64
)


# =====================================================================
# NORMALIZATION
# =====================================================================
#
# Global standardization.
#
# IMPORTANT:
# For final experiments, calculate normalization statistics using
# TRAINING data only.
#
# This step gives us a clean initial dataset.
#
# =====================================================================

mean = X.mean()

std = X.std()

if std < 1e-8:
    std = 1.0

X = (
    X - mean
) / std


# =====================================================================
# METADATA ARRAYS
# =====================================================================

recordings = np.asarray(
    [
        m["recording"]
        for m in metadata
    ]
)

words = np.asarray(
    [
        m["word"]
        for m in metadata
    ]
)

starts = np.asarray(
    [
        m["start"]
        for m in metadata
    ],
    dtype=np.int64
)

ends = np.asarray(
    [
        m["end"]
        for m in metadata
    ],
    dtype=np.int64
)


# =====================================================================
# FINAL SUMMARY
# =====================================================================

print()
print("=" * 70)
print("FINAL DATASET")
print("=" * 70)

print()
print("X shape:")
print(X.shape)

print()
print("y shape:")
print(y_encoded.shape)

print()
print("Expected X:")
print("(N, 8, 24, 9)")

print()
print("Classes:")
print(len(classes))

print()

for i, word in enumerate(classes):

    count = np.sum(
        y_encoded == i
    )

    print(
        f"{i:3d}  {word:15s}  {count:6d}"
    )


# =====================================================================
# NUMERICAL CHECK
# =====================================================================

print()
print("=" * 70)
print("NUMERICAL CHECK")
print("=" * 70)

print()
print("Minimum:", X.min())
print("Maximum:", X.max())
print("Mean   :", X.mean())
print("Std    :", X.std())

print()
print("NaN:", np.isnan(X).sum())
print("Inf:", np.isinf(X).sum())


# =====================================================================
# SAVE
# =====================================================================

print()
print("=" * 70)
print("SAVING DATASET")
print("=" * 70)

np.savez_compressed(
    OUTPUT_PATH,
    X=X,
    y=y_encoded,
    classes=np.asarray(classes),
    recordings=recordings,
    words=words,
    starts=starts,
    ends=ends
)

print()
print("Saved successfully:")
print(OUTPUT_PATH)


# =====================================================================
# FINAL
# =====================================================================

print()
print("=" * 70)
print("DONE")
print("=" * 70)

print()
print("CNN input:")
print("    X ->", X.shape)

print()
print("CNN labels:")
print("    y ->", y_encoded.shape)

print()
print("Number of classes:")
print("   ", len(classes))

print()
print("The dataset is now ready for CNN training.")

