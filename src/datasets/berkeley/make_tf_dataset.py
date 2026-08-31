# `berkeley_tf_dataset.py`

"""
======================================================================
IRIS - BERKELEY WORD-LEVEL TIME-FREQUENCY DATASET
======================================================================

Dataset:
    Berkeley Silent Speech EMG Dataset

Input:
    8-channel EMG

Alignment:
    Praat TextGrid files

TextGrid structure:
    Tier 1 = words
    Tier 2 = phones

IMPORTANT:
    EMG filenames:
        101_emg.npy
        102_emg.npy
        ...

    TextGrid filenames:
        5-9_101_audio.TextGrid
        5-9_102_audio.TextGrid
        ...

    Matching is done using the RECORDING NUMBER.

Output:
    X:
        (samples, 8, frequency_bins, time_frames)

    y:
        word labels

    metadata CSV

======================================================================
"""

import os
import re
import csv
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import stft


# ======================================================================
# CONFIGURATION
# ======================================================================

EMG_ROOT = Path(
    "/Users/vedantdwivedi/Desktop/IRIS/emg_data/closed_vocab/silent"
)

TEXTGRID_ROOT = Path(
    "/Users/vedantdwivedi/Desktop/IRIS/silent_speech_alignments/text_alignments/5-9"
)

OUTPUT_DIR = Path(
    "/Users/vedantdwivedi/Desktop/IRIS/Data/datasets"
)

OUTPUT_NPZ = OUTPUT_DIR / "berkeley_tf_dataset.npz"
OUTPUT_CSV = OUTPUT_DIR / "berkeley_tf_metadata.csv"


# ----------------------------------------------------------------------
# EMG parameters
# ----------------------------------------------------------------------

FS = 600

EXPECTED_CHANNELS = 8


# ----------------------------------------------------------------------
# STFT parameters
# ----------------------------------------------------------------------

# EMG frequency range of interest
FMIN = 20
FMAX = 250

# STFT window
N_FFT = 64

# Hop size
HOP_LENGTH = 16

# Fixed output dimensions
# Number of positive-frequency bins between 20 and 250 Hz
# is determined automatically.
#
# Time dimension is fixed using padding/cropping.

TARGET_TIME_FRAMES = 64


# ----------------------------------------------------------------------
# Minimum word duration
# ----------------------------------------------------------------------

MIN_WORD_DURATION = 0.03


# ======================================================================
# PRINT HEADER
# ======================================================================

def print_header():
    print("=" * 70)
    print("IRIS - BERKELEY WORD-LEVEL TIME-FREQUENCY DATASET")
    print("=" * 70)
    print()
    print("EMG root:")
    print(EMG_ROOT)
    print()
    print("TextGrid root:")
    print(TEXTGRID_ROOT)
    print()
    print("Output:")
    print(OUTPUT_NPZ)
    print()
    print("Sampling rate:", FS)
    print("Expected channels:", EXPECTED_CHANNELS)
    print("STFT n_fft:", N_FFT)
    print("STFT hop:", HOP_LENGTH)
    print("Frequency range:", FMIN, "-", FMAX, "Hz")
    print("Target time frames:", TARGET_TIME_FRAMES)
    print()


# ======================================================================
# FIND EMG FILES
# ======================================================================

def find_emg_files():
    """
    Find all *_emg.npy files recursively.
    """

    files = sorted(EMG_ROOT.rglob("*_emg.npy"))

    return files


# ======================================================================
# FIND TEXTGRID FILES
# ======================================================================

def find_textgrid_files():
    """
    Find all *_audio.TextGrid files recursively.
    """

    files = sorted(TEXTGRID_ROOT.rglob("*_audio.TextGrid"))

    return files


# ======================================================================
# EXTRACT RECORDING ID FROM EMG FILE
# ======================================================================

def extract_emg_id(path):
    """
    Example:

        101_emg.npy -> 101
        5_emg.npy   -> 5
    """

    match = re.search(r"(\d+)_emg\.npy$", path.name)

    if match is None:
        return None

    return int(match.group(1))


# ======================================================================
# EXTRACT RECORDING ID FROM TEXTGRID
# ======================================================================

def extract_textgrid_id(path):
    """
    Example:

        5-9_101_audio.TextGrid -> 101
        5-9_23_audio.TextGrid  -> 23

    IMPORTANT:
        We take the number immediately before "_audio.TextGrid".
    """

    match = re.search(r"_(\d+)_audio\.TextGrid$", path.name)

    if match is None:
        return None

    return int(match.group(1))


# ======================================================================
# BUILD TEXTGRID LOOKUP
# ======================================================================

def build_textgrid_lookup(textgrid_files):

    lookup = {}

    for path in textgrid_files:

        recording_id = extract_textgrid_id(path)

        if recording_id is None:
            continue

        # If duplicate IDs exist, keep the first one.
        if recording_id not in lookup:
            lookup[recording_id] = path

    return lookup


# ======================================================================
# DEBUG MATCHING
# ======================================================================

def diagnose_matching(emg_files, textgrid_lookup):

    print("=" * 70)
    print("FILE MATCHING")
    print("=" * 70)

    emg_ids = []

    for path in emg_files:
        rid = extract_emg_id(path)

        if rid is not None:
            emg_ids.append(rid)

    matched = []
    unmatched = []

    for rid in emg_ids:

        if rid in textgrid_lookup:
            matched.append(rid)
        else:
            unmatched.append(rid)

    print()
    print("EMG recordings:", len(emg_ids))
    print("TextGrid recordings:", len(textgrid_lookup))
    print("Matched recordings:", len(matched))
    print("Unmatched EMG recordings:", len(unmatched))

    print()

    print("First 20 matches:")

    for rid in matched[:20]:

        print(
            f"  {rid}_emg.npy"
            f"  <->  "
            f"{textgrid_lookup[rid].name}"
        )

    print()

    if unmatched:
        print("First 20 unmatched EMG IDs:")

        for rid in unmatched[:20]:
            print(" ", rid)

    print()


# ======================================================================
# LOAD EMG
# ======================================================================

def load_emg(path):
    """
    Load EMG .npy file.

    Expected possible shapes:

        (8, samples)
        (samples, 8)

    Returns:

        (8, samples)
    """

    data = np.load(path)

    data = np.asarray(data)

    # Remove singleton dimensions
    data = np.squeeze(data)

    if data.ndim != 2:
        raise ValueError(
            f"Expected 2D EMG array, got shape {data.shape}"
        )

    # Already channels x samples
    if data.shape[0] == EXPECTED_CHANNELS:
        return data.astype(np.float32)

    # Samples x channels
    if data.shape[1] == EXPECTED_CHANNELS:
        return data.T.astype(np.float32)

    raise ValueError(
        f"Could not identify {EXPECTED_CHANNELS} channels "
        f"in EMG shape {data.shape}"
    )


# ======================================================================
# PARSE TEXTGRID
# ======================================================================

def parse_textgrid_words(path):
    """
    Parse the "words" IntervalTier from a Praat TextGrid.

    Returns:

        [
            {
                "word": "cash",
                "start": 1.350,
                "end": 1.700
            },
            ...
        ]

    Empty intervals are ignored.
    """

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # --------------------------------------------------------------
    # Find the words tier
    # --------------------------------------------------------------

    tier_match = re.search(
        r'name\s*=\s*"words".*?intervals:\s*size\s*=\s*\d+(.*?)(?=\n\s*item\s*\[|\Z)',
        text,
        flags=re.DOTALL
    )

    if tier_match is None:

        # Some TextGrids may have slightly different formatting.
        # Try a more permissive approach.
        tier_match = re.search(
            r'name\s*=\s*"words"(.*?)(?=\n\s*item\s*\[|\Z)',
            text,
            flags=re.DOTALL
        )

    if tier_match is None:
        return []

    tier_text = tier_match.group(1)

    # --------------------------------------------------------------
    # Parse intervals
    # --------------------------------------------------------------

    pattern = re.compile(
        r"""
        intervals\s*\[\d+\]\s*:
        \s*xmin\s*=\s*([0-9.eE+-]+)
        \s*xmax\s*=\s*([0-9.eE+-]+)
        \s*text\s*=\s*"([^"]*)"
        """,
        flags=re.VERBOSE
    )

    words = []

    for match in pattern.finditer(tier_text):

        start = float(match.group(1))
        end = float(match.group(2))
        word = match.group(3).strip()

        # Ignore empty labels
        if not word:
            continue

        # Ignore extremely short intervals
        if end <= start:
            continue

        if (end - start) < MIN_WORD_DURATION:
            continue

        words.append(
            {
                "word": word.upper(),
                "start": start,
                "end": end,
            }
        )

    return words


# ======================================================================
# ALTERNATIVE TEXTGRID PARSER
# ======================================================================

def parse_textgrid_words_fallback(path):
    """
    More robust fallback parser.

    Used if the main parser does not find words.
    """

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    words = []

    current_xmin = None
    current_xmax = None

    in_words_tier = False

    for i, line in enumerate(lines):

        stripped = line.strip()

        # Detect words tier
        if stripped == 'name = "words"':
            in_words_tier = True
            continue

        # If another item starts after words tier,
        # stop parsing this tier.
        if in_words_tier and re.match(r"item \[\d+\]:", stripped):
            if words:
                break

        if not in_words_tier:
            continue

        if stripped.startswith("xmin ="):
            try:
                current_xmin = float(
                    stripped.split("=", 1)[1].strip()
                )
            except Exception:
                current_xmin = None

        elif stripped.startswith("xmax ="):
            try:
                current_xmax = float(
                    stripped.split("=", 1)[1].strip()
                )
            except Exception:
                current_xmax = None

        elif stripped.startswith("text ="):

            match = re.search(
                r'text\s*=\s*"(.*)"',
                stripped
            )

            if match:

                word = match.group(1).strip()

                if (
                    word
                    and current_xmin is not None
                    and current_xmax is not None
                    and current_xmax > current_xmin
                    and (current_xmax - current_xmin)
                    >= MIN_WORD_DURATION
                ):

                    words.append(
                        {
                            "word": word.upper(),
                            "start": current_xmin,
                            "end": current_xmax,
                        }
                    )

                current_xmin = None
                current_xmax = None

    return words


# ======================================================================
# WORD PARSER
# ======================================================================

def get_words(path):

    words = parse_textgrid_words(path)

    if len(words) == 0:
        words = parse_textgrid_words_fallback(path)

    return words


# ======================================================================
# EXTRACT WORD SEGMENT
# ======================================================================

def extract_word_segment(emg, start_time, end_time):

    start_sample = int(round(start_time * FS))
    end_sample = int(round(end_time * FS))

    start_sample = max(0, start_sample)
    end_sample = min(emg.shape[1], end_sample)

    if end_sample <= start_sample:
        return None, None, None

    segment = emg[:, start_sample:end_sample]

    if segment.shape[1] < 4:
        return None, None, None

    return segment, start_sample, end_sample


# ======================================================================
# STFT
# ======================================================================

def compute_channel_stft(signal):

    """
    Compute magnitude STFT for one EMG channel.

    Returns:

        frequency x time
    """

    signal = np.asarray(signal, dtype=np.float32)

    # --------------------------------------------------------------
    # Important:
    #
    # Some words are extremely short.
    #
    # If the signal is shorter than N_FFT, reduce nperseg.
    # This prevents librosa/scipy warnings and invalid outputs.
    # --------------------------------------------------------------

    nperseg = min(N_FFT, len(signal))

    if nperseg < 4:
        return None

    noverlap = min(
        nperseg // 2,
        nperseg - 1
    )

    f, t, Zxx = stft(
        signal,
        fs=FS,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=N_FFT,
        boundary=None,
        padded=False
    )

    magnitude = np.abs(Zxx)

    # --------------------------------------------------------------
    # Frequency selection
    # --------------------------------------------------------------

    mask = (
        (f >= FMIN)
        &
        (f <= FMAX)
    )

    f_selected = f[mask]
    magnitude = magnitude[mask]

    if magnitude.size == 0:
        return None

    # --------------------------------------------------------------
    # Log magnitude
    #
    # log1p keeps zeros numerically safe.
    # --------------------------------------------------------------

    magnitude = np.log1p(magnitude)

    return magnitude.astype(np.float32)


# ======================================================================
# FIX TIME DIMENSION
# ======================================================================

def fix_time_dimension(tf, target_frames=TARGET_TIME_FRAMES):

    """
    Convert:

        (frequency, variable_time)

    into:

        (frequency, target_frames)

    Center crop if too long.

    Zero-pad if too short.
    """

    freq_bins, time_frames = tf.shape

    if time_frames == target_frames:
        return tf

    # --------------------------------------------------------------
    # Crop
    # --------------------------------------------------------------

    if time_frames > target_frames:

        start = (time_frames - target_frames) // 2

        end = start + target_frames

        return tf[:, start:end]

    # --------------------------------------------------------------
    # Pad
    # --------------------------------------------------------------

    output = np.zeros(
        (freq_bins, target_frames),
        dtype=np.float32
    )

    start = (target_frames - time_frames) // 2

    output[:, start:start + time_frames] = tf

    return output


# ======================================================================
# MULTI-CHANNEL TF
# ======================================================================

def compute_multichannel_tf(segment):

    """
    Input:

        (8, samples)

    Output:

        (8, frequency_bins, TARGET_TIME_FRAMES)
    """

    channel_features = []

    for channel in range(segment.shape[0]):

        tf = compute_channel_stft(
            segment[channel]
        )

        if tf is None:
            return None

        tf = fix_time_dimension(tf)

        channel_features.append(tf)

    return np.stack(
        channel_features,
        axis=0
    ).astype(np.float32)


# ======================================================================
# NORMALIZE TF SAMPLE
# ======================================================================

def normalize_tf_sample(tf):

    """
    Per-sample normalization.

    This prevents recordings with larger absolute EMG amplitudes
    from dominating the representation.

    Normalization is performed independently for each channel.
    """

    output = np.zeros_like(tf)

    for c in range(tf.shape[0]):

        x = tf[c]

        mean = np.mean(x)

        std = np.std(x)

        if std < 1e-8:
            output[c] = x - mean

        else:
            output[c] = (x - mean) / std

    return output.astype(np.float32)


# ======================================================================
# MAIN
# ======================================================================

def main():

    print_header()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==============================================================
    # FIND FILES
    # ==============================================================

    print("=" * 70)
    print("SEARCHING FOR FILES")
    print("=" * 70)
    print()

    emg_files = find_emg_files()

    textgrid_files = find_textgrid_files()

    print("EMG files:", len(emg_files))
    print("TextGrid files:", len(textgrid_files))
    print()

    if len(emg_files) == 0:
        raise RuntimeError(
            "No EMG files found."
        )

    if len(textgrid_files) == 0:
        raise RuntimeError(
            "No TextGrid files found."
        )

    # ==============================================================
    # TEXTGRID LOOKUP
    # ==============================================================

    textgrid_lookup = build_textgrid_lookup(
        textgrid_files
    )

    diagnose_matching(
        emg_files,
        textgrid_lookup
    )

    # ==============================================================
    # EXTRACTION
    # ==============================================================

    print("=" * 70)
    print("EXTRACTING WORD-LEVEL TIME-FREQUENCY REPRESENTATIONS")
    print("=" * 70)
    print()

    X_list = []
    y_list = []
    metadata = []

    recordings_with_alignment = 0
    recordings_without_alignment = 0

    total_word_intervals = 0
    total_segments = 0

    failed_emg = 0
    failed_tf = 0

    words_per_recording = []

    # --------------------------------------------------------------
    # Process each EMG file
    # --------------------------------------------------------------

    for index, emg_path in enumerate(emg_files):

        recording_id = extract_emg_id(
            emg_path
        )

        if recording_id is None:
            continue

        # ----------------------------------------------------------
        # Find matching TextGrid
        # ----------------------------------------------------------

        textgrid_path = textgrid_lookup.get(
            recording_id
        )

        if textgrid_path is None:

            recordings_without_alignment += 1

            continue

        recordings_with_alignment += 1

        # ----------------------------------------------------------
        # Load EMG
        # ----------------------------------------------------------

        try:

            emg = load_emg(
                emg_path
            )

        except Exception as e:

            failed_emg += 1

            print(
                f"[EMG ERROR] "
                f"{emg_path.name}: {e}"
            )

            continue

        # ----------------------------------------------------------
        # Get words
        # ----------------------------------------------------------

        words = get_words(
            textgrid_path
        )

        total_word_intervals += len(words)

        words_per_recording.append(
            len(words)
        )

        if len(words) == 0:
            continue

        # ----------------------------------------------------------
        # Process each word
        # ----------------------------------------------------------

        for word_info in words:

            word = word_info["word"]

            start_time = word_info["start"]

            end_time = word_info["end"]

            # ------------------------------------------------------
            # Extract actual aligned EMG segment
            # ------------------------------------------------------

            segment, start_sample, end_sample = (
                extract_word_segment(
                    emg,
                    start_time,
                    end_time
                )
            )

            if segment is None:
                continue

            # ------------------------------------------------------
            # Compute STFT
            # ------------------------------------------------------

            try:

                tf = compute_multichannel_tf(
                    segment
                )

            except Exception as e:

                failed_tf += 1

                print(
                    f"[TF ERROR] "
                    f"{emg_path.name} "
                    f"word={word}: {e}"
                )

                continue

            if tf is None:

                failed_tf += 1

                continue

            # ------------------------------------------------------
            # Normalize
            # ------------------------------------------------------

            tf = normalize_tf_sample(
                tf
            )

            # ------------------------------------------------------
            # Store
            # ------------------------------------------------------

            X_list.append(tf)

            y_list.append(word)

            metadata.append(
                {
                    "recording_id": recording_id,
                    "emg_file": str(emg_path),
                    "textgrid_file": str(textgrid_path),
                    "word": word,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "emg_samples": end_sample - start_sample,
                }
            )

            total_segments += 1

        # ----------------------------------------------------------
        # Progress
        # ----------------------------------------------------------

        if (index + 1) % 50 == 0:

            print(
                f"Processed "
                f"{index + 1}/{len(emg_files)} "
                f"| TF samples: {total_segments}"
            )

    # ==============================================================
    # SUMMARY
    # ==============================================================

    print()
    print("=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    print()

    print("EMG files:", len(emg_files))
    print("TextGrid files:", len(textgrid_files))
    print(
        "Recordings with alignment:",
        recordings_with_alignment
    )
    print(
        "Recordings without alignment:",
        recordings_without_alignment
    )
    print(
        "Word intervals found:",
        total_word_intervals
    )
    print(
        "Word segments extracted:",
        total_segments
    )
    print(
        "Failed EMG files:",
        failed_emg
    )
    print(
        "Failed TF extractions:",
        failed_tf
    )

    print()

    # ==============================================================
    # CHECK DATA
    # ==============================================================

    if len(X_list) == 0:

        raise RuntimeError(
            "\n"
            "NO WORD-LEVEL TF SAMPLES WERE CREATED.\n\n"
            "The EMG/TextGrid matching worked, but no valid "
            "word segments were converted into TF representations."
        )

    # ==============================================================
    # STACK
    # ==============================================================

    print("=" * 70)
    print("BUILDING DATASET")
    print("=" * 70)
    print()

    X = np.stack(
        X_list,
        axis=0
    ).astype(np.float32)

    y = np.asarray(
        y_list
    )

    # ==============================================================
    # PRINT SHAPE
    # ==============================================================

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print()

    print(
        "Expected:",
        "(samples, 8, frequency_bins, time_frames)"
    )

    print(
        "Samples:",
        X.shape[0]
    )

    print(
        "Channels:",
        X.shape[1]
    )

    print(
        "Frequency bins:",
        X.shape[2]
    )

    print(
        "Time frames:",
        X.shape[3]
    )

    # ==============================================================
    # UNIQUE WORDS
    # ==============================================================

    unique_words, counts = np.unique(
        y,
        return_counts=True
    )

    print()
    print(
        "Unique words:",
        len(unique_words)
    )

    print()
    print("Top 30 words:")

    order = np.argsort(
        counts
    )[::-1]

    for idx in order[:30]:

        print(
            f"  {unique_words[idx]:20s}"
            f" {counts[idx]}"
        )

    # ==============================================================
    # SAVE NPZ
    # ==============================================================

    print()
    print("Saving dataset...")

    np.savez_compressed(
        OUTPUT_NPZ,
        X=X,
        y=y,
        classes=unique_words,
        fs=np.array(FS),
        fmin=np.array(FMIN),
        fmax=np.array(FMAX),
        n_fft=np.array(N_FFT),
        hop_length=np.array(HOP_LENGTH),
    )

    # ==============================================================
    # SAVE METADATA CSV
    # ==============================================================

    print("Saving metadata...")

    fieldnames = [
        "recording_id",
        "emg_file",
        "textgrid_file",
        "word",
        "start_time",
        "end_time",
        "duration",
        "start_sample",
        "end_sample",
        "emg_samples",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            metadata
        )

    # ==============================================================
    # FINAL
    # ==============================================================

    print()
    print("=" * 70)
    print("DATASET CREATED SUCCESSFULLY")
    print("=" * 70)
    print()

    print("NPZ:")
    print(OUTPUT_NPZ)

    print()

    print("Metadata:")
    print(OUTPUT_CSV)

    print()

    print("Final X shape:")
    print(X.shape)

    print()

    print("Final y shape:")
    print(y.shape)

    print()

    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()

