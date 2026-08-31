"""
======================================================================
IRIS - BERKELEY WORD-LEVEL MFCC DATASET
======================================================================

Creates an MFCC dataset using EXACTLY the same:
    - EMG recordings
    - TextGrid recordings
    - word intervals
    - word labels
    - word-level segments

as the Berkeley TF dataset.

Therefore:

    TF dataset:
        (N, 8, 24, 64)

    MFCC dataset:
        (N, 8, 13, 64)

The samples and labels are directly comparable.

======================================================================
"""

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import librosa

from pathlib import Path


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

OUTPUT_NPZ = OUTPUT_DIR / "berkeley_mfcc_dataset.npz"
OUTPUT_CSV = OUTPUT_DIR / "berkeley_mfcc_metadata.csv"


# ======================================================================
# SIGNAL PARAMETERS
# ======================================================================

FS = 600

EXPECTED_CHANNELS = 8

# MFCC parameters
N_FFT = 64
HOP_LENGTH = 16

N_MFCC = 13
N_MELS = 20

FMIN = 20
FMAX = 250

TARGET_FRAMES = 64


# ======================================================================
# PRINT CONFIGURATION
# ======================================================================

print("=" * 70)
print("IRIS - BERKELEY WORD-LEVEL MFCC DATASET")
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
print("MFCC n_fft:", N_FFT)
print("MFCC hop:", HOP_LENGTH)
print("MFCC coefficients:", N_MFCC)
print("Mel filters:", N_MELS)
print("Frequency range:", FMIN, "-", FMAX, "Hz")
print("Target time frames:", TARGET_FRAMES)


# ======================================================================
# TEXTGRID PARSER
# ======================================================================

def parse_textgrid(textgrid_path):
    """
    Parse a Praat TextGrid.

    Returns:
        words = list of:
            {
                "start": float,
                "end": float,
                "text": str
            }
    """

    with open(textgrid_path, "r", encoding="utf-8") as f:
        text = f.read()

    # --------------------------------------------------------------
    # Find the "words" tier
    # --------------------------------------------------------------

    words_match = re.search(
        r'name\s*=\s*"words".*?intervals:\s*size\s*=\s*\d+(.*?)(?=\n\s*item \[|\Z)',
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    if words_match is None:
        return []

    words_section = words_match.group(1)

    # --------------------------------------------------------------
    # Extract intervals
    # --------------------------------------------------------------

    pattern = re.compile(
        r'xmin\s*=\s*([0-9.eE+-]+).*?'
        r'xmax\s*=\s*([0-9.eE+-]+).*?'
        r'text\s*=\s*"(.*?)"',
        flags=re.DOTALL
    )

    intervals = []

    for match in pattern.finditer(words_section):

        start = float(match.group(1))
        end = float(match.group(2))
        label = match.group(3).strip()

        # Ignore empty intervals
        if not label:
            continue

        # Normalize label
        label = label.strip().upper()

        intervals.append({
            "start": start,
            "end": end,
            "text": label
        })

    return intervals


# ======================================================================
# RECORDING ID EXTRACTION
# ======================================================================

def get_emg_recording_id(path):
    """
    Berkeley EMG filenames look like:

        100_emg.npy
        101_emg.npy
        5_emg.npy

    Return:
        "100"
        "101"
        "5"
    """

    match = re.match(
        r"^(\d+)_emg\.npy$",
        path.name,
        flags=re.IGNORECASE
    )

    if match:
        return match.group(1)

    return None


def get_textgrid_recording_id(path):
    """
    Berkeley TextGrid filenames look like:

        5-9_100_audio.TextGrid
        5-9_101_audio.TextGrid

    Return:
        "100"
        "101"
    """

    match = re.search(
        r"_(\d+)_audio\.TextGrid$",
        path.name,
        flags=re.IGNORECASE
    )

    if match:
        return match.group(1)

    return None


# ======================================================================
# FIND EMG FILES
# ======================================================================

print()
print("=" * 70)
print("SEARCHING FOR FILES")
print("=" * 70)

emg_files = sorted(
    EMG_ROOT.rglob("*_emg.npy")
)

textgrid_files = sorted(
    TEXTGRID_ROOT.rglob("*.TextGrid")
)

print()
print("EMG files:", len(emg_files))
print("TextGrid files:", len(textgrid_files))


# ======================================================================
# BUILD TEXTGRID INDEX
# ======================================================================

print()
print("=" * 70)
print("BUILDING TEXTGRID INDEX")
print("=" * 70)

textgrid_index = {}

for tg in textgrid_files:

    recording_id = get_textgrid_recording_id(tg)

    if recording_id is None:
        continue

    textgrid_index[recording_id] = tg


print()
print("TextGrid recording IDs:", len(textgrid_index))


# ======================================================================
# MATCH EMG TO TEXTGRID
# ======================================================================

print()
print("=" * 70)
print("FILE MATCHING")
print("=" * 70)

matched = []
unmatched = []

for emg_path in emg_files:

    recording_id = get_emg_recording_id(emg_path)

    if recording_id is None:
        continue

    if recording_id in textgrid_index:

        matched.append(
            (
                emg_path,
                textgrid_index[recording_id],
                recording_id
            )
        )

    else:

        unmatched.append(
            (
                emg_path,
                recording_id
            )
        )


print()
print("EMG recordings:", len(emg_files))
print("TextGrid recordings:", len(textgrid_files))
print("Matched recordings:", len(matched))
print("Unmatched EMG recordings:", len(unmatched))


print()
print("First 20 matches:")

for emg_path, tg_path, recording_id in matched[:20]:

    print(
        f"  {emg_path.name}  <->  {tg_path.name}"
    )


# ======================================================================
# WORD SEGMENT EXTRACTION
# ======================================================================

print()
print("=" * 70)
print("EXTRACTING WORD-LEVEL EMG SEGMENTS")
print("=" * 70)


segments = []

recordings_with_alignment = 0
recordings_without_alignment = 0

word_intervals_found = 0

failed_emg_files = 0


for idx, (emg_path, tg_path, recording_id) in enumerate(matched):

    try:

        # ----------------------------------------------------------
        # Load EMG
        # ----------------------------------------------------------

        emg = np.load(emg_path)

        emg = np.asarray(emg)

        # ----------------------------------------------------------
        # Normalize shape to:
        #
        #     (channels, samples)
        #
        # ----------------------------------------------------------

        if emg.ndim == 1:

            if emg.size % EXPECTED_CHANNELS != 0:
                failed_emg_files += 1
                continue

            emg = emg.reshape(
                EXPECTED_CHANNELS,
                -1
            )

        elif emg.ndim == 2:

            if emg.shape[0] == EXPECTED_CHANNELS:

                pass

            elif emg.shape[1] == EXPECTED_CHANNELS:

                emg = emg.T

            else:

                failed_emg_files += 1
                continue

        else:

            failed_emg_files += 1
            continue


        # ----------------------------------------------------------
        # Verify channels
        # ----------------------------------------------------------

        if emg.shape[0] != EXPECTED_CHANNELS:

            failed_emg_files += 1
            continue


        # ----------------------------------------------------------
        # Parse TextGrid
        # ----------------------------------------------------------

        word_intervals = parse_textgrid(tg_path)

        if len(word_intervals) == 0:

            recordings_without_alignment += 1
            continue

        recordings_with_alignment += 1

        word_intervals_found += len(word_intervals)


        # ----------------------------------------------------------
        # Extract each word
        # ----------------------------------------------------------

        total_samples = emg.shape[1]

        duration = total_samples / FS


        for word_index, interval in enumerate(word_intervals):

            word = interval["text"]

            start_time = interval["start"]
            end_time = interval["end"]


            # ------------------------------------------------------
            # Ignore invalid intervals
            # ------------------------------------------------------

            if end_time <= start_time:
                continue


            # ------------------------------------------------------
            # Convert seconds → samples
            # ------------------------------------------------------

            start_sample = int(
                round(start_time * FS)
            )

            end_sample = int(
                round(end_time * FS)
            )


            # ------------------------------------------------------
            # Clamp to EMG
            # ------------------------------------------------------

            start_sample = max(
                0,
                start_sample
            )

            end_sample = min(
                total_samples,
                end_sample
            )


            if end_sample <= start_sample:
                continue


            # ------------------------------------------------------
            # Extract
            # ------------------------------------------------------

            segment = emg[
                :,
                start_sample:end_sample
            ]


            # ------------------------------------------------------
            # Minimum length
            #
            # We need enough samples for STFT.
            # Very short words are skipped instead of generating
            # meaningless padded MFCCs.
            # ------------------------------------------------------

            if segment.shape[1] < 8:
                continue


            segments.append({

                "emg": segment.astype(
                    np.float32,
                    copy=False
                ),

                "label": word,

                "recording_id": recording_id,

                "emg_file": str(emg_path),

                "textgrid_file": str(tg_path),

                "word_index": word_index,

                "start_time": start_time,

                "end_time": end_time,

                "duration": end_time - start_time,

                "start_sample": start_sample,

                "end_sample": end_sample
            })


    except Exception as e:

        failed_emg_files += 1

        print(
            f"WARNING: failed {emg_path.name}: {e}"
        )


    # --------------------------------------------------------------
    # Progress
    # --------------------------------------------------------------

    if (idx + 1) % 50 == 0:

        print(
            f"Processed {idx + 1}/{len(matched)} "
            f"| Word segments: {len(segments)}"
        )


# ======================================================================
# EXTRACTION SUMMARY
# ======================================================================

print()
print("=" * 70)
print("EXTRACTION SUMMARY")
print("=" * 70)

print()
print("EMG files:", len(emg_files))
print("TextGrid files:", len(textgrid_files))
print("Matched recordings:", len(matched))
print("Recordings with alignment:", recordings_with_alignment)
print("Recordings without alignment:", recordings_without_alignment)
print("Word intervals found:", word_intervals_found)
print("Word segments extracted:", len(segments))
print("Failed EMG files:", failed_emg_files)


if len(segments) == 0:

    raise RuntimeError(
        "\n"
        "NO WORD-LEVEL MFCC SAMPLES WERE CREATED.\n\n"
        "Check EMG/TextGrid matching and TextGrid word parsing."
    )


# ======================================================================
# MFCC EXTRACTION
# ======================================================================

print()
print("=" * 70)
print("EXTRACTING MFCCs")
print("=" * 70)


def extract_channel_mfcc(signal):
    """
    Extract MFCC for one EMG channel.

    Output:
        (13, 64)

    Important:
        Short word segments are handled by zero-padding the
        signal to N_FFT samples before STFT.
    """

    signal = np.asarray(
        signal,
        dtype=np.float32
    )


    # --------------------------------------------------------------
    # Remove NaN / Inf
    # --------------------------------------------------------------

    signal = np.nan_to_num(
        signal,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    # --------------------------------------------------------------
    # Remove DC
    # --------------------------------------------------------------

    if signal.size > 0:

        signal = signal - np.mean(signal)


    # --------------------------------------------------------------
    # Prevent librosa warning:
    #
    # n_fft > signal length
    #
    # Pad only when necessary.
    # --------------------------------------------------------------

    if len(signal) < N_FFT:

        padded = np.zeros(
            N_FFT,
            dtype=np.float32
        )

        padded[:len(signal)] = signal

        signal = padded


    # --------------------------------------------------------------
    # MFCC
    # --------------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=FS,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        center=False
    )


    # --------------------------------------------------------------
    # Force exactly TARGET_FRAMES
    # --------------------------------------------------------------

    if mfcc.shape[1] < TARGET_FRAMES:

        pad_width = (
            0,
            TARGET_FRAMES - mfcc.shape[1]
        )

        mfcc = np.pad(
            mfcc,
            (
                (0, 0),
                pad_width
            ),
            mode="constant"
        )

    elif mfcc.shape[1] > TARGET_FRAMES:

        mfcc = mfcc[
            :,
            :TARGET_FRAMES
        ]


    return mfcc.astype(
        np.float32,
        copy=False
    )


# ======================================================================
# PROCESS ALL WORD SEGMENTS
# ======================================================================

X_list = []
y_list = []

metadata = []

failed_mfcc = 0


total = len(segments)


for i, item in enumerate(segments):

    try:

        emg = item["emg"]


        # ----------------------------------------------------------
        # Must be 8 channels
        # ----------------------------------------------------------

        if emg.shape[0] != EXPECTED_CHANNELS:

            failed_mfcc += 1
            continue


        channel_features = []


        # ----------------------------------------------------------
        # Extract MFCC independently for every EMG channel
        # ----------------------------------------------------------

        for channel in range(EXPECTED_CHANNELS):

            signal = emg[channel]

            mfcc = extract_channel_mfcc(
                signal
            )

            channel_features.append(
                mfcc
            )


        # ----------------------------------------------------------
        # Shape:
        #
        #     (8, 13, 64)
        # ----------------------------------------------------------

        features = np.stack(
            channel_features,
            axis=0
        )


        if features.shape != (
            EXPECTED_CHANNELS,
            N_MFCC,
            TARGET_FRAMES
        ):

            failed_mfcc += 1
            continue


        X_list.append(features)

        y_list.append(item["label"])


        metadata.append({

            "sample_index": len(X_list) - 1,

            "recording_id": item["recording_id"],

            "emg_file": item["emg_file"],

            "textgrid_file": item["textgrid_file"],

            "word_index": item["word_index"],

            "word": item["label"],

            "start_time": item["start_time"],

            "end_time": item["end_time"],

            "duration": item["duration"],

            "start_sample": item["start_sample"],

            "end_sample": item["end_sample"]
        })


    except Exception as e:

        failed_mfcc += 1

        print(
            f"WARNING: MFCC failed for sample {i}: {e}"
        )


    # --------------------------------------------------------------
    # Progress
    # --------------------------------------------------------------

    if (i + 1) % 500 == 0:

        print(
            f"Processed {i + 1}/{total}"
        )


# ======================================================================
# BUILD ARRAYS
# ======================================================================

print()
print("=" * 70)
print("BUILDING DATASET")
print("=" * 70)


if len(X_list) == 0:

    raise RuntimeError(
        "\n"
        "MFCC extraction produced zero samples."
    )


X = np.stack(
    X_list,
    axis=0
)

y = np.asarray(
    y_list
)


# ======================================================================
# FINAL SHAPE CHECK
# ======================================================================

expected_shape = (
    len(X),
    EXPECTED_CHANNELS,
    N_MFCC,
    TARGET_FRAMES
)


if X.shape != expected_shape:

    raise RuntimeError(
        f"\nUnexpected X shape: {X.shape}\n"
        f"Expected: {expected_shape}"
    )


print()
print("X shape:", X.shape)
print("y shape:", y.shape)

print()
print(
    "Expected:",
    f"(samples, {EXPECTED_CHANNELS}, {N_MFCC}, {TARGET_FRAMES})"
)

print()
print("Samples:", X.shape[0])
print("Channels:", X.shape[1])
print("MFCC coefficients:", X.shape[2])
print("Time frames:", X.shape[3])

print()
print("Unique words:", len(np.unique(y)))

print()
print("Top 30 words:")

unique, counts = np.unique(
    y,
    return_counts=True
)

order = np.argsort(
    counts
)[::-1]

for idx in order[:30]:

    print(
        f"  {unique[idx]:20s} {counts[idx]}"
    )


# ======================================================================
# SAVE DATASET
# ======================================================================

print()
print("=" * 70)
print("SAVING DATASET")
print("=" * 70)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


np.savez_compressed(
    OUTPUT_NPZ,
    X=X,
    y=y
)


# ======================================================================
# SAVE METADATA
# ======================================================================

metadata_df = pd.DataFrame(
    metadata
)


metadata_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ======================================================================
# FINAL VALIDATION
# ======================================================================

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
print("Failed MFCC extractions:", failed_mfcc)

print()
print("=" * 70)
print("DONE")
print("=" * 70)
