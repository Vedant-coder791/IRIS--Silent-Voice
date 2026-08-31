"""
======================================================================
IRIS - BERKELEY WORD-LEVEL ALIGNMENT
======================================================================

Uses:
    Berkeley silent EMG recordings
    Berkeley TextGrid word alignments

DO NOT use the JSON "chunks" for word alignment.

Input:
    EMG:
        /Users/vedantdwivedi/Desktop/IRIS/emg_data/closed_vocab/silent

    TextGrids:
        /Users/vedantdwivedi/Desktop/IRIS/silent_speech_alignments/text_alignments

Output:
    /Users/vedantdwivedi/Desktop/IRIS/Data/datasets/
        berkeley_word_segments.npz
        berkeley_word_metadata.csv

Each sample:
    X[i] = word-level EMG
    y[i] = word label

Shape:
    (samples, time, channels)
"""

import os
import re
import json
import csv
import numpy as np


# ======================================================================
# PATHS
# ======================================================================

EMG_ROOT = (
    "/Users/vedantdwivedi/Desktop/IRIS/"
    "emg_data/closed_vocab/silent"
)

TEXTGRID_ROOT = (
    "/Users/vedantdwivedi/Desktop/IRIS/"
    "silent_speech_alignments/text_alignments"
)

OUTPUT_DIR = (
    "/Users/vedantdwivedi/Desktop/IRIS/"
    "Data/datasets"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_NPZ = os.path.join(
    OUTPUT_DIR,
    "berkeley_word_segments.npz"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "berkeley_word_metadata.csv"
)


# ======================================================================
# SETTINGS
# ======================================================================

FS = 600

MIN_WORD_DURATION = 0.05
MAX_WORD_DURATION = 3.0

# Remove these labels
IGNORE_WORDS = {
    "",
    "sil",
    "sp",
    "spn",
    "noise",
    "pau",
}


# ======================================================================
# TEXTGRID PARSER
# ======================================================================

def parse_textgrid(path):
    """
    Parse the 'words' IntervalTier from a Praat TextGrid.

    Returns:
        list of:
            {
                "start": float,
                "end": float,
                "word": str
            }
    """

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # --------------------------------------------------------------
    # Find words tier
    # --------------------------------------------------------------

    words_match = re.search(
        r'name\s*=\s*"words"(.*?)(?=\n\s*item\s*\[\d+\]:|\Z)',
        text,
        flags=re.DOTALL
    )

    if words_match is None:
        return []

    words_section = words_match.group(1)

    # --------------------------------------------------------------
    # Find intervals
    # --------------------------------------------------------------

    pattern = re.compile(
        r'intervals\s*\[\d+\]:\s*'
        r'xmin\s*=\s*([0-9.eE+-]+)\s*'
        r'xmax\s*=\s*([0-9.eE+-]+)\s*'
        r'text\s*=\s*"(.*?)"',
        flags=re.DOTALL
    )

    intervals = []

    for match in pattern.finditer(words_section):

        start = float(match.group(1))
        end = float(match.group(2))
        word = match.group(3).strip()

        word_upper = word.upper()

        if word_upper in IGNORE_WORDS:
            continue

        duration = end - start

        if duration < MIN_WORD_DURATION:
            continue

        if duration > MAX_WORD_DURATION:
            continue

        intervals.append({
            "start": start,
            "end": end,
            "word": word_upper
        })

    return intervals


# ======================================================================
# FIND EMG FILE
# ======================================================================

def find_emg_file(textgrid_path):
    """
    Example:

        5-9_23_audio.TextGrid

    becomes:

        23_emg.npy

    Search recursively because the Berkeley directory structure
    may contain multiple sessions.
    """

    filename = os.path.basename(textgrid_path)

    match = re.search(
        r'_(\d+)_audio\.TextGrid$',
        filename
    )

    if match is None:
        return None

    recording_id = match.group(1)

    target = recording_id + "_emg.npy"

    matches = []

    for root, dirs, files in os.walk(EMG_ROOT):

        if target in files:
            matches.append(
                os.path.join(root, target)
            )

    if len(matches) == 0:
        return None

    # Usually there should be exactly one.
    # If multiple exist, prefer matching session folder.
    textgrid_parent = os.path.basename(
        os.path.dirname(textgrid_path)
    )

    for path in matches:

        if textgrid_parent in path:
            return path

    return matches[0]


# ======================================================================
# FIND SUBJECT / SESSION
# ======================================================================

def get_session_from_textgrid(path):

    parts = os.path.normpath(path).split(os.sep)

    # Usually something like:
    #
    # text_alignments / 5-9 / 5-9_23_audio.TextGrid

    for part in reversed(parts):

        if re.match(r'^\d+-\d+$', part):
            return part

    return "unknown"


# ======================================================================
# MAIN
# ======================================================================

print("=" * 70)
print("IRIS - BERKELEY WORD-LEVEL ALIGNMENT")
print("=" * 70)

print()
print("EMG root:")
print(EMG_ROOT)

print()
print("TextGrid root:")
print(TEXTGRID_ROOT)


# ======================================================================
# FIND TEXTGRIDS
# ======================================================================

textgrids = []

for root, dirs, files in os.walk(TEXTGRID_ROOT):

    for file in files:

        if file.endswith(".TextGrid"):

            textgrids.append(
                os.path.join(root, file)
            )

textgrids.sort()

print()
print("=" * 70)
print("TEXTGRID SEARCH")
print("=" * 70)

print("TextGrids found:", len(textgrids))


# ======================================================================
# EXTRACT WORD SEGMENTS
# ======================================================================

segments = []
metadata = []

missing_emg = 0
no_words = 0
failed = 0

word_counts = {}


for i, tg_path in enumerate(textgrids):

    if (i + 1) % 100 == 0:
        print(
            f"Processed {i + 1}/{len(textgrids)}..."
        )

    try:

        words = parse_textgrid(tg_path)

        if len(words) == 0:
            no_words += 1
            continue

        emg_path = find_emg_file(tg_path)

        if emg_path is None:
            missing_emg += 1
            continue

        emg = np.load(emg_path)

        # ----------------------------------------------------------
        # Check EMG
        # ----------------------------------------------------------

        if emg.ndim != 2:
            failed += 1
            continue

        if emg.shape[1] != 8:
            failed += 1
            continue

        n_samples = emg.shape[0]

        # ----------------------------------------------------------
        # Session
        # ----------------------------------------------------------

        session = get_session_from_textgrid(
            tg_path
        )

        # ----------------------------------------------------------
        # Extract every word
        # ----------------------------------------------------------

        for word_info in words:

            start_time = word_info["start"]
            end_time = word_info["end"]
            word = word_info["word"]

            # Convert seconds -> samples

            start_sample = int(
                round(start_time * FS)
            )

            end_sample = int(
                round(end_time * FS)
            )

            # Clamp to EMG
            start_sample = max(
                0,
                start_sample
            )

            end_sample = min(
                n_samples,
                end_sample
            )

            if end_sample <= start_sample:
                continue

            segment = emg[
                start_sample:end_sample
            ]

            if len(segment) < 20:
                continue

            # ------------------------------------------------------
            # Save
            # ------------------------------------------------------

            segments.append(segment.astype(np.float32))

            metadata.append({
                "word": word,
                "textgrid": tg_path,
                "emg_file": emg_path,
                "session": session,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "samples": len(segment),
                "channels": segment.shape[1],
            })

            word_counts[word] = (
                word_counts.get(word, 0) + 1
            )

    except Exception as e:

        failed += 1

        print(
            "\nFAILED:",
            tg_path,
            "\n",
            repr(e)
        )


# ======================================================================
# SUMMARY
# ======================================================================

print()
print("=" * 70)
print("ALIGNMENT SUMMARY")
print("=" * 70)

print("TextGrids:", len(textgrids))
print("Word segments:", len(segments))
print("Missing EMG:", missing_emg)
print("No words:", no_words)
print("Failed:", failed)


# ======================================================================
# WORD COUNTS
# ======================================================================

print()
print("Unique words:", len(word_counts))

print()
print("Top words:")

for word, count in sorted(
    word_counts.items(),
    key=lambda x: x[1],
    reverse=True
)[:50]:

    print(
        f"{word:20s} {count:5d}"
    )


# ======================================================================
# PAD SEGMENTS
# ======================================================================

if len(segments) == 0:

    raise RuntimeError(
        "No word segments were extracted."
    )


max_length = max(
    len(x)
    for x in segments
)

median_length = int(
    np.median(
        [len(x) for x in segments]
    )
)

print()
print("=" * 70)
print("SEGMENT LENGTH")
print("=" * 70)

print("Minimum:", min(len(x) for x in segments))
print("Maximum:", max(len(x) for x in segments))
print("Median:", median_length)
print("Maximum used for storage:", max_length)


# ======================================================================
# SAVE VARIABLE-LENGTH SEGMENTS
# ======================================================================
#
# We intentionally DO NOT resize here.
#
# The next feature-extraction scripts will standardize the
# representation after extracting the signal features.
#
# Object arrays are used because word durations differ.

X = np.empty(
    len(segments),
    dtype=object
)

for i, segment in enumerate(segments):
    X[i] = segment


y = np.array(
    [m["word"] for m in metadata]
)


# ======================================================================
# SAVE NPZ
# ======================================================================

np.savez_compressed(
    OUTPUT_NPZ,
    X=X,
    y=y
)


# ======================================================================
# SAVE CSV
# ======================================================================

fieldnames = [
    "word",
    "textgrid",
    "emg_file",
    "session",
    "start_time",
    "end_time",
    "duration",
    "start_sample",
    "end_sample",
    "samples",
    "channels",
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

    writer.writerows(metadata)


# ======================================================================
# FINAL
# ======================================================================

print()
print("=" * 70)
print("SAVED")
print("=" * 70)

print()
print(OUTPUT_NPZ)

print()
print(OUTPUT_CSV)

print()
print("=" * 70)
print("DONE")
print("=" * 70)