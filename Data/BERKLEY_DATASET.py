import os
import json
import re
import numpy as np
from collections import Counter, defaultdict


# ============================================================
# BERKELEY SILENT SPEECH DATASET
# WORD FREQUENCY ANALYSIS
#
# Dataset:
# closed_vocab/silent/5-19_silent
#
# Purpose:
# 1. Find all valid silent recordings
# 2. Read their text metadata
# 3. Count how many recordings contain each word
# 4. Count total word occurrences
# 5. Show words with enough examples
# 6. Check sentence/chunk consistency
#
# IMPORTANT:
# This version counts words from the TEXT metadata.
# It does NOT yet assume that chunks = words.
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = (
    "/Users/vedantdwivedi/Desktop/IRIS/"
    "emg_data/closed_vocab/silent/5-19_silent"
)

MIN_EXAMPLES = 5

TOP_N = 100


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("BERKELEY SILENT SPEECH WORD FREQUENCY ANALYSIS")
print("=" * 70)

print("\nDataset directory:")
print(DATASET_DIR)


# ============================================================
# CHECK DIRECTORY
# ============================================================

if not os.path.isdir(DATASET_DIR):

    raise FileNotFoundError(
        "\nDataset directory not found:\n"
        f"{DATASET_DIR}\n\n"
        "Check the path."
    )


# ============================================================
# FIND EMG FILES
# ============================================================

print("\n" + "=" * 70)
print("SEARCHING FOR EMG FILES")
print("=" * 70)


emg_files = []

for filename in os.listdir(DATASET_DIR):

    if filename.endswith("_emg.npy"):

        emg_files.append(
            filename
        )


emg_files.sort()


print(
    "\nEMG files found:",
    len(emg_files)
)


# ============================================================
# WORD NORMALIZATION
# ============================================================

def normalize_text(text):

    """
    Convert sentence text into clean uppercase words.

    Example:

    'Monday February 12'
        ->
    ['MONDAY', 'FEBRUARY', '12']
    """

    text = str(text).upper()

    # Replace punctuation with spaces
    text = re.sub(
        r"[^A-Z0-9]+",
        " ",
        text
    )

    words = text.split()

    return words


# ============================================================
# STORAGE
# ============================================================

total_recordings = 0
valid_labeled_recordings = 0
reference_recordings = 0
broken_recordings = 0


# Number of times a word appears
word_occurrences = Counter()


# Number of different recordings containing the word
word_recordings = Counter()


# Store which recordings contain each word
word_to_recordings = defaultdict(list)


# Sentence lengths
sentence_word_counts = []


# Chunk statistics
chunk_counts = Counter()


# ============================================================
# PROCESS RECORDINGS
# ============================================================

print("\n" + "=" * 70)
print("ANALYZING RECORDINGS")
print("=" * 70)


for index, emg_filename in enumerate(
    emg_files,
    start=1
):

    total_recordings += 1


    # --------------------------------------------------------
    # EMG path
    # --------------------------------------------------------

    emg_path = os.path.join(
        DATASET_DIR,
        emg_filename
    )


    # --------------------------------------------------------
    # Corresponding JSON
    # --------------------------------------------------------

    info_filename = emg_filename.replace(
        "_emg.npy",
        "_info.json"
    )

    info_path = os.path.join(
        DATASET_DIR,
        info_filename
    )


    # --------------------------------------------------------
    # Check JSON
    # --------------------------------------------------------

    if not os.path.isfile(info_path):

        reference_recordings += 1

        continue


    # --------------------------------------------------------
    # Load JSON
    # --------------------------------------------------------

    try:

        with open(
            info_path,
            "r",
            encoding="utf-8"
        ) as f:

            metadata = json.load(f)

    except Exception:

        broken_recordings += 1

        continue


    # --------------------------------------------------------
    # Get sentence
    # --------------------------------------------------------

    text = metadata.get(
        "text",
        ""
    )


    # --------------------------------------------------------
    # Reference recording
    # --------------------------------------------------------

    if not text or not str(text).strip():

        reference_recordings += 1

        continue


    valid_labeled_recordings += 1


    # --------------------------------------------------------
    # Extract words
    # --------------------------------------------------------

    words = normalize_text(
        text
    )


    if len(words) == 0:

        continue


    # --------------------------------------------------------
    # Sentence statistics
    # --------------------------------------------------------

    sentence_word_counts.append(
        len(words)
    )


    # --------------------------------------------------------
    # Count occurrences
    #
    # Example:
    #
    # "THE CAT AND THE DOG"
    #
    # THE appears twice
    #
    # occurrence count = 2
    # --------------------------------------------------------

    for word in words:

        word_occurrences[word] += 1


    # --------------------------------------------------------
    # Count recordings
    #
    # Each word gets counted only once per recording.
    #
    # Example:
    #
    # "THE DOG AND THE CAT"
    #
    # THE -> one recording
    # --------------------------------------------------------

    unique_words = set(words)


    for word in unique_words:

        word_recordings[word] += 1

        word_to_recordings[word].append(
            emg_filename
        )


    # --------------------------------------------------------
    # Chunk information
    # --------------------------------------------------------

    chunks = metadata.get(
        "chunks",
        []
    )


    if isinstance(
        chunks,
        list
    ):

        chunk_counts[
            len(chunks)
        ] += 1


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if index % 50 == 0:

        print(
            f"Processed "
            f"{index}/{len(emg_files)}..."
        )


# ============================================================
# DATASET SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)


print(
    "\nTotal EMG files:",
    total_recordings
)

print(
    "Valid labeled recordings:",
    valid_labeled_recordings
)

print(
    "Reference recordings:",
    reference_recordings
)

print(
    "Broken recordings:",
    broken_recordings
)


# ============================================================
# VOCABULARY
# ============================================================

print("\n" + "=" * 70)
print("VOCABULARY")
print("=" * 70)


print(
    "\nUnique words:",
    len(word_recordings)
)


# ============================================================
# TOP WORDS BY RECORDINGS
# ============================================================

print("\n" + "=" * 70)
print("TOP WORDS BY NUMBER OF RECORDINGS")
print("=" * 70)


print(
    "\n"
    "Rank  Word                 "
    "Recordings   Occurrences"
)

print("-" * 55)


top_words = sorted(
    word_recordings.items(),
    key=lambda x: x[1],
    reverse=True
)


for rank, (
    word,
    recording_count
) in enumerate(
    top_words[:TOP_N],
    start=1
):

    occurrence_count = (
        word_occurrences[word]
    )

    print(
        f"{rank:4d}  "
        f"{word:<20s} "
        f"{recording_count:10d} "
        f"{occurrence_count:11d}"
    )


# ============================================================
# WORDS WITH ENOUGH EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print(
    f"WORDS WITH AT LEAST "
    f"{MIN_EXAMPLES} RECORDINGS"
)
print("=" * 70)


usable_words = [
    (
        word,
        word_recordings[word],
        word_occurrences[word]
    )
    for word in word_recordings
    if word_recordings[word] >= MIN_EXAMPLES
]


usable_words.sort(
    key=lambda x: x[1],
    reverse=True
)


print(
    "\nNumber of usable words:",
    len(usable_words)
)


print(
    "\n"
    "Rank  Word                 "
    "Recordings   Occurrences"
)

print("-" * 55)


for rank, (
    word,
    recordings,
    occurrences
) in enumerate(
    usable_words,
    start=1
):

    print(
        f"{rank:4d}  "
        f"{word:<20s} "
        f"{recordings:10d} "
        f"{occurrences:11d}"
    )


# ============================================================
# DIFFERENT MINIMUM THRESHOLDS
# ============================================================

print("\n" + "=" * 70)
print("DATA AVAILABILITY BY MINIMUM EXAMPLE COUNT")
print("=" * 70)


thresholds = [
    2,
    3,
    5,
    10,
    15,
    20,
    25,
    30,
    40,
    50
]


print(
    "\nMinimum examples    Number of possible words"
)

print("-" * 50)


for threshold in thresholds:

    count = sum(
        1
        for word in word_recordings
        if word_recordings[word] >= threshold
    )

    print(
        f"{threshold:17d}    {count:10d}"
    )


# ============================================================
# POSSIBLE 5-WORD VOCABULARY
# ============================================================

print("\n" + "=" * 70)
print("POSSIBLE 5-WORD VOCABULARY")
print("=" * 70)


for word, recordings, occurrences in usable_words[:5]:

    print(
        f"{word:<15s} "
        f"{recordings:4d} recordings"
    )


# ============================================================
# POSSIBLE 10-WORD VOCABULARY
# ============================================================

print("\n" + "=" * 70)
print("POSSIBLE 10-WORD VOCABULARY")
print("=" * 70)


for word, recordings, occurrences in usable_words[:10]:

    print(
        f"{word:<15s} "
        f"{recordings:4d} recordings"
    )


# ============================================================
# POSSIBLE 20-WORD VOCABULARY
# ============================================================

print("\n" + "=" * 70)
print("POSSIBLE 20-WORD VOCABULARY")
print("=" * 70)


for word, recordings, occurrences in usable_words[:20]:

    print(
        f"{word:<15s} "
        f"{recordings:4d} recordings"
    )


# ============================================================
# SENTENCE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("SENTENCE STATISTICS")
print("=" * 70)


if sentence_word_counts:

    print(
        "\nMinimum words per sentence:",
        min(sentence_word_counts)
    )

    print(
        "Maximum words per sentence:",
        max(sentence_word_counts)
    )

    print(
        "Mean words per sentence:",
        f"{np.mean(sentence_word_counts):.2f}"
    )

    print(
        "Median words per sentence:",
        f"{np.median(sentence_word_counts):.2f}"
    )


# ============================================================
# CHUNK STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("CHUNK STATISTICS")
print("=" * 70)


if chunk_counts:

    chunk_values = []

    for count, number_recordings in chunk_counts.items():

        chunk_values.extend(
            [count] * number_recordings
        )


    print(
        "\nMinimum chunks:",
        min(chunk_values)
    )

    print(
        "Maximum chunks:",
        max(chunk_values)
    )

    print(
        "Mean chunks:",
        f"{np.mean(chunk_values):.2f}"
    )

    print(
        "Median chunks:",
        f"{np.median(chunk_values):.2f}"
    )


# ============================================================
# CHECK CHUNKS VS WORDS
#
# This is VERY important.
#
# We want to know whether:
#
# number of chunks ≈ number of words
#
# If yes, the chunks may correspond to word-level
# EMG segments.
# ============================================================

print("\n" + "=" * 70)
print("CHUNK / WORD ALIGNMENT CHECK")
print("=" * 70)


comparison_count = 0

difference_counter = Counter()


for emg_filename in emg_files:

    info_filename = emg_filename.replace(
        "_emg.npy",
        "_info.json"
    )

    info_path = os.path.join(
        DATASET_DIR,
        info_filename
    )


    if not os.path.isfile(info_path):

        continue


    try:

        with open(
            info_path,
            "r",
            encoding="utf-8"
        ) as f:

            metadata = json.load(f)

    except Exception:

        continue


    text = metadata.get(
        "text",
        ""
    )


    if not text:

        continue


    words = normalize_text(
        text
    )


    chunks = metadata.get(
        "chunks",
        []
    )


    if not isinstance(
        chunks,
        list
    ):

        continue


    comparison_count += 1


    difference = (
        len(chunks) -
        len(words)
    )


    difference_counter[
        difference
    ] += 1


print(
    "\nRecordings compared:",
    comparison_count
)


print(
    "\nChunk count - word count:"
)


for difference, count in sorted(
    difference_counter.items()
):

    print(
        f"{difference:+5d}: "
        f"{count:4d} recordings"
    )


# ============================================================
# SAVE WORD STATISTICS
# ============================================================

OUTPUT_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/"
    "berkeley_word_statistics.csv"
)


try:

    import csv


    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "word",
            "recordings",
            "occurrences"
        ])


        for word, recordings, occurrences in sorted(
            word_recordings.items(),
            key=lambda x: x[1],
            reverse=True
        ):

            writer.writerow([
                word,
                recordings,
                occurrences
            ])


    print("\n" + "=" * 70)
    print("WORD STATISTICS SAVED")
    print("=" * 70)

    print(
        "\nFile:",
        OUTPUT_PATH
    )

except Exception as e:

    print(
        "\nCould not save CSV:",
        e
    )


# ============================================================
# FINAL RECOMMENDATION
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)


print(
    "\nIMPORTANT:"
)

print(
    "These counts are based on words appearing in the "
    "recording TEXT metadata."
)

print(
    "\nWe have NOT yet assumed that each chunk corresponds "
    "to exactly one word."
)

print(
    "\nThe next step should be to verify the chunk-to-word "
    "alignment before extracting individual word-level EMG."
)

print(
    "\nDO NOT TRAIN THE MLP YET."
)

print(
    "\nUse the results above to decide whether "
    "5, 10, or 20 words are realistic."
)

print("\n" + "=" * 70)