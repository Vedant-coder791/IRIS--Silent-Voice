import os
import json
import re
import csv
import numpy as np
from collections import Counter, defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = (
    "/Users/vedantdwivedi/Desktop/IRIS/"
    "emg_data/closed_vocab/silent/5-19_silent"
)

MIN_EXAMPLES = [2, 3, 5, 10, 15, 20, 25, 30, 40, 50]

OUTPUT_DIR = os.path.dirname(DATASET_DIR)

WORD_CSV = os.path.join(
    OUTPUT_DIR,
    "closed_vocab_silent_word_statistics.csv"
)

RECORDING_CSV = os.path.join(
    OUTPUT_DIR,
    "closed_vocab_silent_recording_statistics.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("BERKELEY CLOSED-VOCAB SILENT SPEECH ANALYSIS")
print("=" * 70)

print()
print("Dataset:")
print(DATASET_DIR)

print()


# ============================================================
# FIND EMG FILES
# ============================================================

print("=" * 70)
print("SEARCHING FOR EMG FILES")
print("=" * 70)

emg_files = []

for root, dirs, files in os.walk(DATASET_DIR):

    for filename in files:

        if filename.endswith("_emg.npy"):

            full_path = os.path.join(root, filename)

            emg_files.append(full_path)


emg_files.sort()

print()
print("EMG files found:", len(emg_files))


# ============================================================
# DATA STRUCTURES
# ============================================================

word_recording_count = Counter()
word_occurrence_count = Counter()

# word -> set of recording IDs
word_recordings = defaultdict(set)

recording_information = []

reference_recordings = 0
valid_recordings = 0
broken_recordings = 0

channel_counts = Counter()

emg_lengths = []

sentence_counter = Counter()

chunk_count_counter = Counter()

chunk_middle_lengths = []

alignment_difference_counter = Counter()


# ============================================================
# WORD NORMALIZATION
# ============================================================

def normalize_words(text):

    if not isinstance(text, str):
        return []

    # Convert to uppercase
    text = text.upper()

    # Keep letters, numbers, apostrophes and basic separators
    text = re.sub(r"[^A-Z0-9'\s]", " ", text)

    words = text.split()

    return words


# ============================================================
# ANALYZE RECORDINGS
# ============================================================

print()
print("=" * 70)
print("ANALYZING RECORDINGS")
print("=" * 70)

for index, emg_path in enumerate(emg_files, start=1):

    filename = os.path.basename(emg_path)

    # --------------------------------------------------------
    # Recording ID
    # --------------------------------------------------------

    recording_id = filename.replace("_emg.npy", "")

    info_path = emg_path.replace(
        "_emg.npy",
        "_info.json"
    )

    # --------------------------------------------------------
    # Check metadata
    # --------------------------------------------------------

    if not os.path.exists(info_path):

        reference_recordings += 1

        continue

    try:

        with open(info_path, "r") as f:
            metadata = json.load(f)

    except Exception as e:

        broken_recordings += 1

        print(
            "Could not read:",
            info_path,
            "|",
            e
        )

        continue


    # --------------------------------------------------------
    # Extract text
    # --------------------------------------------------------

    text = metadata.get("text", "")

    if not text:

        reference_recordings += 1

        continue


    words = normalize_words(text)

    if not words:

        reference_recordings += 1

        continue


    # --------------------------------------------------------
    # Load EMG
    # --------------------------------------------------------

    try:

        emg = np.load(emg_path)

    except Exception as e:

        broken_recordings += 1

        print(
            "Could not load:",
            emg_path,
            "|",
            e
        )

        continue


    # --------------------------------------------------------
    # Validate EMG shape
    # --------------------------------------------------------

    if emg.ndim != 2:

        broken_recordings += 1

        continue


    n_samples = emg.shape[0]
    n_channels = emg.shape[1]

    channel_counts[n_channels] += 1

    emg_lengths.append(n_samples)


    # --------------------------------------------------------
    # Chunks
    # --------------------------------------------------------

    chunks = metadata.get("chunks", [])

    chunk_count = len(chunks)

    chunk_count_counter[chunk_count] += 1


    # --------------------------------------------------------
    # Chunk middle values
    #
    # Example:
    # [60, 973, 60]
    #
    # middle value = 973
    # --------------------------------------------------------

    for chunk in chunks:

        if isinstance(chunk, list) and len(chunk) >= 3:

            try:

                middle = float(chunk[1])

                chunk_middle_lengths.append(middle)

            except Exception:
                pass


    # --------------------------------------------------------
    # Sentence
    # --------------------------------------------------------

    sentence_counter[text] += 1


    # --------------------------------------------------------
    # Word statistics
    # --------------------------------------------------------

    unique_words_in_recording = set(words)

    for word in words:

        word_occurrence_count[word] += 1


    for word in unique_words_in_recording:

        word_recording_count[word] += 1

        word_recordings[word].add(recording_id)


    # --------------------------------------------------------
    # Chunk / word comparison
    # --------------------------------------------------------

    alignment_difference = chunk_count - len(words)

    alignment_difference_counter[
        alignment_difference
    ] += 1


    # --------------------------------------------------------
    # Save recording information
    # --------------------------------------------------------

    recording_information.append({

        "recording_id": recording_id,

        "filename": filename,

        "text": text,

        "words": " ".join(words),

        "word_count": len(words),

        "unique_word_count": len(unique_words_in_recording),

        "chunk_count": chunk_count,

        "chunk_minus_word_count":
            alignment_difference,

        "emg_samples": n_samples,

        "channels": n_channels,

        "emg_path": emg_path

    })


    valid_recordings += 1


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if index % 50 == 0:

        print(
            f"Processed {index}/{len(emg_files)}..."
        )


# ============================================================
# BASIC SUMMARY
# ============================================================

print()
print("=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print()
print("Total EMG files:", len(emg_files))
print("Valid labeled recordings:", valid_recordings)
print("Reference recordings:", reference_recordings)
print("Broken recordings:", broken_recordings)


# ============================================================
# CHANNELS
# ============================================================

print()
print("--- CHANNELS ---")

for channels, count in sorted(channel_counts.items()):

    print(
        f"{channels} channels -> {count} recordings"
    )


# ============================================================
# EMG LENGTH
# ============================================================

print()
print("--- EMG LENGTH ---")

if emg_lengths:

    print(
        "Minimum:",
        min(emg_lengths)
    )

    print(
        "Maximum:",
        max(emg_lengths)
    )

    print(
        "Mean:",
        round(np.mean(emg_lengths), 2)
    )

    print(
        "Median:",
        round(np.median(emg_lengths), 2)
    )


# ============================================================
# SENTENCES
# ============================================================

print()
print("--- SENTENCES ---")

print(
    "Unique sentences:",
    len(sentence_counter)
)

print()
print("Most common sentences:")

for sentence, count in sentence_counter.most_common(15):

    print(
        f"{count:4d}x  {sentence}"
    )


# ============================================================
# VOCABULARY
# ============================================================

print()
print("=" * 70)
print("VOCABULARY")
print("=" * 70)

print()
print(
    "Unique words:",
    len(word_recording_count)
)


# ============================================================
# WORD TABLE
# ============================================================

sorted_words = sorted(
    word_recording_count.items(),
    key=lambda x: (-x[1], x[0])
)


print()
print("=" * 70)
print("TOP WORDS BY NUMBER OF RECORDINGS")
print("=" * 70)

print()

print(
    f"{'Rank':>4}  "
    f"{'Word':<20} "
    f"{'Recordings':>12} "
    f"{'Occurrences':>12}"
)

print("-" * 65)


for rank, (word, recordings) in enumerate(
    sorted_words,
    start=1
):

    occurrences = word_occurrence_count[word]

    print(
        f"{rank:4d}  "
        f"{word:<20} "
        f"{recordings:12d} "
        f"{occurrences:12d}"
    )

    if rank >= 100:
        break


# ============================================================
# EXAMPLE COUNT THRESHOLDS
# ============================================================

print()
print("=" * 70)
print("DATA AVAILABILITY BY MINIMUM RECORDING COUNT")
print("=" * 70)

print()
print(
    f"{'Minimum examples':>18} "
    f"{'Possible words':>18}"
)

print("-" * 40)

for minimum in MIN_EXAMPLES:

    possible_words = [

        word

        for word, count
        in word_recording_count.items()

        if count >= minimum

    ]

    print(
        f"{minimum:18d} "
        f"{len(possible_words):18d}"
    )


# ============================================================
# TOP N VOCABULARIES
# ============================================================

for vocabulary_size in [5, 10, 20]:

    print()
    print("=" * 70)
    print(
        f"POSSIBLE {vocabulary_size}-WORD VOCABULARY"
    )
    print("=" * 70)

    print()

    selected = sorted_words[:vocabulary_size]

    for rank, (word, recordings) in enumerate(
        selected,
        start=1
    ):

        occurrences = word_occurrence_count[word]

        print(
            f"{rank:2d}. "
            f"{word:<20} "
            f"{recordings:4d} recordings "
            f"({occurrences} occurrences)"
        )


# ============================================================
# WORDS WITH >= 5 RECORDINGS
# ============================================================

MIN_WORD_RECORDINGS = 5

usable_words = [

    (word, word_recording_count[word])

    for word in word_recording_count

    if word_recording_count[word] >= MIN_WORD_RECORDINGS

]

usable_words.sort(
    key=lambda x: (-x[1], x[0])
)


print()
print("=" * 70)
print(
    f"WORDS WITH AT LEAST {MIN_WORD_RECORDINGS} RECORDINGS"
)
print("=" * 70)

print()

print(
    "Number of usable words:",
    len(usable_words)
)

print()

for rank, (word, recordings) in enumerate(
    usable_words,
    start=1
):

    print(
        f"{rank:4d}  "
        f"{word:<20} "
        f"{recordings:5d} recordings "
        f"{word_occurrence_count[word]:5d} occurrences"
    )


# ============================================================
# CHUNK STATISTICS
# ============================================================

print()
print("=" * 70)
print("CHUNK STATISTICS")
print("=" * 70)

if chunk_count_counter:

    counts = []

    for count, frequency in chunk_count_counter.items():

        counts.extend(
            [count] * frequency
        )

    print()
    print(
        "Minimum chunks:",
        min(counts)
    )

    print(
        "Maximum chunks:",
        max(counts)
    )

    print(
        "Mean chunks:",
        round(np.mean(counts), 2)
    )

    print(
        "Median chunks:",
        round(np.median(counts), 2)
    )


# ============================================================
# CHUNK / WORD ALIGNMENT
# ============================================================

print()
print("=" * 70)
print("CHUNK / WORD ALIGNMENT CHECK")
print("=" * 70)

print()

print(
    "This is ONLY a diagnostic."
)

print(
    "We are NOT assuming one chunk = one word."
)

print()

print(
    "Recordings compared:",
    len(recording_information)
)

print()

for difference, count in sorted(
    alignment_difference_counter.items()
):

    sign = "+" if difference >= 0 else ""

    print(
        f"{sign}{difference:3d}: "
        f"{count:4d} recordings"
    )


# ============================================================
# CHUNK MIDDLE LENGTH
# ============================================================

print()
print("--- CHUNK MIDDLE-LENGTH STATISTICS ---")

if chunk_middle_lengths:

    print(
        "Minimum:",
        round(min(chunk_middle_lengths), 2)
    )

    print(
        "Maximum:",
        round(max(chunk_middle_lengths), 2)
    )

    print(
        "Mean:",
        round(np.mean(chunk_middle_lengths), 2)
    )

    print(
        "Median:",
        round(np.median(chunk_middle_lengths), 2)
    )


# ============================================================
# SENTENCE WORD COUNT
# ============================================================

sentence_word_counts = []

for recording in recording_information:

    sentence_word_counts.append(
        recording["word_count"]
    )


print()
print("=" * 70)
print("SENTENCE WORD STATISTICS")
print("=" * 70)

if sentence_word_counts:

    print()
    print(
        "Minimum words:",
        min(sentence_word_counts)
    )

    print(
        "Maximum words:",
        max(sentence_word_counts)
    )

    print(
        "Mean words:",
        round(np.mean(sentence_word_counts), 2)
    )

    print(
        "Median words:",
        round(np.median(sentence_word_counts), 2)
    )


# ============================================================
# SAVE WORD CSV
# ============================================================

print()
print("=" * 70)
print("SAVING WORD STATISTICS")
print("=" * 70)

try:

    with open(
        WORD_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "rank",
            "word",
            "recordings",
            "occurrences"
        ])

        for rank, (word, recordings) in enumerate(
            sorted_words,
            start=1
        ):

            writer.writerow([
                rank,
                word,
                recordings,
                word_occurrence_count[word]
            ])


    print()
    print(
        "Saved:",
        WORD_CSV
    )

except Exception as e:

    print()
    print(
        "Could not save word CSV:",
        e
    )


# ============================================================
# SAVE RECORDING CSV
# ============================================================

print()
print("=" * 70)
print("SAVING RECORDING STATISTICS")
print("=" * 70)

try:

    fieldnames = [

        "recording_id",
        "filename",
        "text",
        "words",
        "word_count",
        "unique_word_count",
        "chunk_count",
        "chunk_minus_word_count",
        "emg_samples",
        "channels",
        "emg_path"

    ]


    with open(
        RECORDING_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for recording in recording_information:

            writer.writerow(recording)


    print()
    print(
        "Saved:",
        RECORDING_CSV
    )

except Exception as e:

    print()
    print(
        "Could not save recording CSV:",
        e
    )


# ============================================================
# FINAL CONCLUSION
# ============================================================

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print()

print(
    "IMPORTANT:"
)

print(
    "1. Word counts come from the TEXT metadata."
)

print(
    "2. Recording count = number of separate EMG recordings"
)

print(
    "   containing that word."
)

print(
    "3. Occurrence count = total appearances of the word."
)

print(
    "4. Chunks are NOT assumed to correspond to words."
)

print(
    "5. Word-level EMG extraction should only happen"
)

print(
    "   after chunk/word alignment is verified."
)

print()

print(
    "DO NOT TRAIN THE MLP YET."
)

print()