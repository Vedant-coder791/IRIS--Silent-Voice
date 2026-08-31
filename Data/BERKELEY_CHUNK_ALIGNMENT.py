import os
import json
import re
import numpy as np
import csv
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    "/Users/vedantdwivedi/Desktop/IRIS/emg_data/closed_vocab/silent/5-19_silent"
)

# The Berkeley dataset appears to use 600 Hz EMG in our project.
FS = 600

# Number of recordings to print in detail
EXAMPLE_COUNT = 15

# Number of words/chunks to display in examples
MAX_DISPLAY_ITEMS = 20


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("BERKELEY CHUNK / WORD ALIGNMENT INSPECTOR")
print("=" * 70)

print()
print("Dataset:")
print(DATASET_DIR)
print()


# ============================================================
# CHECK DIRECTORY
# ============================================================

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"\nDataset directory does not exist:\n{DATASET_DIR}\n"
    )


# ============================================================
# FIND FILES
# ============================================================

print("=" * 70)
print("SEARCHING FOR FILES")
print("=" * 70)

emg_files = sorted(DATASET_DIR.glob("*_emg.npy"))
info_files = sorted(DATASET_DIR.glob("*_info.json"))

print(f"EMG files found:  {len(emg_files)}")
print(f"INFO files found: {len(info_files)}")


# ============================================================
# MATCH FILES
# ============================================================

info_map = {}

for info_path in info_files:
    stem = info_path.name.replace("_info.json", "")
    info_map[stem] = info_path


matched = []

for emg_path in emg_files:
    stem = emg_path.name.replace("_emg.npy", "")

    if stem in info_map:
        matched.append(
            (
                emg_path,
                info_map[stem]
            )
        )


print(f"Matched EMG/INFO pairs: {len(matched)}")

print()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def tokenize_text(text):
    """
    Convert text into simple word tokens.

    We keep numbers because the closed-vocabulary dataset
    contains things such as:

        09:48 AM
        August 06 1981
    """

    if not isinstance(text, str):
        return []

    return re.findall(r"[A-Za-z]+|\d+", text.upper())


def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"WARNING: Could not read {path}")
        print("Reason:", e)
        return None


def recursive_keys(obj, prefix=""):
    """
    Find all JSON keys recursively.
    """

    found = []

    if isinstance(obj, dict):

        for key, value in obj.items():

            full_key = f"{prefix}.{key}" if prefix else key

            found.append(full_key)

            found.extend(
                recursive_keys(value, full_key)
            )

    elif isinstance(obj, list):

        for i, value in enumerate(obj):

            full_key = f"{prefix}[{i}]"

            found.extend(
                recursive_keys(value, full_key)
            )

    return found


def flatten_chunk_lengths(chunks):

    lengths = []

    for chunk in chunks:

        if isinstance(chunk, list) and len(chunk) >= 2:

            try:
                middle = float(chunk[1])
                lengths.append(middle)

            except (ValueError, TypeError):
                pass

    return lengths


def chunk_to_seconds(samples):

    return samples / FS


# ============================================================
# ANALYSIS STORAGE
# ============================================================

records = []

word_recording_counts = Counter()
word_occurrence_counts = Counter()

chunk_count_distribution = Counter()

chunk_length_all = []

sentence_word_count_distribution = Counter()

json_key_counter = Counter()

potential_alignment_files = []

# Store examples
examples = []


# ============================================================
# PROCESS RECORDINGS
# ============================================================

print("=" * 70)
print("ANALYZING RECORDINGS")
print("=" * 70)

for index, (emg_path, info_path) in enumerate(matched, start=1):

    info = load_json(info_path)

    if info is None:
        continue

    # --------------------------------------------------------
    # Load EMG
    # --------------------------------------------------------

    try:
        emg = np.load(emg_path)

    except Exception as e:
        print(f"Could not load {emg_path.name}: {e}")
        continue

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    text = info.get("text", "")

    chunks = info.get("chunks", [])

    words = tokenize_text(text)

    sentence_word_count = len(words)

    chunk_count = len(chunks)

    chunk_lengths = flatten_chunk_lengths(chunks)

    # --------------------------------------------------------
    # EMG statistics
    # --------------------------------------------------------

    if isinstance(emg, np.ndarray):

        emg_length = emg.shape[0]

        if emg.ndim >= 2:
            channels = emg.shape[1]
        else:
            channels = 1

    else:
        emg_length = 0
        channels = 0

    # --------------------------------------------------------
    # Chunk statistics
    # --------------------------------------------------------

    if chunk_lengths:

        chunk_min = min(chunk_lengths)
        chunk_max = max(chunk_lengths)
        chunk_mean = float(np.mean(chunk_lengths))
        chunk_median = float(np.median(chunk_lengths))

    else:

        chunk_min = 0
        chunk_max = 0
        chunk_mean = 0
        chunk_median = 0

    total_middle_samples = sum(chunk_lengths)

    # --------------------------------------------------------
    # Word statistics
    # --------------------------------------------------------

    for word in set(words):
        word_recording_counts[word] += 1

    for word in words:
        word_occurrence_counts[word] += 1

    # --------------------------------------------------------
    # Distributions
    # --------------------------------------------------------

    chunk_count_distribution[chunk_count] += 1

    sentence_word_count_distribution[sentence_word_count] += 1

    chunk_length_all.extend(chunk_lengths)

    # --------------------------------------------------------
    # JSON key inspection
    # --------------------------------------------------------

    for key in recursive_keys(info):

        root_key = key.split(".")[0]

        json_key_counter[root_key] += 1

    # --------------------------------------------------------
    # Store record
    # --------------------------------------------------------

    record = {
        "file": emg_path.name,
        "info_file": info_path.name,
        "text": text,
        "word_count": sentence_word_count,
        "words": " ".join(words),
        "emg_samples": emg_length,
        "channels": channels,
        "chunk_count": chunk_count,
        "chunk_min": chunk_min,
        "chunk_max": chunk_max,
        "chunk_mean": chunk_mean,
        "chunk_median": chunk_median,
        "total_chunk_middle_samples": total_middle_samples,
        "chunk_to_emg_ratio": (
            total_middle_samples / emg_length
            if emg_length > 0 else 0
        )
    }

    records.append(record)

    # --------------------------------------------------------
    # Save examples
    # --------------------------------------------------------

    if len(examples) < EXAMPLE_COUNT:

        examples.append(
            {
                "emg_path": emg_path,
                "info_path": info_path,
                "info": info,
                "emg": emg,
                "words": words,
                "chunk_lengths": chunk_lengths
            }
        )

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if index % 50 == 0:
        print(f"Processed {index}/{len(matched)}...")


# ============================================================
# BASIC SUMMARY
# ============================================================

print()
print("=" * 70)
print("BASIC SUMMARY")
print("=" * 70)

print(f"Matched recordings analyzed: {len(records)}")

if records:

    total_channels = Counter(
        r["channels"] for r in records
    )

    print()
    print("--- CHANNELS ---")

    for channels, count in sorted(total_channels.items()):
        print(
            f"{channels} channels -> {count} recordings"
        )


# ============================================================
# WORD / CHUNK COMPARISON
# ============================================================

print()
print("=" * 70)
print("WORD COUNT VS CHUNK COUNT")
print("=" * 70)

print()
print(
    "IMPORTANT:"
)
print(
    "This section ONLY compares counts."
)
print(
    "It does NOT assume that chunks are words."
)
print()

comparison = Counter()

for r in records:

    difference = r["chunk_count"] - r["word_count"]

    comparison[difference] += 1


print(
    f"{'Chunk count - Word count':>25}"
    f"{'Recordings':>15}"
)

print("-" * 42)

for difference, count in sorted(comparison.items()):

    print(
        f"{difference:+25d}"
        f"{count:15d}"
    )


# ============================================================
# RATIO ANALYSIS
# ============================================================

print()
print("=" * 70)
print("CHUNK / WORD RATIO ANALYSIS")
print("=" * 70)

ratios = []

for r in records:

    if r["word_count"] > 0:

        ratio = (
            r["chunk_count"] /
            r["word_count"]
        )

        ratios.append(ratio)


if ratios:

    print(
        f"Minimum chunks/word ratio: "
        f"{min(ratios):.2f}"
    )

    print(
        f"Maximum chunks/word ratio: "
        f"{max(ratios):.2f}"
    )

    print(
        f"Mean chunks/word ratio: "
        f"{np.mean(ratios):.2f}"
    )

    print(
        f"Median chunks/word ratio: "
        f"{np.median(ratios):.2f}"
    )


# ============================================================
# CHUNK LENGTH STATISTICS
# ============================================================

print()
print("=" * 70)
print("CHUNK LENGTH ANALYSIS")
print("=" * 70)

if chunk_length_all:

    print(
        f"Minimum chunk middle value: "
        f"{min(chunk_length_all):.2f}"
    )

    print(
        f"Maximum chunk middle value: "
        f"{max(chunk_length_all):.2f}"
    )

    print(
        f"Mean chunk middle value: "
        f"{np.mean(chunk_length_all):.2f}"
    )

    print(
        f"Median chunk middle value: "
        f"{np.median(chunk_length_all):.2f}"
    )

    print()

    print(
        "Approximate duration if middle value is samples:"
    )

    print(
        f"Minimum: "
        f"{min(chunk_length_all) / FS:.3f} seconds"
    )

    print(
        f"Maximum: "
        f"{max(chunk_length_all) / FS:.3f} seconds"
    )

    print(
        f"Median: "
        f"{np.median(chunk_length_all) / FS:.3f} seconds"
    )


# ============================================================
# CHUNK COUNT DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("CHUNK COUNT DISTRIBUTION")
print("=" * 70)

for count, recordings in sorted(
    chunk_count_distribution.items()
):

    print(
        f"{count:4d} chunks -> "
        f"{recordings:4d} recordings"
    )


# ============================================================
# SENTENCE WORD COUNT
# ============================================================

print()
print("=" * 70)
print("SENTENCE WORD COUNT DISTRIBUTION")
print("=" * 70)

for word_count, recordings in sorted(
    sentence_word_count_distribution.items()
):

    print(
        f"{word_count:3d} words -> "
        f"{recordings:4d} recordings"
    )


# ============================================================
# WORD FREQUENCY
# ============================================================

print()
print("=" * 70)
print("WORD FREQUENCY")
print("=" * 70)

print()

print(
    f"{'Rank':>4} "
    f"{'Word':<20} "
    f"{'Recordings':>12} "
    f"{'Occurrences':>12}"
)

print("-" * 52)

sorted_words = sorted(
    word_recording_counts.items(),
    key=lambda x: (-x[1], x[0])
)

for rank, (word, recording_count) in enumerate(
    sorted_words[:100],
    start=1
):

    print(
        f"{rank:4d} "
        f"{word:<20} "
        f"{recording_count:12d} "
        f"{word_occurrence_counts[word]:12d}"
    )


# ============================================================
# WORD AVAILABILITY
# ============================================================

print()
print("=" * 70)
print("WORD AVAILABILITY")
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

print()
print(
    f"{'Minimum recordings':>20}"
    f"{'Possible words':>20}"
)

print("-" * 40)

for threshold in thresholds:

    possible = sum(
        1
        for count in word_recording_counts.values()
        if count >= threshold
    )

    print(
        f"{threshold:20d}"
        f"{possible:20d}"
    )


# ============================================================
# REPRESENTATIVE RECORDINGS
# ============================================================

print()
print("=" * 70)
print("REPRESENTATIVE RECORDINGS")
print("=" * 70)

for i, example in enumerate(examples, start=1):

    info = example["info"]
    words = example["words"]
    chunks = info.get("chunks", [])

    emg = example["emg"]

    print()
    print("-" * 70)
    print(f"EXAMPLE {i}")

    print()
    print("EMG file:")
    print(example["emg_path"])

    print()
    print("INFO file:")
    print(example["info_path"])

    print()
    print("EMG shape:")
    print(emg.shape)

    print()
    print("TEXT:")
    print(repr(info.get("text", "")))

    print()
    print("WORDS:")
    print(words)

    print()
    print("NUMBER OF WORDS:")
    print(len(words))

    print()
    print("NUMBER OF CHUNKS:")
    print(len(chunks))

    print()
    print("FIRST CHUNKS:")

    for j, chunk in enumerate(
        chunks[:MAX_DISPLAY_ITEMS]
    ):

        print(
            f"  Chunk {j + 1:3d}: "
            f"{chunk}"
        )

    if len(chunks) > MAX_DISPLAY_ITEMS:

        print(
            f"  ... "
            f"{len(chunks) - MAX_DISPLAY_ITEMS} more chunks"
        )


# ============================================================
# DETAILED CHUNK ANALYSIS
# ============================================================

print()
print("=" * 70)
print("DETAILED CHUNK INTERPRETATION TEST")
print("=" * 70)

print()
print(
    "We now test whether the chunk structure could"
)
print(
    "plausibly represent whole words."
)
print()

if records:

    # Calculate typical number of chunks per word
    chunk_per_word = []

    for r in records:

        if r["word_count"] > 0:

            chunk_per_word.append(
                r["chunk_count"] /
                r["word_count"]
            )

    if chunk_per_word:

        median_cpw = np.median(chunk_per_word)

        mean_cpw = np.mean(chunk_per_word)

        print(
            f"Median chunks per word: "
            f"{median_cpw:.2f}"
        )

        print(
            f"Mean chunks per word: "
            f"{mean_cpw:.2f}"
        )

        print()

        if median_cpw > 3:

            print(
                "RESULT:"
            )

            print(
                "Chunks are very unlikely to represent "
                "one whole word each."
            )

            print(
                "There are substantially more chunks "
                "than words in a sentence."
            )

        elif median_cpw >= 1.5:

            print(
                "RESULT:"
            )

            print(
                "Chunks do not appear to map "
                "one-to-one with words."
            )

        else:

            print(
                "RESULT:"
            )

            print(
                "The chunk/word relationship requires "
                "additional investigation."
            )


# ============================================================
# INSPECT JSON STRUCTURE
# ============================================================

print()
print("=" * 70)
print("JSON STRUCTURE INSPECTION")
print("=" * 70)

print()

print("JSON keys found:")

for key, count in sorted(
    json_key_counter.items(),
    key=lambda x: (-x[1], x[0])
):

    print(
        f"  {key:<30} "
        f"{count:5d} files"
    )


# ============================================================
# SEARCH FOR POTENTIAL ALIGNMENT FILES
# ============================================================

print()
print("=" * 70)
print("SEARCHING FOR POSSIBLE ALIGNMENT / ANNOTATION FILES")
print("=" * 70)

dataset_root = DATASET_DIR

interesting_extensions = {
    ".json",
    ".txt",
    ".csv",
    ".tsv",
    ".lab",
    ".TextGrid",
    ".align",
    ".phn",
    ".wrd",
    ".words"
}

interesting_keywords = [
    "align",
    "alignment",
    "word",
    "words",
    "phoneme",
    "phonemes",
    "phone",
    "phones",
    "label",
    "labels",
    "transcript",
    "transcription",
    "segment",
    "segments",
    "timing",
    "annotation",
    "annotations"
]

all_files = []

for path in dataset_root.rglob("*"):

    if path.is_file():

        all_files.append(path)

        name_lower = path.name.lower()

        extension_match = (
            path.suffix in interesting_extensions
        )

        keyword_match = any(
            keyword in name_lower
            for keyword in interesting_keywords
        )

        if extension_match or keyword_match:

            potential_alignment_files.append(path)


print()
print(
    f"Total files searched: "
    f"{len(all_files)}"
)

print(
    f"Potential annotation/alignment files: "
    f"{len(potential_alignment_files)}"
)

print()

for path in potential_alignment_files[:100]:

    print(
        " ",
        path.relative_to(dataset_root)
    )

if len(potential_alignment_files) > 100:

    print(
        f"\n... and "
        f"{len(potential_alignment_files) - 100}"
        f" more files."
    )


# ============================================================
# CHECK WHETHER CHUNK BOUNDARIES FIT EMG LENGTH
# ============================================================

print()
print("=" * 70)
print("CHUNK / EMG LENGTH CONSISTENCY")
print("=" * 70)

ratios_emg = []

for r in records:

    if r["emg_samples"] > 0:

        ratios_emg.append(
            r["total_chunk_middle_samples"] /
            r["emg_samples"]
        )

if ratios_emg:

    print(
        f"Minimum chunk-sum / EMG ratio: "
        f"{min(ratios_emg):.4f}"
    )

    print(
        f"Maximum chunk-sum / EMG ratio: "
        f"{max(ratios_emg):.4f}"
    )

    print(
        f"Mean chunk-sum / EMG ratio: "
        f"{np.mean(ratios_emg):.4f}"
    )

    print(
        f"Median chunk-sum / EMG ratio: "
        f"{np.median(ratios_emg):.4f}"
    )


# ============================================================
# SAVE RECORDING CSV
# ============================================================

print()
print("=" * 70)
print("SAVING ALIGNMENT DIAGNOSTIC CSV")
print("=" * 70)

output_csv = (
    DATASET_DIR.parent /
    "closed_vocab_silent_alignment_diagnostic.csv"
)

fieldnames = [
    "file",
    "info_file",
    "text",
    "word_count",
    "words",
    "emg_samples",
    "channels",
    "chunk_count",
    "chunk_min",
    "chunk_max",
    "chunk_mean",
    "chunk_median",
    "total_chunk_middle_samples",
    "chunk_to_emg_ratio"
]

try:

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for record in records:

            writer.writerow(record)

    print()
    print("Saved:")
    print(output_csv)

except Exception as e:

    print()
    print("Could not save CSV:")
    print(e)


# ============================================================
# SAVE WORD STATISTICS
# ============================================================

word_csv = (
    DATASET_DIR.parent /
    "closed_vocab_silent_word_statistics.csv"
)

try:

    with open(
        word_csv,
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

        for word, count in sorted_words:

            writer.writerow([
                word,
                count,
                word_occurrence_counts[word]
            ])

    print()
    print("Saved:")
    print(word_csv)

except Exception as e:

    print()
    print("Could not save word CSV:")
    print(e)


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print()
print("=" * 70)
print("FINAL DIAGNOSTIC")
print("=" * 70)

print()

print("What we know:")

print(
    f"1. {len(records)} usable labeled recordings were analyzed."
)

print(
    f"2. {len(word_recording_counts)} unique words were found."
)

if records:

    print(
        f"3. Sentences contain approximately "
        f"{np.median(list(sentence_word_count_distribution.keys())):.0f} "
        f"words in the current closed-vocabulary dataset."
    )

    print(
        f"4. Typical recordings contain approximately "
        f"{np.median(list(chunk_count_distribution.keys())):.0f} "
        f"chunks."
    )

print()

print("Therefore:")

print(
    "Chunks should NOT currently be treated as words."
)

print()

print(
    "The next task is to identify exactly what the three"
)

print(
    "values inside each chunk represent and whether there"
)

print(
    "is a separate word/phoneme alignment source."
)

print()

print(
    "DO NOT TRAIN THE MLP YET."
)

print()
print("=" * 70)
print("ALIGNMENT INSPECTION COMPLETE")
print("=" * 70)