import os


# ==========================================
# CORPUS PATH
# ==========================================

corpus_path = "/Users/vedantdwivedi/Desktop/IRIS/EMG-UKA-Trial-Corpus"


# ==========================================
# TARGET VOCABULARY
# ==========================================

target_words = {
    "YES",
    "NO",
    "HELP",
    "WATER",
    "STOP"
}


# ==========================================
# SEARCH ALL WORD ALIGNMENT FILES
# ==========================================

word_files_found = 0

word_counts = {
    word: 0
    for word in target_words
}

recordings = {
    word: []
    for word in target_words
}


for root, dirs, files in os.walk(corpus_path):

    for filename in files:

        if not filename.startswith("words_"):
            continue

        if not filename.endswith(".txt"):
            continue

        word_files_found += 1

        filepath = os.path.join(root, filename)

        words_in_file = set()

        with open(filepath, "r") as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 3:
                    continue

                start = int(parts[0])
                end = int(parts[1])
                word = parts[2].upper()

                words_in_file.add(word)

        # Count each target word once per recording
        for target in target_words:

            if target in words_in_file:

                word_counts[target] += 1
                recordings[target].append(filepath)


# ==========================================
# RESULTS
# ==========================================

print("Word alignment files found:", word_files_found)

print("\nTarget word counts:")
print("-------------------")

for word in sorted(target_words):

    print(
        f"{word}: {word_counts[word]} recordings"
    )


# ==========================================
# SHOW EXAMPLES
# ==========================================

print("\nExample recordings:")
print("-------------------")

for word in sorted(target_words):

    print(f"\n{word}:")

    for filepath in recordings[word][:5]:

        print(filepath)


import os
from collections import Counter


# ==========================================
# CORPUS PATH
# ==========================================

corpus_path = "/Users/vedantdwivedi/Desktop/IRIS/EMG-UKA-Trial-Corpus"


# ==========================================
# COUNT WORDS
# ==========================================

word_counts = Counter()

recording_counts = Counter()

for root, dirs, files in os.walk(corpus_path):

    for filename in files:

        if not filename.startswith("words_"):
            continue

        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(root, filename)

        words_in_recording = set()

        with open(filepath, "r") as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 3:
                    continue

                start = int(parts[0])
                end = int(parts[1])
                word = parts[2].upper()

                # Ignore silence markers
                if word == "$":
                    continue

                word_counts[word] += 1
                words_in_recording.add(word)

        # Count how many separate recordings contain each word
        for word in words_in_recording:
            recording_counts[word] += 1


# ==========================================
# RESULTS
# ==========================================

print("Most frequent words")
print("===================")

print(
    f"{'WORD':<20}"
    f"{'OCCURRENCES':<15}"
    f"{'RECORDINGS':<15}"
)

print("-" * 50)

for word, occurrences in word_counts.most_common(50):

    print(
        f"{word:<20}"
        f"{occurrences:<15}"
        f"{recording_counts[word]:<15}"
    )

import os
import numpy as np


# ==========================================
# CORPUS
# ==========================================

CORPUS_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/"
    "EMG-UKA-Trial-Corpus"
)


# ==========================================
# TARGET VOCABULARY
# ==========================================

TARGET_WORDS = [
    "THE",
    "TO",
    "OF",
    "IN",
    "ARE",
    "AND",
    "IS",
    "THEY",
    "THAT",
    "THIS",
    "HAS",
    "HE",
    "I",
    "ON",
    "WERE",
    "WE",
    "IT",
    "YOU",
    "STATE",
    "NEW"
]


# ==========================================
# FIND WORD ALIGNMENT FILES
# ==========================================

word_files = []

for root, dirs, files in os.walk(CORPUS_PATH):

    for filename in files:

        if (
            filename.startswith("words_")
            and filename.endswith(".txt")
        ):
            word_files.append(
                os.path.join(root, filename)
            )


print("Word alignment files found:", len(word_files))


# ==========================================
# FIND RECORDINGS CONTAINING TARGET WORDS
# ==========================================

recordings = []


for word_file in word_files:

    words_in_file = []

    with open(word_file, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 3:
                continue

            start = int(parts[0])
            end = int(parts[1])
            word = parts[2].upper()

            if word in TARGET_WORDS:

                words_in_file.append(
                    (start, end, word)
                )


    if len(words_in_file) > 0:

        recordings.append(
            {
                "word_file": word_file,
                "words": words_in_file
            }
        )


print(
    "Recordings containing target words:",
    len(recordings)
)


# ==========================================
# SHOW FIRST 10
# ==========================================

print("\nFirst 10 matching recordings:")

for recording in recordings[:10]:

    print("\nWord file:")
    print(recording["word_file"])

    print("Target words:")

    for item in recording["words"]:
        print(item)

# ==========================================
# TEST WORD FILE → EMG FILE MATCHING
# ==========================================

print("\nTesting EMG file matching...")
print("--------------------------------")


for recording in recordings[:10]:

    word_file = recording["word_file"]

    # Extract filename
    filename = os.path.basename(word_file)

    # Example:
    # words_008_007_0037.txt
    #
    # We want:
    # e07_008_007_0037.adc

    identifier = filename.replace(
        "words_",
        ""
    ).replace(
        ".txt",
        ""
    )

    # Get the folder containing the word file
    alignment_folder = os.path.dirname(
        word_file
    )

    # Extract 008/007 from the path
    alignment_root = os.path.dirname(
        os.path.dirname(
            alignment_folder
        )
    )

    # Construct EMG folder
    emg_folder = os.path.join(
        CORPUS_PATH,
        "emg",
        os.path.basename(
            os.path.dirname(alignment_folder)
        ),
        os.path.basename(alignment_folder)
    )

    emg_filename = (
        "e07_"
        + identifier
        + ".adc"
    )

    emg_file = os.path.join(
        emg_folder,
        emg_filename
    )

    print("\nWord file:")
    print(word_file)

    print("Expected EMG:")
    print(emg_file)

    print(
        "EXISTS:",
        os.path.isfile(emg_file)
    )

    import os
import numpy as np


# ==========================================
# CORPUS PATH
# ==========================================

CORPUS_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/"
    "EMG-UKA-Trial-Corpus"
)


# ==========================================
# FIND ALL WORD ALIGNMENT FILES
# ==========================================

word_files = []

for root, dirs, files in os.walk(CORPUS_PATH):

    for filename in files:

        if (
            filename.startswith("words_")
            and filename.endswith(".txt")
        ):
            word_files.append(
                os.path.join(root, filename)
            )


print("Word alignment files found:", len(word_files))


# ==========================================
# TARGET WORDS
# ==========================================

TARGET_WORDS = {
    "THE",
    "TO",
    "OF",
    "IN",
    "ARE",
    "AND",
    "IS",
    "THEY",
    "THAT",
    "THIS",
    "HAS",
    "HE",
    "I",
    "ON",
    "WERE",
    "WE",
    "IT",
    "YOU",
    "STATE",
    "NEW"
}


# ==========================================
# FIND MATCHING EMG FILES
# ==========================================

emg_files = []

for word_file in word_files:

    # Read the word alignment
    contains_target = False

    with open(word_file, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 3:
                continue

            word = parts[2].upper()

            if word in TARGET_WORDS:

                contains_target = True
                break

    if not contains_target:
        continue


    # ======================================
    # Construct matching EMG filename
    # ======================================

    filename = os.path.basename(word_file)

    # words_008_007_0031.txt
    #       ↓
    # 008_007_0031

    identifier = filename.replace(
        "words_",
        ""
    ).replace(
        ".txt",
        ""
    )


    # Get 008/007
    alignment_folder = os.path.dirname(
        word_file
    )

    subject = os.path.basename(
        os.path.dirname(alignment_folder)
    )

    session = os.path.basename(
        alignment_folder
    )


    # Construct EMG path
    emg_folder = os.path.join(
        CORPUS_PATH,
        "emg",
        subject,
        session
    )

    emg_filename = (
        "e07_"
        + identifier
        + ".adc"
    )

    emg_file = os.path.join(
        emg_folder,
        emg_filename
    )


    if os.path.isfile(emg_file):

        emg_files.append(emg_file)


print(
    "Matching EMG files found:",
    len(emg_files)
)


# ==========================================
# VALIDATE EMG FILES
# ==========================================

valid_files = 0
failed_files = []

channel_counts = {}
sample_counts = {}


print("\nValidating EMG files...")
print("--------------------------------")


for i, emg_file in enumerate(emg_files):

    try:

        # Read raw ADC data
        data = np.fromfile(
            emg_file,
            dtype=np.float32
        )


        # ==================================
        # CHECK DATA
        # ==================================

        if data.size == 0:

            raise ValueError(
                "File contains no data"
            )


        # ==================================
        # YOUR CORPUS FORMAT
        # ==================================

        # The EMG data is expected to contain
        # 7 channels originally, with one
        # channel being discarded.
        #
        # We therefore reshape based on 7.

        if data.size % 7 != 0:

            raise ValueError(
                f"Data size {data.size} "
                "is not divisible by 7"
            )


        samples = data.size // 7

        data = data.reshape(
            7,
            samples
        )


        # Remove the extra/non-EMG channel
        data = data[:6, :]


        channels = data.shape[0]
        samples = data.shape[1]


        # ==================================
        # RECORD STATISTICS
        # ==================================

        channel_counts[channels] = (
            channel_counts.get(channels, 0) + 1
        )

        sample_counts[samples] = (
            sample_counts.get(samples, 0) + 1
        )


        valid_files += 1


    except Exception as e:

        failed_files.append(
            (emg_file, str(e))
        )


    # Progress indicator
    if (i + 1) % 100 == 0:

        print(
            f"Checked {i + 1}/{len(emg_files)}"
        )


# ==========================================
# RESULTS
# ==========================================

print("\n==========================================")
print("VALIDATION RESULTS")
print("==========================================")

print(
    "Total EMG files:",
    len(emg_files)
)

print(
    "Valid files:",
    valid_files
)

print(
    "Failed files:",
    len(failed_files)
)


# ==========================================
# CHANNEL COUNTS
# ==========================================

print("\nChannel counts:")

for channels, count in sorted(
    channel_counts.items()
):

    print(
        f"{channels} channels: {count} files"
    )


# ==========================================
# SAMPLE COUNTS
# ==========================================

print("\nMost common sample counts:")

for samples, count in sorted(
    sample_counts.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]:

    print(
        f"{samples} samples: {count} files"
    )


# ==========================================
# FAILED FILES
# ==========================================

if failed_files:

    print("\nFAILED FILES")
    print("--------------------------------")

    for filepath, error in failed_files[:20]:

        print(filepath)
        print("Error:", error)
        print()


else:

    print(
        "\n🎉 No EMG files failed validation!"
    )

import numpy as np


file = "/Users/vedantdwivedi/Desktop/IRIS/EMG-UKA-Trial-Corpus/emg/008/007/e07_008_007_0037.adc"

with open(file, "rb") as f:
    data = np.fromfile(f, dtype=np.int16)

print("Total raw values:", len(data))
print("Remainder when divided by 7:", len(data) % 7)
print("First 30 values:")
print(data[:30])

import numpy as np

file = "/Users/vedantdwivedi/Desktop/IRIS/EMG-UKA-Trial-Corpus/emg/008/007/e07_008_007_0031.adc"

with open(file, "rb") as f:
    data = np.fromfile(f, dtype=np.int16)

print("Total raw values:", len(data))
print("Remainder when divided by 7:", len(data) % 7)
print("First 30 values:")
print(data[:30])