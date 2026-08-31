import numpy as np


def read_adc(file_path):

    # Read raw 16-bit ADC data
    data = np.fromfile(file_path, dtype=np.int16)

    print("Total raw values:", len(data))

    # Each sample contains 7 values
    if len(data) % 7 != 0:
        raise ValueError(
            f"Raw data length {len(data)} is not divisible by 7"
        )

    # Reshape into:
    # samples × 7 channels
    data = data.reshape(-1, 7)

    print("Raw shape:", data.shape)

    # Keep the six EMG channels
    # Ignore the 7th constant/reference/status value
    emg = data[:, :6]

    # Transpose to:
    # channels × samples
    emg = emg.T

    print("EMG shape:", emg.shape)

    return emg

file = "/Users/vedantdwivedi/Desktop/IRIS/EMG-UKA-Trial-Corpus/emg/008/007/e07_008_007_0031.adc"

emg = read_adc(file)

print(emg.shape)


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


print(
    "Word alignment files found:",
    len(word_files)
)


# ==========================================
# FIND MATCHING EMG FILES
# ==========================================

emg_files = []

for word_file in word_files:

    contains_target = False

    # Read word alignment
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
    # CONSTRUCT EMG PATH
    # ======================================

    filename = os.path.basename(word_file)

    # Example:
    # words_008_007_0031.txt
    #
    # becomes:
    # 008_007_0031

    identifier = (
        filename
        .replace("words_", "")
        .replace(".txt", "")
    )


    alignment_folder = os.path.dirname(
        word_file
    )

    session = os.path.basename(
        alignment_folder
    )

    subject = os.path.basename(
        os.path.dirname(alignment_folder)
    )


    # EMG folder

    emg_folder = os.path.join(
        CORPUS_PATH,
        "emg",
        subject,
        session
    )


    # EMG filename

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

        emg_files.append(
            emg_file
        )


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

        # ==================================
        # READ RAW ADC DATA
        # ==================================

        # IMPORTANT:
        # EMG-UKA stores the ADC values
        # as 16-bit integers.

        data = np.fromfile(
            emg_file,
            dtype=np.int16
        )


        # ==================================
        # CHECK EMPTY FILE
        # ==================================

        if data.size == 0:

            raise ValueError(
                "File contains no data"
            )


        # ==================================
        # CHECK 7 VALUES PER SAMPLE
        # ==================================

        if data.size % 7 != 0:

            raise ValueError(
                f"Data size {data.size} "
                "is not divisible by 7"
            )


        # ==================================
        # RESHAPE RAW DATA
        # ==================================

        # Raw structure:
        #
        # CH1 CH2 CH3 CH4 CH5 CH6 EXTRA
        # CH1 CH2 CH3 CH4 CH5 CH6 EXTRA
        # CH1 CH2 CH3 CH4 CH5 CH6 EXTRA
        #
        # Therefore:
        #
        # samples × 7

        samples = data.size // 7

        data = data.reshape(
            samples,
            7
        )


        # ==================================
        # KEEP SIX EMG CHANNELS
        # ==================================

        # Remove the 7th value.
        #
        # The corpus contains six usable
        # EMG channels.

        data = data[:, :6]


        # ==================================
        # TRANSPOSE
        # ==================================

        # Convert:
        #
        # samples × channels
        #
        # into:
        #
        # channels × samples

        data = data.T


        # ==================================
        # GET SHAPE
        # ==================================

        channels = data.shape[0]

        samples = data.shape[1]


        # ==================================
        # CHECK CHANNEL COUNT
        # ==================================

        if channels != 6:

            raise ValueError(
                f"Expected 6 channels, "
                f"got {channels}"
            )


        # ==================================
        # RECORD STATISTICS
        # ==================================

        channel_counts[channels] = (
            channel_counts.get(
                channels,
                0
            ) + 1
        )


        sample_counts[samples] = (
            sample_counts.get(
                samples,
                0
            ) + 1
        )


        valid_files += 1


    except Exception as e:

        failed_files.append(
            (
                emg_file,
                str(e)
            )
        )


    # ==================================
    # PROGRESS
    # ==================================

    if (i + 1) % 100 == 0:

        print(
            f"Checked "
            f"{i + 1}/{len(emg_files)}"
        )


# ==========================================
# RESULTS
# ==========================================

print(
    "\n=========================================="
)

print(
    "VALIDATION RESULTS"
)

print(
    "=========================================="
)


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

print(
    "\nChannel counts:"
)

for channels, count in sorted(
    channel_counts.items()
):

    print(
        f"{channels} channels: "
        f"{count} files"
    )


# ==========================================
# SAMPLE COUNTS
# ==========================================

print(
    "\nMost common sample counts:"
)

for samples, count in sorted(
    sample_counts.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]:

    print(
        f"{samples} samples: "
        f"{count} files"
    )


# ==========================================
# FAILED FILES
# ==========================================

if failed_files:

    print(
        "\nFAILED FILES"
    )

    print(
        "--------------------------------"
    )

    for filepath, error in failed_files[:20]:

        print(filepath)

        print(
            "Error:",
            error
        )

        print()


else:

    print(
        "\n🎉 No EMG files failed validation!"
    )