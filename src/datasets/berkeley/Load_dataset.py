import os
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

CORPUS_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/"
    "EMG-UKA-Trial-Corpus"
)


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


# ============================================================
# READ ADC FILE
# ============================================================

def read_adc(file_path):

    # Read raw 16-bit ADC values
    data = np.fromfile(
        file_path,
        dtype=np.int16
    )

    if data.size == 0:
        raise ValueError(
            "ADC file contains no data"
        )

    # Each sample contains 7 values
    if data.size % 7 != 0:
        raise ValueError(
            f"Data size {data.size} "
            "is not divisible by 7"
        )

    # samples × 7
    data = data.reshape(-1, 7)

    # Keep the 6 EMG channels
    data = data[:, :6]

    # channels × samples
    emg = data.T

    if emg.shape[0] != 6:
        raise ValueError(
            f"Expected 6 EMG channels, "
            f"got {emg.shape[0]}"
        )

    return emg


# ============================================================
# READ WORD ALIGNMENT
# ============================================================

def read_word_alignment(word_file):

    words = []

    with open(word_file, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 3:
                continue

            start = int(parts[0])
            end = int(parts[1])
            word = parts[2].upper()

            words.append(
                (start, end, word)
            )

    return words


# ============================================================
# FIND TARGET WORDS
# ============================================================

def find_target_words(words):

    targets = []

    for start, end, word in words:

        if word in TARGET_WORDS:

            targets.append(
                (start, end, word)
            )

    return targets


# ============================================================
# FIND ALL MATCHING RECORDINGS
# ============================================================

def find_recordings():

    recordings = []

    alignment_root = os.path.join(
        CORPUS_PATH,
        "Alignments"
    )

    for root, dirs, files in os.walk(
        alignment_root
    ):

        for filename in files:

            if not (
                filename.startswith("words_")
                and filename.endswith(".txt")
            ):
                continue

            word_file = os.path.join(
                root,
                filename
            )

            # Read alignment
            words = read_word_alignment(
                word_file
            )

            # Find our target words
            targets = find_target_words(
                words
            )

            if not targets:
                continue

            # ------------------------------------------------
            # Extract subject/session
            # ------------------------------------------------

            session = os.path.basename(root)

            subject = os.path.basename(
                os.path.dirname(root)
            )

            # ------------------------------------------------
            # Extract recording identifier
            # ------------------------------------------------

            identifier = (
                filename
                .replace("words_", "")
                .replace(".txt", "")
            )

            # Example:
            #
            # words_008_007_0031.txt
            #
            # identifier:
            #
            # 008_007_0031

            # ------------------------------------------------
            # Construct EMG path
            # ------------------------------------------------

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

            # Make sure EMG exists
            if not os.path.isfile(emg_file):
                continue

            recordings.append({
                "emg_file": emg_file,
                "word_file": word_file,
                "subject": subject,
                "session": session,
                "recording": identifier,
                "targets": targets
            })

    return recordings


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    recordings = find_recordings()

    dataset = []

    print(
        "Matching recordings found:",
        len(recordings)
    )

    print(
        "\nLoading EMG recordings..."
    )

    for i, recording in enumerate(
        recordings
    ):

        try:

            emg = read_adc(
                recording["emg_file"]
            )

            item = {
                "emg": emg,
                "emg_file": recording[
                    "emg_file"
                ],
                "word_file": recording[
                    "word_file"
                ],
                "subject": recording[
                    "subject"
                ],
                "session": recording[
                    "session"
                ],
                "recording": recording[
                    "recording"
                ],
                "targets": recording[
                    "targets"
                ]
            }

            dataset.append(item)

        except Exception as e:

            print(
                "\nFailed:",
                recording["emg_file"]
            )

            print(
                "Error:",
                e
            )

        if (i + 1) % 100 == 0:

            print(
                f"Loaded "
                f"{i + 1}/{len(recordings)}"
            )

    print(
        "\n=========================================="
    )

    print(
        "DATASET LOADED"
    )

    print(
        "=========================================="
    )

    print(
        "Total recordings:",
        len(dataset)
    )

    return dataset


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    dataset = load_dataset()

    if len(dataset) > 0:

        first = dataset[0]

        print(
            "\nFirst recording:"
        )

        print(
            "EMG shape:",
            first["emg"].shape
        )

        print(
            "Subject:",
            first["subject"]
        )

        print(
            "Session:",
            first["session"]
        )

        print(
            "Recording:",
            first["recording"]
        )

        print(
            "Target words:"
        )

        for target in first["targets"]:

            print(
                target
            )