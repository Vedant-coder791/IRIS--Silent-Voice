import numpy as np

from .Load_dataset import load_dataset


# ============================================================
# SETTINGS
# ============================================================

FS = 600
FRAME_DURATION = 0.010


# ============================================================
# EXTRACT WORD SEGMENTS
# ============================================================

def extract_word_segments(dataset):

    word_segments = []

    print("\n==========================================")
    print("EXTRACTING TARGET WORD SEGMENTS")
    print("==========================================")

    for recording in dataset:

        emg = recording["emg"]
        targets = recording["targets"]

        for start_frame, end_frame, word in targets:

            # ------------------------------------------------
            # Convert alignment frames → seconds
            # ------------------------------------------------

            start_time = (
                start_frame * FRAME_DURATION
            )

            end_time = (
                (end_frame + 1) * FRAME_DURATION
            )

            # ------------------------------------------------
            # Convert seconds → EMG samples
            # ------------------------------------------------

            start_sample = int(
                round(start_time * FS)
            )

            end_sample = int(
                round(end_time * FS)
            )

            # ------------------------------------------------
            # Keep indices valid
            # ------------------------------------------------

            start_sample = max(
                0,
                start_sample
            )

            end_sample = min(
                emg.shape[1],
                end_sample
            )

            # ------------------------------------------------
            # Skip invalid segments
            # ------------------------------------------------

            if end_sample <= start_sample:
                continue

            # ------------------------------------------------
            # Extract EMG
            # ------------------------------------------------

            word_emg = emg[
                :,
                start_sample:end_sample
            ]

            # ------------------------------------------------
            # Store segment
            # ------------------------------------------------

            word_segments.append({

                "emg": word_emg,

                "label": word,

                "start_frame": start_frame,
                "end_frame": end_frame,

                "start_sample": start_sample,
                "end_sample": end_sample,

                "subject": recording["subject"],
                "session": recording["session"],
                "recording": recording["recording"],
                "emg_file": recording["emg_file"]
            })

    return word_segments


# ============================================================
# SUMMARY
# ============================================================

def print_summary(word_segments):

    print("\n==========================================")
    print("WORD EXTRACTION RESULTS")
    print("==========================================")

    print(
        "Total word segments:",
        len(word_segments)
    )

    word_counts = {}

    for segment in word_segments:

        word = segment["label"]

        word_counts[word] = (
            word_counts.get(word, 0) + 1
        )

    print("\nWord counts:")
    print("------------------------------------------")

    for word, count in sorted(
        word_counts.items()
    ):

        print(
            f"{word:10s} : {count}"
        )


# ============================================================
# SUBJECT INFORMATION
# ============================================================

def print_subject_distribution(word_segments):

    from collections import Counter

    print("\n==========================================")
    print("SUBJECT INFORMATION")
    print("==========================================")

    subjects = sorted(
        set(
            segment["subject"]
            for segment in word_segments
        )
    )

    print(
        "Number of subjects:",
        len(subjects)
    )

    for subject in subjects:

        count = sum(
            1
            for segment in word_segments
            if segment["subject"] == subject
        )

        print(
            f"Subject {subject}: {count} word segments"
        )

    print("\n==========================================")
    print("SUBJECT / WORD DISTRIBUTION")
    print("==========================================")

    for subject in subjects:

        print("\n------------------------------------------")
        print("SUBJECT:", subject)
        print("------------------------------------------")

        subject_segments = [
            segment
            for segment in word_segments
            if segment["subject"] == subject
        ]

        counts = Counter(
            segment["label"]
            for segment in subject_segments
        )

        for word, count in sorted(
            counts.items()
        ):

            print(
                f"{word:8s}: {count:4d}"
            )

        print(
            "TOTAL:",
            len(subject_segments)
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    dataset = load_dataset()

    word_segments = extract_word_segments(
        dataset
    )

    print_summary(
        word_segments
    )

    print_subject_distribution(
        word_segments
    )

    # --------------------------------------------------------
    # Show first segment
    # --------------------------------------------------------

    if len(word_segments) > 0:

        first = word_segments[0]

        print("\n==========================================")
        print("FIRST EXTRACTED WORD")
        print("==========================================")

        print(
            "Label:",
            first["label"]
        )

        print(
            "EMG shape:",
            first["emg"].shape
        )

        print(
            "Alignment frames:",
            first["start_frame"],
            "→",
            first["end_frame"]
        )

        print(
            "EMG samples:",
            first["start_sample"],
            "→",
            first["end_sample"]
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
