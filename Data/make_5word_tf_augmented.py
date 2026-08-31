
"""
============================================================
IRIS - BERKELEY 5-WORD TF DATASET
============================================================

Purpose:
    Create a clean 5-word time-frequency dataset for CNN
    training from the Berkeley TF dataset.

Vocabulary:
    THE
    AND
    A
    OF
    I

Input:
    berkeley_tf_dataset.npz

Output:
    berkeley_5word_tf_augmented_dataset.npz

Dataset format:
    X: (samples, 8, 24, 64)
    y: (samples,)

Augmentation:
    1. Original
    2. Time-axis speed perturbation
    3. Time masking
    4. Frequency masking

IMPORTANT:
    Augmentation is applied ONLY to the training set.
    Validation and test sets remain untouched.
============================================================
"""

import numpy as np
import pandas as pd

from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path("/Users/vedantdwivedi/Desktop/IRIS")

INPUT_PATH = (
    PROJECT_ROOT
    / "Data"
    / "datasets"
    / "berkeley_tf_dataset.npz"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "Data"
    / "datasets"
    / "berkeley_5word_tf_augmented_dataset.npz"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "Data"
    / "datasets"
    / "berkeley_5word_tf_augmented_metadata.csv"
)


# ------------------------------------------------------------
# FIVE WORD VOCABULARY
# ------------------------------------------------------------

TARGET_WORDS = [
    "THE",
    "AND",
    "A",
    "OF",
    "I",
]


# ------------------------------------------------------------
# RANDOM SEED
# ------------------------------------------------------------

RANDOM_STATE = 42

rng = np.random.default_rng(RANDOM_STATE)


# ============================================================
# AUGMENTATION PARAMETERS
# ============================================================

# Speed perturbation
SPEED_MIN = 0.85
SPEED_MAX = 1.15

# Time masking
MAX_TIME_MASK = 10

# Frequency masking
MAX_FREQ_MASK = 4


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# TIME RESIZING / SPEED PERTURBATION
# ============================================================

def speed_perturbation(sample, speed_factor):
    """
    Apply speed perturbation along the TIME axis.

    Input:
        sample: (channels, frequency_bins, time_frames)

    Output:
        same shape as input
    """

    channels, freq_bins, time_frames = sample.shape

    new_length = max(
        2,
        int(round(time_frames / speed_factor))
    )

    old_x = np.linspace(
        0.0,
        1.0,
        time_frames
    )

    new_x = np.linspace(
        0.0,
        1.0,
        new_length
    )

    resized = np.empty(
        (channels, freq_bins, new_length),
        dtype=np.float32
    )

    for c in range(channels):

        for f in range(freq_bins):

            resized[c, f] = np.interp(
                new_x,
                old_x,
                sample[c, f]
            )

    # Resize back to exactly 64 frames

    final_x = np.linspace(
        0.0,
        1.0,
        time_frames
    )

    resized_final = np.empty(
        (channels, freq_bins, time_frames),
        dtype=np.float32
    )

    resized_old_x = np.linspace(
        0.0,
        1.0,
        new_length
    )

    for c in range(channels):

        for f in range(freq_bins):

            resized_final[c, f] = np.interp(
                final_x,
                resized_old_x,
                resized[c, f]
            )

    return resized_final


# ============================================================
# TIME MASKING
# ============================================================

def time_mask(sample):
    """
    Mask a random section of the TIME axis.

    Input:
        (channels, frequency_bins, time_frames)
    """

    output = sample.copy()

    _, _, time_frames = output.shape

    if time_frames <= 1:
        return output

    max_width = min(
        MAX_TIME_MASK,
        time_frames
    )

    width = rng.integers(
        1,
        max_width + 1
    )

    start_max = time_frames - width

    if start_max <= 0:
        start = 0
    else:
        start = rng.integers(
            0,
            start_max + 1
        )

    # Use zero masking
    output[:, :, start:start + width] = 0.0

    return output


# ============================================================
# FREQUENCY MASKING
# ============================================================

def frequency_mask(sample):
    """
    Mask a random section of the FREQUENCY axis.

    Input:
        (channels, frequency_bins, time_frames)
    """

    output = sample.copy()

    _, freq_bins, _ = output.shape

    if freq_bins <= 1:
        return output

    max_width = min(
        MAX_FREQ_MASK,
        freq_bins
    )

    width = rng.integers(
        1,
        max_width + 1
    )

    start_max = freq_bins - width

    if start_max <= 0:
        start = 0
    else:
        start = rng.integers(
            0,
            start_max + 1
        )

    output[:, start:start + width, :] = 0.0

    return output


# ============================================================
# VALIDATION
# ============================================================

def check_array(name, array):

    print(f"{name} NaN values:", np.isnan(array).sum())
    print(f"{name} Infinite values:", np.isinf(array).sum())

    if np.isnan(array).any():
        raise RuntimeError(
            f"{name} contains NaN values."
        )

    if np.isinf(array).any():
        raise RuntimeError(
            f"{name} contains infinite values."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "IRIS - BERKELEY 5-WORD TF DATASET CREATION"
    )

    print()
    print("Input:")
    print(INPUT_PATH)

    print()
    print("Output:")
    print(OUTPUT_PATH)

    print()
    print("Vocabulary:")
    for word in TARGET_WORDS:
        print("  ", word)

    # ========================================================
    # LOAD DATASET
    # ========================================================

    print_header("LOADING ORIGINAL TF DATASET")

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_PATH}"
        )

    data = np.load(
        INPUT_PATH,
        allow_pickle=True
    )

    print("Dataset keys:")
    print(data.files)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(str)

    print()
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # ========================================================
    # NUMERICAL VALIDATION
    # ========================================================

    print_header("NUMERICAL VALIDATION")

    check_array("X", X)

    # ========================================================
    # ORIGINAL DISTRIBUTION
    # ========================================================

    print_header("ORIGINAL CLASS DISTRIBUTION")

    original_counts = Counter(y)

    print("Original samples:", len(y))
    print("Original classes:", len(original_counts))

    print()
    print("Target vocabulary counts:")

    for word in TARGET_WORDS:

        count = original_counts.get(
            word,
            0
        )

        print(
            f"{word:<10}: {count}"
        )

    # ========================================================
    # VOCABULARY FILTERING
    # ========================================================

    print_header("FILTERING TO FIVE WORDS")

    target_set = set(TARGET_WORDS)

    mask = np.array([
        label in target_set
        for label in y
    ])

    X_filtered = X[mask]
    y_filtered = y[mask]

    print(
        "Samples after vocabulary filtering:",
        len(y_filtered)
    )

    # Check every target word

    missing_words = []

    for word in TARGET_WORDS:

        count = np.sum(
            y_filtered == word
        )

        print(
            f"{word:<10}: {count}"
        )

        if count == 0:
            missing_words.append(word)

    if missing_words:

        raise RuntimeError(
            "The following target words were not found:\n"
            + "\n".join(missing_words)
        )

    # ========================================================
    # DISTRIBUTION
    # ========================================================

    print()
    print("Filtered class distribution:")

    filtered_counts = Counter(y_filtered)

    for word in TARGET_WORDS:

        print(
            f"{word:<10}: "
            f"{filtered_counts[word]}"
        )

    # ========================================================
    # ENCODE LABELS
    # ========================================================

    class_to_index = {
        word: index
        for index, word in enumerate(TARGET_WORDS)
    }

    y_encoded = np.array(
        [
            class_to_index[word]
            for word in y_filtered
        ],
        dtype=np.int64
    )

    classes = np.array(
        TARGET_WORDS
    )

    # ========================================================
    # TRAIN / VALIDATION / TEST SPLIT
    # ========================================================

    print_header(
        "TRAIN / VALIDATION / TEST SPLIT"
    )

    indices = np.arange(
        len(y_encoded)
    )

    # First:
    # 70% train
    # 30% temporary

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )

    # Then split temporary:
    # 15% validation
    # 15% test

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_encoded[temp_idx]
    )

    print("Original filtered dataset:")
    print("Samples:", len(indices))
    print("Classes:", len(classes))

    print()
    print("Train:")
    print(len(train_idx))

    print("Validation:")
    print(len(val_idx))

    print("Test:")
    print(len(test_idx))

    print()
    print("Percentages:")

    print(
        f"Train: "
        f"{100 * len(train_idx) / len(indices):.2f}%"
    )

    print(
        f"Validation: "
        f"{100 * len(val_idx) / len(indices):.2f}%"
    )

    print(
        f"Test: "
        f"{100 * len(test_idx) / len(indices):.2f}%"
    )

    # ========================================================
    # CREATE ORIGINAL SPLITS
    # ========================================================

    X_train_original = X_filtered[train_idx]
    y_train_original = y_encoded[train_idx]

    X_val = X_filtered[val_idx]
    y_val = y_encoded[val_idx]

    X_test = X_filtered[test_idx]
    y_test = y_encoded[test_idx]

    print()
    print(
        "Classes in training:",
        len(np.unique(y_train_original))
    )

    print(
        "Classes in validation:",
        len(np.unique(y_val))
    )

    print(
        "Classes in test:",
        len(np.unique(y_test))
    )

    # ========================================================
    # AUGMENT TRAINING DATA
    # ========================================================

    print_header(
        "AUGMENTING TRAINING DATA"
    )

    print(
        "Each training sample produces:"
    )

    print("  1. Original")
    print("  2. Speed perturbation")
    print("  3. Time masking")
    print("  4. Frequency masking")

    print()
    print(
        "Validation and test are NOT augmented."
    )

    augmented_X = []
    augmented_y = []

    total = len(X_train_original)

    for i in range(total):

        sample = X_train_original[i]

        label = y_train_original[i]

        # ----------------------------------------------------
        # 1. ORIGINAL
        # ----------------------------------------------------

        augmented_X.append(
            sample.astype(np.float32)
        )

        augmented_y.append(
            label
        )

        # ----------------------------------------------------
        # 2. SPEED PERTURBATION
        # ----------------------------------------------------

        speed = rng.uniform(
            SPEED_MIN,
            SPEED_MAX
        )

        speed_sample = speed_perturbation(
            sample,
            speed
        )

        augmented_X.append(
            speed_sample.astype(np.float32)
        )

        augmented_y.append(
            label
        )

        # ----------------------------------------------------
        # 3. TIME MASK
        # ----------------------------------------------------

        time_sample = time_mask(
            sample
        )

        augmented_X.append(
            time_sample.astype(np.float32)
        )

        augmented_y.append(
            label
        )

        # ----------------------------------------------------
        # 4. FREQUENCY MASK
        # ----------------------------------------------------

        freq_sample = frequency_mask(
            sample
        )

        augmented_X.append(
            freq_sample.astype(np.float32)
        )

        augmented_y.append(
            label
        )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            (i + 1) % 100 == 0
            or i + 1 == total
        ):

            print(
                f"Processed {i + 1}/{total} | "
                f"Augmented samples: "
                f"{len(augmented_X)}"
            )

    # ========================================================
    # CONVERT TO ARRAYS
    # ========================================================

    X_train = np.stack(
        augmented_X
    ).astype(np.float32)

    y_train = np.array(
        augmented_y,
        dtype=np.int64
    )

    X_val = X_val.astype(
        np.float32
    )

    X_test = X_test.astype(
        np.float32
    )

    # ========================================================
    # FINAL NUMERICAL CHECK
    # ========================================================

    print_header(
        "FINAL NUMERICAL CHECK"
    )

    check_array(
        "Training",
        X_train
    )

    check_array(
        "Validation",
        X_val
    )

    check_array(
        "Test",
        X_test
    )

    # ========================================================
    # FINAL SHAPES
    # ========================================================

    print_header(
        "FINAL DATASET"
    )

    print(
        "Original filtered samples:",
        len(indices)
    )

    print(
        "Original training samples:",
        len(X_train_original)
    )

    print(
        "Augmented training samples:",
        len(X_train)
    )

    print(
        "Training expansion:",
        f"{len(X_train) / len(X_train_original):.2f}x"
    )

    print()
    print(
        "Validation samples:",
        len(X_val)
    )

    print(
        "Test samples:",
        len(X_test)
    )

    print(
        "Classes:",
        len(classes)
    )

    print()
    print(
        "X_train:",
        X_train.shape
    )

    print(
        "y_train:",
        y_train.shape
    )

    print(
        "X_val:",
        X_val.shape
    )

    print(
        "y_val:",
        y_val.shape
    )

    print(
        "X_test:",
        X_test.shape
    )

    print(
        "y_test:",
        y_test.shape
    )

    # ========================================================
    # CLASS DISTRIBUTION AFTER AUGMENTATION
    # ========================================================

    print_header(
        "FINAL TRAINING CLASS DISTRIBUTION"
    )

    train_counts = Counter(y_train)

    for index, word in enumerate(classes):

        print(
            f"{word:<10}: "
            f"{train_counts[index]}"
        )

    # ========================================================
    # SAVE DATASET
    # ========================================================

    print_header(
        "SAVING DATASET"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_PATH,

        X_train=X_train,
        y_train=y_train,

        X_val=X_val,
        y_val=y_val,

        X_test=X_test,
        y_test=y_test,

        classes=classes,

        fs=np.array(
            data["fs"]
        ),

        fmin=np.array(
            data["fmin"]
        ),

        fmax=np.array(
            data["fmax"]
        ),

        n_fft=np.array(
            data["n_fft"]
        ),

        hop_length=np.array(
            data["hop_length"]
        ),

        vocabulary=np.array(
            TARGET_WORDS
        ),

        random_state=np.array(
            RANDOM_STATE
        ),
    )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata_rows = []

    for i in range(len(y_train)):

        metadata_rows.append({
            "split": "train",
            "sample_index": i,
            "class_index": int(y_train[i]),
            "word": classes[y_train[i]],
            "augmented": True
        })

    for i in range(len(y_val)):

        metadata_rows.append({
            "split": "validation",
            "sample_index": i,
            "class_index": int(y_val[i]),
            "word": classes[y_val[i]],
            "augmented": False
        })

    for i in range(len(y_test)):

        metadata_rows.append({
            "split": "test",
            "sample_index": i,
            "class_index": int(y_test[i]),
            "word": classes[y_test[i]],
            "augmented": False
        })

    metadata = pd.DataFrame(
        metadata_rows
    )

    metadata.to_csv(
        METADATA_PATH,
        index=False
    )

    # ========================================================
    # SUCCESS
    # ========================================================

    print_header(
        "DATASET CREATED SUCCESSFULLY"
    )

    print(
        "Vocabulary:"
    )

    for index, word in enumerate(classes):

        print(
            f"  {index}: {word}"
        )

    print()
    print(
        "X_train:",
        X_train.shape
    )

    print(
        "X_val:",
        X_val.shape
    )

    print(
        "X_test:",
        X_test.shape
    )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print(
        "Metadata:"
    )

    print(
        METADATA_PATH
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()

