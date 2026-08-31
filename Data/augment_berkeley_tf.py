"""
======================================================================
IRIS - BERKELEY TF DATA AUGMENTATION
======================================================================

Input:
    berkeley_tf_dataset.npz

Input shape:
    (samples, 8, 24, 64)

Augmentation:
    1. Original
    2. Speed perturbation
    3. Time masking
    4. Frequency masking

IMPORTANT:
    Dataset is split BEFORE augmentation.

    ONLY training data is augmented.

    Validation and test data remain completely untouched.

======================================================================
"""

import os
import numpy as np

from collections import Counter
from sklearn.model_selection import train_test_split


# =====================================================================
# CONFIGURATION
# =====================================================================

DATASET_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/Data/datasets/"
    "berkeley_tf_dataset.npz"
)

OUTPUT_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/Data/datasets/"
    "berkeley_tf_augmented_dataset.npz"
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# IMPORTANT
#
# We now require at least 6 examples per word.
#
# Why 6?
#
# We need at least:
#     1 train
#     1 validation
#     1 test
#
# while still keeping enough training examples.
#
# Using 6 also avoids the pathological case where a class
# disappears from the temporary split.
# ---------------------------------------------------------------------

MIN_SAMPLES_PER_CLASS = 6


# =====================================================================
# AUGMENTATION PARAMETERS
# =====================================================================

SPEED_FAST = 0.90
SPEED_SLOW = 1.10

MAX_TIME_MASK = 10
MAX_FREQ_MASK = 5


# =====================================================================
# RANDOM GENERATOR
# =====================================================================

rng = np.random.default_rng(RANDOM_STATE)


# =====================================================================
# LOAD DATASET
# =====================================================================

print("=" * 70)
print("IRIS - BERKELEY TF DATA AUGMENTATION")
print("=" * 70)

print("\nLoading:")
print(DATASET_PATH)

data = np.load(
    DATASET_PATH,
    allow_pickle=True
)

print("\nDataset keys:")
print(data.files)

X = data["X"]
y = data["y"]


print("\nOriginal dataset:")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("X dtype:", X.dtype)
print("y dtype:", y.dtype)


# =====================================================================
# VALIDATE DATA
# =====================================================================

print("\n" + "=" * 70)
print("DATA VALIDATION")
print("=" * 70)

if np.isnan(X).any():
    raise RuntimeError("X contains NaN values.")

if np.isinf(X).any():
    raise RuntimeError("X contains infinite values.")

if len(X) != len(y):
    raise RuntimeError(
        "X and y have different numbers of samples."
    )

if X.ndim != 4:
    raise RuntimeError(
        f"Expected X to have 4 dimensions, got {X.ndim}"
    )


CHANNELS = X.shape[1]
FREQ_BINS = X.shape[2]
TIME_FRAMES = X.shape[3]

print("\nNaN values:", np.isnan(X).sum())
print("Infinite values:", np.isinf(X).sum())

print("\nChannels:", CHANNELS)
print("Frequency bins:", FREQ_BINS)
print("Time frames:", TIME_FRAMES)


# =====================================================================
# CLASS COUNTS
# =====================================================================

print("\n" + "=" * 70)
print("ORIGINAL CLASS DISTRIBUTION")
print("=" * 70)

class_counts = Counter(y)

print("\nOriginal samples:", len(y))
print("Original classes:", len(class_counts))

print(
    "Minimum samples/class:",
    min(class_counts.values())
)

print(
    "Maximum samples/class:",
    max(class_counts.values())
)


# =====================================================================
# REMOVE RARE CLASSES
# =====================================================================

print("\n" + "=" * 70)
print("REMOVING RARE WORDS")
print("=" * 70)

valid_classes = {
    cls
    for cls, count in class_counts.items()
    if count >= MIN_SAMPLES_PER_CLASS
}

keep_mask = np.array([
    label in valid_classes
    for label in y
])

X = X[keep_mask]
y = y[keep_mask]

new_counts = Counter(y)

print(
    "\nMinimum samples/class required:",
    MIN_SAMPLES_PER_CLASS
)

print(
    "Classes retained:",
    len(new_counts)
)

print(
    "Classes removed:",
    len(class_counts) - len(new_counts)
)

print(
    "Samples retained:",
    len(y)
)

print(
    "Minimum retained class size:",
    min(new_counts.values())
)


# =====================================================================
# SAFE CLASS-BY-CLASS SPLIT
# =====================================================================
#
# THIS IS THE IMPORTANT FIX.
#
# We do NOT use:
#
#     train_test_split(... stratify=y)
#
# twice.
#
# Instead, each class is split independently.
#
# This guarantees that every retained word gets:
#
#     TRAIN
#     VALIDATION
#     TEST
#
# =====================================================================

print("\n" + "=" * 70)
print("TRAIN / VALIDATION / TEST SPLIT")
print("=" * 70)

all_indices = np.arange(len(y))

train_indices = []
val_indices = []
test_indices = []

unique_classes = np.unique(y)

for class_label in unique_classes:

    class_idx = all_indices[y == class_label]

    # Shuffle samples belonging to this class.
    class_idx = class_idx.copy()
    rng.shuffle(class_idx)

    n = len(class_idx)

    # -------------------------------------------------------------
    # Determine number of samples.
    #
    # We aim for approximately:
    #
    # TRAIN = 70%
    # VAL   = 15%
    # TEST  = 15%
    #
    # Every class gets at least:
    #
    # TRAIN >= 1
    # VAL   >= 1
    # TEST  >= 1
    # -------------------------------------------------------------

    n_val = max(1, int(round(n * 0.15)))
    n_test = max(1, int(round(n * 0.15)))

    n_train = n - n_val - n_test

    # Safety check.
    if n_train < 1:

        # Give one sample back to training.
        n_train = 1

        if n_val > 1:
            n_val -= 1

        elif n_test > 1:
            n_test -= 1

    # -------------------------------------------------------------
    # Final safety check.
    # -------------------------------------------------------------

    if n_train < 1 or n_val < 1 or n_test < 1:

        raise RuntimeError(
            f"Could not safely split class {class_label} "
            f"with {n} samples."
        )

    # -------------------------------------------------------------
    # Assign samples.
    # -------------------------------------------------------------

    train_indices.extend(
        class_idx[:n_train]
    )

    val_indices.extend(
        class_idx[n_train:n_train + n_val]
    )

    test_indices.extend(
        class_idx[n_train + n_val:]
    )


# =====================================================================
# CONVERT INDICES
# =====================================================================

train_idx = np.array(
    train_indices,
    dtype=np.int64
)

val_idx = np.array(
    val_indices,
    dtype=np.int64
)

test_idx = np.array(
    test_indices,
    dtype=np.int64
)


# Shuffle each split.

rng.shuffle(train_idx)
rng.shuffle(val_idx)
rng.shuffle(test_idx)


# =====================================================================
# CHECK FOR OVERLAP
# =====================================================================

if len(set(train_idx) & set(val_idx)) > 0:
    raise RuntimeError(
        "Data leakage detected: train/validation overlap."
    )

if len(set(train_idx) & set(test_idx)) > 0:
    raise RuntimeError(
        "Data leakage detected: train/test overlap."
    )

if len(set(val_idx) & set(test_idx)) > 0:
    raise RuntimeError(
        "Data leakage detected: validation/test overlap."
    )


# =====================================================================
# CREATE SPLITS
# =====================================================================

X_train = X[train_idx]
y_train = y[train_idx]

X_val = X[val_idx]
y_val = y[val_idx]

X_test = X[test_idx]
y_test = y[test_idx]


# =====================================================================
# SPLIT SUMMARY
# =====================================================================

print("\nOriginal filtered dataset:")
print("Samples:", len(X))
print("Classes:", len(np.unique(y)))

print("\nTrain:")
print(len(X_train))

print("Validation:")
print(len(X_val))

print("Test:")
print(len(X_test))

print("\nPercentages:")

print(
    "Train:",
    f"{len(X_train) / len(X) * 100:.2f}%"
)

print(
    "Validation:",
    f"{len(X_val) / len(X) * 100:.2f}%"
)

print(
    "Test:",
    f"{len(X_test) / len(X) * 100:.2f}%"
)


# =====================================================================
# VERIFY EVERY CLASS EXISTS IN EVERY SPLIT
# =====================================================================

train_classes = set(y_train)
val_classes = set(y_val)
test_classes = set(y_test)

print("\nClasses in training:", len(train_classes))
print("Classes in validation:", len(val_classes))
print("Classes in test:", len(test_classes))

if train_classes != val_classes:
    print(
        "\nWARNING: train and validation class sets differ."
    )

if train_classes != test_classes:
    print(
        "WARNING: train and test class sets differ."
    )


# =====================================================================
# SPEED PERTURBATION
# =====================================================================

def speed_perturb_tf(sample, speed_factor):

    """
    Speed perturbation along the time axis.

    Input:
        (channels, frequency_bins, time_frames)

    Output:
        (channels, frequency_bins, time_frames)
    """

    channels, freq_bins, time_frames = sample.shape

    new_length = max(
        2,
        int(round(time_frames / speed_factor))
    )

    old_positions = np.linspace(
        0.0,
        1.0,
        time_frames
    )

    new_positions = np.linspace(
        0.0,
        1.0,
        new_length
    )

    intermediate = np.empty(
        (channels, freq_bins, new_length),
        dtype=np.float32
    )

    for c in range(channels):

        for f in range(freq_bins):

            intermediate[c, f] = np.interp(
                new_positions,
                old_positions,
                sample[c, f]
            )

    # Resize back to original number of frames.

    final_positions = np.linspace(
        0.0,
        1.0,
        time_frames
    )

    old_positions_2 = np.linspace(
        0.0,
        1.0,
        new_length
    )

    output = np.empty(
        (channels, freq_bins, time_frames),
        dtype=np.float32
    )

    for c in range(channels):

        for f in range(freq_bins):

            output[c, f] = np.interp(
                final_positions,
                old_positions_2,
                intermediate[c, f]
            )

    return output


# =====================================================================
# TIME MASKING
# =====================================================================

def time_mask(sample):

    """
    Randomly masks a region along the time axis.
    """

    augmented = sample.copy()

    _, _, time_frames = augmented.shape

    max_width = min(
        MAX_TIME_MASK,
        time_frames - 1
    )

    width = rng.integers(
        1,
        max_width + 1
    )

    start = rng.integers(
        0,
        time_frames - width + 1
    )

    augmented[
        :,
        :,
        start:start + width
    ] = 0.0

    return augmented


# =====================================================================
# FREQUENCY MASKING
# =====================================================================

def frequency_mask(sample):

    """
    Randomly masks a region along the frequency axis.
    """

    augmented = sample.copy()

    _, freq_bins, _ = augmented.shape

    max_width = min(
        MAX_FREQ_MASK,
        freq_bins - 1
    )

    width = rng.integers(
        1,
        max_width + 1
    )

    start = rng.integers(
        0,
        freq_bins - width + 1
    )

    augmented[
        :,
        start:start + width,
        :
    ] = 0.0

    return augmented


# =====================================================================
# AUGMENT TRAINING DATA
# =====================================================================

print("\n" + "=" * 70)
print("AUGMENTING TRAINING DATA")
print("=" * 70)

print(
    "\nEach training sample produces:"
)

print("  1. Original")
print("  2. Speed perturbation")
print("  3. Time masking")
print("  4. Frequency masking")

print(
    "\nValidation and test are NOT augmented."
)


augmented_X = []
augmented_y = []


total_train = len(X_train)


for i in range(total_train):

    sample = X_train[i].astype(
        np.float32,
        copy=False
    )

    label = y_train[i]

    # ---------------------------------------------------------
    # 1. ORIGINAL
    # ---------------------------------------------------------

    augmented_X.append(
        sample.copy()
    )

    augmented_y.append(label)

    # ---------------------------------------------------------
    # 2. SPEED PERTURBATION
    # ---------------------------------------------------------

    if i % 2 == 0:

        speed_factor = SPEED_FAST

    else:

        speed_factor = SPEED_SLOW

    speed_sample = speed_perturb_tf(
        sample,
        speed_factor
    )

    augmented_X.append(speed_sample)
    augmented_y.append(label)

    # ---------------------------------------------------------
    # 3. TIME MASKING
    # ---------------------------------------------------------

    time_sample = time_mask(sample)

    augmented_X.append(time_sample)
    augmented_y.append(label)

    # ---------------------------------------------------------
    # 4. FREQUENCY MASKING
    # ---------------------------------------------------------

    freq_sample = frequency_mask(sample)

    augmented_X.append(freq_sample)
    augmented_y.append(label)

    # ---------------------------------------------------------
    # Progress
    # ---------------------------------------------------------

    if (
        (i + 1) % 100 == 0
        or
        (i + 1) == total_train
    ):

        print(
            f"Processed {i + 1}/{total_train} | "
            f"Augmented samples: {len(augmented_X)}"
        )


# =====================================================================
# STACK AUGMENTED DATA
# =====================================================================

X_train_aug = np.stack(
    augmented_X,
    axis=0
).astype(np.float32)

y_train_aug = np.asarray(
    augmented_y
)


# =====================================================================
# FINAL NUMERICAL CHECK
# =====================================================================

print("\n" + "=" * 70)
print("FINAL NUMERICAL CHECK")
print("=" * 70)

print(
    "\nTraining NaN:",
    np.isnan(X_train_aug).sum()
)

print(
    "Training Inf:",
    np.isinf(X_train_aug).sum()
)

print(
    "Validation NaN:",
    np.isnan(X_val).sum()
)

print(
    "Validation Inf:",
    np.isinf(X_val).sum()
)

print(
    "Test NaN:",
    np.isnan(X_test).sum()
)

print(
    "Test Inf:",
    np.isinf(X_test).sum()
)


# =====================================================================
# FINAL SHAPES
# =====================================================================

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print("\nX_train:")
print(X_train_aug.shape)

print("\ny_train:")
print(y_train_aug.shape)

print("\nX_val:")
print(X_val.shape)

print("\ny_val:")
print(y_val.shape)

print("\nX_test:")
print(X_test.shape)

print("\ny_test:")
print(y_test.shape)


# =====================================================================
# SAVE
# =====================================================================

print("\n" + "=" * 70)
print("SAVING DATASET")
print("=" * 70)

np.savez_compressed(

    OUTPUT_PATH,

    X_train=X_train_aug,
    y_train=y_train_aug,

    X_val=X_val.astype(np.float32),
    y_val=y_val,

    X_test=X_test.astype(np.float32),
    y_test=y_test,

    channels=np.array(CHANNELS),

    frequency_bins=np.array(FREQ_BINS),

    time_frames=np.array(TIME_FRAMES),

    sampling_rate=np.array(600),

    augmentation=np.array(
        [
            "original",
            "speed_perturbation",
            "time_masking",
            "frequency_masking"
        ],
        dtype=object
    ),

    speed_factors=np.array(
        [
            SPEED_FAST,
            SPEED_SLOW
        ]
    ),

    max_time_mask=np.array(
        MAX_TIME_MASK
    ),

    max_frequency_mask=np.array(
        MAX_FREQ_MASK
    ),

    min_samples_per_class=np.array(
        MIN_SAMPLES_PER_CLASS
    ),

    random_state=np.array(
        RANDOM_STATE
    )
)


# =====================================================================
# FINAL SUMMARY
# =====================================================================

print("\n" + "=" * 70)
print("DATASET CREATED SUCCESSFULLY")
print("=" * 70)

print("\nOriginal filtered samples:")
print(len(X))

print("\nOriginal training samples:")
print(len(X_train))

print("\nAugmented training samples:")
print(len(X_train_aug))

print(
    "\nTraining expansion:",
    f"{len(X_train_aug) / len(X_train):.2f}x"
)

print("\nValidation samples:")
print(len(X_val))

print("\nTest samples:")
print(len(X_test))

print("\nClasses:")
print(len(np.unique(y)))

print("\nFinal X_train shape:")
print(X_train_aug.shape)

print("\nFinal X_val shape:")
print(X_val.shape)

print("\nFinal X_test shape:")
print(X_test.shape)

print("\nSaved to:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)