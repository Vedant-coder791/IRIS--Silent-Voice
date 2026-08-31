import sys
import os

import numpy as np

from collections import Counter

from Data.Extract_words import (
    extract_word_segments,
    print_subject_distribution
)

from Filter.Word_filter import filter_emg

from MFCC_extraction.MFCC import extract_mfcc

from sklearn.preprocessing import LabelEncoder, StandardScaler


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# LOAD DATASET
# ============================================================

from Data.Load_dataset import load_dataset


print("\n==========================================")
print("LOADING DATASET")
print("==========================================")

dataset = load_dataset()


# ============================================================
# EXTRACT WORD SEGMENTS
# ============================================================

word_segments = extract_word_segments(
    dataset
)

print_subject_distribution(
    word_segments
)


# ============================================================
# FILTER ALL WORD SEGMENTS
# ============================================================

print("\n==========================================")
print("FILTERING ALL WORD SEGMENTS")
print("==========================================")

filtered_segments = []

failed_segments = []

for i, segment in enumerate(
    word_segments
):

    try:

        emg = segment["emg"]

        filtered_emg = filter_emg(
            emg
        )

        filtered_segment = segment.copy()

        filtered_segment["emg"] = (
            filtered_emg
        )

        filtered_segments.append(
            filtered_segment
        )

    except Exception as e:

        failed_segments.append(
            (i, str(e))
        )

    if (i + 1) % 100 == 0:

        print(
            f"Filtered {i + 1}/{len(word_segments)}"
        )


print("\n==========================================")
print("FILTERING RESULTS")
print("==========================================")

print(
    "Original segments:",
    len(word_segments)
)

print(
    "Successfully filtered:",
    len(filtered_segments)
)

print(
    "Failed:",
    len(failed_segments)
)

if failed_segments:

    print("\nFirst failures:")

    for index, error in failed_segments[:10]:

        print(
            f"Segment {index}: {error}"
        )


# ============================================================
# EXTRACT MFCC FEATURES
# ============================================================

print("\n==========================================")
print("EXTRACTING MFCC FEATURES")
print("==========================================")

feature_segments = []

failed_features = []

for i, segment in enumerate(
    filtered_segments
):

    try:

        emg = segment["emg"]

        mfcc = extract_mfcc(
            emg
        )

        feature_segment = segment.copy()

        feature_segment["mfcc"] = (
            mfcc
        )

        feature_segments.append(
            feature_segment
        )

    except Exception as e:

        failed_features.append(
            (i, str(e))
        )

    if (i + 1) % 100 == 0:

        print(
            f"Processed {i + 1}/{len(filtered_segments)}"
        )


print("\n==========================================")
print("MFCC EXTRACTION RESULTS")
print("==========================================")

print(
    "Filtered segments:",
    len(filtered_segments)
)

print(
    "Successful MFCCs:",
    len(feature_segments)
)

print(
    "Failed:",
    len(failed_features)
)


# ============================================================
# CREATE FIXED-SIZE FEATURES
# ============================================================

print("\n==========================================")
print("CREATING FIXED-SIZE FEATURES")
print("==========================================")

X = []
y = []
subjects = []


for segment in feature_segments:

    mfcc = segment["mfcc"]

    # --------------------------------------------------------
    # MFCC shape:
    #
    # channels × coefficients × time
    # --------------------------------------------------------

    mfcc_mean = np.mean(
        mfcc,
        axis=2
    )

    mfcc_std = np.std(
        mfcc,
        axis=2
    )

    features = np.concatenate(
        [
            mfcc_mean.flatten(),
            mfcc_std.flatten()
        ]
    )

    X.append(
        features
    )

    y.append(
        segment["label"]
    )

    subjects.append(
        segment["subject"]
    )


X = np.asarray(
    X,
    dtype=np.float32
)

y = np.asarray(
    y
)

subjects = np.asarray(
    subjects
)


print("\n==========================================")
print("FEATURE DATASET")
print("==========================================")

print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)

print(
    "Subjects shape:",
    subjects.shape
)


# ============================================================
# LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(
    y
)

print("\n==========================================")
print("WORD CLASSES")
print("==========================================")

for index, word in enumerate(
    label_encoder.classes_
):

    print(
        index,
        "->",
        word
    )


# ============================================================
# SUBJECT-AWARE SPLIT
# ============================================================

print("\n==========================================")
print("SUBJECT-AWARE DATASET SPLIT")
print("==========================================")

unique_subjects = sorted(
    np.unique(subjects)
)

print(
    "Available subjects:",
    unique_subjects
)


# ------------------------------------------------------------
# IMPORTANT:
#
# Subject 002 → TRAINING
# Subject 004 → VALIDATION
# Subject 006 → TEST
# Subject 008 → TRAINING
#
# This tests whether the model can generalize to a
# completely unseen subject.
# ------------------------------------------------------------

TRAIN_SUBJECTS = [
    "002",
    "008"
]

VAL_SUBJECTS = [
    "004"
]

TEST_SUBJECTS = [
    "006"
]


train_mask = np.isin(
    subjects,
    TRAIN_SUBJECTS
)

val_mask = np.isin(
    subjects,
    VAL_SUBJECTS
)

test_mask = np.isin(
    subjects,
    TEST_SUBJECTS
)


X_train = X[train_mask]
y_train = y_encoded[train_mask]

X_val = X[val_mask]
y_val = y_encoded[val_mask]

X_test = X[test_mask]
y_test = y_encoded[test_mask]


print("\nTraining subjects:")
print(TRAIN_SUBJECTS)

print(
    "Training shape:",
    X_train.shape
)

print("\nValidation subject:")
print(VAL_SUBJECTS)

print(
    "Validation shape:",
    X_val.shape
)

print("\nTesting subject:")
print(TEST_SUBJECTS)

print(
    "Testing shape:",
    X_test.shape
)


# ============================================================
# NORMALIZATION
# ============================================================

print("\n==========================================")
print("NORMALIZATION")
print("==========================================")

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_val = scaler.transform(
    X_val
)

X_test = scaler.transform(
    X_test
)


print(
    "Training mean:",
    np.mean(X_train)
)

print(
    "Training standard deviation:",
    np.std(X_train)
)


# ============================================================
# FINAL DATA CHECK
# ============================================================

print("\n==========================================")
print("FINAL DATA CHECK")
print("==========================================")

print(
    "Training finite:",
    np.all(np.isfinite(X_train))
)

print(
    "Validation finite:",
    np.all(np.isfinite(X_val))
)

print(
    "Testing finite:",
    np.all(np.isfinite(X_test))
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n==========================================")
print("FINAL CLASS DISTRIBUTION")
print("==========================================")


print("\nTRAINING")

train_counts = Counter(
    y_train
)

for index, word in enumerate(
    label_encoder.classes_
):

    print(
        f"{word:8s}: "
        f"{train_counts.get(index, 0):4d}"
    )


print("\nVALIDATION")

val_counts = Counter(
    y_val
)

for index, word in enumerate(
    label_encoder.classes_
):

    print(
        f"{word:8s}: "
        f"{val_counts.get(index, 0):4d}"
    )


print("\nTEST")

test_counts = Counter(
    y_test
)

for index, word in enumerate(
    label_encoder.classes_
):

    print(
        f"{word:8s}: "
        f"{test_counts.get(index, 0):4d}"
    )


# ============================================================
# SAVE DATASET
# ============================================================

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "Data",
    "dataset_subject.npz"
)


print("\n==========================================")
print("SAVING SUBJECT-AWARE DATASET")
print("==========================================")


np.savez(
    OUTPUT_FILE,

    X_train=X_train.astype(
        np.float32
    ),

    y_train=y_train,

    X_val=X_val.astype(
        np.float32
    ),

    y_val=y_val,

    X_test=X_test.astype(
        np.float32
    ),

    y_test=y_test,

    classes=label_encoder.classes_,

    train_subjects=np.asarray(
        TRAIN_SUBJECTS
    ),

    val_subjects=np.asarray(
        VAL_SUBJECTS
    ),

    test_subjects=np.asarray(
        TEST_SUBJECTS
    )
)


print(
    "Dataset saved successfully!"
)

print(
    "File:",
    OUTPUT_FILE
)