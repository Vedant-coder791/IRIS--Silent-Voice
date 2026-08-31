
# ============================================================
# IRIS - MLP PROTO 1.7
# Berkeley Closed-Vocabulary Silent Speech
#
# PURPOSE:
#   Numerically stable baseline MLP.
#
# Proto 1.7 changes:
#   - Robust numerical preprocessing
#   - Training-only feature clipping
#   - float64 processing
#   - StandardScaler
#   - Smaller MLP
#   - Stronger L2 regularization
#   - Early stopping OFF
#   - Detailed numerical checks
#   - Classification report
#   - Confusion matrix
#   - Per-class accuracy
#   - Model saving
#
# Dataset:
#   MLP_proto1_4_dataset.npz
#
# Vocabulary:
#   AM
#   PM
#   DECEMBER
#   AUGUST
#   MARCH
# ============================================================


import os
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/AI/"
    "MLP_proto1_4_dataset.npz"
)

MODEL_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/AI/"
    "MLP_proto1_7_model.pkl"
)

RANDOM_STATE = 42

TEST_SIZE = 0.20

EXPECTED_FEATURES = 112

# Training-only percentile clipping.
#
# We calculate these limits ONLY from X_train.
# Therefore the test set is never used to determine
# preprocessing parameters.
CLIP_LOW_PERCENTILE = 1.0
CLIP_HIGH_PERCENTILE = 99.0


# ============================================================
# TARGET VOCABULARY
# ============================================================

TARGET_CLASSES = [
    "AM",
    "AUGUST",
    "DECEMBER",
    "MARCH",
    "PM"
]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("IRIS - MLP PROTO 1.7")
print("Berkeley Closed-Vocabulary Silent Speech")
print("=" * 70)

print()
print("Dataset:")
print(DATASET_PATH)

print()
print("Target vocabulary:")

for i, cls in enumerate(TARGET_CLASSES, 1):
    print(f"  {i}. {cls}")


# ============================================================
# CHECK DATASET
# ============================================================

print()
print("=" * 70)
print("CHECKING DATASET")
print("=" * 70)

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_PATH}"
    )

print("Dataset found.")


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("=" * 70)
print("LOADING FEATURE DATASET")
print("=" * 70)

data = np.load(
    DATASET_PATH,
    allow_pickle=True
)

print("Dataset keys:")
print(list(data.keys()))


# ============================================================
# READ DATA
# ============================================================

if "X" not in data:
    raise KeyError(
        "Dataset does not contain 'X'."
    )

if "y" not in data:
    raise KeyError(
        "Dataset does not contain 'y'."
    )


X = data["X"]
y = data["y"]


if "recording_ids" in data:
    recording_ids = data["recording_ids"]
else:
    recording_ids = np.arange(len(y))


# ============================================================
# FORCE NUMERICAL TYPES
# ============================================================

try:
    X = np.asarray(
        X,
        dtype=np.float64
    )
except Exception as e:
    raise TypeError(
        f"\nCould not convert X to float64.\n"
        f"Original error: {e}"
    )


y = np.asarray(y)


# ============================================================
# DATASET SHAPE
# ============================================================

print()
print("=" * 70)
print("DATASET SHAPE")
print("=" * 70)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

if X.ndim != 2:
    raise ValueError(
        f"X must be 2-dimensional.\n"
        f"Received shape: {X.shape}"
    )

if y.ndim != 1:
    raise ValueError(
        f"y must be 1-dimensional.\n"
        f"Received shape: {y.shape}"
    )

if X.shape[0] != y.shape[0]:
    raise ValueError(
        "Number of X samples does not match "
        "number of y labels."
    )

print(f"Samples: {X.shape[0]}")
print(f"Features: {X.shape[1]}")


# ============================================================
# FEATURE COUNT CHECK
# ============================================================

print()
print("=" * 70)
print("FEATURE COUNT CHECK")
print("=" * 70)

if X.shape[1] != EXPECTED_FEATURES:
    raise ValueError(
        f"\nExpected {EXPECTED_FEATURES} features "
        f"but received {X.shape[1]}."
    )

print(
    f"Feature count is correct: "
    f"{EXPECTED_FEATURES}"
)


# ============================================================
# INITIAL NUMERICAL CHECK
# ============================================================

print()
print("=" * 70)
print("INITIAL NUMERICAL CHECK")
print("=" * 70)

nan_count = int(
    np.isnan(X).sum()
)

inf_count = int(
    np.isinf(X).sum()
)

print(f"NaN values:      {nan_count}")
print(f"Infinite values: {inf_count}")

if nan_count > 0:
    raise ValueError(
        "X contains NaN values."
    )

if inf_count > 0:
    raise ValueError(
        "X contains infinite values."
    )


print(
    f"Minimum feature value: "
    f"{np.min(X):.6g}"
)

print(
    f"Maximum feature value: "
    f"{np.max(X):.6g}"
)

print(
    f"Mean absolute feature value: "
    f"{np.mean(np.abs(X)):.6g}"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

unique_classes, counts = np.unique(
    y,
    return_counts=True
)

for cls, count in zip(
    unique_classes,
    counts
):
    print(
        f"{str(cls):12s}: "
        f"{count:4d} samples"
    )

print()
print(
    f"Number of classes: "
    f"{len(unique_classes)}"
)


# ============================================================
# VERIFY EXACT VOCABULARY
# ============================================================

missing_classes = [
    cls
    for cls in TARGET_CLASSES
    if cls not in unique_classes
]

unexpected_classes = [
    cls
    for cls in unique_classes
    if cls not in TARGET_CLASSES
]


if missing_classes:
    raise ValueError(
        f"\nMissing target classes: "
        f"{missing_classes}"
    )

if unexpected_classes:
    raise ValueError(
        f"\nUnexpected classes found: "
        f"{unexpected_classes}"
    )

if len(unique_classes) != 5:
    raise ValueError(
        "Expected exactly 5 classes."
    )

print()
print("Vocabulary check passed.")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print()
print("=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(f"Test size: {TEST_SIZE * 100:.0f}%")
print(f"Random state: {RANDOM_STATE}")
print("Stratification: ON")

(
    X_train,
    X_test,
    y_train,
    y_test,
    id_train,
    id_test
) = train_test_split(
    X,
    y,
    recording_ids,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


print()
print(
    f"Training samples: "
    f"{len(X_train)}"
)

print(
    f"Testing samples:  "
    f"{len(X_test)}"
)


# ============================================================
# TRAINING DISTRIBUTION
# ============================================================

print()
print("Training class distribution:")

train_classes, train_counts = np.unique(
    y_train,
    return_counts=True
)

for cls, count in zip(
    train_classes,
    train_counts
):
    print(
        f"  {str(cls):12s}: "
        f"{count:4d}"
    )


# ============================================================
# TEST DISTRIBUTION
# ============================================================

print()
print("Testing class distribution:")

test_classes, test_counts = np.unique(
    y_test,
    return_counts=True
)

for cls, count in zip(
    test_classes,
    test_counts
):
    print(
        f"  {str(cls):12s}: "
        f"{count:4d}"
    )


# ============================================================
# CHECK TRAINING DATA NUMERICS
# ============================================================

print()
print("=" * 70)
print("TRAINING DATA NUMERICAL CHECK")
print("=" * 70)

if not np.isfinite(X_train).all():
    raise ValueError(
        "Training data contains NaN or infinite values."
    )

if not np.isfinite(X_test).all():
    raise ValueError(
        "Testing data contains NaN or infinite values."
    )

print("Training data: finite")
print("Testing data:  finite")


# ============================================================
# TRAINING-ONLY PERCENTILE CLIPPING
# ============================================================
#
# IMPORTANT:
#
# The clipping thresholds are calculated ONLY from X_train.
#
# The test set is not used to determine these thresholds.
#
# This prevents test-set information from leaking into
# preprocessing.
# ============================================================

print()
print("=" * 70)
print("TRAINING-ONLY FEATURE CLIPPING")
print("=" * 70)

print(
    f"Lower percentile: "
    f"{CLIP_LOW_PERCENTILE}%"
)

print(
    f"Upper percentile: "
    f"{CLIP_HIGH_PERCENTILE}%"
)


# Calculate one lower and upper limit per feature.

lower_limits = np.percentile(
    X_train,
    CLIP_LOW_PERCENTILE,
    axis=0
)

upper_limits = np.percentile(
    X_train,
    CLIP_HIGH_PERCENTILE,
    axis=0
)


# Make absolutely sure the limits are finite.

if not np.isfinite(
    lower_limits
).all():

    raise ValueError(
        "Lower clipping limits contain "
        "NaN or infinity."
    )


if not np.isfinite(
    upper_limits
).all():

    raise ValueError(
        "Upper clipping limits contain "
        "NaN or infinity."
    )


print()
print(
    "Training-derived clipping limits calculated."
)


# ============================================================
# APPLY CLIPPING
# ============================================================

X_train_clipped = np.clip(
    X_train,
    lower_limits,
    upper_limits
)

X_test_clipped = np.clip(
    X_test,
    lower_limits,
    upper_limits
)


# ============================================================
# CHECK AFTER CLIPPING
# ============================================================

print()
print("=" * 70)
print("POST-CLIPPING NUMERICAL CHECK")
print("=" * 70)

if not np.isfinite(
    X_train_clipped
).all():

    raise ValueError(
        "Training data became invalid after clipping."
    )


if not np.isfinite(
    X_test_clipped
).all():

    raise ValueError(
        "Testing data became invalid after clipping."
    )


print("Training data: finite")
print("Testing data:  finite")

print()
print(
    f"Training minimum: "
    f"{np.min(X_train_clipped):.6g}"
)

print(
    f"Training maximum: "
    f"{np.max(X_train_clipped):.6g}"
)

print(
    f"Testing minimum:  "
    f"{np.min(X_test_clipped):.6g}"
)

print(
    f"Testing maximum:  "
    f"{np.max(X_test_clipped):.6g}"
)


# ============================================================
# BUILD MLP PIPELINE
# ============================================================

print()
print("=" * 70)
print("BUILDING MLP")
print("=" * 70)

print()
print("Preprocessing:")
print("  1. Training-only percentile clipping")
print("  2. StandardScaler")

print()
print("MLP:")
print("  Input features: 112")
print("  Hidden layers: (32, 16)")
print("  Output classes: 5")

print()
print("Activation: ReLU")
print("Solver: Adam")
print("Learning rate: 0.001")
print("L2 regularization: 0.01")
print("Max iterations: 500")
print("Batch size: 32")

print()
print("Early stopping: OFF")
print("Class weighting: OFF")


# ============================================================
# CREATE MLP
# ============================================================

mlp = MLPClassifier(
    hidden_layer_sizes=(32, 16),

    activation="relu",

    solver="adam",

    alpha=0.01,

    learning_rate_init=0.001,

    max_iter=500,

    batch_size=32,

    early_stopping=False,

    tol=1e-4,

    random_state=RANDOM_STATE,

    verbose=True
)


# ============================================================
# CREATE PIPELINE
# ============================================================

model = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),

        (
            "mlp",
            mlp
        )
    ]
)


# ============================================================
# CHECK DATA BEFORE MLP
# ============================================================

print()
print("=" * 70)
print("FINAL PRE-TRAINING CHECK")
print("=" * 70)

print(
    "X_train shape:",
    X_train_clipped.shape
)

print(
    "X_test shape:",
    X_test_clipped.shape
)

print(
    "X_train dtype:",
    X_train_clipped.dtype
)

print(
    "X_test dtype:",
    X_test_clipped.dtype
)

print(
    "X_train finite:",
    np.isfinite(X_train_clipped).all()
)

print(
    "X_test finite:",
    np.isfinite(X_test_clipped).all()
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("TRAINING MLP")
print("=" * 70)

print()
print("Starting training...")

model.fit(
    X_train_clipped,
    y_train
)

print()
print("Training complete.")


# ============================================================
# GET TRAINED MLP
# ============================================================

trained_mlp = model.named_steps["mlp"]


# ============================================================
# TRAINING INFORMATION
# ============================================================

print()
print("=" * 70)
print("TRAINING INFORMATION")
print("=" * 70)

print(
    f"Iterations completed: "
    f"{trained_mlp.n_iter_}"
)

print(
    f"Final training loss: "
    f"{trained_mlp.loss_:.6f}"
)

print(
    f"Number of layers: "
    f"{len(trained_mlp.coefs_) + 1}"
)

print(
    f"Output classes: "
    f"{trained_mlp.classes_}"
)


# ============================================================
# TRAINING PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("TRAINING PERFORMANCE")
print("=" * 70)

y_train_pred = model.predict(
    X_train_clipped
)

train_accuracy = accuracy_score(
    y_train,
    y_train_pred
)

print(
    f"Training accuracy: "
    f"{train_accuracy * 100:.2f}%"
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("TEST PERFORMANCE")
print("=" * 70)

y_test_pred = model.predict(
    X_test_clipped
)

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

print(
    f"Test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# ACCURACY GAP
# ============================================================

print()
print("=" * 70)
print("OVERFITTING CHECK")
print("=" * 70)

accuracy_gap = (
    train_accuracy -
    test_accuracy
)

print(
    f"Training accuracy: "
    f"{train_accuracy * 100:.2f}%"
)

print(
    f"Test accuracy:     "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Accuracy gap:      "
    f"{accuracy_gap * 100:.2f}%"
)


if accuracy_gap > 0.30:

    print()
    print(
        "WARNING: Very large training/test gap."
    )

    print(
        "The MLP is likely overfitting."
    )

elif accuracy_gap > 0.15:

    print()
    print(
        "Moderate training/test gap detected."
    )

    print(
        "Some overfitting may be present."
    )

else:

    print()
    print(
        "Training/test gap is relatively small."
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print()

print(
    classification_report(
        y_test,
        y_test_pred,
        labels=TARGET_CLASSES,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    y_test_pred,
    labels=TARGET_CLASSES
)


print()
print("Actual \\ Predicted")

header = (
    f"{'':15s}"
    +
    "".join(
        f"{cls:>12s}"
        for cls in TARGET_CLASSES
    )
)

print(header)


for i, cls in enumerate(
    TARGET_CLASSES
):

    row = "".join(
        f"{cm[i, j]:12d}"
        for j in range(
            len(TARGET_CLASSES)
        )
    )

    print(
        f"{cls:15s}{row}"
    )


# ============================================================
# PER-CLASS RESULTS
# ============================================================

print()
print("=" * 70)
print("PER-CLASS TEST RESULTS")
print("=" * 70)

for i, cls in enumerate(
    TARGET_CLASSES
):

    total = int(
        np.sum(cm[i])
    )

    correct = int(
        cm[i, i]
    )

    if total > 0:

        class_accuracy = (
            correct / total
        )

    else:

        class_accuracy = 0.0

    print(
        f"{cls:12s}: "
        f"{correct:3d}/{total:3d} "
        f"({class_accuracy * 100:6.2f}%)"
    )


# ============================================================
# SAVE MODEL
# ============================================================
#
# We save:
#
#   - trained sklearn pipeline
#   - classes
#   - feature count
#   - clipping limits
#   - configuration
#
# This allows us to reproduce preprocessing later.
# ============================================================

print()
print("=" * 70)
print("SAVING MODEL")
print("=" * 70)


model_package = {

    "model": model,

    "classes": TARGET_CLASSES,

    "feature_count": EXPECTED_FEATURES,

    "clip_low_percentile":
        CLIP_LOW_PERCENTILE,

    "clip_high_percentile":
        CLIP_HIGH_PERCENTILE,

    "lower_limits":
        lower_limits,

    "upper_limits":
        upper_limits,

    "random_state":
        RANDOM_STATE
}


joblib.dump(
    model_package,
    MODEL_PATH
)


print()
print("Model saved to:")
print(MODEL_PATH)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("PROTO 1.7 COMPLETE")
print("=" * 70)

print()

print("Dataset:")
print(
    f"  Samples:              "
    f"{len(X)}"
)

print(
    f"  Features:             "
    f"{X.shape[1]}"
)

print(
    f"  Classes:              "
    f"{len(TARGET_CLASSES)}"
)


print()

print("Split:")
print(
    f"  Training:             "
    f"{len(X_train)}"
)

print(
    f"  Testing:              "
    f"{len(X_test)}"
)


print()

print("Performance:")

print(
    f"  Training accuracy:    "
    f"{train_accuracy * 100:.2f}%"
)

print(
    f"  Test accuracy:        "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"  Accuracy gap:         "
    f"{accuracy_gap * 100:.2f}%"
)


print()

print("Numerical stabilization:")
print("  float64:              YES")
print("  Training-only clip:   YES")
print("  StandardScaler:       YES")
print("  NaN check:            PASSED")
print("  Infinite check:       PASSED")


print()

print("MLP:")
print("  Architecture:         (32, 16)")
print("  Early stopping:       NO")
print("  L2 alpha:             0.01")
print("  Solver:               Adam")


print()

print("=" * 70)
print("IMPORTANT")
print("=" * 70)

print()

print(
    "Proto 1.7 is a numerically stable baseline."
)

print()

print(
    "The test set is still a random 20% split."
)

print()

print(
    "Therefore, this result does NOT yet prove "
    "generalization to completely unseen subjects."
)

print()

print(
    "The next major experiment should be:"
)

print(
    "  Proto 1.8 -> SUBJECT-INDEPENDENT TESTING"
)

print()

print("=" * 70)
print("DONE")
print("=" * 70)

