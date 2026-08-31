
# ============================================================
# IRIS - MLP PROTO 1.9
# Berkeley Closed-Vocabulary Silent Speech
#
# NUMERICAL-STABILITY + OVERFITTING CONTROL VERSION
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
#
# IMPORTANT:
#   Early stopping is intentionally DISABLED.
#   This avoids the sklearn validation/prediction bug seen
#   in Proto 1.7 / 1.8.
# ============================================================

import os
import sys
import warnings
import pickle

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "/Users/vedantdwivedi/Desktop/IRIS/AI/MLP_proto1_4_dataset.npz"

MODEL_PATH = "/Users/vedantdwivedi/Desktop/IRIS/AI/MLP_proto1_9_model.pkl"

RANDOM_STATE = 42

TEST_SIZE = 0.20

EXPECTED_FEATURES = 112

TARGET_CLASSES = [
    "AM",
    "PM",
    "DECEMBER",
    "AUGUST",
    "MARCH"
]

# ============================================================
# MLP SETTINGS
# ============================================================

HIDDEN_LAYERS = (64, 32)

ACTIVATION = "relu"

SOLVER = "adam"

LEARNING_RATE_INIT = 0.0005

MAX_ITER = 800

ALPHA = 0.02

BATCH_SIZE = 32

# VERY IMPORTANT
EARLY_STOPPING = False

VERBOSE = True


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 70)
print("IRIS - MLP PROTO 1.9")
print("Berkeley Closed-Vocabulary Silent Speech")
print("=" * 70)

print()
print("Dataset:")
print(DATASET_PATH)

print()
print("Target vocabulary:")

for i, word in enumerate(TARGET_CLASSES, 1):
    print(f"  {i}. {word}")

print()
print(f"Expected features: {EXPECTED_FEATURES}")


# ============================================================
# CHECK DATASET
# ============================================================

print()
print("=" * 70)
print("CHECKING DATASET")
print("=" * 70)

if not os.path.exists(DATASET_PATH):

    print()
    print("ERROR: Dataset not found.")
    print(DATASET_PATH)
    sys.exit(1)

print("Dataset found.")


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("=" * 70)
print("LOADING FEATURE DATASET")
print("=" * 70)

try:

    data = np.load(
        DATASET_PATH,
        allow_pickle=True
    )

except Exception as e:

    print()
    print("ERROR loading dataset:")
    print(e)
    sys.exit(1)


print("Dataset keys:")
print(list(data.keys()))


# ============================================================
# CHECK REQUIRED KEYS
# ============================================================

required_keys = ["X", "y"]

missing = [
    key for key in required_keys
    if key not in data
]

if missing:

    print()
    print("ERROR:")
    print("Missing dataset keys:")
    print(missing)
    sys.exit(1)


# ============================================================
# LOAD X AND y
# ============================================================

X = data["X"]

y = data["y"]


# ============================================================
# DATASET SHAPE
# ============================================================

print()
print("=" * 70)
print("DATASET SHAPE")
print("=" * 70)

print("Original X shape:", X.shape)
print("Original y shape:", y.shape)

if X.ndim != 2:

    print()
    print("ERROR: X must be a 2D matrix.")
    print("Received dimensions:", X.ndim)
    sys.exit(1)

if y.ndim != 1:

    y = np.asarray(y).reshape(-1)


n_samples, n_features = X.shape

print(f"Samples: {n_samples}")
print(f"Features: {n_features}")


# ============================================================
# FEATURE COUNT CHECK
# ============================================================

if n_features != EXPECTED_FEATURES:

    print()
    print("WARNING:")
    print(
        f"Expected {EXPECTED_FEATURES} features "
        f"but dataset contains {n_features}."
    )

    print()
    print("Continuing using the actual dataset feature count.")


# ============================================================
# CONVERT FEATURES
# ============================================================

print()
print("=" * 70)
print("CONVERTING FEATURES")
print("=" * 70)

try:

    X = np.asarray(
        X,
        dtype=np.float64
    )

except Exception as e:

    print()
    print("ERROR converting X to float64:")
    print(e)
    sys.exit(1)


print("Feature dtype:", X.dtype)


# ============================================================
# CONVERT LABELS
# ============================================================

y = np.asarray(y)

if y.shape[0] != X.shape[0]:

    print()
    print("ERROR:")
    print("Number of X samples does not match number of labels.")

    print("X samples:", X.shape[0])
    print("y samples:", y.shape[0])

    sys.exit(1)


# ============================================================
# NUMERICAL CHECK - ORIGINAL
# ============================================================

print()
print("=" * 70)
print("NUMERICAL CHECK - BEFORE CLEANING")
print("=" * 70)

nan_count = np.isnan(X).sum()

inf_count = np.isinf(X).sum()

finite_mask = np.isfinite(X)

if finite_mask.any():

    finite_values = X[finite_mask]

    print("NaN values:", nan_count)
    print("Infinite values:", inf_count)

    print(
        "Minimum:",
        np.min(finite_values)
    )

    print(
        "Maximum:",
        np.max(finite_values)
    )

    print(
        "Mean absolute:",
        np.mean(np.abs(finite_values))
    )

else:

    print("No finite values found.")

    sys.exit(1)


# ============================================================
# CLEAN NaN / INF
# ============================================================

print()
print("=" * 70)
print("CLEANING INVALID VALUES")
print("=" * 70)

X = np.nan_to_num(
    X,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

print(
    "NaN values after cleaning:",
    np.isnan(X).sum()
)

print(
    "Infinite values after cleaning:",
    np.isinf(X).sum()
)


# ============================================================
# ROBUST EXTREME-VALUE HANDLING
# ============================================================

print()
print("=" * 70)
print("ROBUST FEATURE NORMALIZATION")
print("=" * 70)

print("Applying per-feature percentile clipping.")

# ------------------------------------------------------------
# IMPORTANT:
# Do NOT clip the whole matrix using one global percentile.
#
# Each feature gets its own limits.
# ------------------------------------------------------------

LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0

for feature_index in range(X.shape[1]):

    column = X[:, feature_index]

    low = np.percentile(
        column,
        LOW_PERCENTILE
    )

    high = np.percentile(
        column,
        HIGH_PERCENTILE
    )

    if np.isfinite(low) and np.isfinite(high):

        if low < high:

            X[:, feature_index] = np.clip(
                column,
                low,
                high
            )


# ============================================================
# SECOND SAFETY LIMIT
# ============================================================

print()
print("Applying global numerical safety limit.")

# ------------------------------------------------------------
# This prevents exceptionally large values from reaching
# matrix multiplication inside sklearn.
# ------------------------------------------------------------

SAFETY_LIMIT = 1e6

X = np.clip(
    X,
    -SAFETY_LIMIT,
    SAFETY_LIMIT
)


# ============================================================
# NUMERICAL CHECK - AFTER CLEANING
# ============================================================

print()
print("=" * 70)
print("NUMERICAL CHECK - AFTER CLEANING")
print("=" * 70)

print("NaN values:", np.isnan(X).sum())

print("Infinite values:", np.isinf(X).sum())

print("Minimum:", np.min(X))

print("Maximum:", np.max(X))

print(
    "Mean absolute:",
    np.mean(np.abs(X))
)


# ============================================================
# FINAL RAW DATA SAFETY CHECK
# ============================================================

if not np.all(np.isfinite(X)):

    print()
    print("ERROR:")
    print("X still contains invalid numerical values.")

    sys.exit(1)


# ============================================================
# LABEL CLEANING
# ============================================================

print()
print("=" * 70)
print("PROCESSING LABELS")
print("=" * 70)

# Convert labels safely to strings.

y = np.asarray(
    y,
    dtype=str
)

y = np.char.strip(y)

print("Labels dtype:", y.dtype)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

unique_labels, counts = np.unique(
    y,
    return_counts=True
)

for label, count in zip(
    unique_labels,
    counts
):

    print(
        f"{label:<12}: {count:4d} samples"
    )


print()
print("Number of classes:", len(unique_labels))


# ============================================================
# CLASS VALIDATION
# ============================================================

print()
print("=" * 70)
print("CLASS VALIDATION")
print("=" * 70)

missing_classes = []

for target in TARGET_CLASSES:

    if target in unique_labels:

        print(
            f"{target:<12} -> FOUND"
        )

    else:

        print(
            f"{target:<12} -> MISSING"
        )

        missing_classes.append(target)


if missing_classes:

    print()
    print("ERROR:")
    print("The following target classes are missing:")

    for item in missing_classes:
        print(" ", item)

    sys.exit(1)


# ============================================================
# FILTER TO TARGET VOCABULARY
# ============================================================

print()
print("=" * 70)
print("FILTERING TARGET VOCABULARY")
print("=" * 70)

target_mask = np.isin(
    y,
    TARGET_CLASSES
)

X = X[target_mask]

y = y[target_mask]

print(
    "Samples after vocabulary filtering:",
    len(y)
)


# ============================================================
# LABEL ENCODING
# ============================================================

print()
print("=" * 70)
print("ENCODING LABELS")
print("=" * 70)

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("Encoded classes:")

for i, cls in enumerate(
    label_encoder.classes_
):

    print(
        f"  {i} -> {cls}"
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print()
print("=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(
    f"Test size: {TEST_SIZE * 100:.0f} %"
)

print(
    f"Random state: {RANDOM_STATE}"
)


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y_encoded,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y_encoded
)


print()
print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples: ",
    len(X_test)
)


# ============================================================
# TRAINING CLASS DISTRIBUTION
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
        f"  {label_encoder.inverse_transform([cls])[0]:<12}: "
        f"{count:4d}"
    )


# ============================================================
# TEST CLASS DISTRIBUTION
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
        f"  {label_encoder.inverse_transform([cls])[0]:<12}: "
        f"{count:4d}"
    )


# ============================================================
# FINAL TRAINING DATA CHECK
# ============================================================

print()
print("=" * 70)
print("FINAL TRAINING DATA CHECK")
print("=" * 70)

print(
    "X_train shape:",
    X_train.shape
)

print(
    "X_test shape:",
    X_test.shape
)

print(
    "X_train dtype:",
    X_train.dtype
)

print(
    "X_test dtype:",
    X_test.dtype
)


if not np.all(np.isfinite(X_train)):

    print()
    print("ERROR: X_train contains NaN or Inf.")

    sys.exit(1)


if not np.all(np.isfinite(X_test)):

    print()
    print("ERROR: X_test contains NaN or Inf.")

    sys.exit(1)


# ============================================================
# BUILD PIPELINE
# ============================================================

print()
print("=" * 70)
print("BUILDING STABLE MLP")
print("=" * 70)

print()
print("Pipeline:")
print("  1. StandardScaler")
print("  2. MLPClassifier")

print()
print("MLP architecture:")
print(
    "  Hidden layers:",
    HIDDEN_LAYERS
)

print(
    "  Activation:",
    ACTIVATION
)

print(
    "  Solver:",
    SOLVER
)

print(
    "  Max iterations:",
    MAX_ITER
)

print(
    "  Learning rate:",
    LEARNING_RATE_INIT
)

print(
    "  L2 regularization:",
    ALPHA
)

print(
    "  Batch size:",
    BATCH_SIZE
)

print()
print("Early stopping: OFF")


# ============================================================
# CREATE MLP
# ============================================================

mlp = MLPClassifier(

    hidden_layer_sizes=HIDDEN_LAYERS,

    activation=ACTIVATION,

    solver=SOLVER,

    alpha=ALPHA,

    batch_size=BATCH_SIZE,

    learning_rate_init=LEARNING_RATE_INIT,

    max_iter=MAX_ITER,

    shuffle=True,

    random_state=RANDOM_STATE,

    early_stopping=False,

    validation_fraction=0.1,

    n_iter_no_change=20,

    tol=1e-4,

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
# TRAINING
# ============================================================

print()
print("=" * 70)
print("TRAINING MLP")
print("=" * 70)

print()
print("Starting training...")


try:

    model.fit(
        X_train,
        y_train
    )

except KeyboardInterrupt:

    print()
    print()
    print("Training interrupted by user.")

    sys.exit(1)

except Exception as e:

    print()
    print()
    print("=" * 70)
    print("TRAINING ERROR")
    print("=" * 70)

    print(type(e).__name__)
    print(e)

    print()
    print("The model could not be trained.")

    sys.exit(1)


print()
print("Training complete.")


# ============================================================
# EXTRACT MLP
# ============================================================

trained_mlp = model.named_steps["mlp"]


# ============================================================
# TRAINING PERFORMANCE
# ============================================================

print()
print("=" * 70)
print("TRAINING PERFORMANCE")
print("=" * 70)

try:

    train_predictions = model.predict(
        X_train
    )

except Exception as e:

    print("ERROR during training prediction:")
    print(e)
    sys.exit(1)


training_accuracy = accuracy_score(
    y_train,
    train_predictions
)


print(
    f"Training accuracy: "
    f"{training_accuracy * 100:.2f}%"
)


# ============================================================
# TEST PERFORMANCE
# ============================================================

print()
print("=" * 70)
print("TEST PERFORMANCE")
print("=" * 70)

try:

    test_predictions = model.predict(
        X_test
    )

except Exception as e:

    print("ERROR during test prediction:")
    print(e)
    sys.exit(1)


test_accuracy = accuracy_score(
    y_test,
    test_predictions
)


print(
    f"Test accuracy: "
    f"{test_accuracy * 100:.2f}%"
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

        test_predictions,

        labels=np.arange(
            len(label_encoder.classes_)
        ),

        target_names=label_encoder.classes_,

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

    test_predictions,

    labels=np.arange(
        len(label_encoder.classes_)
    )
)


print()

print(
    "Actual \\ Predicted"
)

print(
    f"{'':18}",
    end=""
)

for cls in label_encoder.classes_:

    print(
        f"{cls:>12}",
        end=""
    )

print()

for i, cls in enumerate(
    label_encoder.classes_
):

    print(
        f"{cls:<18}",
        end=""
    )

    for value in cm[i]:

        print(
            f"{value:>12}",
            end=""
        )

    print()


# ============================================================
# PER-CLASS RESULTS
# ============================================================

print()
print("=" * 70)
print("PER-CLASS TEST RESULTS")
print("=" * 70)

for i, cls in enumerate(
    label_encoder.classes_
):

    total = cm[i].sum()

    correct = cm[i, i]

    if total > 0:

        accuracy = (
            correct / total
        ) * 100

    else:

        accuracy = 0.0

    print(
        f"{cls:<12}: "
        f"{correct:3d}/{total:3d} "
        f"({accuracy:6.2f}%)"
    )


# ============================================================
# TRAINING INFORMATION
# ============================================================

print()
print("=" * 70)
print("MLP TRAINING INFORMATION")
print("=" * 70)

print()

print(
    "Iterations completed:",
    trained_mlp.n_iter_
)

print(
    "Final training loss:",
    trained_mlp.loss_
)

print(
    "Number of layers:",
    trained_mlp.n_layers_
)

print(
    "Number of outputs:",
    trained_mlp.n_outputs_
)

print(
    "Hidden layers:",
    trained_mlp.hidden_layer_sizes
)

print(
    "Early stopping:",
    trained_mlp.early_stopping
)


# ============================================================
# LOSS HISTORY
# ============================================================

if hasattr(
    trained_mlp,
    "loss_curve_"
):

    losses = trained_mlp.loss_curve_

    print()
    print(
        "Initial loss:",
        losses[0]
    )

    print(
        "Final loss:",
        losses[-1]
    )

    print(
        "Loss reduction:",
        losses[0] - losses[-1]
    )


# ============================================================
# SAVE MODEL
# ============================================================

print()
print("=" * 70)
print("SAVING MODEL")
print("=" * 70)

model_package = {

    "model": model,

    "label_encoder": label_encoder,

    "classes": label_encoder.classes_,

    "feature_count": X.shape[1],

    "target_vocabulary": TARGET_CLASSES,

    "random_state": RANDOM_STATE,

    "test_accuracy": test_accuracy,

    "training_accuracy": training_accuracy,

    "version": "MLP_PROTO_1.9"

}


try:

    with open(
        MODEL_PATH,
        "wb"
    ) as f:

        pickle.dump(
            model_package,
            f
        )

except Exception as e:

    print()
    print("ERROR saving model:")
    print(e)

    sys.exit(1)


print()
print("Model saved to:")
print(MODEL_PATH)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("PROTO 1.9 COMPLETE")
print("=" * 70)

print()

print("Dataset:")
print(
    f"  Samples:       {X.shape[0]}"
)

print(
    f"  Features:      {X.shape[1]}"
)

print(
    f"  Classes:       {len(label_encoder.classes_)}"
)

print()

print("Split:")
print(
    f"  Training:      {len(X_train)}"
)

print(
    f"  Testing:       {len(X_test)}"
)

print()

print("Performance:")

print(
    f"  Training accuracy: "
    f"{training_accuracy * 100:.2f}%"
)

print(
    f"  Test accuracy:     "
    f"{test_accuracy * 100:.2f}%"
)

print()

print("Model:")

print(
    f"  Hidden layers: {HIDDEN_LAYERS}"
)

print(
    "  StandardScaler: YES"
)

print(
    "  Per-feature clipping: YES"
)

print(
    "  Numerical safety clipping: YES"
)

print(
    "  Early stopping: NO"
)

print(
    f"  L2 regularization: {ALPHA}"
)

print()

print("=" * 70)
print("IMPORTANT")
print("=" * 70)

print()
print(
    "Proto 1.9 focuses on numerical stability."
)

print(
    "The model is evaluated using a fixed train/test split."
)

print(
    "Test accuracy is NOT yet a subject-independent"
)

print(
    "generalization measurement."
)

print()

print(
    "If the test accuracy is still low, the next major"
)

print(
    "improvement should be SUBJECT-INDEPENDENT TESTING"
)

print(
    "and improving the feature dataset rather than simply"
)

print(
    "making the MLP larger."
)

print()
print("=" * 70)

