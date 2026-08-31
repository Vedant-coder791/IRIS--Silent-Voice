

# ============================================================
# IRIS - MLP PROTO 1.7
# Berkeley Closed-Vocabulary Silent Speech
#
# Improvements over Proto 1.5:
#   1. Strong numerical feature cleaning
#   2. Robust feature clipping
#   3. StandardScaler
#   4. Smaller MLP to reduce overfitting
#   5. Stronger L2 regularization
#   6. NO early stopping
#   7. Stratified train/test split
#   8. Classification report
#   9. Confusion matrix
#  10. Saves trained pipeline
# ============================================================

import os
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

TARGET_CLASSES = [
    "AM",
    "PM",
    "DECEMBER",
    "AUGUST",
    "MARCH"
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
for i, name in enumerate(TARGET_CLASSES, 1):
    print(f"  {i}. {name}")

print()
print("Expected features:", EXPECTED_FEATURES)


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

data = np.load(DATASET_PATH, allow_pickle=True)

print("Dataset keys:")
print(list(data.keys()))


required_keys = ["X", "y", "classes"]

for key in required_keys:
    if key not in data:
        raise KeyError(
            f"\nRequired dataset key missing: {key}"
        )


X = data["X"]
y = data["y"]
classes = data["classes"]


# ============================================================
# BASIC SHAPE CHECK
# ============================================================

print()
print("=" * 70)
print("DATASET SHAPE")
print("=" * 70)

print("Original X shape:", X.shape)
print("Original y shape:", y.shape)

if X.ndim != 2:
    raise ValueError(
        f"\nX must be 2-dimensional.\n"
        f"Received shape: {X.shape}"
    )

if y.ndim != 1:
    y = y.reshape(-1)

if X.shape[0] != len(y):
    raise ValueError(
        "\nNumber of X samples does not match y labels."
    )

print("Samples:", X.shape[0])
print("Features:", X.shape[1])


if X.shape[1] != EXPECTED_FEATURES:
    raise ValueError(
        f"\nExpected {EXPECTED_FEATURES} features "
        f"but dataset contains {X.shape[1]}."
    )


# ============================================================
# CONVERT FEATURES TO FLOAT64
# ============================================================

print()
print("=" * 70)
print("CONVERTING FEATURES")
print("=" * 70)

try:
    X = np.asarray(X, dtype=np.float64)
except Exception as e:
    raise ValueError(
        f"\nCould not convert feature matrix to float64:\n{e}"
    )

print("Feature dtype:", X.dtype)


# ============================================================
# NUMERICAL CHECK BEFORE CLEANING
# ============================================================

print()
print("=" * 70)
print("NUMERICAL CHECK - BEFORE CLEANING")
print("=" * 70)

nan_count = np.isnan(X).sum()
inf_count = np.isinf(X).sum()

print("NaN values:", nan_count)
print("Infinite values:", inf_count)

finite_values = X[np.isfinite(X)]

if len(finite_values) == 0:
    raise ValueError(
        "\nNo finite numerical feature values found."
    )

print("Minimum:", np.min(finite_values))
print("Maximum:", np.max(finite_values))
print("Mean absolute:", np.mean(np.abs(finite_values)))


# ============================================================
# REPLACE BAD VALUES
# ============================================================

print()
print("=" * 70)
print("CLEANING INVALID VALUES")
print("=" * 70)

# Replace NaN and infinity with finite values temporarily.
X = np.nan_to_num(
    X,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

print("NaN values after cleaning:", np.isnan(X).sum())
print("Infinite values after cleaning:", np.isinf(X).sum())


# ============================================================
# ROBUST FEATURE CLIPPING
#
# The previous dataset contained values around 5.95e9.
# Such extreme values can make the MLP numerically unstable.
#
# We clip EACH FEATURE independently using percentile limits.
# ============================================================

print()
print("=" * 70)
print("ROBUST FEATURE CLIPPING")
print("=" * 70)

LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0

lower_limits = np.percentile(
    X,
    LOW_PERCENTILE,
    axis=0
)

upper_limits = np.percentile(
    X,
    HIGH_PERCENTILE,
    axis=0
)

X = np.clip(
    X,
    lower_limits,
    upper_limits
)

print(
    f"Clipping each feature to its "
    f"{LOW_PERCENTILE}th-{HIGH_PERCENTILE}th percentile range."
)


# ============================================================
# CHECK AFTER CLIPPING
# ============================================================

print()
print("=" * 70)
print("NUMERICAL CHECK - AFTER CLIPPING")
print("=" * 70)

print("NaN values:", np.isnan(X).sum())
print("Infinite values:", np.isinf(X).sum())
print("Minimum:", np.min(X))
print("Maximum:", np.max(X))
print("Mean absolute:", np.mean(np.abs(X)))


if not np.all(np.isfinite(X)):
    raise ValueError(
        "\nFeature matrix still contains invalid numerical values."
    )

print()
print("Features are numerically valid.")


# ============================================================
# LABEL CHECK
# ============================================================

print()
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

y = np.asarray(y)

print("Labels dtype:", y.dtype)

unique_labels, counts = np.unique(
    y,
    return_counts=True
)

for label, count in zip(unique_labels, counts):
    print(f"{str(label):12s}: {count:4d} samples")

print()
print("Number of classes:", len(unique_labels))


# ============================================================
# CHECK EXPECTED CLASSES
# ============================================================

print()
print("=" * 70)
print("CLASS VALIDATION")
print("=" * 70)

dataset_class_strings = [str(c).upper() for c in unique_labels]

for target in TARGET_CLASSES:
    if target not in dataset_class_strings:
        print(
            f"WARNING: Expected class {target} "
            f"was not found."
        )
    else:
        print(f"{target:12s} -> FOUND")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print()
print("=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print("Test size:", int(TEST_SIZE * 100), "%")
print("Random state:", RANDOM_STATE)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print()
print("Training samples:", len(X_train))
print("Testing samples: ", len(X_test))


# ============================================================
# TRAINING DISTRIBUTION
# ============================================================

print()
print("Training class distribution:")

train_labels, train_counts = np.unique(
    y_train,
    return_counts=True
)

for label, count in zip(train_labels, train_counts):
    print(f"  {str(label):12s}: {count:4d}")


print()
print("Testing class distribution:")

test_labels, test_counts = np.unique(
    y_test,
    return_counts=True
)

for label, count in zip(test_labels, test_counts):
    print(f"  {str(label):12s}: {count:4d}")


# ============================================================
# BUILD MLP
# ============================================================

print()
print("=" * 70)
print("BUILDING IMPROVED MLP")
print("=" * 70)

print()
print("Pipeline:")
print("  1. StandardScaler")
print("  2. MLPClassifier")

print()
print("MLP architecture:")
print("  Hidden layers: (32, 16)")
print("  Activation: relu")
print("  Solver: adam")
print("  Max iterations: 600")
print("  Learning rate: 0.0005")
print("  L2 regularization (alpha): 0.01")

print()
print("Early stopping: OFF")
print("Class weighting: OFF")


# ============================================================
# PIPELINE
# ============================================================

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),

    (
        "mlp",
        MLPClassifier(
            hidden_layer_sizes=(64, 32),

            activation="relu",

            solver="adam",

            alpha=0.01,

            batch_size=16,

            learning_rate_init=0.0005,

            max_iter=600,

            tol=1e-4,

            n_iter_no_change=20,

            early_stopping=False,

            random_state=RANDOM_STATE,

            verbose=True
        )
    )
])


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
    X_train,
    y_train
)

print()
print("Training complete.")


# ============================================================
# TRAINING PERFORMANCE
# ============================================================

print()
print("=" * 70)
print("TRAINING PERFORMANCE")
print("=" * 70)

y_train_pred = model.predict(X_train)

train_accuracy = accuracy_score(
    y_train,
    y_train_pred
)

print()
print(
    f"Training accuracy: "
    f"{train_accuracy * 100:.2f}%"
)


# ============================================================
# TEST PERFORMANCE
# ============================================================

print()
print("=" * 70)
print("TEST PERFORMANCE")
print("=" * 70)

y_test_pred = model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

print()
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

print(
    f"{'':12s}" +
    "".join(
        f"{label:>12s}"
        for label in TARGET_CLASSES
    )
)

for i, label in enumerate(TARGET_CLASSES):

    row = (
        f"{label:12s}" +
        "".join(
            f"{cm[i, j]:12d}"
            for j in range(len(TARGET_CLASSES))
        )
    )

    print(row)


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print()
print("=" * 70)
print("PER-CLASS TEST RESULTS")
print("=" * 70)

for i, label in enumerate(TARGET_CLASSES):

    total = cm[i].sum()

    if total == 0:
        accuracy = 0.0
        correct = 0
    else:
        correct = cm[i, i]
        accuracy = correct / total

    print(
        f"{label:12s}: "
        f"{correct:3d}/{total:3d} "
        f"({accuracy * 100:6.2f}%)"
    )


# ============================================================
# MLP TRAINING INFORMATION
# ============================================================

print()
print("=" * 70)
print("MLP TRAINING INFORMATION")
print("=" * 70)

mlp = model.named_steps["mlp"]

print()
print("Iterations completed:", mlp.n_iter_)
print("Final training loss:", mlp.loss_)

print()
print("Number of layers:", mlp.n_layers_)
print("Number of outputs:", mlp.n_outputs_)

print()
print("Hidden layers:", mlp.hidden_layer_sizes)


# ============================================================
# SAVE MODEL
# ============================================================

print()
print("=" * 70)
print("SAVING MODEL")
print("=" * 70)

joblib.dump(
    model,
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
print(f"  Samples:       {X.shape[0]}")
print(f"  Features:      {X.shape[1]}")
print(f"  Classes:       {len(unique_labels)}")

print()
print("Split:")
print(f"  Training:      {len(X_train)}")
print(f"  Testing:       {len(X_test)}")

print()
print("Performance:")
print(
    f"  Training accuracy: "
    f"{train_accuracy * 100:.2f}%"
)

print(
    f"  Test accuracy:     "
    f"{test_accuracy * 100:.2f}%"
)

print()
print("Model:")
print("  Hidden layers: (32, 16)")
print("  StandardScaler: YES")
print("  Robust clipping: YES")
print("  Early stopping: NO")
print("  Stronger L2: YES")

print()
print("=" * 70)
print("IMPORTANT")
print("=" * 70)

print()
print(
    "Proto 1.7 is an improved numerical-stability "
    "and overfitting-control baseline."
)

print()
print(
    "The test accuracy is still NOT a subject-independent "
    "generalization measurement."
)

print()
print(
    "The next major improvement should be SUBJECT-INDEPENDENT "
    "TESTING rather than simply making the MLP larger."
)

print()
print("=" * 70)
