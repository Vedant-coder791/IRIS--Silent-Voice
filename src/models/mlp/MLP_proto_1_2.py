# ============================================================
# IRIS - MLP PROTO 1.5
# Berkeley Closed-Vocabulary Silent Speech
#
# PURPOSE:
#   Train the first actual MLP classifier using the
#   Proto 1.4 feature dataset.
#
# IMPORTANT:
#   This is a BASELINE prototype.
#   No early stopping.
#   No class weighting.
#   No subject-independent split yet.
# ============================================================

import os
import numpy as np

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
    "MLP_proto1_5_model.pkl"
)

RANDOM_STATE = 42

TEST_SIZE = 0.20

# MLP architecture
HIDDEN_LAYERS = (64, 32)

MAX_ITER = 500

LEARNING_RATE_INIT = 0.001

ALPHA = 0.0001


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("IRIS - MLP PROTO 1.5")
print("Berkeley Closed-Vocabulary Silent Speech")
print("=" * 70)

print()
print("Dataset:")
print(DATASET_PATH)

print()
print("Target vocabulary:")
print("  1. AM")
print("  2. PM")
print("  3. DECEMBER")
print("  4. AUGUST")
print("  5. MARCH")


# ============================================================
# CHECK DATASET
# ============================================================

print()
print("=" * 70)
print("CHECKING DATASET")
print("=" * 70)

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_PATH}\n\n"
        "Run MLP Proto 1.4 first."
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
print(data.files)

if "X" not in data:
    raise KeyError(
        "Dataset does not contain 'X'. "
        "Check the Proto 1.4 dataset."
    )

if "y" not in data:
    raise KeyError(
        "Dataset does not contain 'y'. "
        "Check the Proto 1.4 dataset."
    )

X = data["X"]
y = data["y"]


# ============================================================
# BASIC SHAPE CHECK
# ============================================================

print()
print("=" * 70)
print("DATASET SHAPE")
print("=" * 70)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

if X.ndim != 2:
    raise ValueError(
        f"X must be 2-dimensional. Got shape {X.shape}"
    )

if y.ndim != 1:
    y = y.reshape(-1)

if len(X) != len(y):
    raise ValueError(
        f"X and y have different numbers of samples: "
        f"{len(X)} vs {len(y)}"
    )

print(f"Samples: {X.shape[0]}")
print(f"Features: {X.shape[1]}")


# ============================================================
# NUMERICAL SAFETY CHECK
# ============================================================

print()
print("=" * 70)
print("NUMERICAL FEATURE CHECK")
print("=" * 70)

nan_count = np.isnan(X).sum()
inf_count = np.isinf(X).sum()

print(f"NaN values: {nan_count}")
print(f"Infinite values: {inf_count}")

if nan_count > 0:
    raise ValueError("X contains NaN values.")

if inf_count > 0:
    raise ValueError("X contains infinite values.")

# Force floating-point representation
X = X.astype(np.float64)

print(f"Minimum feature value: {np.min(X):.6g}")
print(f"Maximum feature value: {np.max(X):.6g}")
print(f"Mean absolute feature value: {np.mean(np.abs(X)):.6g}")

print()
print("Features are numerically valid.")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

classes, counts = np.unique(y, return_counts=True)

for cls, count in zip(classes, counts):
    print(f"{str(cls):12s}: {count:4d}")

print()
print(f"Number of classes: {len(classes)}")

if len(classes) < 2:
    raise ValueError(
        "MLP requires at least two classes."
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print()
print("=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(f"Test size: {TEST_SIZE * 100:.0f}%")
print(f"Random state: {RANDOM_STATE}")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print()
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# ============================================================
# CHECK SPLIT CLASS DISTRIBUTION
# ============================================================

print()
print("Training class distribution:")

train_classes, train_counts = np.unique(
    y_train,
    return_counts=True
)

for cls, count in zip(train_classes, train_counts):
    print(f"  {str(cls):12s}: {count:4d}")

print()
print("Testing class distribution:")

test_classes, test_counts = np.unique(
    y_test,
    return_counts=True
)

for cls, count in zip(test_classes, test_counts):
    print(f"  {str(cls):12s}: {count:4d}")


# ============================================================
# BUILD MLP PIPELINE
# ============================================================

print()
print("=" * 70)
print("BUILDING MLP")
print("=" * 70)

print()
print("Pipeline:")
print("  1. StandardScaler")
print("  2. MLPClassifier")

print()
print("MLP architecture:")
print(f"  Hidden layers: {HIDDEN_LAYERS}")
print(f"  Max iterations: {MAX_ITER}")
print(f"  Learning rate: {LEARNING_RATE_INIT}")
print(f"  L2 regularization (alpha): {ALPHA}")

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
            hidden_layer_sizes=HIDDEN_LAYERS,

            activation="relu",

            solver="adam",

            alpha=ALPHA,

            learning_rate_init=LEARNING_RATE_INIT,

            max_iter=MAX_ITER,

            random_state=RANDOM_STATE,

            early_stopping=False,

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
print()

model.fit(X_train, y_train)

print()
print("Training complete.")


# ============================================================
# TRAINING ACCURACY
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
print(f"Training accuracy: {train_accuracy * 100:.2f}%")


# ============================================================
# TEST PREDICTION
# ============================================================

print()
print("=" * 70)
print("TEST PERFORMANCE")
print("=" * 70)

y_pred = model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    y_pred
)

print()
print(f"Test accuracy: {test_accuracy * 100:.2f}%")


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
        y_pred,
        labels=classes,
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
    y_pred,
    labels=classes
)

print()

# Header
print("Actual \\ Predicted")

print(
    f"{'':15s}"
    + "".join(
        f"{str(cls):>12s}"
        for cls in classes
    )
)

for i, cls in enumerate(classes):

    row = f"{str(cls):15s}"

    for value in cm[i]:
        row += f"{value:12d}"

    print(row)


# ============================================================
# PER-CLASS TEST ACCURACY
# ============================================================

print()
print("=" * 70)
print("PER-CLASS TEST RESULTS")
print("=" * 70)

for i, cls in enumerate(classes):

    total = np.sum(cm[i])

    correct = cm[i, i]

    if total > 0:
        accuracy = correct / total
    else:
        accuracy = 0.0

    print(
        f"{str(cls):12s}: "
        f"{correct:3d}/{total:3d} "
        f"({accuracy * 100:6.2f}%)"
    )


# ============================================================
# SAVE MODEL
# ============================================================

print()
print("=" * 70)
print("SAVING MODEL")
print("=" * 70)

import joblib

joblib.dump(model, MODEL_PATH)

print()
print(f"Model saved to:")
print(MODEL_PATH)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("PROTO 1.5 COMPLETE")
print("=" * 70)

print()
print("Dataset:")
print(f"  Samples:       {len(X)}")
print(f"  Features:      {X.shape[1]}")
print(f"  Classes:       {len(classes)}")

print()
print("Split:")
print(f"  Training:      {len(X_train)}")
print(f"  Testing:       {len(X_test)}")

print()
print("Performance:")
print(f"  Training accuracy: {train_accuracy * 100:.2f}%")
print(f"  Test accuracy:     {test_accuracy * 100:.2f}%")

print()
print("Model:")
print(f"  Hidden layers: {HIDDEN_LAYERS}")
print("  StandardScaler: YES")
print("  Early stopping: NO")
print("  Class weighting: NO")

print()
print("=" * 70)
print("IMPORTANT")
print("=" * 70)

print()
print("This is a BASELINE MLP.")
print()
print("A good test accuracy here does NOT yet prove")
print("generalization to new people.")
print()
print("Proto 1.5 is only testing whether the extracted")
print("features contain enough information for the MLP")
print("to distinguish the five target words.")
print()
print("Next improvements can include:")
print("  - subject-independent testing")
print("  - better feature extraction")
print("  - class balancing")
print("  - hyperparameter tuning")
print("  - cross-validation")
print("  - confusion-matrix analysis")
print("  - testing on completely unseen recordings")
print()
print("=" * 70)