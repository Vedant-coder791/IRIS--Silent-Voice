import numpy as np
import joblib

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = (
    "/Users/vedantdwivedi/Desktop/IRIS/AI/"
    "proto5_dataset.npz"
)

MODEL_FILE = (
    "/Users/vedantdwivedi/Desktop/IRIS/AI/"
    "MLP_proto5_1_model.pkl"
)


# ============================================================
# WORD CLASSES
# ============================================================

CLASS_NAMES = [
    "AND",
    "ARE",
    "HAS",
    "HE",
    "I",
    "IN",
    "IS",
    "IT",
    "NEW",
    "OF",
    "ON",
    "STATE",
    "THAT",
    "THE",
    "THEY",
    "THIS",
    "TO",
    "WE",
    "WERE",
    "YOU"
]

N_CLASSES = len(CLASS_NAMES)


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("==========================================")
print("LOADING PROTO 5 DATASET")
print("==========================================")

data = np.load(
    DATA_FILE,
    allow_pickle=True
)

print()
print("Available dataset variables:")

for key in data.files:
    print(
        f"  {key:20s}",
        data[key].shape
    )


# ============================================================
# HELPER
# ============================================================

def find_array(data, names):

    for name in names:

        if name in data.files:
            return data[name]

    return None


# ============================================================
# FIND ARRAYS
# ============================================================

X_train = find_array(
    data,
    [
        "X_train",
        "train_X",
        "X_training",
        "training_X"
    ]
)

y_train = find_array(
    data,
    [
        "y_train",
        "train_y",
        "y_training",
        "training_y"
    ]
)

X_val = find_array(
    data,
    [
        "X_val",
        "X_validation",
        "validation_X",
        "val_X"
    ]
)

y_val = find_array(
    data,
    [
        "y_val",
        "y_validation",
        "validation_y",
        "val_y"
    ]
)

X_test = find_array(
    data,
    [
        "X_test",
        "X_testing",
        "testing_X",
        "test_X"
    ]
)

y_test = find_array(
    data,
    [
        "y_test",
        "y_testing",
        "testing_y",
        "test_y"
    ]
)


# ============================================================
# CHECK
# ============================================================

if (
    X_train is None
    or y_train is None
    or X_val is None
    or y_val is None
    or X_test is None
    or y_test is None
):

    print()
    print("ERROR: Could not find all required arrays.")

    print()
    print("Available keys:")

    for key in data.files:
        print(" -", key)

    raise SystemExit


# ============================================================
# CONVERT TYPES
# ============================================================

X_train = np.asarray(
    X_train,
    dtype=np.float32
)

X_val = np.asarray(
    X_val,
    dtype=np.float32
)

X_test = np.asarray(
    X_test,
    dtype=np.float32
)

y_train = np.asarray(
    y_train,
    dtype=np.int64
)

y_val = np.asarray(
    y_val,
    dtype=np.int64
)

y_test = np.asarray(
    y_test,
    dtype=np.int64
)


# ============================================================
# DATASET SUMMARY
# ============================================================

print()
print("==========================================")
print("DATASET SUMMARY")
print("==========================================")

print(
    "Number of classes:",
    N_CLASSES
)

print(
    "Number of input features:",
    X_train.shape[1]
)

print(
    "Training samples:",
    X_train.shape[0]
)

print(
    "Validation samples:",
    X_val.shape[0]
)

print(
    "Testing samples:",
    X_test.shape[0]
)


# ============================================================
# FINITE CHECK
# ============================================================

print()
print("==========================================")
print("FINITE VALUE CHECK")
print("==========================================")

print(
    "Training:",
    np.all(np.isfinite(X_train))
)

print(
    "Validation:",
    np.all(np.isfinite(X_val))
)

print(
    "Testing:",
    np.all(np.isfinite(X_test))
)

if not np.all(np.isfinite(X_train)):
    raise ValueError("Training contains NaN/inf.")

if not np.all(np.isfinite(X_val)):
    raise ValueError("Validation contains NaN/inf.")

if not np.all(np.isfinite(X_test)):
    raise ValueError("Testing contains NaN/inf.")


# ============================================================
# ORIGINAL CLASS DISTRIBUTION
# ============================================================

print()
print("==========================================")
print("TRAINING CLASS DISTRIBUTION")
print("==========================================")

class_counts = np.bincount(
    y_train,
    minlength=N_CLASSES
)

for i in range(N_CLASSES):

    print(
        f"{i:2d} -> "
        f"{CLASS_NAMES[i]:5s} : "
        f"{class_counts[i]:4d}"
    )


# ============================================================
# CLASS-WEIGHTED OVERSAMPLING
# ============================================================
#
# MLPClassifier in some sklearn versions does not accept
# sample_weight.
#
# Therefore we balance ONLY the training data by randomly
# duplicating minority-class samples.
#
# Validation and testing are NEVER oversampled.
#
# IMPORTANT:
# We do NOT force every class to have the same size.
# Instead, we use a capped target to reduce imbalance
# without massively increasing the dataset.
# ============================================================

print()
print("==========================================")
print("BALANCING TRAINING DATA")
print("==========================================")

MIN_TARGET = 100
MAX_TARGET = 300

balanced_indices = []

rng = np.random.RandomState(42)

for class_id in range(N_CLASSES):

    indices = np.where(
        y_train == class_id
    )[0]

    count = len(indices)

    if count == 0:

        print(
            f"Class {class_id}: NO SAMPLES"
        )

        continue

    target = min(
        max(
            count,
            MIN_TARGET
        ),
        MAX_TARGET
    )

    if count < target:

        extra = rng.choice(
            indices,
            size=target - count,
            replace=True
        )

        selected = np.concatenate(
            [
                indices,
                extra
            ]
        )

    else:

        selected = indices

    balanced_indices.extend(
        selected.tolist()
    )

    print(
        f"Class {class_id:2d} "
        f"{CLASS_NAMES[class_id]:5s}: "
        f"{count:4d} -> "
        f"{len(selected):4d}"
    )


balanced_indices = np.asarray(
    balanced_indices,
    dtype=np.int64
)


# Shuffle balanced training data

rng.shuffle(
    balanced_indices
)


X_train_balanced = X_train[
    balanced_indices
]

y_train_balanced = y_train[
    balanced_indices
]


print()
print(
    "Balanced training shape:",
    X_train_balanced.shape
)

print(
    "Balanced labels shape:",
    y_train_balanced.shape
)


# ============================================================
# CREATE PROTO 5.1 MODEL
# ============================================================

print()
print("==========================================")
print("CREATING PROTO 5.1 MLP")
print("==========================================")

model = MLPClassifier(

    # Smaller than Proto 5
    # to reduce memorization
    hidden_layer_sizes=(
        96,
        48
    ),

    activation="relu",

    solver="adam",

    # Stronger L2 regularization
    alpha=0.003,

    # Slightly conservative learning rate
    learning_rate_init=0.0003,

    batch_size=32,

    max_iter=200,

    # Internal early stopping
    early_stopping=True,

    validation_fraction=0.15,

    n_iter_no_change=20,

    tol=0.0005,

    random_state=42,

    verbose=True
)


# ============================================================
# TRAIN
# ============================================================

print()
print("==========================================")
print("TRAINING PROTO 5.1")
print("==========================================")

model.fit(
    X_train_balanced,
    y_train_balanced
)

print()
print("Proto 5.1 training completed!")


# ============================================================
# TRAINING ACCURACY
# ============================================================

print()
print("==========================================")
print("TRAINING RESULTS")
print("==========================================")

train_predictions = model.predict(
    X_train_balanced
)

train_accuracy = accuracy_score(
    y_train_balanced,
    train_predictions
)

print(
    f"Training accuracy: "
    f"{train_accuracy * 100:.2f}%"
)


# ============================================================
# VALIDATION
# ============================================================

print()
print("==========================================")
print("VALIDATION RESULTS")
print("==========================================")

val_predictions = model.predict(
    X_val
)

val_accuracy = accuracy_score(
    y_val,
    val_predictions
)

print(
    f"Validation accuracy: "
    f"{val_accuracy * 100:.2f}%"
)


# ============================================================
# UNSEEN SUBJECT TEST
# ============================================================

print()
print("==========================================")
print("UNSEEN-SUBJECT TEST RESULTS")
print("==========================================")

test_predictions = model.predict(
    X_test
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print(
    f"Unseen-subject test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# VALIDATION CLASSIFICATION REPORT
# ============================================================

print()
print("==========================================")
print("VALIDATION CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_val,
        val_predictions,
        labels=np.arange(N_CLASSES),
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ============================================================
# TEST CLASSIFICATION REPORT
# ============================================================

print()
print("==========================================")
print("UNSEEN-SUBJECT CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        test_predictions,
        labels=np.arange(N_CLASSES),
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("==========================================")
print("UNSEEN-SUBJECT CONFUSION MATRIX")
print("==========================================")

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=np.arange(N_CLASSES)
)

print()

print(
    "Rows    = Actual"
)

print(
    "Columns = Predicted"
)

print()

print(
    "     " +
    " ".join(
        f"{i:3d}"
        for i in range(N_CLASSES)
    )
)

for i, row in enumerate(cm):

    print(
        f"{i:3d}: " +
        " ".join(
            f"{value:3d}"
            for value in row
        )
    )


# ============================================================
# PER-CLASS TEST ACCURACY
# ============================================================

print()
print("==========================================")
print("PER-CLASS TEST ACCURACY")
print("==========================================")

for class_id in range(N_CLASSES):

    mask = (
        y_test == class_id
    )

    total = np.sum(mask)

    if total == 0:

        continue

    correct = np.sum(
        test_predictions[mask]
        == class_id
    )

    accuracy = (
        correct / total
    )

    print(
        f"{CLASS_NAMES[class_id]:5s}: "
        f"{correct:2d}/{total:2d} "
        f"({accuracy * 100:5.1f}%)"
    )


# ============================================================
# SAVE MODEL
# ============================================================

print()
print("==========================================")
print("SAVING PROTO 5.1 MODEL")
print("==========================================")

joblib.dump(
    model,
    MODEL_FILE
)

print(
    "Model saved successfully!"
)

print(
    "File:",
    MODEL_FILE
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("==========================================")
print("PROTO 5.1 FINAL SUMMARY")
print("==========================================")

print(
    "Number of classes:",
    N_CLASSES
)

print(
    "Number of input features:",
    X_train.shape[1]
)

print(
    "Original training samples:",
    X_train.shape[0]
)

print(
    "Balanced training samples:",
    X_train_balanced.shape[0]
)

print(
    "Validation samples:",
    X_val.shape[0]
)

print(
    "Testing samples:",
    X_test.shape[0]
)

print()

print(
    f"Training accuracy: "
    f"{train_accuracy * 100:.2f}%"
)

print(
    f"Validation accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

print(
    f"Unseen-subject test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print()

print(
    "Random 20-class baseline: 5.00%"
)

print()
print(
    "Proto 5.1 completed successfully."
)