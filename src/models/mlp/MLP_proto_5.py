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
    "MLP_proto5_model.pkl"
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


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("==========================================")
print("LOADING PROTO 5 DATASET")
print("==========================================")

try:

    data = np.load(
        DATA_FILE,
        allow_pickle=True
    )

except Exception as e:

    print()
    print("ERROR: Could not load dataset.")
    print(e)
    raise SystemExit


# ============================================================
# SHOW AVAILABLE KEYS
# ============================================================

print()
print("Available dataset variables:")

for key in data.files:

    print(
        f"  {key:20s}",
        data[key].shape
    )


# ============================================================
# HELPER FUNCTION
# ============================================================

def find_array(data, possible_names):

    """
    Finds the first matching array from a list
    of possible names.
    """

    for name in possible_names:

        if name in data.files:

            return data[name]

    return None


# ============================================================
# FIND TRAINING DATA
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


# ============================================================
# FIND VALIDATION DATA
# ============================================================

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


# ============================================================
# FIND TEST DATA
# ============================================================

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
# CHECK DATASET
# ============================================================

print()
print("==========================================")
print("DATASET KEY CHECK")
print("==========================================")


if X_train is None:
    print("ERROR: Training features not found.")

if y_train is None:
    print("ERROR: Training labels not found.")

if X_val is None:
    print("ERROR: Validation features not found.")

if y_val is None:
    print("ERROR: Validation labels not found.")

if X_test is None:
    print("ERROR: Testing features not found.")

if y_test is None:
    print("ERROR: Testing labels not found.")


if (
    X_train is None
    or y_train is None
    or X_val is None
    or y_val is None
    or X_test is None
    or y_test is None
):

    print()
    print("The dataset variable names do not match")
    print("the expected Proto 5 names.")
    print()
    print("Available keys were:")

    for key in data.files:
        print(" -", key)

    raise SystemExit


print("All required arrays found!")


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
# DATASET INFORMATION
# ============================================================

print()
print("==========================================")
print("DATASET INFORMATION")
print("==========================================")

print(
    "Number of classes:",
    len(CLASS_NAMES)
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
# SHAPE CHECK
# ============================================================

print()
print("==========================================")
print("SHAPE CHECK")
print("==========================================")

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


if X_train.shape[0] != y_train.shape[0]:

    raise ValueError(
        "Training X and y have different numbers of samples."
    )

if X_val.shape[0] != y_val.shape[0]:

    raise ValueError(
        "Validation X and y have different numbers of samples."
    )

if X_test.shape[0] != y_test.shape[0]:

    raise ValueError(
        "Testing X and y have different numbers of samples."
    )


if X_train.shape[1] != X_val.shape[1]:

    raise ValueError(
        "Training and validation have different "
        "numbers of features."
    )

if X_train.shape[1] != X_test.shape[1]:

    raise ValueError(
        "Training and testing have different "
        "numbers of features."
    )


# ============================================================
# FINITE VALUE CHECK
# ============================================================

print()
print("==========================================")
print("FINITE VALUE CHECK")
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


if not np.all(np.isfinite(X_train)):

    raise ValueError(
        "Training data contains NaN or infinity."
    )

if not np.all(np.isfinite(X_val)):

    raise ValueError(
        "Validation data contains NaN or infinity."
    )

if not np.all(np.isfinite(X_test)):

    raise ValueError(
        "Testing data contains NaN or infinity."
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()
print("==========================================")
print("TRAINING CLASS DISTRIBUTION")
print("==========================================")

for class_id in range(len(CLASS_NAMES)):

    count = np.sum(
        y_train == class_id
    )

    print(
        f"{class_id:2d} -> "
        f"{CLASS_NAMES[class_id]:5s} : "
        f"{count:4d}"
    )


# ============================================================
# CREATE MLP
# ============================================================

print()
print("==========================================")
print("CREATING PROTO 5 MLP")
print("==========================================")

model = MLPClassifier(

    hidden_layer_sizes=(
        128,
        64
    ),

    activation="relu",

    solver="adam",

    alpha=0.001,

    learning_rate_init=0.0005,

    batch_size=32,

    max_iter=150,

    early_stopping=True,

    validation_fraction=0.15,

    n_iter_no_change=15,

    tol=0.0005,

    random_state=42,

    verbose=True
)


# ============================================================
# TRAIN
# ============================================================

print()
print("==========================================")
print("TRAINING PROTO 5 MLP")
print("==========================================")

model.fit(
    X_train,
    y_train
)

print()
print("MLP training completed!")


# ============================================================
# TRAINING RESULTS
# ============================================================

print()
print("==========================================")
print("TRAINING RESULTS")
print("==========================================")

train_predictions = model.predict(
    X_train
)

train_accuracy = accuracy_score(
    y_train,
    train_predictions
)

print(
    f"Training accuracy: "
    f"{train_accuracy * 100:.2f}%"
)


# ============================================================
# VALIDATION RESULTS
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
        labels=np.arange(
            len(CLASS_NAMES)
        ),
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
        labels=np.arange(
            len(CLASS_NAMES)
        ),
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
    labels=np.arange(
        len(CLASS_NAMES)
    )
)

print()

print(
    "Rows = Actual"
)

print(
    "Columns = Predicted"
)

print()

print(
    "     ",
    " ".join(
        f"{i:3d}"
        for i in range(len(CLASS_NAMES))
    )
)

for i, row in enumerate(cm):

    print(
        f"{i:3d}:",
        " ".join(
            f"{value:3d}"
            for value in row
        )
    )


# ============================================================
# SAVE MODEL
# ============================================================

print()
print("==========================================")
print("SAVING PROTO 5 MODEL")
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
print("PROTO 5 FINAL SUMMARY")
print("==========================================")

print(
    "Number of classes:",
    len(CLASS_NAMES)
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
print("Proto 5 completed successfully.")