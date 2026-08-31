# ============================================================
# MLP PROTO 4
# Subject-Aware Silent Speech Recognition
# ============================================================

import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils import resample
import joblib


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/"
    "Data/dataset_subject.npz"
)

MODEL_OUTPUT = (
    "/Users/vedantdwivedi/Desktop/IRIS/"
    "AI/mlp_proto_4.pkl"
)

RANDOM_STATE = 42


# ============================================================
# LOAD DATASET
# ============================================================

print("\n==========================================")
print("LOADING SUBJECT-AWARE DATASET")
print("==========================================")

data = np.load(DATASET_PATH)

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_val"]
y_val = data["y_val"]

X_test = data["X_test"]
y_test = data["y_test"]

classes = data["classes"]


print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

print("Number of classes:", len(classes))
print("Number of features:", X_train.shape[1])


# ============================================================
# CHECK DATA
# ============================================================

print("\n==========================================")
print("CHECKING DATA")
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


# Stop if invalid values exist

if not np.all(np.isfinite(X_train)):
    raise ValueError(
        "Training dataset contains NaN or infinite values."
    )

if not np.all(np.isfinite(X_val)):
    raise ValueError(
        "Validation dataset contains NaN or infinite values."
    )

if not np.all(np.isfinite(X_test)):
    raise ValueError(
        "Testing dataset contains NaN or infinite values."
    )


# ============================================================
# CHECK FEATURE MAGNITUDE
# ============================================================

print("\n==========================================")
print("FEATURE MAGNITUDE")
print("==========================================")

print(
    "Training minimum:",
    np.min(X_train)
)

print(
    "Training maximum:",
    np.max(X_train)
)

print(
    "Largest absolute value:",
    np.max(np.abs(X_train))
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n==========================================")
print("ORIGINAL TRAINING DISTRIBUTION")
print("==========================================")

unique_classes, counts = np.unique(
    y_train,
    return_counts=True
)

for class_id, count in zip(
    unique_classes,
    counts
):

    print(
        f"Class {class_id:2d}: "
        f"{count:4d} "
        f"-> {str(classes[class_id])}"
    )


# ============================================================
# BALANCE TRAINING DATA
# ============================================================

print("\n==========================================")
print("BALANCING TRAINING DATA")
print("==========================================")


# We use a moderate target size rather than
# massively duplicating rare classes.

TARGET_CLASS_SIZE = 112

X_balanced = []
y_balanced = []


for class_id in unique_classes:

    X_class = X_train[
        y_train == class_id
    ]

    y_class = y_train[
        y_train == class_id
    ]

    original_size = len(X_class)


    # --------------------------------------------------------
    # If class has more than target:
    # keep all original samples
    # --------------------------------------------------------

    if original_size >= TARGET_CLASS_SIZE:

        X_selected = X_class
        y_selected = y_class


    # --------------------------------------------------------
    # If class has fewer than target:
    # randomly oversample
    # --------------------------------------------------------

    else:

        X_selected, y_selected = resample(
            X_class,
            y_class,
            replace=True,
            n_samples=TARGET_CLASS_SIZE,
            random_state=RANDOM_STATE
        )


    X_balanced.append(
        X_selected
    )

    y_balanced.append(
        y_selected
    )

    print(
        f"Class {class_id:2d}: "
        f"{original_size:4d} -> "
        f"{len(X_selected):4d}"
    )


# Combine classes

X_balanced = np.vstack(
    X_balanced
)

y_balanced = np.concatenate(
    y_balanced
)


# ============================================================
# SHUFFLE BALANCED DATA
# ============================================================

rng = np.random.default_rng(
    RANDOM_STATE
)

indices = rng.permutation(
    len(X_balanced)
)

X_balanced = X_balanced[
    indices
]

y_balanced = y_balanced[
    indices
]


print("\nBalanced training shape:")
print(
    X_balanced.shape
)

print(
    y_balanced.shape
)


# ============================================================
# CHECK BALANCED DATA
# ============================================================

print("\n==========================================")
print("BALANCED DATA CHECK")
print("==========================================")

print(
    "Finite:",
    np.all(np.isfinite(X_balanced))
)

print(
    "Minimum:",
    np.min(X_balanced)
)

print(
    "Maximum:",
    np.max(X_balanced)
)


# ============================================================
# CREATE MLP
# ============================================================

print("\n==========================================")
print("CREATING MLP")
print("==========================================")


model = MLPClassifier(

    # Smaller network than Proto 3
    hidden_layer_sizes=(
        128,
        64
    ),

    activation="relu",

    solver="adam",

    # Stronger L2 regularization
    alpha=0.01,

    batch_size=64,

    learning_rate_init=0.0005,

    max_iter=150,

    # Internal validation
    early_stopping=True,

    validation_fraction=0.15,

    n_iter_no_change=15,

    tol=0.0005,

    random_state=RANDOM_STATE,

    verbose=True
)


# ============================================================
# TRAIN
# ============================================================

print("\n==========================================")
print("TRAINING MLP")
print("==========================================")

model.fit(
    X_balanced,
    y_balanced
)


print("\nMLP training completed!")


# ============================================================
# TRAINING INFORMATION
# ============================================================

print("\n==========================================")
print("TRAINING INFORMATION")
print("==========================================")

print(
    "Iterations:",
    model.n_iter_
)

print(
    "Final training loss:",
    model.loss_
)

if hasattr(
    model,
    "best_validation_score_"
):

    print(
        "Best internal validation score:",
        model.best_validation_score_
    )


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print("\n==========================================")
print("VALIDATION RESULTS")
print("==========================================")


y_val_pred = model.predict(
    X_val
)

val_accuracy = accuracy_score(
    y_val,
    y_val_pred
)

print(
    f"Validation accuracy: "
    f"{val_accuracy:.4f}"
)

print(
    f"Validation accuracy: "
    f"{val_accuracy * 100:.2f}%"
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

print("\n==========================================")
print("UNSEEN SUBJECT TEST RESULTS")
print("==========================================")


y_test_pred = model.predict(
    X_test
)

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

print(
    f"Test accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")


# Convert class names to strings.
# This fixes the numpy.int64 error.

class_names = [
    str(x)
    for x in classes
]


report = classification_report(

    y_test,

    y_test_pred,

    labels=np.arange(
        len(class_names)
    ),

    target_names=class_names,

    zero_division=0
)


print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")


cm = confusion_matrix(

    y_test,

    y_test_pred,

    labels=np.arange(
        len(class_names)
    )
)


print(cm)


# ============================================================
# PER-CLASS TEST ACCURACY
# ============================================================

print("\n==========================================")
print("PER-CLASS TEST RESULTS")
print("==========================================")


for class_id, class_name in enumerate(
    class_names
):

    mask = (
        y_test == class_id
    )

    total = np.sum(mask)


    if total == 0:

        print(
            f"{class_name:8s}: "
            "No test samples"
        )

        continue


    correct = np.sum(
        y_test_pred[mask]
        == class_id
    )


    accuracy = (
        correct / total
    )


    print(
        f"{class_name:8s}: "
        f"{correct:3d}/{total:3d} "
        f"({accuracy * 100:6.2f}%)"
    )


# ============================================================
# MOST COMMON PREDICTIONS
# ============================================================

print("\n==========================================")
print("PREDICTION DISTRIBUTION")
print("==========================================")


pred_classes, pred_counts = np.unique(
    y_test_pred,
    return_counts=True
)


for class_id, count in zip(
    pred_classes,
    pred_counts
):

    print(
        f"{str(classes[class_id]):8s}: "
        f"{count}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

print("\n==========================================")
print("SAVING MODEL")
print("==========================================")


joblib.dump(
    model,
    MODEL_OUTPUT
)


print(
    "Model saved successfully!"
)

print(
    "File:"
)

print(
    MODEL_OUTPUT
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n==========================================")
print("PROTO 4 FINAL SUMMARY")
print("==========================================")

print(
    "Number of classes:",
    len(classes)
)

print(
    "Number of input features:",
    X_train.shape[1]
)

print(
    "Training samples:",
    X_balanced.shape[0]
)

print(
    f"Validation accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

print(
    f"Unseen-subject test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    "\nProto 4 completed successfully!"
)

print("==========================================")