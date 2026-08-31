import os
import pickle
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# PROTO 6
# Stable MLP with:
#
# 1. Correct dataset variable names
# 2. Training-only normalization
# 3. Numerical stability protection
# 4. Early stopping
# 5. L2 regularization
# 6. Validation evaluation
# 7. Completely unseen-subject testing
# 8. Classification reports
# 9. Confusion matrix
# 10. Per-class accuracy
# 11. Model saving
#
# NOTE:
# sklearn MLPClassifier.fit() in this environment does NOT
# support sample_weight, so sample weighting has been removed.
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/AI/"
    "proto5_dataset.npz"
)

MODEL_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/AI/"
    "MLP_proto6_model.pkl"
)

RANDOM_STATE = 42


# ============================================================
# HEADER
# ============================================================

print("=" * 42)
print("PROTO 6 MLP")
print("=" * 42)


# ============================================================
# LOAD DATASET
# ============================================================

print("\n==========================================")
print("LOADING DATASET")
print("==========================================")

if not os.path.isfile(DATASET_PATH):

    raise FileNotFoundError(
        "\nDataset not found:\n"
        f"{DATASET_PATH}\n\n"
        "Check that proto5_dataset.npz exists."
    )


data = np.load(
    DATASET_PATH,
    allow_pickle=True
)

print(
    "\nDataset keys:",
    list(data.keys())
)


# ============================================================
# CHECK REQUIRED VARIABLES
# ============================================================

required_keys = [
    "X_train",
    "X_validation",
    "X_test",
    "y_train",
    "y_validation",
    "y_test",
    "subjects_train",
    "subjects_validation",
    "subjects_test",
    "classes"
]

missing = [
    key for key in required_keys
    if key not in data
]

if missing:

    raise KeyError(
        "\nMissing variables:\n"
        f"{missing}\n\n"
        "Available variables:\n"
        f"{list(data.keys())}"
    )


# ============================================================
# LOAD ARRAYS
# ============================================================

X_train = np.asarray(
    data["X_train"],
    dtype=np.float32
)

X_validation = np.asarray(
    data["X_validation"],
    dtype=np.float32
)

X_test = np.asarray(
    data["X_test"],
    dtype=np.float32
)

y_train = np.asarray(
    data["y_train"]
)

y_validation = np.asarray(
    data["y_validation"]
)

y_test = np.asarray(
    data["y_test"]
)

subjects_train = np.asarray(
    data["subjects_train"]
)

subjects_validation = np.asarray(
    data["subjects_validation"]
)

subjects_test = np.asarray(
    data["subjects_test"]
)

classes = np.asarray(
    data["classes"]
)


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n==========================================")
print("DATASET INFORMATION")
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
    len(X_train)
)

print(
    "Validation samples:",
    len(X_validation)
)

print(
    "Testing samples:",
    len(X_test)
)

print("\nClasses:")

for i, class_name in enumerate(classes):

    print(
        f"{i:2d} -> {class_name}"
    )


# ============================================================
# SHAPE CHECK
# ============================================================

print("\n==========================================")
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
    "X_validation:",
    X_validation.shape
)

print(
    "y_validation:",
    y_validation.shape
)

print(
    "X_test:",
    X_test.shape
)

print(
    "y_test:",
    y_test.shape
)


if X_train.shape[0] != len(y_train):

    raise ValueError(
        "Training X/y sample counts do not match."
    )


if X_validation.shape[0] != len(y_validation):

    raise ValueError(
        "Validation X/y sample counts do not match."
    )


if X_test.shape[0] != len(y_test):

    raise ValueError(
        "Testing X/y sample counts do not match."
    )


# ============================================================
# SUBJECT CHECK
# ============================================================

print("\n==========================================")
print("SUBJECT SPLIT")
print("==========================================")

print(
    "Training subjects:",
    np.unique(subjects_train)
)

print(
    "Validation subjects:",
    np.unique(subjects_validation)
)

print(
    "Testing subjects:",
    np.unique(subjects_test)
)


# ============================================================
# CHECK FOR SUBJECT LEAKAGE
# ============================================================

train_subject_set = set(
    subjects_train.astype(str)
)

validation_subject_set = set(
    subjects_validation.astype(str)
)

test_subject_set = set(
    subjects_test.astype(str)
)


train_val_overlap = (
    train_subject_set &
    validation_subject_set
)

train_test_overlap = (
    train_subject_set &
    test_subject_set
)

validation_test_overlap = (
    validation_subject_set &
    test_subject_set
)


print(
    "\nTrain/Validation overlap:",
    train_val_overlap
)

print(
    "Train/Test overlap:",
    train_test_overlap
)

print(
    "Validation/Test overlap:",
    validation_test_overlap
)


if train_val_overlap:

    print(
        "\nWARNING:"
        " Training and validation contain "
        "the same subjects."
    )


if train_test_overlap:

    print(
        "\nWARNING:"
        " Training and testing contain "
        "the same subjects."
    )


if validation_test_overlap:

    print(
        "\nWARNING:"
        " Validation and testing contain "
        "the same subjects."
    )


# ============================================================
# NUMERICAL DATA CHECK
# ============================================================

print("\n==========================================")
print("NUMERICAL DATA CHECK")
print("==========================================")


def check_array(name, array):

    nan_count = np.isnan(array).sum()

    inf_count = np.isinf(array).sum()

    print(
        f"{name}: "
        f"NaN={nan_count}, "
        f"Inf={inf_count}"
    )


check_array(
    "X_train",
    X_train
)

check_array(
    "X_validation",
    X_validation
)

check_array(
    "X_test",
    X_test
)


# ============================================================
# REPLACE NON-FINITE VALUES
# ============================================================

X_train = np.nan_to_num(
    X_train,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

X_validation = np.nan_to_num(
    X_validation,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

X_test = np.nan_to_num(
    X_test,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


# ============================================================
# CHECK EXTREME VALUES
# ============================================================

print("\n==========================================")
print("FEATURE RANGE")
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
    "Training mean:",
    np.mean(X_train)
)

print(
    "Training std:",
    np.std(X_train)
)


# ============================================================
# NORMALIZATION
#
# IMPORTANT:
#
# The scaler is fitted ONLY on training data.
#
# Validation and test data are transformed using
# the training distribution.
#
# This prevents data leakage.
# ============================================================

print("\n==========================================")
print("NORMALIZATION")
print("==========================================")


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)

X_validation_scaled = scaler.transform(
    X_validation
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# FINAL NUMERICAL SAFETY
# ============================================================

X_train_scaled = np.nan_to_num(
    X_train_scaled,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

X_validation_scaled = np.nan_to_num(
    X_validation_scaled,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

X_test_scaled = np.nan_to_num(
    X_test_scaled,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


print(
    "Training mean:",
    np.mean(X_train_scaled)
)

print(
    "Training standard deviation:",
    np.std(X_train_scaled)
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n==========================================")
print("TRAINING CLASS DISTRIBUTION")
print("==========================================")


unique_classes, class_counts = np.unique(
    y_train,
    return_counts=True
)


for label, count in zip(
    unique_classes,
    class_counts
):

    label_int = int(label)

    if label_int < len(classes):

        class_name = str(
            classes[label_int]
        )

    else:

        class_name = str(
            label
        )

    print(
        f"{class_name:12s}: {count:5d}"
    )


# ============================================================
# CREATE MLP
# ============================================================

print("\n==========================================")
print("CREATING PROTO 6 MLP")
print("==========================================")


model = MLPClassifier(

    # --------------------------------------------------------
    # Architecture
    # --------------------------------------------------------

    hidden_layer_sizes=(
        128,
        64,
        32
    ),

    activation="relu",

    # --------------------------------------------------------
    # Regularization
    # --------------------------------------------------------

    alpha=0.001,

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    solver="adam",

    learning_rate="adaptive",

    learning_rate_init=0.0005,

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    max_iter=500,

    batch_size=32,

    early_stopping=True,

    validation_fraction=0.15,

    n_iter_no_change=30,

    tol=1e-4,

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    random_state=RANDOM_STATE,

    verbose=True
)


# ============================================================
# TRAIN
# ============================================================

print("\n==========================================")
print("TRAINING PROTO 6")
print("==========================================")

print(
    "\nTraining without sample_weight."
)

print(
    "MLPClassifier in this environment "
    "does not support sample_weight."
)


model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# TRAINING ACCURACY
# ============================================================

print("\n==========================================")
print("TRAINING PERFORMANCE")
print("==========================================")


train_predictions = model.predict(
    X_train_scaled
)

training_accuracy = accuracy_score(
    y_train,
    train_predictions
)


print(
    "Training accuracy:",
    f"{training_accuracy * 100:.2f}%"
)


# ============================================================
# VALIDATION
# ============================================================

print("\n==========================================")
print("VALIDATION")
print("==========================================")


validation_predictions = model.predict(
    X_validation_scaled
)


validation_accuracy = accuracy_score(
    y_validation,
    validation_predictions
)


print(
    "Validation accuracy:",
    f"{validation_accuracy * 100:.2f}%"
)


# ============================================================
# UNSEEN SUBJECT TEST
# ============================================================

print("\n==========================================")
print("UNSEEN-SUBJECT TEST")
print("==========================================")


test_predictions = model.predict(
    X_test_scaled
)


test_accuracy = accuracy_score(
    y_test,
    test_predictions
)


print(
    "Unseen-subject test accuracy:",
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# VALIDATION CLASSIFICATION REPORT
# ============================================================

print("\n==========================================")
print("VALIDATION CLASSIFICATION REPORT")
print("==========================================")


print(
    classification_report(
        y_validation,
        validation_predictions,
        labels=np.arange(len(classes)),
        target_names=classes.astype(str),
        zero_division=0
    )
)


# ============================================================
# TEST CLASSIFICATION REPORT
# ============================================================

print("\n==========================================")
print("UNSEEN-SUBJECT CLASSIFICATION REPORT")
print("==========================================")


print(
    classification_report(
        y_test,
        test_predictions,
        labels=np.arange(len(classes)),
        target_names=classes.astype(str),
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n==========================================")
print("UNSEEN-SUBJECT CONFUSION MATRIX")
print("==========================================")


cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=np.arange(len(classes))
)


print(
    "\nRows    = Actual"
)

print(
    "Columns = Predicted\n"
)


print(
    "     ",
    " ".join(
        f"{i:3d}"
        for i in range(len(classes))
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
# PER-CLASS TEST ACCURACY
# ============================================================

print("\n==========================================")
print("PER-CLASS TEST ACCURACY")
print("==========================================")


for i, class_name in enumerate(classes):

    total = np.sum(
        y_test == i
    )

    correct = cm[i, i]

    if total > 0:

        class_accuracy = (
            correct / total
        ) * 100

    else:

        class_accuracy = 0.0

    print(
        f"{str(class_name):12s}: "
        f"{correct:3d}/{total:3d} "
        f"({class_accuracy:5.1f}%)"
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\n==========================================")
print("MODEL INFORMATION")
print("==========================================")


print(
    "Hidden layers:",
    model.hidden_layer_sizes
)

print(
    "Activation:",
    model.activation
)

print(
    "Optimizer:",
    model.solver
)

print(
    "Alpha:",
    model.alpha
)

print(
    "Learning rate:",
    model.learning_rate_init
)

print(
    "Iterations used:",
    model.n_iter_
)

print(
    "Number of training samples:",
    model.n_iter_
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\n==========================================")
print("SAVING PROTO 6 MODEL")
print("==========================================")


model_package = {

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    "model": model,

    # --------------------------------------------------------
    # Scaler
    #
    # IMPORTANT:
    # The scaler is required to process future EMG samples
    # in exactly the same way as the training data.
    # --------------------------------------------------------

    "scaler": scaler,

    # --------------------------------------------------------
    # Class names
    # --------------------------------------------------------

    "classes": classes,

    # --------------------------------------------------------
    # Feature count
    # --------------------------------------------------------

    "n_features": X_train.shape[1],

    # --------------------------------------------------------
    # Subject information
    # --------------------------------------------------------

    "training_subjects": np.unique(
        subjects_train
    ),

    "validation_subjects": np.unique(
        subjects_validation
    ),

    "testing_subjects": np.unique(
        subjects_test
    ),

    # --------------------------------------------------------
    # Feature description
    # --------------------------------------------------------

    "feature_type": (
        "Proto 5 MFCC + time + frequency"
    ),

    # --------------------------------------------------------
    # Random state
    # --------------------------------------------------------

    "random_state": RANDOM_STATE,

    # --------------------------------------------------------
    # Model version
    # --------------------------------------------------------

    "model_version": "Proto 6",

    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    "hidden_layer_sizes": (
        128,
        64,
        32
    ),

    "activation": "relu",

    "solver": "adam",

    "alpha": 0.001,

    "learning_rate_init": 0.0005,

    "max_iter": 500,

    "batch_size": 32,

    "early_stopping": True,

    "validation_fraction": 0.15,

    "n_iter_no_change": 30
}


with open(
    MODEL_PATH,
    "wb"
) as f:

    pickle.dump(
        model_package,
        f
    )


print(
    "\nModel saved successfully!"
)

print(
    "File:",
    MODEL_PATH
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n==========================================")
print("PROTO 6 FINAL SUMMARY")
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
    len(X_train)
)

print(
    "Validation samples:",
    len(X_validation)
)

print(
    "Testing samples:",
    len(X_test)
)

print(
    "\nTraining accuracy:",
    f"{training_accuracy * 100:.2f}%"
)

print(
    "Validation accuracy:",
    f"{validation_accuracy * 100:.2f}%"
)

print(
    "Unseen-subject test accuracy:",
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# RANDOM BASELINE
# ============================================================

if len(classes) > 0:

    random_baseline = (
        100 / len(classes)
    )

else:

    random_baseline = 0.0


print(
    "\nRandom",
    f"{len(classes)}-class baseline:",
    f"{random_baseline:.2f}%"
)


# ============================================================
# FINAL
# ============================================================

print("\n==========================================")
print("PROTO 6 COMPLETE")
print("==========================================")