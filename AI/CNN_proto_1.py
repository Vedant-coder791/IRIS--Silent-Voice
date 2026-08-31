# ============================================================
# IRIS - CNN PROTO 1.0
# Berkeley Closed-Vocabulary Silent Speech
#
# Dataset:
# MLP_proto1_4_dataset.npz
#
# Input:
# 112 engineered features
#
# Classes:
# AM, PM, DECEMBER, AUGUST, MARCH
# ============================================================

import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "/Users/vedantdwivedi/Desktop/IRIS/AI/MLP_proto1_4_dataset.npz"

MODEL_PATH = "/Users/vedantdwivedi/Desktop/IRIS/AI/CNN_proto1_model.keras"

RANDOM_STATE = 42

TEST_SIZE = 0.20

TARGET_CLASSES = [
    "AM",
    "PM",
    "DECEMBER",
    "AUGUST",
    "MARCH"
]

EXPECTED_FEATURES = 112


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("IRIS - CNN PROTO 1.0")
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
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_PATH}"
    )

print("Dataset found.")


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

data = np.load(DATASET_PATH, allow_pickle=True)

print("Dataset keys:")
print(list(data.keys()))

if "X" not in data:
    raise KeyError("Dataset does not contain X.")

if "y" not in data:
    raise KeyError("Dataset does not contain y.")

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
    raise ValueError(
        f"Expected X to be 2D, but got shape {X.shape}"
    )

samples, features = X.shape

print(f"Samples: {samples}")
print(f"Features: {features}")


# ============================================================
# FEATURE VALIDATION
# ============================================================

if features != EXPECTED_FEATURES:

    print()
    print(
        f"WARNING: Expected {EXPECTED_FEATURES} features "
        f"but found {features}."
    )

    EXPECTED_FEATURES = features


# ============================================================
# CONVERT FEATURES TO FLOAT32
# ============================================================

print()
print("=" * 70)
print("CONVERTING FEATURES")
print("=" * 70)

X = np.asarray(X, dtype=np.float32)

print("Feature dtype:", X.dtype)


# ============================================================
# CLEAN NUMERICAL VALUES
# ============================================================

print()
print("=" * 70)
print("NUMERICAL CHECK")
print("=" * 70)

print("NaN values:", np.isnan(X).sum())
print("Infinite values:", np.isinf(X).sum())

X = np.nan_to_num(
    X,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

print()
print("After cleaning:")
print("NaN values:", np.isnan(X).sum())
print("Infinite values:", np.isinf(X).sum())


# ============================================================
# LABEL CONVERSION
# ============================================================

print()
print("=" * 70)
print("PROCESSING LABELS")
print("=" * 70)

y = np.asarray(y).astype(str)

print("Labels dtype:", y.dtype)

print()
print("Original classes:")

unique_labels, counts = np.unique(y, return_counts=True)

for label, count in zip(unique_labels, counts):
    print(f"{label:12s}: {count:4d} samples")


# ============================================================
# FILTER TARGET CLASSES
# ============================================================

print()
print("=" * 70)
print("FILTERING TARGET VOCABULARY")
print("=" * 70)

mask = np.isin(y, TARGET_CLASSES)

X = X[mask]
y = y[mask]

print(f"Samples after filtering: {len(y)}")


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

for index, label in enumerate(label_encoder.classes_):
    print(f"  {index} -> {label}")


# ============================================================
# CHECK CLASSES
# ============================================================

if len(label_encoder.classes_) < 2:
    raise ValueError(
        "Need at least two classes for CNN classification."
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print()
print("=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# ============================================================
# FEATURE SCALING
# ============================================================

print()
print("=" * 70)
print("FEATURE SCALING")
print("=" * 70)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to float32 after scaling
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

print("StandardScaler: YES")


# ============================================================
# RESHAPE FOR 1D CNN
# ============================================================

print()
print("=" * 70)
print("RESHAPING FOR 1D CNN")
print("=" * 70)

# CNN expects:
#
# samples × sequence_length × channels
#
# Our data:
#
# samples × 112
#
# becomes:
#
# samples × 112 × 1

X_train = X_train.reshape(
    X_train.shape[0],
    X_train.shape[1],
    1
)

X_test = X_test.reshape(
    X_test.shape[0],
    X_test.shape[1],
    1
)

print("CNN training shape:", X_train.shape)
print("CNN testing shape: ", X_test.shape)


# ============================================================
# BUILD CNN
# ============================================================

print()
print("=" * 70)
print("BUILDING CNN")
print("=" * 70)

model = Sequential([

    # --------------------------------------------------------
    # Convolution block 1
    # --------------------------------------------------------

    Conv1D(
        filters=32,
        kernel_size=5,
        activation="relu",
        padding="same",
        input_shape=(EXPECTED_FEATURES, 1)
    ),

    BatchNormalization(),

    MaxPooling1D(
        pool_size=2
    ),

    Dropout(0.20),


    # --------------------------------------------------------
    # Convolution block 2
    # --------------------------------------------------------

    Conv1D(
        filters=64,
        kernel_size=5,
        activation="relu",
        padding="same"
    ),

    BatchNormalization(),

    MaxPooling1D(
        pool_size=2
    ),

    Dropout(0.25),


    # --------------------------------------------------------
    # Global feature extraction
    # --------------------------------------------------------

    GlobalAveragePooling1D(),


    # --------------------------------------------------------
    # Dense classifier
    # --------------------------------------------------------

    Dense(
        32,
        activation="relu"
    ),

    Dropout(0.30),

    Dense(
        len(label_encoder.classes_),
        activation="softmax"
    )
])


# ============================================================
# MODEL SUMMARY
# ============================================================

print()
print("=" * 70)
print("CNN ARCHITECTURE")
print("=" * 70)

model.summary()


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print("TRAINING CNN")
print("=" * 70)

print()
print("Starting training...")

history = model.fit(
    X_train,
    y_train,

    validation_split=0.20,

    epochs=200,

    batch_size=16,

    callbacks=[
        early_stopping
    ],

    verbose=1
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

train_probabilities = model.predict(
    X_train,
    verbose=0
)

train_predictions = np.argmax(
    train_probabilities,
    axis=1
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
# TEST PERFORMANCE
# ============================================================

print()
print("=" * 70)
print("TEST PERFORMANCE")
print("=" * 70)

test_probabilities = model.predict(
    X_test,
    verbose=0
)

test_predictions = np.argmax(
    test_probabilities,
    axis=1
)

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

print(
    classification_report(
        y_test,
        test_predictions,
        labels=np.arange(len(label_encoder.classes_)),
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
    test_predictions
)

print()

print(
    "Actual \\ Predicted"
)

print(
    f"{'':15s}",
    end=""
)

for label in label_encoder.classes_:
    print(
        f"{label:12s}",
        end=""
    )

print()

for i, label in enumerate(label_encoder.classes_):

    print(
        f"{label:15s}",
        end=""
    )

    for j in range(len(label_encoder.classes_)):

        print(
            f"{cm[i, j]:12d}",
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

for i, label in enumerate(label_encoder.classes_):

    total = np.sum(y_test == i)

    correct = cm[i, i]

    percentage = (
        correct / total * 100
        if total > 0
        else 0
    )

    print(
        f"{label:12s}: "
        f"{correct:3d}/{total:3d} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# TRAINING HISTORY
# ============================================================

print()
print("=" * 70)
print("TRAINING INFORMATION")
print("=" * 70)

print(
    "Epochs completed:",
    len(history.history["loss"])
)

print(
    "Final training loss:",
    history.history["loss"][-1]
)

print(
    "Final validation loss:",
    history.history["val_loss"][-1]
)

print(
    "Final training accuracy:",
    history.history["accuracy"][-1]
)

print(
    "Final validation accuracy:",
    history.history["val_accuracy"][-1]
)


# ============================================================
# SAVE MODEL
# ============================================================

print()
print("=" * 70)
print("SAVING MODEL")
print("=" * 70)

model.save(
    MODEL_PATH
)

print()
print("CNN model saved to:")
print(MODEL_PATH)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("CNN PROTO 1.0 COMPLETE")
print("=" * 70)

print()
print("Dataset:")
print(f"  Samples:       {len(y)}")
print(f"  Features:      {X.shape[1]}")
print(f"  Classes:       {len(label_encoder.classes_)}")

print()
print("Split:")
print(f"  Training:      {len(y_train)}")
print(f"  Testing:       {len(y_test)}")

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
print("CNN:")
print("  Conv1D filters: 32 -> 64")
print("  Kernel size: 5")
print("  Batch normalization: YES")
print("  Max pooling: YES")
print("  Dropout: YES")
print("  Early stopping: YES")
print("  StandardScaler: YES")

print()
print("=" * 70)