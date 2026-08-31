"""
======================================================================
IRIS - BERKELEY 5-WORD 1D CNN PROTO 2.4
======================================================================

Purpose:
    Train a robust 1D CNN on the Berkeley 5-word augmented
    time-frequency dataset.

Vocabulary:
    THE
    AND
    A
    OF
    I

Input:
    X shape = (N, 8, 24, 64)

Converted to:
    (N, 64, 192)

where:
    64  = time steps
    192 = 8 channels × 24 frequency bins

Major changes from Proto 2.3:
    - WeightedRandomSampler
    - Moderate class weighting
    - Label smoothing
    - BatchNorm
    - Dropout
    - Global average pooling
    - Macro-F1 model selection
    - ReduceLROnPlateau
    - Early stopping
    - Prediction sanity check
    - Confusion matrix
    - Classification report

======================================================================
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    WeightedRandomSampler
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# =====================================================================
# CONFIGURATION
# =====================================================================

DATASET_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/Data/datasets/"
    "berkeley_5word_tf_augmented_dataset.npz"
)

MODEL_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/AI/"
    "berkeley_5word_1dcnn_proto2_4_best.pt"
)

CONFUSION_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/AI/"
    "berkeley_5word_1dcnn_proto2_4_confusion_matrix.png"
)

TRAINING_PLOT_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/AI/"
    "berkeley_5word_1dcnn_proto2_4_training.png"
)

ACCURACY_PLOT_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/AI/"
    "berkeley_5word_1dcnn_proto2_4_accuracy.png"
)


VOCABULARY = [
    "THE",
    "AND",
    "A",
    "OF",
    "I"
]


RANDOM_STATE = 42

BATCH_SIZE = 32

EPOCHS = 100

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

DROPOUT = 0.35

LABEL_SMOOTHING = 0.05

PATIENCE = 15

MIN_LR = 1e-6


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# =====================================================================
# HEADER
# =====================================================================

print("=" * 70)
print("IRIS - BERKELEY 5-WORD 1D CNN PROTO 2.4")
print("=" * 70)

print()
print("Dataset:")
print(DATASET_PATH)

print()
print("Device:")
print(DEVICE)

print()
print("Vocabulary:")

for i, word in enumerate(VOCABULARY):
    print(f"  {i}: {word}")


# =====================================================================
# LOAD DATASET
# =====================================================================

print()
print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

data = np.load(
    DATASET_PATH,
    allow_pickle=True
)

print()
print("Dataset keys:")
print(list(data.keys()))


X_train = data["X_train"]
y_train_raw = data["y_train"]

X_val = data["X_val"]
y_val_raw = data["y_val"]

X_test = data["X_test"]
y_test_raw = data["y_test"]


print()
print("=" * 70)
print("DATASET SHAPES")
print("=" * 70)

print()
print("X_train:", X_train.shape)
print("y_train:", y_train_raw.shape)

print()

print("X_val:", X_val.shape)
print("y_val:", y_val_raw.shape)

print()

print("X_test:", X_test.shape)
print("y_test:", y_test_raw.shape)


# =====================================================================
# NUMERICAL VALIDATION
# =====================================================================

print()
print("=" * 70)
print("NUMERICAL VALIDATION")
print("=" * 70)


def check_array(name, array):

    nan_count = np.isnan(array).sum()

    inf_count = np.isinf(array).sum()

    print(
        f"{name} NaN: {nan_count}"
    )

    print(
        f"{name} Inf: {inf_count}"
    )

    if nan_count > 0:
        raise ValueError(
            f"{name} contains NaN values."
        )

    if inf_count > 0:
        raise ValueError(
            f"{name} contains infinite values."
        )


check_array("Training", X_train)

check_array("Validation", X_val)

check_array("Test", X_test)


# =====================================================================
# CHECK INPUT SHAPE
# =====================================================================

if X_train.ndim != 4:

    raise ValueError(
        f"Expected X_train to have 4 dimensions "
        f"(N, C, F, T), got {X_train.shape}"
    )


N_CHANNELS = X_train.shape[1]

N_FREQ = X_train.shape[2]

N_TIME = X_train.shape[3]

FEATURES_PER_TIMESTEP = N_CHANNELS * N_FREQ


print()
print("=" * 70)
print("INPUT CONFIGURATION")
print("=" * 70)

print()
print("Channels:", N_CHANNELS)

print("Frequency bins:", N_FREQ)

print("Time frames:", N_TIME)

print(
    "Features per timestep:",
    FEATURES_PER_TIMESTEP
)


# =====================================================================
# LABEL ENCODING
# =====================================================================

print()
print("=" * 70)
print("ENCODING LABELS")
print("=" * 70)


def encode_labels(labels, vocabulary):

    labels = np.asarray(labels)

    encoded = []

    vocabulary_upper = [
        str(v).upper()
        for v in vocabulary
    ]

    for label in labels:

        # -------------------------------------------------------------
        # Numeric labels
        # -------------------------------------------------------------

        if isinstance(
            label,
            (int, np.integer)
        ):

            value = int(label)

            if 0 <= value < len(vocabulary):

                encoded.append(value)

            else:

                raise ValueError(
                    f"Numeric label {value} is outside "
                    f"the vocabulary range."
                )

        # -------------------------------------------------------------
        # Floating-point labels
        # -------------------------------------------------------------

        elif isinstance(
            label,
            (float, np.floating)
        ):

            value = int(label)

            if float(label) == value and \
               0 <= value < len(vocabulary):

                encoded.append(value)

            else:

                raise ValueError(
                    f"Invalid numeric label: {label}"
                )

        # -------------------------------------------------------------
        # String labels
        # -------------------------------------------------------------

        else:

            text = str(label).strip().upper()

            # Direct word match

            if text in vocabulary_upper:

                encoded.append(
                    vocabulary_upper.index(text)
                )

            # Numeric string such as "0"

            elif text.isdigit():

                value = int(text)

                if 0 <= value < len(vocabulary):

                    encoded.append(value)

                else:

                    raise ValueError(
                        f"Numeric label '{text}' "
                        f"is outside vocabulary."
                    )

            else:

                raise ValueError(
                    f"Unknown label '{label}' "
                    f"not found in vocabulary."
                )

    return np.asarray(
        encoded,
        dtype=np.int64
    )


y_train = encode_labels(
    y_train_raw,
    VOCABULARY
)

y_val = encode_labels(
    y_val_raw,
    VOCABULARY
)

y_test = encode_labels(
    y_test_raw,
    VOCABULARY
)


# =====================================================================
# CLASS DISTRIBUTION
# =====================================================================

print()
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)


for i, word in enumerate(VOCABULARY):

    train_count = np.sum(y_train == i)

    val_count = np.sum(y_val == i)

    test_count = np.sum(y_test == i)

    print(
        f"{word:<10} "
        f"Train: {train_count:<5} "
        f"Val: {val_count:<5} "
        f"Test: {test_count:<5}"
    )


# =====================================================================
# CONVERT TF → 1D SEQUENCE
# =====================================================================

print()
print("=" * 70)
print("CONVERTING TF DATA TO 1D SEQUENCES")
print("=" * 70)


def tf_to_sequence(X):

    # Original:
    # (N, channels, frequency, time)

    # Move time to second dimension:
    # (N, time, channels, frequency)

    X = np.transpose(
        X,
        (0, 3, 1, 2)
    )

    # Flatten channels × frequency:
    # (N, time, channels*frequency)

    X = X.reshape(
        X.shape[0],
        X.shape[1],
        -1
    )

    return X.astype(
        np.float32
    )


X_train_seq = tf_to_sequence(X_train)

X_val_seq = tf_to_sequence(X_val)

X_test_seq = tf_to_sequence(X_test)


print()
print("Original TF shape:")
print(
    "(N, channels, frequency, time)"
)

print()
print("1D CNN sequence shape:")

print(
    "X_train:",
    X_train_seq.shape
)

print(
    "X_val:",
    X_val_seq.shape
)

print(
    "X_test:",
    X_test_seq.shape
)


# =====================================================================
# TRAINING-ONLY NORMALIZATION
# =====================================================================

print()
print("=" * 70)
print("TRAINING-ONLY NORMALIZATION")
print("=" * 70)


# Calculate statistics only from training set.

train_mean = X_train_seq.mean(
    axis=(0, 1),
    keepdims=True
)

train_std = X_train_seq.std(
    axis=(0, 1),
    keepdims=True
)

train_std = np.maximum(
    train_std,
    1e-6
)


X_train_seq = (
    X_train_seq - train_mean
) / train_std


X_val_seq = (
    X_val_seq - train_mean
) / train_std


X_test_seq = (
    X_test_seq - train_mean
) / train_std


print()
print("Normalization complete.")

print(
    "Mean shape:",
    train_mean.shape
)

print(
    "Std shape:",
    train_std.shape
)


# =====================================================================
# TORCH DATASETS
# =====================================================================

X_train_tensor = torch.tensor(
    X_train_seq,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.long
)

X_val_tensor = torch.tensor(
    X_val_seq,
    dtype=torch.float32
)

y_val_tensor = torch.tensor(
    y_val,
    dtype=torch.long
)

X_test_tensor = torch.tensor(
    X_test_seq,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.long
)


train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

val_dataset = TensorDataset(
    X_val_tensor,
    y_val_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)


# =====================================================================
# CLASS WEIGHTS
# =====================================================================

print()
print("=" * 70)
print("CLASS BALANCING")
print("=" * 70)


num_classes = len(VOCABULARY)

class_counts = np.bincount(
    y_train,
    minlength=num_classes
)


# Balanced weights.

raw_class_weights = (
    len(y_train)
    /
    (
        num_classes
        *
        np.maximum(class_counts, 1)
    )
)


# Moderate the weights.

# Square root prevents the majority/minority
# difference from becoming too aggressive.

class_weights = np.sqrt(
    raw_class_weights
)

class_weights = (
    class_weights
    /
    class_weights.mean()
)


for i, word in enumerate(VOCABULARY):

    print(
        f"{word:<10} "
        f"samples={class_counts[i]:<5} "
        f"weight={class_weights[i]:.4f}"
    )


# =====================================================================
# WEIGHTED RANDOM SAMPLER
# =====================================================================

print()
print("=" * 70)
print("CREATING BALANCED TRAINING SAMPLER")
print("=" * 70)


sample_weights = np.array(
    [
        class_weights[label]
        for label in y_train
    ],
    dtype=np.float64
)


sample_weights_tensor = torch.tensor(
    sample_weights,
    dtype=torch.double
)


sampler = WeightedRandomSampler(
    weights=sample_weights_tensor,
    num_samples=len(sample_weights),
    replacement=True
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =====================================================================
# MODEL
# =====================================================================

print()
print("=" * 70)
print("CREATING MODEL")
print("=" * 70)


class CNN1DProto24(nn.Module):

    def __init__(
        self,
        input_features,
        num_classes
    ):

        super().__init__()

        self.network = nn.Sequential(

            # ---------------------------------------------------------
            # Block 1
            # ---------------------------------------------------------

            nn.Conv1d(
                in_channels=input_features,
                out_channels=64,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),

            nn.Dropout(
                DROPOUT
            ),


            # ---------------------------------------------------------
            # Block 2
            # ---------------------------------------------------------

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.Conv1d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),

            nn.Dropout(
                DROPOUT
            ),


            # ---------------------------------------------------------
            # Block 3
            # ---------------------------------------------------------

            nn.Conv1d(
                in_channels=128,
                out_channels=192,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(192),

            nn.ReLU(),

            nn.Conv1d(
                in_channels=192,
                out_channels=192,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(192),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            )
        )


        # Global average pooling.

        self.global_pool = nn.AdaptiveAvgPool1d(
            1
        )


        # Classifier.

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                192,
                96
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                96,
                num_classes
            )
        )


    def forward(self, x):

        # Input:
        # (B, T, F)

        # Conv1D expects:
        # (B, F, T)

        x = x.transpose(
            1,
            2
        )

        x = self.network(x)

        x = self.global_pool(x)

        x = self.classifier(x)

        return x


model = CNN1DProto24(
    input_features=FEATURES_PER_TIMESTEP,
    num_classes=num_classes
)


model = model.to(DEVICE)


print()
print(model)


# =====================================================================
# PARAMETER COUNT
# =====================================================================

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print()
print(
    "Trainable parameters:",
    f"{trainable_parameters:,}"
)


# =====================================================================
# LOSS
# =====================================================================

print()
print("=" * 70)
print("LOSS FUNCTION")
print("=" * 70)


weight_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)


criterion = nn.CrossEntropyLoss(
    weight=weight_tensor,
    label_smoothing=LABEL_SMOOTHING
)


print()
print(
    "Label smoothing:",
    LABEL_SMOOTHING
)

print(
    "Moderate class weighting: enabled"
)


# =====================================================================
# OPTIMIZER
# =====================================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=5,
    min_lr=MIN_LR
)


# =====================================================================
# TRAINING FUNCTION
# =====================================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    predictions = []

    targets = []


    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(
            DEVICE
        )

        y_batch = y_batch.to(
            DEVICE
        )


        optimizer.zero_grad()


        outputs = model(
            X_batch
        )


        loss = criterion(
            outputs,
            y_batch
        )


        loss.backward()


        # Prevent exploding gradients.

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=2.0
        )


        optimizer.step()


        running_loss += (
            loss.item()
            *
            X_batch.size(0)
        )


        preds = torch.argmax(
            outputs,
            dim=1
        )


        predictions.extend(
            preds.detach()
            .cpu()
            .numpy()
        )


        targets.extend(
            y_batch.detach()
            .cpu()
            .numpy()
        )


    epoch_loss = (
        running_loss
        /
        len(train_dataset)
    )


    epoch_accuracy = accuracy_score(
        targets,
        predictions
    )


    epoch_macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )


    return (
        epoch_loss,
        epoch_accuracy,
        epoch_macro_f1
    )


# =====================================================================
# EVALUATION FUNCTION
# =====================================================================

def evaluate(loader):

    model.eval()

    running_loss = 0.0

    predictions = []

    targets = []


    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                DEVICE
            )

            y_batch = y_batch.to(
                DEVICE
            )


            outputs = model(
                X_batch
            )


            loss = criterion(
                outputs,
                y_batch
            )


            running_loss += (
                loss.item()
                *
                X_batch.size(0)
            )


            preds = torch.argmax(
                outputs,
                dim=1
            )


            predictions.extend(
                preds.cpu()
                .numpy()
            )


            targets.extend(
                y_batch.cpu()
                .numpy()
            )


    epoch_loss = (
        running_loss
        /
        len(loader.dataset)
    )


    epoch_accuracy = accuracy_score(
        targets,
        predictions
    )


    balanced_accuracy = balanced_accuracy_score(
        targets,
        predictions
    )


    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0
    )


    return (
        epoch_loss,
        epoch_accuracy,
        balanced_accuracy,
        macro_f1,
        np.asarray(predictions),
        np.asarray(targets)
    )


# =====================================================================
# TRAINING
# =====================================================================

print()
print("=" * 70)
print("TRAINING")
print("=" * 70)


history = {

    "train_loss": [],
    "train_acc": [],
    "train_f1": [],

    "val_loss": [],
    "val_acc": [],
    "val_balanced_acc": [],
    "val_f1": [],

    "lr": []
}


best_val_f1 = -1.0

best_val_accuracy = 0.0

best_epoch = 0

epochs_without_improvement = 0


for epoch in range(
    1,
    EPOCHS + 1
):


    train_loss, train_acc, train_f1 = (
        train_one_epoch()
    )


    (
        val_loss,
        val_acc,
        val_balanced_acc,
        val_f1,
        _,
        _
    ) = evaluate(
        val_loader
    )


    current_lr = optimizer.param_groups[0]["lr"]


    history["train_loss"].append(
        train_loss
    )

    history["train_acc"].append(
        train_acc
    )

    history["train_f1"].append(
        train_f1
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_acc"].append(
        val_acc
    )

    history["val_balanced_acc"].append(
        val_balanced_acc
    )

    history["val_f1"].append(
        val_f1
    )

    history["lr"].append(
        current_lr
    )


    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss {train_loss:.4f} | "
        f"Train Acc {train_acc:.4f} | "
        f"Train F1 {train_f1:.4f} | "
        f"Val Loss {val_loss:.4f} | "
        f"Val Acc {val_acc:.4f} | "
        f"Val Bal Acc {val_balanced_acc:.4f} | "
        f"Val F1 {val_f1:.4f} | "
        f"LR {current_lr:.2e}"
    )


    # ---------------------------------------------------------------
    # Scheduler uses validation Macro-F1
    # ---------------------------------------------------------------

    scheduler.step(
        val_f1
    )


    # ---------------------------------------------------------------
    # Save best model using Macro-F1
    # ---------------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        best_val_accuracy = val_acc

        best_epoch = epoch

        epochs_without_improvement = 0


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "vocabulary":
                    VOCABULARY,

                "input_features":
                    FEATURES_PER_TIMESTEP,

                "n_channels":
                    N_CHANNELS,

                "n_frequency_bins":
                    N_FREQ,

                "n_time_frames":
                    N_TIME,

                "train_mean":
                    train_mean,

                "train_std":
                    train_std,

                "best_val_f1":
                    best_val_f1,

                "best_val_accuracy":
                    best_val_accuracy,

                "epoch":
                    best_epoch
            },
            MODEL_PATH
        )


        print(
            "  -> Best model saved."
        )


    else:

        epochs_without_improvement += 1


    # ---------------------------------------------------------------
    # Early stopping
    # ---------------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print()
        print(
            "Early stopping triggered."
        )

        break


# =====================================================================
# LOAD BEST MODEL
# =====================================================================

print()
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


print()
print(
    "Best epoch:",
    checkpoint["epoch"]
)

print(
    "Best validation Macro F1:",
    checkpoint["best_val_f1"]
)

print(
    "Best validation accuracy:",
    checkpoint["best_val_accuracy"]
)


# =====================================================================
# FINAL VALIDATION
# =====================================================================

print()
print("=" * 70)
print("FINAL VALIDATION")
print("=" * 70)


(
    val_loss,
    val_accuracy,
    val_balanced_accuracy,
    val_macro_f1,
    val_predictions,
    val_targets
) = evaluate(
    val_loader
)


print()
print(
    f"Validation loss: "
    f"{val_loss:.4f}"
)

print(
    f"Validation accuracy: "
    f"{val_accuracy:.4f}"
)

print(
    f"Validation balanced accuracy: "
    f"{val_balanced_accuracy:.4f}"
)

print(
    f"Validation Macro F1: "
    f"{val_macro_f1:.4f}"
)


# =====================================================================
# FINAL TEST
# =====================================================================

print()
print("=" * 70)
print("FINAL TEST")
print("=" * 70)


(
    test_loss,
    test_accuracy,
    test_balanced_accuracy,
    test_macro_f1,
    test_predictions,
    test_targets
) = evaluate(
    test_loader
)


print()
print(
    f"Test loss: "
    f"{test_loss:.4f}"
)

print(
    f"Test accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Test accuracy (%): "
    f"{test_accuracy * 100:.2f}%"
)

print()

print(
    f"Test balanced accuracy: "
    f"{test_balanced_accuracy:.4f}"
)

print(
    f"Test balanced accuracy (%): "
    f"{test_balanced_accuracy * 100:.2f}%"
)

print()

print(
    f"Test Macro F1: "
    f"{test_macro_f1:.4f}"
)

print(
    f"Test Macro F1 (%): "
    f"{test_macro_f1 * 100:.2f}%"
)


# =====================================================================
# PREDICTION SANITY CHECK
# =====================================================================

print()
print("=" * 70)
print("PREDICTION SANITY CHECK")
print("=" * 70)


print()
print("True class distribution:")


for i, word in enumerate(VOCABULARY):

    count = np.sum(
        test_targets == i
    )

    print(
        f"{word:<10}: {count}"
    )


print()
print("Predicted class distribution:")


for i, word in enumerate(VOCABULARY):

    count = np.sum(
        test_predictions == i
    )

    print(
        f"{word:<10}: {count}"
    )


# =====================================================================
# COLLAPSE DETECTION
# =====================================================================

unique_predictions = np.unique(
    test_predictions
)


print()

if len(unique_predictions) == 1:

    predicted_word = VOCABULARY[
        unique_predictions[0]
    ]

    print(
        "WARNING:"
    )

    print(
        "Model collapsed to a single class:"
    )

    print(
        predicted_word
    )

    print(
        "This experiment should NOT be considered "
        "a successful classifier."
    )

elif len(unique_predictions) < num_classes:

    missing = [
        VOCABULARY[i]
        for i in range(num_classes)
        if i not in unique_predictions
    ]

    print(
        "WARNING:"
    )

    print(
        "Model did not predict every class."
    )

    print(
        "Missing predicted classes:",
        missing
    )

else:

    print(
        "Prediction sanity check:"
    )

    print(
        "All 5 classes were predicted."
    )


# =====================================================================
# CLASSIFICATION REPORT
# =====================================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)


report = classification_report(
    test_targets,
    test_predictions,
    labels=np.arange(num_classes),
    target_names=VOCABULARY,
    zero_division=0
)


print(
    report
)


# =====================================================================
# CONFUSION MATRIX
# =====================================================================

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)


cm = confusion_matrix(
    test_targets,
    test_predictions,
    labels=np.arange(num_classes)
)


print()
print(cm)


fig = plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Berkeley 5-Word 1D CNN Proto 2.4"
)

plt.colorbar()


tick_marks = np.arange(
    num_classes
)

plt.xticks(
    tick_marks,
    VOCABULARY,
    rotation=45
)

plt.yticks(
    tick_marks,
    VOCABULARY
)


threshold = cm.max() / 2.0


for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            horizontalalignment="center",
            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            )
        )


plt.ylabel(
    "True label"
)

plt.xlabel(
    "Predicted label"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_PATH,
    dpi=200
)

plt.close()


print()
print(
    "Confusion matrix saved:"
)

print(
    CONFUSION_PATH
)


# =====================================================================
# TRAINING CURVES
# =====================================================================

print()
print("=" * 70)
print("SAVING TRAINING CURVES")
print("=" * 70)


epochs_completed = len(
    history["train_loss"]
)


epoch_numbers = np.arange(
    1,
    epochs_completed + 1
)


# ---------------------------------------------------------------------
# Loss plot
# ---------------------------------------------------------------------

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    epoch_numbers,
    history["train_loss"],
    label="Train Loss"
)

plt.plot(
    epoch_numbers,
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "1D CNN Training / Validation Loss"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    TRAINING_PLOT_PATH,
    dpi=200
)

plt.close()


print()
print(
    "Training plot saved:"
)

print(
    TRAINING_PLOT_PATH
)


# ---------------------------------------------------------------------
# Accuracy plot
# ---------------------------------------------------------------------

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    epoch_numbers,
    history["train_acc"],
    label="Train Accuracy"
)

plt.plot(
    epoch_numbers,
    history["val_acc"],
    label="Validation Accuracy"
)

plt.plot(
    epoch_numbers,
    history["val_balanced_acc"],
    label="Validation Balanced Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Score"
)

plt.title(
    "1D CNN Accuracy"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    ACCURACY_PLOT_PATH,
    dpi=200
)

plt.close()


print()
print(
    "Accuracy plot saved:"
)

print(
    ACCURACY_PLOT_PATH
)


# =====================================================================
# FINAL SUMMARY
# =====================================================================

print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print()

print("Vocabulary:")

for i, word in enumerate(VOCABULARY):

    print(
        f"  {i}: {word}"
    )


print()

print("Original TF input:")

print(
    "  Channels:",
    N_CHANNELS
)

print(
    "  Frequency bins:",
    N_FREQ
)

print(
    "  Time frames:",
    N_TIME
)


print()

print("1D CNN input:")

print(
    "  Timesteps:",
    N_TIME
)

print(
    "  Features per timestep:",
    FEATURES_PER_TIMESTEP
)


print()

print(
    "Training samples:",
    len(X_train_seq)
)

print(
    "Validation samples:",
    len(X_val_seq)
)

print(
    "Test samples:",
    len(X_test_seq)
)


print()

print(
    f"Best validation Macro F1: "
    f"{best_val_f1 * 100:.2f}%"
)

print(
    f"Final validation accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

print(
    f"Final test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Final test balanced accuracy: "
    f"{test_balanced_accuracy * 100:.2f}%"
)

print(
    f"Final test Macro F1: "
    f"{test_macro_f1 * 100:.2f}%"
)


print()

print(
    "Model:"
)

print(
    MODEL_PATH
)


print()
print("=" * 70)
print("DONE")
print("=" * 70)