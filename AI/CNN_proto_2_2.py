
"""
======================================================================
IRIS - BERKELEY 5-WORD TIME-FREQUENCY CNN
======================================================================

Dataset:
    berkeley_5word_tf_augmented_dataset.npz

Vocabulary:
    0 = THE
    1 = AND
    2 = A
    3 = OF
    4 = I

Input:
    (B, 8, 24, 64)

Meaning:
    B  = batch
    8  = EMG channels
    24 = frequency bins
    64 = time frames

Model:
    2D CNN

IMPORTANT:
    The 8 EMG channels are treated as CNN input channels.
    The CNN convolves over frequency x time.

Features:
    - Class-weighted loss
    - Batch normalization
    - Dropout
    - AdamW optimizer
    - Learning-rate scheduler
    - Early stopping
    - Best-model checkpoint
    - Validation monitoring
    - Test accuracy
    - Classification report
    - Confusion matrix
======================================================================
"""

import os
import random
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# =====================================================================
# CONFIGURATION
# =====================================================================

PROJECT_ROOT = Path(
    "/Users/vedantdwivedi/Desktop/IRIS"
)

DATASET_PATH = (
    PROJECT_ROOT
    / "Data"
    / "datasets"
    / "berkeley_5word_tf_augmented_dataset.npz"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "AI"
    / "berkeley_5word_tf_cnn_best.pt"
)

CONFUSION_MATRIX_PATH = (
    PROJECT_ROOT
    / "AI"
    / "berkeley_5word_tf_cnn_confusion_matrix.png"
)

LOSS_PLOT_PATH = (
    PROJECT_ROOT
    / "AI"
    / "berkeley_5word_tf_cnn_training.png"
)


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =====================================================================
# TRAINING SETTINGS
# =====================================================================

BATCH_SIZE = 32

EPOCHS = 100

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

DROPOUT = 0.30

PATIENCE = 15

NUM_WORKERS = 0


# =====================================================================
# DEVICE
# =====================================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

elif torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

else:

    DEVICE = torch.device("cpu")


# =====================================================================
# VOCABULARY
# =====================================================================

CLASS_NAMES = [
    "THE",
    "AND",
    "A",
    "OF",
    "I"
]

NUM_CLASSES = len(CLASS_NAMES)


# =====================================================================
# HEADER
# =====================================================================

print()
print("=" * 70)
print("IRIS - BERKELEY 5-WORD TIME-FREQUENCY CNN")
print("=" * 70)

print()
print("Dataset:")
print(DATASET_PATH)

print()
print("Device:")
print(DEVICE)

print()
print("Vocabulary:")

for i, word in enumerate(CLASS_NAMES):

    print(
        f"  {i}: {word}"
    )


# =====================================================================
# LOAD DATASET
# =====================================================================

print()
print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

if not DATASET_PATH.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_PATH}"
    )


data = np.load(
    DATASET_PATH,
    allow_pickle=True
)


print()
print("Dataset keys:")
print(data.files)


X_train = data["X_train"].astype(
    np.float32
)

y_train = data["y_train"].astype(
    np.int64
)

X_val = data["X_val"].astype(
    np.float32
)

y_val = data["y_val"].astype(
    np.int64
)

X_test = data["X_test"].astype(
    np.float32
)

y_test = data["y_test"].astype(
    np.int64
)


# =====================================================================
# DATASET SHAPES
# =====================================================================

print()
print("=" * 70)
print("DATASET SHAPES")
print("=" * 70)

print()
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print()
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print()
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# =====================================================================
# EXPECTED INPUT CHECK
# =====================================================================

if X_train.ndim != 4:

    raise RuntimeError(
        f"Expected X_train to have 4 dimensions.\n"
        f"Got: {X_train.shape}"
    )


if X_train.shape[1:] != (8, 24, 64):

    raise RuntimeError(
        f"Expected input shape (8, 24, 64).\n"
        f"Got: {X_train.shape[1:]}"
    )


# =====================================================================
# NUMERICAL VALIDATION
# =====================================================================

print()
print("=" * 70)
print("NUMERICAL VALIDATION")
print("=" * 70)

print()
print(
    "Training NaN:",
    np.isnan(X_train).sum()
)

print(
    "Training Inf:",
    np.isinf(X_train).sum()
)

print(
    "Validation NaN:",
    np.isnan(X_val).sum()
)

print(
    "Validation Inf:",
    np.isinf(X_val).sum()
)

print(
    "Test NaN:",
    np.isnan(X_test).sum()
)

print(
    "Test Inf:",
    np.isinf(X_test).sum()
)


if np.isnan(X_train).any() or np.isinf(X_train).any():

    raise RuntimeError(
        "Invalid values detected in training data."
    )


# =====================================================================
# CLASS DISTRIBUTION
# =====================================================================

print()
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)


for i, word in enumerate(CLASS_NAMES):

    train_count = np.sum(
        y_train == i
    )

    val_count = np.sum(
        y_val == i
    )

    test_count = np.sum(
        y_test == i
    )

    print(
        f"{word:<10}"
        f" Train: {train_count:<5}"
        f" Val: {val_count:<5}"
        f" Test: {test_count}"
    )


# =====================================================================
# STANDARDIZATION
# =====================================================================
#
# IMPORTANT:
# Calculate statistics ONLY from training data.
#
# This avoids information leakage from validation/test.
#
# We normalize each EMG channel independently.
# =====================================================================

print()
print("=" * 70)
print("TRAINING-ONLY NORMALIZATION")
print("=" * 70)


# Mean/std across:
# samples, frequency, time
#
# Keeping channel dimension.

train_mean = X_train.mean(
    axis=(0, 2, 3),
    keepdims=True
)

train_std = X_train.std(
    axis=(0, 2, 3),
    keepdims=True
)

train_std = np.maximum(
    train_std,
    1e-6
)


X_train = (
    X_train - train_mean
) / train_std


X_val = (
    X_val - train_mean
) / train_std


X_test = (
    X_test - train_mean
) / train_std


print(
    "Normalization complete."
)

print(
    "Mean shape:",
    train_mean.shape
)

print(
    "Std shape:",
    train_std.shape
)


# =====================================================================
# CONVERT TO PYTORCH
# =====================================================================

X_train_tensor = torch.from_numpy(
    X_train
)

y_train_tensor = torch.from_numpy(
    y_train
)

X_val_tensor = torch.from_numpy(
    X_val
)

y_val_tensor = torch.from_numpy(
    y_val
)

X_test_tensor = torch.from_numpy(
    X_test
)

y_test_tensor = torch.from_numpy(
    y_test
)


# =====================================================================
# DATASETS
# =====================================================================

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
# DATALOADERS
# =====================================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# =====================================================================
# CLASS WEIGHTS
# =====================================================================

print()
print("=" * 70)
print("CLASS WEIGHTS")
print("=" * 70)


class_counts = np.bincount(
    y_train,
    minlength=NUM_CLASSES
)

total_train = len(
    y_train
)


class_weights = (
    total_train
    /
    (
        NUM_CLASSES
        * class_counts
    )
)


for i, word in enumerate(CLASS_NAMES):

    print(
        f"{word:<10}"
        f" samples={class_counts[i]:<5}"
        f" weight={class_weights[i]:.4f}"
    )


class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)


# =====================================================================
# MODEL
# =====================================================================

class TF_CNN(nn.Module):

    def __init__(
        self,
        num_classes=5,
        dropout=0.30
    ):

        super().__init__()


        # --------------------------------------------------------------
        # BLOCK 1
        # --------------------------------------------------------------

        self.block1 = nn.Sequential(

            nn.Conv2d(
                in_channels=8,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            nn.Dropout2d(
                dropout
            )
        )


        # Input:
        # 24 x 64
        #
        # After pool:
        # 12 x 32


        # --------------------------------------------------------------
        # BLOCK 2
        # --------------------------------------------------------------

        self.block2 = nn.Sequential(

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            nn.Dropout2d(
                dropout
            )
        )


        # Input:
        # 12 x 32
        #
        # After pool:
        # 6 x 16


        # --------------------------------------------------------------
        # BLOCK 3
        # --------------------------------------------------------------

        self.block3 = nn.Sequential(

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )


        # --------------------------------------------------------------
        # CLASSIFIER
        # --------------------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                64,
                num_classes
            )
        )


    def forward(self, x):

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.classifier(x)

        return x


# =====================================================================
# CREATE MODEL
# =====================================================================

print()
print("=" * 70)
print("CREATING MODEL")
print("=" * 70)


model = TF_CNN(
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
).to(DEVICE)


print()
print(model)


# =====================================================================
# PARAMETER COUNT
# =====================================================================

num_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print()
print(
    "Trainable parameters:",
    f"{num_parameters:,}"
)


# =====================================================================
# LOSS
# =====================================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor
)


# =====================================================================
# OPTIMIZER
# =====================================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# =====================================================================
# LEARNING RATE SCHEDULER
# =====================================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5
)


# =====================================================================
# TRAINING FUNCTION
# =====================================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    total_loss = 0.0

    correct = 0

    total = 0


    for X_batch, y_batch in loader:

        X_batch = X_batch.to(
            DEVICE
        )

        y_batch = y_batch.to(
            DEVICE
        )


        optimizer.zero_grad()


        logits = model(
            X_batch
        )


        loss = criterion(
            logits,
            y_batch
        )


        loss.backward()


        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )


        optimizer.step()


        total_loss += (
            loss.item()
            * X_batch.size(0)
        )


        predictions = torch.argmax(
            logits,
            dim=1
        )


        correct += (
            predictions == y_batch
        ).sum().item()


        total += y_batch.size(0)


    avg_loss = (
        total_loss / total
    )

    accuracy = (
        correct / total
    )


    return avg_loss, accuracy


# =====================================================================
# VALIDATION FUNCTION
# =====================================================================

def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    correct = 0

    total = 0


    all_predictions = []

    all_labels = []


    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                DEVICE
            )

            y_batch = y_batch.to(
                DEVICE
            )


            logits = model(
                X_batch
            )


            loss = criterion(
                logits,
                y_batch
            )


            total_loss += (
                loss.item()
                * X_batch.size(0)
            )


            predictions = torch.argmax(
                logits,
                dim=1
            )


            correct += (
                predictions == y_batch
            ).sum().item()


            total += y_batch.size(0)


            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                y_batch.cpu().numpy()
            )


    avg_loss = (
        total_loss / total
    )

    accuracy = (
        correct / total
    )


    return (
        avg_loss,
        accuracy,
        np.array(all_labels),
        np.array(all_predictions)
    )


# =====================================================================
# TRAINING
# =====================================================================

print()
print("=" * 70)
print("TRAINING")
print("=" * 70)


best_val_loss = float(
    "inf"
)

best_val_accuracy = 0.0

epochs_without_improvement = 0


history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}


MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


for epoch in range(
    1,
    EPOCHS + 1
):


    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )


    val_loss, val_accuracy, _, _ = evaluate(
        model,
        val_loader,
        criterion
    )


    scheduler.step(
        val_loss
    )


    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_accuracy"].append(
        val_accuracy
    )


    current_lr = optimizer.param_groups[0]["lr"]


    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss {train_loss:.4f} | "
        f"Train Acc {train_accuracy:.4f} | "
        f"Val Loss {val_loss:.4f} | "
        f"Val Acc {val_accuracy:.4f} | "
        f"LR {current_lr:.2e}"
    )


    # --------------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_val_accuracy = val_accuracy

        epochs_without_improvement = 0


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "classes":
                    CLASS_NAMES,

                "train_mean":
                    train_mean,

                "train_std":
                    train_std,

                "best_val_loss":
                    best_val_loss,

                "best_val_accuracy":
                    best_val_accuracy,

                "epoch":
                    epoch,

                "input_shape":
                    (8, 24, 64),

                "model_type":
                    "2D CNN"
            },
            MODEL_PATH
        )


        print(
            "  -> Best model saved."
        )


    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

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


# IMPORTANT:
# weights_only=False is intentional because the checkpoint contains
# numpy arrays and metadata in addition to the PyTorch state_dict.
#
# This fixes the PyTorch 2.6+ loading error encountered previously.

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
    "Best validation loss:",
    checkpoint["best_val_loss"]
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


val_loss, val_accuracy, val_labels, val_predictions = evaluate(
    model,
    val_loader,
    criterion
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


# =====================================================================
# FINAL TEST
# =====================================================================

print()
print("=" * 70)
print("FINAL TEST")
print("=" * 70)


test_loss, test_accuracy, test_labels, test_predictions = evaluate(
    model,
    test_loader,
    criterion
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


# =====================================================================
# CLASSIFICATION REPORT
# =====================================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)


print(
    classification_report(
        test_labels,
        test_predictions,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# =====================================================================
# CONFUSION MATRIX
# =====================================================================

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)


cm = confusion_matrix(
    test_labels,
    test_predictions,
    labels=np.arange(NUM_CLASSES)
)


print()
print(cm)


# =====================================================================
# PLOT CONFUSION MATRIX
# =====================================================================

fig, ax = plt.subplots(
    figsize=(7, 6)
)


im = ax.imshow(
    cm
)


ax.set_xticks(
    np.arange(NUM_CLASSES)
)

ax.set_yticks(
    np.arange(NUM_CLASSES)
)


ax.set_xticklabels(
    CLASS_NAMES
)

ax.set_yticklabels(
    CLASS_NAMES
)


ax.set_xlabel(
    "Predicted"
)

ax.set_ylabel(
    "Actual"
)

ax.set_title(
    "Berkeley 5-Word TF CNN Confusion Matrix"
)


for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


fig.colorbar(
    im,
    ax=ax
)


plt.tight_layout()


plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=200
)


plt.close()


print()
print(
    "Confusion matrix saved:"
)

print(
    CONFUSION_MATRIX_PATH
)


# =====================================================================
# TRAINING CURVES
# =====================================================================

print()
print("=" * 70)
print("SAVING TRAINING CURVES")
print("=" * 70)


fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    history["train_loss"],
    label="Train Loss"
)

ax.plot(
    history["val_loss"],
    label="Validation Loss"
)

ax.set_xlabel(
    "Epoch"
)

ax.set_ylabel(
    "Loss"
)

ax.set_title(
    "CNN Training / Validation Loss"
)

ax.legend()

plt.tight_layout()

plt.savefig(
    LOSS_PLOT_PATH,
    dpi=200
)

plt.close()


print(
    "Training plot saved:"
)

print(
    LOSS_PLOT_PATH
)


# =====================================================================
# FINAL SUMMARY
# =====================================================================

print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print()
print(
    "Vocabulary:"
)

for i, word in enumerate(CLASS_NAMES):

    print(
        f"  {i}: {word}"
    )


print()
print(
    "Input shape:"
)

print(
    "(8, 24, 64)"
)


print()
print(
    "Training samples:",
    len(X_train)
)

print(
    "Validation samples:",
    len(X_val)
)

print(
    "Test samples:",
    len(X_test)
)


print()
print(
    "Best validation accuracy:",
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    "Final test accuracy:",
    f"{test_accuracy * 100:.2f}%"
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

