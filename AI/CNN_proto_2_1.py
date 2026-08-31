
"""
======================================================================
IRIS - CNN PROTO 2
Berkeley Silent Speech Recognition
Time-Frequency CNN
======================================================================

Dataset:
    berkeley_tf_augmented_dataset.npz

Input:
    (N, 8, 24, 64)

    N      = samples
    8      = EMG channels
    24     = frequency bins
    64     = time frames

The dataset has already been augmented:
    Original
    Speed perturbation
    Time masking
    Frequency masking

Therefore this script DOES NOT perform additional augmentation.

Features:
    - Training-only normalization
    - CNN classifier
    - Class-safe loading
    - Early stopping
    - ReduceLROnPlateau
    - Best checkpoint saving
    - Test evaluation
    - Classification report
    - Confusion matrix
    - PyTorch 2.6-compatible checkpoint loading
======================================================================
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt


# ======================================================================
# CONFIGURATION
# ======================================================================

DATASET_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/Data/datasets/"
    "berkeley_tf_augmented_dataset.npz"
)

MODEL_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/AI/"
    "CNN_proto2_best.pt"
)

REPORT_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/AI/"
    "CNN_proto2_classification_report.txt"
)

CONFUSION_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/AI/"
    "CNN_proto2_confusion_matrix.png"
)


# Training settings
BATCH_SIZE = 32
EPOCHS = 100

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

PATIENCE = 15
LR_PATIENCE = 5

NUM_WORKERS = 0

RANDOM_STATE = 42


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


# ======================================================================
# DEVICE
# ======================================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


print("=" * 70)
print("IRIS - CNN PROTO 2")
print("BERKELEY SILENT SPEECH TIME-FREQUENCY CNN")
print("=" * 70)

print()
print("Device:", DEVICE)
print()


# ======================================================================
# LOAD DATASET
# ======================================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_PATH}\n"
    )

data = np.load(DATASET_PATH, allow_pickle=True)

print("\nDataset keys:")
print(data.files)


# ======================================================================
# LOAD ARRAYS
# ======================================================================

X_train = data["X_train"].astype(np.float32)
y_train_raw = data["y_train"]

X_val = data["X_val"].astype(np.float32)
y_val_raw = data["y_val"]

X_test = data["X_test"].astype(np.float32)
y_test_raw = data["y_test"]


# ======================================================================
# LOAD CLASS LIST
# ======================================================================

if "classes" in data.files:
    classes_raw = data["classes"]
else:
    classes_raw = np.unique(
        np.concatenate(
            [
                y_train_raw,
                y_val_raw,
                y_test_raw
            ]
        )
    )

classes = [str(x) for x in classes_raw]

print("\nClasses found:", len(classes))


# ======================================================================
# CLASS ENCODING
# ======================================================================

print()
print("=" * 70)
print("ENCODING LABELS")
print("=" * 70)

class_to_idx = {
    cls: i for i, cls in enumerate(classes)
}


def encode_labels(labels):
    encoded = []

    for label in labels:
        label = str(label)

        if label not in class_to_idx:
            raise ValueError(
                f"Unknown class '{label}' encountered."
            )

        encoded.append(class_to_idx[label])

    return np.asarray(encoded, dtype=np.int64)


y_train = encode_labels(y_train_raw)
y_val = encode_labels(y_val_raw)
y_test = encode_labels(y_test_raw)


# ======================================================================
# DATASET SHAPE CHECK
# ======================================================================

print("\nDataset shapes:")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


if X_train.ndim != 4:
    raise ValueError(
        f"Expected X_train to have 4 dimensions, "
        f"got {X_train.ndim}"
    )


channels = X_train.shape[1]
frequency_bins = X_train.shape[2]
time_frames = X_train.shape[3]

print()
print("Channels:", channels)
print("Frequency bins:", frequency_bins)
print("Time frames:", time_frames)


# ======================================================================
# NUMERICAL VALIDATION
# ======================================================================

print()
print("=" * 70)
print("NUMERICAL VALIDATION")
print("=" * 70)

print("\nNaN values:")
print("Train:", np.isnan(X_train).sum())
print("Val:  ", np.isnan(X_val).sum())
print("Test: ", np.isnan(X_test).sum())

print("\nInfinite values:")
print("Train:", np.isinf(X_train).sum())
print("Val:  ", np.isinf(X_val).sum())
print("Test: ", np.isinf(X_test).sum())


if not np.isfinite(X_train).all():
    raise ValueError("X_train contains NaN or infinite values.")

if not np.isfinite(X_val).all():
    raise ValueError("X_val contains NaN or infinite values.")

if not np.isfinite(X_test).all():
    raise ValueError("X_test contains NaN or infinite values.")


# ======================================================================
# CLASS DISTRIBUTION
# ======================================================================

print()
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)


def print_distribution(labels, name):
    unique, counts = np.unique(
        labels,
        return_counts=True
    )

    print()
    print(name)
    print("-" * 50)

    pairs = sorted(
        zip(unique, counts),
        key=lambda x: x[1],
        reverse=True
    )

    for idx, count in pairs[:30]:
        print(
            f"{classes[idx]:20s} {count:5d}"
        )


print_distribution(y_train, "TRAIN")
print_distribution(y_val, "VALIDATION")
print_distribution(y_test, "TEST")


# ======================================================================
# TRAINING-ONLY NORMALIZATION
# ======================================================================

print()
print("=" * 70)
print("NORMALIZATION")
print("=" * 70)

print("\nCalculating normalization statistics from TRAINING DATA ONLY...")


# Per-channel normalization.
#
# Shape:
#   X_train = (N, C, F, T)
#
# Mean/std:
#   (1, C, 1, 1)

train_mean = X_train.mean(
    axis=(0, 2, 3),
    keepdims=True
)

train_std = X_train.std(
    axis=(0, 2, 3),
    keepdims=True
)

# Avoid division by zero.
train_std[train_std < 1e-6] = 1.0


X_train = (
    X_train - train_mean
) / train_std

X_val = (
    X_val - train_mean
) / train_std

X_test = (
    X_test - train_mean
) / train_std


print("\nNormalization complete.")

print(
    "Training mean after normalization:",
    float(X_train.mean())
)

print(
    "Training std after normalization:",
    float(X_train.std())
)


# ======================================================================
# PYTORCH DATASET
# ======================================================================

class TFEMGDataset(Dataset):
    """
    Dataset for time-frequency EMG representations.

    Input:
        X -> (N, C, F, T)

    Output:
        tensor -> (C, F, T)
        label  -> integer
    """

    def __init__(self, X, y):
        self.X = torch.from_numpy(
            X.astype(np.float32)
        )

        self.y = torch.from_numpy(
            y.astype(np.int64)
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = TFEMGDataset(
    X_train,
    y_train
)

val_dataset = TFEMGDataset(
    X_val,
    y_val
)

test_dataset = TFEMGDataset(
    X_test,
    y_test
)


# ======================================================================
# DATALOADERS
# ======================================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False
)


print()
print("=" * 70)
print("DATALOADERS")
print("=" * 70)

print("\nTraining batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Test batches:", len(test_loader))


# ======================================================================
# CNN MODEL
# ======================================================================

class TFEMGCNN(nn.Module):
    """
    CNN for 2-D time-frequency EMG representations.

    Input:
        (B, 8, 24, 64)

    Output:
        (B, number_of_classes)
    """

    def __init__(
        self,
        input_channels,
        num_classes
    ):
        super().__init__()

        self.features = nn.Sequential(

            # ----------------------------------------------------------
            # BLOCK 1
            # ----------------------------------------------------------

            nn.Conv2d(
                input_channels,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            nn.Dropout2d(0.10),

            # ----------------------------------------------------------
            # BLOCK 2
            # ----------------------------------------------------------

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            nn.Dropout2d(0.15),

            # ----------------------------------------------------------
            # BLOCK 3
            # ----------------------------------------------------------

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            nn.Dropout2d(0.20),

            # ----------------------------------------------------------
            # BLOCK 4
            # ----------------------------------------------------------

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(),

            # Adaptive pooling avoids hardcoding
            # the flattened dimension.

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.40),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ======================================================================
# CREATE MODEL
# ======================================================================

num_classes = len(classes)

model = TFEMGCNN(
    input_channels=channels,
    num_classes=num_classes
)

model = model.to(DEVICE)


# ======================================================================
# MODEL SUMMARY
# ======================================================================

print()
print("=" * 70)
print("MODEL")
print("=" * 70)

print(model)


parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print()
print("Total parameters:", parameter_count)
print("Trainable parameters:", trainable_parameters)


# ======================================================================
# LOSS
# ======================================================================

criterion = nn.CrossEntropyLoss()


# ======================================================================
# OPTIMIZER
# ======================================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ======================================================================
# LEARNING RATE SCHEDULER
# ======================================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=LR_PATIENCE
)


# ======================================================================
# TRAINING FUNCTIONS
# ======================================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(
            outputs,
            y_batch
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * X_batch.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == y_batch
        ).sum().item()

        total += X_batch.size(0)

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ======================================================================
# VALIDATION
# ======================================================================

def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            outputs = model(X_batch)

            loss = criterion(
                outputs,
                y_batch
            )

            running_loss += (
                loss.item()
                * X_batch.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == y_batch
            ).sum().item()

            total += X_batch.size(0)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                y_batch.cpu().numpy()
            )

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return (
        epoch_loss,
        epoch_accuracy,
        np.asarray(all_labels),
        np.asarray(all_predictions)
    )


# ======================================================================
# TRAINING
# ======================================================================

print()
print("=" * 70)
print("TRAINING")
print("=" * 70)

print()
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
print("Weight decay:", WEIGHT_DECAY)
print("Early stopping patience:", PATIENCE)


best_val_loss = float("inf")

best_val_accuracy = 0.0

epochs_without_improvement = 0

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}


for epoch in range(1, EPOCHS + 1):

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

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

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

    print(
        f"Epoch {epoch:03d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy * 100:.2f}% | "
        f"LR: {current_lr:.2e}"
    )

    # --------------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_val_accuracy = val_accuracy

        epochs_without_improvement = 0

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_accuracy,
            "classes": classes,
            "train_mean": train_mean,
            "train_std": train_std,
            "input_shape": (
                channels,
                frequency_bins,
                time_frames
            )
        }

        torch.save(
            checkpoint,
            MODEL_PATH
        )

        print(
            f"  -> Best model saved "
            f"(Val Acc: {val_accuracy * 100:.2f}%)"
        )

    else:

        epochs_without_improvement += 1

    # --------------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print()
        print(
            f"Early stopping triggered after "
            f"{epoch} epochs."
        )

        break


# ======================================================================
# LOAD BEST CHECKPOINT
# ======================================================================

print()
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Best checkpoint was not created:\n{MODEL_PATH}"
    )


# IMPORTANT:
#
# PyTorch 2.6 changed torch.load() so that
# weights_only=True is the default.
#
# Our checkpoint contains NumPy objects such as
# train_mean and train_std.
#
# Since this checkpoint was created locally by this script,
# weights_only=False is safe here.

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
    f"{checkpoint['best_val_accuracy'] * 100:.2f}%"
)


# ======================================================================
# TEST SET
# ======================================================================

print()
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

test_loss, test_accuracy, test_labels, test_predictions = evaluate(
    model,
    test_loader,
    criterion
)

print()
print(
    f"Test loss: {test_loss:.4f}"
)

print(
    f"Test accuracy: {test_accuracy * 100:.2f}%"
)


# ======================================================================
# CLASSIFICATION REPORT
# ======================================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

# Only report classes actually represented in the test set.
present_labels = np.unique(test_labels)

present_names = [
    classes[i]
    for i in present_labels
]

report = classification_report(
    test_labels,
    test_predictions,
    labels=present_labels,
    target_names=present_names,
    zero_division=0
)

print()
print(report)


# ======================================================================
# SAVE CLASSIFICATION REPORT
# ======================================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "IRIS - CNN PROTO 2\n"
    )

    f.write(
        "Berkeley Silent Speech Recognition\n\n"
    )

    f.write(
        f"Best epoch: {checkpoint['epoch']}\n"
    )

    f.write(
        f"Best validation accuracy: "
        f"{checkpoint['best_val_accuracy'] * 100:.2f}%\n"
    )

    f.write(
        f"Test loss: {test_loss:.4f}\n"
    )

    f.write(
        f"Test accuracy: "
        f"{test_accuracy * 100:.2f}%\n\n"
    )

    f.write(
        report
    )


# ======================================================================
# CONFUSION MATRIX
# ======================================================================

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    test_labels,
    test_predictions,
    labels=present_labels
)


# Save a figure.
#
# For 41 classes, labels are intentionally kept small.

plt.figure(
    figsize=(16, 14)
)

plt.imshow(cm)

plt.title(
    "CNN Proto 2 - Test Confusion Matrix"
)

plt.xlabel(
    "Predicted class"
)

plt.ylabel(
    "True class"
)

plt.xticks(
    range(len(present_names)),
    present_names,
    rotation=90,
    fontsize=6
)

plt.yticks(
    range(len(present_names)),
    present_names,
    fontsize=6
)

plt.colorbar()

plt.tight_layout()

plt.savefig(
    CONFUSION_PATH,
    dpi=200
)

plt.close()


print(
    "Saved:",
    CONFUSION_PATH
)


# ======================================================================
# TRAINING CURVES
# ======================================================================

print()
print("=" * 70)
print("TRAINING CURVES")
print("=" * 70)


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history["train_loss"],
    label="Train Loss"
)

plt.plot(
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "CNN Proto 2 - Loss"
)

plt.legend()

plt.tight_layout()

loss_path = (
    "/Users/vedantdwivedi/Desktop/IRIS/AI/"
    "CNN_proto2_loss.png"
)

plt.savefig(
    loss_path,
    dpi=200
)

plt.close()


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    np.asarray(history["train_accuracy"]) * 100,
    label="Train Accuracy"
)

plt.plot(
    np.asarray(history["val_accuracy"]) * 100,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.title(
    "CNN Proto 2 - Accuracy"
)

plt.legend()

plt.tight_layout()

accuracy_path = (
    "/Users/vedantdwivedi/Desktop/IRIS/AI/"
    "CNN_proto2_accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=200
)

plt.close()


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print()

print("Dataset:")
print("  Training:", X_train.shape)
print("  Validation:", X_val.shape)
print("  Test:", X_test.shape)

print()

print("Input:")
print("  Channels:", channels)
print("  Frequency bins:", frequency_bins)
print("  Time frames:", time_frames)

print()

print("Classes:", num_classes)

print()

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Final test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print()

print("Model:")
print(MODEL_PATH)

print()

print("Classification report:")
print(REPORT_PATH)

print()

print("Confusion matrix:")
print(CONFUSION_PATH)

print()

print("Loss curve:")
print(loss_path)

print()

print("Accuracy curve:")
print(accuracy_path)

print()
print("=" * 70)
print("DONE")
print("=" * 70)

