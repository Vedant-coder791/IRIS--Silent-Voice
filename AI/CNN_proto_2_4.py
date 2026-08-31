
"""
======================================================================
IRIS - BERKELEY 5-WORD 2D CNN
PROTO 2.5

Improved 2D CNN experiment for silent-speech sEMG
using word-level time-frequency representations.

Dataset:
    berkeley_5word_tf_augmented_dataset.npz

Input:
    (N, 8, 24, 64)

    8  = EMG channels
    24 = frequency bins
    64 = time frames

Vocabulary:
    THE
    AND
    A
    OF
    I

Improvements:
    - 2D CNN over frequency x time
    - Residual convolution blocks
    - Batch normalization
    - Dropout
    - Training-only normalization
    - Class-balanced loss
    - WeightedRandomSampler
    - Dynamic time masking
    - Dynamic frequency masking
    - Gaussian noise augmentation
    - Label smoothing
    - AdamW
    - Cosine learning-rate scheduling
    - Gradient clipping
    - Early stopping using validation Macro-F1
    - Accuracy
    - Balanced accuracy
    - Macro F1
    - Classification report
    - Confusion matrix
    - Training curves
    - Prediction sanity check
    - PyTorch 2.6-safe checkpoint loading
======================================================================
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ======================================================================
# CONFIGURATION
# ======================================================================

DATASET_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/Data/datasets/"
    "berkeley_5word_tf_augmented_dataset.npz"
)

MODEL_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/AI/"
    "berkeley_5word_2dcnn_proto2_5_best.pt"
)

CONFUSION_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/AI/"
    "berkeley_5word_2dcnn_proto2_5_confusion_matrix.png"
)

TRAINING_PLOT_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/AI/"
    "berkeley_5word_2dcnn_proto2_5_training.png"
)

ACCURACY_PLOT_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/AI/"
    "berkeley_5word_2dcnn_proto2_5_accuracy.png"
)

RANDOM_STATE = 42

BATCH_SIZE = 32
EPOCHS = 100

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

LABEL_SMOOTHING = 0.05

PATIENCE = 18
MIN_DELTA = 1e-4

GRAD_CLIP = 1.0

# Augmentation probabilities
TIME_MASK_PROB = 0.50
FREQ_MASK_PROB = 0.50
NOISE_PROB = 0.30

# Maximum mask sizes
MAX_TIME_MASK = 10
MAX_FREQ_MASK = 4

NOISE_STD = 0.015

NUM_WORKERS = 0


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_STATE)


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
print("IRIS - BERKELEY 5-WORD 2D CNN PROTO 2.5")
print("=" * 70)

print()
print("Dataset:")
print(DATASET_PATH)

print()
print("Device:")
print(DEVICE)


# ======================================================================
# LOAD DATASET
# ======================================================================

print()
print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

data = np.load(DATASET_PATH, allow_pickle=True)

print()
print("Dataset keys:")
print(list(data.keys()))

X_train = data["X_train"].astype(np.float32)
y_train_raw = data["y_train"]

X_val = data["X_val"].astype(np.float32)
y_val_raw = data["y_val"]

X_test = data["X_test"].astype(np.float32)
y_test_raw = data["y_test"]

if "vocabulary" in data:
    vocabulary_raw = data["vocabulary"]
else:
    vocabulary_raw = np.array(["THE", "AND", "A", "OF", "I"])


# ======================================================================
# NORMALIZE VOCABULARY
# ======================================================================

vocabulary = []

for item in vocabulary_raw:
    if isinstance(item, bytes):
        item = item.decode("utf-8")

    vocabulary.append(str(item).strip().upper())

vocabulary = np.array(vocabulary)

print()
print("Vocabulary:")

for i, word in enumerate(vocabulary):
    print(f"  {i}: {word}")


# ======================================================================
# LABEL ENCODING
# ======================================================================

print()
print("=" * 70)
print("ENCODING LABELS")
print("=" * 70)


word_to_index = {
    word: i for i, word in enumerate(vocabulary)
}


def encode_labels(labels):

    encoded = []

    for label in labels:

        if isinstance(label, bytes):
            label = label.decode("utf-8")

        label_string = str(label).strip().upper()

        # Case 1:
        # Already integer class index
        try:
            numeric_value = int(label_string)

            if 0 <= numeric_value < len(vocabulary):
                encoded.append(numeric_value)
                continue

        except ValueError:
            pass

        # Case 2:
        # Word label
        if label_string in word_to_index:
            encoded.append(word_to_index[label_string])
        else:
            raise ValueError(
                f"Unknown label '{label_string}'. "
                f"Vocabulary = {list(vocabulary)}"
            )

    return np.array(encoded, dtype=np.int64)


y_train = encode_labels(y_train_raw)
y_val = encode_labels(y_val_raw)
y_test = encode_labels(y_test_raw)


# ======================================================================
# DATASET SHAPES
# ======================================================================

print()
print("=" * 70)
print("DATASET SHAPES")
print("=" * 70)

print()
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ======================================================================
# EXPECTED INPUT CHECK
# ======================================================================

if X_train.ndim != 4:
    raise RuntimeError(
        f"Expected 4D input, got {X_train.ndim}D"
    )

if X_train.shape[1] != 8:
    raise RuntimeError(
        f"Expected 8 channels, got {X_train.shape[1]}"
    )

if X_train.shape[2] != 24:
    raise RuntimeError(
        f"Expected 24 frequency bins, got {X_train.shape[2]}"
    )

if X_train.shape[3] != 64:
    raise RuntimeError(
        f"Expected 64 time frames, got {X_train.shape[3]}"
    )


# ======================================================================
# NUMERICAL VALIDATION
# ======================================================================

print()
print("=" * 70)
print("NUMERICAL VALIDATION")
print("=" * 70)

print()
print("Training NaN:", np.isnan(X_train).sum())
print("Training Inf:", np.isinf(X_train).sum())

print("Validation NaN:", np.isnan(X_val).sum())
print("Validation Inf:", np.isinf(X_val).sum())

print("Test NaN:", np.isnan(X_test).sum())
print("Test Inf:", np.isinf(X_test).sum())

if (
    np.isnan(X_train).any()
    or np.isinf(X_train).any()
    or np.isnan(X_val).any()
    or np.isinf(X_val).any()
    or np.isnan(X_test).any()
    or np.isinf(X_test).any()
):
    raise RuntimeError("NaN or infinite values detected.")


# ======================================================================
# CLASS DISTRIBUTION
# ======================================================================

print()
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)


def print_distribution(name, labels):

    counts = Counter(labels)

    print()
    print(name)

    for i, word in enumerate(vocabulary):

        print(
            f"{word:<10}: "
            f"{counts.get(i, 0)}"
        )


print_distribution("TRAIN", y_train)
print_distribution("VALIDATION", y_val)
print_distribution("TEST", y_test)


# ======================================================================
# TRAINING-ONLY NORMALIZATION
# ======================================================================

print()
print("=" * 70)
print("TRAINING-ONLY NORMALIZATION")
print("=" * 70)

# Mean/std calculated ONLY from training set.
#
# Shape:
# (1, channels, 1, 1)
#
# This prevents validation/test leakage.

mean = X_train.mean(axis=(0, 2, 3), keepdims=True)

std = X_train.std(axis=(0, 2, 3), keepdims=True)

std = np.maximum(std, 1e-6)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

print()
print("Mean shape:", mean.shape)
print("Std shape:", std.shape)

print()
print("Normalized training mean:",
      float(X_train.mean()))

print("Normalized training std:",
      float(X_train.std()))


# ======================================================================
# DATA AUGMENTATION
# ======================================================================

class TimeFrequencyAugment:
    """
    Dynamic augmentation for TF representations.

    Applied ONLY during training.

    Operations:
        1. Time masking
        2. Frequency masking
        3. Small Gaussian noise
    """

    def __init__(
        self,
        time_mask_prob=0.5,
        freq_mask_prob=0.5,
        noise_prob=0.3,
        max_time_mask=10,
        max_freq_mask=4,
        noise_std=0.015,
    ):

        self.time_mask_prob = time_mask_prob
        self.freq_mask_prob = freq_mask_prob
        self.noise_prob = noise_prob

        self.max_time_mask = max_time_mask
        self.max_freq_mask = max_freq_mask

        self.noise_std = noise_std

    def __call__(self, x):

        # x:
        # (C, F, T)

        x = x.copy()

        channels, freq_bins, time_frames = x.shape

        # --------------------------------------------------------------
        # TIME MASKING
        # --------------------------------------------------------------

        if random.random() < self.time_mask_prob:

            mask_width = random.randint(
                1,
                min(self.max_time_mask, time_frames)
            )

            start = random.randint(
                0,
                time_frames - mask_width
            )

            # Use zero because the input is normalized.
            x[:, :, start:start + mask_width] = 0.0

        # --------------------------------------------------------------
        # FREQUENCY MASKING
        # --------------------------------------------------------------

        if random.random() < self.freq_mask_prob:

            mask_width = random.randint(
                1,
                min(self.max_freq_mask, freq_bins)
            )

            start = random.randint(
                0,
                freq_bins - mask_width
            )

            x[:, start:start + mask_width, :] = 0.0

        # --------------------------------------------------------------
        # SMALL GAUSSIAN NOISE
        # --------------------------------------------------------------

        if random.random() < self.noise_prob:

            noise = np.random.normal(
                0.0,
                self.noise_std,
                size=x.shape
            ).astype(np.float32)

            x += noise

        return x


train_augment = TimeFrequencyAugment(
    time_mask_prob=TIME_MASK_PROB,
    freq_mask_prob=FREQ_MASK_PROB,
    noise_prob=NOISE_PROB,
    max_time_mask=MAX_TIME_MASK,
    max_freq_mask=MAX_FREQ_MASK,
    noise_std=NOISE_STD,
)


# ======================================================================
# PYTORCH DATASET
# ======================================================================

class TFEMGDataset(Dataset):

    def __init__(
        self,
        X,
        y,
        augment=None
    ):

        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = self.X[idx]

        y = self.y[idx]

        if self.augment is not None:

            x = self.augment(x)

        x = torch.from_numpy(
            x.astype(np.float32)
        )

        y = torch.tensor(
            y,
            dtype=torch.long
        )

        return x, y


train_dataset = TFEMGDataset(
    X_train,
    y_train,
    augment=train_augment
)

val_dataset = TFEMGDataset(
    X_val,
    y_val,
    augment=None
)

test_dataset = TFEMGDataset(
    X_test,
    y_test,
    augment=None
)


# ======================================================================
# CLASS WEIGHTS
# ======================================================================

print()
print("=" * 70)
print("CLASS BALANCING")
print("=" * 70)

train_counts = Counter(y_train)

num_classes = len(vocabulary)

class_weights = np.zeros(num_classes, dtype=np.float32)

for i in range(num_classes):

    class_weights[i] = (
        len(y_train)
        /
        (num_classes * train_counts[i])
    )

print()

for i, word in enumerate(vocabulary):

    print(
        f"{word:<10} "
        f"samples={train_counts[i]:<5} "
        f"weight={class_weights[i]:.4f}"
    )


# ======================================================================
# WEIGHTED RANDOM SAMPLER
# ======================================================================

sample_weights = np.array(
    [
        class_weights[label]
        for label in y_train
    ],
    dtype=np.float64
)

sampler = WeightedRandomSampler(
    weights=torch.from_numpy(sample_weights),
    num_samples=len(sample_weights),
    replacement=True
)


# ======================================================================
# DATA LOADERS
# ======================================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)


# ======================================================================
# RESIDUAL 2D BLOCK
# ======================================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=0.15,
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        self.dropout = nn.Dropout2d(
            dropout
        )

        if in_channels != out_channels:

            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    out_channels
                )
            )

        else:

            self.shortcut = nn.Identity()

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.gelu(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity

        out = F.gelu(out)

        return out


# ======================================================================
# 2D CNN MODEL
# ======================================================================

class TF2DCNN(nn.Module):

    def __init__(
        self,
        num_classes=5,
    ):

        super().__init__()

        # --------------------------------------------------------------
        # STEM
        # --------------------------------------------------------------

        self.stem = nn.Sequential(

            nn.Conv2d(
                8,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(32),

            nn.GELU(),
        )

        # --------------------------------------------------------------
        # BLOCK 1
        # --------------------------------------------------------------

        self.block1 = ResidualBlock(
            32,
            32,
            dropout=0.10,
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )

        # --------------------------------------------------------------
        # BLOCK 2
        # --------------------------------------------------------------

        self.block2 = ResidualBlock(
            32,
            64,
            dropout=0.15,
        )

        self.pool2 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )

        # --------------------------------------------------------------
        # BLOCK 3
        # --------------------------------------------------------------

        self.block3 = ResidualBlock(
            64,
            128,
            dropout=0.20,
        )

        # --------------------------------------------------------------
        # BLOCK 4
        # --------------------------------------------------------------

        self.block4 = ResidualBlock(
            128,
            128,
            dropout=0.20,
        )

        # --------------------------------------------------------------
        # GLOBAL POOLING
        # --------------------------------------------------------------

        self.avg_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.max_pool = nn.AdaptiveMaxPool2d(
            (1, 1)
        )

        # Concatenated:
        #
        # 128 average features
        # 128 maximum features
        #
        # total = 256

        self.classifier = nn.Sequential(

            nn.Linear(
                256,
                128
            ),

            nn.GELU(),

            nn.Dropout(0.35),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        # x:
        # (B, 8, 24, 64)

        x = self.stem(x)

        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)

        x = self.block4(x)

        avg = self.avg_pool(x)
        maxv = self.max_pool(x)

        avg = avg.flatten(1)
        maxv = maxv.flatten(1)

        x = torch.cat(
            [avg, maxv],
            dim=1
        )

        x = self.classifier(x)

        return x


# ======================================================================
# CREATE MODEL
# ======================================================================

print()
print("=" * 70)
print("CREATING 2D CNN MODEL")
print("=" * 70)

model = TF2DCNN(
    num_classes=num_classes
)

model = model.to(DEVICE)

print()
print(model)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print()
print(
    f"Trainable parameters: "
    f"{trainable_parameters:,}"
)


# ======================================================================
# LOSS
# ======================================================================

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE
    ),
    label_smoothing=LABEL_SMOOTHING,
)


# ======================================================================
# OPTIMIZER
# ======================================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ======================================================================
# LR SCHEDULER
# ======================================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6,
)


# ======================================================================
# METRICS
# ======================================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    balanced = balanced_accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return accuracy, balanced, macro_f1


# ======================================================================
# TRAIN ONE EPOCH
# ======================================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0

    all_true = []
    all_pred = []

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        logits = model(X_batch)

        loss = criterion(
            logits,
            y_batch
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
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

        all_true.extend(
            y_batch.detach().cpu().numpy()
        )

        all_pred.extend(
            predictions.detach().cpu().numpy()
        )

    average_loss = (
        total_loss
        /
        len(train_dataset)
    )

    accuracy, balanced, macro_f1 = calculate_metrics(
        all_true,
        all_pred
    )

    return (
        average_loss,
        accuracy,
        balanced,
        macro_f1
    )


# ======================================================================
# EVALUATION
# ======================================================================

@torch.no_grad()
def evaluate(loader):

    model.eval()

    total_loss = 0.0

    all_true = []
    all_pred = []

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        logits = model(X_batch)

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

        all_true.extend(
            y_batch.cpu().numpy()
        )

        all_pred.extend(
            predictions.cpu().numpy()
        )

    average_loss = (
        total_loss
        /
        len(loader.dataset)
    )

    accuracy, balanced, macro_f1 = calculate_metrics(
        all_true,
        all_pred
    )

    return (
        average_loss,
        accuracy,
        balanced,
        macro_f1,
        np.array(all_true),
        np.array(all_pred),
    )


# ======================================================================
# TRAINING
# ======================================================================

print()
print("=" * 70)
print("TRAINING")
print("=" * 70)

history = {

    "train_loss": [],
    "val_loss": [],

    "train_acc": [],
    "val_acc": [],

    "train_balanced": [],
    "val_balanced": [],

    "train_f1": [],
    "val_f1": [],
}


best_val_f1 = -np.inf
best_val_loss = np.inf

best_epoch = 0

epochs_without_improvement = 0


for epoch in range(1, EPOCHS + 1):

    (
        train_loss,
        train_acc,
        train_balanced,
        train_f1,
    ) = train_one_epoch()

    (
        val_loss,
        val_acc,
        val_balanced,
        val_f1,
        _,
        _,
    ) = evaluate(val_loader)

    scheduler.step()

    current_lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(
        train_loss
    )

    history["val_loss"].append(
        val_loss
    )

    history["train_acc"].append(
        train_acc
    )

    history["val_acc"].append(
        val_acc
    )

    history["train_balanced"].append(
        train_balanced
    )

    history["val_balanced"].append(
        val_balanced
    )

    history["train_f1"].append(
        train_f1
    )

    history["val_f1"].append(
        val_f1
    )

    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss {train_loss:.4f} | "
        f"Train Acc {train_acc:.4f} | "
        f"Train Bal {train_balanced:.4f} | "
        f"Train F1 {train_f1:.4f} | "
        f"Val Loss {val_loss:.4f} | "
        f"Val Acc {val_acc:.4f} | "
        f"Val Bal {val_balanced:.4f} | "
        f"Val F1 {val_f1:.4f} | "
        f"LR {current_lr:.2e}"
    )

    # --------------------------------------------------------------
    # MODEL SELECTION
    #
    # Primary metric:
    # validation Macro-F1
    #
    # Tie-breaker:
    # validation loss
    # --------------------------------------------------------------

    improved = False

    if val_f1 > best_val_f1 + MIN_DELTA:

        improved = True

    elif (
        abs(val_f1 - best_val_f1) <= MIN_DELTA
        and val_loss < best_val_loss
    ):

        improved = True

    if improved:

        best_val_f1 = val_f1
        best_val_loss = val_loss
        best_epoch = epoch

        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "epoch": epoch,

                "best_val_f1":
                    best_val_f1,

                "best_val_loss":
                    best_val_loss,

                "vocabulary":
                    vocabulary.tolist(),

                "mean":
                    mean,

                "std":
                    std,

                "config": {
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "label_smoothing":
                        LABEL_SMOOTHING,
                },
            },
            MODEL_PATH
        )

        print(
            "  -> Best model saved."
        )

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:

        print()
        print(
            "Early stopping triggered."
        )

        break


# ======================================================================
# LOAD BEST MODEL
# ======================================================================

print()
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

print()
print("Best epoch:", best_epoch)

print(
    "Best validation Macro-F1:",
    best_val_f1
)

print(
    "Best validation loss:",
    best_val_loss
)


# PyTorch 2.6 compatibility.
#
# This checkpoint was created by this script and is trusted.
#
# weights_only=False is necessary because the checkpoint contains
# numpy arrays such as mean/std.

try:

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

except TypeError:

    # Compatibility with older PyTorch versions.

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)


# ======================================================================
# FINAL VALIDATION
# ======================================================================

print()
print("=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

(
    val_loss,
    val_accuracy,
    val_balanced,
    val_macro_f1,
    val_true,
    val_pred,
) = evaluate(val_loader)

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
    f"Validation accuracy (%): "
    f"{val_accuracy * 100:.2f}%"
)

print(
    f"Validation balanced accuracy: "
    f"{val_balanced:.4f}"
)

print(
    f"Validation balanced accuracy (%): "
    f"{val_balanced * 100:.2f}%"
)

print(
    f"Validation Macro F1: "
    f"{val_macro_f1:.4f}"
)

print(
    f"Validation Macro F1 (%): "
    f"{val_macro_f1 * 100:.2f}%"
)


# ======================================================================
# FINAL TEST
# ======================================================================

print()
print("=" * 70)
print("FINAL TEST")
print("=" * 70)

(
    test_loss,
    test_accuracy,
    test_balanced,
    test_macro_f1,
    test_true,
    test_pred,
) = evaluate(test_loader)

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

print(
    f"Test balanced accuracy: "
    f"{test_balanced:.4f}"
)

print(
    f"Test balanced accuracy (%): "
    f"{test_balanced * 100:.2f}%"
)

print(
    f"Test Macro F1: "
    f"{test_macro_f1:.4f}"
)

print(
    f"Test Macro F1 (%): "
    f"{test_macro_f1 * 100:.2f}%"
)


# ======================================================================
# PREDICTION SANITY CHECK
# ======================================================================

print()
print("=" * 70)
print("PREDICTION SANITY CHECK")
print("=" * 70)

true_counts = Counter(test_true)
pred_counts = Counter(test_pred)

print()
print("True class distribution:")

for i, word in enumerate(vocabulary):

    print(
        f"{word:<10}: "
        f"{true_counts.get(i, 0)}"
    )

print()
print("Predicted class distribution:")

for i, word in enumerate(vocabulary):

    print(
        f"{word:<10}: "
        f"{pred_counts.get(i, 0)}"
    )

predicted_classes = set(test_pred.tolist())

missing_classes = [
    i
    for i in range(num_classes)
    if i not in predicted_classes
]

print()

if len(missing_classes) == 0:

    print(
        "Prediction sanity check:"
    )

    print(
        "All 5 classes were predicted."
    )

else:

    print(
        "WARNING:"
    )

    print(
        "The following classes were never predicted:"
    )

    for i in missing_classes:

        print(
            f"  {i}: {vocabulary[i]}"
        )


# ======================================================================
# CLASSIFICATION REPORT
# ======================================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print()

print(
    classification_report(
        test_true,
        test_pred,
        labels=list(range(num_classes)),
        target_names=vocabulary,
        zero_division=0,
    )
)


# ======================================================================
# CONFUSION MATRIX
# ======================================================================

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    test_true,
    test_pred,
    labels=list(range(num_classes))
)

print()
print(cm)


plt.figure(
    figsize=(8, 7)
)

plt.imshow(cm)

plt.title(
    "IRIS Berkeley 5-Word 2D CNN\n"
    "Proto 2.5 Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "True"
)

plt.xticks(
    range(num_classes),
    vocabulary,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(num_classes),
    vocabulary
)

for i in range(num_classes):

    for j in range(num_classes):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.colorbar()

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


# ======================================================================
# TRAINING CURVES
# ======================================================================

print()
print("=" * 70)
print("SAVING TRAINING CURVES")
print("=" * 70)

epochs_completed = range(
    1,
    len(history["train_loss"]) + 1
)


# ----------------------------------------------------------------------
# LOSS
# ----------------------------------------------------------------------

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    epochs_completed,
    history["train_loss"],
    label="Train Loss"
)

plt.plot(
    epochs_completed,
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "IRIS Berkeley 5-Word 2D CNN\n"
    "Proto 2.5 Loss"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    TRAINING_PLOT_PATH,
    dpi=200
)

plt.close()


# ----------------------------------------------------------------------
# ACCURACY / F1
# ----------------------------------------------------------------------

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    epochs_completed,
    history["train_acc"],
    label="Train Accuracy"
)

plt.plot(
    epochs_completed,
    history["val_acc"],
    label="Validation Accuracy"
)

plt.plot(
    epochs_completed,
    history["train_f1"],
    label="Train Macro-F1"
)

plt.plot(
    epochs_completed,
    history["val_f1"],
    label="Validation Macro-F1"
)

plt.xlabel("Epoch")
plt.ylabel("Score")

plt.title(
    "IRIS Berkeley 5-Word 2D CNN\n"
    "Proto 2.5 Performance"
)

plt.legend()

plt.grid(
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
    "Training plot saved:"
)

print(
    TRAINING_PLOT_PATH
)

print()
print(
    "Accuracy/F1 plot saved:"
)

print(
    ACCURACY_PLOT_PATH
)


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print()

print("Vocabulary:")

for i, word in enumerate(vocabulary):

    print(
        f"  {i}: {word}"
    )

print()

print("Input:")
print("  Channels:", X_train.shape[1])
print("  Frequency bins:", X_train.shape[2])
print("  Time frames:", X_train.shape[3])

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
    f"Best validation Macro F1: "
    f"{best_val_f1 * 100:.2f}%"
)

print(
    f"Final validation accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

print(
    f"Final validation balanced accuracy: "
    f"{val_balanced * 100:.2f}%"
)

print(
    f"Final validation Macro F1: "
    f"{val_macro_f1 * 100:.2f}%"
)

print()

print(
    f"Final test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Final test balanced accuracy: "
    f"{test_balanced * 100:.2f}%"
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

