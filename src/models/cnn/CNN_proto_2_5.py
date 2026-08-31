"""
======================================================================
IRIS - BERKELEY 5-WORD IMPROVED 2D CNN
PROTO 2.6

Vocabulary:
    THE
    AND
    A
    OF
    I

Input:
    (N, 8, 24, 64)

Architecture:
    Residual 2D CNN
    Channel normalization
    Balanced sampling
    SpecAugment-style masking
    AdamW
    Cosine LR scheduler
    Label smoothing
    Gradient clipping
    Early stopping on Macro-F1

IMPORTANT:
    Validation and test data are NEVER augmented.

======================================================================
"""

import os
import random
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

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
    "berkeley_5word_2dcnn_proto2_6_best.pt"
)

CONFUSION_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/AI/"
    "berkeley_5word_2dcnn_proto2_6_confusion_matrix.png"
)

TRAINING_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/AI/"
    "berkeley_5word_2dcnn_proto2_6_training.png"
)

METRICS_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/AI/"
    "berkeley_5word_2dcnn_proto2_6_accuracy.png"
)


# ---------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------

VOCABULARY = [
    "THE",
    "AND",
    "A",
    "OF",
    "I"
]

NUM_CLASSES = 5

RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

BATCH_SIZE = 32

MAX_EPOCHS = 100

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

LABEL_SMOOTHING = 0.05

DROPOUT = 0.25

PATIENCE = 18

GRAD_CLIP = 1.0


# ---------------------------------------------------------------------
# SpecAugment
# ---------------------------------------------------------------------

USE_SPEC_AUGMENT = True

TIME_MASK_MAX = 8

FREQ_MASK_MAX = 4

NUM_TIME_MASKS = 1

NUM_FREQ_MASKS = 1

SPEC_AUG_PROB = 0.50


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


# =====================================================================
# HEADER
# =====================================================================

print("=" * 70)
print("IRIS - BERKELEY 5-WORD IMPROVED 2D CNN")
print("PROTO 2.6")
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
# DATASET
# =====================================================================

class EMGDataset(Dataset):

    def __init__(
        self,
        X,
        y,
        training=False,
        mean=None,
        std=None
    ):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.long
        )

        self.training = training

        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.X)

    def apply_spec_augment(self, x):

        if not self.training:
            return x

        if not USE_SPEC_AUGMENT:
            return x

        if random.random() > SPEC_AUG_PROB:
            return x

        # -------------------------------------------------------------
        # x shape:
        #
        # (8, 24, 64)
        #
        # channel × frequency × time
        # -------------------------------------------------------------

        x = x.clone()

        channels, freq, time = x.shape

        # -------------------------------------------------------------
        # Time masking
        # -------------------------------------------------------------

        for _ in range(NUM_TIME_MASKS):

            if time <= 1:
                continue

            mask_width = random.randint(
                0,
                min(TIME_MASK_MAX, time - 1)
            )

            if mask_width == 0:
                continue

            start = random.randint(
                0,
                time - mask_width
            )

            x[:, :, start:start + mask_width] = 0.0

        # -------------------------------------------------------------
        # Frequency masking
        # -------------------------------------------------------------

        for _ in range(NUM_FREQ_MASKS):

            if freq <= 1:
                continue

            mask_width = random.randint(
                0,
                min(FREQ_MASK_MAX, freq - 1)
            )

            if mask_width == 0:
                continue

            start = random.randint(
                0,
                freq - mask_width
            )

            x[:, start:start + mask_width, :] = 0.0

        return x

    def __getitem__(self, idx):

        x = self.X[idx]

        y = self.y[idx]

        # -------------------------------------------------------------
        # Training-only normalization
        # -------------------------------------------------------------

        if self.mean is not None and self.std is not None:

            x = (
                x - self.mean.squeeze(0)
            ) / self.std.squeeze(0)

        # -------------------------------------------------------------
        # Training-only augmentation
        # -------------------------------------------------------------

        x = self.apply_spec_augment(x)

        return x, y


# =====================================================================
# LOAD DATA
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


# =====================================================================
# LABEL ENCODING
# =====================================================================

def encode_labels(y, vocabulary):

    result = []

    for label in y:

        # -------------------------------------------------------------
        # Already numerical
        # -------------------------------------------------------------

        if isinstance(label, (int, np.integer)):

            value = int(label)

            if 0 <= value < len(vocabulary):

                result.append(value)

            else:

                raise ValueError(
                    f"Numerical label {value} outside vocabulary."
                )

        else:

            label_string = str(label).strip().upper()

            if label_string not in vocabulary:

                raise ValueError(
                    f"Unknown label '{label_string}'. "
                    f"Expected one of {vocabulary}"
                )

            result.append(
                vocabulary.index(label_string)
            )

    return np.asarray(
        result,
        dtype=np.int64
    )


print()
print("=" * 70)
print("ENCODING LABELS")
print("=" * 70)

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
# SHAPES
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
# NUMERICAL VALIDATION
# =====================================================================

print()
print("=" * 70)
print("NUMERICAL VALIDATION")
print("=" * 70)

print()
print("Training NaN:", np.isnan(X_train).sum())
print("Training Inf:", np.isinf(X_train).sum())

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


# =====================================================================
# ASSERT SHAPE
# =====================================================================

if X_train.ndim != 4:

    raise ValueError(
        f"Expected X_train to have 4 dimensions. "
        f"Got {X_train.shape}"
    )

channels = X_train.shape[1]

freq_bins = X_train.shape[2]

time_frames = X_train.shape[3]


print()
print("Input channels:", channels)
print("Frequency bins:", freq_bins)
print("Time frames:", time_frames)


# =====================================================================
# CLASS DISTRIBUTION
# =====================================================================

def print_distribution(name, y):

    print()
    print(name)

    for i, word in enumerate(VOCABULARY):

        count = int(
            np.sum(y == i)
        )

        print(
            f"{word:<10}: {count}"
        )


print()
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print_distribution(
    "Training:",
    y_train
)

print_distribution(
    "Validation:",
    y_val
)

print_distribution(
    "Test:",
    y_test
)


# =====================================================================
# TRAINING-ONLY NORMALIZATION
# =====================================================================

print()
print("=" * 70)
print("TRAINING-ONLY NORMALIZATION")
print("=" * 70)

# Calculate statistics over:
#
# samples × frequency × time
#
# separately for every EMG channel.

mean_np = X_train.mean(
    axis=(0, 2, 3),
    keepdims=True
)

std_np = X_train.std(
    axis=(0, 2, 3),
    keepdims=True
)

std_np = np.maximum(
    std_np,
    1e-6
)


mean = torch.tensor(
    mean_np,
    dtype=torch.float32
)

std = torch.tensor(
    std_np,
    dtype=torch.float32
)

print()
print("Mean shape:", mean.shape)
print("Std shape:", std.shape)


# =====================================================================
# DATASETS
# =====================================================================

train_dataset = EMGDataset(
    X_train,
    y_train,
    training=True,
    mean=mean,
    std=std
)

val_dataset = EMGDataset(
    X_val,
    y_val,
    training=False,
    mean=mean,
    std=std
)

test_dataset = EMGDataset(
    X_test,
    y_test,
    training=False,
    mean=mean,
    std=std
)


# =====================================================================
# BALANCED SAMPLER
# =====================================================================

print()
print("=" * 70)
print("CREATING BALANCED TRAINING SAMPLER")
print("=" * 70)


class_counts = np.bincount(
    y_train,
    minlength=NUM_CLASSES
)

print()

for i, word in enumerate(VOCABULARY):

    print(
        f"{word:<10}: {class_counts[i]}"
    )


# Inverse-frequency weights
class_weights = (
    len(y_train)
    /
    (
        NUM_CLASSES
        *
        np.maximum(class_counts, 1)
    )
)


print()
print("Sampling weights:")

for i, word in enumerate(VOCABULARY):

    print(
        f"{word:<10}: "
        f"{class_weights[i]:.4f}"
    )


sample_weights = class_weights[y_train]

sample_weights = torch.tensor(
    sample_weights,
    dtype=torch.double
)


sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(y_train),
    replacement=True
)


# =====================================================================
# DATALOADERS
# =====================================================================

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
# RESIDUAL BLOCK
# =====================================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        dropout=0.0
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        self.dropout = nn.Dropout2d(
            dropout
        )

        if (
            stride != 1
            or in_channels != out_channels
        ):

            self.shortcut = nn.Sequential(

                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
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

        out = F.relu(
            out,
            inplace=True
        )

        out = self.dropout(out)

        out = self.conv2(out)

        out = self.bn2(out)

        out = out + identity

        out = F.relu(
            out,
            inplace=True
        )

        return out


# =====================================================================
# IMPROVED 2D CNN
# =====================================================================

class ImprovedTF_CNN(nn.Module):

    def __init__(
        self,
        input_channels=8,
        num_classes=5
    ):

        super().__init__()

        # -------------------------------------------------------------
        # Initial feature extraction
        # -------------------------------------------------------------

        self.stem = nn.Sequential(

            nn.Conv2d(
                input_channels,
                32,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True)
        )

        # -------------------------------------------------------------
        # Residual block 1
        #
        # 24 x 64
        # ->
        # 12 x 32
        # -------------------------------------------------------------

        self.block1 = ResidualBlock(
            32,
            48,
            stride=2,
            dropout=0.08
        )

        # -------------------------------------------------------------
        # Residual block 2
        #
        # 12 x 32
        # ->
        # 6 x 16
        # -------------------------------------------------------------

        self.block2 = ResidualBlock(
            48,
            64,
            stride=2,
            dropout=0.10
        )

        # -------------------------------------------------------------
        # Residual block 3
        #
        # 6 x 16
        # ->
        # 3 x 8
        # -------------------------------------------------------------

        self.block3 = ResidualBlock(
            64,
            96,
            stride=2,
            dropout=0.12
        )

        # -------------------------------------------------------------
        # Extra feature block
        # -------------------------------------------------------------

        self.block4 = ResidualBlock(
            96,
            128,
            stride=1,
            dropout=0.15
        )

        # -------------------------------------------------------------
        # Global pooling
        # -------------------------------------------------------------

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        # -------------------------------------------------------------
        # Classifier
        # -------------------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128,
                64
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(inplace=True),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                64,
                num_classes
            )
        )

    def forward(self, x):

        x = self.stem(x)

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.pool(x)

        x = self.classifier(x)

        return x


# =====================================================================
# CREATE MODEL
# =====================================================================

print()
print("=" * 70)
print("CREATING IMPROVED 2D CNN")
print("=" * 70)

model = ImprovedTF_CNN(
    input_channels=channels,
    num_classes=NUM_CLASSES
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

criterion = nn.CrossEntropyLoss(
    label_smoothing=LABEL_SMOOTHING
)

print()
print(
    "CrossEntropyLoss"
)

print(
    "Label smoothing:",
    LABEL_SMOOTHING
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
# SCHEDULER
# =====================================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=MAX_EPOCHS,
    eta_min=1e-6
)


# =====================================================================
# MOVE NORMALIZATION TO DEVICE
# =====================================================================

mean_device = mean.to(DEVICE)

std_device = std.to(DEVICE)


# =====================================================================
# TRAINING FUNCTION
# =====================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion
):

    model.train()

    total_loss = 0.0

    correct = 0

    total = 0

    all_true = []

    all_pred = []


    for X, y in loader:

        X = X.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        # -------------------------------------------------------------
        # Forward
        # -------------------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(X)

        loss = criterion(
            logits,
            y
        )

        # -------------------------------------------------------------
        # Backpropagation
        # -------------------------------------------------------------

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        optimizer.step()

        # -------------------------------------------------------------
        # Metrics
        # -------------------------------------------------------------

        total_loss += (
            loss.item()
            * X.size(0)
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        correct += (
            predictions == y
        ).sum().item()

        total += X.size(0)

        all_true.extend(
            y.detach()
            .cpu()
            .numpy()
        )

        all_pred.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )


    average_loss = (
        total_loss / total
    )

    accuracy = (
        correct / total
    )

    macro_f1 = f1_score(
        all_true,
        all_pred,
        average="macro",
        zero_division=0
    )

    balanced_acc = balanced_accuracy_score(
        all_true,
        all_pred
    )

    return (
        average_loss,
        accuracy,
        balanced_acc,
        macro_f1
    )


# =====================================================================
# EVALUATION FUNCTION
# =====================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    total = 0

    all_true = []

    all_pred = []


    for X, y in loader:

        X = X.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        logits = model(X)

        loss = criterion(
            logits,
            y
        )

        total_loss += (
            loss.item()
            * X.size(0)
        )

        total += X.size(0)

        predictions = torch.argmax(
            logits,
            dim=1
        )

        all_true.extend(
            y.cpu().numpy()
        )

        all_pred.extend(
            predictions.cpu().numpy()
        )


    average_loss = (
        total_loss / total
    )

    accuracy = accuracy_score(
        all_true,
        all_pred
    )

    balanced_acc = balanced_accuracy_score(
        all_true,
        all_pred
    )

    macro_f1 = f1_score(
        all_true,
        all_pred,
        average="macro",
        zero_division=0
    )

    return (
        average_loss,
        accuracy,
        balanced_acc,
        macro_f1,
        np.asarray(all_true),
        np.asarray(all_pred)
    )


# =====================================================================
# TRAINING
# =====================================================================

print()
print("=" * 70)
print("TRAINING")
print("=" * 70)

print()

history = {

    "train_loss": [],
    "val_loss": [],

    "train_acc": [],
    "val_acc": [],

    "train_balanced": [],
    "val_balanced": [],

    "train_f1": [],
    "val_f1": []
}


best_model_state = None

best_val_f1 = -np.inf

best_val_loss = np.inf

best_epoch = 0

epochs_without_improvement = 0


for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    # ---------------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------------

    (
        train_loss,
        train_acc,
        train_balanced,
        train_f1
    ) = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion
    )


    # ---------------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------------

    (
        val_loss,
        val_acc,
        val_balanced,
        val_f1,
        _,
        _
    ) = evaluate(
        model,
        val_loader,
        criterion
    )


    # ---------------------------------------------------------------
    # Scheduler
    # ---------------------------------------------------------------

    scheduler.step()


    current_lr = optimizer.param_groups[0]["lr"]


    # ---------------------------------------------------------------
    # Save history
    # ---------------------------------------------------------------

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


    # ---------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------

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


    # ---------------------------------------------------------------
    # Model selection
    #
    # PRIMARY:
    #     validation Macro-F1
    #
    # SECONDARY:
    #     validation loss
    # ---------------------------------------------------------------

    improved = False


    if val_f1 > best_val_f1 + 1e-4:

        improved = True

    elif (
        abs(val_f1 - best_val_f1) <= 1e-4
        and val_loss < best_val_loss
    ):

        improved = True


    if improved:

        best_val_f1 = val_f1

        best_val_loss = val_loss

        best_epoch = epoch

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict":
                    best_model_state,

                "vocabulary":
                    VOCABULARY,

                "input_channels":
                    channels,

                "freq_bins":
                    freq_bins,

                "time_frames":
                    time_frames,

                "best_epoch":
                    best_epoch,

                "best_val_f1":
                    best_val_f1,

                "best_val_loss":
                    best_val_loss,

                "mean":
                    mean_np,

                "std":
                    std_np
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


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


print()
print(
    "Best epoch:",
    checkpoint["best_epoch"]
)

print(
    "Best validation Macro-F1:",
    checkpoint["best_val_f1"]
)

print(
    "Best validation loss:",
    checkpoint["best_val_loss"]
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
    val_acc,
    val_balanced,
    val_f1,
    y_val_true,
    y_val_pred
) = evaluate(
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
    f"{val_acc:.4f}"
)

print(
    f"Validation accuracy (%): "
    f"{val_acc * 100:.2f}%"
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
    f"{val_f1:.4f}"
)

print(
    f"Validation Macro F1 (%): "
    f"{val_f1 * 100:.2f}%"
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
    test_acc,
    test_balanced,
    test_f1,
    y_test_true,
    y_test_pred
) = evaluate(
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
    f"{test_acc:.4f}"
)

print(
    f"Test accuracy (%): "
    f"{test_acc * 100:.2f}%"
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
    f"{test_f1:.4f}"
)

print(
    f"Test Macro F1 (%): "
    f"{test_f1 * 100:.2f}%"
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
        y_test_true == i
    )

    print(
        f"{word:<10}: {count}"
    )


print()
print("Predicted class distribution:")

for i, word in enumerate(VOCABULARY):

    count = np.sum(
        y_test_pred == i
    )

    print(
        f"{word:<10}: {count}"
    )


missing_predictions = []

for i, word in enumerate(VOCABULARY):

    if np.sum(y_test_pred == i) == 0:

        missing_predictions.append(
            (i, word)
        )


print()

if len(missing_predictions) == 0:

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

    for i, word in missing_predictions:

        print(
            f"  {i}: {word}"
        )


# =====================================================================
# CLASSIFICATION REPORT
# =====================================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print()

report = classification_report(
    y_test_true,
    y_test_pred,
    labels=np.arange(NUM_CLASSES),
    target_names=VOCABULARY,
    digits=2,
    zero_division=0
)

print(report)


# =====================================================================
# CONFUSION MATRIX
# =====================================================================

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)


cm = confusion_matrix(
    y_test_true,
    y_test_pred,
    labels=np.arange(NUM_CLASSES)
)


print()

print(cm)


plt.figure(
    figsize=(8, 7)
)

plt.imshow(cm)

plt.title(
    "IRIS - Berkeley 5-Word Improved 2D CNN"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "True"
)

plt.xticks(
    np.arange(NUM_CLASSES),
    VOCABULARY,
    rotation=45
)

plt.yticks(
    np.arange(NUM_CLASSES),
    VOCABULARY
)

for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

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


# =====================================================================
# TRAINING CURVES
# =====================================================================

print()
print("=" * 70)
print("SAVING TRAINING CURVES")
print("=" * 70)


epochs = np.arange(
    1,
    len(history["train_loss"]) + 1
)


# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    epochs,
    history["train_loss"],
    label="Train Loss"
)

plt.plot(
    epochs,
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
    "Training and Validation Loss"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    TRAINING_PATH,
    dpi=200
)

plt.close()


# ---------------------------------------------------------------------
# Accuracy / F1
# ---------------------------------------------------------------------

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    epochs,
    history["train_acc"],
    label="Train Accuracy"
)

plt.plot(
    epochs,
    history["val_acc"],
    label="Validation Accuracy"
)

plt.plot(
    epochs,
    history["train_f1"],
    label="Train Macro-F1"
)

plt.plot(
    epochs,
    history["val_f1"],
    label="Validation Macro-F1"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Score"
)

plt.title(
    "Accuracy and Macro-F1"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    METRICS_PATH,
    dpi=200
)

plt.close()


print()
print(
    "Training plot saved:"
)

print(
    TRAINING_PATH
)

print()
print(
    "Accuracy/F1 plot saved:"
)

print(
    METRICS_PATH
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

print("Input:")

print(
    f"Channels: {channels}"
)

print(
    f"Frequency bins: {freq_bins}"
)

print(
    f"Time frames: {time_frames}"
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
    f"Best validation Macro-F1: "
    f"{best_val_f1 * 100:.2f}%"
)

print(
    f"Final validation accuracy: "
    f"{val_acc * 100:.2f}%"
)

print(
    f"Final validation balanced accuracy: "
    f"{val_balanced * 100:.2f}%"
)

print(
    f"Final validation Macro-F1: "
    f"{val_f1 * 100:.2f}%"
)


print()

print(
    f"Final test accuracy: "
    f"{test_acc * 100:.2f}%"
)

print(
    f"Final test balanced accuracy: "
    f"{test_balanced * 100:.2f}%"
)

print(
    f"Final test Macro-F1: "
    f"{test_f1 * 100:.2f}%"
)


print()

print("Model:")

print(
    MODEL_PATH
)


print()
print("=" * 70)
print("DONE")
print("=" * 70)
