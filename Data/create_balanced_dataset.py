import numpy as np


# ==========================================
# PATHS
# ==========================================

INPUT_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/"
    "Data/dataset.npz"
)

OUTPUT_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/"
    "Data/dataset_balanced.npz"
)


# ==========================================
# SETTINGS
# ==========================================

TARGET_COUNT = 150

RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)


# ==========================================
# WORD CLASSES
# ==========================================

WORD_CLASSES = [
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


# ==========================================
# LOAD ORIGINAL DATASET
# ==========================================

print("==========================================")
print("LOADING ORIGINAL DATASET")
print("==========================================")

data = np.load(INPUT_PATH)

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_val"]
y_val = data["y_val"]

X_test = data["X_test"]
y_test = data["y_test"]


print("Original training set:", X_train.shape)
print("Validation set:", X_val.shape)
print("Testing set:", X_test.shape)


# ==========================================
# ORIGINAL CLASS COUNTS
# ==========================================

print("\n==========================================")
print("ORIGINAL TRAINING DISTRIBUTION")
print("==========================================")

original_counts = np.bincount(
    y_train,
    minlength=len(WORD_CLASSES)
)

for i, word in enumerate(WORD_CLASSES):

    print(
        f"{word:<8}: {original_counts[i]:4d}"
    )


# ==========================================
# OVERSAMPLE MINORITY CLASSES
# ==========================================

X_balanced_parts = []
y_balanced_parts = []


print("\n==========================================")
print("BALANCING TRAINING DATA")
print("==========================================")


for class_id, word in enumerate(WORD_CLASSES):

    # Find all samples belonging to this class
    indices = np.where(
        y_train == class_id
    )[0]

    current_count = len(indices)

    # --------------------------------------
    # Class already has enough samples
    # --------------------------------------

    if current_count >= TARGET_COUNT:

        selected_indices = indices

    # --------------------------------------
    # Class needs oversampling
    # --------------------------------------

    else:

        additional_count = (
            TARGET_COUNT - current_count
        )

        extra_indices = rng.choice(
            indices,
            size=additional_count,
            replace=True
        )

        selected_indices = np.concatenate(
            [
                indices,
                extra_indices
            ]
        )

    X_balanced_parts.append(
        X_train[selected_indices]
    )

    y_balanced_parts.append(
        y_train[selected_indices]
    )

    print(
        f"{word:<8}: "
        f"{current_count:4d} -> "
        f"{len(selected_indices):4d}"
    )


# ==========================================
# COMBINE CLASSES
# ==========================================

X_train_balanced = np.concatenate(
    X_balanced_parts,
    axis=0
)

y_train_balanced = np.concatenate(
    y_balanced_parts,
    axis=0
)


# ==========================================
# SHUFFLE
# ==========================================

shuffle_indices = rng.permutation(
    len(X_train_balanced)
)

X_train_balanced = (
    X_train_balanced[shuffle_indices]
)

y_train_balanced = (
    y_train_balanced[shuffle_indices]
)


# ==========================================
# FINAL DISTRIBUTION
# ==========================================

print("\n==========================================")
print("BALANCED TRAINING DISTRIBUTION")
print("==========================================")

balanced_counts = np.bincount(
    y_train_balanced,
    minlength=len(WORD_CLASSES)
)

for i, word in enumerate(WORD_CLASSES):

    print(
        f"{word:<8}: "
        f"{balanced_counts[i]:4d}"
    )


# ==========================================
# SAVE DATASET
# ==========================================

print("\n==========================================")
print("SAVING BALANCED DATASET")
print("==========================================")

np.savez(
    OUTPUT_PATH,

    X_train=X_train_balanced,
    y_train=y_train_balanced,

    X_val=X_val,
    y_val=y_val,

    X_test=X_test,
    y_test=y_test
)


# ==========================================
# FINAL INFORMATION
# ==========================================

print("\n==========================================")
print("BALANCED DATASET CREATED")
print("==========================================")

print(
    "Original training samples:",
    len(X_train)
)

print(
    "Balanced training samples:",
    len(X_train_balanced)
)

print(
    "Validation samples:",
    len(X_val)
)

print(
    "Testing samples:",
    len(X_test)
)

print(
    "Features per sample:",
    X_train_balanced.shape[1]
)

print("\nSaved to:")
print(OUTPUT_PATH)