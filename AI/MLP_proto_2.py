# import numpy as np

# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score, classification_report


# # ==========================================
# # LOAD DATASET
# # ==========================================

# DATASET_PATH = (
#     "/Users/vedantdwivedi/Desktop/IRIS/"
#     "Data/dataset.npz"
# )

# data = np.load(DATASET_PATH)

# X_train = data["X_train"]
# y_train = data["y_train"]

# X_val = data["X_val"]
# y_val = data["y_val"]

# X_test = data["X_test"]
# y_test = data["y_test"]

# word_classes = [
#     "AND",
#     "ARE",
#     "HAS",
#     "HE",
#     "I",
#     "IN",
#     "IS",
#     "IT",
#     "NEW",
#     "OF",
#     "ON",
#     "STATE",
#     "THAT",
#     "THE",
#     "THEY",
#     "THIS",
#     "TO",
#     "WE",
#     "WERE",
#     "YOU"
# ]


# print("==========================================")
# print("IMPROVED MLP")
# print("==========================================")

# print("Training:", X_train.shape)
# print("Validation:", X_val.shape)
# print("Testing:", X_test.shape)


# # ==========================================
# # CHECK DATA
# # ==========================================

# print("\n==========================================")
# print("DATA CHECK")
# print("==========================================")

# print("Training finite:",
#       np.isfinite(X_train).all())

# print("Validation finite:",
#       np.isfinite(X_val).all())

# print("Testing finite:",
#       np.isfinite(X_test).all())


# # ==========================================
# # CREATE MLP
# # ==========================================

# print("\n==========================================")
# print("CREATING MLP")
# print("==========================================")

# model = MLPClassifier(

#     # Two hidden layers
#     hidden_layer_sizes=(128, 64),

#     # Activation function
#     activation="relu",

#     # Optimizer
#     solver="adam",

#     # Learning rate
#     learning_rate_init=0.001,

#     # Regularization
#     alpha=0.0001,

#     # Training
#     max_iter=100,

#     # Early stopping
#     early_stopping=True,

#     validation_fraction=0.15,

#     n_iter_no_change=10,

#     # Reproducibility
#     random_state=42,

#     verbose=True
# )


# print("MLP created successfully!")


# # ==========================================
# # TRAIN
# # ==========================================

# print("\n==========================================")
# print("TRAINING MLP")
# print("==========================================")

# model.fit(
#     X_train,
#     y_train
# )

# print("\nMLP training completed!")


# # ==========================================
# # VALIDATION
# # ==========================================

# print("\n==========================================")
# print("VALIDATION RESULTS")
# print("==========================================")

# val_predictions = model.predict(X_val)

# val_accuracy = accuracy_score(
#     y_val,
#     val_predictions
# )

# print(
#     "Validation accuracy:",
#     val_accuracy
# )


# # ==========================================
# # TEST
# # ==========================================

# print("\n==========================================")
# print("TEST RESULTS")
# print("==========================================")

# test_predictions = model.predict(X_test)

# test_accuracy = accuracy_score(
#     y_test,
#     test_predictions
# )

# print(
#     "Test accuracy:",
#     test_accuracy
# )


# # ==========================================
# # CLASSIFICATION REPORT
# # ==========================================

# print("\n==========================================")
# print("CLASSIFICATION REPORT")
# print("==========================================")

# print(
#     classification_report(
#         y_test,
#         test_predictions,
#         target_names=word_classes,
#         zero_division=0
#     )
# )


# # ==========================================
# # TRAINING INFORMATION
# # ==========================================

# print("\n==========================================")
# print("TRAINING INFORMATION")
# print("==========================================")

# print(
#     "Iterations:",
#     model.n_iter_
# )

# print(
#     "Final loss:",
#     model.loss_
# )

# print(
#     "Best validation score:",
#     model.best_validation_score_
# )

# word_classes

# import numpy as np


# # ==========================================
# # LOAD DATASET
# # ==========================================

# DATASET_PATH = (
#     "/Users/vedantdwivedi/Desktop/IRIS/"
#     "Data/dataset.npz"
# )

# data = np.load(DATASET_PATH)

# X_train = data["X_train"]
# X_val = data["X_val"]
# X_test = data["X_test"]


# print("==========================================")
# print("FEATURE DIAGNOSTICS")
# print("==========================================")

# print("X_train:", X_train.shape)
# print("X_val:", X_val.shape)
# print("X_test:", X_test.shape)


# # ==========================================
# # FEATURE STATISTICS
# # ==========================================

# means = np.mean(X_train, axis=0)

# stds = np.std(X_train, axis=0)

# mins = np.min(X_train, axis=0)

# maxs = np.max(X_train, axis=0)


# # ==========================================
# # BASIC INFORMATION
# # ==========================================

# print("\n==========================================")
# print("GLOBAL STATISTICS")
# print("==========================================")

# print("Overall minimum:", np.min(X_train))
# print("Overall maximum:", np.max(X_train))

# print("Mean of feature means:",
#       np.mean(means))

# print("Mean feature std:",
#       np.mean(stds))

# print("Smallest feature std:",
#       np.min(stds))

# print("Largest feature std:",
#       np.max(stds))


# # ==========================================
# # LOW VARIANCE FEATURES
# # ==========================================

# print("\n==========================================")
# print("LOW VARIANCE FEATURES")
# print("==========================================")

# threshold = 1e-6

# low_variance = np.where(
#     stds < threshold
# )[0]

# print(
#     "Number of near-zero variance features:",
#     len(low_variance)
# )

# if len(low_variance) > 0:

#     print("Feature indices:")

#     print(low_variance)


# # ==========================================
# # HIGHEST VARIANCE FEATURES
# # ==========================================

# print("\n==========================================")
# print("HIGHEST VARIANCE FEATURES")
# print("==========================================")

# indices = np.argsort(stds)[::-1]

# for i in indices[:20]:

#     print(
#         f"Feature {i:3d} | "
#         f"mean={means[i]: .6f} | "
#         f"std={stds[i]: .6f} | "
#         f"min={mins[i]: .6f} | "
#         f"max={maxs[i]: .6f}"
#     )


# # ==========================================
# # SMALLEST VARIANCE FEATURES
# # ==========================================

# print("\n==========================================")
# print("SMALLEST VARIANCE FEATURES")
# print("==========================================")

# indices = np.argsort(stds)

# for i in indices[:20]:

#     print(
#         f"Feature {i:3d} | "
#         f"mean={means[i]: .6f} | "
#         f"std={stds[i]: .6f} | "
#         f"min={mins[i]: .6f} | "
#         f"max={maxs[i]: .6f}"
#     )


# # ==========================================
# # FINITE CHECK
# # ==========================================

# print("\n==========================================")
# print("FINITE CHECK")
# print("==========================================")

# print(
#     "Training finite:",
#     np.isfinite(X_train).all()
# )

# print(
#     "Validation finite:",
#     np.isfinite(X_val).all()
# )

# print(
#     "Testing finite:",
#     np.isfinite(X_test).all()
# )


# # ==========================================
# # CONDITION CHECK
# # ==========================================

# print("\n==========================================")
# print("FEATURE SCALE CHECK")
# print("==========================================")

# print(
#     "Features with std < 0.01:",
#     np.sum(stds < 0.01)
# )

# print(
#     "Features with std > 2:",
#     np.sum(stds > 2)
# )

# print(
#     "Features with std > 5:",
#     np.sum(stds > 5)
# )

# print("\nDiagnostics complete!")



import numpy as np


# ==========================================
# LOAD DATASET
# ==========================================

DATASET_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/"
    "Data/dataset.npz"
)

data = np.load(DATASET_PATH)

y_train = data["y_train"]
y_val = data["y_val"]
y_test = data["y_test"]


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
# TRAINING DISTRIBUTION
# ==========================================

print("==========================================")
print("CLASS BALANCE ANALYSIS")
print("==========================================")

print("\nTRAINING SET")
print("------------------------------------------")

counts = np.bincount(
    y_train,
    minlength=len(WORD_CLASSES)
)

for i, word in enumerate(WORD_CLASSES):

    print(
        f"{word:<8}: {counts[i]:4d} samples"
    )


# ==========================================
# STATISTICS
# ==========================================

print("\n==========================================")
print("BALANCE STATISTICS")
print("==========================================")

print("Largest class:",
      WORD_CLASSES[np.argmax(counts)])

print("Largest class count:",
      np.max(counts))

print("Smallest class:",
      WORD_CLASSES[np.argmin(counts)])

print("Smallest class count:",
      np.min(counts))

print(
    "Imbalance ratio:",
    np.max(counts) / np.min(counts)
)


# ==========================================
# IDEAL BALANCED SIZE
# ==========================================

print("\n==========================================")
print("BALANCING INFORMATION")
print("==========================================")

target = np.median(counts)

print(
    "Median class size:",
    target
)

print("\nSamples needed to reach median:")

for i, word in enumerate(WORD_CLASSES):

    needed = max(
        0,
        int(target - counts[i])
    )

    print(
        f"{word:<8}: +{needed}"
    )


print("\nClass balance analysis complete!")