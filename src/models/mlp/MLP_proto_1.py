import os
import numpy as np


# ==========================================
# DATASET PATH
# ==========================================

DATASET_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/"
    "Data/dataset.npz"
)


# ==========================================
# LOAD DATASET
# ==========================================

print("\n==========================================")
print("LOADING DATASET")
print("==========================================")

data = np.load(DATASET_PATH)


X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_val"]
y_val = data["y_val"]

X_test = data["X_test"]
y_test = data["y_test"]

classes = data["classes"]


# ==========================================
# DISPLAY DATASET INFORMATION
# ==========================================

print("\n==========================================")
print("DATASET LOADED")
print("==========================================")

print("Training:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nValidation:")
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("\nTesting:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ==========================================
# DISPLAY WORD CLASSES
# ==========================================

print("\n==========================================")
print("WORD CLASSES")
print("==========================================")

for i, word in enumerate(classes):

    print(
        i,
        "->",
        word
    )


# ==========================================
# BASIC VALIDATION
# ==========================================

print("\n==========================================")
print("DATASET CHECK")
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
    X_train.shape[0]
)

print(
    "Validation samples:",
    X_val.shape[0]
)

print(
    "Testing samples:",
    X_test.shape[0]
)


# ==========================================
# CHECK FOR NaN / INFINITY
# ==========================================

print("\nChecking for invalid values...")

print(
    "NaN in training:",
    np.isnan(X_train).any()
)

print(
    "Infinity in training:",
    np.isinf(X_train).any()
)

print(
    "NaN in validation:",
    np.isnan(X_val).any()
)

print(
    "NaN in testing:",
    np.isnan(X_test).any()
)


# ==========================================
# SUCCESS
# ==========================================

print("\n==========================================")
print("DATASET VERIFICATION COMPLETE")
print("==========================================")


# ==========================================
# NUMERICAL DATA CHECK
# ==========================================

print("\n==========================================")
print("NUMERICAL DATA CHECK")
print("==========================================")

print("X_train dtype:", X_train.dtype)
print("X_val dtype:", X_val.dtype)
print("X_test dtype:", X_test.dtype)

print("\nX_train contiguous:",
      X_train.flags["C_CONTIGUOUS"])

print("X_val contiguous:",
      X_val.flags["C_CONTIGUOUS"])

print("X_test contiguous:",
      X_test.flags["C_CONTIGUOUS"])

print("\nX_train shape:", X_train.shape)

print("\nChecking finite values:")

print(
    "Training:",
    np.all(np.isfinite(X_train))
)

print(
    "Validation:",
    np.all(np.isfinite(X_val))
)

print(
    "Testing:",
    np.all(np.isfinite(X_test))
)

print("\n==========================================")
print("MATRIX MULTIPLICATION TEST")
print("==========================================")

test_matrix = X_train[:10]

weights = np.random.randn(
    20,
    X_train.shape[1]
).astype(np.float64)

result = test_matrix.astype(
    np.float64
) @ weights.T

print("Matrix multiplication successful!")
print("Result shape:", result.shape)
print("Result min:", np.min(result))
print("Result max:", np.max(result))
print(
    "Result contains NaN:",
    np.any(np.isnan(result))
)
print(
    "Result contains infinity:",
    np.any(np.isinf(result))
)

# ==========================================
# CLASS FEATURE ANALYSIS
# ==========================================

print("\n==========================================")
print("CLASS FEATURE ANALYSIS")
print("==========================================")

for class_id, word in enumerate(classes):

    class_samples = X_train[
        y_train == class_id
    ]

    print(
        f"{word:8s} : "
        f"{len(class_samples):4d} samples | "
        f"mean = {np.mean(class_samples): .4f} | "
        f"std = {np.std(class_samples): .4f}"
    )

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report




# ==========================================
# CREATE MLP MODEL
# ==========================================

print("\n==========================================")
print("CREATING MLP MODEL")
print("==========================================")

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size=32,
    learning_rate_init=0.001,
    max_iter=100,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=10,
    random_state=42,
    verbose=True
)

print("MLP created successfully!")


# ==========================================
# TRAIN MODEL
# ==========================================

print("\n==========================================")
print("TRAINING MLP")
print("==========================================")

model.fit(
    X_train,
    y_train
)

print("\nMLP training completed!")


# ==========================================
# VALIDATION
# ==========================================

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
    "Validation accuracy:",
    val_accuracy
)


# ==========================================
# TEST
# ==========================================

print("\n==========================================")
print("TEST RESULTS")
print("==========================================")

y_test_pred = model.predict(
    X_test
)

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

print(
    "Test accuracy:",
    test_accuracy
)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=classes
    )
)

# ==========================================
# CONVERT FEATURES TO FLOAT64
# ==========================================

X_train = np.ascontiguousarray(
    X_train,
    dtype=np.float64
)

X_val = np.ascontiguousarray(
    X_val,
    dtype=np.float64
)

X_test = np.ascontiguousarray(
    X_test,
    dtype=np.float64
)

print("\n==========================================")
print("FINAL NUMERICAL FORMAT")
print("==========================================")

print("X_train dtype:", X_train.dtype)
print("X_val dtype:", X_val.dtype)
print("X_test dtype:", X_test.dtype)

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

from sklearn.linear_model import LogisticRegression

print("\n==========================================")
print("LOGISTIC REGRESSION BASELINE")
print("==========================================")

logreg = LogisticRegression(
    max_iter=2000,
    solver="liblinear",
    C=1.0,
    random_state=42
)

logreg.fit(X_train, y_train)

val_accuracy = logreg.score(X_val, y_val)
test_accuracy = logreg.score(X_test, y_test)

print("Validation accuracy:", val_accuracy)
print("Test accuracy:", test_accuracy)


# ==========================================
# VALIDATION
# ==========================================

y_val_logistic = logreg.predict(
    X_val
)

logistic_val_accuracy = accuracy_score(
    y_val,
    y_val_logistic
)


# ==========================================
# TEST
# ==========================================

y_test_logistic = logreg.predict(
    X_test
)

logistic_test_accuracy = accuracy_score(
    y_test,
    y_test_logistic
)


print(
    "Validation accuracy:",
    logistic_val_accuracy
)

print(
    "Test accuracy:",
    logistic_test_accuracy
)