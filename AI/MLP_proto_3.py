import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score


# ==========================================
# PATH
# ==========================================

DATASET_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/"
    "Data/dataset_balanced.npz"
)


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
# LOAD DATASET
# ==========================================

print("==========================================")
print("LOADING BALANCED DATASET")
print("==========================================")

data = np.load(DATASET_PATH)

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_val"]
y_val = data["y_val"]

X_test = data["X_test"]
y_test = data["y_test"]


print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Testing:", X_test.shape, y_test.shape)


# ==========================================
# DATA CHECK
# ==========================================

print("\n==========================================")
print("DATASET CHECK")
print("==========================================")

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

print(
    "Training dtype:",
    X_train.dtype
)

print(
    "Number of features:",
    X_train.shape[1]
)

print(
    "Number of classes:",
    len(WORD_CLASSES)
)


# ==========================================
# CLASS DISTRIBUTION
# ==========================================

print("\n==========================================")
print("TRAINING CLASS DISTRIBUTION")
print("==========================================")

counts = np.bincount(
    y_train,
    minlength=len(WORD_CLASSES)
)

for i, word in enumerate(WORD_CLASSES):

    print(
        f"{word:<8}: {counts[i]:4d}"
    )


# ==========================================
# CREATE MLP
# ==========================================

print("\n==========================================")
print("CREATING MLP MODEL")
print("==========================================")


model = MLPClassifier(

    hidden_layer_sizes=(128, 64),

    activation="relu",

    solver="adam",

    alpha=0.0005,

    batch_size=32,

    learning_rate_init=0.001,

    max_iter=100,

    early_stopping=True,

    validation_fraction=0.15,

    n_iter_no_change=15,

    tol=0.0001,

    random_state=42,

    verbose=True
)


print("MLP created successfully!")


# ==========================================
# TRAIN
# ==========================================

print("\n==========================================")
print("TRAINING MLP PROTO 3")
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

y_val_pred = model.predict(X_val)

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

y_test_pred = model.predict(X_test)

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
        target_names=WORD_CLASSES,
        zero_division=0
    )
)


# ==========================================
# TRAINING INFORMATION
# ==========================================

print("\n==========================================")
print("TRAINING INFORMATION")
print("==========================================")

print(
    "Iterations:",
    model.n_iter_
)

print(
    "Final loss:",
    model.loss_
)

if hasattr(model, "best_validation_score_"):

    print(
        "Best internal validation score:",
        model.best_validation_score_
    )


# ==========================================
# SAVE MODEL
# ==========================================

MODEL_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/"
    "Data/mlp_proto_3.npz"
)

print("\n==========================================")
print("SAVING MODEL")
print("==========================================")

np.savez(
    MODEL_PATH,

    coefs=np.array(
        model.coefs_,
        dtype=object
    ),

    intercepts=np.array(
        model.intercepts_,
        dtype=object
    ),

    classes=model.classes_
)

print(
    "Model parameters saved to:"
)

print(MODEL_PATH)