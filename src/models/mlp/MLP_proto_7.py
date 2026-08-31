# ============================================================
# IRIS - MLP PROTO 1
# Berkeley Closed-Vocabulary Silent Speech Dataset
#
# STAGE 1:
#   Full-recording EMG
#   -> preprocessing
#   -> MFCC-style features
#   -> MLP classifier
#
# IMPORTANT:
# Chunks are NOT treated as words.
# ============================================================

import os
import json
import numpy as np

from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


# ============================================================
# 1. SETTINGS
# ============================================================

DATASET_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/emg_data/"
    "closed_vocab/silent/5-19_silent"
)

RANDOM_STATE = 42

# Sampling rate of Berkeley EMG
FS = 1000

# ------------------------------------------------------------
# Proto 1 vocabulary
# ------------------------------------------------------------

TARGET_WORDS = [
    "AM",
    "PM",
    "DECEMBER",
    "AUGUST",
    "MARCH"
]

# ------------------------------------------------------------
# Signal processing
# ------------------------------------------------------------

LOWCUT = 20.0
HIGHCUT = 450.0

NOTCH_FREQ = 50.0
NOTCH_Q = 30.0

# ------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------

N_MELS = 20
N_MFCC = 13

N_FFT = 128
HOP_LENGTH = 64

# ============================================================
# 2. PRINT CONFIGURATION
# ============================================================

print("=" * 70)
print("IRIS - MLP PROTO 1")
print("Berkeley Closed-Vocabulary Silent Speech")
print("=" * 70)

print("\nDataset:")
print(DATASET_PATH)

print("\nTarget vocabulary:")
for i, word in enumerate(TARGET_WORDS, 1):
    print(f"  {i}. {word}")

print("\nSampling rate:", FS, "Hz")


# ============================================================
# 3. FILTER FUNCTIONS
# ============================================================

def bandpass_filter(signal, fs=FS, lowcut=LOWCUT, highcut=HIGHCUT):

    nyquist = fs / 2.0

    low = lowcut / nyquist
    high = highcut / nyquist

    sos = butter(
        4,
        [low, high],
        btype="bandpass",
        output="sos"
    )

    return sosfiltfilt(sos, signal)


def notch_filter(signal, fs=FS, freq=NOTCH_FREQ, q=NOTCH_Q):

    w0 = freq / (fs / 2.0)

    b, a = iirnotch(w0, q)

    return filtfilt(b, a, signal)


# ============================================================
# 4. PREPROCESS ONE CHANNEL
# ============================================================

def preprocess_channel(signal):

    signal = np.asarray(signal, dtype=np.float64)

    # Remove DC offset
    signal = signal - np.mean(signal)

    # Bandpass
    signal = bandpass_filter(signal)

    # 50 Hz notch
    signal = notch_filter(signal)

    # Normalize
    std = np.std(signal)

    if std > 1e-8:
        signal = signal / std

    return signal


# ============================================================
# 5. SIMPLE EMG FEATURE EXTRACTION
# ============================================================

def extract_channel_features(signal):

    signal = np.asarray(signal)

    # --------------------------------------------------------
    # Time-domain features
    # --------------------------------------------------------

    mean_abs = np.mean(np.abs(signal))

    rms = np.sqrt(np.mean(signal ** 2))

    std = np.std(signal)

    peak = np.max(np.abs(signal))

    waveform_length = np.sum(
        np.abs(np.diff(signal))
    )

    zero_crossings = np.sum(
        signal[:-1] * signal[1:] < 0
    )

    # --------------------------------------------------------
    # Frequency-domain features
    # --------------------------------------------------------

    fft = np.fft.rfft(signal)

    magnitude = np.abs(fft)

    frequencies = np.fft.rfftfreq(
        len(signal),
        d=1.0 / FS
    )

    power = magnitude ** 2

    total_power = np.sum(power) + 1e-12

    spectral_centroid = (
        np.sum(frequencies * power)
        / total_power
    )

    spectral_bandwidth = np.sqrt(
        np.sum(
            ((frequencies - spectral_centroid) ** 2)
            * power
        )
        / total_power
    )

    # --------------------------------------------------------
    # Band powers
    # --------------------------------------------------------

    def band_power(low, high):

        mask = (
            (frequencies >= low)
            &
            (frequencies < high)
        )

        return np.sum(power[mask])

    power_20_50 = band_power(20, 50)
    power_50_100 = band_power(50, 100)
    power_100_200 = band_power(100, 200)
    power_200_400 = band_power(200, 400)

    # Normalize powers
    power_20_50 /= total_power
    power_50_100 /= total_power
    power_100_200 /= total_power
    power_200_400 /= total_power

    return np.array([
        mean_abs,
        rms,
        std,
        peak,
        waveform_length,
        zero_crossings,
        spectral_centroid,
        spectral_bandwidth,
        power_20_50,
        power_50_100,
        power_100_200,
        power_200_400
    ], dtype=np.float64)


# ============================================================
# 6. EXTRACT FEATURES FROM ALL CHANNELS
# ============================================================

def extract_recording_features(emg):

    # Expected shape:
    # samples x channels

    if emg.ndim != 2:
        raise ValueError(
            f"Expected 2D EMG array, got {emg.shape}"
        )

    # Make sure channels are columns
    if emg.shape[1] != 8 and emg.shape[0] == 8:
        emg = emg.T

    if emg.shape[1] != 8:
        raise ValueError(
            f"Expected 8 channels, got {emg.shape}"
        )

    all_features = []

    for channel in range(8):

        signal = emg[:, channel]

        signal = preprocess_channel(signal)

        features = extract_channel_features(signal)

        all_features.extend(features)

    return np.asarray(all_features)


# ============================================================
# 7. READ JSON
# ============================================================

def read_info(info_path):

    with open(info_path, "r") as f:
        info = json.load(f)

    return info


# ============================================================
# 8. FIND TARGET WORD IN RECORDING
# ============================================================

def get_target_label(info):

    text = info.get("text", "")

    if not isinstance(text, str):
        return None

    text = text.upper()

    words = text.replace(":", " ").split()

    matching_words = [
        word
        for word in words
        if word in TARGET_WORDS
    ]

    # --------------------------------------------------------
    # IMPORTANT
    #
    # A sentence can contain multiple target words.
    #
    # To avoid assigning one recording to multiple classes,
    # Proto 1 only keeps recordings containing EXACTLY ONE
    # target word.
    # --------------------------------------------------------

    unique_matches = list(set(matching_words))

    if len(unique_matches) != 1:
        return None

    return unique_matches[0]


# ============================================================
# 9. LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASET")
print("=" * 70)

X = []
y = []

files_seen = 0
files_used = 0
files_skipped = 0

class_counts = {
    word: 0
    for word in TARGET_WORDS
}


emg_files = sorted(
    [
        f
        for f in os.listdir(DATASET_PATH)
        if f.endswith("_emg.npy")
    ],
    key=lambda x: int(
        x.replace("_emg.npy", "")
    )
)


print("\nEMG recordings found:", len(emg_files))


for filename in emg_files:

    files_seen += 1

    emg_path = os.path.join(
        DATASET_PATH,
        filename
    )

    info_filename = filename.replace(
        "_emg.npy",
        "_info.json"
    )

    info_path = os.path.join(
        DATASET_PATH,
        info_filename
    )

    if not os.path.exists(info_path):
        files_skipped += 1
        continue

    try:

        info = read_info(info_path)

        label = get_target_label(info)

        if label is None:
            files_skipped += 1
            continue

        emg = np.load(emg_path)

        if emg.ndim != 2:
            files_skipped += 1
            continue

        if emg.shape[1] != 8:

            if emg.shape[0] == 8:
                emg = emg.T

            else:
                files_skipped += 1
                continue

        # ----------------------------------------------------
        # Extract features
        # ----------------------------------------------------

        features = extract_recording_features(emg)

        if not np.all(np.isfinite(features)):
            files_skipped += 1
            continue

        X.append(features)
        y.append(label)

        class_counts[label] += 1
        files_used += 1

    except Exception as e:

        print(
            f"WARNING: Could not process {filename}: {e}"
        )

        files_skipped += 1


X = np.asarray(X)
y = np.asarray(y)


# ============================================================
# 10. DATASET SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE DATASET SUMMARY")
print("=" * 70)

print("\nFiles seen:", files_seen)
print("Files used:", files_used)
print("Files skipped:", files_skipped)

print("\nFeature matrix shape:")
print(X.shape)

print("\nNumber of features per recording:")
print(X.shape[1])

print("\nClass distribution:")

for word in TARGET_WORDS:
    print(
        f"  {word:10s}: {class_counts[word]:4d}"
    )


# ============================================================
# 11. CHECK DATASET
# ============================================================

if len(X) == 0:

    raise RuntimeError(
        "\nNo usable recordings were found."
    )


for word in TARGET_WORDS:

    if class_counts[word] < 5:

        print(
            f"\nWARNING: Only "
            f"{class_counts[word]} samples for {word}"
        )


# ============================================================
# 12. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 13. MLP MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING MLP")
print("=" * 70)

model = Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "mlp",
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",

            alpha=0.001,

            batch_size=16,

            learning_rate_init=0.001,

            max_iter=500,

            early_stopping=True,

            validation_fraction=0.15,

            n_iter_no_change=30,

            random_state=RANDOM_STATE,

            verbose=True
        )
    )
])


# ============================================================
# 14. TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

model.fit(
    X_train,
    y_train
)


# ============================================================
# 15. PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION")
print("=" * 70)

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    f"\nTest accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

print("\nClassification report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        labels=TARGET_WORDS,
        zero_division=0
    )
)


# ============================================================
# 17. SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

for i in range(
    min(20, len(X_test))
):

    print(
        f"Actual: {y_test[i]:10s} "
        f"Predicted: {y_pred[i]:10s}"
    )


# ============================================================
# 18. SAVE MODEL
# ============================================================

import pickle

MODEL_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/AI/"
    "MLP_proto1_model.pkl"
)

with open(MODEL_PATH, "wb") as f:

    pickle.dump(
        {
            "model": model,
            "target_words": TARGET_WORDS,
            "sampling_rate": FS,
            "feature_count": X.shape[1]
        },
        f
    )


print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print("\nSaved to:")
print(MODEL_PATH)

print("\n" + "=" * 70)
print("MLP PROTO 1 COMPLETE")
print("=" * 70)