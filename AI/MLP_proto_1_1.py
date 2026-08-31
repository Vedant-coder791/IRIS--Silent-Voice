# ============================================================
# IRIS - MLP PROTO 1.1
# Berkeley Closed-Vocabulary Silent Speech Dataset
#
# PURPOSE:
# Fix numerical instability encountered in Proto 1.
#
# Pipeline:
#   EMG recording
#       ↓
#   preprocessing
#       ↓
#   stable EMG features
#       ↓
#   feature cleaning / clipping
#       ↓
#   StandardScaler
#       ↓
#   MLP
#       ↓
#   evaluation
#
# IMPORTANT:
# Chunks are NOT treated as words.
# Each recording is assigned a label only when it contains
# exactly ONE of the selected target words.
# ============================================================

import os
import json
import pickle

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

MODEL_PATH = (
    "/Users/vedantdwivedi/Desktop/IRIS/AI/"
    "MLP_proto1_1_model.pkl"
)

RANDOM_STATE = 42

# ------------------------------------------------------------
# Target vocabulary
# ------------------------------------------------------------

TARGET_WORDS = [
    "AM",
    "PM",
    "DECEMBER",
    "AUGUST",
    "MARCH"
]

# ------------------------------------------------------------
# EMG settings
# ------------------------------------------------------------

FS = 1000

LOWCUT = 20.0
HIGHCUT = 450.0

NOTCH_FREQ = 50.0
NOTCH_Q = 30.0


# ============================================================
# 2. HEADER
# ============================================================

print("=" * 70)
print("IRIS - MLP PROTO 1.1")
print("Berkeley Closed-Vocabulary Silent Speech")
print("=" * 70)

print("\nDataset:")
print(DATASET_PATH)

print("\nTarget vocabulary:")

for i, word in enumerate(TARGET_WORDS, 1):
    print(f"  {i}. {word}")

print("\nSampling rate:", FS, "Hz")


# ============================================================
# 3. FILTERS
# ============================================================

def bandpass_filter(signal, fs=FS):

    nyquist = fs / 2.0

    low = LOWCUT / nyquist
    high = HIGHCUT / nyquist

    sos = butter(
        4,
        [low, high],
        btype="bandpass",
        output="sos"
    )

    return sosfiltfilt(sos, signal)


def notch_filter(signal, fs=FS):

    w0 = NOTCH_FREQ / (fs / 2.0)

    b, a = iirnotch(
        w0,
        NOTCH_Q
    )

    return filtfilt(
        b,
        a,
        signal
    )


# ============================================================
# 4. PREPROCESS CHANNEL
# ============================================================

def preprocess_channel(signal):

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    # Remove NaN / infinity before processing
    signal = np.nan_to_num(
        signal,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Remove DC
    signal = signal - np.mean(signal)

    # Bandpass
    signal = bandpass_filter(signal)

    # Notch
    signal = notch_filter(signal)

    # Remove any numerical problems caused by filtering
    signal = np.nan_to_num(
        signal,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return signal


# ============================================================
# 5. STABLE FEATURE EXTRACTION
# ============================================================

def extract_channel_features(signal):

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Absolute amplitude
    # --------------------------------------------------------

    abs_signal = np.abs(signal)

    mean_abs = np.mean(abs_signal)

    rms = np.sqrt(
        np.mean(signal ** 2)
    )

    std = np.std(signal)

    peak = np.max(abs_signal)

    # --------------------------------------------------------
    # Log amplitude features
    #
    # log1p prevents very large values from dominating.
    # --------------------------------------------------------

    log_mean_abs = np.log1p(
        min(mean_abs, 1e10)
    )

    log_rms = np.log1p(
        min(rms, 1e10)
    )

    log_std = np.log1p(
        min(std, 1e10)
    )

    log_peak = np.log1p(
        min(peak, 1e10)
    )

    # --------------------------------------------------------
    # Waveform statistics
    # --------------------------------------------------------

    if len(signal) > 1:

        diff = np.diff(signal)

        waveform_length = np.sum(
            np.abs(diff)
        )

        waveform_length = min(
            waveform_length,
            1e12
        )

        log_waveform_length = np.log1p(
            waveform_length
        )

        zero_crossings = np.sum(
            signal[:-1] * signal[1:] < 0
        )

    else:

        log_waveform_length = 0.0
        zero_crossings = 0.0

    # Normalize zero crossings by signal length
    zero_crossing_rate = (
        zero_crossings / max(len(signal), 1)
    )

    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------

    fft = np.fft.rfft(signal)

    magnitude = np.abs(fft)

    frequencies = np.fft.rfftfreq(
        len(signal),
        d=1.0 / FS
    )

    power = magnitude ** 2

    # Remove problematic values
    power = np.nan_to_num(
        power,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    total_power = np.sum(power)

    if not np.isfinite(total_power):
        total_power = 0.0

    total_power = max(
        total_power,
        1e-12
    )

    # --------------------------------------------------------
    # Spectral centroid
    # --------------------------------------------------------

    spectral_centroid = (
        np.sum(frequencies * power)
        / total_power
    )

    if not np.isfinite(spectral_centroid):
        spectral_centroid = 0.0

    # --------------------------------------------------------
    # Spectral bandwidth
    # --------------------------------------------------------

    spectral_bandwidth = np.sqrt(
        np.sum(
            (
                frequencies
                - spectral_centroid
            ) ** 2
            * power
        )
        / total_power
    )

    if not np.isfinite(spectral_bandwidth):
        spectral_bandwidth = 0.0

    # --------------------------------------------------------
    # Frequency band power
    # --------------------------------------------------------

    def normalized_band_power(
        low,
        high
    ):

        mask = (
            (frequencies >= low)
            &
            (frequencies < high)
        )

        value = np.sum(
            power[mask]
        )

        value /= total_power

        if not np.isfinite(value):
            value = 0.0

        return value

    power_20_50 = normalized_band_power(
        20,
        50
    )

    power_50_100 = normalized_band_power(
        50,
        100
    )

    power_100_200 = normalized_band_power(
        100,
        200
    )

    power_200_400 = normalized_band_power(
        200,
        400
    )

    # --------------------------------------------------------
    # Log total power
    # --------------------------------------------------------

    log_total_power = np.log1p(
        min(total_power, 1e20)
    )

    # --------------------------------------------------------
    # Final feature vector
    #
    # 14 features / channel
    # × 8 channels
    # = 112 features
    # --------------------------------------------------------

    features = np.array([
        log_mean_abs,
        log_rms,
        log_std,
        log_peak,
        log_waveform_length,
        zero_crossing_rate,
        spectral_centroid,
        spectral_bandwidth,
        power_20_50,
        power_50_100,
        power_100_200,
        power_200_400,
        log_total_power,
        len(signal)
    ], dtype=np.float64)

    # Final safety check
    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return features


# ============================================================
# 6. RECORDING FEATURES
# ============================================================

def extract_recording_features(emg):

    if emg.ndim != 2:

        raise ValueError(
            f"Expected 2D EMG array, got {emg.shape}"
        )

    # Convert 8 × samples → samples × 8
    if (
        emg.shape[1] != 8
        and emg.shape[0] == 8
    ):

        emg = emg.T

    if emg.shape[1] != 8:

        raise ValueError(
            f"Expected 8 channels, got {emg.shape}"
        )

    all_features = []

    for channel in range(8):

        signal = emg[:, channel]

        signal = preprocess_channel(
            signal
        )

        features = extract_channel_features(
            signal
        )

        all_features.extend(
            features
        )

    features = np.asarray(
        all_features,
        dtype=np.float64
    )

    # Final protection
    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return features


# ============================================================
# 7. READ INFO JSON
# ============================================================

def read_info(path):

    with open(path, "r") as f:

        return json.load(f)


# ============================================================
# 8. GET LABEL
# ============================================================

def get_target_label(info):

    text = info.get(
        "text",
        ""
    )

    if not isinstance(
        text,
        str
    ):

        return None

    text = text.upper()

    words = (
        text
        .replace(":", " ")
        .split()
    )

    matches = [
        word
        for word in words
        if word in TARGET_WORDS
    ]

    unique_matches = list(
        set(matches)
    )

    # Keep ONLY recordings containing
    # exactly one target vocabulary word.
    if len(unique_matches) != 1:

        return None

    return unique_matches[0]


# ============================================================
# 9. FIND EMG FILES
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASET")
print("=" * 70)

emg_files = [
    f
    for f in os.listdir(
        DATASET_PATH
    )
    if f.endswith("_emg.npy")
]


def numeric_sort(filename):

    try:

        return int(
            filename
            .replace(
                "_emg.npy",
                ""
            )
        )

    except:

        return 999999


emg_files.sort(
    key=numeric_sort
)

print(
    "\nEMG recordings found:",
    len(emg_files)
)


# ============================================================
# 10. LOAD DATA
# ============================================================

X = []
y = []

files_seen = 0
files_used = 0
files_skipped = 0

class_counts = {
    word: 0
    for word in TARGET_WORDS
}


for filename in emg_files:

    files_seen += 1

    emg_path = os.path.join(
        DATASET_PATH,
        filename
    )

    info_path = os.path.join(
        DATASET_PATH,
        filename.replace(
            "_emg.npy",
            "_info.json"
        )
    )

    if not os.path.exists(
        info_path
    ):

        files_skipped += 1
        continue

    try:

        info = read_info(
            info_path
        )

        label = get_target_label(
            info
        )

        if label is None:

            files_skipped += 1
            continue

        emg = np.load(
            emg_path
        )

        if emg.ndim != 2:

            files_skipped += 1
            continue

        if emg.shape[1] != 8:

            if emg.shape[0] == 8:

                emg = emg.T

            else:

                files_skipped += 1
                continue

        features = extract_recording_features(
            emg
        )

        # ----------------------------------------------------
        # Numerical safety
        # ----------------------------------------------------

        if not np.all(
            np.isfinite(features)
        ):

            print(
                "WARNING: non-finite features:",
                filename
            )

            files_skipped += 1
            continue

        X.append(features)

        y.append(label)

        class_counts[label] += 1

        files_used += 1

    except Exception as e:

        print(
            f"WARNING: {filename}: {e}"
        )

        files_skipped += 1


X = np.asarray(
    X,
    dtype=np.float64
)

y = np.asarray(
    y,
    dtype=str
)


# ============================================================
# 11. DATASET SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE DATASET SUMMARY")
print("=" * 70)

print("\nFiles seen:", files_seen)
print("Files used:", files_used)
print("Files skipped:", files_skipped)

print("\nFeature matrix shape:")
print(X.shape)

print(
    "\nFeatures per recording:",
    X.shape[1]
)

print("\nClass distribution:")

for word in TARGET_WORDS:

    print(
        f"  {word:10s}: "
        f"{class_counts[word]:4d}"
    )


# ============================================================
# 12. NUMERICAL DIAGNOSTICS
# ============================================================

print("\n" + "=" * 70)
print("NUMERICAL FEATURE CHECK")
print("=" * 70)

nan_count = np.isnan(X).sum()
inf_count = np.isinf(X).sum()

print("\nNaN values:", nan_count)
print("Infinity values:", inf_count)

print(
    "\nAll features finite:",
    np.all(np.isfinite(X))
)

print(
    "\nMinimum feature:",
    np.min(X)
)

print(
    "Maximum feature:",
    np.max(X)
)

print(
    "Mean absolute feature:",
    np.mean(np.abs(X))
)


# ============================================================
# 13. FINAL FEATURE CLIPPING
# ============================================================

# This prevents a single unusual recording from
# dominating the MLP.

FEATURE_LIMIT = 50.0

X = np.clip(
    X,
    -FEATURE_LIMIT,
    FEATURE_LIMIT
)

print(
    "\nFeatures clipped to:",
    f"+/- {FEATURE_LIMIT}"
)


# ============================================================
# 14. TRAIN / TEST SPLIT
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

print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# 15. MLP
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

            hidden_layer_sizes=(
                64,
                32
            ),

            activation="relu",

            solver="adam",

            alpha=0.001,

            batch_size=16,

            learning_rate_init=0.0005,

            max_iter=500,

            # IMPORTANT:
            # Disabled for Proto 1.1.
            # This lets us diagnose the MLP
            # without validation-score issues.
            early_stopping=False,

            random_state=RANDOM_STATE,

            verbose=True
        )
    )
])


# ============================================================
# 16. TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

model.fit(
    X_train,
    y_train
)


# ============================================================
# 17. PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION")
print("=" * 70)

y_pred = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    f"\nTest accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

print(
    "\nClassification report:\n"
)

print(
    classification_report(
        y_test,
        y_pred,
        labels=TARGET_WORDS,
        zero_division=0
    )
)


# ============================================================
# 19. SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

for i in range(
    min(20, len(X_test))
):

    print(
        f"Actual: "
        f"{y_test[i]:10s} "
        f"Predicted: "
        f"{y_pred[i]:10s}"
    )


# ============================================================
# 20. SAVE MODEL
# ============================================================

with open(
    MODEL_PATH,
    "wb"
) as f:

    pickle.dump(
        {
            "model": model,
            "target_words": TARGET_WORDS,
            "sampling_rate": FS,
            "feature_count": X.shape[1],
            "prototype": "MLP Proto 1.1"
        },
        f
    )


print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print("\nSaved to:")
print(MODEL_PATH)

print("\n" + "=" * 70)
print("MLP PROTO 1.1 COMPLETE")
print("=" * 70)