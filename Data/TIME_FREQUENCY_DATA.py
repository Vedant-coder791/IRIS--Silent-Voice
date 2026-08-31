

# ============================================================
# IRIS - BERKELEY SILENT SPEECH
# TIME-FREQUENCY FEATURE EXTRACTION
#
# Version 1.0
#
# Representation:
#       Raw EMG
#          ↓
#       Windowing
#          ↓
#       STFT
#          ↓
#       Power Spectrogram
#          ↓
#       Log Power
#          ↓
#       Time-Frequency Representation
#
# Dataset:
#       Berkeley Silent Speech EMG
#
# Input:
#       .npy EMG files
#
# Expected EMG shape:
#       (samples, channels)
#
# Example:
#       (2664, 8)
#
# Output:
#       TIME_FREQUENCY_DATA.npz
#
# ============================================================


import os
import json
import numpy as np

from scipy.signal import stft, butter, sosfiltfilt


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "/Users/vedantdwivedi/Desktop/IRIS/emg_data"

# We specifically start with the silent closed vocabulary
SILENT_PATH = os.path.join(
    DATASET_PATH,
    "closed_vocab",
    "silent"
)

OUTPUT_PATH = "/Users/vedantdwivedi/Desktop/IRIS/Data/TIME_FREQUENCY_DATA.npz"


# ============================================================
# EMG SETTINGS
# ============================================================

# Berkeley EMG sampling rate
FS = 600


# Number of channels expected
N_CHANNELS = 8


# ============================================================
# WINDOW SETTINGS
# ============================================================

# 200 samples at 600 Hz
# ≈ 333 ms
WINDOW_SIZE = 200

# 50% overlap
STEP_SIZE = 100


# ============================================================
# STFT SETTINGS
# ============================================================

# STFT FFT size
N_FFT = 64

# STFT window
STFT_WINDOW = 64

# STFT hop
STFT_HOP = 16


# Frequency range useful for EMG
FMIN = 20
FMAX = 250


# ============================================================
# FILTER SETTINGS
# ============================================================

HIGH_PASS = 20.0


# ============================================================
# LIMITS
# ============================================================

# Set to None to process everything.
#
# During testing, you can set:
#
# MAX_FILES = 20
#
# Once everything works:
#
# MAX_FILES = None

MAX_FILES = None


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("=" * 70)
print("IRIS - BERKELEY TIME-FREQUENCY EXTRACTION")
print("=" * 70)

print()
print("Dataset:")
print(DATASET_PATH)

print()
print("Silent dataset:")
print(SILENT_PATH)

print()
print("Sampling rate:", FS, "Hz")
print("Expected channels:", N_CHANNELS)
print("Window size:", WINDOW_SIZE)
print("Step size:", STEP_SIZE)
print("FFT size:", N_FFT)
print("STFT hop:", STFT_HOP)

print()


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.isdir(DATASET_PATH):

    raise RuntimeError(
        "\nDataset directory does not exist:\n"
        + DATASET_PATH
    )


if not os.path.isdir(SILENT_PATH):

    raise RuntimeError(
        "\nSilent dataset directory does not exist:\n"
        + SILENT_PATH
    )


print("Dataset directory: OK")
print("Silent directory: OK")


# ============================================================
# FIND EMG FILES
# ============================================================

print()
print("=" * 70)
print("SEARCHING FOR SILENT EMG FILES")
print("=" * 70)


emg_files = []


for root, dirs, files in os.walk(SILENT_PATH):

    for filename in files:

        if filename.endswith("_emg.npy"):

            full_path = os.path.join(root, filename)

            emg_files.append(full_path)


emg_files.sort()


print()
print("EMG files found:", len(emg_files))


if len(emg_files) == 0:

    raise RuntimeError(
        "\nNo *_emg.npy files were found.\n"
        "Check the Berkeley dataset structure."
    )


if MAX_FILES is not None:

    emg_files = emg_files[:MAX_FILES]

    print("Testing with first", MAX_FILES, "files.")


# ============================================================
# HIGH-PASS FILTER
# ============================================================

def highpass_filter(signal, fs, cutoff=20.0):

    """
    High-pass filter for EMG.

    Removes very-low-frequency drift.
    """

    sos = butter(
        4,
        cutoff,
        btype="highpass",
        fs=fs,
        output="sos"
    )

    return sosfiltfilt(sos, signal)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_channel(signal):

    """
    Robust channel normalization.

    Centers the signal and scales by standard deviation.
    """

    signal = signal.astype(np.float64)

    signal = signal - np.mean(signal)

    std = np.std(signal)

    if std < 1e-8:

        return np.zeros_like(signal)

    return signal / std


# ============================================================
# TIME-FREQUENCY EXTRACTION
# ============================================================

def extract_time_frequency(
    signal,
    fs=FS
):

    """
    Convert one EMG window into a
    time-frequency representation.

    Returns:

        spectrogram shape:

        (frequency_bins, time_bins)
    """

    # --------------------------------------------------------
    # STFT
    # --------------------------------------------------------

    frequencies, times, Zxx = stft(
        signal,
        fs=fs,
        window="hann",
        nperseg=STFT_WINDOW,
        noverlap=STFT_WINDOW - STFT_HOP,
        nfft=N_FFT,
        boundary=None,
        padded=False
    )

    # --------------------------------------------------------
    # POWER
    # --------------------------------------------------------

    power = np.abs(Zxx) ** 2


    # --------------------------------------------------------
    # SELECT EMG FREQUENCIES
    # --------------------------------------------------------

    frequency_mask = (
        (frequencies >= FMIN)
        &
        (frequencies <= FMAX)
    )

    power = power[frequency_mask]


    # --------------------------------------------------------
    # LOG POWER
    #
    # Log compression makes large amplitudes
    # less dominant.
    # --------------------------------------------------------

    power = np.log10(
        power + 1e-10
    )


    # --------------------------------------------------------
    # STANDARDIZE SPECTROGRAM
    # --------------------------------------------------------

    mean = np.mean(power)
    std = np.std(power)

    if std > 1e-8:

        power = (power - mean) / std

    else:

        power = np.zeros_like(power)


    return power.astype(np.float32)


# ============================================================
# PROCESS ONE EMG RECORDING
# ============================================================

def process_recording(filepath):

    """
    Load and process one Berkeley EMG recording.

    Returns:
        list of time-frequency windows
    """

    try:

        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        emg = np.load(filepath)


        # ----------------------------------------------------
        # CHECK DIMENSIONS
        # ----------------------------------------------------

        if emg.ndim != 2:

            raise ValueError(
                f"Expected 2D EMG array, got {emg.shape}"
            )


        # ----------------------------------------------------
        # HANDLE ORIENTATION
        # ----------------------------------------------------

        # Berkeley example:
        #
        # (2664, 8)
        #
        # = samples × channels

        if emg.shape[1] == N_CHANNELS:

            # Already correct
            pass

        elif emg.shape[0] == N_CHANNELS:

            # Transpose
            emg = emg.T

        else:

            raise ValueError(
                f"Unexpected EMG shape: {emg.shape}"
            )


        # ----------------------------------------------------
        # LIMIT TO EXPECTED CHANNELS
        # ----------------------------------------------------

        emg = emg[:, :N_CHANNELS]


        # ----------------------------------------------------
        # PROCESS CHANNELS
        # ----------------------------------------------------

        processed = np.zeros_like(
            emg,
            dtype=np.float64
        )


        for channel in range(N_CHANNELS):

            x = emg[:, channel]

            # High-pass
            x = highpass_filter(
                x,
                FS,
                HIGH_PASS
            )

            # Normalize
            x = normalize_channel(x)

            processed[:, channel] = x


        # ----------------------------------------------------
        # CREATE WINDOWS
        # ----------------------------------------------------

        n_samples = processed.shape[0]

        windows = []


        start = 0

        while start + WINDOW_SIZE <= n_samples:

            end = start + WINDOW_SIZE

            window = processed[
                start:end
            ]


            # ------------------------------------------------
            # EXTRACT T-F FROM EACH CHANNEL
            # ------------------------------------------------

            channel_spectrograms = []


            for channel in range(N_CHANNELS):

                signal = window[:, channel]

                tf = extract_time_frequency(
                    signal,
                    FS
                )

                channel_spectrograms.append(tf)


            # ------------------------------------------------
            # STACK CHANNELS
            # ------------------------------------------------

            #
            # Result:
            #
            # (channels, frequency, time)
            #

            tf_window = np.stack(
                channel_spectrograms,
                axis=0
            )


            windows.append(tf_window)


            start += STEP_SIZE


        return windows


    except Exception as e:

        raise RuntimeError(
            f"Failed processing {filepath}: {e}"
        )


# ============================================================
# MAIN EXTRACTION
# ============================================================

all_features = []
metadata = []

valid_recordings = 0
failed_recordings = 0
total_windows = 0


print()
print("=" * 70)
print("STARTING EXTRACTION")
print("=" * 70)


for index, filepath in enumerate(emg_files):

    filename = os.path.basename(filepath)


    try:

        windows = process_recording(
            filepath
        )


        if len(windows) == 0:

            print(
                f"[SKIP] {filename}: "
                "recording too short"
            )

            continue


        valid_recordings += 1


        # ----------------------------------------------------
        # SAVE EACH WINDOW
        # ----------------------------------------------------

        for window_index, tf_window in enumerate(windows):

            all_features.append(
                tf_window
            )


            metadata.append({
                "file": filepath,
                "filename": filename,
                "window": window_index
            })


            total_windows += 1


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            valid_recordings % 25 == 0
            or index == len(emg_files) - 1
        ):

            print(
                f"Processed: {index + 1}/{len(emg_files)} | "
                f"Valid: {valid_recordings} | "
                f"Windows: {total_windows}"
            )


    except Exception as e:

        failed_recordings += 1

        print(
            f"[FAILED] {filename}"
        )

        print(
            "         ",
            str(e)
        )


# ============================================================
# CHECK RESULTS
# ============================================================

print()
print("=" * 70)
print("EXTRACTION COMPLETE")
print("=" * 70)

print()
print("Valid recordings:", valid_recordings)
print("Failed recordings:", failed_recordings)
print("Total windows:", total_windows)


if len(all_features) == 0:

    raise RuntimeError(
        "\nNo time-frequency features were extracted."
    )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X = np.stack(
    all_features,
    axis=0
)


# ============================================================
# DATA TYPE
# ============================================================

X = X.astype(
    np.float32
)


# ============================================================
# PRINT SHAPE
# ============================================================

print()
print("=" * 70)
print("FEATURE DATASET")
print("=" * 70)

print()
print("X shape:", X.shape)

print()
print("Interpretation:")

print(
    "Samples      :", X.shape[0]
)

print(
    "Channels     :", X.shape[1]
)

print(
    "Frequency bins:", X.shape[2]
)

print(
    "Time bins    :", X.shape[3]
)


# ============================================================
# CHECK NUMERICAL STABILITY
# ============================================================

print()
print("=" * 70)
print("NUMERICAL CHECK")
print("=" * 70)

print()
print("Minimum:", np.min(X))
print("Maximum:", np.max(X))
print("Mean   :", np.mean(X))
print("Std    :", np.std(X))

print()
print(
    "NaN values:",
    np.isnan(X).sum()
)

print(
    "Inf values:",
    np.isinf(X).sum()
)


# ============================================================
# REMOVE INVALID VALUES IF ANY
# ============================================================

if (
    np.isnan(X).any()
    or np.isinf(X).any()
):

    print()
    print(
        "WARNING: Invalid values detected."
    )

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    print(
        "Invalid values replaced with 0."
    )


# ============================================================
# SAVE METADATA
# ============================================================

metadata_json = json.dumps(
    metadata
)


# ============================================================
# SAVE DATASET
# ============================================================

print()
print("=" * 70)
print("SAVING DATASET")
print("=" * 70)


np.savez_compressed(
    OUTPUT_PATH,
    X=X,
    metadata=np.array(
        metadata_json
    )
)


print()
print("Saved successfully:")

print(
    OUTPUT_PATH
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print()
print("Input files:", len(emg_files))
print("Valid recordings:", valid_recordings)
print("Failed recordings:", failed_recordings)
print("Total windows:", total_windows)

print()
print("Feature shape:", X.shape)

print()
print("Representation:")
print(
    "8-channel EMG → STFT → log-power spectrogram"
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)