import numpy as np
import librosa
from scipy.signal import welch


# ============================================================
# SETTINGS
# ============================================================

FS = 600

N_FFT = 64
HOP_LENGTH = 32
N_MFCC = 13
N_MELS = 20

FMIN = 20
FMAX = 250


# ============================================================
# MFCC EXTRACTION
# ============================================================

def extract_mfcc_test(signal, fs=FS):

    signal = np.asarray(signal, dtype=np.float32)

    # Prevent numerical problems
    signal = signal - np.mean(signal)

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=fs,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mfcc=N_MFCC,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX
    )

    return mfcc.astype(np.float32)


# ============================================================
# TIME-DOMAIN FEATURES
# ============================================================

def extract_time_features(signal):

    signal = np.asarray(signal, dtype=np.float32)

    # Mean Absolute Value
    mav = np.mean(np.abs(signal))

    # Root Mean Square
    rms = np.sqrt(np.mean(signal ** 2))

    # Waveform Length
    waveform_length = np.sum(
        np.abs(np.diff(signal))
    )

    # Zero Crossing Rate
    zero_crossings = np.sum(
        np.diff(np.signbit(signal))
    )

    zero_crossing_rate = (
        zero_crossings / len(signal)
    )

    return np.array([
        mav,
        rms,
        waveform_length,
        zero_crossing_rate
    ], dtype=np.float32)


# ============================================================
# FREQUENCY-DOMAIN FEATURES
# ============================================================

def extract_frequency_features(signal, fs=FS):

    signal = np.asarray(signal, dtype=np.float32)

    frequencies, power = welch(
        signal,
        fs=fs,
        nperseg=min(64, len(signal))
    )

    total_power = np.sum(power)

    if total_power == 0:

        spectral_centroid = 0.0
        median_frequency = 0.0

    else:

        # Spectral centroid
        spectral_centroid = (
            np.sum(frequencies * power)
            / total_power
        )

        # Median frequency
        cumulative_power = np.cumsum(power)

        median_index = np.searchsorted(
            cumulative_power,
            total_power / 2
        )

        median_index = min(
            median_index,
            len(frequencies) - 1
        )

        median_frequency = frequencies[
            median_index
        ]

    return np.array([
        total_power,
        spectral_centroid,
        median_frequency
    ], dtype=np.float32)


# ============================================================
# ALL FEATURES FROM ONE CHANNEL
# ============================================================

def extract_channel_features(signal, fs=FS):

    # -------------------------
    # MFCC
    # -------------------------

    mfcc = extract_mfcc_test(
        signal,
        fs
    )

    # Mean of each MFCC coefficient
    mfcc_mean = np.mean(
        mfcc,
        axis=1
    )

    # Standard deviation of each MFCC coefficient
    mfcc_std = np.std(
        mfcc,
        axis=1
    )

    # 13 + 13 = 26 MFCC features
    mfcc_features = np.concatenate([
        mfcc_mean,
        mfcc_std
    ])

    # -------------------------
    # Time-domain
    # -------------------------

    time_features = extract_time_features(
        signal
    )

    # 4 features
    # MAV
    # RMS
    # Waveform length
    # Zero crossing rate

    # -------------------------
    # Frequency-domain
    # -------------------------

    frequency_features = extract_frequency_features(
        signal,
        fs
    )

    # 3 features
    # Total power
    # Spectral centroid
    # Median frequency

    # -------------------------
    # Combine everything
    # -------------------------

    return np.concatenate([
        mfcc_features,
        time_features,
        frequency_features
    ]).astype(np.float32)


# ============================================================
# ALL FEATURES FROM ONE EMG WINDOW
# ============================================================

def extract_window_features(window, fs=FS):

    window = np.asarray(
        window,
        dtype=np.float32
    )

    channel_features = []

    # window shape should be:
    #
    # (channels, samples)
    #
    # Example:
    # (6, 120)

    for channel in window:

        features = extract_channel_features(
            channel,
            fs
        )

        channel_features.append(
            features
        )

    # Flatten all channels
    #
    # Example:
    # 6 channels × 33 features
    # = 198 features

    return np.concatenate(
        channel_features
    ).astype(np.float32)


# ============================================================
# ALL FEATURES FROM ALL WINDOWS
# ============================================================

def extract_all_features(windows, fs=FS):

    all_features = []

    for i, window in enumerate(windows):

        features = extract_window_features(
            window,
            fs
        )

        all_features.append(
            features
        )

    return np.asarray(
        all_features,
        dtype=np.float32
    )