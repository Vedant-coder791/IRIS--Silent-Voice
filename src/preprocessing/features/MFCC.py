import numpy as np
import librosa


# ==========================================
# SETTINGS
# ==========================================

FS = 600

N_FFT = 64
HOP_LENGTH = 32
N_MFCC = 13
N_MELS = 20

FMIN = 20
FMAX = 280


# ==========================================
# EXTRACT MFCC FEATURES
# ==========================================

def extract_mfcc(emg):
    
    channel_features = []

    for channel in range(emg.shape[0]):

        signal = emg[channel].astype(
            np.float32
        )

        # Make sure signal is long enough
        if len(signal) < N_FFT:

            signal = np.pad(
                signal,
                (0, N_FFT - len(signal))
            )

        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=FS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mfcc=N_MFCC,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX
        )

        channel_features.append(mfcc)

    return np.array(channel_features)

