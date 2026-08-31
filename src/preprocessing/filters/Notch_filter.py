import numpy as np
from scipy.signal import iirnotch, filtfilt

def notch_filter(signal, notch_freq=50, fs=600, quality_factor=30):
    signal = np.asarray(signal)

    b, a = iirnotch(notch_freq, quality_factor, fs)

    if signal.ndim == 1:
        return filtfilt(b, a, signal)

    elif signal.ndim == 2:
        return filtfilt(b, a, signal, axis=1)

    else:
        raise ValueError("Signal must be 1D or 2D")