import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


# ==========================================
# HIGH-PASS FILTER
# ==========================================

from Filter.Highpass_filter import highpass_filter
# ==========================================
# NOTCH FILTER
# ==========================================

from Filter.Notch_filter import notch_filter

# ==========================================
# LOW-PASS FILTER
# ==========================================

from Filter.Lowpass_filter import lowpass_filter

# ==========================================
# COMPLETE EMG FILTER
# ==========================================

def filter_emg(emg):


    filtered = np.zeros_like(
        emg,
        dtype=np.float64
    )

    for channel in range(emg.shape[0]):

        signal = emg[channel]

        # High-pass
        signal = highpass_filter(signal)

        # 50 Hz notch
        signal = notch_filter(signal)

        # Low-pass
        signal = lowpass_filter(signal)

        filtered[channel] = signal

    return filtered



    """
    Input:
        emg → shape (6, samples)

    Output:
        filtered EMG → shape (6, samples)
    """