import numpy as np

def create_windows(emg, window_size=120, step_size=60):
    """
    Create overlapping windows from EMG data.

    Parameters
    ----------
    emg : ndarray
        Shape = (channels, samples)

    window_size : int
        Number of samples per window

    step_size : int
        Number of samples between consecutive windows

    Returns
    -------
    ndarray
        Shape = (num_windows, channels, window_size)
    """

    channels, samples = emg.shape

    windows = []

    for start in range(0, samples - window_size + 1, step_size):

        end = start + window_size

        window = emg[:, start:end]

        windows.append(window)

    return np.array(windows)



