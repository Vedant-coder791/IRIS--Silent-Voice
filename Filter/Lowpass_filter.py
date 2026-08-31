from scipy.signal import butter, filtfilt


def lowpass_filter(
    signal,
    cutoff=250,
    fs=600,
    order=4
):

    b, a = butter(
        order,
        cutoff,
        btype="lowpass",
        fs=fs
    )

    return filtfilt(
        b,
        a,
        signal,
        axis=-1
    )