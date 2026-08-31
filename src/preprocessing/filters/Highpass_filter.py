from scipy.signal import butter, filtfilt

def highpass_filter(signal, cutoff=20, fs=600, order=4):

    b, a = butter(order, cutoff, btype='highpass', fs=fs)

    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal


