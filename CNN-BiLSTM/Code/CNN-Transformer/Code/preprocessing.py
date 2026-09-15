"""
===========================================================
preprocessing.py

ECG Signal Preprocessing

1. Band-pass Filter
2. Notch Filter
3. Wavelet Denoising
4. Z-score Normalization

Works for 12-lead ECG signals

===========================================================
"""

import numpy as np
import pywt

from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import iirnotch
from scipy.signal import medfilt
from scipy.signal import savgol_filter

from config import *


# ==========================================================
# Butterworth Band-pass Filter
# ==========================================================

def butter_bandpass(lowcut, highcut, fs, order=4):

    nyquist = 0.5 * fs

    low = lowcut / nyquist

    high = highcut / nyquist

    b, a = butter(order,
                  [low, high],
                  btype='band')

    return b, a


def bandpass_filter(signal):

    b, a = butter_bandpass(
        LOWCUT,
        HIGHCUT,
        SAMPLING_RATE,
        order=4
    )

    filtered = filtfilt(b, a, signal)

    return filtered


# ==========================================================
# Notch Filter
# ==========================================================

def notch_filter(signal):

    quality_factor = 30

    b, a = iirnotch(
        POWERLINE_FREQ,
        quality_factor,
        SAMPLING_RATE
    )

    filtered = filtfilt(b, a, signal)

    return filtered


# ==========================================================
# Wavelet Denoising
# ==========================================================

def wavelet_denoise(signal):

    coeffs = pywt.wavedec(
        signal,
        WAVELET,
        level=WAVELET_LEVEL
    )

    sigma = np.median(np.abs(coeffs[-1])) / 0.6745

    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    coeffs = [
        pywt.threshold(
            c,
            threshold,
            mode="soft"
        )
        for c in coeffs
    ]

    reconstructed = pywt.waverec(
        coeffs,
        WAVELET
    )

    reconstructed = reconstructed[:len(signal)]

    return reconstructed



# ==========================================================
# Median Filter
# ==========================================================
def median_filter(signal,kernel_size=5):
    return medfilt(signal,kernel_size=kernel_size)

# ==========================================================
# Savitzky-Golay Filter
# ==========================================================
def savitzky_golay_filter(signal,window_length=11,polyorder=3):
    return savgol_filter(signal,window_length=window_length,polyorder=polyorder)





# ==========================================================
# Z-score Normalization
# ==========================================================

def normalize(signal):

    mean = np.mean(signal)

    std = np.std(signal)

    if std == 0:

        std = 1

    return (signal - mean) / std




# ==========================================================
# Preprocess One Lead
# ==========================================================

def preprocess_lead(signal):

    signal = bandpass_filter(signal)

    signal = notch_filter(signal)

    signal = wavelet_denoise(signal)

    signal = normalize(signal)

    return signal

# =========================================================
# preprocess one lead using bandpass and notch
# ==========================================================

def preprocess_bandpass_notch(signal):
    signal=bandpass_filter(signal)
    signal=notch_filter(signal)
    signal=normalize(signal)
    return signal

# ==========================================================
# preprocess one lead using bandpass and wavelet
# ==========================================================
def preprocess_bandpass_wavelet(signal):
    signal=bandpass_filter(signal)
    signal=wavelet_denoise(signal)
    signal=normalize(signal)
    return signal

# ==========================================================
# preprocess one lead using bandpass and median
# ==========================================================


def preprocess_bandpass_median(signal):
    signal=bandpass_filter(signal)
    signal=median_filter(signal)
    signal=normalize(signal)
    return signal


# ==========================================================
# preprocess one lead using bandpass and savitzky 
# ==========================================================
def preprocess_bandpass_savgol(signal):
    signal=bandpass_filter(signal)
    signal=savitzky_golay_filter(signal)
    signal=normalize(signal)
    return signal


#---------------------------------------------------------
# preprocess using bandpass
#--------------------------------------------------------

def preprocess_bandpass(signal):
    signal=bandpass_filter(signal)
    signal=normalize(signal)
    return signal

#-----------------------------------------
#preprocess using notch
#----------------------------------------

def preprocess_notch(signal):

    signal=notch_filter(signal)
    signal=normalize(signal)
    return signal

# ==========================================================
# Preprocess 12-Lead ECG
# ==========================================================

def preprocess_ecg(ecg):

    processed = np.zeros_like(ecg)

    leads = ecg.shape[1]

    for lead in range(leads):

        processed[:, lead] = preprocess_bandpass_notch(
           ecg[:, lead]
         #processed[:, lead] = preprocess_lead(
          # ecg[:, lead]
         
        )

    return processed


# ==========================================================
# Preprocess Entire Dataset
# ==========================================================

def preprocess_dataset(X):

    

    total = len(X)

    print("\nPreprocessing ECG Signals...\n")

    for i in range(total):

        X[i]=preprocess_ecg(X[i])

        if (i + 1) % 100 == 0:

            print(f"{i+1}/{total} completed")

    return X


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    from dataset_loader import load_dataset

    X, y = load_dataset()

    X = preprocess_dataset(X)

    print()

    print("Processed Shape :", X.shape)

    print("Done.")
