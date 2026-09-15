"""
===========================================================
config.py
Configuration File
===========================================================
"""

import os

# ===========================================================
# DATASET PATH
# ===========================================================

# Change this to your dataset path
DATASET_PATH = r"/libx32/home/sdg/Downloads/sreedevi/MI_NORMAL"

MI_FOLDER = os.path.join(DATASET_PATH, "MI")
NORMAL_FOLDER = os.path.join(DATASET_PATH, "Normal")

# ===========================================================
# ECG PARAMETERS
# ===========================================================

SAMPLING_RATE = 500          # Hz
SIGNAL_LENGTH = 5000         # 10 seconds

# ===========================================================
# FILTER PARAMETERS
# ===========================================================

LOWCUT = 0.5               # Hz
HIGHCUT = 40                 # Hz
POWERLINE_FREQ = 50         # India

# ===========================================================
# WAVELET PARAMETERS
# ===========================================================

WAVELET = "db4"
WAVELET_LEVEL = 4

# ===========================================================
# TRAINING PARAMETERS
# ===========================================================

BATCH_SIZE = 32
EPOCHS = 100

LEARNING_RATE = 0.0005

USER_PREPROCESSING=True

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

RANDOM_STATE = 42

# ===========================================================
# MODEL SAVE
# ===========================================================

MODEL_NAME = "MI_Classifier.weights.h5"

# ===========================================================
# CLASS LABELS
# ===========================================================

CLASS_NAMES = {
    0: "Normal",
    1: "MI"
}
