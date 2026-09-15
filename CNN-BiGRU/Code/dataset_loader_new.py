"""
===========================================================
dataset_loader.py

Generator-based Dataset Loader
Loads ECG batch-by-batch to avoid RAM overflow.

===========================================================
"""

import os
import math
import wfdb
import numpy as np

from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split

from config import *
from preprocessing import preprocess_ecg


# ===========================================================
# Find all ECG records
# ===========================================================

def find_records(folder):

    records = []

    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.endswith(".hea"):

                record = os.path.splitext(file)[0]

                records.append(
                    os.path.join(root, record)
                )

    return sorted(records)


# ===========================================================
# Read ECG Record
# ===========================================================

def load_record(record_path):

    try:

        record = wfdb.rdrecord(record_path)

        return record.p_signal

    except Exception as e:

        print(f"Error reading {record_path}")

        print(e)

        return None


# ===========================================================
# Fix Signal Length
# ===========================================================

def fix_length(signal, target_length=SIGNAL_LENGTH):

    current = signal.shape[0]

    if current > target_length:

        signal = signal[:target_length]

    elif current < target_length:

        pad = target_length - current

        signal = np.pad(
            signal,
            ((0, pad), (0, 0)),
            mode="constant"
        )

    return signal



# ===========================================================
# ECG Data Generator
# ===========================================================

class ECGGenerator(Sequence):

    def __init__(
        self,
        records,
        labels,
        batch_size=BATCH_SIZE,
        shuffle=True
    ):

        self.records = records
        self.labels = np.array(labels)

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indexes = np.arange(len(self.records))

        self.on_epoch_end()

    def __len__(self):

        return math.ceil(
            len(self.records) / self.batch_size
        )

    def __getitem__(self, index):

        batch_indexes = self.indexes[
            index*self.batch_size :
            (index+1)*self.batch_size
        ]

        X = []
        y = []

        for i in batch_indexes:

            signal = load_record(
                self.records[i]
            )

            if signal is None:
                print(f"Skipping missing file:{self.records[i]}")
                continue
                

            signal = fix_length(signal)
            if USER_PREPROCESSING:
                
               signal = preprocess_ecg(signal)
            if np.isnan(signal).any() or np.isinf(signal).any():
                print(f"skipping invalid signal:{self.records[i]}")
                continue
            #X.append(signal)
            X.append(signal.astype(np.float32))
            

            y.append(self.labels[i])

        X = np.array(
            X,
            dtype=np.float32
        )

        y = np.array(
            y,
            dtype=np.float32
        )
        

        return X, y

    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(
                self.indexes
            )

# ===========================================================
# Prepare Train / Validation / Test Generators
# ===========================================================

def prepare_generators():

    mi_records = find_records(MI_FOLDER)
    normal_records = find_records(NORMAL_FOLDER)
    mi_records=mi_records[:5000]
    normal_records=normal_records[:5000]
    # save selected mi records
    with open("mi_records.txt","w") as f:
        for record in mi_records:
            f.write(record+"\n")
     # save selected normal records
    with open("normal_records.txt","w") as f:
        for record in normal_records:
            f.write(record+"\n")
    print("successfully record lists saved")
    
    
    print(f"Using {len(mi_records)}MI records")
    print(f"Using {len(normal_records)} Normal records")

    records = mi_records + normal_records

    labels = (
        [1] * len(mi_records)
        +
        [0] * len(normal_records)
    )

    print("-" * 60)
    print("MI Records :", len(mi_records))
    print("Normal Records :", len(normal_records))
    print("Total Records :", len(records))
    print("-" * 60)

    train_records, test_records, train_labels, test_labels = train_test_split(
        records,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    train_records, val_records, train_labels, val_labels = train_test_split(
        train_records,
        train_labels,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_labels
    )

    train_generator = ECGGenerator(
        train_records,
        train_labels,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_generator = ECGGenerator(
        val_records,
        val_labels,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_generator = ECGGenerator(
        test_records,
        test_labels,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return (
        train_generator,
        val_generator,
        test_generator
    )


# ===========================================================
# Test
# ===========================================================

if __name__ == "__main__":

    train_gen, val_gen, test_gen = prepare_generators()

    print("Train Batches :", len(train_gen))
    print("Validation Batches :", len(val_gen))
    print("Test Batches :", len(test_gen))

    X, y = train_gen[0]

    print("One Batch Shape :", X.shape)
    print("Labels Shape :", y.shape)

