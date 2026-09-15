
"""
===========================================================
model.py

1D CNN + BiLSTM Model
For MI vs Normal ECG Classification

Input Shape:
(samples, 5000, 12)

===========================================================
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam

from config import *


def build_model():

    inputs = Input(shape=(SIGNAL_LENGTH, 12))

    x = Conv1D(
        filters=32,
        kernel_size=7,
        activation="relu",
        padding="same")(inputs)

    x = BatchNormalization()(x)

    x = MaxPooling1D(pool_size=2)(x)

    x = Dropout(0.2)(x)


    x = Conv1D(
        filters=64,
        kernel_size=5,
        activation="relu",
        padding="same")(x)

    x = BatchNormalization()(x)

    x = MaxPooling1D(pool_size=2)(x)

    x = Dropout(0.2)(x)


    x = Conv1D(
        filters=128,
        kernel_size=3,
        activation="relu",
        padding="same")(x)

    x = BatchNormalization()(x)

    x = MaxPooling1D(pool_size=2)(x)

    x = Dropout(0.3)(x)


    x = Bidirectional(
        LSTM(
            64,
            return_sequences=True
        )
    )(x)


    x = Dropout(0.3)(x)


    x = GlobalAveragePooling1D()(x)


    x = Dense(
        128,
        activation="relu")(x)

    x = Dropout(0.4)(x)


    outputs = Dense(
        1,
        activation="sigmoid")(x)


    model = Model(inputs, outputs)


    model.compile(

        optimizer=Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="binary_crossentropy",

        metrics=[
            "accuracy"
        ]
    )

    return model


if __name__ == "__main__":

    model = build_model()

    model.summary()