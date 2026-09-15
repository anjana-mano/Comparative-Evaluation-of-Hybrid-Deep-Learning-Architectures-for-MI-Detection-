
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    Dropout,
    Bidirectional,
    GRU,
    Dense
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from config import *


def build_model():

    # Input
    inputs = Input(shape=(SIGNAL_LENGTH, 12))

    # CNN Block 1
    x = Conv1D(
        filters=64,
        kernel_size=7,
        padding='same',
        activation='relu'
    )(inputs)

    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # CNN Block 2
    x = Conv1D(
        filters=128,
        kernel_size=5,
        padding='same',
        activation='relu'
    )(x)

    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # BiGRU
    x = Bidirectional(
        GRU(
            units=64,
            return_sequences=False
        )
    )(x)

    x = Dropout(0.3)(x)

    # Output
    outputs = Dense(1, activation='sigmoid')(x)

    # Model
    model = Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            AUC(name='val_auc')
        ]
    )

    return model


if __name__ == "__main__":

    model = build_model()

    model.summary()