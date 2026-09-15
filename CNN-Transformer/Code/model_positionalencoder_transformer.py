
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    Dropout,
    Dense,
    Add,
    LayerNormalization,
    GlobalAveragePooling1D,
    MultiHeadAttention
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import register_keras_serializable

from config import *


@register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)

        self.seq_len = seq_len
        self.d_model = d_model

        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]

        angle_rates = 1 / np.power(
            10000,
            (2 * (i // 2)) / np.float32(d_model)
        )

        angle_rads = pos * angle_rates

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        self.pos_encoding = tf.cast(
            angle_rads[np.newaxis, ...],
            dtype=tf.float32
        )

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "seq_len": self.seq_len,
            "d_model": self.d_model
        })
        return config


def build_model():

    inputs = Input(shape=(SIGNAL_LENGTH, 12))

    # CNN Block 1
    x = Conv1D(
        filters=64,
        kernel_size=7,
        padding="same",
        activation="relu"
    )(inputs)

    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # CNN Block 2
    x = Conv1D(
        filters=128,
        kernel_size=5,
        padding="same",
        activation="relu"
    )(x)

    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Dense(64)(x)

    x = PositionalEncoding(
        seq_len=1250,
        d_model=64
    )(x)

    # Transformer Block
    attn_output = MultiHeadAttention(
        num_heads=4,
        key_dim=64
    )(x, x)

    attn_output = Dropout(0.1)(attn_output)

    x = Add()([x, attn_output])

    x = LayerNormalization(
        epsilon=1e-6
    )(x)
	

    # Feed Forward Network
    ff = Dense(
        128,
        activation="relu"
    )(x)

    ff = Dropout(0.1)(ff)

    ff = Dense(64)(ff)

    x = Add()([x, ff])

    x = LayerNormalization(
        epsilon=1e-6
    )(x)

    # Classification Head
    x = GlobalAveragePooling1D()(x)

    x = Dense(
        64,
        activation="relu"
    )(x)

    x = Dropout(0.3)(x)

    outputs = Dense(
        1,
        activation="sigmoid"
    )(x)

    # Model
    model = Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        optimizer=Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            AUC(name="auc")
        ]
    )

    return model


if __name__ == "__main__":

    model = build_model()

    model.summary()
	


