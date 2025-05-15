from imports import keras
from imports import tensorflow as tf


def get_model() -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Dense(64, activation = "swish", input_shape = (2,)),
            keras.layers.Dense(32, activation = "swish"),
            keras.layers.Dense(1)
        ]

    )

    model.compile(optimizer = "adam", loss = "mse")
    return model
