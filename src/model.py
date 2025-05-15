from imports import keras
from imports import tensorflow as tf



inputs = keras.Input(shape = 2, dtype = tf.float32)
model = keras.Sequential(
    [
        keras.layers.Dense(64, activation = "lelu", input_shape = inputs),
        keras.layers.Dense(32, activation = "lelu"),
        keras.layers.Dense(1)
    ]

)