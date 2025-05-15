from imports import keras
from imports import tensorflow as tf

def get_model() -> keras.Model:
    # add lstm? 
    model = keras.Sequential(
        [
            keras.layers.Dense(64, activation = "swish", input_shape = (2,)),
            keras.layers.Dense(32, activation = "swish"),
            keras.layers.Dense(2)
        ]
    )

    # TODO: other loss function / custom loss function
    model.compile(optimizer = "adam", loss = "mse")
    return model

def get_lstm_model(seq_len=5, input_dim=2) -> keras.Model:
    inputs = keras.layers.Input(shape=(seq_len, input_dim))
    lstm_out = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(inputs)

    # Attention mechanism
    attention_scores = keras.layers.Dense(1, activation='tanh')(lstm_out)
    attention_weights = keras.layers.Softmax(axis=1)(attention_scores)
    context_vector = keras.layers.Dot(axes=1)([attention_weights, lstm_out])
    context_vector = keras.layers.Flatten()(context_vector)

    dense_out = keras.layers.Dense(32, activation='swish')(context_vector)
    output = keras.layers.Dense(1)(dense_out)

    model = keras.models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

def get_circ_model(seq_len=5, input_dim=2):
    inputs = keras.Input(shape=(seq_len, input_dim))
    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(inputs)
    
    attention = keras.layers.Dense(1, activation='tanh')(x)
    attention = keras.layers.Softmax(axis=1)(attention)
    context = keras.layers.Dot(axes=1)([attention, x])
    context = keras.layers.Flatten()(context)

    x = keras.layers.Dense(32, activation="swish")(context)
    output = keras.layers.Dense(2)(x)  # Predict sin & cos

    model = keras.Model(inputs, output)

    model.compile(optimizer='adam', loss=angle_cosine_loss)

    return model

def angle_cosine_loss(y_true, y_pred):
        y_true = tf.math.l2_normalize(y_true, axis=1)
        y_pred = tf.math.l2_normalize(y_pred, axis=1)
        return 1 - tf.reduce_mean(tf.reduce_sum(y_true * y_pred, axis=1))