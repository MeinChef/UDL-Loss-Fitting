from imports import keras

def get_dense_model() -> keras.Model:
    inputs = keras.layers.Input(shape = (3,))
    x = keras.layers.Dense(64, activation = "tanh")(inputs)
    x = keras.layers.Dense(32, activation = "tanh")(x)
    output = keras.layers.Dense(1)(x)
    model = keras.models.Model(inputs = inputs, outputs = output)
    return model

def get_lstm_model(
        seq_len = 5, 
        input_dim = 3
    ) -> keras.Model:
    inputs = keras.layers.Input(shape = (seq_len, input_dim))
    lstm_out = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences = True))(inputs)

    # Attention mechanism
    # attention_scores = keras.layers.Dense(1, activation='tanh')(lstm_out)
    # attention_weights = keras.layers.Softmax(axis = 1)(attention_scores)
    # context_vector = keras.layers.Dot(axes = 1)([attention_weights, lstm_out])
    context_vector = keras.layers.Flatten()(lstm_out)

    dense_out = keras.layers.Dense(32, activation = 'swish')(context_vector)
    output = keras.layers.Dense(1)(dense_out)

    model = keras.models.Model(inputs = inputs, outputs = output)
    return model

def get_circ_model(seq_len = 5, input_dim = 3):
    inputs = keras.Input(shape = (seq_len, input_dim))
    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences = True))(inputs)
    
    # attention = keras.layers.Dense(1, activation = 'tanh')(x)
    # attention = keras.layers.Softmax(axis = 1)(attention)
    # context = keras.layers.Dot(axes = 1)([attention, x])
    context = keras.layers.Flatten()(x)

    x = keras.layers.Dense(32, activation = "swish")(context)
    output = keras.layers.Dense(2)(x)  # Predict sin & cos

    model = keras.Model(inputs, output)
    return model