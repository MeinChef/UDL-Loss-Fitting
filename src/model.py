from imports import keras

def get_dense_model(
        num_out: int = 1
    ) -> keras.Model:
    
    inputs = keras.layers.Input(shape = (3,))
    x = keras.layers.Dense(64, activation = "tanh")(inputs)
    x = keras.layers.Dense(32, activation = "tanh")(x)
    output = keras.layers.Dense(num_out)(x)
    model = keras.models.Model(inputs = inputs, outputs = output)
    return model

def get_lstm_model(
        seq_len:int = 5, 
        input_dim:int = 3,
        num_out: int = 2
    ) -> keras.Model:
    inputs = keras.layers.Input(shape = (seq_len, input_dim))
    lstm_out = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences = True))(inputs)
    context_vector = keras.layers.Flatten()(lstm_out)

    dense_out = keras.layers.Dense(32, activation = 'swish')(context_vector)
    output = keras.layers.Dense(num_out)(dense_out)

    model = keras.models.Model(inputs = inputs, outputs = output)
    return model