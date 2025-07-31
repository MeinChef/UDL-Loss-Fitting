from loss import VonMises, CustomMSE
from imports import keras


def get_model(
        cfg: dict
    ) -> keras.Model:

    if cfg["loss"] == "vm":
        loss = VonMises()
        cfg["out"] = 1
    elif cfg["loss"] == "mse":
        loss = CustomMSE(axis = -1)
        cfg["out"] = 2
    else:
        raise ValueError(f"Unknown loss function: {cfg['loss']}")

    if cfg["model"] == "dense":
        model = get_dense_model(
            num_out = cfg["out"]
        )
    elif cfg["model"] == "lstm":
        model = get_lstm_model(
            seq_len = cfg["seq_len"],
            num_out = cfg["out"]
        )
    else: 
        raise ValueError(f"Unknown model type: {cfg['model']}")
    
    model.compile(optimizer = 'adam', loss = loss)
    return model

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
        num_out: int = 2
    ) -> keras.Model:
    inputs = keras.layers.Input(shape = (seq_len, 3))
    lstm_out = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences = True))(inputs)
    context_vector = keras.layers.Flatten()(lstm_out)

    dense_out = keras.layers.Dense(32, activation = 'swish')(context_vector)
    output = keras.layers.Dense(num_out)(dense_out)

    model = keras.models.Model(inputs = inputs, outputs = output)
    return model