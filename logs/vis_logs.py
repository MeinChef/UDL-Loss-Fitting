import matplotlib.pyplot as plt
import numpy as np


logs_lstm = np.genfromtxt(
    fname = "logs/lstm-mse.csv",
    dtype = np.float32,
    delimiter=",",
    skip_header=1,
    usecols=(1,2)
)
logs_dense = np.genfromtxt(
    fname = "logs/dense-mse.csv",
    dtype = np.float32,
    delimiter=",",
    skip_header=1,
    usecols=(1,2)
)
bad_lstm_mse = np.genfromtxt(
    fname = "logs/bad_lstm-mse.csv",
    dtype = np.float32,
    delimiter=",",
    skip_header=1,
    usecols=(1,2)
)

plt.plot(logs_lstm[:,1], label = "LSTM")
plt.plot(logs_dense[:,1], label = "Dense")
plt.plot(bad_lstm_mse[:,1], label = "Bad Init")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss of Models with Sine-Cosine loss")
plt.show()