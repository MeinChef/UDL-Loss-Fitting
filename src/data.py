from imports import tensorflow as tf
from imports import numpy as np
from imports import yaml
from imports import os
from imports import math

def load_config(path: str = "config.yml") -> tuple[dict, dict]:
    with open(path, "r") as file:
        config = yaml.safe_load(file)

    return config

def load_data(cfg: dict) -> tuple[tuple, tuple]:
    """
        A function to load a csv, path specified in config dictionary.
        Returns Tuple of (Train, Test), where Train and Test (Data, Target)

        :param cfg: A Dictionary containing the key "data_path"
        :type cfg: dictionary, required

        :return: Tuple of Tuple of Dataloaders
        :rtype: tuple[tuple[tf.Dataloader, tf.Dataloader], tuple[tf.Dataloader, tf.Dataloader]]
    """

    # load csv with numpy
    df = np.genfromtxt(
        os.path.join(cfg["data_path"], cfg["data_name"]), 
        dtype = np.float32,
        delimiter = ";",
        skip_header = 3,
        # only use columns pressure, speed and direction
        usecols = (2,6,4)
        )
    # remove any inf or nan values
    df = df[~np.ma.fix_invalid(df).mask.any(axis=1)]
    
    # shuffle for better distribution of training/test data
    np.random.seed(42)
    np.random.shuffle(df)
    # split in train/test halves
    train = df[:int(df.shape[0]*0.8)]
    test = df[int(df.shape[0]*0.8):]

    # turn them into dataloaders, and use col 0+1 for data, col 2 for target
    train = tf.data.Dataset.from_tensor_slices((train[:,:2], train[:,2]))
    test = tf.data.Dataset.from_tensor_slices((test[:,:2], test[:,2]))

    train = train.batch(cfg["batch"]).prefetch(tf.data.AUTOTUNE)
    test  =  test.batch(cfg["batch"]).prefetch(tf.data.AUTOTUNE)

    return train, test


def load_lstm_data(cfg: dict) -> tuple[tuple, tuple]:
    # load csv with numpy
    df = np.genfromtxt(
        os.path.join(cfg["data_path"], cfg["data_name"]), 
        dtype = np.float32,
        delimiter = ";",
        skip_header = 3,
        # only use columns pressure, speed and direction
        usecols = (2,6,4)
        )
    # remove any inf or nan values
    df = df[~np.ma.fix_invalid(df).mask.any(axis=1)] # Shape (2159, 3)

    # shuffle for better distribution of training/test data
    np.random.seed(42)
    np.random.shuffle(df)

    data = df[:,:2] # Shape (2159, 2)
    label = df[:,2] # Shape (2159,)

    # Prepare sequences
    X_seq = []
    y_seq = []

    for i in range(len(label) - cfg["seq_len"]):
        X_seq.append(data[i:i + cfg["seq_len"]])       # Shape (5, 2)
        y_seq.append(label[i + cfg["seq_len"]])        # Predict the value after the 5 inputs

    data = np.array(X_seq)  # Shape (2159, 5, 2)
    label = np.array(y_seq)  # Shape (2159,)

    # split in train/test halves
    train_data  =  data[:int(df.shape[0]*0.8)]
    train_label = label[:int(df.shape[0]*0.8)]
    test_data  =  data[int(df.shape[0]*0.8):]
    test_label = label[int(df.shape[0]*0.8):]

    # turn them into dataloaders, and use col 0+1 
    train = tf.data.Dataset.from_tensor_slices((train_data, train_label))
    test = tf.data.Dataset.from_tensor_slices((test_data, test_label))

    train = train.batch(cfg["batch"]).prefetch(tf.data.AUTOTUNE)
    test  =  test.batch(cfg["batch"]).prefetch(tf.data.AUTOTUNE)

    return train, test

def load_circ_data(cfg: dict) -> tuple[tuple, tuple]:
    # load csv with numpy
    df = np.genfromtxt(
        os.path.join(cfg["data_path"], cfg["data_name"]), 
        dtype = np.float32,
        delimiter = ";",
        skip_header = 3,
        # only use columns pressure, speed and direction
        usecols = (2,6,4)
        )
    # remove any inf or nan values
    df = df[~np.ma.fix_invalid(df).mask.any(axis=1)] # Shape (2159, 3)

    # shuffle for better distribution of training/test data
    np.random.seed(42)
    np.random.shuffle(df)

    data = df[:,:2] # Shape (2159, 2)
    label = df[:,2] # Shape (2159,)

    # Prepare sequences
    X_seq = []
    y_seq = []

    for i in range(len(label) - cfg["seq_len"]):
        X_seq.append(data[i:i + cfg["seq_len"]])       # Shape (5, 2)
        y_seq.append(label[i + cfg["seq_len"]])        # Predict the value after the 5 inputs

    data = np.array(X_seq)  # Shape (2159, 5, 2)
    label = np.array(y_seq)  # Shape (2159,)

    # split in train/test halves
    train_data  =  data[:int(df.shape[0]*0.8)]
    train_label = label[:int(df.shape[0]*0.8)]
    test_data  =  data[int(df.shape[0]*0.8):]
    test_label = label[int(df.shape[0]*0.8):]

    # turn them into dataloaders, and use col 0+1 
    train = tf.data.Dataset.from_tensor_slices((train_data, train_label))
    test = tf.data.Dataset.from_tensor_slices((test_data, test_label))


    def angle_to_vector(x, y_deg):
        # y_rad = tf.math.radians(y_deg)  # Convert degrees to radians
        y_rad = y_deg * math.pi / 180
        y_sin = tf.math.sin(y_rad)
        y_cos = tf.math.cos(y_rad)
        y_vec = tf.stack([y_sin, y_cos], axis=-1)  # shape (2,)
        return x, y_vec

    train = train.map(angle_to_vector)
    test  =  test.map(angle_to_vector)

    train = train.batch(cfg["batch"]).prefetch(tf.data.AUTOTUNE)
    test  =  test.batch(cfg["batch"]).prefetch(tf.data.AUTOTUNE)

    return train, test