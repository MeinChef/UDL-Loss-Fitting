from imports import tensorflow as tf
from imports import numpy as np
from imports import yaml
from imports import os

def load_config(path: str = "config.yml") -> dict:
    """
        Loads a .yml file as dictionary and returns it.

        :param path: A string or Path-Like object pointing to the file.
        :type path: str or Path-Like

        :return: Dictionary
        :rtype: dict
    """
    with open(path, "r") as file:
        config = yaml.safe_load(file)

    return config

def load_data(cfg: dict) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
        A function to load a csv, path specified in config dictionary.
        Returns Tuple of (Train, Test), where Train and Test are tf.data.Dataset objects.

        :param cfg: A Dictionary containing the key "data_path" (path to file), "data_name" (name of file) and "data_prep" (generated from network type).
        :type cfg: dictionary, required

        :return: Tuple of Dataloaders
        :rtype: tuple[tf.data.Dataset, tf.data.Dataset]
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
    
    # different pipelines for data
    if cfg["data_prep"] == "dense":
        data, target =  dense_data(df, cfg)
    elif cfg["data_prep"] == "lstm":
        data, target = lstm_data(df, cfg)
    else:
        raise ValueError(f"Unknown data preparation type: {cfg['data_prep']}")

    # shuffle for better distribution of training/test data
    data, target = unison_shuffled_copies(data, target)

    # turn them into dataloaders, and use the fi
    train = tf.data.Dataset.from_tensor_slices((
        data[:int(df.shape[0] * cfg["split"])], 
        target[:int(df.shape[0] * cfg["split"])]
        ))
    
    test = tf.data.Dataset.from_tensor_slices((
        data[int(df.shape[0] * cfg["split"]):], 
        target[int(df.shape[0] * cfg["split"]):]
        ))

    train = train.batch(cfg["batch"]).prefetch(tf.data.AUTOTUNE)
    test  =  test.batch(cfg["batch"]).prefetch(tf.data.AUTOTUNE)

    return train, test

def dense_data(
        df: np.ndarray, 
        cfg: dict
    ) -> tuple[np.ndarray, np.ndarray]:
    """
        Prepares data for a dense model.
        :param df: The dataframe containing the data.
        :type df: np.ndarray, required
        :param cfg: Configuration dictionary containing batch size.
        :type cfg: dict, required
        :return: Tuple of data and target arrays.
        :rtype: tuple[np.ndarray, np.ndarray]
    """
    # last datapoint gets discarded
    # since the target of the "next hour" does not exist
    data = df[:-1]
    # for the "zero hour" there is no -1st datapoint
    # and only direction as to be predicted
    target = df[1:,2]

    return data, target

def lstm_data(
        df: np.ndarray,
        cfg: dict
    ) -> tuple[np.ndarray, np.ndarray]:
    """
        Prepares data for an LSTM model.
        :param df: The dataframe containing the data.
        :type df: np.ndarray, required
        :param cfg: Configuration dictionary containing sequence length.
        :type cfg: dict, required
        :return: Tuple of data and target arrays.
        :rtype: tuple[np.ndarray, np.ndarray]
    """

    total = df.shape[0]
    data = df  # Use first two columns as data
    target = df[cfg["seq_len"]:,2]  # Use third column as target

    lstm = np.full(
        shape = (total - cfg["seq_len"], cfg["seq_len"], 3), 
        fill_value = np.nan,
        dtype = np.float32
    )

    # construct LSTM data
    for i in range(total - cfg["seq_len"]):
        # use the past cfg["seq_len"] datapoints for one sample
        lstm[i] = data[i:i+cfg["seq_len"]]
    
    return lstm, target

def unison_shuffled_copies(
        arr_0: np.ndarray, 
        arr_1: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
    """
        Shuffle two arrays in unison.
        :param arr_0: First array to shuffle.
        :type arr_0: np.ndarray, required
        :param arr_1: Second array to shuffle.
        :type arr_1: np.ndarray, required
        :return: Tuple of shuffled arrays.
        :rtype: tuple[np.ndarray, np.ndarray]
    """
    # set random seed for reproducibility
    np.random.seed(42)
    assert len(arr_0) == len(arr_1)

    p = np.random.permutation(len(arr_0))
    return arr_0[p], arr_1[p]
