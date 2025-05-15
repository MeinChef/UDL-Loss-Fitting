from imports import pandas as pd
from imports import tensorflow as tf
from imports import numpy as np
from imports import keras
from imports import yaml
from imports import os

def load_config(path: str = "config.yml") -> tuple[dict, dict]:
    with open(path, "r") as file:
        config = yaml.safe_load(file)

    return config

def load_data(cfg: dict) -> tuple[tuple, tuple]:
    """
        A function to load a csv, path specified in config dictionary.
        Returns Tuple of (Train, Test), where Train and Test (Data, Target)

        :param cfg: A Dictionary containing the key "data_pth"
        :type cfg: dictionary, required

        :return: Tuple of Tuple of Dataloaders
        :rtype: tuple[tuple[tf.Dataloader, tf.Dataloader], tuple[tf.Dataloader, tf.Dataloader]]
    """

    # load csv with numpy
    df = np.genfromtxt(
        os.path.join(cfg["data_pth"], cfg["data_name"]), 
        dtype = np.float32,
        delimiter = ";",
        skip_header = 3,
        # only use columns pressure, speed and direction
        usecols = (2,6,4)
        )
    
    # shuffle for better distribution of training/test data
    np.random.seed(42)
    np.random.shuffle(df)
    train = df[:int(df.shape[0]*0.8)]
    test = df[int(df.shape[0]*0.8):]

    # turn them into dataloaders
    train = tf.data.Dataset.from_tensors((train[:,:2], train[:,2]))
    test = tf.data.Dataset.from_tensors((test[:,:2], test[:,2]))

    train = train.batch(cfg["batch"]).prefetch(tf.data.AUTOTUNE)
    test  =  test.batch(cfg["batch"]).prefetch(tf.data.AUTOTUNE)

    return train, test


