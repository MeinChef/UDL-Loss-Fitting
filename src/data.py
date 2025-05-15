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
    breakpoint()

    # turn them into dataloaders, and use col 0+1 
    train = tf.data.Dataset.from_tensor_slices((train[:,:2], train[:,2]))
    test = tf.data.Dataset.from_tensor_slices((test[:,:2], test[:,2]))

    train = train.batch(cfg["batch"]).prefetch(tf.data.AUTOTUNE)
    test  =  test.batch(cfg["batch"]).prefetch(tf.data.AUTOTUNE)

    return train, test


