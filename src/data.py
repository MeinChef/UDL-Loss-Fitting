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

class DataLoader(object):
    def __init__(
            self,
            cfg: dict
        ) -> None:
        
        self.cfg = cfg

    def load_data(
            self
        ) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """
            A function to load a csv, path specified in config dictionary.
            Returns Tuple of (Train, Test), where Train and Test are tf.data.Dataset objects.

            :return: Tuple of Dataloaders: Training Data and Test Data
            :rtype: tuple[tf.data.Dataset, tf.data.Dataset]
        """

        # load csv with numpy
        df = np.genfromtxt(
            os.path.join(self.cfg["data_path"], self.cfg["data_name"]), 
            dtype = np.float32,
            delimiter = ";",
            skip_header = 3,
            # only use columns pressure, speed and direction
            usecols = (2,6,4)
            )
        
        # remove any inf or nan values
        df = df[~np.ma.fix_invalid(df).mask.any(axis=1)]
        # since the distributions are defined from -pi to pi, we need the direction in rad
        df[:,2] = np.deg2rad(df[:,2])
        
        # different pipelines for data
        if self.cfg["data_prep"] == "dense":
            data, target = self.dense_data_(df)
        elif self.cfg["data_prep"] == "lstm":
            data, target = self.lstm_data_(df)
        elif self.cfg["data_prep"] == "circ":
            data, target = self.circ_data_(df)
        else:
            raise ValueError(f"Unknown data preparation type: {self.cfg['data_prep']}")

        # shuffle for better distribution of training/test data
        data, target = unison_shuffled_copies(data, target)

        # save in class
        self.data = data
        self.target = target

        # turn them into dataloaders, and use the fi
        train = tf.data.Dataset.from_tensor_slices((
            data[:int(df.shape[0] * self.cfg["split"])], 
            target[:int(df.shape[0] * self.cfg["split"])]
            ))
        
        test = tf.data.Dataset.from_tensor_slices((
            data[int(df.shape[0] * self.cfg["split"]):], 
            target[int(df.shape[0] * self.cfg["split"]):]
            ))

        train = train.batch(self.cfg["batch"]).prefetch(tf.data.AUTOTUNE)
        test  =  test.batch(self.cfg["batch"]).prefetch(tf.data.AUTOTUNE)

        return train, test

    def dense_data_(
            self,
            df: np.ndarray, 
        ) -> tuple[np.ndarray, np.ndarray]:
        """
            Prepares data for a dense model.

            :param df: The dataframe containing the data.
            :type df: np.ndarray, required
            :return: Tuple of data and target arrays.
            :rtype: tuple[np.ndarray, np.ndarray]
        """
        # last datapoint gets discarded
        # since the target of the "next hour" does not exist
        data = df[:-1]
        # for the "zero hour" there is no -1st datapoint
        # and select the columns according to comment on line 39
        target = df[1:,2]
        return data, target

    def lstm_data_(
            self,
            df: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
        """
            Prepares data for an LSTM model.

            :param df: The dataframe containing the data.
            :type df: np.ndarray, required
            :return: Tuple of data and target arrays.
            :rtype: tuple[np.ndarray, np.ndarray]
        """

        total = df.shape[0]
        data = df

        # target needs to start after the first sequence has ended
        # and select the columns according to comment on line 39
        target = df[self.cfg["seq_len"]:,2]

        lstm = np.full(
            shape = (total - self.cfg["seq_len"], self.cfg["seq_len"], 3), 
            fill_value = np.nan,
            dtype = np.float32
        )

        # construct LSTM data
        for i in range(total - self.cfg["seq_len"]):
            # use the past self.cfg["seq_len"] datapoints for one sample
            lstm[i] = data[i:i+self.cfg["seq_len"]]
        
        return lstm, target

    def circ_data_(
            self,
            df: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:

        """
            Prepares data for an LSTM model that predicts sine and cosine embedding of an angle.

            :param df: The dataframe containing the data.
            :type df: np.ndarray, required
            :return: Tuple of data and target arrays.
            :rtype: tuple[np.ndarray, np.ndarray]
        """
        total = df.shape[0]
        data = df
            
        # initialize data and target arrays
        lstm = np.full(
                shape = (total - self.cfg["seq_len"], self.cfg["seq_len"], 3), 
                fill_value = np.nan,
                dtype = np.float32
            )
        
        target = np.empty(
            shape = (total - self.cfg["seq_len"], 2),
            dtype = np.float32
        )

        # construct LSTM data
        for i in range(total - self.cfg["seq_len"]):
            # use the past self.cfg["seq_len"] datapoints for one sample
            lstm[i] = data[i:i + self.cfg["seq_len"]]

            # target is sine and cosine of the angle to be predicted
            target[i,:] = np.array([
                np.sin(data[i + self.cfg["seq_len"],2]),
                np.cos(data[i + self.cfg["seq_len"],2])
            ])
        
        return lstm, target
    
    def get_data_numpy(
            self
        ) -> tuple[np.ndarray, np.ndarray]:
        
        if not self.data or not self.target:
            _ = self.load_data()
        
        return self.data, self.target


def unison_shuffled_copies(
        arr_0: np.ndarray, 
        arr_1: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
    """
        Convenience function to shuffle two arrays in unison.
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
