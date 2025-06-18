from imports import plt
from imports import numpy as np
from imports import os
import data

def visualise_data(
        data: tuple,
        cfg: dict = None,
        title:str = ""
):
    """
    Visualise the data given in a dataset.

    :param data: A Tuple of Datasets, containing the data to be visualised.
    :type data: tuple[tf.data.Dataset, tf.data.Dataset]
    :param type_visualisatin: Type of visualisation to be used. Can be "dense" or "lstm". Default is "dense". 
                    "Dense" refers to a plot aimed at strictly feed-forward networks, while "lstm" tries to capture the timing-variable nature of the data.
    :type type_visualisatin: str
    :param title: Title of the plot.
    :type title: str

    :return: A matplotlib figure object.
    :rtype: plt.Figure
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

    # make data accessible as numpy array
    # data = list(data)
    train = data[0].unbatch()
    test = data[1].unbatch()
    dat = train.concatenate(test)

    ds = np.full(
        shape = df.shape, 
        fill_value = np.nan,
        dtype = np.float32
    )
    ds[:,:2] = np.asarray(list(dat.map(lambda x, y: x)))
    ds[:,2] = np.asarray(list(dat.map(lambda x, y: y)))
    breakpoint()
    



    fig, axes = plt.subplots(
        nrows = 1,
        ncols = 2,
        figsize = (15, 6),
        sharex = True,
        sharey = True,
        subplot_kw = {'projection': 'polar'}
    )

    fig.suptitle(title, fontsize=16)
    axes[0].set_title("Data directly loaded")
    axes[0].grid(True)
    axes[0].set_theta_zero_location("N")
    axes[0].scatter(
        x = df[:,2],
        y = df[:,1],
        c = df[:,0],
        cmap = "viridis",
        alpha = 0.75
    )
    
    axes[1].set_title("Data after preprocessing")
    axes[1].grid(True)
    axes[1].set_theta_zero_location("N")
    axes[1].scatter(
        x = ds[:,2],
        y = ds[:,1],
        c = ds[:,0],
        cmap = "viridis",
        alpha = 0.75
    )

    # for lstm take average of direction/speed and compare to the target

    return fig




def visualise_loss_surface():
    raise NotImplementedError("Visualisation of the loss surface is not implemented yet.")


if __name__ == "__main__":
    # load config
    cfg = data.load_config("cfg/cfg.yml")

    # load data
    train, test = data.load_data(cfg)

    # visualise the data
    fig = visualise_data((train, test), cfg = cfg, title="Training and Test Data Visualisation")
    fig.show()
    breakpoint()