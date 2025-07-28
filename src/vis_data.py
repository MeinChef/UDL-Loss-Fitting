from imports import plt
from imports import numpy as np
from imports import os

def visualise_data(
        data: tuple,
        cfg: dict,
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
    df[:,2] = np.deg2rad(df[:,2])

    # make data accessible as numpy array
    train = data[0].unbatch()
    test = data[1].unbatch()
    dat = train.concatenate(test)

    ds = np.asarray(list(dat.map(lambda x, y: x)))


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
    return fig


def vis_test_gt(model, data):
    """
    Visualise the predictions of the test against the ground truth 
    """

    gt = []
    pred = []
    for x, y in data:
        pred.append(model(x))
        gt.append(y)

    # dimension zero is the batch_size
    out = len(gt[0].shape)
    gt = np.concatenate(gt)
    pred = np.concatenate(pred).squeeze()
    
    if out == 2:
        gt = np.arctan2(gt[:,1], gt[:,0])
        pred = np.arctan2(pred[:,1], pred[:,0])


    fig, axes = plt.subplots(
        nrows = 1,
        ncols = 3,
        figsize = (18,6),
        subplot_kw = {'projection': 'polar'}
    )

    # we are just plotting the samples from inward out, 
    # since they do not have a speed attribute anymore
    axes[0].set_title("Ground Truth")
    axes[0].grid(True)
    axes[0].set_theta_zero_location("N")
    axes[0].scatter(
        x = gt,
        y = np.linspace(0,1,len(gt)),
        color = "orange",
        alpha = 0.75
    )

    axes[1].set_title("Predictions")
    axes[1].grid(True)
    axes[1].set_theta_zero_location("N")
    axes[1].scatter(
        x = pred,
        y = np.linspace(0,1,len(pred)),
        color = "blue",
        alpha = 0.75
    )

    axes[2].set_title("Overlapping")
    axes[2].grid(True)
    axes[2].set_theta_zero_location("N")
    axes[2].scatter(
        x = np.concatenate([
            gt, 
            pred
        ]),
        y = np.concatenate([
            np.linspace(0,1,len(gt)), 
            np.linspace(0,1,len(pred))
        ]),
        c = np.concatenate([
            np.zeros_like(gt), 
            np.ones_like(pred)
        ]),
        cmap = "viridis",
        alpha = 0.75
    )

    return fig