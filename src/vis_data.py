from imports import plt
from imports import numpy as np
from imports import os
import data

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
    # data = list(data)
    train = data[0].unbatch()
    test = data[1].unbatch()
    dat = train.concatenate(test)

    ds = np.asarray(list(dat.map(lambda x, y: x)))
    # ds = np.full(
    #     shape = df.shape, 
    #     fill_value = np.nan,
    #     dtype = np.float32
    # )
    # ds[:,:2] = np.asarray(list(dat.map(lambda x, y: x)))
    # ds[:,2] = np.asarray(list(dat.map(lambda x, y: y)))
    



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

def vis_trend(cfg: dict) -> plt:

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
    dir = df[:,1:]
    dir = np.deg2rad(dir)
    save = np.empty(
        (len(dir) - 1,),
        dtype = np.float32
    )
    
    def angle_btwn_vec(a,b):
        # Calculate dot product
        dot_product = np.dot(a, b)

        # Calculate magnitudes (lengths of the vectors)
        magnitude_a = np.linalg.norm(a)
        magnitude_b = np.linalg.norm(b)

        # Calculate angle in radians
        angle_radians = np.arccos(dot_product / (magnitude_a * magnitude_b))
        return angle_radians
    
    for i in range(len(dir) - 1):
        save[i] = angle_btwn_vec(dir[i], dir[i+1])

    fig, axes = plt.subplots(
        nrows = 1,
        ncols = 1,
        figsize = (6, 6),
        subplot_kw = {'projection': 'polar'}
    )

    axes.set_title("Change in Direction in regard to previous hour")
    axes.grid(True)
    axes.set_theta_zero_location("N")
    # axes.set_thetamax(355)
    # axes.set_thetamin(90)
    axes.scatter(
        x = save,
        y = df[1:,1],
        # c = df[:,0],
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

    gt = np.concatenate(gt)
    pred = np.concatenate(pred).squeeze()
    # gt[:,0] = np.arcsin(gt[:,0])
    # pred[:,0] = np.arcsin(pred[:,0])
    # gt_1 = np.arccos(gt[:,1])
    # pred_1 = np.arccos(pred[:,1])
    # gt = gt[:,0]
    # pred = pred[:,0]
    gt = np.arctan(gt[:,0]/gt[:,1])/2
    pred = np.arctan(pred[:,0]/pred[:,1])/2

 
    fig, axes = plt.subplots(
        nrows = 1,
        ncols = 3,
        figsize = (18, 12),
        subplot_kw = {'projection': 'polar'}
    )

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
        x = np.concatenate([gt, pred]),
        y = np.concatenate([np.linspace(0,1,len(gt)), np.linspace(0,1,len(pred))]),
        c = np.concatenate([np.zeros_like(gt), np.ones_like(pred)]),
        cmap = "viridis",
        alpha = 0.75
    )


    # axes[0,0].set_title("Ground Truth")
    # axes[0,0].grid(True)
    # axes[0,0].set_theta_zero_location("N")
    # axes[0,0].scatter(
    #     x = gt_1,
    #     y = np.linspace(0,1,len(gt_1)),
    #     color = "orange",
    #     alpha = 0.75
    # )

    # axes[0,1].set_title("Predictions")
    # axes[0,1].grid(True)
    # axes[0,1].set_theta_zero_location("N")
    # axes[0,1].scatter(
    #     x = pred_1,
    #     y = np.linspace(0,1,len(pred_1)),
    #     color = "blue",
    #     alpha = 0.75
    # )

    # axes[0,2].set_title("Overlapping")
    # axes[0,2].grid(True)
    # axes[0,2].set_theta_zero_location("N")
    # axes[0,2].scatter(
    #     x = np.concatenate([gt_1, pred_1]),
    #     y = np.concatenate([np.linspace(0,1,len(gt_1)), np.linspace(0,1,len(pred_1))]),
    #     c = np.concatenate([np.zeros_like(gt_1), np.ones_like(pred_1)]),
    #     cmap = "viridis",
    #     alpha = 0.75
    # )

    return fig

if __name__ == "__main__":
    # load config
    cfg = data.load_config("cfg/cfg.yml")
    cfg["data_prep"] = 'dense'
    cfg["problem"] = 'direction'

    # load data
    train, test = data.load_data(cfg)

    # visualise the data
    # fig = visualise_data((train, test), cfg = cfg, title="Training and Test Data Visualisation")
    fig = vis_trend(cfg)
    fig.show()
    breakpoint()