from data import DataLoader, load_config
from model import get_model
from vis_loss import PCACoordinates, LossSurface
from vis_data import vis_test_gt

from imports import os
from imports import argparse
from imports import keras
from imports import numpy as np
from imports import plt
from tqdm.keras import TqdmCallback

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Run the main script for UDL Loss Fitting."
    )

    parser.add_argument(
        "--model", 
        type = str, 
        choices = ["dense", "lstm"], 
        default = "lstm",
        help = "Optional argument. Choose the model type to run: 'dense', 'lstm'. " +
                "Visualisations for the different model architectures can be found in img/.\nDefault: lstm"
    )

    parser.add_argument(
        "--loss",
        type = str,
        choices = ["mse","vm"],
        default = "mse",
        help = "Optional argument. Choose the loss function to use: 'mse', 'vm'. " +
        "The loss functions are defined in src/loss.py. 'vm' refers to the von Mises loss."+
        "\nDefault: mse"
    )

    parser.add_argument(
        "--show-performance",
        action = "store_true",
        help = "Optional argument. If set visualises the models performance on the test set after training."
    )

    return parser.parse_args()  

def resolve_args(args:argparse.Namespace, cfg: dict) -> tuple[keras.Model, dict]:

    cfg["visualise_test"] = args.show_performance
    cfg["model"] = args.model.lower()
    cfg["loss"] = args.loss.lower()

    # get the model with the specified parameters
    model = get_model(cfg)
    
    return model, cfg

if __name__ == "__main__":
    # parse command line arguments
    parsed_args = parse_args()
    
    # fetch config
    cfg = load_config(os.path.join("cfg", "cfg.yml"))
    model, cfg = resolve_args(parsed_args, cfg)

    # load data
    loader = DataLoader(cfg)
    train, test = loader.load_data()

    # print configurations
    print(f"Using model type {cfg['model']} and Loss {cfg['loss']}:")
    model.summary()

    # preparation for recording the training and visualising the loss surface
    # capturing the initial state
    training_path = [model.get_weights()]
    collect_weights = keras.callbacks.LambdaCallback(
        on_epoch_end = (
            lambda batch, logs: training_path.append(
                    model.get_weights()
                )
        )
    )

    # fit the model
    print("Data Preparation Complete, Training...")
    history = model.fit(
        x = train,
        validation_data = test, 
        epochs = cfg["epochs"],
        callbacks = [collect_weights, TqdmCallback(verbose = 0)],
        verbose = 0
    )

    # append the recording 
    print("Done!")

    if cfg["visualise_test"]:
        figure = vis_test_gt(model, test)
        plt.show()

    # create the loss surface
    loss_surface = LossSurface(
        model = model,
        inputs = loader.data,
        outputs = loader.target
    )

    # project it into 2D using PCA
    coords = PCACoordinates(training_path)
    loss_surface.compile(
        points = 30,
        coords = coords,
        range = 1
    )
    
    # plot the loss surface and the training path
    fig, ax = loss_surface.plot_surfc_and_loss(
        coords = coords,
        training_path = training_path,
        dpi = 300
    )
    
    # ax.view_init(elev = 5+ 25*np.sin(np.radians(180)), azim = 25)
    plt.draw()

    # and have a nice rotating animation to the plot
    try:
        for i in range(360*2):
            # Calculate azimuth and elevation for rotation
            azim = i
            elev = 5 + 25 * np.sin(np.radians(i)) % 360
            ax.view_init(elev = elev, azim = azim)

            plt.draw()
            plt.pause(0.01)
    except KeyboardInterrupt:
        pass

    plt.show()
