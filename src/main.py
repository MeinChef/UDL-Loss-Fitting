# from imports import os
# from imports import logging
# logging.disable(logging.WARNING)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


import model as md
import data
import vis_loss
from loss import VonMisesFisher, CosineSimilarity, VonMises, CustomMSE

from imports import os
from imports import argparse
from imports import keras
from imports import plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Run the main script for UDL Loss Fitting."
    )

    parser.add_argument(
        "--model", 
        type = str, 
        choices = ["dense", "lstm", "sine_cosine"], 
        default = "lstm",
        help = "Choose the model type to run: 'dense', 'lstm' or 'sine_cosine'."
    )

    parser.add_argument(
        "--loss",
        type = str,
        choices = ["mse","vM"],
        default = "mse",
        help = "Choose the loss function to use: 'mse', 'vM'."
    )

    return parser.parse_args()  

def resolve_args(args:argparse.Namespace, cfg: dict) -> tuple[keras.Model, dict]:

    if args.model.lower() == "dense":
        model = md.get_dense_model()
        cfg["data_prep"] = "dense"

    elif args.model.lower() == "lstm":
        model = md.get_lstm_model(
            seq_len = cfg["seq_len"]
        )
        cfg["data_prep"] = "lstm"

    elif args.model.lower() == "sine_cosine":
        model = md.get_sincos_model(
            seq_len = cfg["seq_len"],
            input_dim = 3
        )
        cfg["data_prep"] = "sine_cosine"
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    cfg["model"] = args.model.lower()

    if args.loss.lower() == "vm":
        cfg["loss"] = VonMises()
    elif args.loss.lower() == "mse":
        cfg["loss"] = CustomMSE(axis = 0)
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")
    
    return model, cfg

if __name__ == "__main__":
    # parse command line arguments
    parsed_args = parse_args()
    
    # fetch config
    cfg = data.load_config(os.path.join("cfg", "cfg.yml"))
    model, cfg = resolve_args(parsed_args, cfg)

    # load data
    loader = data.DataLoader(cfg)
    train, test = loader.load_data()
    
    # compile the model
    model.compile(optimizer = "adam", loss = cfg["loss"])

    # print configurations
    print(f"Using model type {cfg['model']} and Loss {cfg['loss'].__str__()}:")
    model.summary()

    
    # preparation for recording the training and visualising the loss surface
    # initial state
    training_path = [model.weights]
    collect_weights = keras.callbacks.LambdaCallback(
        on_epoch_end = (
            lambda batch, logs: training_path.append(
                    model.weights
                )
        )
    )


    # fit the model
    history = model.fit(
        train, 
        epochs = cfg["epochs"],
        callbacks = collect_weights,
        verbose = 1
    )
    model.evaluate(test)
    # fig = vis.vis_test_gt(model, test)
    # fig.show()

    # create the loss surface
    loss_surface = vis_loss.LossSurface(
        model = model,
        inputs = loader.data,
        outputs = loader.target
    )

    coords = vis_loss.PCACoordinates(training_path)
    loss_surface.compile(
        points = 30,
        coords = coords,
        range = 5 
    )

    # and plot it
    ax = loss_surface.plot(dpi = 300)
    plt.show()
    breakpoint()