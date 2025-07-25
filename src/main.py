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
        choices = ["dense", "lstm", "circular"], 
        default = "lstm",
        help = "Choose the model type to run: 'dense', 'lstm' or 'circular'."
    )

    parser.add_argument(
        "--loss",
        type = str,
        choices = ["mse", "vMF","vM", "cosine"],
        default = "mse",
        help = "Choose the loss function to use: 'mse', 'vMF', 'vM', 'cosine'."
    )

    parser.add_argument(
        "--problem",
        type = str,
        choices = ["direction", "speed", "mnist"],
        default = "direction",
        help = "Chosse the type of problem to be solved. Which value is to be predicted: 'direction' or 'speed'. Speed is incompatible with model type 'circular'"
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

    elif args.model.lower() == "circular":
        model = md.get_circ_model(
            seq_len = cfg["seq_len"],
            input_dim = 3
        )
        cfg["data_prep"] = "circ"
        if args.problem.lower() == "speed":
            raise ValueError(f"Options Model: {args.model} and Problem: {args.problem} are incompatible."\
                             "\nPlease choose a different Model or Problem.")
        
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    cfg["model"] = args.model.lower()

    if args.loss.lower() == "vmf":
        cfg["loss"] = VonMisesFisher(
            kappa = 1.0,
            reduction = "sum_over_batch_size" 
            )
    elif args.loss.lower() == "vm":
        cfg["loss"] = VonMises()
    elif args.loss.lower() == "mse":
        cfg["loss"] = CustomMSE(axis = 0)
    elif args.loss.lower() == "cosine":
        cfg["loss"] = CosineSimilarity()
        # cfg["loss"] = "cosine_similarity"
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")
    
    cfg["problem"] = args.problem.lower()

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