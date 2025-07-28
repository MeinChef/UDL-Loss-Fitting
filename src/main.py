from data import DataLoader, load_config
from model import get_dense_model, get_lstm_model
from loss import VonMises, CustomMSE
from vis_loss import PCACoordinates, LossSurface
from vis_data import vis_test_gt

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
        choices = ["dense", "lstm"], 
        default = "lstm",
        help = "Optional argument. Choose the model type to run: 'dense', 'lstm'. " +
                "Visualisations for the different models can be found in img/.\nDefault: lstm"
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
        "--from-pretrained",
        action = "store_true",
        help = "Optional argument. Load model weights from a pretrained checkpoint present in data/."
    )

    return parser.parse_args()  

def resolve_args(args:argparse.Namespace, cfg: dict) -> tuple[keras.Model, dict]:

    if args.loss.lower() == "vm":
        cfg["loss"] = VonMises()
        cfg["out"] = 1
    elif args.loss.lower() == "mse":
        cfg["loss"] = CustomMSE(axis = 0)
        cfg["out"] = 2
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")

    if args.from_pretrained:
        model = keras.saving.load_model(
            os.path.join(
                cfg["data_path"], 
                f"{args.model}_{args.loss}.keras"
            ),
            compile = False
        )      
    else:
        if args.model.lower() == "dense":
            model = get_dense_model(
                num_out = cfg["out"]
            )
            cfg["data_prep"] = "dense"

        elif args.model.lower() == "lstm":
            model = get_lstm_model(
                seq_len = cfg["seq_len"],
                num_out = cfg["out"]
            )
            cfg["data_prep"] = "lstm"

        else:
            raise ValueError(f"Unknown model type: {args.model}")
    cfg["model"] = args.model.lower()


    
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
    figure = vis_test_gt(model, test)
    breakpoint()

    # create the loss surface
    loss_surface = LossSurface(
        model = model,
        inputs = loader.data,
        outputs = loader.target
    )

    coords = PCACoordinates(training_path)
    loss_surface.compile(
        points = 30,
        coords = coords,
        range = 5 
    )

    # and plot it
    ax = loss_surface.plot(dpi = 300)
    plt.show()
    breakpoint()