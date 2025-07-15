import model as md
import data
from imports import os
from imports import argparse
from loss import VonMisesFisher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Run the main script for UDL Loss Fitting."
    )

    parser.add_argument(
        "--model", 
        type = str, 
        choices = ["dense", "lstm"], 
        default = "lstm",
        help = "Choose the model type to run: 'dense' or 'lstm'."
    )

    parser.add_argument(
        "--loss",
        type = str,
        choices = ["mse", "von_Mises", "cosine"],
        default = "mse",
        help = "Choose the loss function to use: 'mse' or 'von_Mises'."
    )

    return parser.parse_args()  

def resolve_args(args:argparse.Namespace) -> tuple:

    if args.model == "dense":
        model = md.get_dense_model()
        prep = "dense"
    elif args.model == "lstm":
        model = md.get_lstm_model()
        prep = "lstm"
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    if args.loss == "von_Mises":
        loss = VonMisesFisher(kappa=1.0)
    elif args.loss == "mse":
        loss = "mse"
    elif args.loss == "cosine":
        loss = "cosine_similarity"
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")
    
    return model, {
            "loss": loss,
            "data_prep": prep
        }

if __name__ == "__main__":
    # parse command line arguments
    parsed_args = parse_args()
    model, args = resolve_args(parsed_args)
    
    # fetch config
    cfg = data.load_config(os.path.join("cfg", "cfg.yml"))
    cfg["data_prep"] = args["data_prep"]

    # load data
    train, test = data.load_data(cfg)
    

    # compile the model
    model.compile(optimizer = "adam", loss = args["loss"])
    model.summary()

    # fit the model
    model.fit(train, epochs = cfg["epochs"])
    model.evaluate(test)