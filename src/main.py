import model as md
import data
import vis
from imports import os
from imports import argparse
from loss import VonMisesFisher, CosineSimilarity, VonMises, CustomMSE
from imports import keras


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
        choices = ["direction", "speed"],
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
    train, test = data.load_data(cfg)
    

    # compile the model
    print(f"Using model type {cfg['model']} and Loss {cfg['loss'].__str__()}:")
    model.compile(optimizer = "adam", loss = cfg["loss"])
    model.summary()

    # fit the model
    model.fit(train, epochs = cfg["epochs"])
    # model.evaluate(test)
    fig = vis.vis_test_gt(model, test)
    fig.show()
    for x, y in test:
        breakpoint()
    breakpoint()