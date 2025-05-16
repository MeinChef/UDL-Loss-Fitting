import model as md
import data
from imports import os

def main():
    cfg = data.load_config(os.path.join("cfg", "cfg.yml"))
    train, test = data.load_data(cfg)
    model = md.get_model()
    model.summary()
    model.fit(train, epochs = cfg["epochs"])
    model.evaluate(test)

    # TODO: nice visualisation of loss surfaces, and maybe interactive stuff

def main_lstm():
    cfg = data.load_config(os.path.join("cfg", "cfg.yml"))
    train, test = data.load_lstm_data(cfg)
    model = md.get_lstm_model()
    model.summary()
    model.fit(train, epochs = cfg["epochs"])
    model.evaluate(test)

def main_circular():
    cfg = data.load_config(os.path.join("cfg", "cfg.yml"))
    train, test = data.load_circ_data(cfg)
    model = md.get_circ_model()
    model.summary()
    model.fit(train, epochs = cfg["epochs"])
    model.evaluate(test)

if __name__ == "__main__":
    main()
    