import model as md
import data
from imports import os

def main():
    cfg = data.load_config(os.path.join("cfg", "cfg.yml"))
    train, test = data.load_data(cfg)
    model = md.get_model()
    model.fit(train, epochs = cfg["epochs"])
    model.evaluate(test)


if __name__ == "__main__":
    main()
    