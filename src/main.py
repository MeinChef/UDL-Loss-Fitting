import model
import data
from imports import os

def main():
    cfg = data.load_config(os.path.join("cfg", "cfg.yml"))
    train, test = data.load_data(cfg)


if __name__ == "__main__":
    main()
    