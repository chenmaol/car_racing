import argparse
from datasets import get_loaders
from model import Model

def main(args):
    # TODO: add dataset, dataloader
    train_dataloader, val_dataloader = get_loaders(args)

    # TODO: add model
    model = Model(args)

    # TODO: training loop

    # TODO: evaluation loop
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    args = parser.parse_args()
    main(args)
