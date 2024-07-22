import torch
import torch.nn as nn
import argparse


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        # TODO
        pass

    def forward(self, x):
        # TODO
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="config")
    args = parser.parse_args()
    model = Model(args)
    x = torch.randn(1, 3, 64, 128, 128)
    a = model(x)
    print(a)
