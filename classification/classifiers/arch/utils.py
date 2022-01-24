from torch import nn


class Identity(nn.Module):
    def __init__(self):
        pass

    def forward(x):
        return x
