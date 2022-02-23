from math import sqrt

import numpy as np
import torch
from torch.utils.data import TensorDataset


def noise(train_size, sigma):
    return torch.randn(train_size) * sigma


def f_1d(train_size, sigma):
    X = torch.linspace(-1.0, 1.0, train_size).view(-1, 1)
    return X, torch.sin(2 * np.pi * X) + noise(train_size, sigma)


def f_2d(train_size, sigma):
    x_space = torch.linspace(-1.0, 1.0, int(sqrt(train_size)))
    X = torch.cartesian_prod(x_space, x_space)
    y = 0.5 * torch.sin(2 * np.pi * X[:, 0]) + 0.5 * torch.sin(2 * np.pi * X[:, 1]) + noise(train_size, sigma)

    return X, y.view(-1, 1)


class NoisySineDataset(TensorDataset):
    def __init__(self, dim=1, sigma=0.1, train_size=256):
        if dim == 1:
            X, y = f_1d(train_size, sigma)
        elif dim == 2:
            X, y = f_2d(train_size, sigma)
        else:
            raise NotImplementedError()

        super(NoisySineDataset, self).__init__(X, y)
