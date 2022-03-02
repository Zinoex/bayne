from math import sqrt

import numpy as np
import torch
from torch import distributions
from torch.utils.data import TensorDataset


def noise(train_size, sigma):
    dist1 = distributions.Normal(-0.4, sigma)
    dist2 = distributions.Normal(0.4, sigma)
    # dist = distributions.Beta(0.5, sigma)

    train_size = (train_size,)
    return torch.cat([dist1.sample(train_size), dist2.sample(train_size)])


def f_1d(train_size, sigma):
    X = torch.linspace(-1.0, 1.0, train_size).repeat(2).view(-1, 1)
    return X, torch.sin(2 * np.pi * X) + noise(train_size, sigma).view(-1, 1)


def f_2d(train_size, sigma):
    x_space = torch.linspace(-1.0, 1.0, int(sqrt(train_size)))
    X = torch.cartesian_prod(x_space, x_space)
    y = 0.5 * torch.sin(2 * np.pi * X[:, 0]) + 0.5 * torch.sin(2 * np.pi * X[:, 1]) + noise(train_size, sigma)

    return X, y.view(-1, 1)


class NoisySineDataset(TensorDataset):
    def __init__(self, dim=1, sigma=0.2, train_size=2 ** 10):
        if dim == 1:
            X, y = f_1d(train_size, sigma)
        elif dim == 2:
            X, y = f_2d(train_size, sigma)
        else:
            raise NotImplementedError()

        super(NoisySineDataset, self).__init__(X, y)
