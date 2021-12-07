import numpy as np
import torch
from torch.utils import data
from torch.utils.data import TensorDataset


def f(x, sigma):
    epsilon = torch.randn_like(x) * sigma
    return torch.sin(2 * np.pi * x) + epsilon


train_size = 256
noise = 0.2

X = torch.linspace(-1.0, 1.0, train_size, dtype=torch.float32).view(-1, 1)
y = f(X, sigma=noise)


class NoisySineDataset(TensorDataset):
    def __init__(self):
        super(NoisySineDataset, self).__init__(X, y)
