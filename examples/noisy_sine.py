import numpy as np
import torch
from torch.utils import data


def f(x, sigma):
    epsilon = np.random.randn(*x.shape) * sigma
    return 10 * np.sin(2 * np.pi * x) + epsilon


train_size = 1000
noise = 2.0

X = np.linspace(-1.0, 1.0, train_size).reshape(-1, 1).astype(np.float32)
y = f(X, sigma=noise).astype(np.float32)


class NoisySineDataset(data.Dataset):
    def __init__(self):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.X[idx], self.y[idx]
