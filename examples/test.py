import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from examples.weather import WeatherHistoryDataset, y_scaler


@torch.no_grad()
def test(model):
    num_samples = 1000

    criterion = MSELoss()

    dataset = WeatherHistoryDataset(train=False)

    # Average MSE test
    dataloader = DataLoader(dataset, batch_size=len(dataset), num_workers=0)
    X, y = next(iter(dataloader))

    y_dist = model.predict_dist(X, num_samples=num_samples)
    y_pred = y_dist.mean(0)
    y_var = y_dist.var(0)

    avg_loss = criterion(y_pred, y)

    print(f'Average MSE: {avg_loss.item()}')
    print(f'Average var: {y_var.mean()}')

    # Distribution test
    dataloader = DataLoader(dataset, num_workers=0)
    X, y = next(iter(dataloader))

    y_dist = model.predict_dist(X, num_samples=num_samples)
    y_dist = y_scaler.inverse_transform(y_dist.numpy())[:, 0]

    plt.hist(y_dist)
    plt.axvline(y_scaler.inverse_transform(y), color='r', linestyle='dashed', linewidth=1)
    plt.show()
