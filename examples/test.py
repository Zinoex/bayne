import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from examples.weather import WeatherHistoryDataset, y_scaler, X_scaler


@torch.no_grad()
def test(model):
    num_samples = 1000

    criterion = MSELoss()

    dataset = WeatherHistoryDataset(train=False)
    dataloader = DataLoader(dataset, batch_size=len(dataset), num_workers=0)
    X, y = next(iter(dataloader))

    y_dist = model.predict_dist(X, num_samples=num_samples)
    y_mean = y_dist.mean(0)

    avg_loss = criterion(y_mean, y)

    print(f'Average MSE: {avg_loss.item()}')

    X = X[..., 0]
    y_dist = y_dist[..., 0]

    sort_idx = np.argsort(X)
    X = X[sort_idx]
    y_dist = y_dist[:, sort_idx]
    y = y[sort_idx]

    X = X_scaler.inverse_transform(X)
    y_dist = y_scaler.inverse_transform(y_dist)
    y = y_scaler.inverse_transform(y)

    y_mean = y_dist.mean(0)
    y_sigma = y_dist.std(0)

    plt.plot(X, y_mean, 'r-', label='Predictive mean')
    plt.scatter(X, y, marker='+', label='Training data')
    plt.fill_between(X.ravel(), y_mean + 2 * y_sigma, y_mean - 2 * y_sigma,
                     alpha=0.5, label='Epistemic uncertainty')
    plt.title('Prediction')
    plt.legend()
    plt.show()
