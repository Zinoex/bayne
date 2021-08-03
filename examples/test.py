import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from examples.noisy_sine import NoisySineDataset


@torch.no_grad()
def test(model):
    num_samples = 500

    criterion = MSELoss()

    dataset = NoisySineDataset()
    dataloader = DataLoader(dataset, batch_size=len(dataset), num_workers=0)
    X, y = next(iter(dataloader))

    y_mean = model.predict_mean(X, num_samples=num_samples)
    avg_loss = criterion(y_mean, y)

    print(f'Average MSE: {avg_loss.item()}')

    X_test = torch.linspace(-3.0, 3.0, 1000).view(-1, 1)
    y_dist = model.predict_dist(X_test, num_samples=num_samples)[..., 0]
    X_test = X_test[..., 0]

    y_mean = y_dist.mean(0)
    y_sigma = y_dist.std(0)

    plt.plot(X_test, y_mean, 'r-', label='Predictive mean')
    plt.scatter(X, y, marker='+', label='Training data')
    plt.fill_between(X_test.ravel(), y_mean + 2 * y_sigma, y_mean - 2 * y_sigma,
                     alpha=0.5, label='Epistemic Uncertainty')
    plt.title('Prediction')
    plt.legend()
    plt.show()
