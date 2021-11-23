import torch
from matplotlib import pyplot as plt
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch import nn

from bayne.bounds.ibp import SampleIntervalBoundPropagation
from bayne.util import timer
from examples.noisy_sine import NoisySineDataset


@torch.no_grad()
def plot_bounds(model):
    ibp = SampleIntervalBoundPropagation()
    num_slices = 100
    boundaries = torch.linspace(-2, 2, num_slices + 1).view(-1, 1)
    input_bounds = boundaries[:-1], boundaries[1:]
    sequential_network = model if isinstance(model, nn.Sequential) else model.network
    output_bounds = timer(ibp.interval_bounds)(sequential_network, input_bounds)

    for i in range(num_slices):
        x1, x2 = input_bounds[0][i], input_bounds[1][i]
        y1, y2 = output_bounds[0][i], output_bounds[1][i]

        plt.plot([x1, x2], [y1, y1], color='blue', linestyle='dashed', label='IBP' if i == 0 else None)
        plt.plot([x1, x2], [y2, y2], color='blue', linestyle='dashed')


@torch.no_grad()
def test(model, label):
    num_samples = 1000

    criterion = MSELoss()

    dataset = NoisySineDataset()
    dataloader = DataLoader(dataset, batch_size=len(dataset), num_workers=0)
    X, y = next(iter(dataloader))

    y_mean = timer(model.predict_mean)(X, num_samples=num_samples)
    avg_loss = criterion(y_mean, y)

    print(f'Average MSE: {avg_loss.item()}')

    X_test = torch.linspace(-2.0, 2.0, 1000).view(-1, 1)
    y_dist = model.predict_dist(X_test, num_samples=num_samples)
    X_test, y_dist = X_test[..., 0], y_dist[..., 0]

    y_mean = y_dist.mean(0)
    y_sigma = y_dist.std(0)

    plt.ylim(-4, 4)

    plt.plot(X_test, y_mean, 'r-', label='Predictive mean')
    plt.scatter(X, y, marker='+', label='Training data')
    plt.fill_between(X_test.ravel(), y_mean + 2 * y_sigma, y_mean - 2 * y_sigma, alpha=0.5, label='Epistemic Uncertainty')

    plot_bounds(model)

    plt.title(f'{label} prediction')
    plt.legend()

    plt.show()
