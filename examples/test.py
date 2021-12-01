import torch
from matplotlib import pyplot as plt
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch import nn

from bayne.bounds import CROWNIntervalBoundPropagation, SampleIntervalBoundPropagation
from bayne.util import timer
from examples.noisy_sine import NoisySineDataset


@torch.no_grad()
def plot_bounds(model, device):
    num_slices = 100
    boundaries = torch.linspace(-2, 2, num_slices + 1).view(-1, 1).to(device)
    input_bounds = boundaries[:-1], boundaries[1:]
    sequential_network = model if isinstance(model, nn.Sequential) else model.network

    ibp = SampleIntervalBoundPropagation()
    interval_output_bounds = timer(ibp.interval_bounds)(sequential_network, input_bounds)

    crown = CROWNIntervalBoundPropagation()
    linear_output_bounds = timer(crown.linear_bounds)(sequential_network, input_bounds)

    input_bounds = input_bounds[0].cpu(), input_bounds[1].cpu()
    interval_output_bounds = interval_output_bounds[0].cpu(), interval_output_bounds[1].cpu()
    linear_output_bounds = (linear_output_bounds[0][0].cpu(), linear_output_bounds[0][1].cpu()), (linear_output_bounds[1][0].cpu(), linear_output_bounds[1][1].cpu())

    for i in range(num_slices):
        x1, x2 = input_bounds[0][i], input_bounds[1][i]
        y1, y2 = interval_output_bounds[0][i], interval_output_bounds[1][i]

        x1, x2 = x1.item(), x2.item()
        y1, y2 = y1.item(), y2.item()

        plt.plot([x1, x2], [y1, y1], color='blue', linestyle='dashed', label='IBP' if i == 0 else None)
        plt.plot([x1, x2], [y2, y2], color='blue', linestyle='dashed')

        y1, y2 = linear_output_bounds[0][0][i, 0, 0] * x1 + linear_output_bounds[0][1][i, 0], linear_output_bounds[0][0][i, 0, 0] * x2 + linear_output_bounds[0][1][i, 0]
        y3, y4 = linear_output_bounds[1][0][i, 0, 0] * x1 + linear_output_bounds[1][1][i, 0], linear_output_bounds[1][0][i, 0, 0] * x2 + linear_output_bounds[1][1][i, 0]

        y1, y2 = y1.item(), y2.item()
        y3, y4 = y3.item(), y4.item()

        plt.plot([x1, x2], [y1, y2], color='green', linestyle='dashed', label='CROWN-IBP' if i == 0 else None)
        plt.plot([x1, x2], [y3, y4], color='green', linestyle='dashed')

    X = torch.linspace(-2, 2, 1000, device=device).view(-1, 1)
    y = model(X)
    X, y = X.cpu().numpy(), y.cpu().numpy()

    plt.plot(X, y, color='blueviolet', label='Function to bound')


@torch.no_grad()
def test(model, device, label):
    num_samples = 1000

    criterion = MSELoss()

    dataset = NoisySineDataset()
    dataloader = DataLoader(dataset, batch_size=len(dataset), num_workers=0)
    X_train, y_train = next(iter(dataloader))
    X, y = X_train.to(device), y_train.to(device)

    y_mean = timer(model.predict_mean)(X, num_samples=num_samples)
    avg_loss = criterion(y_mean, y)

    print(f'Average MSE: {avg_loss.item()}')

    X_test = torch.linspace(-2.0, 2.0, 1000).view(-1, 1).to(device)
    y_dist = model.predict_dist(X_test, num_samples=num_samples)
    X_test, y_dist = X_test[..., 0].cpu().numpy(), y_dist[..., 0].cpu()

    y_mean = y_dist.mean(0).numpy()
    y_std = y_dist.var(0).numpy()

    plt.ylim(-4, 4)

    plt.plot(X_test, y_mean, 'r-', label='Predictive mean')
    plt.scatter(X_train.numpy(), y_train.numpy(), marker='+', label='Training data')
    plt.fill_between(X_test.ravel(), y_mean + y_std * 3, y_mean - y_std * 3, alpha=0.5, label='Epistemic Uncertainty')

    plot_bounds(model, device)

    plt.title(f'{label} prediction')
    plt.legend()

    plt.show()
