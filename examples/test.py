import os

import torch
from matplotlib import pyplot as plt
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from bayne.util import timer
from examples.noisy_sine import NoisySineDataset


@torch.no_grad()
def plot_bounds(model, device, label):
    num_slices = 20
    boundaries = torch.linspace(-2, 2, num_slices + 1).view(-1, 1).to(device)
    lower_x, upper_x = boundaries[:-1], boundaries[1:]

    bound_indices = torch.arange(0, 1000)

    lower_ibp, upper_ibp = timer(model.func_index)(model.ibp, bound_indices, lower_x, upper_x)
    lower_lbp, upper_lbp = timer(model.func_index)(model.crown, bound_indices, lower_x, upper_x)

    sample_to_plot = 900

    lower_x, upper_x = lower_x.cpu(), upper_x.cpu()
    lower_ibp, upper_ibp = lower_ibp[sample_to_plot].cpu(), upper_ibp[sample_to_plot].cpu()
    lower_lbp, upper_lbp = (lower_lbp[0][sample_to_plot].cpu(), lower_lbp[1][sample_to_plot].cpu()), (upper_lbp[0][sample_to_plot].cpu(), upper_lbp[1][sample_to_plot].cpu())

    plt.clf()
    plt.figure(figsize=(6.4 * 2, 4.8 * 2))

    plt.ylim(-4, 4)

    for i in range(num_slices):
        x1, x2 = lower_x[i].item(), upper_x[i].item()
        y1, y2 = lower_ibp[i].item(), upper_ibp[i].item()

        plt.plot([x1, x2], [y1, y1], color='blue', label='IBP' if i == 0 else None)
        plt.plot([x1, x2], [y2, y2], color='blue')

        y1, y2 = lower_lbp[0][i, 0, 0] * x1 + lower_lbp[1][i, 0], lower_lbp[0][i, 0, 0] * x2 + lower_lbp[1][i, 0]
        y3, y4 = upper_lbp[0][i, 0, 0] * x1 + upper_lbp[1][i, 0], upper_lbp[0][i, 0, 0] * x2 + upper_lbp[1][i, 0]

        y1, y2 = y1.item(), y2.item()
        y3, y4 = y3.item(), y4.item()

        plt.plot([x1, x2], [y1, y2], color='green', label='CROWN' if i == 0 else None)
        plt.plot([x1, x2], [y3, y4], color='green')

    X = torch.linspace(-2, 2, 1000, device=device).view(-1, 1)
    y = model.predict_index(bound_indices, X)[sample_to_plot]
    X, y = X.cpu().numpy(), y.cpu().numpy()

    plt.plot(X, y, color='blueviolet', label='Function to bound')

    plt.title(f'Bound propagation')
    plt.legend()

    plt.savefig(f'visualization/{label}_lbp.png', dpi=300)


def plot_bnn(model, device, label):
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
    y_dist = timer(model.predict_dist)(X_test, num_samples=num_samples)
    X_test, y_dist = X_test[..., 0].cpu().numpy(), y_dist[..., 0].cpu()

    y_mean = y_dist.mean(0).numpy()
    y_std = y_dist.std(0).numpy()

    plt.clf()
    plt.figure(figsize=(6.4 * 2, 4.8 * 2))

    plt.ylim(-4, 4)

    plt.plot(X_test, y_mean, 'r-', label='Predictive mean')
    plt.scatter(X_train.numpy(), y_train.numpy(), marker='+', label='Training data')
    plt.fill_between(X_test.ravel(), y_mean + y_std * 3, y_mean - y_std * 3, alpha=0.5, label='Uncertainty')

    plt.title(f'{label} BNN prediction')
    plt.legend()

    plt.savefig(f'visualization/{label}_bnn.png', dpi=300)


@torch.no_grad()
def test(model, device, label):
    os.makedirs('visualization', exist_ok=True)
    plot_bnn(model, device, label)
    plot_bounds(model, device, label)
