import os

import torch
from matplotlib import pyplot as plt
from torch.nn import MSELoss

from bayne.util import timer


def evaluate(model, dataset, criterion, args):

    X_train, y_train = dataset[:]
    X, y = X_train.to(args.device), y_train.to(args.device)

    y_dist = timer(model.predict_dist)(X, num_samples=model.mcmc.num_samples)
    avg_loss = criterion(y_dist, y.unsqueeze(0).expand(y_dist.size(0), *y.size()))

    print(f'Average "loss": {avg_loss.item()}')


def plot_bnn(model, dataset, args):
    X_train, y_train = dataset[:]

    X_test = torch.linspace(-2.0, 2.0, 1000).view(-1, 1).to(args.device)
    y_dist = timer(model.predict_dist)(X_test, num_samples=model.mcmc.num_samples)
    X_test, y_dist = X_test[..., 0].cpu(), y_dist[..., 0].cpu()

    y_mean = y_dist.mean(0)
    y_std = y_dist.std(0)

    plt.figure(figsize=(6.4 * 2, 4.8 * 2))

    plt.ylim(-4, 4)

    for i in range(y_dist.size(0)):
        plt.plot(X_test, y_dist[i], color='y', alpha=0.1, label='Uncertainty' if i == 0 else None)

    plt.plot(X_test, y_mean, 'r-', label='Predictive mean')
    plt.scatter(X_train[::4], y_train[::4], marker='+', color='b', label='Training data')
    plt.fill_between(X_test.ravel(), y_mean + y_std * 3, y_mean - y_std * 3, alpha=0.5, label='Uncertainty')

    plt.title('MCMC BNN prediction')
    plt.legend()

    plt.savefig('visualization/mcmc_bnn.png', dpi=300)
    plt.show()


@torch.no_grad()
def test(model, dataset, criterion, args):
    os.makedirs('visualization', exist_ok=True)

    evaluate(model, dataset, criterion, args)

    if args.dataset == 'noisy-sine' and args.dim == 1:
        plot_bnn(model, dataset, args)
