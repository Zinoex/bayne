import math
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from bayne.bounds.crown_ibp import linear_bound_propagation
from bayne.mcmc import PyroMCMCBNN, PyroBatchLinear, PyroTanh, PyroReLU
from bayne.bounds.ibp import interval_bound_propagation
from examples.noisy_sine import NoisySineDataset
from examples.test import test


################################################################
# An example of Markov chain Monte Carlo BNN for regression on
# a noisy sine dataset
################################################################


def train(model, device):
    dataset = NoisySineDataset()
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

    X, y = next(iter(dataloader))
    X, y = X.to(device), y.to(device)

    # We use so many samples because we need the velocity to be resampled much.
    # Could be improve by modifying Pyro HMC to allow resampling at each iteration.
    model.sample(X, y, num_samples=10000, reject=2000)


def main(args):
    device = torch.device(args.device)

    net = linear_bound_propagation(interval_bound_propagation(PyroMCMCBNN(
            PyroBatchLinear(1, 16),
            PyroTanh(),
            PyroBatchLinear(16, 16),
            PyroTanh(),
            PyroBatchLinear(16, 1),
            sigma=0.2,
            num_steps=100
    ))).to(device)

    train(net, device)
    test(net, device, 'HMC')


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='Select device for tensor operations')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
