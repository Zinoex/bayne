from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from bayne.bounds import ibp, crown_ibp, crown
from bayne.mcmc import PyroMCMCBNN, PyroBatchLinear, PyroTanh
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

    model.sample(X, y, num_samples=1000, reject=200)


def main(args):
    device = torch.device(args.device)

    net = crown(crown_ibp(ibp(PyroMCMCBNN(
            PyroBatchLinear(1, 16),
            PyroTanh(),
            PyroBatchLinear(16, 16),
            PyroTanh(),
            PyroBatchLinear(16, 1),
            sigma=0.2,
            num_steps=100
    )))).to(device)

    train(net, device)
    test(net, device, 'MCMC')


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='Select device for tensor operations')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
