from argparse import ArgumentParser

import torch
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

from bayne.mcmc import MonteCarloBNN
from bayne.nll import MinibatchGaussianNegativeLogProb, GaussianNegativeLogProb
from bayne.sampler import StochasticGradientHMC, CyclicalStochasticGradientHMC, HamiltonianMonteCarlo
from examples.noisy_sine import NoisySineDataset
from examples.test import test


################################################################
# An example of Markov chain Monte Carlo BNN for regression on
# a noisy sine dataset
################################################################


class ExampleMonteCarloBNN(nn.Sequential):
    def __init__(self, in_features, out_features):
        super().__init__(
            Linear(in_features, 8),
            nn.Tanh(),
            Linear(8, 8),
            nn.Tanh(),
            Linear(8, out_features)
        )

        nn.init.xavier_normal_(self[0].weight)
        nn.init.xavier_normal_(self[2].weight)


def train(model, device):
    dataset = NoisySineDataset()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    negative_log_prob = MinibatchGaussianNegativeLogProb(model, dataloader, device, noise=0.3)

    model.sample(negative_log_prob, num_samples=500, reject=500)


def main(args):
    device = torch.device(args.device)

    subnetwork = ExampleMonteCarloBNN(1, 1).to(device)
    model = MonteCarloBNN(subnetwork, sampler=CyclicalStochasticGradientHMC(initial_step_size=8e-5, momentum_decay=0.05, grad_noise=0.01, num_steps=50))

    train(model, device)
    test(model, device, 'HMC')


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='Select device for tensor operations')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
